"""Main entry point for Mil Mapper 2.0 with PySide6 MVC architecture."""

import argparse
import faulthandler
import logging
import os
import shutil
import sys
from pathlib import Path

# Python traceback on SIGSEGV/SIGFPE/etc from C++ extensions (Qt/numpy/PIL).
faulthandler.enable()

# Limit glibc per-thread malloc arenas BEFORE any C extension (numpy / cv2 /
# PIL) is imported, and BEFORE the persistent worker is spawned. Default on
# 12-core machines is 8 × CPUs = 96 arenas, each reserving its own VMS pool;
# during a SATELLITE z=16 retina build this inflates VMS post-build to
# ~6 GB and stops the 2nd build dead with MemoryError under RLIMIT_AS.
# MALLOC_ARENA_MAX=2 cuts post-build VMS to ~4.4 GB (and RSS to ~1.3 GB)
# without measurable throughput cost. Honour an existing env if the user
# already set it.
os.environ.setdefault('MALLOC_ARENA_MAX', '2')

from PySide6.QtCore import QLocale, QtMsgType, qInstallMessageHandler
from PySide6.QtGui import QIcon

from gui.theme import apply_dark_title_bar
from gui.view import create_application
from shared.constants import LOG_FSYNC_TO_FILE, MEMORY_MIN_TOTAL_MB, MEMORY_RLIMIT_RATIO
from shared.diagnostics import log_memory_usage
from shared.portable import get_portable_path, is_portable_mode

logger = logging.getLogger(__name__)


class _FsyncFileHandler(logging.FileHandler):
    """
    FileHandler that flushes after every record.

    Previously also called os.fsync() per record — that gave the bulk of
    "log writes are slow" overhead (~10-15ms each on SSD). flush() alone
    pushes data to the kernel which survives SIGKILL/MemoryError/Python
    crashes; only sudden power loss or kernel panic would lose recent
    lines. For application debugging that's enough.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.stream.flush()
        except Exception:
            logger.debug('flush failed for log file', exc_info=True)


def setup_logging() -> tuple[Path, Path]:
    """
    Configure application logging to LOCALAPPDATA and ensure user dirs.
    В portable режиме использует локальные папки относительно exe.

    Returns:
        Tuple (appdata_base, local_base) for further use.

    """
    # Determine user profile dirs
    if is_portable_mode():
        # Portable режим: все данные в папке с exe
        # appdata_base должен указывать на корень, потому что код добавляет /configs
        appdata_base = Path(sys.argv[0]).resolve().parent
        local_base = get_portable_path('data')
        log_dir = get_portable_path('logs')
    else:
        appdata_env = os.getenv('APPDATA')
        local_env = os.getenv('LOCALAPPDATA')
        if appdata_env and local_env:
            # Windows
            appdata_base = Path(appdata_env) / 'SK42'
            local_base = Path(local_env) / 'SK42'
            log_dir = local_base / 'log'
        else:
            # Linux/macOS — XDG-совместимые пути
            xdg_data = os.getenv('XDG_DATA_HOME', '')
            xdg_state = os.getenv('XDG_STATE_HOME', '')
            appdata_base = (
                Path(xdg_data) / 'sk42mapper'
                if xdg_data
                else Path.home() / '.local' / 'share' / 'sk42mapper'
            )
            local_base = (
                Path(xdg_data) / 'sk42mapper'
                if xdg_data
                else Path.home() / '.local' / 'share' / 'sk42mapper'
            )
            log_dir = (
                Path(xdg_state) / 'sk42mapper' / 'log'
                if xdg_state
                else Path.home() / '.local' / 'state' / 'sk42mapper' / 'log'
            )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'mil_mapper.log'

    # Лог открываем в append-режиме: при OOM-падении worker'а / GUI
    # Vladimir перезапускает приложение немедленно — с mode='w' crash-лог
    # терялся (см. след. сессию = 43 строки чистого старта без NSU).
    # Каждый запуск помечается явным баннером ниже.
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if LOG_FSYNC_TO_FILE:
        handlers.append(_FsyncFileHandler(str(log_file), mode='a', encoding='utf-8'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )
    return appdata_base, local_base


def _migrate_from_old_name(appdata_base: Path, local_base: Path) -> None:
    """Одноразовая миграция данных из SK42mapper → SK42."""
    old_appdata = appdata_base.parent / 'SK42mapper'
    old_local = local_base.parent / 'SK42mapper'

    if not old_appdata.exists():
        return

    logger.info('Миграция: обнаружена папка %s', old_appdata)

    # .secrets.env (API ключ)
    old_secrets = old_appdata / '.secrets.env'
    new_secrets = appdata_base / '.secrets.env'
    if old_secrets.exists() and not new_secrets.exists():
        appdata_base.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_secrets, new_secrets)
        logger.info('Миграция: скопирован .secrets.env')

    # Профили
    old_profiles = old_appdata / 'configs' / 'profiles'
    new_profiles = appdata_base / 'configs' / 'profiles'
    if old_profiles.exists():
        new_profiles.mkdir(parents=True, exist_ok=True)
        for f in old_profiles.glob('*.toml'):
            dst = new_profiles / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                logger.info('Миграция: скопирован профиль %s', f.name)

    # Сохранённые карты
    old_maps = old_appdata / 'maps'
    new_maps = appdata_base / 'maps'
    if old_maps.exists():
        new_maps.mkdir(parents=True, exist_ok=True)
        for f in old_maps.iterdir():
            if f.is_file():
                dst = new_maps / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    logger.info('Миграция: скопирована карта %s', f.name)

    # Кэш тайлов (переименование)
    old_cache = old_local / '.cache' / 'tiles'
    new_cache = local_base / '.cache' / 'tiles'
    if old_cache.exists() and not new_cache.exists():
        new_cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            old_cache.rename(new_cache)
            logger.info('Миграция: перемещён кэш тайлов')
        except OSError:
            logger.warning('Миграция: не удалось переместить кэш, будет скачан заново')

    logger.info('Миграция из SK42mapper завершена')


def _check_system_memory() -> bool:
    """
    Check that total system RAM meets minimum requirements.

    Returns True if OK, False if insufficient.
    """
    try:
        import psutil

        total_mb = psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        # psutil unavailable — skip the check
        return True

    if total_mb < MEMORY_MIN_TOTAL_MB:
        logger.error(
            'Недостаточно оперативной памяти: %.0f МБ (требуется минимум %d МБ)',
            total_mb,
            MEMORY_MIN_TOTAL_MB,
        )
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(
                None,
                'Недостаточно памяти',
                f'Для работы приложения требуется минимум '
                f'{MEMORY_MIN_TOTAL_MB} МБ оперативной памяти.\n'
                f'Обнаружено: {total_mb:.0f} МБ.',
            )
        except Exception:
            logger.debug('Could not show low-memory dialog', exc_info=True)
        return False
    return True


def _set_memory_limit() -> None:
    """
    Set GUI process memory limit via RLIMIT_AS (Linux only).

    Cap = MEMORY_RLIMIT_RATIO × RAM (NOT RAM + swap). The old
    (RAM + swap) × 0.85 formula allowed the GUI to address ~14.8 GB
    virtual on a 13 GB RAM + 4 GB swap system — meaning a runaway
    leak could drag the kernel into thrashing swap and trigger
    system-wide OOM that took the IDE down with us. Bounding to a
    fraction of physical RAM ensures we MemoryError inside our own
    process long before the system as a whole is starved.

    The worker process gets its own (lower) limit from
    _set_child_memory_limit; together the two processes are bounded
    to leave guaranteed headroom for IDE / OS / page cache.
    """
    try:
        import resource

        import psutil
    except ImportError:
        return

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    # RAM-only cap (NOT RAM+swap): MemoryError must fire inside this
    # process before physical RAM is exhausted, so the IDE keeps working.
    limit_bytes = int(mem.total * MEMORY_RLIMIT_RATIO)

    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit_bytes = min(limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        logger.info(
            'RLIMIT_AS set to %.0f MB (RAM=%.0f MB × ratio=%.2f, reserve '
            '~%.0f MB for IDE/OS; swap=%.0f MB not counted in limit)',
            limit_bytes / (1024 * 1024),
            mem.total / (1024 * 1024),
            MEMORY_RLIMIT_RATIO,
            (mem.total - limit_bytes) / (1024 * 1024),
            swap.total / (1024 * 1024),
        )
    except (ValueError, OSError) as e:
        logger.warning('Failed to set RLIMIT_AS: %s', e)


def main() -> int:
    """Main application entry point."""
    appdata_base, local_base = setup_logging()
    logger.info('=' * 72)
    logger.info('Starting Mil Mapper 2.0 (pid=%d)', os.getpid())
    logger.info('=' * 72)

    _set_memory_limit()

    if not _check_system_memory():
        return 1

    # Миграция из SK42mapper → SK42 (при первом запуске новой версии)
    if not is_portable_mode():
        try:
            _migrate_from_old_name(appdata_base, local_base)
        except Exception as e:
            logger.warning('Миграция не удалась: %s', e)

    # Ensure user data directories and bootstrap defaults
    try:
        # Create user dirs
        if is_portable_mode():
            # Portable режим: создаем папки в директории приложения
            get_portable_path('configs/profiles').mkdir(parents=True, exist_ok=True)
            get_portable_path('maps').mkdir(parents=True, exist_ok=True)
            get_portable_path('cache/tiles').mkdir(parents=True, exist_ok=True)
        else:
            # Обычный режим
            (appdata_base / 'configs' / 'profiles').mkdir(parents=True, exist_ok=True)
            (appdata_base / 'maps').mkdir(parents=True, exist_ok=True)
            (local_base / '.cache' / 'tiles').mkdir(parents=True, exist_ok=True)
        # Remove legacy flat-format profiles (will be replaced by sectioned defaults)
        _profiles_dir = (
            get_portable_path('configs/profiles')
            if is_portable_mode()
            else appdata_base / 'configs' / 'profiles'
        )
        if _profiles_dir.exists():
            for _pf in list(_profiles_dir.glob('*.toml')):
                try:
                    _head = _pf.read_text(encoding='utf-8').lstrip()
                    if _head and not _head.startswith('['):
                        _pf.unlink()
                        logger.info('Удалён устаревший профиль: %s', _pf.name)
                except Exception:
                    logger.debug(
                        'Не удалось удалить профиль %s', _pf.name, exc_info=True
                    )

        # Copy default configs if not present
        install_dir = Path(sys.argv[0]).resolve().parent
        # Приоритет поиска configs/:
        # 1. sys._MEIPASS (PyInstaller onefile — данные во временной папке)
        # 2. install_dir/_internal/configs (PyInstaller onedir — стандартная структура)
        # 3. install_dir/configs (рядом с exe/скриптом)
        # 4. install_dir/../configs (разработка: src/../configs)
        meipass = getattr(sys, '_MEIPASS', None)
        default_cfg_root = Path(meipass) / 'configs' if meipass else None
        if default_cfg_root is None or not default_cfg_root.exists():
            default_cfg_root = install_dir / '_internal' / 'configs'
        if not default_cfg_root.exists():
            default_cfg_root = install_dir / 'configs'
        if not default_cfg_root.exists():
            default_cfg_root = install_dir.parent / 'configs'

        logger.info(f'Looking for default configs in: {default_cfg_root}')
        if default_cfg_root.exists():
            logger.info(
                f'Found default configs, copying to: {appdata_base / "configs"}'
            )
            # Copy files only if missing in user configs
            for src in default_cfg_root.rglob('*'):
                if src.is_file():
                    rel = src.relative_to(default_cfg_root)
                    dst = appdata_base / 'configs' / rel
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(src, dst)
                            logger.info(f'Copied config: {rel}')
                        except Exception as e:
                            logger.warning(f'Failed to copy {rel}: {e}')
        else:
            logger.warning(f'Default configs not found at: {default_cfg_root}')
    except Exception as e:
        logger.warning(f'User data bootstrap failed: {e}')

    parser = argparse.ArgumentParser(
        description='Mil Mapper 2.0 - создание топографических карт',
    )
    parser.add_argument(
        '--mode',
        choices=['gui'],
        default='gui',
        help='Режим запуска приложения (только gui в версии 2.0)',
    )

    parser.parse_args()

    # Suppress harmless "qt.svg: Duplicate unique style id" warnings
    # originating from broken SVGs in system icon themes (e.g. elementary).
    # Все остальные Qt-сообщения маршрутизируем в Python-лог — раньше шли
    # в stderr и терялись при крэше; warnings часто предвестники падений.
    _orig_handler = None
    _qt_logger = logging.getLogger('Qt')
    _qt_level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _qt_msg_filter(msg_type, context, message) -> None:  # noqa: ANN001
        if msg_type == QtMsgType.QtWarningMsg and message.startswith('qt.svg:'):
            return  # swallow
        level = _qt_level_map.get(msg_type, logging.WARNING)
        cat = getattr(context, 'category', None) or ''
        _qt_logger.log(level, '[%s] %s', cat, message)
        if _orig_handler is not None:
            _orig_handler(msg_type, context, message)

    _orig_handler = qInstallMessageHandler(_qt_msg_filter)

    try:
        # Create PySide6 application
        log_memory_usage('before creating application')
        app, window, model, controller = create_application()
        log_memory_usage('after creating application')

        # Configure application properties
        app.setQuitOnLastWindowClosed(True)

        # Set Russian locale for all Qt dialogs
        QLocale.setDefault(QLocale(QLocale.Language.Russian, QLocale.Country.Russia))

        # Set application icon if available
        install_dir2 = Path(sys.argv[0]).resolve().parent
        icon_path = install_dir2 / 'img' / 'icon.ico'
        if not icon_path.exists():
            # fallback to source path (dev mode)
            icon_path = Path(__file__).parent.parent / 'img' / 'icon.ico'
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))

        # Dark title bar on Windows 10/11
        apply_dark_title_bar(int(window.winId()))

        # Show main window
        window.showMaximized()

        logger.info('Application started successfully')

        # Run application event loop
        result = app.exec()
    except Exception:
        logger.exception('Failed to start application')
        return 1
    else:
        return result


if __name__ == '__main__':
    import multiprocessing

    # 'spawn' вместо 'fork': на Linux fork() после инициализации OpenMP
    # (numpy, cv2) вызывает deadlock/crash. 'spawn' создаёт чистый процесс.
    # На Windows 'spawn' — уже дефолт, force=True безопасно.
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()
    sys.exit(main())

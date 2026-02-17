"""Main entry point for Mil Mapper 2.0 with PySide6 MVC architecture."""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QLocale
from PySide6.QtGui import QIcon

from gui.theme import apply_dark_title_bar
from gui.view import create_application
from shared.diagnostics import log_memory_usage
from shared.portable import get_portable_path, is_portable_mode

logger = logging.getLogger(__name__)


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
        # Обычный режим: стандартные пути Windows
        appdata_base = (
            Path(os.getenv('APPDATA') or Path.home() / 'AppData' / 'Roaming') / 'SK42'
        )
        local_base = (
            Path(os.getenv('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
            / 'SK42'
        )
        log_dir = local_base / 'log'

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'mil_mapper.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding='utf-8'),
        ],
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


def main() -> int:
    """Main application entry point."""
    appdata_base, local_base = setup_logging()
    logger.info('Starting Mil Mapper 2.0')

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
        # Copy default configs if not present
        install_dir = Path(sys.argv[0]).resolve().parent
        # В PyInstaller onedir сборке данные находятся в _internal/
        # В onefile или при разработке - рядом с exe/скриптом
        default_cfg_root = install_dir / '_internal' / 'configs'
        if not default_cfg_root.exists():
            default_cfg_root = install_dir / 'configs'

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
    # Обязательно для multiprocessing на Windows (особенно PyInstaller)
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())

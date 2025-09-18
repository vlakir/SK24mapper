"""
Diagnostic utilities.

This module monitors system resources and detects potential hanging issues.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import psutil
from datetime import timedelta
import contextlib

_PSUTIL_AVAILABLE = True

logger = logging.getLogger(__name__)

# --- Кэш/HTTP проверка для Terrain-RGB
try:
    from aiohttp_client_cache import CachedSession, SQLiteBackend
except Exception:  # pragma: no cover
    CachedSession = None  # type: ignore[assignment]
    SQLiteBackend = None  # type: ignore[assignment]


def get_memory_info() -> dict[str, Any]:
    """Get comprehensive memory usage information."""
    if not _PSUTIL_AVAILABLE:
        return {'error': 'psutil not available'}

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        return {
            'process_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'process_vms_mb': round(memory_info.vms / 1024 / 1024, 2),
            'system_total_mb': round(system_memory.total / 1024 / 1024, 2),
            'system_available_mb': round(system_memory.available / 1024 / 1024, 2),
            'system_used_percent': system_memory.percent,
            'process_memory_percent': round(process.memory_percent(), 2),
        }
    except Exception as e:
        return {'error': f'Failed to get memory info: {e}'}


def get_thread_info() -> dict[str, Any]:
    """Get information about active threads."""
    try:
        active_threads = threading.active_count()
        thread_names = [t.name for t in threading.enumerate()]

        info = {
            'active_count': active_threads,
            'thread_names': thread_names,
            'main_thread_alive': threading.main_thread().is_alive(),
        }

        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                info['system_threads'] = process.num_threads()
            except Exception as e:
                logger.debug(f'Failed to get system thread count: {e}')

        return info
    except Exception as e:
        return {'error': f'Failed to get thread info: {e}'}


def get_file_descriptor_info() -> dict[str, Any]:
    """Get information about open file descriptors."""
    if not _PSUTIL_AVAILABLE:
        return {'error': 'psutil not available'}

    try:
        process = psutil.Process()
        open_files = len(process.open_files())
        connections = len(process.connections())

        return {
            'open_files': open_files,
            'network_connections': connections,
            'pid': process.pid,
        }
    except Exception as e:
        return {'error': f'Failed to get file descriptor info: {e}'}


def get_sqlite_info() -> dict[str, Any]:
    """Get SQLite connection information from cache directories."""
    try:
        # Check cache directory for SQLite files
        repo_root = Path(__file__).resolve().parent.parent
        cache_dir = repo_root / '.cache'

        sqlite_files = []
        if cache_dir.exists():
            for sqlite_file in cache_dir.rglob('*.sqlite*'):
                try:
                    # Try to get file info
                    stat = sqlite_file.stat()
                    sqlite_files.append(
                        {
                            'file': str(sqlite_file),
                            'size_mb': round(stat.st_size / 1024 / 1024, 2),
                            'modified': time.ctime(stat.st_mtime),
                        },
                    )
                except Exception as e:
                    logger.debug(
                        f'Failed to get info for SQLite file {sqlite_file}: {e}',
                    )

        return {
            'cache_dir': str(cache_dir),
            'sqlite_files': sqlite_files,
            'total_files': len(sqlite_files),
        }
    except Exception as e:
        return {'error': f'Failed to get SQLite info: {e}'}


def get_system_load() -> dict[str, Any]:
    """Get system load and CPU information."""
    if not _PSUTIL_AVAILABLE:
        return {'error': 'psutil not available'}

    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None

        info = {
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
        }

        if load_avg:
            info['load_avg_1min'] = load_avg[0]
            info['load_avg_5min'] = load_avg[1]
            info['load_avg_15min'] = load_avg[2]

        return info
    except Exception as e:
        return {'error': f'Failed to get system load: {e}'}


def log_comprehensive_diagnostics(
    operation: str = 'general',
    level: int = logging.INFO,
) -> None:
    """Log comprehensive diagnostic information."""
    logger.log(level, f'=== DIAGNOSTIC INFO: {operation.upper()} ===')

    # Memory information
    memory_info = get_memory_info()
    logger.log(
        level,
        f'Memory - RSS: {memory_info.get("process_rss_mb", "N/A")}MB, '
        f'VMS: {memory_info.get("process_vms_mb", "N/A")}MB, '
        f'System Available: {memory_info.get("system_available_mb", "N/A")}MB '
        f'({memory_info.get("system_used_percent", "N/A")}% used)',
    )

    # Thread information
    thread_info = get_thread_info()
    logger.log(
        level,
        f'Threads - Active: {thread_info.get("active_count", "N/A")}, '
        f'System: {thread_info.get("system_threads", "N/A")}, '
        f'Main alive: {thread_info.get("main_thread_alive", "N/A")}',
    )

    # File descriptor information
    fd_info = get_file_descriptor_info()
    logger.log(
        level,
        f'Resources - Open files: {fd_info.get("open_files", "N/A")}, '
        f'Network connections: {fd_info.get("network_connections", "N/A")}',
    )

    # System load
    load_info = get_system_load()
    logger.log(
        level,
        f'System - CPU: {load_info.get("cpu_percent", "N/A")}%, '
        f'Load avg: {load_info.get("load_avg_1min", "N/A")}',
    )

    # SQLite cache info
    sqlite_info = get_sqlite_info()
    logger.log(level, f'SQLite - Cache files: {sqlite_info.get("total_files", "N/A")}')

    # Thread names for debugging
    thread_info = get_thread_info()
    if 'thread_names' in thread_info:
        logger.log(level, f'Active threads: {", ".join(thread_info["thread_names"])}')

    logger.log(level, f'=== END DIAGNOSTIC INFO: {operation.upper()} ===')


def log_memory_usage(context: str = '') -> None:
    """Quick memory usage logging."""
    memory_info = get_memory_info()
    logger.info(
        f'Memory usage{" (" + context + ")" if context else ""}: '
        f'RSS={memory_info.get("process_rss_mb", "N/A")}MB, '
        f'Available={memory_info.get("system_available_mb", "N/A")}MB',
    )


def log_thread_status(context: str = '') -> None:
    """Quick thread status logging."""
    thread_info = get_thread_info()
    logger.info(
        f'Thread status{" (" + context + ")" if context else ""}: '
        f'Active={thread_info.get("active_count", "N/A")}, '
        f'System={thread_info.get("system_threads", "N/A")}',
    )


async def run_deep_verification(*, api_key: str, settings: Any) -> None:
    """Глубокая проверка перед стартом тяжёлой обработки.

    Проверяет:
    - Наличие API-ключа.
    - Доступность соответствующего источника (Styles или Terrain-RGB) в зависимости от типа карты.
    - Возможность записи в каталог кэша HTTP.
    - Возможность записи в каталог вывода.
    - Для Terrain-RGB: пробную загрузку маленького тайла и декодирование DEM -> цвет.

    Не логирует токен; в логах — только пути без query.
    В случае проблем выбрасывает RuntimeError с понятным сообщением для пользователя.
    """
    from constants import (
        HTTP_CACHE_DIR,
        MapType,
        default_map_type,
        map_type_to_style_id,
        MAPBOX_STATIC_BASE,
        MAPBOX_TERRAIN_RGB_PATH,
    )
    from topography import (
        colorize_dem_to_image,
        decode_terrain_rgb_to_elevation_m,
    )
    import aiohttp
    import asyncio
    from PIL import Image
    import io

    # 1) API key presence
    if not api_key:
        raise RuntimeError('API-ключ не найден. Задайте переменную окружения API_KEY или файл .env/.secrets.env.')

    # 2) Resolve map type
    try:
        mt = getattr(settings, 'map_type', default_map_type())
        mt_enum = MapType(mt) if not isinstance(mt, MapType) else mt
    except Exception:
        mt_enum = default_map_type()

    # 3) Cache directory writability
    try:
        cache_dir = Path(HTTP_CACHE_DIR)
        if not cache_dir.is_absolute():
            # Mirror logic similar to service._resolve_cache_dir()
            local = os.getenv('LOCALAPPDATA')
            cache_dir = (Path(local) / 'SK42mapper' / '.cache' / 'tiles').resolve() if local else (Path.home() / '.sk42mapper_cache' / 'tiles').resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        test_file = cache_dir / '.write_test.tmp'
        test_file.write_text('ok', encoding='utf-8')
        test_file.unlink(missing_ok=True)
    except Exception:
        raise RuntimeError('Кэш недоступен для записи. Проверьте права на каталог кэша.')

    # 4) Output path directory writability
    try:
        out_path = Path(getattr(settings, 'output_path', '../maps/map.jpg'))
        out_dir = out_path.resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        test_file = out_dir / '.write_test.tmp'
        test_file.write_text('ok', encoding='utf-8')
        test_file.unlink(missing_ok=True)
    except Exception:
        raise RuntimeError('Каталог вывода недоступен для записи. Измените путь сохранения в настройках.')

    # 5) Network/source checks
    async def _check_styles(style_id: str) -> None:
        path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/256/0/0/0'
        url = f'{path}?access_token={api_key}'
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession() as client:
            async with client.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    await resp.read()
                    return
                if resp.status in (401, 403):
                    raise RuntimeError('Неверный или недействительный API-ключ. Проверьте ключ и попробуйте снова.')
                raise RuntimeError(f'Ошибка доступа к серверу карт (HTTP {resp.status}). Повторите попытку позже.')

    async def _check_terrain_small() -> None:
        # Try z=0/x=0/y=0
        path = f'{MAPBOX_TERRAIN_RGB_PATH}/0/0/0.pngraw'
        url = f'{path}?access_token={api_key}'
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession() as client:
            async with client.get(url, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.read()
                elif resp.status in (401, 403):
                    raise RuntimeError('Неверный или недействительный API-ключ. Проверьте ключ и попробуйте снова.')
                else:
                    raise RuntimeError(f'Ошибка доступа к серверу карт (HTTP {resp.status}). Повторите попытку позже.')
        # Decode tiny image and colorize to ensure pipeline works
        img = Image.open(io.BytesIO(data)).convert('RGB')
        dem = decode_terrain_rgb_to_elevation_m(img)
        _ = colorize_dem_to_image(dem)

    if mt_enum in (MapType.SATELLITE, MapType.HYBRID, MapType.STREETS, MapType.OUTDOORS):
        style_id = map_type_to_style_id(mt_enum) or map_type_to_style_id(default_map_type())
        logger.info('Глубокая проверка: стилевой режим %s, style_id=%s', mt_enum, style_id)
        try:
            await _check_styles(style_id)
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError('Не удалось выполнить сетевую проверку. Проверьте подключение к интернету.')
    elif mt_enum == MapType.ELEVATION_COLOR:
        logger.info('Глубокая проверка: Terrain-RGB (цветовая шкала)')
        try:
            await _check_terrain_small()
            # Дополнительно: глубокая проверка кэширования Terrain-RGB
            try:
                await verify_cache_for_terrain(api_key)
            except Exception as e:
                logger.warning('Проверка кэширования не удалась или пропущена: %s', e)
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError('Не удалось выполнить проверку Terrain-RGB. Проверьте интернет и повторите попытку.')
    else:
        # For unimplemented modes verify base style
        base_style = map_type_to_style_id(default_map_type())
        try:
            await _check_styles(base_style)
        except Exception:
            raise RuntimeError('Проверка базового стиля не удалась. Проверьте интернет и токен.')

    logger.info('Глубокая проверка успешно пройдена')


async def _make_cached_session_for_diag() -> Any:
    """Создаёт CachedSession с теми же параметрами, что в сервисе, для проверки кэша.
    Возвращает aiohttp.ClientSession, если кэш отключён или библиотека недоступна.
    """
    from constants import (
        HTTP_CACHE_DIR,
        HTTP_CACHE_ENABLED,
        HTTP_CACHE_EXPIRE_HOURS,
        HTTP_CACHE_RESPECT_HEADERS,
        HTTP_CACHE_STALE_IF_ERROR_HOURS,
    )
    if not HTTP_CACHE_ENABLED or CachedSession is None or SQLiteBackend is None:
        import aiohttp
        return aiohttp.ClientSession()

    # Разрешаем каталог кэша аналогично service._resolve_cache_dir
    raw_dir = Path(HTTP_CACHE_DIR)
    if raw_dir.is_absolute():
        cache_dir = raw_dir
    else:
        local = os.getenv('LOCALAPPDATA')
        cache_dir = (Path(local) / 'SK42mapper' / '.cache' / 'tiles').resolve() if local else (Path.home() / '.sk42mapper_cache' / 'tiles').resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / 'http_cache.sqlite'
    # Инициализация WAL
    try:
        import sqlite3
        if not cache_path.exists():
            with sqlite3.connect(cache_path) as _conn:
                _conn.execute('PRAGMA journal_mode=WAL;')
    except Exception:
        pass

    expire_td = timedelta(hours=max(0, int(HTTP_CACHE_EXPIRE_HOURS)))
    stale_hours = int(HTTP_CACHE_STALE_IF_ERROR_HOURS)
    stale_param = timedelta(hours=stale_hours) if stale_hours > 0 else False

    backend = SQLiteBackend(str(cache_path), expire_after=expire_td)
    return CachedSession(
        cache=backend,
        expire_after=expire_td,
        cache_control=bool(HTTP_CACHE_RESPECT_HEADERS),
        stale_if_error=stale_param,
    )


async def verify_cache_for_terrain(api_key: str) -> None:
    """Глубокая проверка кэширования Terrain‑RGB: два запроса одного и того же тайла.
    Логируются статус, признак from_cache и размер тела. Токен не печатается.
    """
    from constants import MAPBOX_TERRAIN_RGB_PATH, HTTP_CACHE_ENABLED
    import aiohttp

    if not HTTP_CACHE_ENABLED:
        logger.info('Кэширование HTTP отключено — проверка кэширования пропущена')
        return

    scale_suffix = ''  # retina disabled for elevation
    z, x, y = 0, 0, 0
    path = f'{MAPBOX_TERRAIN_RGB_PATH}/{z}/{x}/{y}{scale_suffix}.pngraw'
    url = f'{path}?access_token={api_key}'

    async with await _make_cached_session_for_diag() as client:  # type: ignore[misc]
        async def fetch(iter_no: int) -> tuple[int, int, bool, int]:
            try:
                resp = await client.get(url, timeout=aiohttp.ClientTimeout(total=10))
                try:
                    status = getattr(resp, 'status', 0)
                    from_cache = bool(getattr(resp, 'from_cache', False))
                    body = await resp.read()
                    size = len(body or b'')
                finally:
                    with contextlib.suppress(Exception):
                        close = getattr(resp, 'close', None)
                        if callable(close):
                            close()
                        release = getattr(resp, 'release', None)
                        if callable(release):
                            release()
                logger.info('Cache check iter=%s z/x/y=%s/%s/%s status=%s from_cache=%s size=%s bytes path=%s', iter_no, z, x, y, status, from_cache, size, path)
                return status, size, from_cache, iter_no
            except Exception as e:
                logger.warning('Cache check failed (iter=%s) for path=%s: %s', iter_no, path, e)
                return 0, 0, False, iter_no

        s1, b1, c1, _ = await fetch(1)
        s2, b2, c2, _ = await fetch(2)
        hits = int(c2 or s2 == 304)
        logger.info('Cache deep verification summary: second-iter hits/revalidated=%s/1; bytes first=%s, second=%s', hits, b1, b2)


def monitor_resource_changes(
    before_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monitor changes in resource usage."""
    current_info = {
        'memory': get_memory_info(),
        'threads': get_thread_info(),
        'files': get_file_descriptor_info(),
    }

    if before_info:
        changes: dict[str, dict[str, str]] = {}
        for category in current_info:
            if category in before_info:
                changes[category] = {}
                for key in current_info[category]:
                    if key in before_info[category]:
                        if isinstance(current_info[category][key], (int, float)):
                            diff = (
                                current_info[category][key] - before_info[category][key]
                            )
                            if diff != 0:
                                changes[category][key] = (
                                    f'{before_info[category][key]} -> {current_info[category][key]} ({diff:+})'
                                )

        if any(changes.values()):
            logger.info(f'Resource changes detected: {changes}')

    return current_info


class ResourceMonitor:
    """Context manager for monitoring resource usage during operations."""

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self.start_time = None
        self.start_resources = None

    def __enter__(self):
        self.start_time = time.time()
        self.start_resources = monitor_resource_changes()
        log_comprehensive_diagnostics(f'{self.operation_name} - START')
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        assert self.start_time is not None
        duration = time.time() - self.start_time
        logger.info(
            f"Operation '{self.operation_name}' completed in {duration:.2f} seconds",
        )

        # Log final resources and changes
        monitor_resource_changes(self.start_resources)
        log_comprehensive_diagnostics(f'{self.operation_name} - END')

        if exc_type:
            logger.error(
                f"Operation '{self.operation_name}' failed with "
                f'{exc_type.__name__}: {exc_val}',
            )


# Check if psutil is available and log warning if not
if not _PSUTIL_AVAILABLE:
    logger.warning(
        'psutil library not available - memory and system monitoring will be limited',
    )

"""Diagnostic utilities.

This module monitors system resources and detects potential hanging issues.
"""

import asyncio
import contextlib
import io
import logging
import os
import ssl
import threading
import time
import types
from pathlib import Path
from typing import Any

import aiohttp
import certifi
import psutil
from PIL import Image

from geo.topography import (
    colorize_dem_to_image,
    decode_terrain_rgb_to_elevation_m,
)
from shared.constants import (
    ELEVATION_USE_RETINA,
    HTTP_FORBIDDEN,
    HTTP_OK,
    HTTP_UNAUTHORIZED,
    MAPBOX_STATIC_BASE,
    MAPBOX_TERRAIN_RGB_PATH,
    TILE_CACHE_DIR,
    MapType,
    default_map_type,
    map_type_to_style_id,
)
from shared.constants import PSUTIL_AVAILABLE as _PSUTIL_AVAILABLE

logger = logging.getLogger(__name__)


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
            'system_available_mb': round(
                system_memory.available / 1024 / 1024,
                2,
            ),
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
                logger.debug('Failed to get system thread count: %s', e)
    except Exception as e:
        return {'error': f'Failed to get thread info: {e}'}
    else:
        return info


def get_file_descriptor_info() -> dict[str, Any]:
    """Get information about open file descriptors."""
    if not _PSUTIL_AVAILABLE:
        return {'error': 'psutil not available'}

    try:
        process = psutil.Process()
        open_files = len(process.open_files())
        try:
            connections = len(process.net_connections())
        except AttributeError:
            connections = len(process.connections())
    except Exception as e:
        return {'error': f'Failed to get file descriptor info: {e}'}
    else:
        return {
            'open_files': open_files,
            'network_connections': connections,
            'pid': process.pid,
        }


def get_sqlite_info() -> dict[str, Any]:
    """Get SQLite connection information from cache directories."""
    try:
        # Check tile cache directory for SQLite files
        cache_dir = Path(TILE_CACHE_DIR)
        if not cache_dir.is_absolute():
            local = os.getenv('LOCALAPPDATA')
            cache_dir = (
                (Path(local) / 'SK42mapper' / '.cache' / 'tiles').resolve()
                if local
                else (Path.home() / '.sk42mapper_cache' / 'tiles').resolve()
            )

        sqlite_files = []
        if cache_dir.exists():
            for sqlite_file in cache_dir.rglob('*.db'):
                try:
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
                        'Failed to get info for SQLite file %s: %s',
                        sqlite_file,
                        e,
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
    except Exception as e:
        return {'error': f'Failed to get system load: {e}'}
    else:
        return info


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
        'Memory - RSS: %sMB, VMS: %sMB, System Available: %sMB (%s%% used)',
        memory_info.get('process_rss_mb', 'N/A'),
        memory_info.get('process_vms_mb', 'N/A'),
        memory_info.get('system_available_mb', 'N/A'),
        memory_info.get('system_used_percent', 'N/A'),
    )

    # Thread information
    thread_info = get_thread_info()
    logger.log(
        level,
        'Threads - Active: %s, System: %s, Main alive: %s',
        thread_info.get('active_count', 'N/A'),
        thread_info.get('system_threads', 'N/A'),
        thread_info.get('main_thread_alive', 'N/A'),
    )

    # File descriptor information
    fd_info = get_file_descriptor_info()
    logger.log(
        level,
        'Resources - Open files: %s, Network connections: %s',
        fd_info.get('open_files', 'N/A'),
        fd_info.get('network_connections', 'N/A'),
    )

    # System load
    load_info = get_system_load()
    logger.log(
        level,
        'System - CPU: %s%%, Load avg: %s',
        load_info.get('cpu_percent', 'N/A'),
        load_info.get('load_avg_1min', 'N/A'),
    )

    # SQLite cache info
    sqlite_info = get_sqlite_info()
    logger.log(
        level,
        'SQLite - Cache files: %s',
        sqlite_info.get('total_files', 'N/A'),
    )

    # Thread names for debugging
    thread_info = get_thread_info()
    thread_names = thread_info.get('thread_names')
    if thread_names:
        logger.log(
            level,
            'Active threads: %s',
            ', '.join(thread_names),
        )

    logger.log(level, f'=== END DIAGNOSTIC INFO: {operation.upper()} ===')


def log_memory_usage(context: str = '') -> None:
    """Quick memory usage logging."""
    memory_info = get_memory_info()
    context_label = f' ({context})' if context else ''
    logger.info(
        'Memory usage%s: RSS=%sMB, Available=%sMB',
        context_label,
        memory_info.get('process_rss_mb', 'N/A'),
        memory_info.get('system_available_mb', 'N/A'),
    )


def log_thread_status(context: str = '') -> None:
    """Quick thread status logging."""
    thread_info = get_thread_info()
    context_label = f' ({context})' if context else ''
    logger.info(
        'Thread status%s: Active=%s, System=%s',
        context_label,
        thread_info.get('active_count', 'N/A'),
        thread_info.get('system_threads', 'N/A'),
    )


def _ensure_writable_dir(path: Path) -> None:
    """Ensure directory exists and is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / '.write_test.tmp'
        test_file.write_text('ok', encoding='utf-8')
        test_file.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError('Недоступна запись в каталог: ' + str(path)) from e


async def run_deep_verification(*, api_key: str, settings: object) -> None:
    """Deep verification before starting heavy processing.

    Checks:
    - API key presence
    - Source availability (Styles or Terrain-RGB) based on map type
    - Write access to tile cache directory
    - Write access to output directory
    - For Terrain-RGB: test download and DEM decode

    Does not log the token; paths in logs without query params.
    Raises RuntimeError with user-friendly message on failure.
    """
    # 1) API key presence
    if not api_key:
        msg = (
            'API-ключ не найден. Задайте переменную окружения API_KEY '
            'или файл .env/.secrets.env.'
        )
        raise RuntimeError(msg)

    # 2) Resolve map type
    try:
        mt = getattr(settings, 'map_type', default_map_type())
        mt_enum = MapType(mt) if not isinstance(mt, MapType) else mt
    except Exception:
        mt_enum = default_map_type()

    # 3) Cache directory writability
    cache_dir = Path(TILE_CACHE_DIR)
    if not cache_dir.is_absolute():
        local = os.getenv('LOCALAPPDATA')
        cache_dir = (
            (Path(local) / 'SK42mapper' / '.cache' / 'tiles').resolve()
            if local
            else (Path.home() / '.sk42mapper_cache' / 'tiles').resolve()
        )
    _ensure_writable_dir(cache_dir)

    # 4) Output path directory writability
    out_path = Path(getattr(settings, 'output_path', '../maps/map.jpg'))
    out_dir = out_path.resolve().parent
    _ensure_writable_dir(out_dir)

    # 5) Network/source checks
    async def _check_styles(style_id: str) -> None:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/256/0/0/0'
        url = f'{path}?access_token={api_key}'
        timeout = aiohttp.ClientTimeout(
            total=10,
            connect=10,
            sock_connect=10,
            sock_read=10,
        )
        attempts = 3
        async with aiohttp.ClientSession(connector=connector) as client:
            for i in range(attempts):
                try:
                    async with client.get(url, timeout=timeout) as resp:
                        if resp.status == HTTP_OK:
                            with contextlib.suppress(Exception):
                                await resp.read()
                            return
                        if resp.status in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                            msg = (
                                'Неверный или недействительный API-ключ. '
                                'Проверьте ключ и попробуйте снова.'
                            )
                            raise RuntimeError(msg)
                        msg = (
                            f'Ошибка доступа к серверу карт (HTTP {resp.status}). '
                            f'Повторите попытку позже.'
                        )
                        raise RuntimeError(msg)
                except asyncio.CancelledError:
                    raise
                except (TimeoutError, aiohttp.ClientError, OSError) as e:
                    if i < attempts - 1:
                        backoff = 0.5 * (2**i)
                        logger.warning(
                            'Проблема сети при проверке стиля (попытка %s/%s): %s; '
                            'повтор через %.1fs',
                            i + 1,
                            attempts,
                            e,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    msg = (
                        'Не удалось связаться с сервером карт. '
                        'Проверьте подключение к интернету и попробуйте снова.'
                    )
                    raise RuntimeError(msg) from e

    async def _check_terrain_small() -> None:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)

        path = f'{MAPBOX_TERRAIN_RGB_PATH}/0/0/0.pngraw'
        url = f'{path}?access_token={api_key}'
        timeout = aiohttp.ClientTimeout(
            total=10, connect=10, sock_connect=10, sock_read=10
        )
        attempts = 3
        last_exc: Exception | None = None
        async with aiohttp.ClientSession(connector=connector) as client:
            for i in range(attempts):
                try:
                    async with client.get(url, timeout=timeout) as resp:
                        if resp.status == HTTP_OK:
                            data = await resp.read()
                        elif resp.status in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                            msg = (
                                'Неверный или недействительный API-ключ. '
                                'Проверьте ключ и попробуйте снова.'
                            )
                            raise RuntimeError(msg)
                        else:
                            msg = (
                                f'Ошибка доступа к серверу карт (HTTP {resp.status}). '
                                f'Повторите попытку позже.'
                            )
                            raise RuntimeError(msg)
                        # Decode tiny image and colorize to ensure pipeline works
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                        dem = decode_terrain_rgb_to_elevation_m(img)
                        _ = colorize_dem_to_image(dem)
                        return
                except asyncio.CancelledError:
                    raise
                except (TimeoutError, aiohttp.ClientError, OSError) as e:
                    last_exc = e
                    if i < attempts - 1:
                        backoff = 0.5 * (2**i)
                        logger.warning(
                            'Проблема сети при проверке Terrain-RGB '
                            '(попытка %s/%s): %s; повтор через %.1fs',
                            i + 1,
                            attempts,
                            e,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
        msg = (
            'Не удалось выполнить проверку Terrain-RGB: сеть недоступна. '
            'Проверьте подключение к интернету.'
        )
        raise RuntimeError(msg) from last_exc

    if mt_enum in (
        MapType.SATELLITE,
        MapType.HYBRID,
        MapType.STREETS,
        MapType.OUTDOORS,
    ):
        style_id = map_type_to_style_id(mt_enum) or map_type_to_style_id(
            default_map_type()
        )
        logger.info(
            'Глубокая проверка: стилевой режим %s, style_id=%s', mt_enum, style_id
        )
        try:
            await _check_styles(style_id)
        except RuntimeError:
            raise
        except Exception as e:
            msg = (
                'Не удалось выполнить сетевую проверку. '
                'Проверьте подключение к интернету.'
            )
            raise RuntimeError(msg) from e
    elif mt_enum in (MapType.ELEVATION_COLOR, MapType.ELEVATION_CONTOURS):
        logger.info('Глубокая проверка: Terrain-RGB (%s)', mt_enum)
        try:
            await _check_terrain_small()
        except RuntimeError:
            raise
        except Exception as e:
            msg = (
                'Не удалось выполнить проверку Terrain-RGB. Проверьте интернет '
                'и повторите попытку.'
            )
            raise RuntimeError(msg) from e
    else:
        # For unimplemented modes verify base style
        base_style = map_type_to_style_id(default_map_type())
        try:
            await _check_styles(base_style)
        except Exception as e:
            msg = 'Проверка базового стиля не удалась. Проверьте интернет и токен.'
            raise RuntimeError(msg) from e

    logger.info('Глубокая проверка успешно пройдена')


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
        for category, cat_data in current_info.items():
            if category in before_info:
                changes[category] = {}
                before_cat = before_info[category]
                for key, val in cat_data.items():
                    if key in before_cat:
                        before_val = before_cat[key]
                        if isinstance(val, (int, float)):
                            diff = val - before_val
                            if diff != 0:
                                changes[category][key] = (
                                    f'{before_val} -> {val} ({diff:+})'
                                )
        if any(changes.values()):
            logger.info(f'Resource changes detected: {changes}')

    return current_info


class ResourceMonitor:
    """Context manager for monitoring resource usage during operations."""

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self.start_time: float | None = None
        self.start_resources: dict[str, Any] | None = None

    def __enter__(self) -> 'ResourceMonitor':
        start_time = time.time()
        self.start_time = start_time
        start_resources = monitor_resource_changes()
        self.start_resources = start_resources
        log_comprehensive_diagnostics(f'{self.operation_name} - START')
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        _ = (exc_type, exc_val, exc_tb)
        if self.start_time is None:
            msg = 'Unexpected missing start_time in ResourceMonitor'
            raise RuntimeError(msg)
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

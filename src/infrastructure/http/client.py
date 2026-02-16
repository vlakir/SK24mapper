from __future__ import annotations

import contextlib
import os
import sqlite3
import ssl
import time
from datetime import timedelta
from pathlib import Path

import aiohttp
import certifi
from aiohttp_client_cache import CachedSession, SQLiteBackend

from shared.constants import (
    HTTP_CACHE_DIR,
    HTTP_CACHE_ENABLED,
    HTTP_CACHE_EXPIRE_HOURS,
    HTTP_CACHE_RESPECT_HEADERS,
    HTTP_CACHE_STALE_IF_ERROR_HOURS,
    HTTP_FORBIDDEN,
    HTTP_OK,
    HTTP_UNAUTHORIZED,
    MAPBOX_STATIC_BASE,
    MAPBOX_TERRAIN_RGB_PATH,
)
from shared.portable import get_portable_path, is_portable_mode


def resolve_cache_dir() -> Path | None:
    # Portable режим: кэш в папке приложения
    if is_portable_mode():
        return get_portable_path('cache/tiles')

    # Обычный режим
    raw_dir = Path(HTTP_CACHE_DIR)
    if raw_dir.is_absolute():
        return raw_dir

    local = os.getenv('LOCALAPPDATA')
    if local:
        return (Path(local) / 'SK42' / '.cache' / 'tiles').resolve()
    # Fallback: user's home directory
    return (Path.home() / '.sk42mapper_cache' / 'tiles').resolve()


def cleanup_sqlite_cache(cache_dir: Path) -> None:
    """Force cleanup of SQLite cache connections."""
    cache_file = cache_dir / 'http_cache.sqlite'
    if cache_file.exists():
        # Close any remaining SQLite connections
        conn = sqlite3.connect(cache_file)
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE);')
        conn.close()

        time.sleep(0.1)


def make_http_session(cache_dir: Path | None) -> aiohttp.ClientSession:
    # Создать SSL-контекст с сертификатами из certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    use_cache = HTTP_CACHE_ENABLED
    if use_cache and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_dir is not None:
        cache_path = cache_dir / 'http_cache.sqlite'
        with contextlib.suppress(Exception):
            if not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(cache_path) as _conn:
                    _conn.execute('PRAGMA journal_mode=WAL;')
        expire_td = timedelta(hours=max(0, int(HTTP_CACHE_EXPIRE_HOURS)))
        stale_hours = int(HTTP_CACHE_STALE_IF_ERROR_HOURS)
        stale_param: bool | timedelta
        stale_param = timedelta(hours=stale_hours) if stale_hours > 0 else False
        backend = SQLiteBackend(str(cache_path), expire_after=expire_td)
        return CachedSession(
            cache=backend,
            connector=connector,
            expire_after=expire_td,
            cache_control=bool(HTTP_CACHE_RESPECT_HEADERS),
            stale_if_error=stale_param,
        )
    return aiohttp.ClientSession(connector=connector)


async def validate_style_api(api_key: str, style_id: str) -> None:
    """Проверяет доступность стилей Mapbox (Styles API tiles endpoint)."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    test_path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/256/0/0/0'
    test_url = f'{test_path}?access_token={api_key}'
    timeout = aiohttp.ClientTimeout(total=10, connect=10, sock_connect=10, sock_read=10)
    try:
        async with (
            aiohttp.ClientSession(connector=connector) as client,
            client.get(test_url, timeout=timeout) as resp,
        ):
            sc = resp.status
            if sc == HTTP_OK:
                with contextlib.suppress(Exception):
                    await resp.read()
                return
            if sc in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                msg = (
                    'Неверный или недействительный API-ключ. '
                    'Проверьте ключ и попробуйте снова.'
                )
                raise RuntimeError(msg)
            msg = f'Ошибка доступа к серверу карт (HTTP {sc}). Повторите попытку позже.'
            raise RuntimeError(msg)
    except (TimeoutError, aiohttp.ClientConnectorError, aiohttp.ClientOSError):
        msg = (
            'Нет соединения с интернетом или сервер недоступен. '
            'Проверьте подключение к сети.'
        )
        raise RuntimeError(msg) from None


async def validate_terrain_api(api_key: str) -> None:
    """Быстрая проверка доступности Terrain-RGB источника."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    test_path = f'{MAPBOX_TERRAIN_RGB_PATH}/0/0/0.pngraw'
    test_url = f'{test_path}?access_token={api_key}'
    timeout = aiohttp.ClientTimeout(total=10, connect=10, sock_connect=10, sock_read=10)
    try:
        async with (
            aiohttp.ClientSession(connector=connector) as client,
            client.get(test_url, timeout=timeout) as resp,
        ):
            sc = resp.status
            if sc == HTTP_OK:
                with contextlib.suppress(Exception):
                    await resp.read()
                return
            if sc in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                msg = (
                    'Неверный или недействительный API-ключ. '
                    'Проверьте ключ и попробуйте снова.'
                )
                raise RuntimeError(msg)
            msg = f'Ошибка доступа к серверу карт (HTTP {sc}). Повторите попытку позже.'
            raise RuntimeError(msg)
    except (TimeoutError, aiohttp.ClientConnectorError, aiohttp.ClientOSError):
        msg = (
            'Нет соединения с интернетом или сервер недоступен. '
            'Проверьте подключение к сети.'
        )
        raise RuntimeError(msg) from None

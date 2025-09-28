from __future__ import annotations

import contextlib
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import aiohttp
from aiohttp_client_cache import CachedSession, SQLiteBackend

from constants import (
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


def resolve_cache_dir() -> Path | None:
    raw_dir = Path(HTTP_CACHE_DIR)
    if raw_dir.is_absolute():
        return raw_dir
    # Prefer user LOCALAPPDATA for writable cache dir
    import os

    local = os.getenv('LOCALAPPDATA')
    if local:
        return (Path(local) / 'SK42mapper' / '.cache' / 'tiles').resolve()
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
            expire_after=expire_td,
            cache_control=bool(HTTP_CACHE_RESPECT_HEADERS),
            stale_if_error=stale_param,
        )
    return aiohttp.ClientSession()


async def validate_style_api(api_key: str, style_id: str) -> None:
    """Проверяет доступность стилей Mapbox (Styles API tiles endpoint)."""
    test_path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/256/0/0/0'
    test_url = f'{test_path}?access_token={api_key}'
    timeout = aiohttp.ClientTimeout(total=10, connect=10, sock_connect=10, sock_read=10)
    try:
        async with (
            aiohttp.ClientSession() as client,
            client.get(test_url, timeout=timeout) as resp,
        ):
            sc = resp.status
            if sc == HTTP_OK:
                with contextlib.suppress(Exception):
                    await resp.read()
                return
            if sc in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                msg = 'Неверный или недействительный API-ключ. Проверьте ключ и попробуйте снова.'
                raise RuntimeError(msg)
            msg = f'Ошибка доступа к серверу карт (HTTP {sc}). Повторите попытку позже.'
            raise RuntimeError(msg)
    except (TimeoutError, aiohttp.ClientConnectorError, aiohttp.ClientOSError):
        msg = 'Нет соединения с интернетом или сервер недоступен. Проверьте подключение к сети.'
        raise RuntimeError(msg) from None


async def validate_terrain_api(api_key: str) -> None:
    """Быстрая проверка доступности Terrain-RGB источника."""
    test_path = f'{MAPBOX_TERRAIN_RGB_PATH}/0/0/0.pngraw'
    test_url = f'{test_path}?access_token={api_key}'
    timeout = aiohttp.ClientTimeout(total=10, connect=10, sock_connect=10, sock_read=10)
    try:
        async with (
            aiohttp.ClientSession() as client,
            client.get(test_url, timeout=timeout) as resp,
        ):
            sc = resp.status
            if sc == HTTP_OK:
                with contextlib.suppress(Exception):
                    await resp.read()
                return
            if sc in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                msg = 'Неверный или недействительный API-ключ. Проверьте ключ и попробуйте снова.'
                raise RuntimeError(msg)
            msg = f'Ошибка доступа к серверу карт (HTTP {sc}). Повторите попытку позже.'
            raise RuntimeError(msg)
    except (TimeoutError, aiohttp.ClientConnectorError, aiohttp.ClientOSError):
        msg = 'Нет соединения с интернетом или сервер недоступен. Проверьте подключение к сети.'
        raise RuntimeError(msg) from None

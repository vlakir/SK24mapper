"""HTTP client utilities for Mapbox API access.

Note: Tile caching is now handled by tiles.cache.TileCache.
This module only provides basic HTTP session creation and API validation.
"""

from __future__ import annotations

import contextlib
import os
import ssl
from pathlib import Path

import aiohttp
import certifi

from shared.constants import (
    HTTP_FORBIDDEN,
    HTTP_OK,
    HTTP_UNAUTHORIZED,
    MAPBOX_STATIC_BASE,
    MAPBOX_TERRAIN_RGB_PATH,
    TILE_CACHE_DIR,
)


def resolve_cache_dir() -> Path:
    """Resolve tile cache directory path.

    Returns:
        Path to the tile cache directory.
    """
    raw_dir = Path(TILE_CACHE_DIR)
    if raw_dir.is_absolute():
        return raw_dir

    local = os.getenv('LOCALAPPDATA')
    if local:
        return (Path(local) / 'SK42mapper' / '.cache' / 'tiles').resolve()
    # Fallback: user's home directory
    return (Path.home() / '.sk42mapper_cache' / 'tiles').resolve()


def cleanup_sqlite_cache(cache_dir: Path) -> None:
    """No-op for backward compatibility.

    Tile cache cleanup is now handled by TileCache.cleanup_lru().
    """
    pass


def make_http_session(cache_dir: Path | None = None) -> aiohttp.ClientSession:
    """Create aiohttp session for API requests.

    Note: HTTP response caching is no longer used. Tile caching is handled
    by TileCache at the tile level, which is more efficient.

    Args:
        cache_dir: Ignored (kept for backward compatibility).

    Returns:
        aiohttp.ClientSession configured with SSL.
    """
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    return aiohttp.ClientSession(connector=connector)


async def validate_style_api(api_key: str, style_id: str) -> None:
    """Validate Mapbox Styles API access.

    Args:
        api_key: Mapbox API access token.
        style_id: Style ID to validate (e.g., 'mapbox/satellite-v9').

    Raises:
        RuntimeError: If API is not accessible or credentials are invalid.
    """
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
    """Validate Mapbox Terrain-RGB API access.

    Args:
        api_key: Mapbox API access token.

    Raises:
        RuntimeError: If API is not accessible or credentials are invalid.
    """
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

"""HTTP fetcher with caching integration.

This module provides TileFetcher class that downloads tiles from
Mapbox API with automatic caching, TTL checking, and retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from http import HTTPStatus
from io import BytesIO
from typing import TYPE_CHECKING

import aiohttp
from PIL import Image

from shared.constants import (
    HTTP_5XX_MAX,
    HTTP_5XX_MIN,
    HTTP_BACKOFF_FACTOR,
    HTTP_RETRIES_DEFAULT,
    HTTP_TIMEOUT_DEFAULT,
    MAPBOX_STATIC_BASE,
    MAPBOX_TERRAIN_RGB_PATH,
    TILE_CACHE_TTL_HOURS,
    TILE_SIZE,
    TILE_SIZE_512,
    TILE_SOURCE_SATELLITE,
    TILE_SOURCE_TERRAIN_RGB,
)

if TYPE_CHECKING:
    from tiles.cache import TileCache
    from tiles.writer import CacheWriter

logger = logging.getLogger(__name__)


class TileFetcher:
    """Tile fetcher with caching integration.

    Features:
    - Checks cache before making HTTP requests
    - TTL-based cache validation
    - Automatic retry with exponential backoff
    - Async writes via CacheWriter
    - Supports XYZ (satellite/hybrid/streets) and terrain-RGB tiles

    Usage:
        cache = TileCache()
        writer = CacheWriter(cache)
        writer.start()

        fetcher = TileFetcher(
            cache=cache,
            writer=writer,
            api_key='your-mapbox-token',
        )

        async with aiohttp.ClientSession() as session:
            # Fetch XYZ tile
            img = await fetcher.fetch_xyz(
                client=session,
                style_id='mapbox/satellite-v9',
                z=15, x=100, y=200,
            )

            # Fetch terrain-RGB tile
            img = await fetcher.fetch_terrain(
                client=session,
                z=15, x=100, y=200,
            )

        writer.stop()
        cache.close()
    """

    def __init__(
        self,
        cache: TileCache,
        writer: CacheWriter | None = None,
        api_key: str = '',
        ttl_hours: int | None = None,
        retries: int = HTTP_RETRIES_DEFAULT,
        backoff: float = HTTP_BACKOFF_FACTOR,
        timeout: float = HTTP_TIMEOUT_DEFAULT,
        offline: bool = False,
    ) -> None:
        """Initialize tile fetcher.

        Args:
            cache: TileCache instance for reading cached tiles.
            writer: CacheWriter for async cache writes. If None, writes directly.
            api_key: Mapbox API access token.
            ttl_hours: Cache TTL in hours. Defaults to TILE_CACHE_TTL_HOURS.
            retries: Number of retry attempts.
            backoff: Backoff factor for exponential delay.
            timeout: Request timeout in seconds.
            offline: If True, only use cached tiles without network requests.
        """
        self.cache = cache
        self.writer = writer
        self.api_key = api_key
        self.ttl_hours = ttl_hours if ttl_hours is not None else TILE_CACHE_TTL_HOURS
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.offline = offline

        # Statistics
        self._stats_cache_hits = 0
        self._stats_cache_misses = 0
        self._stats_downloads = 0
        self._stats_errors = 0
        self._stats_offline_misses = 0

    @property
    def stats(self) -> dict:
        """Get fetcher statistics."""
        return {
            'cache_hits': self._stats_cache_hits,
            'cache_misses': self._stats_cache_misses,
            'downloads': self._stats_downloads,
            'errors': self._stats_errors,
            'offline_misses': self._stats_offline_misses,
        }

    def _is_expired(self, fetched_at: int) -> bool:
        """Check if tile is expired based on TTL."""
        if self.ttl_hours <= 0:
            return False
        now = int(time.time())
        ttl_seconds = self.ttl_hours * 3600
        return (now - fetched_at) > ttl_seconds

    def _get_source_for_style(self, style_id: str) -> str:
        """Get source name from style ID."""
        if 'satellite' in style_id.lower():
            return TILE_SOURCE_SATELLITE
        # For other styles, use the style_id itself as source
        return style_id.replace('/', '-')

    async def fetch_xyz(
        self,
        client: aiohttp.ClientSession,
        style_id: str,
        z: int,
        x: int,
        y: int,
        *,
        tile_size: int = TILE_SIZE_512,
        use_retina: bool = True,
        force_download: bool = False,
    ) -> Image.Image:
        """Fetch XYZ tile from cache or API.

        Args:
            client: aiohttp session for making requests.
            style_id: Mapbox style ID (e.g., 'mapbox/satellite-v9').
            z: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            tile_size: Base tile size (256 or 512).
            use_retina: Whether to request @2x tiles.
            force_download: Skip cache and always download.

        Returns:
            PIL Image in RGB mode.

        Raises:
            RuntimeError: If download fails after retries.
        """
        source = self._get_source_for_style(style_id)

        # Try cache first
        if not force_download:
            cached = self._get_cached(z, x, y, source)
            if cached is not None:
                return Image.open(BytesIO(cached)).convert('RGB')

        # In offline mode, return placeholder or raise error
        if self.offline:
            self._stats_offline_misses += 1
            logger.warning('Offline mode: tile z%d/%d/%d (%s) not in cache', z, x, y, source)
            return self._create_placeholder_tile()

        # Download from API
        data = await self._download_xyz(
            client=client,
            style_id=style_id,
            z=z,
            x=x,
            y=y,
            tile_size=tile_size,
            use_retina=use_retina,
        )

        # Write to cache
        self._write_to_cache(z, x, y, source, data)

        return Image.open(BytesIO(data)).convert('RGB')

    async def fetch_terrain(
        self,
        client: aiohttp.ClientSession,
        z: int,
        x: int,
        y: int,
        *,
        use_retina: bool = True,
        force_download: bool = False,
    ) -> Image.Image:
        """Fetch terrain-RGB tile from cache or API.

        Args:
            client: aiohttp session for making requests.
            z: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            use_retina: Whether to request @2x tiles.
            force_download: Skip cache and always download.

        Returns:
            PIL Image in RGB mode (terrain-RGB encoded).

        Raises:
            RuntimeError: If download fails after retries.
        """
        source = TILE_SOURCE_TERRAIN_RGB

        # Try cache first
        if not force_download:
            cached = self._get_cached(z, x, y, source)
            if cached is not None:
                return Image.open(BytesIO(cached)).convert('RGB')

        # In offline mode, return placeholder or raise error
        if self.offline:
            self._stats_offline_misses += 1
            logger.warning('Offline mode: terrain tile z%d/%d/%d not in cache', z, x, y)
            return self._create_placeholder_terrain_tile()

        # Download from API
        data = await self._download_terrain(
            client=client,
            z=z,
            x=x,
            y=y,
            use_retina=use_retina,
        )

        # Write to cache
        self._write_to_cache(z, x, y, source, data)

        return Image.open(BytesIO(data)).convert('RGB')

    async def fetch_xyz_raw(
        self,
        client: aiohttp.ClientSession,
        style_id: str,
        z: int,
        x: int,
        y: int,
        *,
        tile_size: int = TILE_SIZE_512,
        use_retina: bool = True,
        force_download: bool = False,
    ) -> bytes:
        """Fetch XYZ tile raw bytes from cache or API.

        Similar to fetch_xyz but returns raw bytes without decoding.
        Useful when you need to store/forward the data.

        Returns:
            Raw tile bytes (JPEG/PNG/WebP).
        """
        source = self._get_source_for_style(style_id)

        if not force_download:
            cached = self._get_cached(z, x, y, source)
            if cached is not None:
                return cached

        # In offline mode, return placeholder bytes
        if self.offline:
            self._stats_offline_misses += 1
            logger.warning('Offline mode: tile z%d/%d/%d (%s) not in cache', z, x, y, source)
            return self._create_placeholder_tile_bytes()

        data = await self._download_xyz(
            client=client,
            style_id=style_id,
            z=z,
            x=x,
            y=y,
            tile_size=tile_size,
            use_retina=use_retina,
        )

        self._write_to_cache(z, x, y, source, data)
        return data

    async def fetch_terrain_raw(
        self,
        client: aiohttp.ClientSession,
        z: int,
        x: int,
        y: int,
        *,
        use_retina: bool = True,
        force_download: bool = False,
    ) -> bytes:
        """Fetch terrain-RGB tile raw bytes from cache or API.

        Returns:
            Raw PNG bytes.
        """
        source = TILE_SOURCE_TERRAIN_RGB

        if not force_download:
            cached = self._get_cached(z, x, y, source)
            if cached is not None:
                return cached

        # In offline mode, return placeholder bytes
        if self.offline:
            self._stats_offline_misses += 1
            logger.warning('Offline mode: terrain tile z%d/%d/%d not in cache', z, x, y)
            return self._create_placeholder_terrain_bytes()

        data = await self._download_terrain(
            client=client,
            z=z,
            x=x,
            y=y,
            use_retina=use_retina,
        )

        self._write_to_cache(z, x, y, source, data)
        return data

    def _get_cached(self, z: int, x: int, y: int, source: str) -> bytes | None:
        """Get tile from cache if valid."""
        info = self.cache.get_info(z, x, y, source)
        if info is None:
            self._stats_cache_misses += 1
            return None

        if self._is_expired(info.fetched_at):
            logger.debug('Tile z%d/%d/%d (%s) expired', z, x, y, source)
            self._stats_cache_misses += 1
            return None

        data = self.cache.get(z, x, y, source)
        if data is not None:
            self._stats_cache_hits += 1
        return data

    def _write_to_cache(self, z: int, x: int, y: int, source: str, data: bytes) -> None:
        """Write tile to cache via writer or directly."""
        if self.writer is not None and self.writer.is_running():
            self.writer.put(zoom=z, x=x, y=y, source=source, data=data)
        else:
            self.cache.put(zoom=z, x=x, y=y, source=source, data=data)

    async def _download_xyz(
        self,
        client: aiohttp.ClientSession,
        style_id: str,
        z: int,
        x: int,
        y: int,
        tile_size: int,
        use_retina: bool,
    ) -> bytes:
        """Download XYZ tile from Mapbox API."""
        ts = TILE_SIZE_512 if tile_size >= TILE_SIZE_512 else TILE_SIZE
        scale_suffix = '@2x' if use_retina else ''
        path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/{ts}/{z}/{x}/{y}{scale_suffix}'
        url = f'{path}?access_token={self.api_key}'

        return await self._download_with_retry(client, url, path, 'XYZ', z, x, y)

    async def _download_terrain(
        self,
        client: aiohttp.ClientSession,
        z: int,
        x: int,
        y: int,
        use_retina: bool,
    ) -> bytes:
        """Download terrain-RGB tile from Mapbox API."""
        scale_suffix = '@2x' if use_retina else ''
        path = f'{MAPBOX_TERRAIN_RGB_PATH}/{z}/{x}/{y}{scale_suffix}.pngraw'
        url = f'{path}?access_token={self.api_key}'

        return await self._download_with_retry(client, url, path, 'terrain', z, x, y)

    async def _download_with_retry(
        self,
        client: aiohttp.ClientSession,
        url: str,
        path: str,
        tile_type: str,
        z: int,
        x: int,
        y: int,
    ) -> bytes:
        """Download with retry and exponential backoff."""
        last_exc: Exception | None = None

        for attempt in range(self.retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                resp = await client.get(url, timeout=timeout)
                try:
                    sc = resp.status
                    if sc == HTTPStatus.OK:
                        data = await resp.read()
                        self._stats_downloads += 1
                        return data

                    if sc in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                        msg = (
                            f'Access denied (HTTP {sc}) for {tile_type} tile '
                            f'z/x/y={z}/{x}/{y}. Check API token.'
                        )
                        self._stats_errors += 1
                        raise RuntimeError(msg)

                    if sc == HTTPStatus.NOT_FOUND:
                        msg = f'Not found (404) for {tile_type} tile z/x/y={z}/{x}/{y}'
                        self._stats_errors += 1
                        raise RuntimeError(msg)

                    is_retryable = (sc == HTTPStatus.TOO_MANY_REQUESTS) or (
                        HTTP_5XX_MIN <= sc < HTTP_5XX_MAX
                    )
                    if is_retryable:
                        last_exc = RuntimeError(
                            f'HTTP {sc} for {tile_type} tile z/x/y={z}/{x}/{y}'
                        )
                    else:
                        msg = f'Unexpected HTTP {sc} for {tile_type} tile z/x/y={z}/{x}/{y}'
                        self._stats_errors += 1
                        raise RuntimeError(msg)

                finally:
                    with suppress(Exception):
                        close = getattr(resp, 'close', None)
                        if callable(close):
                            close()
                        release = getattr(resp, 'release', None)
                        if callable(release):
                            release()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e

            # Exponential backoff before retry
            if attempt < self.retries - 1:
                delay = self.backoff * (2 ** attempt)
                logger.warning(
                    '%s tile z/x/y=%d/%d/%d: attempt %d failed, retrying in %.1fs',
                    tile_type,
                    z,
                    x,
                    y,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        self._stats_errors += 1
        msg = f'Failed to download {tile_type} tile z/x/y={z}/{x}/{y} after {self.retries} attempts'
        if last_exc:
            raise RuntimeError(msg) from last_exc
        raise RuntimeError(msg)

    def _create_placeholder_tile(self) -> Image.Image:
        """Create a placeholder tile for offline mode (gray with pattern)."""
        # Gray tile with diagonal lines pattern
        img = Image.new('RGB', (512, 512), color=(180, 180, 180))
        return img

    def _create_placeholder_terrain_tile(self) -> Image.Image:
        """Create a placeholder terrain tile for offline mode.

        Terrain-RGB encoding: elevation = -10000 + (R*256*256 + G*256 + B) * 0.1
        For sea level (0m): R*256*256 + G*256 + B = 100000
        That gives approximately R=1, G=134, B=160 for 0m elevation
        """
        # Encode 0m elevation in terrain-RGB format
        img = Image.new('RGB', (512, 512), color=(1, 134, 160))
        return img

    def _create_placeholder_tile_bytes(self) -> bytes:
        """Create placeholder tile as PNG bytes."""
        img = self._create_placeholder_tile()
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def _create_placeholder_terrain_bytes(self) -> bytes:
        """Create placeholder terrain tile as PNG bytes."""
        img = self._create_placeholder_terrain_tile()
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def check_tiles_in_cache(
        self,
        tiles: list[tuple[int, int]],
        zoom: int,
        source: str,
    ) -> tuple[int, int]:
        """Check how many tiles are available in cache.

        Args:
            tiles: List of (x, y) tile coordinates.
            zoom: Zoom level.
            source: Tile source name.

        Returns:
            Tuple of (cached_count, total_count).
        """
        cached = 0
        for x, y in tiles:
            if self.cache.get_info(zoom, x, y, source) is not None:
                cached += 1
        return cached, len(tiles)

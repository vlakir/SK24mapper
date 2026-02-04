"""Terrain-RGB tile provider with DEM decoding.

Uses TileFetcher for all tile fetching and caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

import aiohttp
from PIL import Image

from geo.topography import decode_terrain_rgb_to_elevation_m

if TYPE_CHECKING:
    from tiles.fetcher import TileFetcher


@dataclass(frozen=True)
class TileKey:
    """Key for tile identification in memory cache."""

    z: int
    x: int
    y: int
    retina: bool


class ElevationTileProvider:
    """Provider for Terrain-RGB tiles with DEM decoding.

    Uses TileFetcher for tile fetching (with SQLite caching).
    Maintains in-memory cache for decoded DEM data.

    Usage:
        provider = ElevationTileProvider(
            client=session,
            tile_fetcher=fetcher,
            use_retina=True,
        )
        img = await provider.get_tile_image(z=15, x=100, y=200)
        dem = await provider.get_tile_dem(z=15, x=100, y=200)
    """

    def __init__(
        self,
        client: aiohttp.ClientSession,
        tile_fetcher: TileFetcher,
        *,
        use_retina: bool = True,
        max_mem_tiles: int = 512,
        # Legacy parameters (ignored, kept for backward compatibility)
        api_key: str = '',
        cache_root: str | None = None,
    ) -> None:
        """Initialize elevation tile provider.

        Args:
            client: aiohttp session for HTTP requests.
            tile_fetcher: TileFetcher instance for fetching tiles.
            use_retina: Whether to request @2x tiles.
            max_mem_tiles: Maximum tiles to keep in memory cache.
            api_key: Ignored (TileFetcher has its own api_key).
            cache_root: Ignored (TileFetcher uses TileCache).
        """
        self.client = client
        self.tile_fetcher = tile_fetcher
        self.use_retina = bool(use_retina)
        self._mem_raw: dict[TileKey, bytes] = {}
        self._mem_lru: list[TileKey] = []
        self._mem_dem: dict[TileKey, list[list[float]]] = {}
        self._max_mem = max(16, int(max_mem_tiles))

    def _key(self, z: int, x: int, y: int) -> TileKey:
        """Create tile key for memory cache."""
        return TileKey(int(z), int(x), int(y), self.use_retina)

    def _remember_raw(self, key: TileKey, data: bytes) -> None:
        """Store raw tile data in memory cache with LRU eviction."""
        self._mem_raw[key] = data
        self._mem_lru.append(key)
        if len(self._mem_lru) > self._max_mem:
            old = self._mem_lru.pop(0)
            self._mem_raw.pop(old, None)
            self._mem_dem.pop(old, None)

    def _touch(self, key: TileKey) -> None:
        """Move key to end of LRU list."""
        if key in self._mem_lru:
            self._mem_lru.remove(key)
            self._mem_lru.append(key)

    async def _fetch_raw(self, key: TileKey) -> bytes:
        """Fetch raw tile data via TileFetcher."""
        data = await self.tile_fetcher.fetch_terrain_raw(
            client=self.client,
            z=key.z,
            x=key.x,
            y=key.y,
            use_retina=key.retina,
        )
        self._remember_raw(key, data)
        return data

    async def get_tile_image(self, z: int, x: int, y: int) -> Image.Image:
        """Get terrain tile as PIL Image.

        Args:
            z: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.

        Returns:
            PIL Image in RGB mode.
        """
        key = self._key(z, x, y)
        raw = self._mem_raw.get(key)
        if raw is None:
            raw = await self._fetch_raw(key)
        else:
            self._touch(key)
        return Image.open(BytesIO(raw)).convert('RGB')

    async def get_tile_dem(self, z: int, x: int, y: int) -> list[list[float]]:
        """Get decoded DEM (elevation in meters) from terrain tile.

        Args:
            z: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.

        Returns:
            2D list of elevation values in meters.
        """
        key = self._key(z, x, y)
        dem = self._mem_dem.get(key)
        if dem is not None:
            self._touch(key)
            return dem
        img = await self.get_tile_image(z, x, y)
        dem = decode_terrain_rgb_to_elevation_m(img)
        self._mem_dem[key] = dem
        return dem

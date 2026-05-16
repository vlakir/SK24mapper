"""
Process-wide in-memory LRU cache for raw tile bytes.

Sits in front of aiohttp_client_cache (SQLite). Hit path: dict lookup
+ move_to_end. Miss path: bypass, normal HTTP/SQLite fetch happens, then
the bytes are put here. Bytes-level (not decoded Image) keeps memory
low — ~50-200 KB per tile, 500 entries ≈ 25-100 MB.

Single-event-loop access only (no lock); safe for the asyncio model used
by the map download service.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

DEFAULT_MAX_ENTRIES = 500


@dataclass(frozen=True)
class TileKey:
    source: str  # style_id for XYZ, 'terrain_rgb' for elevation
    z: int
    x: int
    y: int
    tile_size: int
    retina: bool


class TileMemoryCache:
    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._cache: OrderedDict[TileKey, bytes] = OrderedDict()
        self._max = max_entries
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: TileKey) -> bytes | None:
        data = self._cache.get(key)
        if data is None:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return data

    def put(self, key: TileKey, data: bytes) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = data
            return
        self._cache[key] = data
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
            self._evictions += 1

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
        }


_global_cache: TileMemoryCache | None = None


def get_global_cache() -> TileMemoryCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = TileMemoryCache(DEFAULT_MAX_ENTRIES)
    return _global_cache

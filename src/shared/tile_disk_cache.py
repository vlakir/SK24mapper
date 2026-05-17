"""
On-disk per-tile bytes cache for XYZ map tiles.

Mirrors the pattern used by ElevationTileProvider for terrain_rgb: store
the raw tile bytes (whatever the server returned — png/jpg/webp) in a flat
directory tree, keyed by (style, tile_size, retina, z, x, y). Survives
across worker processes and app restarts; the OS file cache keeps the
hot working set in RAM, so warm reads are tens of microseconds.

This lives in front of aiohttp_client_cache to avoid the per-tile SQLite
lock + CachedResponse pickling that profiling identified as the ~210ms
per-tile warm-rebuild bottleneck.
"""

from __future__ import annotations

from pathlib import Path

from infrastructure.http.client import resolve_cache_dir
from shared.constants import HTTP_CACHE_ENABLED


def xyz_disk_path(
    *,
    style_id: str,
    tile_size: int,
    use_retina: bool,
    z: int,
    x: int,
    y: int,
) -> Path | None:
    """
    Return disk path for an XYZ tile, or None if disk caching is disabled.
    """
    if not HTTP_CACHE_ENABLED:
        return None
    root = resolve_cache_dir()
    if root is None:
        return None
    style_safe = style_id.replace('/', '__').replace('\\', '__')
    size_seg = f'{tile_size}@2x' if use_retina else f'{tile_size}'
    return root / 'xyz' / style_safe / size_seg / str(z) / str(x) / f'{y}.bin'


def write_tile_bytes(path: Path, data: bytes) -> None:
    """Write tile bytes to disk, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish: write to tmp then rename to avoid partial file on crash.
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)

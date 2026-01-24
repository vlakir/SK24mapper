"""
Tile fetching utilities for map download service.

This module contains helper functions for downloading and processing map tiles.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from shared.constants import CONTOUR_LOG_MEMORY_EVERY_TILES
from shared.diagnostics import log_memory_usage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import aiohttp
    from PIL import Image

logger = logging.getLogger(__name__)


async def fetch_xyz_tiles_batch(
    client: aiohttp.ClientSession,
    tiles: list[tuple[int, int]],
    *,
    api_key: str,
    style_id: str,
    zoom: int,
    tile_size: int,
    use_retina: bool,
    semaphore: asyncio.Semaphore,
    on_progress: Callable[[int], Awaitable[None]] | None = None,
) -> list[tuple[int, Image.Image]]:
    """
    Fetch a batch of XYZ tiles concurrently.

    Args:
        client: HTTP client session
        tiles: List of (x, y) tile coordinates
        api_key: Mapbox API key
        style_id: Mapbox style ID
        zoom: Zoom level
        tile_size: Tile size in pixels
        use_retina: Whether to use retina tiles
        semaphore: Concurrency limiter
        on_progress: Optional callback for progress updates

    Returns:
        List of (index, image) tuples sorted by index

    """
    from geo.topography import async_fetch_xyz_tile

    tile_count = 0

    async def bound_fetch(
        idx_xy: tuple[int, tuple[int, int]],
    ) -> tuple[int, Image.Image]:
        nonlocal tile_count
        idx, (tx, ty) = idx_xy
        async with semaphore:
            img = await async_fetch_xyz_tile(
                client=client,
                api_key=api_key,
                style_id=style_id,
                tile_size=tile_size,
                z=zoom,
                x=tx,
                y=ty,
                use_retina=use_retina,
            )
            if on_progress:
                await on_progress(1)
            tile_count += 1
            if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                log_memory_usage(f'after {tile_count} tiles')
            return idx, img

    tasks = [bound_fetch(pair) for pair in enumerate(tiles)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda t: t[0])
    return results

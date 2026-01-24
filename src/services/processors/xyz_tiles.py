"""XYZ tiles processor - standard map tiles (satellite, hybrid, outdoors)."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from geo.topography import async_fetch_xyz_tile
from imaging import assemble_and_crop
from shared.constants import (
    CONTOUR_LOG_MEMORY_EVERY_TILES,
    XYZ_TILE_SIZE,
    XYZ_USE_RETINA,
)
from shared.diagnostics import log_memory_usage, log_thread_status
from shared.progress import ConsoleProgress

if TYPE_CHECKING:
    from PIL import Image

    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def process_xyz_tiles(ctx: MapDownloadContext) -> Image.Image:
    """
    Process standard XYZ tiles (satellite, hybrid, streets, outdoors).

    Args:
        ctx: Map download context with all necessary parameters.

    Returns:
        Assembled and cropped image from XYZ tiles.

    """
    tile_progress = ConsoleProgress(total=len(ctx.tiles), label='Загрузка XYZ-тайлов')
    tile_count = 0

    async def bound_fetch(
        idx_xy: tuple[int, tuple[int, int]],
    ) -> tuple[int, Image.Image]:
        nonlocal tile_count
        idx, (tx, ty) = idx_xy
        async with ctx.semaphore:
            img = await async_fetch_xyz_tile(
                client=ctx.client,
                api_key=ctx.api_key,
                style_id=ctx.style_id,
                tile_size=XYZ_TILE_SIZE,
                z=ctx.zoom,
                x=tx,
                y=ty,
                use_retina=XYZ_USE_RETINA,
            )
            await tile_progress.step(1)
            tile_count += 1
            if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                log_memory_usage(f'after {tile_count} tiles')
            return idx, img

    import asyncio

    tasks = [bound_fetch(pair) for pair in enumerate(ctx.tiles)]
    results = await asyncio.gather(*tasks)
    tile_progress.close()

    log_memory_usage('after all tiles downloaded')
    log_thread_status('after all tiles downloaded')

    results.sort(key=lambda t: t[0])
    images: list[Image.Image] = [img for _, img in results]

    eff_tile_px = XYZ_TILE_SIZE * (2 if XYZ_USE_RETINA else 1)
    result = assemble_and_crop(
        images=images,
        tiles_x=ctx.tiles_x,
        tiles_y=ctx.tiles_y,
        eff_tile_px=eff_tile_px,
        crop_rect=ctx.crop_rect,
    )

    with contextlib.suppress(Exception):
        images.clear()

    return result

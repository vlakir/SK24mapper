"""XYZ tiles processor - standard map tiles (satellite, hybrid, outdoors)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from geo.topography import async_fetch_xyz_tile
from services.tile_streaming import stream_fetch_assemble_xyz
from shared.constants import (
    XYZ_TILE_SIZE,
    xyz_use_retina_for_style,
)

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
    use_retina = xyz_use_retina_for_style(ctx.style_id)
    eff_tile_px = XYZ_TILE_SIZE * (2 if use_retina else 1)

    async def fetch_one(tx: int, ty: int) -> Image.Image:
        return await async_fetch_xyz_tile(
            client=ctx.client,
            api_key=ctx.api_key,
            style_id=ctx.style_id,
            tile_size=XYZ_TILE_SIZE,
            z=ctx.zoom,
            x=tx,
            y=ty,
            use_retina=use_retina,
        )

    t_start = time.monotonic()
    result = await stream_fetch_assemble_xyz(
        ctx,
        eff_tile_px=eff_tile_px,
        fetch_one=fetch_one,
        label='Загрузка XYZ-тайлов',
    )
    elapsed = time.monotonic() - t_start
    logger.info(
        'PROFILE xyz: stream fetch+assemble=%.2fs (N=%d)',
        elapsed,
        len(ctx.tiles),
    )
    return result

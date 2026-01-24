"""Elevation color processor - DEM colorization with color ramp."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from shared.constants import (
    CONTOUR_LOG_MEMORY_EVERY_TILES,
    CONTOUR_PASS2_QUEUE_MAXSIZE,
    ELEVATION_LEGEND_STEP_M,
    ELEVATION_USE_RETINA,
)
from contours.helpers import tx_ty_from_index
from shared.diagnostics import log_memory_usage
from elevation.provider import ElevationTileProvider
from geo.geometry import tile_overlap_rect_common
from geo.topography import (
    ELEV_MIN_RANGE_M,
    ELEV_PCTL_HI,
    ELEV_PCTL_LO,
    ELEVATION_COLOR_RAMP,
)
from services.color_utils import ColorMapper
from shared.progress import ConsoleProgress

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def process_elevation_color(ctx: MapDownloadContext) -> Image.Image:
    """Process elevation color map using DEM tiles.
    
    Two-pass approach:
    1. Sample elevations to determine range
    2. Colorize tiles using the computed range
    
    Args:
        ctx: Map download context with all necessary parameters.
        
    Returns:
        Colorized elevation image.
    """
    from elevation.stats import compute_elevation_range, sample_elevation_percentiles
    
    color_mapper = ColorMapper(ELEVATION_COLOR_RAMP, lut_size=2048)
    
    provider = ElevationTileProvider(
        client=ctx.client, api_key=ctx.api_key, use_retina=ELEVATION_USE_RETINA
    )
    
    full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)
    
    # Pass A: Sample elevations
    tile_progress = ConsoleProgress(total=len(ctx.tiles), label='Загрузка и анализ DEM')
    
    async def get_dem_tile(xw: int, yw: int) -> Image.Image:
        return await provider.get_tile_image(ctx.zoom, xw, yw)
    
    samples, seen_count, tile_cache = await sample_elevation_percentiles(
        enumerate(ctx.tiles),
        tiles_x=ctx.tiles_x,
        crop_rect=ctx.crop_rect,
        full_eff_tile_px=full_eff_tile_px,
        get_tile_image=get_dem_tile,
        max_samples=50000,
        rng_seed=42,
        on_progress=tile_progress.step,
        semaphore=ctx.semaphore,
        cache_tiles=True,
    )
    
    with contextlib.suppress(Exception):
        tile_progress.close()
    
    logger.info('DEM sampling reservoir: kept=%s seen~=%s', len(samples), seen_count)
    
    lo, hi = compute_elevation_range(
        samples,
        p_lo=ELEV_PCTL_LO,
        p_hi=ELEV_PCTL_HI,
        min_range_m=ELEV_MIN_RANGE_M,
    )
    
    step_m = ELEVATION_LEGEND_STEP_M
    lo_rounded = math.floor(lo / step_m) * step_m
    hi_rounded = math.ceil(hi / step_m) * step_m
    if hi_rounded <= lo_rounded:
        hi_rounded = lo_rounded + step_m
    inv = 1.0 / (hi_rounded - lo_rounded)
    
    # Store elevation range for legend
    ctx.elev_min_m = lo_rounded
    ctx.elev_max_m = hi_rounded
    
    # Pass B: Colorize tiles
    result = Image.new('RGB', (ctx.crop_rect[2], ctx.crop_rect[3]))
    tile_progress = ConsoleProgress(total=len(ctx.tiles), label='Окрашивание DEM')
    tile_count = 0
    
    queue: asyncio.Queue[tuple[int, int, int, int, int, int, Image.Image]] = (
        asyncio.Queue(maxsize=CONTOUR_PASS2_QUEUE_MAXSIZE)
    )
    paste_lock = asyncio.Lock()
    
    async def producer(idx_xy: tuple[int, tuple[int, int]]) -> None:
        nonlocal tile_count
        idx, (tile_x_world, tile_y_world) = idx_xy
        tx, ty = tx_ty_from_index(idx, ctx.tiles_x)
        ov = tile_overlap_rect_common(tx, ty, ctx.crop_rect, full_eff_tile_px)
        if ov is None:
            await tile_progress.step(1)
            return
        # Use cached tile from Pass A
        if tile_cache and (tile_x_world, tile_y_world) in tile_cache:
            img = tile_cache[(tile_x_world, tile_y_world)]
        else:
            async with ctx.semaphore:
                img = await provider.get_tile_image(ctx.zoom, tile_x_world, tile_y_world)
        x0, y0, x1, y1 = ov
        await queue.put((tx, ty, x0, y0, x1, y1, img))
    
    async def consumer() -> None:
        nonlocal tile_count
        ar = 0.1 * 65536.0 * inv
        ag = 0.1 * 256.0 * inv
        ab = 0.1 * 1.0 * inv
        a0 = (-10000.0 - lo_rounded) * inv
        
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            tx, ty, x0, y0, x1, y1, img = item
            try:
                cx, cy, _, _ = ctx.crop_rect
                dx0 = x0 - cx
                dy0 = y0 - cy
                base_x = tx * full_eff_tile_px
                base_y = ty * full_eff_tile_px
                
                arr = np.asarray(img, dtype=np.uint8)
                sub = arr[
                    y0 - base_y : y1 - base_y,
                    x0 - base_x : x1 - base_x,
                    :3,
                ]
                
                t = (
                    ar * sub[..., 0].astype(np.float32)
                    + ag * sub[..., 1].astype(np.float32)
                    + ab * sub[..., 2].astype(np.float32)
                    + a0
                )
                
                _lut = color_mapper.lut
                _lut_size = len(_lut)
                _l = np.clip(t * (_lut_size - 1), 0, _lut_size - 1).astype(np.int32)
                rgb = np.stack([_lut[_l, c] for c in range(3)], axis=-1).astype(np.uint8)
                
                patch = Image.fromarray(rgb, mode='RGB')
                async with paste_lock:
                    result.paste(patch, (dx0, dy0))
                
                await tile_progress.step(1)
                tile_count += 1
                if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                    log_memory_usage(f'pass2(color) after {tile_count} tiles')
            except Exception as e:
                logger.warning('Error colorizing tile (%d,%d): %s', tx, ty, e)
                await tile_progress.step(1)
            finally:
                queue.task_done()
    
    # Run producer/consumer
    num_consumers = min(4, len(ctx.tiles))
    consumers = [asyncio.create_task(consumer()) for _ in range(num_consumers)]
    producers = [asyncio.create_task(producer(pair)) for pair in enumerate(ctx.tiles)]
    
    try:
        await asyncio.gather(*producers)
        for _ in consumers:
            await queue.put(None)
        await queue.join()
        await asyncio.gather(*consumers)
    except Exception:
        for task in producers + consumers:
            task.cancel()
        for _ in consumers:
            await queue.put(None)
        await queue.join()
        await asyncio.gather(*consumers, return_exceptions=True)
        raise
    finally:
        tile_progress.close()
        if tile_cache:
            for img in tile_cache.values():
                with contextlib.suppress(Exception):
                    img.close()
            tile_cache.clear()
    
    return result

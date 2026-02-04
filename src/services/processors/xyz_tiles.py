"""XYZ tiles processor - standard map tiles (satellite, hybrid, outdoors)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from imaging.streaming import StreamingImage
from shared.constants import (
    CONTOUR_LOG_MEMORY_EVERY_TILES,
    STREAMING_TEMP_DIR,
    XYZ_TILE_SIZE,
    XYZ_USE_RETINA,
)
from shared.diagnostics import log_memory_usage, log_thread_status
from shared.progress import ConsoleProgress

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)

# Timing accumulators for profiling
_timing_fetch_total = 0.0
_timing_process_total = 0.0
_timing_paste_total = 0.0


async def process_xyz_tiles(ctx: MapDownloadContext) -> StreamingImage:
    """Process standard XYZ tiles (satellite, hybrid, streets, outdoors).

    Tiles are fetched via TileFetcher (with SQLite caching), then
    assembled into a StreamingImage in a streaming fashion.

    Args:
        ctx: Map download context with all necessary parameters.

    Returns:
        StreamingImage with assembled and cropped tiles.
    """
    eff_tile_px = XYZ_TILE_SIZE * (2 if XYZ_USE_RETINA else 1)
    crop_x, crop_y, crop_w, crop_h = ctx.crop_rect

    logger.info(
        'XYZ processor: создание StreamingImage %dx%d (%.1fM пикселей, ~%.1f GB)...',
        crop_w,
        crop_h,
        crop_w * crop_h / 1_000_000,
        crop_w * crop_h * 3 / 1_000_000_000,
    )

    # Create result StreamingImage
    temp_dir = getattr(ctx, 'temp_dir', STREAMING_TEMP_DIR)
    result = StreamingImage(crop_w, crop_h, temp_dir=temp_dir)

    logger.info('StreamingImage создан успешно')

    logger.info(
        'Processing %d tiles (%dx%d) into %dx%d image, eff_tile_px=%d',
        len(ctx.tiles),
        ctx.tiles_x,
        ctx.tiles_y,
        crop_w,
        crop_h,
        eff_tile_px,
    )

    tile_progress = ConsoleProgress(total=len(ctx.tiles), label='Загрузка XYZ-тайлов')
    tile_count = 0

    # Timing accumulators (use global for profiling)
    global _timing_fetch_total, _timing_process_total, _timing_paste_total
    _timing_fetch_total = 0.0
    _timing_process_total = 0.0
    _timing_paste_total = 0.0
    timing_lock = asyncio.Lock()

    async def fetch_and_write(idx_xy: tuple[int, tuple[int, int]]) -> None:
        """Fetch a single tile and write it directly to StreamingImage."""
        nonlocal tile_count
        global _timing_fetch_total, _timing_process_total, _timing_paste_total

        idx, (tx, ty) = idx_xy

        # Calculate tile position in grid
        tile_j = idx // ctx.tiles_x  # row
        tile_i = idx % ctx.tiles_x  # column

        # Tile coordinates on full canvas
        tile_x0 = tile_i * eff_tile_px
        tile_y0 = tile_j * eff_tile_px

        # Calculate intersection with crop_rect
        inter_x0 = max(tile_x0, crop_x)
        inter_y0 = max(tile_y0, crop_y)
        inter_x1 = min(tile_x0 + eff_tile_px, crop_x + crop_w)
        inter_y1 = min(tile_y0 + eff_tile_px, crop_y + crop_h)

        # Skip if no intersection
        if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
            await tile_progress.step(1)
            return

        # TIMING: fetch from cache or network
        t0_fetch = time.monotonic()
        async with ctx.semaphore:
            img = await ctx.tile_fetcher.fetch_xyz(
                client=ctx.client,
                style_id=ctx.style_id,
                z=ctx.zoom,
                x=tx,
                y=ty,
                tile_size=XYZ_TILE_SIZE,
                use_retina=XYZ_USE_RETINA,
            )
        t1_fetch = time.monotonic()

        # TIMING: process (convert, resize, crop)
        t0_process = time.monotonic()
        # Convert PIL.Image to numpy array and close immediately
        tile_arr = np.array(img)
        img.close()

        # Resize if needed
        th, tw = tile_arr.shape[:2]
        if (tw, th) != (eff_tile_px, eff_tile_px):
            if tile_count == 0:
                logger.warning(
                    'RESIZE DETECTED: tile actual size %dx%d, expected %dx%d',
                    tw,
                    th,
                    eff_tile_px,
                    eff_tile_px,
                )
            tile_arr = cv2.resize(
                tile_arr,
                (eff_tile_px, eff_tile_px),
                interpolation=cv2.INTER_LINEAR,
            )

        # Extract the part of tile that intersects with crop_rect
        src_x0 = inter_x0 - tile_x0
        src_y0 = inter_y0 - tile_y0
        src_x1 = inter_x1 - tile_x0
        src_y1 = inter_y1 - tile_y0

        tile_crop = tile_arr[src_y0:src_y1, src_x0:src_x1]

        # Destination coordinates in result image
        dst_x = inter_x0 - crop_x
        dst_y = inter_y0 - crop_y
        t1_process = time.monotonic()

        # TIMING: paste to StreamingImage
        t0_paste = time.monotonic()
        # Write directly without lock - tiles are in non-overlapping regions
        result.paste_tile(tile_crop, dst_x, dst_y)
        t1_paste = time.monotonic()

        # Update timing accumulators
        async with timing_lock:
            _timing_fetch_total += t1_fetch - t0_fetch
            _timing_process_total += t1_process - t0_process
            _timing_paste_total += t1_paste - t0_paste

        # Free memory immediately
        del tile_arr
        del tile_crop

        await tile_progress.step(1)
        tile_count += 1
        if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
            log_memory_usage(f'after {tile_count} tiles')

    # Process all tiles concurrently but with semaphore limiting parallelism
    tiles_start_time = time.monotonic()
    tasks = [fetch_and_write(pair) for pair in enumerate(ctx.tiles)]
    await asyncio.gather(*tasks)
    tiles_elapsed = time.monotonic() - tiles_start_time
    tile_progress.close()

    # Flush mmap to disk once after all tiles are written
    result.flush()

    log_memory_usage('after all tiles processed')
    log_thread_status('after all tiles processed')

    # Log detailed timing breakdown
    logger.info(
        'Tile timing breakdown (cumulative across %d tiles):\n'
        '  - Fetch (cache/network): %.2fs\n'
        '  - Process (np/resize):   %.2fs\n'
        '  - Paste (mmap write):    %.2fs\n'
        '  - Total wall clock:      %.2fs',
        len(ctx.tiles),
        _timing_fetch_total,
        _timing_process_total,
        _timing_paste_total,
        tiles_elapsed,
    )

    logger.info('Tile processing complete: %dx%d', crop_w, crop_h)

    return result

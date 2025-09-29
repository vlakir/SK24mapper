from __future__ import annotations

import contextlib
import random
from typing import TYPE_CHECKING

from constants import CONTOUR_LOG_MEMORY_EVERY_TILES
from contours_helpers import tx_ty_from_index
from geometry import tile_overlap_rect_common as _tile_overlap_rect_common
from topography import compute_percentiles, decode_terrain_rgb_to_elevation_m

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from PIL import Image


async def sample_elevation_percentiles(
    tiles: Iterable[tuple[int, tuple[int, int]]],
    *,
    tiles_x: int,
    crop_rect: tuple[int, int, int, int],
    full_eff_tile_px: int,
    get_tile_image: Callable[[int, int], Awaitable[Image.Image]],
    max_samples: int = 50_000,
    rng_seed: int = 42,
    on_progress: Callable[[int], Awaitable[None]] | None = None,
    semaphore=None,
) -> tuple[list[float], int]:
    """
    Reservoir-sample elevations over a set of tiles for percentile estimation.

    Args:
        tiles: Iterable of (idx, (x_world, y_world)) tile refs.
        tiles_x: Tile columns count for mapping linear idx -> (tx,ty).
        crop_rect: Crop rectangle (x, y, w, h) in pixels to filter overlapping tiles.
        full_eff_tile_px: Effective tile size in pixels (considering retina factor) for overlap check.
        get_tile_image: Async function to fetch tile image for (zoom, x_world, y_world).
        max_samples: Reservoir capacity.
        rng_seed: RNG seed for reproducibility.
        on_progress: Optional async callback to report progress per processed tile.
        semaphore: Optional asyncio.Semaphore to limit concurrency.

    Returns:
        (samples, seen_count): sampled elevation values and total number of seen samples.

    """
    rng = random.Random(rng_seed)
    samples: list[float] = []
    seen_count = 0
    tile_count = 0

    async def _step_progress() -> None:
        if on_progress is not None:
            with contextlib.suppress(Exception):
                await on_progress(1)

    async def _fetch_and_sample(idx_xy: tuple[int, tuple[int, int]]) -> None:
        nonlocal samples, seen_count, tile_count
        idx, (tile_x_world, tile_y_world) = idx_xy
        tx, ty = tx_ty_from_index(idx, tiles_x)
        if _tile_overlap_rect_common(tx, ty, crop_rect, full_eff_tile_px) is None:
            await _step_progress()
            return
        if semaphore is None:
            img = await get_tile_image(
                tile_x_world, tile_y_world
            )  # zoom embedded by closure
        else:
            async with semaphore:
                img = await get_tile_image(tile_x_world, tile_y_world)
        dem_tile = decode_terrain_rgb_to_elevation_m(img)
        # iterate coarse grid (approx 32x32) within tile to limit CPU
        h = len(dem_tile)
        w = len(dem_tile[0]) if h else 0
        if h and w:
            step_y = max(1, h // 32)
            step_x = max(1, w // 32)
            off_y = rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
            off_x = rng.randrange(0, min(step_x, w)) if step_x > 1 else 0
            for ry in range(off_y, h, step_y):
                row = dem_tile[ry]
                for rx in range(off_x, w, step_x):
                    v = row[rx]
                    seen_count += 1
                    if len(samples) < max_samples:
                        samples.append(v)
                    else:
                        j = rng.randrange(0, seen_count)
                        if j < max_samples:
                            samples[j] = v
        await _step_progress()
        tile_count += 1
        if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
            try:
                from diagnostics import log_memory_usage

                log_memory_usage(f'elev pass1 after {tile_count} tiles')
            except Exception:
                pass

    # Run sequentially here — service should orchestrate concurrency around us if needed.
    # This keeps the module reusable and free of task lifecycle concerns.
    for idx_xy in tiles:
        await _fetch_and_sample(idx_xy)

    return samples, seen_count


def compute_elevation_range(
    samples: list[float], *, p_lo: float, p_hi: float, min_range_m: float
) -> tuple[float, float]:
    """Compute robust elevation range [mn, mx] using percentiles with a safety floor."""
    if not samples:
        return (0.0, 0.0)
    lo, hi = compute_percentiles(samples, p_lo, p_hi)
    # Ensure minimal dynamic range
    if hi - lo < min_range_m:
        mid = 0.5 * (lo + hi)
        lo = mid - 0.5 * min_range_m
        hi = mid + 0.5 * min_range_m
    return lo, hi

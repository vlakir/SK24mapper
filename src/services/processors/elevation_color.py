"""Elevation color processor - DEM colorization with color ramp."""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from elevation.stats import compute_elevation_range
from geo.topography import (
    ELEV_MIN_RANGE_M,
    ELEV_PCTL_HI,
    ELEV_PCTL_LO,
    ELEVATION_COLOR_RAMP,
)
from services.color_utils import ColorMapper
from services.processors.radio_horizon import _load_dem, _load_topo
from shared.constants import (
    ELEVATION_COLOR_USE_RETINA,
    ELEVATION_LEGEND_STEP_M,
)

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def process_elevation_color(ctx: MapDownloadContext) -> Image.Image:
    """
    Process elevation color map using DEM tiles.

    Single-pass approach:
    1. Load DEM via shared _load_dem (same as radio_horizon)
    2. Sample elevations from numpy array to determine range
    3. Colorize DEM directly using LUT

    Args:
        ctx: Map download context with all necessary parameters.

    Returns:
        Colorized elevation image.

    """
    color_mapper = ColorMapper(ELEVATION_COLOR_RAMP, lut_size=2048)

    # 1. Load DEM (reuse shared loader from radio_horizon)
    dem_full, ds_factor = await _load_dem(
        ctx, use_retina=ELEVATION_COLOR_USE_RETINA, label='карты высот'
    )

    # 2. Sample elevations from numpy DEM for percentile estimation
    rng = random.Random(42)  # noqa: S311
    flat = dem_full.ravel()
    sample_count = min(50_000, len(flat))
    indices = rng.sample(range(len(flat)), sample_count)
    samples = [float(flat[i]) for i in indices]

    logger.info('DEM sampling: kept=%d from %d pixels', len(samples), len(flat))

    # 3. Compute elevation range
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

    ctx.elev_min_m = lo_rounded
    ctx.elev_max_m = hi_rounded

    # 4. Colorize numpy DEM directly (no RGB decoding needed)
    inv = 1.0 / (hi_rounded - lo_rounded)
    t = np.clip((dem_full - lo_rounded) * inv, 0.0, 1.0)
    lut = color_mapper.lut
    lut_indices = (t * (len(lut) - 1)).astype(np.int32)
    rgb = lut[lut_indices]  # (H, W, 3)

    # 5. Scale back if downsampled
    target_w, target_h = ctx.crop_rect[2], ctx.crop_rect[3]
    result = Image.fromarray(rgb)
    if result.size != (target_w, target_h):
        result = result.resize((target_w, target_h), Image.Resampling.BILINEAR)

    # 6. Load topographic base and blend (DRY — same _load_topo as НСУ/РЛС)
    topo_base = await _load_topo(
        ctx, use_retina=ELEVATION_COLOR_USE_RETINA, label='карты высот'
    )
    if topo_base.size != result.size:
        topo_base = topo_base.resize(result.size, Image.Resampling.BILINEAR)

    blend_alpha = 1.0 - ctx.settings.radio_horizon_overlay_alpha
    topo_gray = topo_base.convert('L').convert('RGBA')
    result = result.convert('RGBA')

    # Cache layers for interactive alpha (same mechanism as НСУ/РЛС)
    ctx.rh_cache_topo_base = topo_gray
    ctx.rh_cache_coverage = result.copy()

    return Image.blend(topo_gray, result, blend_alpha)

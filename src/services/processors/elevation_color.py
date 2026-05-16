"""Elevation color processor - DEM colorization with color ramp."""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import random
import time
from typing import TYPE_CHECKING

import cv2
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
from shared import dem_topo_cache
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

    # 1. Start DEM and topo fetches in parallel.
    #
    # DEM (Terrain-RGB) and topo (style raster) come from different Mapbox
    # endpoints with independent caches and decoders. Previously these were
    # two sequential `await`s with ~3s of DEM + ~1.5s of topo + idle gaps
    # between them. Running both as tasks lets the event loop overlap them:
    # while we're CPU-bound inside the colorisation block below (halve →
    # sample → LUT → resize), topo continues to download and decode on its
    # own worker threads (both _load_dem and _load_topo offload PIL/numpy
    # decode work via asyncio.to_thread). Net: total run_processor wall
    # time falls toward max(dem_with_color, topo) instead of their sum.
    _t_loads_start = time.monotonic()
    dem_task = asyncio.create_task(
        _load_dem(ctx, use_retina=ELEVATION_COLOR_USE_RETINA, label='карты высот')
    )
    topo_task = asyncio.create_task(
        _load_topo(ctx, use_retina=ELEVATION_COLOR_USE_RETINA, label='карты высот')
    )

    # DEM is needed first for colorisation; topo continues to load in
    # background while we crunch the color image. If DEM raises, cancel
    # the topo fetch so it doesn't keep downloading.
    try:
        dem_full, ds_factor = await dem_task
    except BaseException:
        topo_task.cancel()
        raise
    logger.info(
        'elev_color: DEM ready in %.3fs (topo still loading in parallel)',
        time.monotonic() - _t_loads_start,
    )

    # 1a. Halve the DEM that feeds the colorisation step.
    #
    # The colorised image is upscaled back to target size via PIL.BILINEAR
    # at step 5 anyway, so we lose only a tiny bit of color-band detail
    # for a ×4 reduction in colorise + resize memory traffic. NOTE: the
    # cursor-elevation lookup uses ctx.raw_dem_for_cursor (saved at full
    # resolution INSIDE _load_dem before any downsampling), so this does
    # NOT affect "H: X м" accuracy under the mouse.
    _t_ds = time.monotonic()
    orig_h, orig_w = dem_full.shape
    dem_full = cv2.resize(
        dem_full,
        (max(1, orig_w // 2), max(1, orig_h // 2)),
        interpolation=cv2.INTER_AREA,
    )
    logger.info(
        'elev_color: extra DEM halve %dx%d → %dx%d in %.3fs',
        orig_w, orig_h, dem_full.shape[1], dem_full.shape[0],
        time.monotonic() - _t_ds,
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

    # 4. Colorize numpy DEM directly (no RGB decoding needed). Output is
    # numpy uint8 (H/2, W/2, 3) RGB — we KEEP it in numpy from here to the
    # end of the function and only wrap as PIL at the very last step.
    inv = 1.0 / (hi_rounded - lo_rounded)
    t = np.clip((dem_full - lo_rounded) * inv, 0.0, 1.0)
    lut = color_mapper.lut
    lut_indices = (t * (len(lut) - 1)).astype(np.int32)
    rgb = lut[lut_indices]  # (H/2, W/2, 3) uint8 RGB

    del dem_full, t, lut_indices
    gc.collect()

    # 5. Upscale colorised image to target size using cv2 (multi-threaded,
    # SIMD) instead of PIL.resize (single-threaded). For 1197² → 9580² this
    # alone saves ~0.4-0.5s vs PIL.
    _t_upscale = time.monotonic()
    target_w, target_h = ctx.crop_rect[2], ctx.crop_rect[3]
    if rgb.shape[:2] != (target_h, target_w):
        rgb = cv2.resize(
            rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
    logger.info(
        'elev_color: upscale rgb to %dx%d in %.3fs',
        target_w, target_h, time.monotonic() - _t_upscale,
    )

    # 6. Await the topo fetch started at step 1.
    _t_topo_wait = time.monotonic()
    topo_base_pil = await topo_task
    logger.info(
        'elev_color: topo ready after %.3fs additional wait',
        time.monotonic() - _t_topo_wait,
    )

    # 7. Move topo to numpy and prepare 3-channel grayscale for blending.
    #
    # The whole post-topo block used to do PIL operations on 9580×9580
    # images: topo.resize + topo.convert('L').convert('RGBA') + result.
    # convert('RGBA') + 2× .resize(half) + Image.blend. PIL is single-
    # threaded and each step allocates a full ~367 MB RGBA buffer, so the
    # block cost ~2.5s on this machine.
    #
    # cv2 is multi-threaded (12 cores set globally) and SIMD-accelerated;
    # numpy concatenate is memcpy-fast. Same visual output, ~10× less
    # wall-clock time on these sizes.
    _t_post = time.monotonic()
    _t = time.monotonic()
    topo_arr = np.asarray(topo_base_pil, dtype=np.uint8)
    # Some Mapbox style tiles arrive as RGBA — strip alpha; downstream
    # operations all assume 3-channel RGB.
    if topo_arr.ndim == 3 and topo_arr.shape[2] == 4:
        topo_arr = topo_arr[..., :3]
    _t_asarray = time.monotonic() - _t

    _t = time.monotonic()
    topo_orig_shape = topo_arr.shape[:2]
    if topo_arr.shape[:2] != (target_h, target_w):
        topo_arr = cv2.resize(
            topo_arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
    _t_topo_resize = time.monotonic() - _t

    # Grayscale → broadcast to 3 channels (matches PIL's convert('L').
    # convert('RGB') visually — R=G=B=luminance).
    _t = time.monotonic()
    gray = cv2.cvtColor(topo_arr, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    del topo_arr, topo_base_pil, gray
    _t_gray = time.monotonic() - _t

    # 8. rh_cache layers at HALF target size — for interactive alpha
    # slider only (ELEVATION_COLOR has no *_fast recompute path). Half-
    # res blend is upscaled BILINEAR in GUI; visually indistinguishable
    # from full-res. Smaller payload → less shm transfer, less GUI RSS,
    # less transient inside RhDisplayCacheWorker's rotate.
    #
    # Cache the two output bitmaps: rh_topo_half depends only on the
    # (area + zoom + retina + topo style) and so reuses the same key as
    # cached topo; rh_cov_half also folds in the elevation range that
    # defines the color gradient. On a warm rebuild same-area + same-
    # range the cv2.resize + np.concatenate (~390 ms combined) becomes a
    # cache lookup + Image.fromarray view.
    _t = time.monotonic()
    half_w = max(1, target_w // 2)
    half_h = max(1, target_h // 2)
    topo_half_key = dem_topo_cache.make_area_key(
        kind_tag='rh_topo_half_v1',
        zoom=ctx.zoom,
        eff_scale=ctx.eff_scale,
        use_retina=ELEVATION_COLOR_USE_RETINA,
        center_lat_wgs=ctx.center_lat_wgs,
        center_lng_wgs=ctx.center_lng_wgs,
        width_m=ctx.width_m,
        height_m=ctx.height_m,
    )
    cov_half_key = dem_topo_cache.make_area_key(
        kind_tag='rh_cov_half_v1',
        zoom=ctx.zoom,
        eff_scale=ctx.eff_scale,
        use_retina=ELEVATION_COLOR_USE_RETINA,
        center_lat_wgs=ctx.center_lat_wgs,
        center_lng_wgs=ctx.center_lng_wgs,
        width_m=ctx.width_m,
        height_m=ctx.height_m,
        extras=(round(float(lo_rounded), 2), round(float(hi_rounded), 2)),
    )
    topo_half_arr = dem_topo_cache.get('rh_topo_half', topo_half_key)
    cov_half_arr = dem_topo_cache.get('rh_cov_half', cov_half_key)
    if topo_half_arr is None:
        gray_3ch_half = cv2.resize(
            gray_3ch, (half_w, half_h), interpolation=cv2.INTER_LINEAR
        )
        alpha_half = np.full((half_h, half_w, 1), 255, dtype=np.uint8)
        # np.concatenate is a memcpy — fast. Image.fromarray is a zero-
        # copy view over the numpy buffer (uint8 contiguous arrays).
        topo_half_arr = np.concatenate([gray_3ch_half, alpha_half], axis=2)
        dem_topo_cache.put('rh_topo_half', topo_half_key, topo_half_arr)
    if cov_half_arr is None:
        rgb_half = cv2.resize(
            rgb, (half_w, half_h), interpolation=cv2.INTER_LINEAR
        )
        alpha_half = np.full((half_h, half_w, 1), 255, dtype=np.uint8)
        cov_half_arr = np.concatenate([rgb_half, alpha_half], axis=2)
        dem_topo_cache.put('rh_cov_half', cov_half_key, cov_half_arr)
    ctx.rh_cache_topo_base = Image.fromarray(topo_half_arr)
    ctx.rh_cache_coverage = Image.fromarray(cov_half_arr)
    _t_rh_cache = time.monotonic() - _t

    # 9. Blend at full target size: out = topo*(1-α) + result*α.
    #    cv2.addWeighted(src1, α1, src2, α2, γ) = src1*α1 + src2*α2 + γ.
    _t = time.monotonic()
    blend_alpha = 1.0 - ctx.settings.radio_horizon_overlay_alpha
    blended = cv2.addWeighted(
        gray_3ch, 1.0 - blend_alpha, rgb, blend_alpha, 0.0
    )
    # Return as RGB. Downstream postprocess (paste-with-mask trick) pastes
    # RGBA overlays onto this RGB base: PIL drops the overlay's alpha after
    # using it as the mask, so result stays RGB through the pipeline. The
    # alternative — returning RGBA — forced the final preview to be
    # serialised as 282 MB raw bytes instead of a 10 MB JPEG, adding
    # ~0.4 s to worker save and a similar slice to GUI deserialise.
    _t_blend = time.monotonic() - _t

    logger.info(
        'elev_color post-topo timings: asarray=%.3fs '
        'topo_resize=%.3fs (%dx%d→%dx%d) gray=%.3fs rh_cache_half=%.3fs '
        'blend=%.3fs TOTAL=%.3fs',
        _t_asarray, _t_topo_resize,
        topo_orig_shape[1], topo_orig_shape[0], target_w, target_h,
        _t_gray, _t_rh_cache, _t_blend,
        time.monotonic() - _t_post,
    )
    return Image.fromarray(blended)

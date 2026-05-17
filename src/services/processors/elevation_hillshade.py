"""Elevation hillshade processor - DEM grayscale hillshading."""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from elevation.provider import ElevationTileProvider
from geo.topography import (
    assemble_dem,
    compute_hillshade,
    meters_per_pixel,
)
from infrastructure.http.client import resolve_cache_dir
from services.processors.radio_horizon import _load_topo
from services.tile_coverage import compute_tile_coverage
from shared.constants import (
    HILLSHADE_ALTITUDE_DEG,
    HILLSHADE_AZIMUTH_DEG,
    HILLSHADE_DEM_ZOOM,
    HILLSHADE_SMOOTH_SIGMA,
    HILLSHADE_USE_RETINA,
    HILLSHADE_Z_FACTOR,
)
from shared.diagnostics import log_memory_usage
from shared.progress import ConsoleProgress

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def _load_dem_native(
    ctx: MapDownloadContext,
    *,
    dem_zoom: int,
    use_retina: bool,
) -> tuple[np.ndarray, float]:
    """
    Load DEM at a specific zoom level (typically z14 = native Mapbox resolution).

    Returns (dem_array, pixel_size_m) where dem_array covers the same
    geographic extent as ctx but at the DEM zoom resolution.
    """
    eff_scale = 2 if use_retina else 1
    dem_mpp = meters_per_pixel(ctx.center_lat_wgs, dem_zoom, scale=eff_scale)

    # Compute tile coverage at the DEM zoom for the same geographic area
    coverage = compute_tile_coverage(
        center_lat_wgs=ctx.center_lat_wgs,
        center_lng_wgs=ctx.center_lng_wgs,
        width_m=ctx.width_m,
        height_m=ctx.height_m,
        zoom=dem_zoom,
        eff_scale=eff_scale,
    )
    logger.info(
        'Hillshade DEM: zoom=%d (display zoom=%d), tiles=%d, mpp=%.2f m',
        dem_zoom,
        ctx.zoom,
        len(coverage.tiles),
        dem_mpp,
    )

    # Load DEM tiles
    provider = ElevationTileProvider(
        client=ctx.client,
        api_key=ctx.api_key,
        use_retina=use_retina,
        cache_root=resolve_cache_dir(),
    )
    eff_tile_px = 256 * eff_scale

    progress = ConsoleProgress(
        total=len(coverage.tiles), label='Загрузка DEM (hillshade)'
    )

    async def fetch(idx_xy: tuple[int, tuple[int, int]]) -> tuple[int, np.ndarray]:
        idx, (tx, ty) = idx_xy
        async with ctx.semaphore:
            tile = await provider.get_tile_dem(dem_zoom, tx, ty)
            await progress.step(1)
            return idx, tile

    tasks = [fetch(pair) for pair in enumerate(coverage.tiles)]
    results = await asyncio.gather(*tasks)
    progress.close()

    results.sort(key=lambda t: t[0])
    tiles_data = [dem for _, dem in results]

    dem = assemble_dem(
        tiles_data=tiles_data,
        tiles_x=coverage.tiles_x,
        tiles_y=coverage.tiles_y,
        eff_tile_px=eff_tile_px,
        crop_rect=coverage.crop_rect,
    )
    del tiles_data
    gc.collect()

    # Save for cursor elevation display and contour overlay.
    # Must match ctx.crop_rect size expected by downstream code.
    display_w, display_h = ctx.crop_rect[2], ctx.crop_rect[3]
    if dem.shape[1] != display_w or dem.shape[0] != display_h:
        raw_dem_display = cv2.resize(
            dem, (display_w, display_h), interpolation=cv2.INTER_LINEAR
        )
        ctx.raw_dem_for_cursor = raw_dem_display
    else:
        ctx.raw_dem_for_cursor = dem.copy()
    ctx.rh_cache_crop_size = (display_w, display_h)

    log_memory_usage('after hillshade DEM load')
    return dem, dem_mpp


async def process_elevation_hillshade(ctx: MapDownloadContext) -> Image.Image:
    """
    Process elevation hillshade map using DEM tiles.

    Pipeline:
    1. Load DEM at native zoom (z14) — no oversampling, no downsampling
    2. Compute hillshade with Gaussian smoothing
    3. Convert float32 → grayscale → RGB
    4. Resize to display resolution with BILINEAR (smooth upscale)
    5. Load topographic base (Outdoors, grayscale)
    6. Blend hillshade with topo base (interactive alpha)

    Args:
        ctx: Map download context with all necessary parameters.

    Returns:
        Hillshade image blended with topographic base.

    """
    # 1. Load DEM at native Mapbox resolution
    dem_zoom = min(HILLSHADE_DEM_ZOOM, ctx.zoom)
    dem, dem_mpp = await _load_dem_native(
        ctx, dem_zoom=dem_zoom, use_retina=HILLSHADE_USE_RETINA
    )
    logger.info(
        'DEM loaded: %dx%d (%.1f m/px)',
        dem.shape[1],
        dem.shape[0],
        dem_mpp,
    )

    # 2. Compute hillshade
    hs = compute_hillshade(
        dem,
        pixel_size_m=dem_mpp,
        azimuth_deg=HILLSHADE_AZIMUTH_DEG,
        altitude_deg=HILLSHADE_ALTITUDE_DEG,
        z_factor=HILLSHADE_Z_FACTOR,
        smooth_sigma=HILLSHADE_SMOOTH_SIGMA,
    )
    del dem
    gc.collect()

    logger.info(
        'Hillshade computed: shape=%s, min=%.3f, max=%.3f',
        hs.shape,
        float(hs.min()),
        float(hs.max()),
    )

    # 3. Convert float32[0..1] → uint8 grayscale → RGB
    hs_uint8 = (hs * 255).astype(np.uint8)
    del hs
    result = Image.fromarray(hs_uint8).convert('RGB')
    del hs_uint8

    # 4. Resize to display resolution (smooth BILINEAR upscale)
    target_w, target_h = ctx.crop_rect[2], ctx.crop_rect[3]
    if result.size != (target_w, target_h):
        logger.info(
            'Hillshade upscale: %dx%d -> %dx%d',
            result.size[0],
            result.size[1],
            target_w,
            target_h,
        )
        result = result.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 5. Load topographic base
    topo_base = await _load_topo(
        ctx, use_retina=HILLSHADE_USE_RETINA, label='карты высот (hillshade)'
    )
    if topo_base.size != result.size:
        topo_base = topo_base.resize(result.size, Image.Resampling.BILINEAR)

    # 6. Blend hillshade with topo base
    blend_alpha = 1.0 - ctx.settings.radio_horizon_overlay_alpha
    topo_gray = topo_base.convert('L').convert('RGBA')
    result = result.convert('RGBA')

    # 7. Cache layers for interactive alpha slider
    ctx.rh_cache_topo_base = topo_gray
    ctx.rh_cache_coverage = result.copy()

    return Image.blend(topo_gray, result, blend_alpha)

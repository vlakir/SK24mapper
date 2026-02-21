"""Radio horizon / radar coverage processor - compute and visualize LOS coverage."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
from typing import TYPE_CHECKING

from PIL import Image

from elevation.provider import ElevationTileProvider
from geo.topography import (
    assemble_dem,
    async_fetch_xyz_tile,
    latlng_to_pixel_xy,
    meters_per_pixel,
)
from imaging import assemble_and_crop
from infrastructure.http.client import resolve_cache_dir
from services.radio_horizon import (
    compute_and_colorize_coverage,
    compute_downsample_factor,
    downsample_dem,
)
from shared.constants import (
    CONTOUR_LOG_MEMORY_EVERY_TILES,
    MAPBOX_STYLE_BY_TYPE,
    RADIO_HORIZON_USE_RETINA,
    TILE_SIZE,
    MapType,
)
from shared.diagnostics import log_memory_usage
from shared.progress import ConsoleProgress, LiveSpinner

if TYPE_CHECKING:
    import numpy as np

    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def _load_dem(
    ctx: MapDownloadContext,
    *,
    use_retina: bool = False,
    label: str = 'DEM',
    max_dem_pixels: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Загрузка и сборка DEM. Возвращает (dem_full, ds_factor).

    Args:
        ctx: Контекст загрузки карты.
        use_retina: Использовать ретина-тайлы.
        label: Метка для прогресса.
        max_dem_pixels: Порог даунсэмплинга
            (None = дефолт RADIO_HORIZON_MAX_DEM_PIXELS).

    Сохраняет ctx.raw_dem_for_cursor (полноразмерный DEM).

    """
    provider = ElevationTileProvider(
        client=ctx.client,
        api_key=ctx.api_key,
        use_retina=use_retina,
        cache_root=resolve_cache_dir(),
    )

    full_eff_tile_px = 256 * (2 if use_retina else 1)

    tile_progress = ConsoleProgress(
        total=len(ctx.tiles), label=f'Загрузка DEM для {label}'
    )
    tile_count = 0

    async def fetch_dem_tile(
        idx_xy: tuple[int, tuple[int, int]],
    ) -> tuple[int, np.ndarray]:
        nonlocal tile_count
        idx, (tile_x_world, tile_y_world) = idx_xy
        async with ctx.semaphore:
            dem_tile = await provider.get_tile_dem(ctx.zoom, tile_x_world, tile_y_world)
            await tile_progress.step(1)
            tile_count += 1
            if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                log_memory_usage(f'{label} after {tile_count} tiles')
            return idx, dem_tile

    tasks = [fetch_dem_tile(pair) for pair in enumerate(ctx.tiles)]
    results = await asyncio.gather(*tasks)
    tile_progress.close()

    results.sort(key=lambda t: t[0])
    dem_tiles_data = [dem for _, dem in results]

    dem_full = assemble_dem(
        tiles_data=dem_tiles_data,
        tiles_x=ctx.tiles_x,
        tiles_y=ctx.tiles_y,
        eff_tile_px=full_eff_tile_px,
        crop_rect=ctx.crop_rect,
    )
    del dem_tiles_data
    gc.collect()

    ctx.raw_dem_for_cursor = dem_full.copy()

    dem_h_orig, dem_w_orig = dem_full.shape
    ctx.rh_cache_crop_size = (dem_w_orig, dem_h_orig)
    ds_kwargs = {} if max_dem_pixels is None else {'max_pixels': max_dem_pixels}
    ds_factor = compute_downsample_factor(dem_h_orig, dem_w_orig, **ds_kwargs)

    if ds_factor > 1:
        logger.info(
            '%s: DEM слишком большой (%dx%d = %d Mpx), даунсэмплинг в %d раз',
            label,
            dem_w_orig,
            dem_h_orig,
            dem_w_orig * dem_h_orig // 1_000_000,
            ds_factor,
        )
        dem_full = downsample_dem(dem_full, ds_factor)
        gc.collect()

    return dem_full, ds_factor


async def _load_topo(
    ctx: MapDownloadContext,
    *,
    use_retina: bool = False,
    label: str = 'карты',
) -> Image.Image:
    """Загрузка топографической основы (стиль Outdoors). Возвращает PIL Image."""
    logger.info('Загрузка топографической основы для %s', label)
    sp_topo = LiveSpinner('Загрузка топографической основы')
    sp_topo.start()

    topo_style_id = MAPBOX_STYLE_BY_TYPE[MapType.OUTDOORS]
    topo_tile_size = TILE_SIZE
    topo_use_retina = use_retina

    async def fetch_topo_tile(
        idx_xy: tuple[int, tuple[int, int]],
    ) -> tuple[int, Image.Image]:
        idx, (tx, ty) = idx_xy
        async with ctx.semaphore:
            img = await async_fetch_xyz_tile(
                client=ctx.client,
                api_key=ctx.api_key,
                style_id=topo_style_id,
                tile_size=topo_tile_size,
                z=ctx.zoom,
                x=tx,
                y=ty,
                use_retina=topo_use_retina,
            )
            return idx, img

    topo_tasks = [fetch_topo_tile(pair) for pair in enumerate(ctx.tiles)]
    topo_results = await asyncio.gather(*topo_tasks)
    topo_results.sort(key=lambda t: t[0])
    topo_images = [img for _, img in topo_results]

    eff_tile_px_topo = topo_tile_size * (2 if topo_use_retina else 1)
    topo_base = assemble_and_crop(
        images=topo_images,
        tiles_x=ctx.tiles_x,
        tiles_y=ctx.tiles_y,
        eff_tile_px=eff_tile_px_topo,
        crop_rect=ctx.crop_rect,
    )

    with contextlib.suppress(Exception):
        topo_images.clear()

    sp_topo.stop('Топографическая основа загружена')
    return topo_base


async def _load_dem_and_topo(
    ctx: MapDownloadContext,
    *,
    use_retina: bool = RADIO_HORIZON_USE_RETINA,
    label: str = 'радиогоризонта',
) -> tuple[np.ndarray, Image.Image, int, int, float, float, int]:
    """
    Общая загрузка DEM и топоосновы для radio_horizon и radar_coverage.

    Returns:
        Tuple of (dem, topo_base, antenna_row, antenna_col,
        pixel_size_m, cp_elevation, ds_factor).

    """
    dem_full, ds_factor = await _load_dem(ctx, use_retina=use_retina, label=label)

    full_eff_tile_px = 256 * (2 if use_retina else 1)

    # Compute antenna position in DEM pixels
    cp_lng_sk42, cp_lat_sk42 = ctx.t_sk42_from_gk.transform(
        ctx.settings.control_point_x_sk42_gk,
        ctx.settings.control_point_y_sk42_gk,
    )
    control_lng_wgs, control_lat_wgs = ctx.t_sk42_to_wgs.transform(
        cp_lng_sk42, cp_lat_sk42
    )

    ant_px_x, ant_px_y = latlng_to_pixel_xy(control_lat_wgs, control_lng_wgs, ctx.zoom)

    cx_crop, cy_crop, _cw, _ch = ctx.crop_rect
    first_tile_x, first_tile_y = ctx.tiles[0]
    global_origin_x = first_tile_x * full_eff_tile_px
    global_origin_y = first_tile_y * full_eff_tile_px

    antenna_col_orig = int(
        ant_px_x * (full_eff_tile_px / 256) - global_origin_x - cx_crop
    )
    antenna_row_orig = int(
        ant_px_y * (full_eff_tile_px / 256) - global_origin_y - cy_crop
    )

    antenna_col = antenna_col_orig // ds_factor
    antenna_row = antenna_row_orig // ds_factor

    dem_h, dem_w = dem_full.shape
    antenna_row = max(0, min(antenna_row, dem_h - 1))
    antenna_col = max(0, min(antenna_col, dem_w - 1))

    logger.info(
        '%s: DEM размер %dx%d, антенна в пикселях (%d, %d)',
        label,
        dem_w,
        dem_h,
        antenna_col,
        antenna_row,
    )

    pixel_size_m = (
        meters_per_pixel(ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale) * ds_factor
    )

    cp_elevation = float(dem_full[antenna_row, antenna_col])
    ctx.control_point_elevation = cp_elevation

    # Load topographic base (DRY — shared _load_topo)
    topo_base = await _load_topo(ctx, use_retina=use_retina, label=label)

    # Save topo base for cache at DEM size (matches downsampled DEM dimensions).
    # This ensures aligned crop+resize during recompute (same transform as coverage).
    dem_cache_size = (dem_full.shape[1], dem_full.shape[0])
    if topo_base.size != dem_cache_size:
        topo_for_cache = topo_base.resize(dem_cache_size, Image.Resampling.BILINEAR)
    else:
        topo_for_cache = topo_base.copy()

    ctx.rh_cache_topo_base = topo_for_cache.convert('L').convert('RGBA')

    # Save DEM for cache
    ctx.rh_cache_dem_full = ctx.raw_dem_for_cursor  # полноразмерный DEM для курсора
    ctx.rh_cache_dem = dem_full.copy()
    ctx.rh_cache_antenna_row = antenna_row
    ctx.rh_cache_antenna_col = antenna_col
    ctx.rh_cache_pixel_size_m = pixel_size_m

    return (
        dem_full,
        topo_base,
        antenna_row,
        antenna_col,
        pixel_size_m,
        cp_elevation,
        ds_factor,
    )


async def process_radio_horizon(ctx: MapDownloadContext) -> Image.Image:
    """Process radio horizon map (360° coverage for НСУ БпЛА)."""
    (
        dem_full,
        topo_base,
        antenna_row,
        antenna_col,
        pixel_size_m,
        cp_elevation,
        ds_factor,
    ) = await _load_dem_and_topo(
        ctx, use_retina=RADIO_HORIZON_USE_RETINA, label='радиогоризонта'
    )

    sp = LiveSpinner('Вычисление радиогоризонта')
    sp.start()

    result = compute_and_colorize_coverage(
        dem=dem_full,
        antenna_row=antenna_row,
        antenna_col=antenna_col,
        antenna_height_m=ctx.settings.antenna_height_m,
        pixel_size_m=pixel_size_m,
        max_height_m=ctx.settings.max_flight_height_m,
        uav_height_reference=ctx.settings.uav_height_reference,
        cp_elevation=cp_elevation,
    )

    sp.stop('Радиогоризонт вычислен')

    del dem_full
    gc.collect()

    # Resize if downsampled
    if ds_factor > 1:
        target_size = (ctx.crop_rect[2], ctx.crop_rect[3])
        result = result.resize(target_size, Image.Resampling.BILINEAR)

    # Resize topo base to match result for current blend
    if topo_base.size != result.size:
        topo_base = topo_base.resize(result.size, Image.Resampling.BILINEAR)

    # Blend
    blend_alpha = 1.0 - ctx.settings.radio_horizon_overlay_alpha
    topo_base = topo_base.convert('L').convert('RGBA')
    result = result.convert('RGBA')
    # Сохраняем слой покрытия до blend для интерактивной альфы
    ctx.rh_cache_coverage = result.copy()
    result = Image.blend(topo_base, result, blend_alpha)

    del topo_base
    gc.collect()

    logger.info('Карта радиогоризонта построена')
    return result

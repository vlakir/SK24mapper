"""Radio horizon / radar coverage processor - compute and visualize LOS coverage."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
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
from services.tile_streaming import (
    stream_fetch_assemble_dem,
    stream_fetch_assemble_xyz,
)
from services.radio_horizon import (
    compute_and_colorize_coverage,
    compute_downsample_factor,
    downsample_dem,
)
from shared import dem_topo_cache
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
    # Try cache layers first: in-memory (same persistent worker) → disk
    # (across worker restarts). On hit, skip 1444 tile fetches + decodes +
    # assembly (~3 s on z=17 retina). Key folds in area + zoom + retina.
    cache_key = dem_topo_cache.make_area_key(
        kind_tag='dem_v1',
        zoom=ctx.zoom,
        eff_scale=ctx.eff_scale,
        use_retina=use_retina,
        center_lat_wgs=ctx.center_lat_wgs,
        center_lng_wgs=ctx.center_lng_wgs,
        width_m=ctx.width_m,
        height_m=ctx.height_m,
        tile_size=256,
    )
    dem_full = dem_topo_cache.get_inmem('dem', cache_key)
    cache_layer = None
    if dem_full is not None:
        cache_layer = 'in-mem'
    else:
        disk_hit = dem_topo_cache.load('dem', cache_key)
        if disk_hit is not None:
            dem_full = disk_hit
            cache_layer = 'disk'
            dem_topo_cache.set_inmem('dem', cache_key, dem_full)

    if dem_full is None:
        provider = ElevationTileProvider(
            client=ctx.client,
            api_key=ctx.api_key,
            use_retina=use_retina,
            cache_root=resolve_cache_dir(),
        )

        full_eff_tile_px = 256 * (2 if use_retina else 1)

        async def fetch_one(tx: int, ty: int) -> np.ndarray:
            return await provider.get_tile_dem(ctx.zoom, tx, ty)

        def _on_tile_done(tile_count: int) -> None:
            if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                log_memory_usage(f'{label} after {tile_count} tiles')

        # Streaming fetch+assemble: each Terrain-RGB tile is written into
        # the result array and dropped immediately, so peak memory stays
        # at ~concurrency × tile_size (Mb-class) instead of N × tile_size
        # (Gb-class). Critical when running concurrently with topo at
        # retina.
        dem_full = await stream_fetch_assemble_dem(
            ctx,
            eff_tile_px=full_eff_tile_px,
            fetch_one=fetch_one,
            label=f'Загрузка DEM для {label}',
            on_tile_done=_on_tile_done,
        )
        gc.collect()
        dem_topo_cache.set_inmem('dem', cache_key, dem_full)
        dem_topo_cache.save('dem', cache_key, dem_full)
    else:
        logger.info(
            '%s: DEM cache HIT (%s), skipped %d tile fetches + decodes',
            label, cache_layer, len(ctx.tiles),
        )

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
    style: MapType = MapType.OUTDOORS,
    tile_size: int = TILE_SIZE,
) -> Image.Image:
    """
    Загрузка топографической основы. Возвращает PIL Image.

    tile_size + use_retina определяют URL у Mapbox (`tiles/<size>/...{@2x}?`).
    Чтобы кэш переиспользовался между map types, нужно вызывать
    с теми же `tile_size` и `use_retina`, что использует XYZ processor.
    Например, для elev_color/RH/Radar при общем кэше с HYBRID:
    `tile_size=TILE_SIZE_512, use_retina=True`.
    """
    from shared.diagnostics import crash_log

    logger.info('Загрузка топографической основы для %s', label)
    sp_topo = LiveSpinner('Загрузка топографической основы')
    sp_topo.start()

    topo_style_id = MAPBOX_STYLE_BY_TYPE[style]
    topo_tile_size = tile_size
    topo_use_retina = use_retina

    crash_log(
        f'_load_topo: START style={style.value}, tiles={len(ctx.tiles)}, '
        f'zoom={ctx.zoom}, retina={topo_use_retina}'
    )

    eff_tile_px_topo = topo_tile_size * (2 if topo_use_retina else 1)

    # Cache layers (in-mem then disk) before paying for fetch + decode +
    # assembly. Key includes style/tile_size since the same area at
    # different styles produces different topo bitmaps.
    cache_key = dem_topo_cache.make_area_key(
        kind_tag='topo_v1',
        zoom=ctx.zoom,
        eff_scale=ctx.eff_scale,
        use_retina=topo_use_retina,
        center_lat_wgs=ctx.center_lat_wgs,
        center_lng_wgs=ctx.center_lng_wgs,
        width_m=ctx.width_m,
        height_m=ctx.height_m,
        style_id=topo_style_id,
        tile_size=topo_tile_size,
    )
    cached_arr = dem_topo_cache.get_inmem('topo', cache_key)
    cache_layer = None
    if cached_arr is not None:
        cache_layer = 'in-mem'
    else:
        disk_hit = dem_topo_cache.load('topo', cache_key)
        if disk_hit is not None:
            cached_arr = disk_hit
            cache_layer = 'disk'
            dem_topo_cache.set_inmem('topo', cache_key, cached_arr)

    if cached_arr is not None:
        topo_base = Image.fromarray(cached_arr)
        # Stash the underlying numpy array on ctx so downstream
        # processors (elev_color in particular) can skip the
        # np.asarray(pil) round-trip — on a 9580² RGB topo PIL.Image
        # .fromarray + np.asarray together copy ~275 MB even though
        # they look like views, because PIL repacks RGB into its own
        # internal stride and np has to copy to read it back contiguously.
        # ~293 ms saving per warm elev_color rebuild.
        ctx.topo_cached_array = cached_arr
        sp_topo.stop('Топографическая основа загружена')
        logger.info(
            'topo cache HIT (%s), skipped %d tile fetches + decodes',
            cache_layer, len(ctx.tiles),
        )
        crash_log(
            f'_load_topo: cache HIT ({cache_layer}), size={topo_base.size}, '
            f'mode={topo_base.mode}',
        )
        return topo_base

    async def fetch_one(tx: int, ty: int) -> Image.Image:
        return await async_fetch_xyz_tile(
            client=ctx.client,
            api_key=ctx.api_key,
            style_id=topo_style_id,
            tile_size=topo_tile_size,
            z=ctx.zoom,
            x=tx,
            y=ty,
            use_retina=topo_use_retina,
        )

    # Streaming fetch+assemble: peak memory ~concurrency × tile instead of
    # all_tiles × tile. Critical for sharing the HYBRID retina cache —
    # without this, switching elev_color to 512+retina (1024² tiles) at
    # z=17 would peak at 4.5 GB during fetch and force auto-zoom-downgrade.
    crash_log(
        f'_load_topo: streaming fetch+assemble '
        f'({ctx.tiles_x}x{ctx.tiles_y} tiles, eff={eff_tile_px_topo}px)'
    )
    topo_base = await stream_fetch_assemble_xyz(
        ctx,
        eff_tile_px=eff_tile_px_topo,
        fetch_one=fetch_one,
        label=f'Топография для {label}',
    )
    crash_log(f'_load_topo: DONE, result={topo_base.size}, mode={topo_base.mode}')

    # Cache as numpy for fast round-trip through .npy memmap on disk.
    # np.asarray is a zero-copy view over PIL's buffer; .copy() snapshots
    # before set_inmem / save so the topo_base PIL object stays independent.
    cache_arr = np.asarray(topo_base).copy()
    dem_topo_cache.set_inmem('topo', cache_key, cache_arr)
    dem_topo_cache.save('topo', cache_key, cache_arr)
    # Hand the just-cached array to downstream processors directly so
    # they can avoid an np.asarray(topo_base) re-copy that costs ~293 ms
    # on 9580² RGB.
    ctx.topo_cached_array = cache_arr

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
    # Start DEM and topo fetches in parallel — same idiom as elev_color.
    # DEM is needed first (antenna_row/col / cp_elevation depend on it),
    # but topo continues to load in the background while we do the cheap
    # antenna math and wait at the second await.
    _t_loads = time.monotonic()
    dem_task = asyncio.create_task(
        _load_dem(ctx, use_retina=use_retina, label=label)
    )
    topo_task = asyncio.create_task(
        _load_topo(ctx, use_retina=use_retina, label=label)
    )
    try:
        dem_full, ds_factor = await dem_task
    except BaseException:
        topo_task.cancel()
        raise
    logger.info(
        '%s: DEM ready in %.3fs (topo still loading in parallel)',
        label, time.monotonic() - _t_loads,
    )

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

    # Await topo task started above. By this point DEM is done and we've
    # done the antenna-position math, so topo download/decode has had
    # the most overlap room possible.
    _t_topo_wait = time.monotonic()
    topo_base = await topo_task
    logger.info(
        '%s: topo ready after %.3fs additional wait',
        label, time.monotonic() - _t_topo_wait,
    )

    # Save topo base for cache at DEM size (matches downsampled DEM dimensions).
    # This ensures aligned crop+resize during recompute (same transform as coverage).
    dem_cache_size = (dem_full.shape[1], dem_full.shape[0])
    if topo_base.size != dem_cache_size:
        topo_for_cache = topo_base.resize(dem_cache_size, Image.Resampling.BILINEAR)
    else:
        topo_for_cache = topo_base.copy()

    ctx.rh_cache_topo_base = topo_for_cache.convert('L').convert('RGBA')

    # Save DEM for cache
    ctx.rh_cache_dem = dem_full  # same object; caller does `del dem_full` after use
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


def _blend_coverage_with_topo(
    ctx: MapDownloadContext,
    result: Image.Image,
    topo_base: Image.Image,
    label: str,
    *,
    cache_attr: str = 'rh_cache_coverage',
    overlay_alpha: float | None = None,
) -> Image.Image:
    """
    Post-coverage block shared by radio_horizon / radar_coverage / nsu_optimizer:
      1) move colorised coverage (PIL → numpy), strip alpha if present;
      2) cache RGBA at DEM-size for GUI's alpha slider (NOT target-size —
         GUI rescales to topo_base.size before blending anyway, so caching
         at full target is pure shm waste; 367 MB → 23 MB per z=17 build);
      3) upscale coverage to target size via cv2.resize (multi-thread);
      4) move topo to numpy + resize to target + convert to grayscale-3ch;
      5) blend via cv2.addWeighted (SIMD); wrap as PIL via fromarray.

    PIL only at the entry (close immediately after np.array forces a copy)
    and the exit (Image.fromarray on uint8 RGB is a zero-copy wrap).

    Replaces ~1.5 s of PIL .resize() + .convert() + Image.blend() with a
    ~0.7 s cv2/numpy chain on z=17 retina inputs.

    cache_attr/overlay_alpha let NSU plug in (nsu_cache_coverage / nsu_overlay_
    alpha); defaults match radio_horizon and radar_coverage.
    """
    _t_post = time.monotonic()
    target_w = ctx.crop_rect[2]
    target_h = ctx.crop_rect[3]
    if overlay_alpha is None:
        overlay_alpha = ctx.settings.radio_horizon_overlay_alpha

    # 1. Coverage → numpy. result is RGB at DEM size.
    rgb = np.array(result)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    result.close()
    del result

    # 2. Cache coverage at DEM size for interactive alpha slider.
    rgb_h_pre, rgb_w_pre = rgb.shape[:2]
    alpha_cache = np.full((rgb_h_pre, rgb_w_pre, 1), 255, dtype=np.uint8)
    coverage_rgba_cache = np.concatenate([rgb, alpha_cache], axis=2)
    setattr(ctx, cache_attr, Image.fromarray(coverage_rgba_cache))
    del alpha_cache, coverage_rgba_cache

    # 3. Upscale coverage to target size for blend onto base map.
    if rgb.shape[:2] != (target_h, target_w):
        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 4. Topo → numpy + resize + grayscale-as-RGB.
    topo_arr = np.array(topo_base)
    if topo_arr.ndim == 3 and topo_arr.shape[2] == 4:
        topo_arr = topo_arr[..., :3]
    topo_base.close()
    del topo_base
    if topo_arr.shape[:2] != (target_h, target_w):
        topo_arr = cv2.resize(
            topo_arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
    gray = cv2.cvtColor(topo_arr, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    del topo_arr, gray

    # 5. Blend: out = topo*(1-α) + coverage*α.
    blend_alpha = 1.0 - overlay_alpha
    blended = cv2.addWeighted(
        gray_3ch, 1.0 - blend_alpha, rgb, blend_alpha, 0.0
    )
    out = Image.fromarray(blended)

    logger.info(
        '%s: post-coverage numpy/cv2 block in %.3fs',
        label, time.monotonic() - _t_post,
    )
    del rgb, gray_3ch, blended
    gc.collect()
    return out


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

    result = _blend_coverage_with_topo(ctx, result, topo_base, 'radio_horizon')
    logger.info('Карта радиогоризонта построена')
    return result

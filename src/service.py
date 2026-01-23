import asyncio
import contextlib
import gc
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

from constants import (
    ASYNC_MAX_CONCURRENCY,
    CENTER_CROSS_COLOR,
    CENTER_CROSS_LENGTH_M,
    CENTER_CROSS_LINE_WIDTH_M,
    CONTOUR_BLOCK_EDGE_PAD_PX,
    CONTOUR_COLOR,
    CONTOUR_INDEX_COLOR,
    CONTOUR_INDEX_EVERY,
    CONTOUR_INDEX_WIDTH,
    CONTOUR_INTERVAL_M,
    CONTOUR_LABEL_EDGE_MARGIN_M,
    CONTOUR_LABEL_GAP_ENABLED,
    CONTOUR_LABEL_GAP_PADDING_M,
    CONTOUR_LABEL_MIN_SEG_LEN_M,
    CONTOUR_LABEL_SPACING_M,
    CONTOUR_LABELS_ENABLED,
    CONTOUR_LOG_MEMORY_EVERY_TILES,
    CONTOUR_PASS2_QUEUE_MAXSIZE,
    CONTOUR_SEED_DOWNSAMPLE,
    CONTOUR_WIDTH,
    CONTROL_POINT_COLOR,
    CONTROL_POINT_SIZE_M,
    DOWNLOAD_CONCURRENCY,
    EARTH_RADIUS_M,
    ELEVATION_LEGEND_STEP_M,
    ELEVATION_USE_RETINA,
    GRID_COLOR,
    GRID_STEP_M,
    MAPBOX_STYLE_BY_TYPE,
    MAX_OUTPUT_PIXELS,
    MAX_ZOOM,
    PIL_DISABLE_LIMIT,
    RADIO_HORIZON_COLOR_RAMP,
    RADIO_HORIZON_TOPO_OVERLAY_ALPHA,
    RADIO_HORIZON_USE_RETINA,
    ROTATION_PAD_MIN_PX,
    ROTATION_PAD_RATIO,
    TILE_SIZE,
    XYZ_TILE_SIZE,
    XYZ_USE_RETINA,
    MapType,
    default_map_type,
    map_type_to_style_id,
)
from contours import draw_contour_labels as _draw_contour_labels
from contours_helpers import tx_ty_from_index
from coords_sk42 import build_sk42_gk_crs as _build_sk42_gk_crs
from coords_sk42 import determine_zone as _determine_zone
from coords_sk42 import validate_sk42_bounds as _validate_sk42_bounds
from diagnostics import log_memory_usage, log_thread_status
from domen import MapSettings
from elevation_provider import ElevationTileProvider

# Common helpers extracted to reduce duplication
from geometry import tile_overlap_rect_common as _tile_overlap_rect_common
from http_client import cleanup_sqlite_cache as _cleanup_sqlite_cache
from http_client import make_http_session as _make_http_session
from http_client import resolve_cache_dir as _resolve_cache_dir
from http_client import validate_style_api as _validate_api_and_connectivity
from http_client import validate_terrain_api as _validate_terrain_api
from image import (
    assemble_and_crop,
    center_crop,
    draw_axis_aligned_km_grid,
    draw_elevation_legend,
    draw_label_with_bg,
    draw_label_with_subscript_bg,
    load_grid_font,
    rotate_keep_size,
)
from image_io import build_save_kwargs as _build_save_kwargs
from image_io import save_jpeg as _save_jpeg
from preview import publish_preview_image
from progress import ConsoleProgress, LiveSpinner
from radio_horizon import (
    compute_and_colorize_radio_horizon,
    compute_downsample_factor,
    downsample_dem,
)
from render.contours_builder import build_seed_polylines
from topography import (
    ELEV_MIN_RANGE_M,
    ELEV_PCTL_HI,
    ELEV_PCTL_LO,
    ELEVATION_COLOR_RAMP,
    async_fetch_xyz_tile,
    build_transformers_sk42,
    choose_zoom_with_limit,
    compute_rotation_deg_for_east_axis,
    compute_xyz_coverage,
    crs_sk42_geog,
    decode_terrain_rgb_to_elevation_m,
    effective_scale_for_xyz,
    estimate_crop_size_px,
    latlng_to_pixel_xy,
    meters_per_pixel,
)

logger = logging.getLogger(__name__)


async def download_satellite_rectangle(
    center_x_sk42_gk: float,
    center_y_sk42_gk: float,
    width_m: float,
    height_m: float,
    api_key: str,
    output_path: str,
    max_zoom: int = MAX_ZOOM,
    settings: MapSettings | None = None,
) -> str:
    """Полный конвейер."""
    overall_start_time = time.monotonic()
    logger.info('=== ОБЩИЙ ТАЙМЕР: старт download_satellite_rectangle ===')

    # Handle default settings if not provided
    if settings is None:
        settings = MapSettings(
            from_x_high=54,
            from_y_high=74,
            to_x_high=54,
            to_y_high=74,
            from_x_low=14,
            from_y_low=43,
            to_x_low=23,
            to_y_low=49,
            output_path=output_path,
            grid_width_m=5.0,
            grid_font_size_m=100.0,
            grid_text_margin_m=50.0,
            grid_label_bg_padding_m=10.0,
            mask_opacity=0.35,
        )

    # Определяем тип карты и стратегию
    mt = getattr(settings, 'map_type', default_map_type())
    try:
        mt_enum = MapType(mt) if not isinstance(mt, MapType) else mt
    except Exception:
        mt_enum = default_map_type()

    style_id: str | None = None
    if mt_enum in (
        MapType.SATELLITE,
        MapType.HYBRID,
        MapType.STREETS,
        MapType.OUTDOORS,
    ):
        style_id = map_type_to_style_id(mt_enum)
        logger.info(
            'Тип карты: %s; style_id=%s; tile_size=%s; retina=%s',
            mt_enum,
            style_id,
            XYZ_TILE_SIZE,
            XYZ_USE_RETINA,
        )
        await _validate_api_and_connectivity(api_key, style_id)
    elif mt_enum == MapType.ELEVATION_COLOR:
        logger.info(
            'Тип карты: %s (Terrain-RGB, цветовая шкала); retina=%s',
            mt_enum,
            ELEVATION_USE_RETINA,
        )
    elif mt_enum == MapType.ELEVATION_CONTOURS:
        logger.info(
            'Тип карты: %s (Terrain-RGB, контуры); retina=%s',
            mt_enum,
            ELEVATION_USE_RETINA,
        )
    elif mt_enum == MapType.RADIO_HORIZON:
        if not settings.control_point_enabled:
            msg = 'Для карты радиогоризонта необходимо включить контрольную точку'
            raise ValueError(msg)
        logger.info(
            'Тип карты: %s (радиогоризонт); высота антенны=%s м; retina=%s',
            mt_enum,
            settings.antenna_height_m,
            ELEVATION_USE_RETINA,
        )
    else:
        # Нереализованные режимы пока откатываются к Спутнику
        logger.warning(
            'Выбран режим высот (%s), пока не реализовано. Используется Спутник.',
            mt_enum,
        )
        style_id = map_type_to_style_id(default_map_type())
        await _validate_api_and_connectivity(api_key, style_id)

    # Выбор масштаба под тип карты
    is_elev_color = False
    is_elev_contours = False
    is_radio_horizon = False
    try:
        mt_val = MapType(mt) if not isinstance(mt, MapType) else mt
        is_elev_color = mt_val == MapType.ELEVATION_COLOR
        is_elev_contours = mt_val == MapType.ELEVATION_CONTOURS
        is_radio_horizon = mt_val == MapType.RADIO_HORIZON
    except Exception:
        is_elev_color = False
        is_elev_contours = False
        is_radio_horizon = False

    # Флаг оверлея изолиний поверх выбранного типа карты
    overlay_contours = bool(getattr(settings, 'overlay_contours', False))

    # Переменные для хранения диапазона высот (для легенды)
    elev_min_m: float | None = None
    elev_max_m: float | None = None

    if is_elev_color or is_elev_contours:
        # Для Terrain-RGB базовый тайл 256px; @2x даёт 512
        eff_scale = effective_scale_for_xyz(256, use_retina=ELEVATION_USE_RETINA)
        # Ранняя проверка доступности Terrain-RGB
        await _validate_terrain_api(api_key)
    elif is_radio_horizon:
        # Для радиогоризонта используем пониженное разрешение для экономии памяти
        eff_scale = effective_scale_for_xyz(256, use_retina=RADIO_HORIZON_USE_RETINA)
        await _validate_terrain_api(api_key)
    else:
        eff_scale = effective_scale_for_xyz(XYZ_TILE_SIZE, use_retina=XYZ_USE_RETINA)

    # Единый ретина-фактор для контурного оверлея (синхронизирован с базовой картой)
    if is_elev_color or is_elev_contours:
        contour_use_retina = ELEVATION_USE_RETINA
    elif is_radio_horizon:
        contour_use_retina = RADIO_HORIZON_USE_RETINA
    else:
        contour_use_retina = XYZ_USE_RETINA

    # Подготовка — конвертация из Гаусса-Крюгера в географические координаты СК-42
    sp = LiveSpinner('Подготовка: определение зоны')
    sp.start()

    # Зона и система координат
    zone = _determine_zone(center_x_sk42_gk)
    crs_sk42_gk = _build_sk42_gk_crs(zone)
    sp.stop('Подготовка: зона определена')

    sp = LiveSpinner('Подготовка: конвертация из ГК в СК-42')
    sp.start()
    # Конвертируем из Гаусса-Крюгера в географические СК-42
    t_sk42_from_gk = Transformer.from_crs(crs_sk42_gk, crs_sk42_geog, always_xy=True)
    center_lng_sk42, center_lat_sk42 = t_sk42_from_gk.transform(
        center_x_sk42_gk,
        center_y_sk42_gk,
    )
    sp.stop('Подготовка: координаты СК-42 готовы')

    # Проверка зоны применимости СК-42 (без формирования карты вне зоны)
    _validate_sk42_bounds(center_lng_sk42, center_lat_sk42)

    # Проверка контрольной точки на попадание в границы карты
    if settings.control_point_enabled:
        # Вычисляем границы карты в СК-42 ГК
        half_width = width_m / 2
        half_height = height_m / 2
        map_left = center_x_sk42_gk - half_width
        map_right = center_x_sk42_gk + half_width
        map_bottom = center_y_sk42_gk - half_height
        map_top = center_y_sk42_gk + half_height

        control_x = settings.control_point_x_sk42_gk
        control_y = settings.control_point_y_sk42_gk

        if not (
            map_left <= control_x <= map_right and map_bottom <= control_y <= map_top
        ):
            # Note on naming: map_left/right are along easting (восток), map_bottom/top are along northing (север).
            # Military printout below uses X=northing, Y=easting for clarity.
            msg = (
                f'Контрольная точка X(север)={control_y:.0f}, Y(восток)={control_x:.0f} выходит за пределы карты. '
                f'Границы карты: Y(восток)=[{map_left:.0f}, {map_right:.0f}], X(север)=[{map_bottom:.0f}, {map_top:.0f}]'
            )
            raise ValueError(msg)

    sp = LiveSpinner('Подготовка: создание трансформеров')
    sp.start()
    # Создаем трансформеры для работы с полученными координатами
    custom_helmert = getattr(settings, 'custom_helmert', None)
    if not custom_helmert:
        logger.warning(
            'Дата трансформация СК-42↔WGS84 выполняется без явных региональных параметров; '
            'возможен систематический сдвиг 100–300 м. Укажите параметры Helmert в профиле.'
        )
    else:
        logger.info(
            'Используются пользовательские параметры Helmert: '
            f'dx={custom_helmert[0]}, dy={custom_helmert[1]}, dz={custom_helmert[2]}, '
            f'rx={custom_helmert[3]}", ry={custom_helmert[4]}", rz={custom_helmert[5]}", '
            f'ds={custom_helmert[6]}ppm'
        )
    t_sk42_to_wgs, t_wgs_to_sk42 = build_transformers_sk42(
        custom_helmert=custom_helmert,
    )
    sp.stop('Подготовка: трансформеры готовы')

    sp = LiveSpinner('Подготовка: конвертация центра в WGS84')
    sp.start()
    center_lng_wgs, center_lat_wgs = t_sk42_to_wgs.transform(
        center_lng_sk42,
        center_lat_sk42,
    )
    sp.stop('Подготовка: центр WGS84 готов')

    # Log control point coordinates using military notation: X=northing (север), Y=easting (восток)
    # Note: Internally and in PROJ/pyproj, the axis order with always_xy=True is (easting, northing).
    # Here control_x is GK easting, control_y is GK northing. For military notation we print X=control_y, Y=control_x.
    try:
        if getattr(settings, 'control_point_enabled', False):
            control_x = settings.control_point_x_sk42_gk  # GK easting (восток)
            control_y = settings.control_point_y_sk42_gk  # GK northing (север)
            # Convert GK -> SK-42 geographic using the same mechanism as grid (input order: easting, northing)
            cp_lng_sk42, cp_lat_sk42 = t_sk42_from_gk.transform(control_x, control_y)
            # Convert SK-42 geographic -> WGS84 using Helmert-aware transformer (input order: lon, lat)
            cp_lng_wgs, cp_lat_wgs = t_sk42_to_wgs.transform(cp_lng_sk42, cp_lat_sk42)
            logger.info(
                'Контрольная точка: СК-42 ГК X(север)=%.3f, Y(восток)=%.3f; WGS84 lat=%.8f, lon=%.8f',
                control_y,
                control_x,
                cp_lat_wgs,
                cp_lng_wgs,
            )
    except Exception as e:
        logger.warning('Не удалось вывести координаты контрольной точки: %s', e)

    sp = LiveSpinner('Подготовка: подбор zoom')
    sp.start()
    zoom = choose_zoom_with_limit(
        center_lat=center_lat_wgs,
        width_m=width_m,
        height_m=height_m,
        desired_zoom=max_zoom,
        scale=eff_scale,
        max_pixels=MAX_OUTPUT_PIXELS,
    )
    sp.stop('Подготовка: zoom выбран')

    if PIL_DISABLE_LIMIT:
        Image.MAX_IMAGE_PIXELS = None

    sp = LiveSpinner('Подготовка: оценка размера')
    sp.start()
    target_w_px, target_h_px, _ = estimate_crop_size_px(
        center_lat_wgs,
        width_m,
        height_m,
        zoom,
        eff_scale,
    )
    sp.stop('Подготовка: размер оценён')

    sp = LiveSpinner('Подготовка: расчёт покрытия XYZ')
    sp.start()
    base_pad = round(min(target_w_px, target_h_px) * ROTATION_PAD_RATIO)
    pad_px = max(base_pad, ROTATION_PAD_MIN_PX)
    tiles, (tiles_x, tiles_y), crop_rect, map_params = compute_xyz_coverage(
        center_lat=center_lat_wgs,
        center_lng=center_lng_wgs,
        width_m=width_m,
        height_m=height_m,
        zoom=zoom,
        eff_scale=eff_scale,
        pad_px=pad_px,
    )
    sp.stop('Подготовка: покрытие рассчитано')

    tile_label = (
        'Загрузка Terrain-RGB тайлов'
        if (is_elev_color or is_elev_contours)
        else 'Загрузка XYZ-тайлов'
    )
    tile_progress = ConsoleProgress(total=len(tiles), label=tile_label)
    semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY)

    cache_dir_resolved = _resolve_cache_dir()
    session_ctx = _make_http_session(cache_dir_resolved)

    log_memory_usage('before tile download')
    log_thread_status('before tile download')

    try:
        async with session_ctx as client:
            tile_count = 0

            if is_elev_color:
                # Two-pass streaming without storing full DEM (color ramp)
                # Helper: build LUT from color ramp for fast palette lookup
                def _lerp(a: float, b: float, t: float) -> float:
                    return a + (b - a) * t

                lut_size = 2048
                _lut: list[tuple[int, int, int]] = []
                ramp = ELEVATION_COLOR_RAMP
                # Precompute cumulative ramp into fixed-size LUT
                for i in range(lut_size):
                    t = i / (lut_size - 1)
                    # find segment
                    for j in range(1, len(ramp)):
                        t0, c0 = ramp[j - 1]
                        t1, c1 = ramp[j]
                        if t <= t1 or j == len(ramp) - 1:
                            local = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                            r = round(_lerp(c0[0], c1[0], local))
                            g = round(_lerp(c0[1], c1[1], local))
                            b = round(_lerp(c0[2], c1[2], local))
                            _lut.append((r, g, b))
                            break

                def _color_at(t: float) -> tuple[int, int, int]:
                    # Clamp and map to LUT index
                    if t <= 0.0:
                        return _lut[0]
                    if t >= 1.0:
                        return _lut[-1]
                    idx = int(t * (lut_size - 1))
                    return _lut[idx]

                provider_main = ElevationTileProvider(
                    client=client, api_key=api_key, use_retina=ELEVATION_USE_RETINA
                )

                full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)

                # Single pass: load tiles, sample elevations, and cache for colorization
                from elevation.stats import compute_elevation_range as _elev_range
                from elevation.stats import sample_elevation_percentiles as _sample_elev

                tile_progress.label = 'Загрузка и анализ DEM'

                async def _get_dem_tile(xw: int, yw: int) -> Image.Image:
                    return await provider_main.get_tile_image(zoom, xw, yw)

                samples, seen_count, tile_cache = await _sample_elev(
                    enumerate(tiles),
                    tiles_x=tiles_x,
                    crop_rect=crop_rect,
                    full_eff_tile_px=full_eff_tile_px,
                    get_tile_image=_get_dem_tile,
                    max_samples=50000,
                    rng_seed=42,
                    on_progress=tile_progress.step,
                    semaphore=semaphore,
                    cache_tiles=True,
                )
                with contextlib.suppress(Exception):
                    tile_progress.close()
                logger.info(
                    'DEM sampling reservoir: kept=%s seen~=%s', len(samples), seen_count
                )
                lo, hi = _elev_range(
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
                # Сохраняем диапазон высот для легенды
                elev_min_m = lo_rounded
                elev_max_m = hi_rounded

                # Pass B: render directly to output image using cached tiles (no network)
                result = Image.new('RGB', (crop_rect[2], crop_rect[3]))
                tile_progress = ConsoleProgress(
                    total=len(tiles), label='Окрашивание DEM'
                )
                tile_count = 0

                queue: asyncio.Queue[
                    tuple[int, int, int, int, int, int, Image.Image]
                ] = asyncio.Queue(maxsize=CONTOUR_PASS2_QUEUE_MAXSIZE)
                paste_lock = asyncio.Lock()

                async def producer(idx_xy: tuple[int, tuple[int, int]]) -> None:
                    nonlocal tile_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = tx_ty_from_index(idx, tiles_x)
                    ov = _tile_overlap_rect_common(tx, ty, crop_rect, full_eff_tile_px)
                    if ov is None:
                        await tile_progress.step(1)
                        return
                    # Use cached tile from Pass A (no network request)
                    if tile_cache and (tile_x_world, tile_y_world) in tile_cache:
                        img = tile_cache[(tile_x_world, tile_y_world)]
                    else:
                        # Fallback to network if cache miss (shouldn't happen)
                        async with semaphore:
                            img = await provider_main.get_tile_image(
                                zoom, tile_x_world, tile_y_world
                            )
                    # Enqueue with metadata for consumers
                    x0, y0, x1, y1 = ov
                    await queue.put((tx, ty, x0, y0, x1, y1, img))
                    # Progress is stepped when consumer finishes to reflect actual painting

                # Precompute normalization coeffs for consumers (used with DEM values)
                async def consumer() -> None:
                    nonlocal tile_count
                    # Precompute linear coefficients for t = ar*R + ag*G + ab*B + a0
                    ar = 0.1 * 65536.0 * inv
                    ag = 0.1 * 256.0 * inv
                    ab = 0.1 * 1.0 * inv
                    a0 = (-10000.0 - lo_rounded) * inv

                    while True:
                        item = await queue.get()
                        if item is None:  # type: ignore[comparison-overlap]
                            queue.task_done()
                            break
                        tx, ty, x0, y0, x1, y1, img = item
                        try:
                            cx, cy, _, _ = crop_rect
                            dx0 = x0 - cx
                            dy0 = y0 - cy
                            base_x = tx * full_eff_tile_px
                            base_y = ty * full_eff_tile_px
                            # NumPy fast path for DEM colorization
                            arr = np.asarray(img, dtype=np.uint8)  # HxWx3
                            # Slice overlap
                            sub = arr[
                                y0 - base_y : y1 - base_y,
                                x0 - base_x : x1 - base_x,
                                :3,
                            ]
                            # Compute t linear combination
                            t = (
                                ar * sub[..., 0].astype(np.float32)
                                + ag * sub[..., 1].astype(np.float32)
                                + ab * sub[..., 2].astype(np.float32)
                                + a0
                            )
                            # Clamp and map to LUT
                            _l = np.clip(
                                (t * (lut_size - 1)).astype(np.int32),
                                0,
                                lut_size - 1,
                            )
                            lut = np.asarray(_lut, dtype=np.uint8)
                            rgb = lut[_l]
                            block_img = Image.fromarray(rgb, mode='RGB')
                            # Paste into result under lock (PIL isn't thread-safe)
                            async with paste_lock:
                                result.paste(block_img, (dx0, dy0))
                        finally:
                            # Don't close img here - it's from cache, will be closed later
                            queue.task_done()
                            tile_count += 1
                            if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                                log_memory_usage(f'colorize after {tile_count} tiles')
                            await tile_progress.step(1)

                # Launch producers/consumers with guarded cancellation
                producers = [
                    asyncio.create_task(producer(pair)) for pair in enumerate(tiles)
                ]
                cpu_workers = max(1, min(os.cpu_count() or 2, 4))
                consumers = [
                    asyncio.create_task(consumer()) for _ in range(cpu_workers)
                ]
                try:
                    await asyncio.gather(*producers)
                    for _ in consumers:
                        await queue.put(None)  # type: ignore[arg-type]
                    await queue.join()
                    await asyncio.gather(*consumers)
                except Exception:
                    for task in producers + consumers:
                        task.cancel()
                    # attempt to drain queue and stop consumers
                    for _ in consumers:
                        await queue.put(None)  # type: ignore[arg-type]
                    await queue.join()
                    await asyncio.gather(*consumers, return_exceptions=True)
                    raise
                finally:
                    tile_progress.close()
                    # Clean up tile cache
                    if tile_cache:
                        for img in tile_cache.values():
                            with contextlib.suppress(Exception):
                                img.close()
                        tile_cache.clear()
            elif is_elev_contours:
                # Two-pass streaming without storing full DEM (contours)

                full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)

                # Pass A: sample min/max elevations and build global low-res DEM seed
                max_samples = 50000
                samples_contours: list[float] = []
                seen = 0

                rng = random.Random(42)  # noqa: S311
                tile_progress.label = 'Проверка диапазона высот (проход 1/2)'

                # Prepare low-res DEM seed canvas in crop coordinates

                seed_ds = max(2, int(CONTOUR_SEED_DOWNSAMPLE))
                cx, cy, cw, ch = crop_rect
                seed_w = max(1, (cw + seed_ds - 1) // seed_ds)
                seed_h = max(1, (ch + seed_ds - 1) // seed_ds)
                # initialize with None to track unfilled cells; will average contributions
                seed_sum: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
                seed_cnt: list[list[int]] = [[0] * seed_w for _ in range(seed_h)]

                async def fetch_and_sample2(
                    idx_xy: tuple[int, tuple[int, int]],
                ) -> None:
                    nonlocal seen, samples_contours, tile_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = tx_ty_from_index(idx, tiles_x)
                    if (
                        _tile_overlap_rect_common(tx, ty, crop_rect, full_eff_tile_px)
                        is None
                    ):
                        await tile_progress.step(1)
                        return
                    async with semaphore:
                        img = await provider_main.get_tile_image(
                            zoom, tile_x_world, tile_y_world
                        )
                        dem_tile = decode_terrain_rgb_to_elevation_m(img)
                        h = len(dem_tile)
                        w = len(dem_tile[0]) if h else 0
                        if h and w:
                            # reservoir sampling for min/max
                            step_y = max(1, h // 32)
                            step_x = max(1, w // 32)
                            off_y = (
                                rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
                            )
                            off_x = (
                                rng.randrange(0, min(step_x, w)) if step_x > 1 else 0
                            )
                            for ry in range(off_y, h, step_y):
                                row = dem_tile[ry]
                                for rx in range(off_x, w, step_x):
                                    v = row[rx]
                                    seen += 1
                                    if len(samples_contours) < max_samples:
                                        samples_contours.append(v)
                                    else:
                                        j = rng.randrange(0, seen)
                                        if j < max_samples:
                                            samples_contours[j] = v
                            # accumulate into low-res seed for this tile overlap
                            ov = _tile_overlap_rect_common(
                                tx, ty, crop_rect, full_eff_tile_px
                            )
                            if ov is not None:
                                x0, y0, x1, y1 = ov
                                base_x = tx * full_eff_tile_px
                                base_y = ty * full_eff_tile_px
                                # iterate cropped region at stride seed_ds
                                for yy in range(y0, y1, seed_ds):
                                    sy = (yy - cy) // seed_ds
                                    if sy < 0 or sy >= seed_h:
                                        continue
                                    src_y = yy - base_y
                                    row_src = dem_tile[src_y]
                                    for xx in range(x0, x1, seed_ds):
                                        sx = (xx - cx) // seed_ds
                                        if sx < 0 or sx >= seed_w:
                                            continue
                                        src_x = xx - base_x
                                        v = row_src[src_x]
                                        seed_sum[sy][sx] += v
                                        seed_cnt[sy][sx] += 1
                    await tile_progress.step(1)
                    tile_count += 1
                    if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                        log_memory_usage(f'pass1(contours) after {tile_count} tiles')

                try:
                    async with asyncio.TaskGroup() as tg:
                        for pair in enumerate(tiles):
                            tg.create_task(fetch_and_sample2(pair))
                finally:
                    tile_progress.close()
                if samples_contours:
                    mn = min(samples_contours)
                    mx = max(samples_contours)
                else:
                    mn, mx = 0.0, 1.0
                if mx < mn:
                    mn, mx = mx, mn

                # Finalize low-res DEM seed (average accumulated values)
                seed_dem: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
                filled = 0
                for sy in range(seed_h):
                    row_s = seed_sum[sy]
                    row_c = seed_cnt[sy]
                    out = seed_dem[sy]
                    for sx in range(seed_w):
                        c = row_c[sx]
                        if c > 0:
                            out[sx] = row_s[sx] / float(c)
                            filled += 1
                        else:
                            out[sx] = (mn + mx) * 0.5  # fallback to mid
                logger.info(
                    'Seed DEM low-res size=%sx%s, filled=%s/%s',
                    seed_w,
                    seed_h,
                    filled,
                    seed_w * seed_h,
                )

                # Build contour levels

                if CONTOUR_INTERVAL_M <= 0:
                    interval = 25.0
                else:
                    interval = float(CONTOUR_INTERVAL_M)
                start = math.floor(mn / interval) * interval
                end = math.ceil(mx / interval) * interval
                levels: list[float] = []
                k = 0
                v = start
                while v <= end:
                    levels.append(v)
                    k += 1
                    v = start + k * interval

                # Pass B: draw contours into result image using global low-res seed polylines
                result = Image.new(
                    'RGB', (crop_rect[2], crop_rect[3]), color=(255, 255, 255)
                )
                tile_progress = ConsoleProgress(
                    total=len(tiles), label='Построение изогипс (проход 2/2)'
                )
                tile_count = 0

                # Build global polylines from seed_dem via marching squares
                seed_polylines = build_seed_polylines(seed_dem, levels, seed_h, seed_w)

                queue_contours: asyncio.Queue[tuple[int, int, int, int, int, int]] = (
                    asyncio.Queue(maxsize=CONTOUR_PASS2_QUEUE_MAXSIZE)
                )
                paste_lock = asyncio.Lock()

                async def producer2(idx_xy: tuple[int, tuple[int, int]]) -> None:
                    nonlocal tile_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = tx_ty_from_index(idx, tiles_x)
                    ov = _tile_overlap_rect_common(tx, ty, crop_rect, full_eff_tile_px)
                    if ov is None:
                        await tile_progress.step(1)
                        return
                    # We don't need image in pass B anymore; only metadata for block
                    await queue_contours.put((tx, ty, *ov))

                async def consumer2() -> None:
                    nonlocal tile_count
                    while True:
                        item = await queue_contours.get()
                        if item is None:  # type: ignore[comparison-overlap]
                            queue_contours.task_done()
                            break
                        tx, ty, x0, y0, x1, y1 = item
                        try:
                            cx, cy, _, _ = crop_rect
                            dx0 = x0 - cx
                            dy0 = y0 - cy
                            block_w = x1 - x0
                            block_h = y1 - y0

                            # local RGBA buffer with 1px edge pad
                            edge_pad = CONTOUR_BLOCK_EDGE_PAD_PX
                            pad_w = block_w + 2 * edge_pad
                            pad_h = block_h + 2 * edge_pad
                            tmp = Image.new('RGBA', (pad_w, pad_h), (0, 0, 0, 0))
                            draw = ImageDraw.Draw(tmp)

                            # Block bbox in crop coords
                            bx0, by0, bx1, by1 = x0, y0, x1, y1

                            # iterate levels and draw clipped polylines
                            for li, _level in enumerate(levels):
                                is_index = (li % max(1, int(CONTOUR_INDEX_EVERY))) == 0
                                color = (
                                    CONTOUR_INDEX_COLOR if is_index else CONTOUR_COLOR
                                )
                                width = int(
                                    CONTOUR_INDEX_WIDTH if is_index else CONTOUR_WIDTH
                                )
                                for poly in seed_polylines.get(li, []):
                                    # map seed coords to crop pixel coords by multiplying by seed_ds and adding crop origin
                                    pts_crop: list[tuple[float, float]] = [
                                        (cx + p[0] * seed_ds, cy + p[1] * seed_ds)
                                        for p in poly
                                    ]
                                    # clip to block bbox (simple bbox clip by skipping if all outside and no crossing)
                                    # quick bbox test
                                    minx = min(p[0] for p in pts_crop)
                                    maxx = max(p[0] for p in pts_crop)
                                    miny = min(p[1] for p in pts_crop)
                                    maxy = max(p[1] for p in pts_crop)
                                    if (
                                        maxx < bx0
                                        or minx > bx1
                                        or maxy < by0
                                        or miny > by1
                                    ):
                                        continue
                                    # render as polyline in block-local coords
                                    prev = None
                                    for px, py in pts_crop:
                                        lx = px - bx0 + edge_pad
                                        ly = py - by0 + edge_pad
                                        if prev is not None:
                                            draw.line(
                                                (prev[0], prev[1], lx, ly),
                                                fill=(*list(color), 255),
                                                width=width,
                                            )
                                        prev = (lx, ly)

                            composed = tmp.crop(
                                (
                                    edge_pad,
                                    edge_pad,
                                    edge_pad + block_w,
                                    edge_pad + block_h,
                                )
                            )
                            async with paste_lock:
                                result.paste(composed, (dx0, dy0), composed)
                        finally:
                            queue_contours.task_done()
                            tile_count += 1
                            if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                                log_memory_usage(
                                    f'pass2(contours) after {tile_count} tiles'
                                )
                            await tile_progress.step(1)

                # Launch producers and consumers with guarded cancellation
                producers = [
                    asyncio.create_task(producer2(pair)) for pair in enumerate(tiles)
                ]
                cpu_workers = max(1, min(os.cpu_count() or 2, 4))
                consumers = [
                    asyncio.create_task(consumer2()) for _ in range(cpu_workers)
                ]
                try:
                    await asyncio.gather(*producers)
                    for _ in consumers:
                        await queue_contours.put(None)  # type: ignore[arg-type]
                    await queue_contours.join()
                    await asyncio.gather(*consumers)
                except Exception:
                    for task in producers + consumers:
                        task.cancel()
                    with contextlib.suppress(Exception):
                        for _ in consumers:
                            await queue_contours.put(None)  # type: ignore[arg-type]
                        await queue_contours.join()
                        await asyncio.gather(*consumers, return_exceptions=True)
                    raise
                finally:
                    tile_progress.close()

                if is_elev_contours and CONTOUR_LABELS_ENABLED:
                    try:
                        # Diagnostics for labels context
                        logger.info(
                            'Подписи изогипс: подготовка (zoom=%d, crop=%dx%d at (%d,%d), seed_ds=%d, levels=%d)',
                            zoom,
                            crop_rect[2],
                            crop_rect[3],
                            crop_rect[0],
                            crop_rect[1],
                            int(seed_ds),
                            len(levels),
                        )
                        lat_rad = math.radians(center_lat_wgs)
                        # Web Mercator meters-per-pixel at given zoom and latitude
                        # Учитываем retina-фактор для корректного mpp
                        elev_retina_factor = 2 if ELEVATION_USE_RETINA else 1
                        mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
                            TILE_SIZE * elev_retina_factor * (2**zoom)
                        )
                        logger.info(
                            'Подписи изогипс: mpp=%.6f (TILE_SIZE=%d, retina=%s)', mpp, TILE_SIZE, ELEVATION_USE_RETINA
                        )
                        # Первый проход: получаем позиции подписей БЕЗ размещения на изображении
                        label_bboxes = _draw_contour_labels(
                            result,
                            seed_polylines,
                            levels,
                            crop_rect,
                            seed_ds,
                            mpp,
                            dry_run=True,
                        )
                        logger.info(
                            'Подписи изогипс: dry_run завершён, кандидатов=%d',
                            len(label_bboxes),
                        )
                        if not label_bboxes:
                            logger.warning(
                                'Подписи изогипс: dry_run вернул 0 кандидатов — проверьте пороги (spacing=%d,min_len=%d,edge=%d) и геометрию полилиний',
                                int(CONTOUR_LABEL_SPACING_M),
                                int(CONTOUR_LABEL_MIN_SEG_LEN_M),
                                int(CONTOUR_LABEL_EDGE_MARGIN_M),
                            )

                        # Создаем разрывы линий контуров в местах подписей

                        if CONTOUR_LABEL_GAP_ENABLED and label_bboxes:
                            draw = ImageDraw.Draw(result)
                            gap_padding = max(
                                1,
                                round(
                                    CONTOUR_LABEL_GAP_PADDING_M / max(1e-9, mpp)
                                ),
                            )
                            for bbox in label_bboxes:
                                x0, y0, x1, y1 = bbox
                                # Расширяем область на gap_padding
                                gap_area = (
                                    max(0, x0 - gap_padding),
                                    max(0, y0 - gap_padding),
                                    min(result.width, x1 + gap_padding),
                                    min(result.height, y1 + gap_padding),
                                )
                                # "Стираем" контуры в этой области (заливаем белым)
                                draw.rectangle(gap_area, fill=(255, 255, 255))

                        # Второй проход: размещаем подписи поверх созданных разрывов
                        placed_after = _draw_contour_labels(
                            result,
                            seed_polylines,
                            levels,
                            crop_rect,
                            seed_ds,
                            mpp,
                            dry_run=False,
                        )
                        logger.info(
                            'Подписи изогипс: финальная отрисовка, размещено=%d',
                            len(placed_after),
                        )
                    except Exception as e:
                        logger.warning(
                            'Не удалось нанести подписи изогипс: %s', e, exc_info=True
                        )
            elif is_radio_horizon:
                # Карта радиогоризонта: загружаем DEM и вычисляем минимальные высоты БпЛА
                from topography import assemble_dem

                provider_rh = ElevationTileProvider(
                    client=client, api_key=api_key, use_retina=RADIO_HORIZON_USE_RETINA
                )

                full_eff_tile_px = 256 * (2 if RADIO_HORIZON_USE_RETINA else 1)

                tile_progress.label = 'Загрузка DEM для радиогоризонта'

                # Загружаем все DEM тайлы (numpy arrays)
                dem_tiles_data: list[np.ndarray] = []

                async def fetch_dem_tile(
                    idx_xy: tuple[int, tuple[int, int]],
                ) -> tuple[int, np.ndarray]:
                    nonlocal tile_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    async with semaphore:
                        dem_tile = await provider_rh.get_tile_dem(
                            zoom, tile_x_world, tile_y_world
                        )
                        await tile_progress.step(1)
                        tile_count += 1
                        if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                            log_memory_usage(f'radio_horizon after {tile_count} tiles')
                        return idx, dem_tile

                tasks_rh = [fetch_dem_tile(pair) for pair in enumerate(tiles)]
                results_rh = await asyncio.gather(*tasks_rh)
                tile_progress.close()

                results_rh.sort(key=lambda t: t[0])
                dem_tiles_data = [dem for _, dem in results_rh]

                # Собираем единую DEM (numpy array)
                dem_full = assemble_dem(
                    tiles_data=dem_tiles_data,
                    tiles_x=tiles_x,
                    tiles_y=tiles_y,
                    eff_tile_px=full_eff_tile_px,
                    crop_rect=crop_rect,
                )
                del dem_tiles_data  # Освобождаем память
                gc.collect()

                # Проверяем, нужен ли даунсэмплинг DEM
                dem_h_orig, dem_w_orig = dem_full.shape
                ds_factor = compute_downsample_factor(dem_h_orig, dem_w_orig)

                if ds_factor > 1:
                    logger.info(
                        'Радиогоризонт: DEM слишком большой (%dx%d = %d Mpx), '
                        'даунсэмплинг в %d раз',
                        dem_w_orig,
                        dem_h_orig,
                        dem_w_orig * dem_h_orig // 1_000_000,
                        ds_factor,
                    )
                    dem_full = downsample_dem(dem_full, ds_factor)
                    gc.collect()

                # Вычисляем позицию антенны в пикселях DEM
                # Сначала конвертируем координаты контрольной точки из ГК в географические СК-42
                cp_lng_sk42, cp_lat_sk42 = t_sk42_from_gk.transform(
                    settings.control_point_x_sk42_gk,
                    settings.control_point_y_sk42_gk,
                )
                # Затем из СК-42 в WGS84
                control_lng_wgs, control_lat_wgs = t_sk42_to_wgs.transform(
                    cp_lng_sk42, cp_lat_sk42
                )

                # Конвертируем в пиксели относительно crop_rect
                ant_px_x, ant_px_y = latlng_to_pixel_xy(
                    control_lat_wgs, control_lng_wgs, zoom
                )
                # Преобразуем в локальные координаты crop (с учётом даунсэмплинга)
                cx, cy, cw, ch = crop_rect
                # Начало тайловой сетки в глобальных пикселях
                first_tile_x, first_tile_y = tiles[0]
                global_origin_x = first_tile_x * full_eff_tile_px
                global_origin_y = first_tile_y * full_eff_tile_px

                # Координаты антенны в оригинальном масштабе
                antenna_col_orig = int(
                    ant_px_x * (full_eff_tile_px / 256) - global_origin_x - cx
                )
                antenna_row_orig = int(
                    ant_px_y * (full_eff_tile_px / 256) - global_origin_y - cy
                )

                # Применяем коэффициент даунсэмплинга
                antenna_col = antenna_col_orig // ds_factor
                antenna_row = antenna_row_orig // ds_factor

                # Проверяем границы
                dem_h, dem_w = dem_full.shape
                antenna_row = max(0, min(antenna_row, dem_h - 1))
                antenna_col = max(0, min(antenna_col, dem_w - 1))

                logger.info(
                    'Радиогоризонт: DEM размер %dx%d, антенна в пикселях (%d, %d)',
                    dem_w,
                    dem_h,
                    antenna_col,
                    antenna_row,
                )

                # Вычисляем размер пикселя в метрах (с учётом даунсэмплинга)
                pixel_size_m = (
                    meters_per_pixel(
                        center_lat_wgs, zoom, scale=full_eff_tile_px // 256
                    )
                    * ds_factor
                )

                # Вычисляем и раскрашиваем радиогоризонт за один проход
                sp_rh = LiveSpinner('Расчёт радиогоризонта')
                sp_rh.start()

                result = compute_and_colorize_radio_horizon(
                    dem=dem_full,
                    antenna_row=antenna_row,
                    antenna_col=antenna_col,
                    antenna_height_m=settings.antenna_height_m,
                    pixel_size_m=pixel_size_m,
                    max_height_m=settings.max_flight_height_m,
                )

                sp_rh.stop('Радиогоризонт рассчитан')
                del dem_full  # Освобождаем память
                gc.collect()  # Принудительная сборка мусора

                # Масштабируем результат до целевого размера (если был даунсэмплинг)
                target_size = (cw, ch)
                if result.size != target_size:
                    logger.info(
                        'Радиогоризонт: масштабирование %s -> %s',
                        result.size,
                        target_size,
                    )
                    result = result.resize(target_size, Image.Resampling.BILINEAR)

                # Накладываем цветовую карту радиогоризонта на топографическую основу
                logger.info('Загрузка топографической основы для радиогоризонта')
                sp_topo = LiveSpinner('Загрузка топографической основы')
                sp_topo.start()

                topo_style_id = MAPBOX_STYLE_BY_TYPE[MapType.OUTDOORS]

                # Используем те же настройки тайлов, что и для DEM (RADIO_HORIZON_USE_RETINA)
                # чтобы crop_rect соответствовал и не было огромного потребления памяти
                topo_tile_size = TILE_SIZE
                topo_use_retina = RADIO_HORIZON_USE_RETINA

                async def fetch_topo_tile(
                    idx_xy: tuple[int, tuple[int, int]],
                ) -> tuple[int, Image.Image]:
                    idx, (tx, ty) = idx_xy
                    async with semaphore:
                        img = await async_fetch_xyz_tile(
                            client=client,
                            api_key=api_key,
                            style_id=topo_style_id,
                            tile_size=topo_tile_size,
                            z=zoom,
                            x=tx,
                            y=ty,
                            use_retina=topo_use_retina,
                        )
                        return idx, img

                topo_tasks = [fetch_topo_tile(pair) for pair in enumerate(tiles)]
                topo_results = await asyncio.gather(*topo_tasks)
                topo_results.sort(key=lambda t: t[0])
                topo_images: list[Image.Image] = [img for _, img in topo_results]
                eff_tile_px_topo = topo_tile_size * (2 if topo_use_retina else 1)
                topo_base = assemble_and_crop(
                    images=topo_images,
                    tiles_x=tiles_x,
                    tiles_y=tiles_y,
                    eff_tile_px=eff_tile_px_topo,
                    crop_rect=crop_rect,
                )
                with contextlib.suppress(Exception):
                    topo_images.clear()

                sp_topo.stop('Топографическая основа загружена')

                # Масштабируем топооснову до размера результата если нужно
                if topo_base.size != result.size:
                    topo_base = topo_base.resize(result.size, Image.Resampling.BILINEAR)

                # Накладываем цветовую карту радиогоризонта на топооснову с прозрачностью
                logger.info(
                    'Наложение радиогоризонта на топооснову (alpha=%.2f)',
                    RADIO_HORIZON_TOPO_OVERLAY_ALPHA,
                )
                # Конвертируем топооснову в оттенки серого, затем в RGBA для смешивания
                topo_base = topo_base.convert('L').convert('RGBA')
                result = result.convert('RGBA')
                result = Image.blend(
                    topo_base, result, RADIO_HORIZON_TOPO_OVERLAY_ALPHA
                )
                del topo_base
                gc.collect()

                logger.info('Карта радиогоризонта построена')

            else:

                async def bound_fetch(
                    idx_xy: tuple[int, tuple[int, int]],
                ) -> tuple[int, Image.Image]:
                    nonlocal tile_count
                    idx, (tx, ty) = idx_xy
                    async with semaphore:
                        img = await async_fetch_xyz_tile(
                            client=client,
                            api_key=api_key,
                            style_id=style_id,  # type: ignore[arg-type]
                            tile_size=XYZ_TILE_SIZE,
                            z=zoom,
                            x=tx,
                            y=ty,
                            use_retina=XYZ_USE_RETINA,
                        )
                        await tile_progress.step(1)
                        tile_count += 1
                        if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                            log_memory_usage(f'after {tile_count} tiles')
                        return idx, img

                tasks = [bound_fetch(pair) for pair in enumerate(tiles)]
                results = await asyncio.gather(*tasks)
                tile_progress.close()
                log_memory_usage('after all tiles downloaded')
                log_thread_status('after all tiles downloaded')
                results.sort(key=lambda t: t[0])
                images: list[Image.Image] = [img for _, img in results]
                eff_tile_px = XYZ_TILE_SIZE * (2 if XYZ_USE_RETINA else 1)
                result = assemble_and_crop(
                    images=images,
                    tiles_x=tiles_x,
                    tiles_y=tiles_y,
                    eff_tile_px=eff_tile_px,
                    crop_rect=crop_rect,
                )
                with contextlib.suppress(Exception):
                    images.clear()
    finally:
        # Explicit cleanup of HTTP session resources
        try:
            if hasattr(session_ctx, '_cache') and session_ctx._cache:
                # Close SQLite cache backend if it exists
                if hasattr(session_ctx._cache, '_cache') and hasattr(
                    session_ctx._cache._cache,
                    'close',
                ):
                    await session_ctx._cache._cache.close()
        except Exception:
            logging.getLogger(__name__).debug('Error during HTTP session cleanup')

        if cache_dir_resolved:
            _cleanup_sqlite_cache(cache_dir_resolved)

    # Перед поворотом: при необходимости наложим изолинии поверх основы
    logger.info(
        'Проверка overlay_contours: overlay_contours=%s, is_elev_contours=%s',
        overlay_contours,
        is_elev_contours,
    )
    if overlay_contours and not is_elev_contours:
        logger.info('=== НАЧАЛО ПОСТРОЕНИЯ OVERLAY ИЗОЛИНИЙ ===')
        try:
            # Быстрая проверка доступности Terrain-RGB перед началом фазы оверлея
            logger.info('Изолинии: проверка доступности Terrain-RGB API')
            await _validate_terrain_api(api_key)
            logger.info('Изолинии: Terrain-RGB API доступен')

            # Строим Terrain-RGB оверлей изолиний независимо, затем масштабируем к размеру основы
            eff_scale_cont = effective_scale_for_xyz(256, use_retina=contour_use_retina)
            base_pad = round(min(target_w_px, target_h_px) * ROTATION_PAD_RATIO)
            pad_px_cont = max(base_pad, ROTATION_PAD_MIN_PX)
            tiles_c, (tiles_x_c, tiles_y_c), crop_rect_c, _ = compute_xyz_coverage(
                center_lat=center_lat_wgs,
                center_lng=center_lng_wgs,
                width_m=width_m,
                height_m=height_m,
                zoom=zoom,
                eff_scale=eff_scale_cont,
                pad_px=pad_px_cont,
            )

            # Создаём собственную HTTP-сессию для Terrain-RGB (одну на фазу) и ограничитель параллелизма
            cache_dir_resolved2 = _resolve_cache_dir()
            session_ctx2 = _make_http_session(cache_dir_resolved2)

            full_eff_tile_px = 256 * (2 if contour_use_retina else 1)

            # Pass A: gather samples and build low-res seed
            max_samples = 50000
            samples_overlay: list[float] = []
            seen = 0

            rng = random.Random(42)  # noqa: S311

            seed_ds = max(2, int(CONTOUR_SEED_DOWNSAMPLE))
            cx_c, cy_c, cw_c, ch_c = crop_rect_c
            seed_w = max(1, (cw_c + seed_ds - 1) // seed_ds)
            seed_h = max(1, (ch_c + seed_ds - 1) // seed_ds)
            seed_sum_c: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
            seed_cnt_c: list[list[int]] = [[0] * seed_w for _ in range(seed_h)]

            # Ограничитель параллелизма для overlay-запросов
            overlay_semaphore = asyncio.Semaphore(
                DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY
            )

            async def fetch_and_sample_overlay(
                idx_xy: tuple[int, tuple[int, int]], client2: aiohttp.ClientSession
            ) -> None:
                nonlocal seen, samples_overlay
                try:
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = tx_ty_from_index(idx, tiles_x_c)
                    if (
                        _tile_overlap_rect_common(tx, ty, crop_rect_c, full_eff_tile_px)
                        is None
                    ):
                        return
                    async with overlay_semaphore:
                        img = await provider2.get_tile_image(
                            zoom, tile_x_world, tile_y_world
                        )
                    dem_tile = decode_terrain_rgb_to_elevation_m(img)
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
                                seen += 1
                                if len(samples_overlay) < max_samples:
                                    samples_overlay.append(v)
                                else:
                                    j = rng.randrange(0, seen)
                                    if j < max_samples:
                                        samples_overlay[j] = v
                        ov = _tile_overlap_rect_common(
                            tx, ty, crop_rect_c, full_eff_tile_px
                        )
                        if ov is not None:
                            x0, y0, x1, y1 = ov
                            base_x = tx * full_eff_tile_px
                            base_y = ty * full_eff_tile_px
                            for yy in range(y0, y1, seed_ds):
                                sy = (yy - cy_c) // seed_ds
                                if sy < 0 or sy >= seed_h:
                                    continue
                                src_y = yy - base_y
                                row_src = dem_tile[src_y]
                                for xx in range(x0, x1, seed_ds):
                                    sx = (xx - cx_c) // seed_ds
                                    if sx < 0 or sx >= seed_w:
                                        continue
                                    src_x = xx - base_x
                                    v = row_src[src_x]
                                    seed_sum_c[sy][sx] += v
                                    seed_cnt_c[sy][sx] += 1
                except Exception as ex:
                    # Ignore individual tile errors during overlay pass A
                    logger.warning('Overlay tile failed: %s', ex)
                finally:
                    with contextlib.suppress(Exception):
                        await overlay_progress_a.step(1)

            # Прогресс-индикатор для overlay-прохода A
            logger.info('Изолинии: старт прохода 1/2')
            overlay_progress_a = ConsoleProgress(
                total=len(tiles_c), label='Изолинии: загрузка Terrain-RGB (проход 1/2)'
            )
            async with session_ctx2 as client2:
                provider2 = ElevationTileProvider(
                    client=client2, api_key=api_key, use_retina=contour_use_retina
                )
                await asyncio.gather(
                    *[
                        fetch_and_sample_overlay(pair, client2)
                        for pair in enumerate(tiles_c)
                    ],
                    return_exceptions=True,
                )
            overlay_progress_a.close()
            # Логи по завершении прохода A
            try:
                tiles_len = len(tiles_c)
                samples_len = len(samples_overlay)
                logger.info(
                    'Изолинии: завершён проход 1/2: tiles=%d, samples=%d',
                    tiles_len,
                    samples_len,
                )
            except Exception:
                logger.info('Изолинии: завершён проход 1/2')

            logger.info('Изолинии: вычисление диапазона высот для overlay')
            if samples_overlay:
                mn = min(samples_overlay)
                mx = max(samples_overlay)
                logger.info(
                    'Изолинии overlay: диапазон высот min=%.2f, max=%.2f', mn, mx
                )
            else:
                mn, mx = 0.0, 1.0
                logger.warning(
                    'Изолинии overlay: нет данных высот, используются значения по умолчанию'
                )
            if mx < mn:
                mn, mx = mx, mn
                logger.warning('Изолинии overlay: min > max, значения переставлены')

            seed_dem_c: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
            for sy in range(seed_h):
                row_s = seed_sum_c[sy]
                row_c = seed_cnt_c[sy]
                out = seed_dem_c[sy]
                for sx in range(seed_w):
                    c = row_c[sx]
                    out[sx] = (row_s[sx] / float(c)) if c > 0 else (mn + mx) * 0.5

            interval = float(CONTOUR_INTERVAL_M) if CONTOUR_INTERVAL_M > 0 else 25.0
            start = math.floor(mn / interval) * interval
            end = math.ceil(mx / interval) * interval
            levels_c: list[float] = []
            k = 0
            v = start
            while v <= end:
                levels_c.append(v)
                k += 1
                v = start + k * interval

            logger.info(
                'Изолинии: построение seed polylines для %d уровней', len(levels_c)
            )
            seed_polylines = build_seed_polylines(seed_dem_c, levels_c, seed_h, seed_w)
            total_polylines = sum(len(polys) for polys in seed_polylines.values())
            logger.info('Изолинии: создано %d полилиний', total_polylines)

            # Draw overlay RGBA with transparent background
            logger.info(
                'Изолинии: создание overlay изображения размером %dx%d',
                crop_rect_c[2],
                crop_rect_c[3],
            )
            overlay = Image.new('RGBA', (crop_rect_c[2], crop_rect_c[3]), (0, 0, 0, 0))
            overlay_progress_b = ConsoleProgress(
                total=len(levels_c),
                label='Изолинии: построение и рисование (проход 2/2)',
            )
            logger.info('Изолинии: старт прохода 2/2: levels=%d', len(levels_c))
            
            # Сглаживание отключено для оверлея ради производительности и согласованности
            smoothed_polylines = seed_polylines
            
            # Создаём ImageDraw один раз перед циклами
            draw = ImageDraw.Draw(overlay)
            
            for li, _level in enumerate(levels_c):
                is_index = (li % max(1, int(CONTOUR_INDEX_EVERY))) == 0
                color = CONTOUR_INDEX_COLOR if is_index else CONTOUR_COLOR
                width = int(CONTOUR_INDEX_WIDTH if is_index else CONTOUR_WIDTH)
                fill_color = (*list(color), 255)
                
                for poly in smoothed_polylines.get(li, []):
                    # draw full polylines; block optimization skipped for simplicity
                    pts_crop = [
                        (int(p[0] * seed_ds), int(p[1] * seed_ds))
                        for p in poly
                    ]
                    if len(pts_crop) < 2:  # noqa: PLR2004
                        continue
                    for i in range(1, len(pts_crop)):
                        x0, y0 = pts_crop[i - 1]
                        x1, y1 = pts_crop[i]
                        draw.line((x0, y0, x1, y1), fill=fill_color, width=width)
                        
                # step progress per level
                with contextlib.suppress(Exception):
                    overlay_progress_b.step_sync(1)

                # periodic diagnostics every 5 levels
                if (li + 1) % 5 == 0 or (li + 1) == len(levels_c):
                    log_memory_usage(
                        f'Contours pass 2/2 after level {li + 1}/{len(levels_c)}'
                    )
                    log_thread_status(
                        f'Contours pass 2/2 after level {li + 1}/{len(levels_c)}'
                    )

            logger.info('Изолинии: завершён проход 2/2')

            labels_start_time = time.monotonic()
            logger.info('Изолинии: подписи (оверлей) — старт')
            log_memory_usage('before overlay labels')

            if CONTOUR_SEED_DOWNSAMPLE:
                try:
                    # прогресс для подписей можно отразить тем же лейблом
                    lat_rad = math.radians(center_lat_wgs)
                    # Учитываем retina-фактор overlay для корректного mpp
                    overlay_retina_factor = 2 if contour_use_retina else 1
                    mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
                        TILE_SIZE * overlay_retina_factor * (2**zoom)
                    )

                    # Reuse overlay label drawer from dedicated module
                    def _draw_contour_labels_overlay(
                        img: Image.Image,
                        seed_polylines: dict[int, list[list[tuple[float, float]]]],
                        levels: list[float],
                        seed_ds: int,
                        mpp: float,
                        *,
                        dry_run: bool = False,
                    ) -> list[tuple[int, int, int, int]]:
                        from contours.labels_overlay import (
                            draw_contour_labels_overlay as _impl,
                        )

                        return _impl(
                            img,
                            seed_polylines,
                            levels,
                            mpp,
                            seed_ds=seed_ds,
                            dry_run=dry_run,
                        )

                    # Первый проход: получаем позиции подписей БЕЗ размещения на изображении
                    overlay_label_bboxes = _draw_contour_labels_overlay(
                        overlay, seed_polylines, levels_c, seed_ds, mpp, dry_run=True
                    )

                    # Создаем разрывы линий контуров в местах подписей для overlay

                    if CONTOUR_LABEL_GAP_ENABLED and overlay_label_bboxes:
                        draw_overlay = ImageDraw.Draw(overlay)
                        gap_padding = max(
                            1,
                            round(CONTOUR_LABEL_GAP_PADDING_M / max(1e-9, mpp)),
                        )
                        for bbox in overlay_label_bboxes:
                            x0, y0, x1, y1 = bbox
                            # Расширяем область на gap_padding
                            gap_area = (
                                max(0, x0 - gap_padding),
                                max(0, y0 - gap_padding),
                                min(overlay.width, x1 + gap_padding),
                                min(overlay.height, y1 + gap_padding),
                            )
                            # "Стираем" контуры в этой области (делаем прозрачным для RGBA)
                            draw_overlay.rectangle(gap_area, fill=(0, 0, 0, 0))

                    # Второй проход: размещаем подписи поверх созданных разрывов
                    _draw_contour_labels_overlay(
                        overlay, seed_polylines, levels_c, seed_ds, mpp, dry_run=False
                    )
                except Exception as e:
                    logger.warning(
                        'Не удалось нанести подписи изолиний (оверлей): %s',
                        e,
                        exc_info=True,
                    )

            labels_elapsed = time.monotonic() - labels_start_time
            logger.info(
                'Изолинии: подписи (оверлей) — завершены (%.2fs)', labels_elapsed
            )
            log_memory_usage('after overlay labels')

            # Composite overlay onto base (pre-rotation)
            composite_start_time = time.monotonic()
            logger.info('Изолинии: компоновка overlay на базу — старт')
            try:
                logger.info(
                    'Изолинии: размер overlay=%s, размер базы=%s',
                    overlay.size,
                    (crop_rect[2], crop_rect[3]),
                )
                if overlay.size != (crop_rect[2], crop_rect[3]):
                    logger.info('Изолинии: масштабирование overlay до размера базы')
                    overlay = overlay.resize(
                        (crop_rect[2], crop_rect[3]), Image.Resampling.BICUBIC
                    )
                # Convert base to RGBA, composite overlay, then convert back to RGB
                prev_result = result
                logger.info('Изолинии: конвертация базы в RGBA')
                base_rgba = prev_result.convert('RGBA')
                logger.info('Изолинии: применение alpha_composite')
                base_rgba.alpha_composite(overlay)
                logger.info('Изолинии: конвертация результата обратно в RGB')
                result = base_rgba.convert('RGB')
                # Explicitly close large temporary images to free memory
                with contextlib.suppress(Exception):
                    overlay.close()
                with contextlib.suppress(Exception):
                    base_rgba.close()
                with contextlib.suppress(Exception):
                    prev_result.close()
                logger.info('Изолинии: наложение успешно завершено')
            except Exception as e:
                logger.error('Не удалось наложить изолинии: %s', e, exc_info=True)
            composite_elapsed = time.monotonic() - composite_start_time
            logger.info(
                'Изолинии: компоновка overlay на базу — завершена (%.2fs)',
                composite_elapsed,
            )
            log_memory_usage('after overlay composite')
            logger.info('=== ЗАВЕРШЕНИЕ ПОСТРОЕНИЯ OVERLAY ИЗОЛИНИЙ ===')
        except Exception as e:
            logger.error('Построение оверлея изолиний не удалось: %s', e, exc_info=True)

    # Image rotation
    rotation_start_time = time.monotonic()
    logger.info('Поворот изображения — старт')
    rotation_deg = compute_rotation_deg_for_east_axis(
        center_lat_sk42=center_lat_sk42,
        center_lng_sk42=center_lng_sk42,
        map_params=map_params,
        crs_sk42_gk=crs_sk42_gk,
        t_sk42_to_wgs=t_sk42_to_wgs,
    )
    prev_result = result
    result = rotate_keep_size(prev_result, rotation_deg, fill=(255, 255, 255))
    with contextlib.suppress(Exception):
        prev_result.close()
    rotation_elapsed = time.monotonic() - rotation_start_time
    logger.info('Поворот изображения — завершён (%.2fs)', rotation_elapsed)
    log_memory_usage('after rotation')

    # Cropping
    crop_start_time = time.monotonic()
    logger.info('Обрезка к целевому размеру — старт')
    prev_result = result
    result = center_crop(prev_result, target_w_px, target_h_px)
    with contextlib.suppress(Exception):
        prev_result.close()
    crop_elapsed = time.monotonic() - crop_start_time
    logger.info('Обрезка к целевому размеру — завершена (%.2fs)', crop_elapsed)
    log_memory_usage('after cropping')

    # Grid drawing first (legend will be drawn on top)
    grid_start_time = time.monotonic()
    logger.info('Рисование км-сетки — старт')
    draw_axis_aligned_km_grid(
        img=result,
        center_lat_sk42=center_lat_sk42,
        center_lng_sk42=center_lng_sk42,
        center_lat_wgs=center_lat_wgs,
        center_lng_wgs=center_lng_wgs,
        zoom=zoom,
        crs_sk42_gk=crs_sk42_gk,
        t_sk42_to_wgs=t_sk42_to_wgs,
        step_m=GRID_STEP_M,
        color=GRID_COLOR,
        width_m=settings.grid_width_m,
        scale=eff_scale,
        grid_font_size_m=settings.grid_font_size_m,
        grid_text_margin_m=settings.grid_text_margin_m,
        grid_label_bg_padding_m=settings.grid_label_bg_padding_m,
        legend_bounds=None,
        display_grid=settings.display_grid,
        rotation_deg=rotation_deg,
    )
    grid_elapsed = time.monotonic() - grid_start_time
    logger.info('Рисование км-сетки — завершено (%.2fs)', grid_elapsed)
    log_memory_usage('after grid drawing')
    log_thread_status('after grid drawing')

    # Draw elevation legend on top of grid (if needed)
    if is_elev_color and elev_min_m is not None and elev_max_m is not None:
        legend_start_time = time.monotonic()
        logger.info('Рисование легенды высот — старт')
        try:
            draw_elevation_legend(
                img=result,
                color_ramp=ELEVATION_COLOR_RAMP,
                min_elevation_m=elev_min_m,
                max_elevation_m=elev_max_m,
                center_lat_wgs=center_lat_wgs,
                zoom=zoom,
                scale=eff_scale,
                title='Высота',
                label_step_m=ELEVATION_LEGEND_STEP_M,
            )
            legend_elapsed = time.monotonic() - legend_start_time
            logger.info('Рисование легенды высот — завершено (%.2fs)', legend_elapsed)
        except Exception as e:
            logger.warning('Не удалось нарисовать легенду высот: %s', e)

    # Draw radio horizon legend on top of grid (if needed)
    elif is_radio_horizon:
        legend_start_time = time.monotonic()
        logger.info('Рисование легенды радиогоризонта — старт')
        try:
            draw_elevation_legend(
                img=result,
                color_ramp=RADIO_HORIZON_COLOR_RAMP,
                min_elevation_m=0.0,
                max_elevation_m=settings.max_flight_height_m,
                center_lat_wgs=center_lat_wgs,
                zoom=zoom,
                scale=eff_scale,
                title='Минимальная высота БпЛА',
                label_step_m=ELEVATION_LEGEND_STEP_M,
            )
            legend_elapsed = time.monotonic() - legend_start_time
            logger.info(
                'Рисование легенды радиогоризонта — завершено (%.2fs)', legend_elapsed
            )
        except Exception as e:
            logger.warning('Не удалось нарисовать легенду радиогоризонта: %s', e)

    # Draw center cross and log its coordinates (military notation)
    # PROJ/pyproj uses (easting, northing) with always_xy=True. Military notation wants X=northing, Y=easting.
    try:
        # Compute center coordinates in SK-42 GK using same mechanics as grid (result: x0_gk=easting, y0_gk=northing)
        t_sk42gk_from_sk42 = Transformer.from_crs(
            crs_sk42_geog,
            crs_sk42_gk,
            always_xy=True,
        )
        x0_gk, y0_gk = t_sk42gk_from_sk42.transform(center_lng_sk42, center_lat_sk42)
        # Compute center coordinates in WGS84 using Helmert-aware transformer
        center_lng_wgs, center_lat_wgs = t_sk42_to_wgs.transform(
            center_lng_sk42, center_lat_sk42
        )
        logger.info(
            'Центральный крест: СК-42 ГК X(север)=%.3f, Y(восток)=%.3f; WGS84 lat=%.8f, lon=%.8f',
            y0_gk,
            x0_gk,
            center_lat_wgs,
            center_lng_wgs,
        )

        # Draw the cross at the image center
        mpp_center = meters_per_pixel(center_lat_wgs, zoom, scale=eff_scale)
        ppm_center = 1.0 / mpp_center if mpp_center > 0 else 0.0
        cx = result.width // 2
        cy = result.height // 2
        length_px = max(1, round(CENTER_CROSS_LENGTH_M * ppm_center))
        half = max(1, length_px // 2)
        line_w = max(1, round(CENTER_CROSS_LINE_WIDTH_M * ppm_center))
        draw = ImageDraw.Draw(result)
        color = tuple(CENTER_CROSS_COLOR)
        draw.line([(cx, cy - half), (cx, cy + half)], fill=color, width=line_w)
        draw.line([(cx - half, cy), (cx + half, cy)], fill=color, width=line_w)
    except Exception as e:
        logger.warning(
            'Не удалось нарисовать центрированный крест или вывести координаты: %s', e
        )

    # Draw control point as red triangle (unified style for all map types)
    try:
        cp_enabled = getattr(settings, 'control_point_enabled', False)
        logger.info(
            'CP DEBUG ENTRY: map_type=%s, control_point_enabled=%s, is_radio_horizon=%s, '
            'result.size=%s, result.mode=%s, eff_scale=%s, zoom=%s',
            getattr(settings, 'map_type', 'UNKNOWN'),
            cp_enabled,
            is_radio_horizon,
            result.size,
            result.mode,
            eff_scale,
            zoom,
        )
        if cp_enabled:
            logger.info('Отрисовка контрольной точки включена')
            # Get control point coordinates in SK-42 GK (easting, northing)
            cp_x_gk = float(settings.control_point_x_sk42_gk)
            cp_y_gk = float(settings.control_point_y_sk42_gk)

            # Pixels-per-meter at center latitude in WGS84
            mpp = meters_per_pixel(center_lat_wgs, zoom, scale=eff_scale)
            ppm = 1.0 / mpp if mpp > 0 else 0.0

            # Преобразуем координаты контрольной точки через полную цепочку:
            # СК-42 ГК → СК-42 географические → WGS-84 → Web Mercator пиксели
            # Используем существующий трансформер t_sk42_from_gk (определён в начале функции)
            cp_lng_sk42, cp_lat_sk42 = t_sk42_from_gk.transform(cp_x_gk, cp_y_gk)
            cp_lng_wgs, cp_lat_wgs = t_sk42_to_wgs.transform(cp_lng_sk42, cp_lat_sk42)

            # Детальное логирование для диагностики
            logger.info(
                'CP DEBUG: input control_point_x=%d, control_point_y=%d',
                settings.control_point_x,
                settings.control_point_y,
            )
            logger.info(
                'CP DEBUG: cp_x_gk(easting)=%.3f, cp_y_gk(northing)=%.3f',
                cp_x_gk,
                cp_y_gk,
            )
            logger.info(
                'CP DEBUG: cp_lng_sk42=%.8f, cp_lat_sk42=%.8f',
                cp_lng_sk42,
                cp_lat_sk42,
            )
            logger.info(
                'CP DEBUG: center_lng_wgs=%.8f, center_lat_wgs=%.8f',
                center_lng_wgs,
                center_lat_wgs,
            )

            # Вычисляем "мировые" пиксельные координаты центра и контрольной точки
            cx_world, cy_world = latlng_to_pixel_xy(
                center_lat_wgs, center_lng_wgs, zoom
            )
            cp_x_world, cp_y_world = latlng_to_pixel_xy(cp_lat_wgs, cp_lng_wgs, zoom)

            logger.info(
                'CP DEBUG: cx_world=%.3f, cy_world=%.3f, cp_x_world=%.3f, cp_y_world=%.3f',
                cx_world,
                cy_world,
                cp_x_world,
                cp_y_world,
            )

            # Преобразуем в пиксели изображения (относительно центра, до поворота)
            img_cx = result.width / 2.0
            img_cy = result.height / 2.0
            x_pre = img_cx + (cp_x_world - cx_world) * eff_scale
            y_pre = img_cy + (cp_y_world - cy_world) * eff_scale

            logger.info(
                'CP DEBUG: img_cx=%.1f, img_cy=%.1f, x_pre=%.1f, y_pre=%.1f, eff_scale=%d',
                img_cx,
                img_cy,
                x_pre,
                y_pre,
                eff_scale,
            )

            # Применяем поворот вокруг центра изображения (как в gk_to_pixel)
            rotation_rad = math.radians(-rotation_deg)
            cos_rot = math.cos(rotation_rad)
            sin_rot = math.sin(rotation_rad)
            dx = x_pre - img_cx
            dy = y_pre - img_cy
            cx_img = img_cx + dx * cos_rot - dy * sin_rot
            cy_img = img_cy + dx * sin_rot + dy * cos_rot

            logger.info(
                'CP DEBUG: rotation_deg=%.3f, dx=%.1f, dy=%.1f, cx_img=%.1f, cy_img=%.1f',
                rotation_deg,
                dx,
                dy,
                cx_img,
                cy_img,
            )
            # Показываем координаты как процент от размера изображения для диагностики
            logger.info(
                'CP DEBUG BOUNDS: cx_img=%.1f (%.1f%% of width=%d), cy_img=%.1f (%.1f%% of height=%d)',
                cx_img,
                100.0 * cx_img / result.width,
                result.width,
                cy_img,
                100.0 * cy_img / result.height,
                result.height,
            )
            logger.info(
                'Контрольная точка: СК-42 ГК X(север)=%.3f, Y(восток)=%.3f; WGS84 lat=%.8f, lon=%.8f',
                cp_y_gk,
                cp_x_gk,
                cp_lat_wgs,
                cp_lng_wgs,
            )

            # Draw control point if inside image bounds
            if 0 <= cx_img <= result.width and 0 <= cy_img <= result.height:
                draw = ImageDraw.Draw(result)
                cx_i = round(cx_img)
                cy_i = round(cy_img)

                # Единый стиль контрольной точки для всех типов карт: красный треугольник
                # Размер треугольника в пикселях (из метров на местности)
                triangle_size_px = max(10, round(CONTROL_POINT_SIZE_M * ppm))
                # Высота равнобедренного треугольника (вершина вверху)
                h_tri = triangle_size_px
                # Ширина основания = высоте для равнобедренного треугольника
                half_base = triangle_size_px // 2
                # Вершины треугольника: вершина вверху, основание внизу
                p1 = (cx_i, cy_i - h_tri // 2)  # верхняя вершина
                p2 = (cx_i - half_base, cy_i + h_tri // 2)  # левый нижний угол
                p3 = (cx_i + half_base, cy_i + h_tri // 2)  # правый нижний угол
                color = tuple(CONTROL_POINT_COLOR)
                draw.polygon([p1, p2, p3], fill=color, outline=(0, 0, 0))
                marker_bottom_y = cy_i + h_tri // 2

                # Рисуем название контрольной точки с высотой антенны
                cp_name = getattr(settings, 'control_point_name', '')
                antenna_h = round(getattr(settings, 'antenna_height_m', 10.0))

                # Формируем текст: "<Название> (hант = <высота>)" или "(hант = <высота>)" если имя пустое
                # Размер шрифта такой же как у подписей сетки (в метрах → пиксели)
                label_font_size = max(10, round(settings.grid_font_size_m * ppm))
                subscript_font_size = max(8, round(label_font_size * 0.65))
                try:
                    label_font = load_grid_font(label_font_size)
                    subscript_font = load_grid_font(subscript_font_size)
                except Exception:
                    from PIL import ImageFont

                    label_font = ImageFont.load_default()
                    subscript_font = label_font

                # Отступ текста от маркера = настройка "Отступ текста" из параметров сетки
                text_margin_px = max(5, round(settings.grid_text_margin_m * ppm))
                # Позиция текста: под маркером с отступом
                label_x = cx_i
                label_y = marker_bottom_y + text_margin_px

                # Внутренний отступ подложки (как у подписей сетки)
                bg_padding_px = max(2, round(settings.grid_label_bg_padding_m * ppm))

                # Рисуем название и высоту антенны на отдельных строках
                # Формат: "<Название>" на первой строке, "(hант = <высота> м)" на второй
                current_y = label_y

                # Первая строка: название точки (если есть)
                if cp_name:
                    draw_label_with_bg(
                        draw,
                        (label_x, current_y),
                        cp_name,
                        font=label_font,
                        anchor='mt',  # middle-top
                        img_size=result.size,
                        padding=bg_padding_px,
                    )
                    # Вычисляем высоту первой строки для отступа
                    name_bbox = draw.textbbox(
                        (0, 0), cp_name, font=label_font, anchor='lt'
                    )
                    name_height = name_bbox[3] - name_bbox[1]
                    # Отступ между строками = высота текста + padding
                    current_y += name_height + bg_padding_px * 2

                # Вторая строка: высота антенны с подстрочным индексом
                height_parts = [
                    ('(h', False),
                    ('ант', True),  # подстрочный индекс
                    (f' = {int(antenna_h)} м)', False),
                ]

                draw_label_with_subscript_bg(
                    draw,
                    (label_x, current_y),
                    height_parts,
                    font=label_font,
                    subscript_font=subscript_font,
                    anchor='mt',  # middle-top
                    img_size=result.size,
                    padding=bg_padding_px,
                )
            else:
                logger.warning(
                    'Контрольная точка вне кадра: (%.2f, %.2f) not in [0..%d]x[0..%d]',
                    cx_img,
                    cy_img,
                    result.width,
                    result.height,
                )
    except Exception as e:
        logger.warning('Не удалось нарисовать контрольную точку: %s', e)

    # Preview publishing
    preview_start_time = time.monotonic()
    logger.info('Публикация предпросмотра — старт')
    did_publish = False
    try:
        # Send a copy to GUI to avoid holding the processing buffer; GUI will own the copy
        gui_image = None
        try:
            gui_image = result.copy()
        except Exception:
            gui_image = None
        if gui_image is not None:
            did_publish = publish_preview_image(gui_image)
        else:
            did_publish = publish_preview_image(result)
    except Exception:
        did_publish = False
    preview_elapsed = time.monotonic() - preview_start_time
    logger.info(
        'Публикация предпросмотра — %s (%.2fs)',
        'успех' if did_publish else 'пропущено',
        preview_elapsed,
    )

    # Если GUI подписан на предпросмотр (did_publish=True), не сохраняем автоматически.
    if not did_publish:
        save_start_time = time.monotonic()
        logger.info('Сохранение файла — старт')
        sp = LiveSpinner('Сохранение файла')
        sp.start()

        out_path = Path(output_path)
        # Принудительно используем расширение .jpg
        if out_path.suffix.lower() not in ('.jpg', '.jpeg'):
            out_path = out_path.with_suffix('.jpg')
        out_path.resolve().parent.mkdir(parents=True, exist_ok=True)
        # Use default quality (95%) for automatic saves in CLI/preview mode
        save_kwargs = _build_save_kwargs(out_path, quality=95)

        _save_jpeg(result, out_path, save_kwargs)

        sp.stop('Сохранение файла: готово')
        save_elapsed = time.monotonic() - save_start_time
        logger.info('Сохранение файла — завершено (%.2fs)', save_elapsed)
        log_memory_usage('after file save')
        # Close the main result image after saving to release memory
        with contextlib.suppress(Exception):
            result.close()
    else:
        # GUI consumed a copy; close original processing buffer to free memory
        with contextlib.suppress(Exception):
            result.close()

    # Garbage collection
    gc_start_time = time.monotonic()
    logger.info('Сборка мусора — старт')
    try:
        gc.collect()
        gc_elapsed = time.monotonic() - gc_start_time
        logger.info('Сборка мусора — завершена (%.2fs)', gc_elapsed)
        log_memory_usage('after garbage collection')
        log_thread_status('final cleanup')
    except Exception as e:
        logger.debug(f'Garbage collection failed: {e}')

    overall_elapsed = time.monotonic() - overall_start_time
    logger.info(
        '=== ОБЩИЙ ТАЙМЕР: завершён download_satellite_rectangle (%.2fs) ===',
        overall_elapsed,
    )
    return output_path

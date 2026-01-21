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
from typing import cast

import aiohttp
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

from constants import (
    ASYNC_MAX_CONCURRENCY,
    CENTER_CROSS_COLOR,
    CENTER_CROSS_LENGTH_PX,
    CENTER_CROSS_LINE_WIDTH_PX,
    CONTOUR_BLOCK_EDGE_PAD_PX,
    CONTOUR_COLOR,
    CONTOUR_INDEX_COLOR,
    CONTOUR_INDEX_EVERY,
    CONTOUR_INDEX_WIDTH,
    CONTOUR_INTERVAL_M,
    CONTOUR_LABEL_EDGE_MARGIN_PX,
    CONTOUR_LABEL_GAP_ENABLED,
    CONTOUR_LABEL_GAP_PADDING,
    CONTOUR_LABEL_MIN_SEG_LEN_PX,
    CONTOUR_LABEL_SPACING_PX,
    CONTOUR_LABELS_ENABLED,
    CONTOUR_LOG_MEMORY_EVERY_TILES,
    CONTOUR_PASS2_QUEUE_MAXSIZE,
    CONTOUR_SEED_DOWNSAMPLE,
    CONTOUR_SEED_SMOOTHING,
    CONTOUR_WIDTH,
    CONTROL_POINT_CROSS_COLOR,
    CONTROL_POINT_CROSS_LENGTH_PX,
    CONTROL_POINT_CROSS_LINE_WIDTH_PX,
    DOWNLOAD_CONCURRENCY,
    EARTH_RADIUS_M,
    ELEVATION_USE_RETINA,
    GRID_COLOR,
    GRID_STEP_M,
    MARCHING_SQUARES_CENTER_WEIGHT,
    MAX_OUTPUT_PIXELS,
    MAX_ZOOM,
    MIN_POINTS_FOR_SMOOTHING,
    MS_AMBIGUOUS_CASES,
    MS_CONNECT_LEFT_BOTTOM,
    MS_CONNECT_LEFT_RIGHT,
    MS_CONNECT_RIGHT_BOTTOM,
    MS_CONNECT_TOP_BOTTOM,
    MS_CONNECT_TOP_LEFT,
    MS_CONNECT_TOP_RIGHT,
    MS_MASK_TL_BR,
    MS_NO_CONTOUR_CASES,
    PIL_DISABLE_LIMIT,
    ROTATION_PAD_MIN_PX,
    ROTATION_PAD_RATIO,
    SEED_POLYLINE_QUANT_FACTOR,
    TILE_SIZE,
    USE_NUMPY_FASTPATH,
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
    rotate_keep_size,
)
from image_io import build_save_kwargs as _build_save_kwargs
from image_io import save_jpeg as _save_jpeg
from preview import publish_preview_image
from progress import ConsoleProgress, LiveSpinner
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
    try:
        mt_val = MapType(mt) if not isinstance(mt, MapType) else mt
        is_elev_color = mt_val == MapType.ELEVATION_COLOR
        is_elev_contours = mt_val == MapType.ELEVATION_CONTOURS
    except Exception:
        is_elev_color = False
        is_elev_contours = False

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
    else:
        eff_scale = effective_scale_for_xyz(XYZ_TILE_SIZE, use_retina=XYZ_USE_RETINA)

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

                # Pass A: sample elevations for percentiles (reservoir sampling)
                from elevation.stats import compute_elevation_range as _elev_range
                from elevation.stats import sample_elevation_percentiles as _sample_elev

                tile_progress.label = 'Проверка диапазона высот (проход 1/2)'

                async def _get_dem_tile(xw: int, yw: int) -> Image.Image:
                    return await provider_main.get_tile_image(zoom, xw, yw)

                samples, seen_count = await _sample_elev(
                    enumerate(tiles),
                    tiles_x=tiles_x,
                    crop_rect=crop_rect,
                    full_eff_tile_px=full_eff_tile_px,
                    get_tile_image=_get_dem_tile,
                    max_samples=50000,
                    rng_seed=42,
                    on_progress=tile_progress.step,
                    semaphore=semaphore,
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
                inv = 1.0 / (hi - lo) if hi > lo else 0.0
                # Сохраняем диапазон высот для легенды
                elev_min_m = lo
                elev_max_m = hi

                # Pass B: render directly to output image using producer–consumer (I/O vs CPU)
                result = Image.new('RGB', (crop_rect[2], crop_rect[3]))
                tile_progress = ConsoleProgress(
                    total=len(tiles), label='Окрашивание DEM (проход 2/2)'
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
                    a0 = (-10000.0 - lo) * inv

                    try:
                        _numpy_ok = bool(USE_NUMPY_FASTPATH)
                    except Exception:
                        _numpy_ok = False
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
                            # Build raw RGB buffer for the overlap block (scanline writing)
                            block_w = x1 - x0
                            block_h = y1 - y0
                            if _numpy_ok:
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
                            else:
                                # Fallback: per-pixel loop without NumPy
                                buf = bytearray(block_w * block_h * 3)
                                out_idx = 0
                                pix = img.load()
                                assert pix is not None
                                for yy in range(y0, y1):
                                    for xx in range(x0, x1):
                                        r0, g0, b0 = cast(
                                            'tuple[int, int, int]',
                                            pix[xx - base_x, yy - base_y],
                                        )[:3]
                                        t = ar * r0 + ag * g0 + ab * b0 + a0
                                        cr, cg, cb = _color_at(t)
                                        buf[out_idx] = cr
                                        buf[out_idx + 1] = cg
                                        buf[out_idx + 2] = cb
                                        out_idx += 3
                                block_img = Image.frombytes(
                                    'RGB', (block_w, block_h), bytes(buf)
                                )
                            # Paste into result under lock (PIL isn't thread-safe)
                            async with paste_lock:
                                result.paste(block_img, (dx0, dy0))
                        finally:
                            img.close()
                            queue.task_done()
                            tile_count += 1
                            if tile_count % CONTOUR_LOG_MEMORY_EVERY_TILES == 0:
                                log_memory_usage(f'pass2 after {tile_count} tiles')
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

                # Build global polylines from seed_dem via marching squares once (in crop pixels / seed ds)
                def build_seed_polylines() -> dict[
                    int, list[list[tuple[float, float]]]
                ]:
                    hs = seed_h
                    ws = seed_w
                    polylines_by_level: dict[int, list[list[tuple[float, float]]]] = {}

                    # Simple marching squares on seed grid; coordinates will be in seed-cell space
                    def interp(
                        p0: float, p1: float, v0: float, v1: float, level: float
                    ) -> float:
                        if v1 == v0:
                            return p0
                        t = (level - v0) / (v1 - v0)
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        return p0 + (p1 - p0) * t

                    # For each level index (use indexes into levels)
                    for li, level in enumerate(levels):
                        segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
                        for j in range(hs - 1):
                            row0 = seed_dem[j]
                            row1 = seed_dem[j + 1]
                            for i in range(ws - 1):
                                v00 = row0[i]
                                v10 = row0[i + 1]
                                v11 = row1[i + 1]
                                v01 = row1[i]
                                mask = (
                                    (1 if v00 >= level else 0)
                                    | ((1 if v10 >= level else 0) << 1)
                                    | ((1 if v11 >= level else 0) << 2)
                                    | ((1 if v01 >= level else 0) << 3)
                                )
                                if mask in MS_NO_CONTOUR_CASES:
                                    continue
                                x = i
                                y = j
                                yl = interp(y, y + 1, v00, v01, level)
                                yr = interp(y, y + 1, v10, v11, level)
                                xt = interp(x, x + 1, v00, v10, level)
                                xb = interp(x, x + 1, v01, v11, level)

                                def add(
                                    a: tuple[float, float],
                                    b: tuple[float, float],
                                    segs_list: list[
                                        tuple[tuple[float, float], tuple[float, float]]
                                    ] = segs,
                                ) -> None:
                                    segs_list.append((a, b))

                                if mask in MS_CONNECT_TOP_LEFT:
                                    add((xt, y), (x, yl))
                                elif mask in MS_CONNECT_TOP_RIGHT:
                                    add((xt, y), (x + 1, yr))
                                elif mask in MS_CONNECT_LEFT_RIGHT:
                                    add((x, yl), (x + 1, yr))
                                elif mask in MS_CONNECT_RIGHT_BOTTOM:
                                    add((x + 1, yr), (xb, y + 1))
                                elif mask in MS_AMBIGUOUS_CASES:
                                    center = (
                                        v00 + v10 + v11 + v01
                                    ) * MARCHING_SQUARES_CENTER_WEIGHT
                                    choose_diag = center >= level
                                    if choose_diag:
                                        if mask == MS_MASK_TL_BR:
                                            add((xt, y), (xb, y + 1))
                                        else:
                                            add((x, yl), (x + 1, yr))
                                    elif mask == MS_MASK_TL_BR:
                                        add((x, yl), (x + 1, yr))
                                    else:
                                        add((xt, y), (xb, y + 1))
                                elif mask in MS_CONNECT_TOP_BOTTOM:
                                    add((xt, y), (xb, y + 1))
                                elif mask in MS_CONNECT_LEFT_BOTTOM:
                                    add((x, yl), (xb, y + 1))
                        # Chain segments into polylines (simple greedy)
                        polylines: list[list[tuple[float, float]]] = []
                        if segs:
                            buckets = defaultdict(list)

                            def key(p: tuple[float, float]) -> tuple[int, int]:
                                # quantize in seed grid to reduce splits
                                qx = round(p[0] * SEED_POLYLINE_QUANT_FACTOR)
                                qy = round(p[1] * SEED_POLYLINE_QUANT_FACTOR)
                                return qx, qy

                            unused = {}
                            for idx, (a, b) in enumerate(segs):
                                unused[idx] = True
                                buckets[key(a)].append((idx, 0))
                                buckets[key(b)].append((idx, 1))
                            for si in range(len(segs)):
                                if si not in unused:
                                    continue
                                # start a polyline from seg si
                                unused.pop(si, None)
                                a, b = segs[si]
                                poly = [a, b]
                                end = b
                                # extend forward
                                while True:
                                    k = key(end)
                                    found = None
                                    for idx, endpos in buckets.get(k, []):
                                        if idx in unused:
                                            aa, bb = segs[idx]
                                            if endpos == 0:
                                                end = bb
                                                poly.append(bb)
                                            else:
                                                end = aa
                                                poly.append(aa)
                                            unused.pop(idx, None)
                                            found = True
                                            break
                                    if not found:
                                        break
                                polylines.append(poly)
                        polylines_by_level[li] = polylines
                    return polylines_by_level

                from contours.seeds import build_seed_polylines as _build_seeds

                seed_polylines = _build_seeds(seed_dem, levels)

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
                        mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
                            TILE_SIZE * (2**zoom)
                        )
                        logger.info(
                            'Подписи изогипс: mpp=%.6f (TILE_SIZE=%d)', mpp, TILE_SIZE
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
                                int(CONTOUR_LABEL_SPACING_PX),
                                int(CONTOUR_LABEL_MIN_SEG_LEN_PX),
                                int(CONTOUR_LABEL_EDGE_MARGIN_PX),
                            )

                        # Создаем разрывы линий контуров в местах подписей

                        if CONTOUR_LABEL_GAP_ENABLED and label_bboxes:
                            draw = ImageDraw.Draw(result)
                            gap_padding = int(CONTOUR_LABEL_GAP_PADDING)
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
            eff_scale_cont = effective_scale_for_xyz(
                256, use_retina=ELEVATION_USE_RETINA
            )
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

            full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)

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
                    client=client2, api_key=api_key, use_retina=ELEVATION_USE_RETINA
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

            def build_seed_polylines() -> dict[int, list[list[tuple[float, float]]]]:
                h_s = seed_h
                w_s = seed_w
                polylines_by_level: dict[int, list[list[tuple[float, float]]]] = {}

                def interp(
                    p0: float, p1: float, v0: float, v1: float, level: float
                ) -> float:
                    if v1 == v0:
                        return p0
                    t = (level - v0) / (v1 - v0)
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    return p0 + (p1 - p0) * t

                for li, level in enumerate(levels_c):
                    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
                    for j in range(h_s - 1):
                        row0 = seed_dem_c[j]
                        row1 = seed_dem_c[j + 1]
                        for i in range(w_s - 1):
                            v00 = row0[i]
                            v10 = row0[i + 1]
                            v11 = row1[i + 1]
                            v01 = row1[i]
                            mask = (
                                (1 if v00 >= level else 0)
                                | ((1 if v10 >= level else 0) << 1)
                                | ((1 if v11 >= level else 0) << 2)
                                | ((1 if v01 >= level else 0) << 3)
                            )
                            if mask in MS_NO_CONTOUR_CASES:
                                continue
                            x = i
                            y = j
                            yl = interp(y, y + 1, v00, v01, level)
                            yr = interp(y, y + 1, v10, v11, level)
                            xt = interp(x, x + 1, v00, v10, level)
                            xb = interp(x, x + 1, v01, v11, level)

                            def add(
                                a: tuple[float, float],
                                b: tuple[float, float],
                                _segs=segs,
                            ) -> None:
                                _segs.append((a, b))

                            if mask in MS_CONNECT_TOP_LEFT:
                                add((xt, y), (x, yl))
                            elif mask in MS_CONNECT_TOP_RIGHT:
                                add((xt, y), (x + 1, yr))
                            elif mask in MS_CONNECT_LEFT_RIGHT:
                                add((x, yl), (x + 1, yr))
                            elif mask in MS_CONNECT_RIGHT_BOTTOM:
                                add((x + 1, yr), (xb, y + 1))
                            elif mask in MS_AMBIGUOUS_CASES:
                                center = (
                                    v00 + v10 + v11 + v01
                                ) * MARCHING_SQUARES_CENTER_WEIGHT
                                choose_diag = center >= level
                                if choose_diag:
                                    if mask == MS_AMBIGUOUS_CASES[0]:
                                        add((xt, y), (xb, y + 1))
                                    else:
                                        add((x, yl), (x + 1, yr))
                                elif mask == MS_AMBIGUOUS_CASES[0]:
                                    add((x, yl), (x + 1, yr))
                                else:
                                    add((xt, y), (xb, y + 1))
                            elif mask in MS_CONNECT_TOP_BOTTOM:
                                add((xt, y), (xb, y + 1))
                            elif mask in MS_CONNECT_LEFT_BOTTOM:
                                add((x, yl), (xb, y + 1))
                    polylines: list[list[tuple[float, float]]] = []
                    if segs:
                        buckets = defaultdict(list)

                        def key(p: tuple[float, float]) -> tuple[int, int]:
                            qx = round(p[0] * SEED_POLYLINE_QUANT_FACTOR)
                            qy = round(p[1] * SEED_POLYLINE_QUANT_FACTOR)
                            return qx, qy

                        unused = {}
                        for idx, (a, b) in enumerate(segs):
                            unused[idx] = True
                            buckets[key(a)].append((idx, 0))
                            buckets[key(b)].append((idx, 1))
                        for si in range(len(segs)):
                            if si not in unused:
                                continue
                            a, b = segs[si]
                            poly = [a, b]
                            end = b
                            while True:
                                k = key(end)
                                found = None
                                for idx, endpos in buckets.get(k, []):
                                    if idx in unused:
                                        aa, bb = segs[idx]
                                        if endpos == 0:
                                            end = bb
                                            poly.append(bb)
                                        else:
                                            end = aa
                                            poly.append(aa)
                                        unused.pop(idx, None)
                                        found = True
                                        break
                                if not found:
                                    break
                            polylines.append(poly)
                    polylines_by_level[li] = polylines
                return polylines_by_level

            from contours.seeds import build_seed_polylines as _build_seeds

            logger.info(
                'Изолинии: построение seed polylines для %d уровней', len(levels_c)
            )
            seed_polylines = _build_seeds(seed_dem_c, levels_c)
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
            for li, _level in enumerate(levels_c):
                is_index = (li % max(1, int(CONTOUR_INDEX_EVERY))) == 0
                color = CONTOUR_INDEX_COLOR if is_index else CONTOUR_COLOR
                width = int(CONTOUR_INDEX_WIDTH if is_index else CONTOUR_WIDTH)
                for poly in seed_polylines.get(li, []):
                    # Применяем сглаживание если включено
                    if CONTOUR_SEED_SMOOTHING and len(poly) >= MIN_POINTS_FOR_SMOOTHING:
                        from contours.seeds import smooth_polyline

                        smoothed_poly = smooth_polyline(poly)
                    else:
                        smoothed_poly = poly

                    # draw full polylines; block optimization skipped for simplicity
                    pts_crop = [
                        (int(p[0] * seed_ds), int(p[1] * seed_ds))
                        for p in smoothed_poly
                    ]
                    if len(pts_crop) < 2:  # noqa: PLR2004
                        continue
                    draw = ImageDraw.Draw(overlay)
                    for i in range(1, len(pts_crop)):
                        x0, y0 = pts_crop[i - 1]
                        x1, y1 = pts_crop[i]
                        draw.line(
                            (x0, y0, x1, y1), fill=(*list(color), 255), width=width
                        )
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
                    mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
                        TILE_SIZE * (2**zoom)
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
                        gap_padding = int(CONTOUR_LABEL_GAP_PADDING)
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

    # Draw elevation legend first (if needed) to get bounds for grid line breaking
    legend_bounds: tuple[int, int, int, int] | None = None
    if is_elev_color and elev_min_m is not None and elev_max_m is not None:
        legend_start_time = time.monotonic()
        logger.info('Рисование легенды высот — старт')
        try:
            legend_bounds = draw_elevation_legend(
                img=result,
                color_ramp=ELEVATION_COLOR_RAMP,
                min_elevation_m=elev_min_m,
                max_elevation_m=elev_max_m,
                center_lat_wgs=center_lat_wgs,
                zoom=zoom,
                scale=eff_scale,
            )
            legend_elapsed = time.monotonic() - legend_start_time
            logger.info('Рисование легенды высот — завершено (%.2fs)', legend_elapsed)
        except Exception as e:
            logger.warning('Не удалось нарисовать легенду высот: %s', e)
            legend_bounds = None

    # Grid drawing (with legend bounds for line breaking if legend was drawn)
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
        legend_bounds=legend_bounds,
        display_grid=settings.display_grid,
        rotation_deg=rotation_deg,
    )
    grid_elapsed = time.monotonic() - grid_start_time
    logger.info('Рисование км-сетки — завершено (%.2fs)', grid_elapsed)
    log_memory_usage('after grid drawing')
    log_thread_status('after grid drawing')

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
        cx = result.width // 2
        cy = result.height // 2
        half = max(1, int(CENTER_CROSS_LENGTH_PX) // 2)
        line_w = max(1, int(CENTER_CROSS_LINE_WIDTH_PX))
        draw = ImageDraw.Draw(result)
        color = tuple(CENTER_CROSS_COLOR)
        draw.line([(cx, cy - half), (cx, cy + half)], fill=color, width=line_w)
        draw.line([(cx - half, cy), (cx + half, cy)], fill=color, width=line_w)
    except Exception as e:
        logger.warning(
            'Не удалось нарисовать центрированный крест или вывести координаты: %s', e
        )

    # Draw control point as red cross (using same mechanics as grid/center cross)
    try:
        if getattr(settings, 'control_point_enabled', False):
            # Get control point coordinates in SK-42 GK (easting, northing)
            cp_x_gk = float(settings.control_point_x_sk42_gk)
            cp_y_gk = float(settings.control_point_y_sk42_gk)

            # Compute center in GK as above
            t_sk42gk_from_sk42 = Transformer.from_crs(
                crs_sk42_geog,
                crs_sk42_gk,
                always_xy=True,
            )
            x0_gk, y0_gk = t_sk42gk_from_sk42.transform(
                center_lng_sk42, center_lat_sk42
            )

            # Pixels-per-meter at center latitude in WGS84
            mpp = meters_per_pixel(center_lat_wgs, zoom, scale=eff_scale)
            ppm = 1.0 / mpp if mpp > 0 else 0.0

            # Map GK offsets to pixel offsets (screen Y grows down)
            dx_m = cp_x_gk - x0_gk
            dy_m = cp_y_gk - y0_gk
            cx_img = result.width / 2.0 + dx_m * ppm
            cy_img = result.height / 2.0 - dy_m * ppm

            # Optional: log WGS84 of control point via Helmert-aware transformer

            t_sk42_from_gk = Transformer.from_crs(
                crs_sk42_gk, crs_sk42_geog, always_xy=True
            )
            cp_lng_sk42, cp_lat_sk42 = t_sk42_from_gk.transform(cp_x_gk, cp_y_gk)
            cp_lng_wgs, cp_lat_wgs = t_sk42_to_wgs.transform(cp_lng_sk42, cp_lat_sk42)
            logger.info(
                'Контрольная точка: СК-42 ГК X(север)=%.3f, Y(восток)=%.3f; WGS84 lat=%.8f, lon=%.8f',
                cp_y_gk,
                cp_x_gk,
                cp_lat_wgs,
                cp_lng_wgs,
            )

            # Draw cross if inside image bounds
            if 0 <= cx_img <= result.width and 0 <= cy_img <= result.height:
                draw = ImageDraw.Draw(result)
                half = max(1, int(CONTROL_POINT_CROSS_LENGTH_PX) // 2)
                line_w = max(1, int(CONTROL_POINT_CROSS_LINE_WIDTH_PX))
                color = tuple(CONTROL_POINT_CROSS_COLOR)
                cx_i = round(cx_img)
                cy_i = round(cy_img)
                draw.line(
                    [(cx_i, cy_i - half), (cx_i, cy_i + half)], fill=color, width=line_w
                )
                draw.line(
                    [(cx_i - half, cy_i), (cx_i + half, cy_i)], fill=color, width=line_w
                )
            else:
                logger.debug(
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

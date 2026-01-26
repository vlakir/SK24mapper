"""Elevation contours processor - DEM contour lines generation."""

from __future__ import annotations

import asyncio
import logging
import math
import random
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw

from contours import draw_contour_labels
from contours.adaptive import ContourAdaptiveParams, compute_contour_adaptive_params
from contours.helpers import tx_ty_from_index
from elevation.provider import ElevationTileProvider
from geo.geometry import tile_overlap_rect_common
from geo.topography import decode_terrain_rgb_to_elevation_m
from infrastructure.http.client import resolve_cache_dir
from render.contours_builder import build_seed_polylines
from shared.constants import (
    CONTOUR_COLOR,
    CONTOUR_INDEX_COLOR,
    CONTOUR_INDEX_EVERY,
    CONTOUR_INDEX_WIDTH,
    CONTOUR_LABEL_GAP_ENABLED,
    CONTOUR_LABELS_ENABLED,
    CONTOUR_SEED_DOWNSAMPLE,
    CONTOUR_WIDTH,
    EARTH_RADIUS_M,
    ELEVATION_USE_RETINA,
    MIN_POINTS_FOR_SEGMENT,
    TILE_SIZE,
)
from shared.progress import ConsoleProgress, LiveSpinner

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def process_elevation_contours(ctx: MapDownloadContext) -> Image.Image:
    """
    Process elevation contours map.

    Two-pass approach:
    1. Sample elevations and build low-res DEM seed
    2. Generate contour lines using marching squares

    Args:
        ctx: Map download context with all necessary parameters.

    Returns:
        Image with contour lines.

    """
    full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)

    provider = ElevationTileProvider(
        client=ctx.client,
        api_key=ctx.api_key,
        use_retina=ELEVATION_USE_RETINA,
        cache_root=resolve_cache_dir(),
    )

    # Pass A: sample min/max elevations and build global low-res DEM seed
    max_samples = 50000
    samples_contours: list[float] = []

    rng = random.Random(42)  # noqa: S311
    tile_progress = ConsoleProgress(
        total=len(ctx.tiles), label='Проверка диапазона высот (проход 1/2)'
    )

    # Prepare low-res DEM seed canvas (numpy arrays для быстрого доступа)
    seed_ds = max(2, int(CONTOUR_SEED_DOWNSAMPLE))
    cx, cy, cw, ch = ctx.crop_rect
    seed_w = max(1, (cw + seed_ds - 1) // seed_ds)
    seed_h = max(1, (ch + seed_ds - 1) // seed_ds)
    seed_sum = np.zeros((seed_h, seed_w), dtype=np.float64)
    seed_cnt = np.zeros((seed_h, seed_w), dtype=np.int32)

    # Список для сбора результатов от всех тайлов (без lock)
    tile_results: list[tuple[list[float], list[tuple[int, int, float]]]] = []
    results_lock = asyncio.Lock()

    async def fetch_and_sample(idx_xy: tuple[int, tuple[int, int]]) -> None:
        idx, (tile_x_world, tile_y_world) = idx_xy
        tx, ty = tx_ty_from_index(idx, ctx.tiles_x)
        if tile_overlap_rect_common(tx, ty, ctx.crop_rect, full_eff_tile_px) is None:
            tile_progress.step_sync(1)
            return

        async with ctx.semaphore:
            img = await provider.get_tile_image(ctx.zoom, tile_x_world, tile_y_world)
            dem_tile = decode_terrain_rgb_to_elevation_m(img)
            h = len(dem_tile)
            w = len(dem_tile[0]) if h else 0

            if h and w:
                # Reservoir sampling for min/max
                step_y = max(1, h // 32)
                step_x = max(1, w // 32)
                off_y = rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
                off_x = rng.randrange(0, min(step_x, w)) if step_x > 1 else 0

                # Собираем локальные данные для этого тайла
                local_samples: list[float] = []
                for ry in range(off_y, h, step_y):
                    local_samples.extend(dem_tile[ry][off_x:w:step_x])

                # Accumulate into low-res seed (локальные данные)
                local_updates: list[tuple[int, int, float]] = []
                ov = tile_overlap_rect_common(tx, ty, ctx.crop_rect, full_eff_tile_px)
                if ov is not None:
                    x0, y0, x1, y1 = ov
                    base_x = tx * full_eff_tile_px
                    base_y = ty * full_eff_tile_px

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
                            local_updates.append((sy, sx, v))

                # Сохраняем результаты для batch-обработки в конце
                if local_samples or local_updates:
                    async with results_lock:
                        tile_results.append((local_samples, local_updates))

        tile_progress.step_sync(1)

    try:
        async with asyncio.TaskGroup() as tg:
            for pair in enumerate(ctx.tiles):
                tg.create_task(fetch_and_sample(pair))
    finally:
        tile_progress.close()

    # Batch-обработка всех результатов (без contention)
    all_samples: list[float] = []
    for local_samples, local_updates in tile_results:
        all_samples.extend(local_samples)
        for sy, sx, v in local_updates:
            seed_sum[sy, sx] += v
            seed_cnt[sy, sx] += 1

    # Reservoir sampling из собранных samples
    if len(all_samples) > max_samples:
        rng.shuffle(all_samples)
        samples_contours = all_samples[:max_samples]
    else:
        samples_contours = all_samples

    if samples_contours:
        mn = min(samples_contours)
        mx = max(samples_contours)
    else:
        mn, mx = 0.0, 1.0
    if mx < mn:
        mn, mx = mx, mn

    # Finalize low-res DEM seed (векторизованно через numpy)
    with np.errstate(divide='ignore', invalid='ignore'):
        seed_dem_arr = np.where(seed_cnt > 0, seed_sum / seed_cnt, (mn + mx) * 0.5)
    filled = int(np.sum(seed_cnt > 0))

    logger.info(
        'Seed DEM low-res size=%sx%s, filled=%s/%s',
        seed_w,
        seed_h,
        filled,
        seed_w * seed_h,
    )

    map_size_m = max(float(ctx.width_m), float(ctx.height_m))
    adaptive_params = compute_contour_adaptive_params(map_size_m)
    logger.info(
        'Адаптация изолиний: map_size=%.1f м, scale=%.3f, interval=%.2f м',
        map_size_m,
        adaptive_params.scale,
        adaptive_params.interval_m,
    )

    # Build contour levels
    if adaptive_params.interval_m <= 0:
        interval = 25.0
    else:
        interval = float(adaptive_params.interval_m)

    start_v = math.floor(mn / interval) * interval
    end_v = math.ceil(mx / interval) * interval
    levels: list[float] = [
        start_v + k * interval for k in range(int((end_v - start_v) / interval) + 1)
    ]

    # Pass B: draw contours
    sp = LiveSpinner('Построение изолиний')
    sp.start()
    try:
        result = Image.new(
            'RGB', (ctx.crop_rect[2], ctx.crop_rect[3]), color=(255, 255, 255)
        )

        # Build global polylines from seed_dem
        seed_polylines = build_seed_polylines(seed_dem_arr, levels, seed_h, seed_w)

        # Draw contours directly on result image using OpenCV (batch drawing)
        img_w, img_h = result.size
        lat_rad = math.radians(ctx.center_lat_wgs)
        elev_retina_factor = 2 if ELEVATION_USE_RETINA else 1
        mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
            TILE_SIZE * elev_retina_factor * (2**ctx.zoom)
        )

        # Конвертируем PIL Image в numpy array для OpenCV
        result_array = np.array(result)

        for li, _level in enumerate(levels):
            is_index = (li % max(1, int(CONTOUR_INDEX_EVERY))) == 0
            color = CONTOUR_INDEX_COLOR if is_index else CONTOUR_COLOR
            target_width_m = CONTOUR_INDEX_WIDTH if is_index else CONTOUR_WIDTH
            width = max(1, round(target_width_m / max(1e-9, mpp)))

            # Собираем все полилинии для этого уровня
            polylines_for_level: list[np.ndarray] = []
            for poly in seed_polylines.get(li, []):
                # Координаты в системе результирующего изображения
                pts_array = np.array(poly, dtype=np.float32) * seed_ds

                # Проверка, что полилиния в пределах изображения
                minx_p, miny_p = pts_array.min(axis=0)
                maxx_p, maxy_p = pts_array.max(axis=0)

                if maxx_p < 0 or minx_p > img_w or maxy_p < 0 or miny_p > img_h:
                    continue

                # OpenCV требует int32 координаты
                polylines_for_level.append(pts_array.astype(np.int32))

            # Batch-отрисовка всех полилиний уровня через cv2.polylines
            if polylines_for_level:
                # OpenCV использует BGR, PIL — RGB
                cv_color = (color[2], color[1], color[0])
                cv2.polylines(
                    result_array,
                    polylines_for_level,
                    isClosed=False,
                    color=cv_color,
                    thickness=width,
                    lineType=cv2.LINE_AA,
                )

        # Конвертируем обратно в PIL Image
        result = Image.fromarray(result_array)

        # Add contour labels if enabled
        if CONTOUR_LABELS_ENABLED:
            result = _add_contour_labels(
                result,
                seed_polylines,
                levels,
                ctx.crop_rect,
                seed_ds,
                ctx.center_lat_wgs,
                ctx.zoom,
                label_params=adaptive_params,
            )
    finally:
        sp.stop('Построение изолиний завершено')

    return result


def _add_contour_labels(
    result: Image.Image,
    seed_polylines: dict[int, list[list[tuple[float, float]]]],
    levels: list[float],
    crop_rect: tuple[int, int, int, int],
    seed_ds: int,
    center_lat_wgs: float,
    zoom: int,
    *,
    is_overlay: bool = False,
    overlay_retina_factor: int | None = None,
    label_params: ContourAdaptiveParams | None = None,
) -> Image.Image:
    """Add labels to contour lines."""
    try:
        logger.info(
            'Подписи изогипс: подготовка (zoom=%d, crop=%dx%d at (%d,%d), '
            'seed_ds=%d, levels=%d)',
            zoom,
            crop_rect[2],
            crop_rect[3],
            crop_rect[0],
            crop_rect[1],
            seed_ds,
            len(levels),
        )

        lat_rad = math.radians(center_lat_wgs)
        elev_retina_factor = overlay_retina_factor or (2 if ELEVATION_USE_RETINA else 1)
        mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
            TILE_SIZE * elev_retina_factor * (2**zoom)
        )

        logger.info(
            'Подписи изогипс: mpp=%.6f (TILE_SIZE=%d, retina_factor=%d)',
            mpp,
            TILE_SIZE,
            elev_retina_factor,
        )

        if label_params is None:
            map_size_m = max(float(crop_rect[2]), float(crop_rect[3])) * mpp
            label_params = compute_contour_adaptive_params(map_size_m)
            logger.info(
                'Адаптация подписей: map_size=%.1f м, scale=%.3f, spacing=%.1f м, '
                'font=%.1f м',
                map_size_m,
                label_params.scale,
                label_params.label_spacing_m,
                label_params.label_font_m,
            )

        # Dry run to get label positions
        label_bboxes = draw_contour_labels(
            result,
            seed_polylines,
            levels,
            crop_rect,
            seed_ds,
            mpp,
            dry_run=True,
            label_spacing_m=label_params.label_spacing_m,
            label_min_seg_len_m=label_params.label_min_seg_len_m,
            label_edge_margin_m=label_params.label_edge_margin_m,
            label_font_m=label_params.label_font_m,
        )

        logger.info(
            'Подписи изогипс: dry_run завершён, кандидатов=%d', len(label_bboxes)
        )

        if not label_bboxes:
            logger.warning(
                'Подписи изогипс: dry_run вернул 0 кандидатов — проверьте пороги '
                '(spacing=%d,min_len=%d,edge=%d)',
                int(label_params.label_spacing_m),
                int(label_params.label_min_seg_len_m),
                int(label_params.label_edge_margin_m),
            )

        # Create gaps in contour lines at label positions
        # Для overlay режима не рисуем белые прямоугольники —
        # подписи выводятся прямо на карту
        if CONTOUR_LABEL_GAP_ENABLED and label_bboxes and not is_overlay:
            draw = ImageDraw.Draw(result)
            gap_padding = max(
                1, round(label_params.label_gap_padding_m / max(1e-9, mpp))
            )

            for bbox in label_bboxes:
                x0, y0, x1, y1 = bbox
                gap_area = (
                    x0 - gap_padding,
                    y0 - gap_padding,
                    x1 + gap_padding,
                    y1 + gap_padding,
                )
                draw.rectangle(gap_area, fill=(255, 255, 255))

        # Final pass: draw labels
        placed_after = draw_contour_labels(
            result,
            seed_polylines,
            levels,
            crop_rect,
            seed_ds,
            mpp,
            dry_run=False,
            label_spacing_m=label_params.label_spacing_m,
            label_min_seg_len_m=label_params.label_min_seg_len_m,
            label_edge_margin_m=label_params.label_edge_margin_m,
            label_font_m=label_params.label_font_m,
        )

        logger.info(
            'Подписи изогипс: финальная отрисовка, размещено=%d', len(placed_after)
        )

    except Exception as e:
        logger.warning('Не удалось нанести подписи изогипс: %s', e, exc_info=True)

    return result


async def apply_contours_to_image(
    ctx: MapDownloadContext,
    base_image: Image.Image,
) -> Image.Image:
    """
    Apply contour lines overlay to an existing image.

    This function fetches elevation data and draws contour lines
    on top of the provided base image (e.g., satellite or streets map).

    Args:
        ctx: Map download context with all necessary parameters.
        base_image: The base image to overlay contours on.

    Returns:
        Image with contour lines overlaid.

    """
    # Terrain-RGB тайлы поддерживают только 256px или 512px (@2x)
    overlay_retina_factor = max(1, round(ctx.full_eff_tile_px / 256))
    overlay_retina_factor = min(overlay_retina_factor, 2)
    elev_tile_px = 256 * overlay_retina_factor

    # ctx.crop_rect и ctx.full_eff_tile_px рассчитаны для XYZ тайлов,
    # которые могут иметь другой размер. Нужно пересчитать crop_rect
    # для Elevation тайлов.
    scale_factor = elev_tile_px / ctx.full_eff_tile_px
    orig_cx, orig_cy, orig_cw, orig_ch = ctx.crop_rect

    # Пересчитываем crop_rect для elevation тайлов
    elev_cx = int(orig_cx * scale_factor)
    elev_cy = int(orig_cy * scale_factor)
    elev_cw = int(orig_cw * scale_factor)
    elev_ch = int(orig_ch * scale_factor)
    elev_crop_rect = (elev_cx, elev_cy, elev_cw, elev_ch)

    logger.info(
        'Overlay: base tile_px=%d, Elev tile_px=%d, scale=%.3f, '
        'orig_crop=(%d,%d,%d,%d), elev_crop=(%d,%d,%d,%d)',
        ctx.full_eff_tile_px,
        elev_tile_px,
        scale_factor,
        orig_cx,
        orig_cy,
        orig_cw,
        orig_ch,
        elev_cx,
        elev_cy,
        elev_cw,
        elev_ch,
    )

    provider = ElevationTileProvider(
        client=ctx.client, api_key=ctx.api_key, use_retina=(overlay_retina_factor > 1)
    )

    # Pass A: sample elevations and build global low-res DEM seed
    max_samples = 50000
    samples_contours: list[float] = []

    rng = random.Random(42)  # noqa: S311
    tile_progress = ConsoleProgress(
        total=len(ctx.tiles), label='Загрузка высот для изолиний'
    )

    # Prepare low-res DEM seed canvas (numpy arrays для быстрого доступа)
    seed_ds = max(2, int(CONTOUR_SEED_DOWNSAMPLE))
    cx, cy, cw, ch = elev_crop_rect
    seed_w = max(1, (cw + seed_ds - 1) // seed_ds)
    seed_h = max(1, (ch + seed_ds - 1) // seed_ds)
    seed_sum = np.zeros((seed_h, seed_w), dtype=np.float64)
    seed_cnt = np.zeros((seed_h, seed_w), dtype=np.int32)

    # Список для сбора результатов от всех тайлов (без lock)
    tile_results: list[tuple[list[float], list[tuple[int, int, float]]]] = []
    results_lock = asyncio.Lock()

    async def fetch_and_sample(idx_xy: tuple[int, tuple[int, int]]) -> None:
        idx, (tile_x_world, tile_y_world) = idx_xy
        tx, ty = tx_ty_from_index(idx, ctx.tiles_x)
        if tile_overlap_rect_common(tx, ty, elev_crop_rect, elev_tile_px) is None:
            await tile_progress.step(1)
            return

        try:
            async with ctx.semaphore:
                img = await provider.get_tile_image(
                    ctx.zoom, tile_x_world, tile_y_world
                )
                dem_tile = decode_terrain_rgb_to_elevation_m(img)
                h = len(dem_tile)
                w = len(dem_tile[0]) if h else 0

                if h and w:
                    # Reservoir sampling for min/max
                    step_y = max(1, h // 32)
                    step_x = max(1, w // 32)
                    off_y = rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
                    off_x = rng.randrange(0, min(step_x, w)) if step_x > 1 else 0

                    # Собираем локальные данные для этого тайла
                    local_samples: list[float] = []
                    for ry in range(off_y, h, step_y):
                        local_samples.extend(dem_tile[ry][off_x:w:step_x])

                    # Accumulate into low-res seed (локальные данные)
                    local_updates: list[tuple[int, int, float]] = []
                    ov = tile_overlap_rect_common(tx, ty, elev_crop_rect, elev_tile_px)
                    if ov is not None:
                        x0, y0, x1, y1 = ov
                        base_x = tx * elev_tile_px
                        base_y = ty * elev_tile_px

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
                                local_updates.append((sy, sx, v))

                    # Сохраняем результаты для batch-обработки в конце
                    if local_samples or local_updates:
                        async with results_lock:
                            tile_results.append((local_samples, local_updates))
        except Exception as exc:
            logger.warning(
                'Не удалось получить/обработать terrain тайл z/x/y=%d/%d/%d: %s',
                ctx.zoom,
                tile_x_world,
                tile_y_world,
                exc,
                exc_info=True,
            )
        finally:
            await tile_progress.step(1)

    try:
        async with asyncio.TaskGroup() as tg:
            for pair in enumerate(ctx.tiles):
                tg.create_task(fetch_and_sample(pair))
    finally:
        tile_progress.close()

    # Batch-обработка всех результатов (без contention)
    all_samples: list[float] = []
    for local_samples, local_updates in tile_results:
        all_samples.extend(local_samples)
        for sy, sx, v in local_updates:
            seed_sum[sy, sx] += v
            seed_cnt[sy, sx] += 1

    # Reservoir sampling из собранных samples
    if len(all_samples) > max_samples:
        rng.shuffle(all_samples)
        samples_contours = all_samples[:max_samples]
    else:
        samples_contours = all_samples

    # Finalize low-res DEM seed (векторизованно через numpy)
    with np.errstate(divide='ignore', invalid='ignore'):
        seed_dem = np.where(seed_cnt > 0, seed_sum / seed_cnt, 0.0)
    filled_count = int(np.sum(seed_cnt > 0))

    logger.info(
        'Overlay seed_dem: size=%dx%d, filled=%d/%d (%.1f%%), crop_rect=(%d,%d,%d,%d)',
        seed_w,
        seed_h,
        filled_count,
        seed_w * seed_h,
        100.0 * filled_count / max(1, seed_w * seed_h),
        cx,
        cy,
        cw,
        ch,
    )

    if not samples_contours:
        logger.warning('Нет данных высот для изолиний')
        return base_image

    mn = min(samples_contours)
    mx = max(samples_contours)
    logger.info('Диапазон высот для изолиний: %.1f – %.1f м', mn, mx)

    map_size_m = max(float(ctx.width_m), float(ctx.height_m))
    adaptive_params = compute_contour_adaptive_params(map_size_m)
    logger.info(
        'Overlay адаптация изолиний: map_size=%.1f м, scale=%.3f, interval=%.2f м',
        map_size_m,
        adaptive_params.scale,
        adaptive_params.interval_m,
    )

    # Build contour levels
    interval = adaptive_params.interval_m
    start = math.floor(mn / interval) * interval
    end = math.ceil(mx / interval) * interval
    levels: list[float] = []
    k = 0
    v = start
    while v <= end:
        levels.append(v)
        k += 1
        v = start + k * interval

    # Pass B: draw contours
    sp = LiveSpinner('Построение изолиний')
    sp.start()
    try:
        # Build global polylines from seed_dem
        seed_polylines = build_seed_polylines(seed_dem, levels, seed_h, seed_w)

        total_polys = sum(len(polys) for polys in seed_polylines.values())
        logger.info(
            'Overlay polylines: levels=%d, total_polylines=%d, base_image=%dx%d',
            len(levels),
            total_polys,
            base_image.width,
            base_image.height,
        )

        # Pass B: draw contours on base image
        # Convert to RGBA if needed for compositing
        if base_image.mode != 'RGBA':
            result = base_image.convert('RGBA')
        else:
            result = base_image.copy()

        img_w, img_h = result.size

        # Координаты полилиний в системе elev_crop_rect, но base_image имеет
        # размер orig_cw x orig_ch.
        # Нужно масштабировать координаты: elev_coords * seed_ds -> elev_px -> base_px
        # elev_px / scale_factor = base_px, т.е. множитель = seed_ds / scale_factor
        coord_scale = seed_ds / scale_factor
        effective_seed_ds = round(coord_scale)

        # Для создания разрывов в линиях под подписями:
        # 1. Сначала получаем позиции подписей (dry_run)
        # 2. Рисуем линии, пропуская сегменты в местах подписей
        # 3. Рисуем подписи
        label_bboxes: list[tuple[int, int, int, int]] = []
        lat_rad = math.radians(ctx.center_lat_wgs)
        mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
            TILE_SIZE * overlay_retina_factor * (2**ctx.zoom)
        )
        if CONTOUR_LABELS_ENABLED:
            label_bboxes = draw_contour_labels(
                result,
                seed_polylines,
                levels,
                (0, 0, img_w, img_h),
                effective_seed_ds,
                mpp,
                dry_run=True,
                label_spacing_m=adaptive_params.label_spacing_m,
                label_min_seg_len_m=adaptive_params.label_min_seg_len_m,
                label_edge_margin_m=adaptive_params.label_edge_margin_m,
                label_font_m=adaptive_params.label_font_m,
            )
            logger.info(
                'Overlay: получено %d позиций подписей для создания разрывов',
                len(label_bboxes),
            )

        # Функция проверки пересечения сегмента с bbox (с учётом padding)
        gap_padding = max(
            1, round(adaptive_params.label_gap_padding_m / max(1e-9, mpp))
        )

        # Построение пространственного индекса (grid-based spatial hash)
        # для быстрого поиска пересечений O(1) вместо O(n)
        grid_cell_size = 64  # размер ячейки в пикселях
        bbox_grid: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

        # Расширяем bbox на padding и добавляем в grid
        expanded_bboxes: list[tuple[int, int, int, int]] = []
        for bbox in label_bboxes:
            bx0, by0, bx1, by1 = bbox
            exp_bbox = (
                bx0 - gap_padding,
                by0 - gap_padding,
                bx1 + gap_padding,
                by1 + gap_padding,
            )
            expanded_bboxes.append(exp_bbox)
            # Добавляем bbox во все ячейки, которые он покрывает
            cell_x0 = exp_bbox[0] // grid_cell_size
            cell_y0 = exp_bbox[1] // grid_cell_size
            cell_x1 = exp_bbox[2] // grid_cell_size
            cell_y1 = exp_bbox[3] // grid_cell_size
            for cy in range(cell_y0, cell_y1 + 1):
                for cx in range(cell_x0, cell_x1 + 1):
                    key = (cx, cy)
                    if key not in bbox_grid:
                        bbox_grid[key] = []
                    bbox_grid[key].append(exp_bbox)

        def segment_intersects_any_bbox(
            x1: float,
            y1: float,
            x2: float,
            y2: float,
        ) -> bool:
            """Проверяет пересечение сегмента с любым bbox через spatial hash."""
            if not bbox_grid:
                return False
            # Bounding box сегмента
            seg_minx, seg_maxx = (x1, x2) if x1 < x2 else (x2, x1)
            seg_miny, seg_maxy = (y1, y2) if y1 < y2 else (y2, y1)
            # Ячейки, которые покрывает сегмент
            cell_x0 = int(seg_minx) // grid_cell_size
            cell_y0 = int(seg_miny) // grid_cell_size
            cell_x1 = int(seg_maxx) // grid_cell_size
            cell_y1 = int(seg_maxy) // grid_cell_size
            # Проверяем только bbox в релевантных ячейках
            checked: set[tuple[int, int, int, int]] = set()
            for cy in range(cell_y0, cell_y1 + 1):
                for cx in range(cell_x0, cell_x1 + 1):
                    for bbox in bbox_grid.get((cx, cy), []):
                        if bbox in checked:
                            continue
                        checked.add(bbox)
                        bx0, by0, bx1, by1 = bbox
                        # Быстрая проверка пересечения bbox
                        if seg_maxx < bx0 or seg_minx > bx1:
                            continue
                        if seg_maxy < by0 or seg_miny > by1:
                            continue
                        # Точка внутри bbox — пересекает
                        if bx0 <= x1 <= bx1 and by0 <= y1 <= by1:
                            return True
                        if bx0 <= x2 <= bx1 and by0 <= y2 <= by1:
                            return True
            return False

        # Draw contours directly on result image using OpenCV (batch drawing)
        # Конвертируем PIL Image в numpy array для OpenCV
        result_array = np.array(result)

        for li, _level in enumerate(levels):
            is_index = (li % max(1, int(CONTOUR_INDEX_EVERY))) == 0
            color = CONTOUR_INDEX_COLOR if is_index else CONTOUR_COLOR
            target_width_m = CONTOUR_INDEX_WIDTH if is_index else CONTOUR_WIDTH
            width = max(1, round(target_width_m / max(1e-9, mpp)))

            # Собираем полилинии для этого уровня, фильтруя сегменты под подписями
            polylines_for_level: list[np.ndarray] = []

            for poly in seed_polylines.get(li, []):
                # Координаты в системе base_image
                pts_array = np.array(poly, dtype=np.float32) * coord_scale

                # Проверка, что полилиния в пределах изображения
                minx, miny = pts_array.min(axis=0)
                maxx, maxy = pts_array.max(axis=0)

                if maxx < 0 or minx > img_w or maxy < 0 or miny > img_h:
                    continue

                if label_bboxes:
                    # Разбиваем полилинию на сегменты, пропуская те, что под подписями
                    current_segment: list[tuple[float, float]] = []
                    pts_list = pts_array.tolist()

                    for i, (px, py) in enumerate(pts_list):
                        if i == 0:
                            current_segment.append((px, py))
                            continue

                        prev_x, prev_y = pts_list[i - 1]
                        # Проверяем пересечение с bbox подписей через spatial hash
                        skip_segment = segment_intersects_any_bbox(
                            prev_x, prev_y, px, py
                        )

                        if skip_segment:
                            # Сохраняем текущий сегмент и начинаем новый
                            if len(current_segment) >= MIN_POINTS_FOR_SEGMENT:
                                polylines_for_level.append(
                                    np.array(current_segment, dtype=np.int32)
                                )
                            current_segment = [(px, py)]
                        else:
                            current_segment.append((px, py))

                    # Добавляем последний сегмент
                    if len(current_segment) >= MIN_POINTS_FOR_SEGMENT:
                        polylines_for_level.append(
                            np.array(current_segment, dtype=np.int32)
                        )
                else:
                    # Нет подписей — добавляем всю полилинию
                    polylines_for_level.append(pts_array.astype(np.int32))

            # Batch-отрисовка всех полилиний уровня через cv2.polylines
            if polylines_for_level:
                # OpenCV использует BGR, а result может быть RGBA
                cv_color: tuple[int, int, int] | tuple[int, int, int, int]
                if result_array.shape[2] == 4:  # noqa: PLR2004
                    cv_color = (color[2], color[1], color[0], 255)
                else:
                    cv_color = (color[2], color[1], color[0])
                cv2.polylines(
                    result_array,
                    polylines_for_level,
                    isClosed=False,
                    color=cv_color,
                    thickness=width,
                    lineType=cv2.LINE_AA,
                )

        # Конвертируем обратно в PIL Image
        result = Image.fromarray(result_array)

        # Add contour labels if enabled (final pass - actual drawing)
        if CONTOUR_LABELS_ENABLED:
            result = _add_contour_labels(
                result,
                seed_polylines,
                levels,
                (0, 0, img_w, img_h),
                effective_seed_ds,
                ctx.center_lat_wgs,
                ctx.zoom,
                is_overlay=True,
                overlay_retina_factor=overlay_retina_factor,
                label_params=adaptive_params,
            )

        # Convert back to RGB if original was RGB
        if base_image.mode == 'RGB':
            result = result.convert('RGB')
    finally:
        sp.stop('Построение изолиний завершено')

    return result

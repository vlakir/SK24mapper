"""
NSU Optimizer processor — тепловая карта оптимальных позиций НСУ.

Для каждой потенциальной позиции НСУ вычисляет максимальную высоту полёта,
необходимую для поддержания прямой видимости со ВСЕМИ целевыми точками.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from geo.topography import latlng_to_pixel_xy, meters_per_pixel
from services.coordinate_transformer import sk42_raw_to_gk
from services.map_postprocessing import draw_control_point_triangle
from services.processors.radio_horizon import _load_dem, _load_topo
from services.radio_horizon import (
    compute_and_colorize_nsu_optimizer,
)
from shared.constants import (
    NSU_OPTIMIZER_POINT_COLORS,
    NSU_OPTIMIZER_USE_RETINA,
)
from shared.progress import LiveSpinner

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


def _sk42_to_dem_pixels(
    x_sk42: int,
    y_sk42: int,
    ctx: MapDownloadContext,
    full_eff_tile_px: int,
    ds_factor: int,
) -> tuple[int, int]:
    """Конвертация координаты СК-42 в пиксели DEM (row, col)."""
    gk_x, gk_y = sk42_raw_to_gk(x_sk42, y_sk42)  # easting, northing

    # GK → geodetic SK-42
    lng_sk42, lat_sk42 = ctx.t_sk42_from_gk.transform(gk_x, gk_y)
    # SK-42 → WGS-84
    lng_wgs, lat_wgs = ctx.t_sk42_to_wgs.transform(lng_sk42, lat_sk42)

    # WGS-84 → pixel coords
    px_x, px_y = latlng_to_pixel_xy(lat_wgs, lng_wgs, ctx.zoom)

    # Global tile origin
    cx_crop, cy_crop, _cw, _ch = ctx.crop_rect
    first_tile_x, first_tile_y = ctx.tiles[0]
    global_origin_x = first_tile_x * full_eff_tile_px
    global_origin_y = first_tile_y * full_eff_tile_px

    col_orig = int(px_x * (full_eff_tile_px / 256) - global_origin_x - cx_crop)
    row_orig = int(px_y * (full_eff_tile_px / 256) - global_origin_y - cy_crop)

    col = col_orig // ds_factor
    row = row_orig // ds_factor

    return row, col


def _draw_target_markers(
    image: Image.Image,
    target_points: list[tuple[int, int]],
    ctx: MapDownloadContext,
    full_eff_tile_px: int,
    ds_factor: int,
    dem_h: int,
    dem_w: int,
) -> None:
    """Отрисовка маркеров целевых точек через draw_control_point_triangle."""
    img_w, img_h = image.size
    mpp = meters_per_pixel(ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale)
    if mpp <= 0:
        return

    for i, (x_sk42, y_sk42) in enumerate(target_points):
        row, col = _sk42_to_dem_pixels(x_sk42, y_sk42, ctx, full_eff_tile_px, ds_factor)
        px = int(col * img_w / dem_w)
        py = int(row * img_h / dem_h)

        if 0 <= px < img_w and 0 <= py < img_h:
            color = NSU_OPTIMIZER_POINT_COLORS[i % len(NSU_OPTIMIZER_POINT_COLORS)]
            draw_control_point_triangle(
                image, px, py, mpp, color=color, adaptive_outline=True
            )


async def process_nsu_optimizer(ctx: MapDownloadContext) -> Image.Image:
    """
    Процессор карты оптимального размещения НСУ.

    1. Загружает DEM + топооснову
    2. Если есть целевые точки → вычисляет тепловую карту + маркеры
    3. Если точек нет → чистая топооснова (точки добавятся интерактивно)
    """
    use_retina = NSU_OPTIMIZER_USE_RETINA

    # Load DEM
    dem_full, ds_factor = await _load_dem(
        ctx,
        use_retina=use_retina,
        label='НСУ Optimizer',
    )

    full_eff_tile_px = 256 * (2 if use_retina else 1)

    # Load topo base
    topo_base = await _load_topo(
        ctx,
        use_retina=use_retina,
        label='НСУ Optimizer',
    )

    # Pixel size for DEM (accounting for downsampling)
    pixel_size_m = (
        meters_per_pixel(ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale) * ds_factor
    )

    # Cache for interactive recalculation
    dem_cache_size = (dem_full.shape[1], dem_full.shape[0])
    if topo_base.size != dem_cache_size:
        topo_for_cache = topo_base.resize(dem_cache_size, Image.Resampling.BILINEAR)
    else:
        topo_for_cache = topo_base.copy()

    ctx.nsu_cache_dem = dem_full
    ctx.nsu_cache_topo_base = topo_for_cache.convert('L').convert('RGBA')
    ctx.nsu_cache_pixel_size_m = pixel_size_m
    ctx.nsu_cache_crop_size = ctx.rh_cache_crop_size

    # Parse target points
    target_points = ctx.settings.nsu_target_points
    dem_h, dem_w = dem_full.shape

    if target_points:
        sp = LiveSpinner('Вычисление оптимальных позиций НСУ')
        sp.start()

        # Convert target points to DEM pixel coordinates
        target_rows = []
        target_cols = []
        valid_points = []
        for x_sk42, y_sk42 in target_points:
            try:
                row, col = _sk42_to_dem_pixels(
                    x_sk42, y_sk42, ctx, full_eff_tile_px, ds_factor
                )
                row = max(0, min(row, dem_h - 1))
                col = max(0, min(col, dem_w - 1))
                target_rows.append(row)
                target_cols.append(col)
                valid_points.append((x_sk42, y_sk42))
            except Exception:
                logger.warning(
                    'Не удалось преобразовать точку (%d, %d)', x_sk42, y_sk42
                )

        if valid_points:
            target_rows_arr = np.array(target_rows, dtype=np.int32)
            target_cols_arr = np.array(target_cols, dtype=np.int32)

            result = compute_and_colorize_nsu_optimizer(
                dem=dem_full,
                target_rows=target_rows_arr,
                target_cols=target_cols_arr,
                antenna_height_m=ctx.settings.nsu_antenna_height_m,
                pixel_size_m=pixel_size_m,
                max_height_m=ctx.settings.nsu_max_flight_height_m,
            )

            sp.stop('Покрытие НСУ вычислено')

            del dem_full
            gc.collect()

            # Resize if downsampled
            if ds_factor > 1:
                target_size = (ctx.crop_rect[2], ctx.crop_rect[3])
                result = result.resize(target_size, Image.Resampling.BILINEAR)

            if topo_base.size != result.size:
                topo_base = topo_base.resize(result.size, Image.Resampling.BILINEAR)

            # Blend
            blend_alpha = 1.0 - ctx.settings.nsu_overlay_alpha
            topo_base = topo_base.convert('L').convert('RGBA')
            result = result.convert('RGBA')
            ctx.nsu_cache_coverage = result.copy()
            result = Image.blend(topo_base, result, blend_alpha)

            # Draw target markers
            _draw_target_markers(
                result,
                valid_points,
                ctx,
                full_eff_tile_px,
                ds_factor,
                dem_h,
                dem_w,
            )

            del topo_base
            gc.collect()

            logger.info(
                'Карта НСУ Optimizer построена (%d целевых точек)', len(valid_points)
            )
            return result
        sp.stop('Нет валидных целевых точек')

    # No target points — return clean topo base
    del dem_full
    gc.collect()
    logger.info('Карта НСУ Optimizer: топооснова без покрытия (нет целевых точек)')
    return topo_base

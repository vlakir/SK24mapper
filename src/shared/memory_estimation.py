"""Memory estimation and safe zoom selection for OOM prevention."""

import logging

from geo.topography import estimate_crop_size_px
from shared.constants import (
    MEMORY_MIN_FREE_MB,
    MEMORY_SAFETY_RATIO,
    ROTATION_PAD_RATIO,
)

logger = logging.getLogger(__name__)

# Bytes per pixel for different data types
_BYTES_PER_PX_RGB = 3       # PIL RGB image
_BYTES_PER_PX_F32 = 4       # numpy float32 DEM
_MB = 1024 * 1024

# Padding увеличивает линейные размеры с каждой стороны
# → площадь растёт как (1 + 2*pad_ratio)²
_PAD_AREA_FACTOR = (1 + 2 * ROTATION_PAD_RATIO) ** 2  # ~1.30


def estimate_map_memory_mb(
    tiles_count: int,
    eff_tile_px: int,
    crop_w: int,
    crop_h: int,
    is_dem: bool = False,
    has_contours: bool = False,
) -> dict:
    """
    Estimate peak memory consumption for map building.

    Учитывает:
    - Тайлы в памяти (PIL RGB)
    - Результирующее изображение (с padding под поворот)
    - Промежуточные numpy-массивы для контуров (np.array ↔ Image.fromarray)
    - DEM для курсора (повторная загрузка + resize)
    - DEM-тайлы и DEM-canvas для elevation-режимов

    Returns dict with component breakdown and peak estimate in MB.
    """
    tile_pixels = eff_tile_px * eff_tile_px

    # Размер изображения с padding (до поворота/обрезки)
    padded_pixels = crop_w * crop_h * _PAD_AREA_FACTOR

    # Tile list memory (all tiles loaded before assembly)
    tiles_mb = tiles_count * tile_pixels * _BYTES_PER_PX_RGB / _MB

    # Result image with padding (PIL RGB)
    canvas_mb = padded_pixels * _BYTES_PER_PX_RGB / _MB

    # DEM array (if applicable)
    dem_mb = 0.0
    if is_dem:
        # DEM tiles (float32 per pixel)
        dem_tiles_mb = tiles_count * tile_pixels * _BYTES_PER_PX_F32 / _MB
        # DEM result (direct-crop, padded size)
        dem_canvas_mb = padded_pixels * _BYTES_PER_PX_F32 / _MB
        dem_mb = dem_tiles_mb + dem_canvas_mb

    # DEM for cursor elevation (loaded separately in postprocessing)
    # Загружается как отдельный набор DEM-тайлов + resize до target размера
    dem_cursor_mb = (
        tiles_count * tile_pixels * _BYTES_PER_PX_F32 / _MB  # DEM tiles
        + crop_w * crop_h * _BYTES_PER_PX_F32 / _MB          # resized grid
    )

    # Contours overlay: PIL ImageDraw.line() рисует in-place на PIL Image,
    # без промежуточного numpy-массива → дополнительная память не нужна.
    contour_mb = 0.0

    base_mb = tiles_mb + canvas_mb + dem_mb + dem_cursor_mb + contour_mb

    # Overhead: rotation buffer, intermediate PIL operations, Python objects
    overhead_mb = base_mb * 0.2

    peak_mb = base_mb + overhead_mb

    return {
        'tiles_mb': round(tiles_mb, 1),
        'canvas_mb': round(canvas_mb, 1),
        'dem_mb': round(dem_mb, 1),
        'dem_cursor_mb': round(dem_cursor_mb, 1),
        'contour_mb': round(contour_mb, 1),
        'overhead_mb': round(overhead_mb, 1),
        'total_mb': round(peak_mb, 1),
        'peak_mb': round(peak_mb, 1),
    }


def get_available_memory_mb() -> float:
    """Return available system memory in MB. Returns 0 if psutil unavailable."""
    try:
        import psutil
        return psutil.virtual_memory().available / _MB
    except Exception:
        return 0.0


def choose_safe_zoom(
    center_lat: float,
    width_m: float,
    height_m: float,
    desired_zoom: int,
    eff_scale: int,
    max_pixels: int,
    is_dem: bool = False,
    has_contours: bool = False,
    safety_ratio: float = MEMORY_SAFETY_RATIO,
    min_free_mb: float = MEMORY_MIN_FREE_MB,
) -> tuple[int, dict]:
    """
    Choose maximum zoom that fits both pixel limit and available RAM.

    Returns (zoom, info_dict) where info_dict contains memory estimates
    and whether zoom was reduced.
    """
    available_mb = get_available_memory_mb()
    memory_budget_mb = available_mb * safety_ratio - min_free_mb

    zoom = desired_zoom
    while zoom >= 0:
        crop_w, crop_h, total_pixels = estimate_crop_size_px(
            center_lat, width_m, height_m, zoom, eff_scale,
        )

        # Check pixel limit
        if total_pixels > max_pixels:
            zoom -= 1
            continue

        # Approximate tile count from crop size and tile size
        # (tiles cover more than crop due to padding, estimate ~1.3x)
        eff_tile_px = 256 * eff_scale
        tiles_x = (crop_w // eff_tile_px) + 3  # rough estimate with padding
        tiles_y = (crop_h // eff_tile_px) + 3
        tiles_count = tiles_x * tiles_y

        mem_est = estimate_map_memory_mb(
            tiles_count=tiles_count,
            eff_tile_px=eff_tile_px,
            crop_w=crop_w,
            crop_h=crop_h,
            is_dem=is_dem,
            has_contours=has_contours,
        )

        # If psutil unavailable (available_mb == 0), skip memory check
        if available_mb > 0 and mem_est['peak_mb'] > memory_budget_mb:
            logger.info(
                'Zoom %d: peak ~%.0f MB > budget ~%.0f MB, снижаем',
                zoom, mem_est['peak_mb'], memory_budget_mb,
            )
            zoom -= 1
            continue

        # This zoom fits
        info = {
            **mem_est,
            'desired_zoom': desired_zoom,
            'actual_zoom': zoom,
            'zoom_reduced': zoom < desired_zoom,
            'available_mb': round(available_mb, 0),
            'budget_mb': round(memory_budget_mb, 0),
            'crop_w': crop_w,
            'crop_h': crop_h,
            'tiles_count_est': tiles_count,
        }

        if zoom < desired_zoom:
            logger.warning(
                'Zoom снижен %d → %d для экономии RAM '
                '(peak ~%.0f MB, available ~%.0f MB)',
                desired_zoom, zoom, mem_est['peak_mb'], available_mb,
            )

        return zoom, info

    # Fallback: zoom 0
    return 0, {
        'desired_zoom': desired_zoom,
        'actual_zoom': 0,
        'zoom_reduced': desired_zoom > 0,
        'available_mb': round(available_mb, 0),
        'budget_mb': round(memory_budget_mb, 0),
        'peak_mb': 0,
    }

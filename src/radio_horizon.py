"""
Модуль расчёта радиогоризонта.

Вычисляет минимальную высоту БпЛА над поверхностью земли,
необходимую для обеспечения прямой радиовидимости с наземной станцией.
Учитывает рельеф местности и кривизну земной поверхности.

Оптимизирован с использованием numpy для экономии памяти и ускорения вычислений.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from constants import (
    EARTH_RADIUS_M,
    RADIO_HORIZON_COLOR_RAMP,
    RADIO_HORIZON_EMPTY_IMAGE_COLOR,
    RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX,
    RADIO_HORIZON_GRID_STEP,
    RADIO_HORIZON_GRID_STEP_LARGE,
    RADIO_HORIZON_GRID_STEP_MEDIUM,
    RADIO_HORIZON_GRID_STEP_SMALL,
    RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_LARGE,
    RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_MEDIUM,
    RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL,
    RADIO_HORIZON_INTERPOLATION_EDGE_EPSILON,
    RADIO_HORIZON_LEGEND_UNIT_LABEL,
    RADIO_HORIZON_LOS_MIN_DISTANCE_PX_SQ,
    RADIO_HORIZON_LOS_STEP_DIVISOR,
    RADIO_HORIZON_LOS_STEPS_MAX,
    RADIO_HORIZON_LOS_STEPS_MIN,
    RADIO_HORIZON_LUT_SIZE,
    RADIO_HORIZON_MAX_DEM_PIXELS,
    RADIO_HORIZON_MAX_ELEVATION_ANGLE_INIT,
    RADIO_HORIZON_MAX_ELEVATION_ANGLE_NO_DATA_THRESHOLD,
    RADIO_HORIZON_MAX_HEIGHT_M,
    RADIO_HORIZON_MIN_HEIGHT_M,
    RADIO_HORIZON_REFRACTION_K,
    RADIO_HORIZON_TARGET_HEIGHT_CLEARANCE_M,
    RADIO_HORIZON_UNREACHABLE_COLOR,
)


def downsample_dem(
    dem: np.ndarray,
    factor: int,
) -> np.ndarray:
    """
    Уменьшает разрешение DEM в factor раз путём усреднения блоков.

    Args:
        dem: Исходная матрица высот (numpy array, dtype=float32)
        factor: Коэффициент уменьшения (2, 4, 8, ...)

    Returns:
        Уменьшенная матрица высот (numpy array, dtype=float32)

    """
    if factor <= 1:
        return dem

    h, w = dem.shape

    # Обрезаем до размера, кратного factor
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor

    if new_h == 0 or new_w == 0:
        return dem

    # Reshape и усреднение по блокам
    cropped = dem[:new_h, :new_w]
    reshaped = cropped.reshape(new_h // factor, factor, new_w // factor, factor)
    return reshaped.mean(axis=(1, 3)).astype(np.float32)


def compute_downsample_factor(
    dem_h: int,
    dem_w: int,
    max_pixels: int = RADIO_HORIZON_MAX_DEM_PIXELS,
) -> int:
    """
    Вычисляет необходимый коэффициент даунсэмплинга.

    Returns:
        Коэффициент (1, 2, 4, 8, ...) или 1 если даунсэмплинг не нужен

    """
    total_pixels = dem_h * dem_w
    if total_pixels <= max_pixels:
        return 1

    # Находим минимальный коэффициент (степень двойки), при котором
    # количество пикселей будет <= max_pixels
    factor = 1
    while (dem_h // factor) * (dem_w // factor) > max_pixels:
        factor *= 2

    return factor


def _bilinear_interpolate_np(
    dem: np.ndarray,
    row: float,
    col: float,
) -> float:
    """
    Билинейная интерполяция высоты в дробной позиции DEM (numpy версия).

    Args:
        dem: Матрица высот (numpy array)
        row: Дробная строка (y)
        col: Дробный столбец (x)

    Returns:
        Интерполированная высота в метрах

    """
    h, w = dem.shape

    # Ограничиваем координаты
    row = max(0.0, min(row, h - RADIO_HORIZON_INTERPOLATION_EDGE_EPSILON))
    col = max(0.0, min(col, w - RADIO_HORIZON_INTERPOLATION_EDGE_EPSILON))

    r0 = int(row)
    c0 = int(col)
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)

    dr = row - r0
    dc = col - c0

    # Билинейная интерполяция
    v00 = dem[r0, c0]
    v01 = dem[r0, c1]
    v10 = dem[r1, c0]
    v11 = dem[r1, c1]

    v0 = v00 + (v01 - v00) * dc
    v1 = v10 + (v11 - v10) * dc
    return float(v0 + (v1 - v0) * dr)


def _trace_line_of_sight_np(
    dem: np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_abs_height: float,
    target_row: int,
    target_col: int,
    pixel_size_m: float,
    effective_earth_radius: float,
) -> float:
    """
    Трассировка луча от антенны к целевой точке (numpy версия).

    Returns:
        Минимальная высота БпЛА над поверхностью в целевой точке (метры).

    """
    # Расстояние в пикселях
    dx = target_col - antenna_col
    dy = target_row - antenna_row
    dist_px_sq = dx * dx + dy * dy

    if dist_px_sq < RADIO_HORIZON_LOS_MIN_DISTANCE_PX_SQ:
        return 0.0

    dist_px = math.sqrt(dist_px_sq)
    dist_m = dist_px * pixel_size_m

    # Высота поверхности в целевой точке
    target_surface_height = _bilinear_interpolate_np(dem, target_row, target_col)

    # Количество шагов (оптимизировано - не более 200 шагов)
    num_steps = min(
        RADIO_HORIZON_LOS_STEPS_MAX,
        max(RADIO_HORIZON_LOS_STEPS_MIN, int(dist_px / RADIO_HORIZON_LOS_STEP_DIVISOR)),
    )

    # Максимальный угол затенения
    max_elevation_angle = RADIO_HORIZON_MAX_ELEVATION_ANGLE_INIT

    # Трассируем луч от антенны к цели
    for i in range(1, num_steps):
        t = i / num_steps

        # Текущая позиция на луче
        curr_col = antenna_col + dx * t
        curr_row = antenna_row + dy * t

        # Расстояние от антенны до текущей точки
        curr_dist_m = dist_m * t

        # Поправка на кривизну Земли
        earth_curvature_drop = (curr_dist_m * curr_dist_m) / (
            2.0 * effective_earth_radius
        )

        # Высота поверхности + поправка на кривизну
        surface_height = _bilinear_interpolate_np(dem, curr_row, curr_col)
        effective_height = surface_height + earth_curvature_drop

        # Угол от антенны до текущей точки рельефа
        height_diff = effective_height - antenna_abs_height
        elevation_angle = height_diff / curr_dist_m
        max_elevation_angle = max(max_elevation_angle, elevation_angle)

    if max_elevation_angle < RADIO_HORIZON_MAX_ELEVATION_ANGLE_NO_DATA_THRESHOLD:
        return 0.0

    # Поправка на кривизну в целевой точке
    target_curvature_drop = (dist_m * dist_m) / (2.0 * effective_earth_radius)

    # Абсолютная высота линии визирования в целевой точке
    los_height_at_target = antenna_abs_height + max_elevation_angle * dist_m

    # Эффективная высота поверхности в целевой точке
    effective_target_surface = target_surface_height + target_curvature_drop

    # Минимальная высота БпЛА над поверхностью + запас
    min_uav_height = (
        los_height_at_target
        - effective_target_surface
        + RADIO_HORIZON_TARGET_HEIGHT_CLEARANCE_M
    )

    return max(0.0, min_uav_height)


def _build_color_lut(
    color_ramp: list[tuple[float, tuple[int, int, int]]],
    unreachable_color: tuple[int, int, int],
    lut_size: int = RADIO_HORIZON_LUT_SIZE,
) -> np.ndarray:
    """Строит LUT для быстрого маппинга значений в цвета (numpy версия)."""
    sorted_ramp = sorted(color_ramp, key=lambda x: x[0])
    lut = np.zeros((lut_size + 1, 3), dtype=np.uint8)

    for i in range(lut_size):
        t = i / (lut_size - 1)

        # Находим сегмент
        color = sorted_ramp[-1][1]
        for j in range(len(sorted_ramp) - 1):
            t0, c0 = sorted_ramp[j]
            t1, c1 = sorted_ramp[j + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    color = c0
                else:
                    ratio = (t - t0) / (t1 - t0)
                    color = (
                        int(c0[0] + (c1[0] - c0[0]) * ratio),
                        int(c0[1] + (c1[1] - c0[1]) * ratio),
                        int(c0[2] + (c1[2] - c0[2]) * ratio),
                    )
                break

        lut[i] = color

    # Цвет для недостижимых точек в конце
    lut[lut_size] = unreachable_color

    return lut


def compute_radio_horizon(
    dem: np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_height_m: float,
    pixel_size_m: float,
    earth_radius_m: float = EARTH_RADIUS_M,
    refraction_k: float = RADIO_HORIZON_REFRACTION_K,
    grid_step: int = RADIO_HORIZON_GRID_STEP,
) -> np.ndarray:
    """
    Вычисляет матрицу минимальных высот БпЛА для радиовидимости.

    Оптимизированная numpy версия: вычисляет на грубой сетке и интерполирует.

    Args:
        dem: Матрица высот (numpy array, dtype=float32)
        antenna_row: Индекс строки положения антенны в DEM
        antenna_col: Индекс столбца положения антенны в DEM
        antenna_height_m: Высота антенны над поверхностью, м
        pixel_size_m: Размер пикселя DEM в метрах
        earth_radius_m: Радиус Земли в метрах для расчёта кривизны
        refraction_k: Коэффициент рефракции для эффективного радиуса Земли
        grid_step: Шаг грубой сетки для ускорения расчёта

    Returns:
        Матрица минимальных высот БпЛА (numpy array, dtype=float32)

    """
    h, w = dem.shape

    if h == 0 or w == 0:
        return np.array([], dtype=np.float32)

    # Адаптивный шаг сетки для экономии памяти и ускорения
    total_pixels = h * w
    if total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_LARGE:  # > 8000x8000
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_LARGE)
    elif total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_MEDIUM:  # > 4000x4000
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_MEDIUM)
    elif total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL:  # > 2000x2000
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_SMALL)

    # Ограничиваем позицию антенны
    antenna_row = max(0, min(antenna_row, h - 1))
    antenna_col = max(0, min(antenna_col, w - 1))

    # Абсолютная высота антенны
    antenna_abs_height = float(dem[antenna_row, antenna_col]) + antenna_height_m

    # Эффективный радиус Земли
    effective_earth_radius = earth_radius_m * refraction_k

    # Размеры грубой сетки
    grid_h = (h + grid_step - 1) // grid_step
    grid_w = (w + grid_step - 1) // grid_step

    # Вычисляем значения на грубой сетке
    grid_values = np.zeros((grid_h, grid_w), dtype=np.float32)

    for gr in range(grid_h):
        row = min(gr * grid_step, h - 1)
        for gc in range(grid_w):
            col = min(gc * grid_step, w - 1)
            val = _trace_line_of_sight_np(
                dem,
                antenna_row,
                antenna_col,
                antenna_abs_height,
                row,
                col,
                pixel_size_m,
                effective_earth_radius,
            )
            grid_values[gr, gc] = val

    # Интерполируем до полного размера с помощью scipy или вручную
    # Используем простую билинейную интерполяцию
    result = np.zeros((h, w), dtype=np.float32)
    inv_grid_step = 1.0 / grid_step

    for row in range(h):
        gr0 = row // grid_step
        gr1 = min(gr0 + 1, grid_h - 1)
        dr = (row - gr0 * grid_step) * inv_grid_step

        for col in range(w):
            gc0 = col // grid_step
            gc1 = min(gc0 + 1, grid_w - 1)
            dc = (col - gc0 * grid_step) * inv_grid_step

            # Интерполяция
            v00 = grid_values[gr0, gc0]
            v01 = grid_values[gr0, gc1]
            v10 = grid_values[gr1, gc0]
            v11 = grid_values[gr1, gc1]

            v0 = v00 + (v01 - v00) * dc
            v1 = v10 + (v11 - v10) * dc
            result[row, col] = v0 + (v1 - v0) * dr

    return result


def colorize_radio_horizon(
    horizon_matrix: np.ndarray,
    max_height_m: float = RADIO_HORIZON_MAX_HEIGHT_M,
    color_ramp: list[tuple[float, tuple[int, int, int]]] | None = None,
    unreachable_color: tuple[int, int, int] = RADIO_HORIZON_UNREACHABLE_COLOR,
) -> Image.Image:
    """
    Преобразует матрицу высот радиогоризонта в цветное изображение.

    Оптимизированная numpy версия с векторизованными операциями.
    """
    if color_ramp is None:
        color_ramp = RADIO_HORIZON_COLOR_RAMP

    if horizon_matrix.size == 0:
        return Image.new(
            'RGB', RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX, RADIO_HORIZON_EMPTY_IMAGE_COLOR
        )

    h, w = horizon_matrix.shape

    if h == 0 or w == 0:
        return Image.new(
            'RGB', RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX, RADIO_HORIZON_EMPTY_IMAGE_COLOR
        )

    # Строим LUT
    lut_size = RADIO_HORIZON_LUT_SIZE
    lut = _build_color_lut(color_ramp, unreachable_color, lut_size)
    unreachable_idx = lut_size

    # Нормализация и индексация
    inv_max = (lut_size - 1) / max_height_m if max_height_m > 0 else 0.0

    # Вычисляем индексы в LUT для всех пикселей
    safe_horizon = np.nan_to_num(
        horizon_matrix,
        nan=0.0,
        posinf=max_height_m + 1,
        neginf=0.0,
    )
    indices = (safe_horizon * inv_max).astype(np.int32)
    indices = np.clip(indices, 0, lut_size - 1)

    # Помечаем точки за пределами максимальной высоты полёта серым цветом
    unreachable_mask = (horizon_matrix > max_height_m) | ~np.isfinite(horizon_matrix)
    indices[unreachable_mask] = unreachable_idx

    # Применяем LUT
    rgb = lut[indices]

    return Image.fromarray(rgb)


def compute_and_colorize_radio_horizon(
    dem: np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_height_m: float,
    pixel_size_m: float,
    earth_radius_m: float = EARTH_RADIUS_M,
    refraction_k: float = RADIO_HORIZON_REFRACTION_K,
    grid_step: int = RADIO_HORIZON_GRID_STEP,
    max_height_m: float = RADIO_HORIZON_MAX_HEIGHT_M,
    color_ramp: list[tuple[float, tuple[int, int, int]]] | None = None,
    unreachable_color: tuple[int, int, int] = RADIO_HORIZON_UNREACHABLE_COLOR,
) -> Image.Image:
    """
    Оптимизированная функция: вычисляет и раскрашивает радиогоризонт.

    Numpy версия с оптимизированным использованием памяти.
    """
    if color_ramp is None:
        color_ramp = RADIO_HORIZON_COLOR_RAMP

    if dem.size == 0:
        return Image.new(
            'RGB', RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX, RADIO_HORIZON_EMPTY_IMAGE_COLOR
        )

    h, w = dem.shape

    if h == 0 or w == 0:
        return Image.new(
            'RGB', RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX, RADIO_HORIZON_EMPTY_IMAGE_COLOR
        )

    # Адаптивный шаг сетки для экономии памяти и ускорения
    total_pixels = h * w
    if total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_LARGE:  # > 8000x8000
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_LARGE)
    elif total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_MEDIUM:  # > 4000x4000
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_MEDIUM)
    elif total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL:  # > 2000x2000
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_SMALL)

    # Ограничиваем позицию антенны
    antenna_row = max(0, min(antenna_row, h - 1))
    antenna_col = max(0, min(antenna_col, w - 1))

    # Абсолютная высота антенны
    antenna_abs_height = float(dem[antenna_row, antenna_col]) + antenna_height_m
    effective_earth_radius = earth_radius_m * refraction_k

    # Размеры грубой сетки
    grid_h = (h + grid_step - 1) // grid_step
    grid_w = (w + grid_step - 1) // grid_step

    # Вычисляем значения на грубой сетке
    grid_values = np.zeros((grid_h, grid_w), dtype=np.float32)

    for gr in range(grid_h):
        row = min(gr * grid_step, h - 1)
        for gc in range(grid_w):
            col = min(gc * grid_step, w - 1)
            val = _trace_line_of_sight_np(
                dem,
                antenna_row,
                antenna_col,
                antenna_abs_height,
                row,
                col,
                pixel_size_m,
                effective_earth_radius,
            )
            grid_values[gr, gc] = val

    # Строим LUT
    lut_size = RADIO_HORIZON_LUT_SIZE
    lut = _build_color_lut(color_ramp, unreachable_color, lut_size)
    unreachable_idx = lut_size
    inv_max = (lut_size - 1) / max_height_m if max_height_m > 0 else 0.0
    inv_grid_step = 1.0 / grid_step

    # Создаём результирующий массив RGB
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for row in range(h):
        gr0 = row // grid_step
        gr1 = min(gr0 + 1, grid_h - 1)
        dr = (row - gr0 * grid_step) * inv_grid_step

        for col in range(w):
            gc0 = col // grid_step
            gc1 = min(gc0 + 1, grid_w - 1)
            dc = (col - gc0 * grid_step) * inv_grid_step

            # Интерполяция
            v00 = grid_values[gr0, gc0]
            v01 = grid_values[gr0, gc1]
            v10 = grid_values[gr1, gc0]
            v11 = grid_values[gr1, gc1]

            v0 = v00 + (v01 - v00) * dc
            v1 = v10 + (v11 - v10) * dc
            val = v0 + (v1 - v0) * dr

            # Цвет: значения > max_height_m отображаются серым
            if val > max_height_m:
                idx = unreachable_idx
            else:
                idx = int(val * inv_max)
                idx = max(0, min(idx, lut_size - 1))

            rgb[row, col] = lut[idx]

    return Image.fromarray(rgb, mode='RGB')


def get_radio_horizon_legend_params(
    max_height_m: float = RADIO_HORIZON_MAX_HEIGHT_M,
) -> tuple[float, float, str]:
    """
    Возвращает параметры для легенды радиогоризонта.

    Returns:
        (min_value, max_value, unit_label)

    """
    return RADIO_HORIZON_MIN_HEIGHT_M, max_height_m, RADIO_HORIZON_LEGEND_UNIT_LABEL


# === Функции совместимости для работы со старым кодом (list[list[float]]) ===


def _list_to_numpy(dem_list: list[list[float]]) -> np.ndarray:
    """Конвертирует list[list[float]] в numpy array."""
    return np.array(dem_list, dtype=np.float32)


def _numpy_to_list(dem_np: np.ndarray) -> list[list[float]]:
    """Конвертирует numpy array в list[list[float]]."""
    return dem_np.tolist()


# Алиасы для обратной совместимости (принимают list, конвертируют внутри)
def _bilinear_interpolate(
    dem: list[list[float]] | np.ndarray,
    row: float,
    col: float,
    h: int,
    w: int,
) -> float:
    """Обёртка для совместимости с тестами."""
    if isinstance(dem, list):
        dem = _list_to_numpy(dem)
    return _bilinear_interpolate_np(dem, row, col)


def _trace_line_of_sight(
    dem: list[list[float]] | np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_abs_height: float,
    target_row: int,
    target_col: int,
    pixel_size_m: float,
    effective_earth_radius: float,
    h: int,
    w: int,
) -> float:
    """Обёртка для совместимости с тестами."""
    if isinstance(dem, list):
        dem = _list_to_numpy(dem)
    return _trace_line_of_sight_np(
        dem,
        antenna_row,
        antenna_col,
        antenna_abs_height,
        target_row,
        target_col,
        pixel_size_m,
        effective_earth_radius,
    )

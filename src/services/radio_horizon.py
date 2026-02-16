"""
Модуль расчёта радиогоризонта.

Вычисляет минимальную высоту БпЛА над поверхностью земли,
необходимую для обеспечения прямой радиовидимости с наземной станцией.
Учитывает рельеф местности и кривизну земной поверхности.

Оптимизирован с использованием numpy для экономии памяти и ускорения вычислений.
"""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np
from numba import njit, prange
from PIL import Image

from imaging.transforms import center_crop, rotate_keep_size
from shared.constants import (
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
    RADIO_HORIZON_LEGEND_UNIT_LABEL,
    RADIO_HORIZON_LUT_SIZE,
    RADIO_HORIZON_MAX_DEM_PIXELS,
    RADIO_HORIZON_MAX_HEIGHT_M,
    RADIO_HORIZON_MIN_HEIGHT_M,
    RADIO_HORIZON_REFRACTION_K,
    RADIO_HORIZON_UNREACHABLE_COLOR,
    ROTATION_EPSILON,
    UavHeightReference,
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


@njit(cache=True)
def _bilinear_interpolate_numba(
    dem: np.ndarray,
    row: float,
    col: float,
    h: int,
    w: int,
) -> float:
    """Билинейная интерполяция высоты (Numba JIT версия)."""
    edge_eps = 1.0001

    # Ограничиваем координаты
    if row < 0.0:
        row = 0.0
    elif row > h - edge_eps:
        row = h - edge_eps
    if col < 0.0:
        col = 0.0
    elif col > w - edge_eps:
        col = w - edge_eps

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
    return v0 + (v1 - v0) * dr


@njit(cache=True)
def _trace_line_of_sight_numba(
    dem: np.ndarray,
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
    """
    Трассировка луча от антенны к целевой точке (Numba JIT версия).

    Returns:
        Минимальная высота БпЛА над поверхностью в целевой точке (метры).

    """
    # Константы (инлайн для Numba)
    min_dist_px_sq = 4.0  # RADIO_HORIZON_LOS_MIN_DISTANCE_PX_SQ
    los_steps_max = 200  # RADIO_HORIZON_LOS_STEPS_MAX
    los_steps_min = 20  # RADIO_HORIZON_LOS_STEPS_MIN
    los_step_divisor = 4.0  # RADIO_HORIZON_LOS_STEP_DIVISOR
    max_elev_init = -1e9  # RADIO_HORIZON_MAX_ELEVATION_ANGLE_INIT
    no_data_threshold = -1e8  # RADIO_HORIZON_MAX_ELEVATION_ANGLE_NO_DATA_THRESHOLD
    clearance_m = 5.0  # RADIO_HORIZON_TARGET_HEIGHT_CLEARANCE_M

    # Расстояние в пикселях
    dx = target_col - antenna_col
    dy = target_row - antenna_row
    dist_px_sq = float(dx * dx + dy * dy)

    if dist_px_sq < min_dist_px_sq:
        return 0.0

    dist_px = math.sqrt(dist_px_sq)
    dist_m = dist_px * pixel_size_m

    # Высота поверхности в целевой точке
    target_surface_height = _bilinear_interpolate_numba(
        dem, float(target_row), float(target_col), h, w
    )

    # Количество шагов
    num_steps_calc = int(dist_px / los_step_divisor)
    num_steps_calc = max(num_steps_calc, los_steps_min)
    num_steps_calc = min(num_steps_calc, los_steps_max)
    num_steps = num_steps_calc

    # Максимальный угол затенения
    max_elevation_angle = max_elev_init

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
        surface_height = _bilinear_interpolate_numba(dem, curr_row, curr_col, h, w)
        effective_height = surface_height + earth_curvature_drop

        # Угол от антенны до текущей точки рельефа
        height_diff = effective_height - antenna_abs_height
        elevation_angle = height_diff / curr_dist_m
        max_elevation_angle = max(max_elevation_angle, elevation_angle)

    if max_elevation_angle < no_data_threshold:
        return 0.0

    # Поправка на кривизну в целевой точке
    target_curvature_drop = (dist_m * dist_m) / (2.0 * effective_earth_radius)

    # Абсолютная высота линии визирования в целевой точке
    los_height_at_target = antenna_abs_height + max_elevation_angle * dist_m

    # Эффективная высота поверхности в целевой точке
    effective_target_surface = target_surface_height + target_curvature_drop

    # Минимальная высота БпЛА над поверхностью + запас
    min_uav_height = los_height_at_target - effective_target_surface + clearance_m

    if min_uav_height < 0.0:
        return 0.0
    return min_uav_height


@njit(parallel=True, cache=True)
def _compute_grid_values_parallel(
    dem: np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_abs_height: float,
    pixel_size_m: float,
    effective_earth_radius: float,
    grid_step: int,
    h: int,
    w: int,
) -> np.ndarray:
    """Вычисляет значения на грубой сетке параллельно (Numba JIT)."""
    grid_h = (h + grid_step - 1) // grid_step
    grid_w = (w + grid_step - 1) // grid_step
    grid_values = np.zeros((grid_h, grid_w), dtype=np.float32)

    for gr in prange(grid_h):
        row = gr * grid_step
        row = min(row, h - 1)
        for gc in range(grid_w):
            col = gc * grid_step
            col = min(col, w - 1)
            grid_values[gr, gc] = _trace_line_of_sight_numba(
                dem,
                antenna_row,
                antenna_col,
                antenna_abs_height,
                row,
                col,
                pixel_size_m,
                effective_earth_radius,
                h,
                w,
            )
    return grid_values


@njit(parallel=True, cache=True)
def _compute_coverage_grid_parallel(
    dem: np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_abs_height: float,
    pixel_size_m: float,
    effective_earth_radius: float,
    grid_step: int,
    h: int,
    w: int,
    sector_enabled: int = 0,
    radar_azimuth_rad: float = 0.0,
    radar_half_sector_rad: float = math.pi,
    elevation_min_rad: float = 0.0,
    elevation_max_rad: float = 1.5707963,
    max_range_px: float = 1e9,
) -> np.ndarray:
    """
    Обобщённый Numba-kernel для расчёта покрытия (360° или сектор).

    При sector_enabled=False поведение идентично _compute_grid_values_parallel.
    При sector_enabled=True добавляются проверки: азимут, дальность, углы места.
    Пиксели вне сектора → NaN; мёртвая воронка (ниже elevation_min) → +inf.
    """
    grid_h = (h + grid_step - 1) // grid_step
    grid_w = (w + grid_step - 1) // grid_step
    grid_values = np.full((grid_h, grid_w), np.nan, dtype=np.float32)

    for gr in prange(grid_h):
        row = gr * grid_step
        row = min(row, h - 1)
        for gc in range(grid_w):
            col = gc * grid_step
            col = min(col, w - 1)

            if sector_enabled:
                # Вектор от антенны к целевой точке
                dy = -(row - antenna_row)  # ось Y инвертирована (вниз = юг)
                dx = col - antenna_col

                # Дальность в пикселях
                dist_px = math.sqrt(float(dx * dx + dy * dy))
                if dist_px > max_range_px:
                    continue  # Вне дальности → NaN

                if dist_px < 1.0:
                    grid_values[gr, gc] = 0.0
                    continue

                # Азимут к целевой точке (от севера, по часовой)
                target_azimuth = math.atan2(float(dx), float(dy))
                # Разница азимутов с учётом циклического перехода
                angle_diff = target_azimuth - radar_azimuth_rad
                # Нормализация в [-pi, pi]
                while angle_diff > math.pi:
                    angle_diff -= 2.0 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2.0 * math.pi

                if abs(angle_diff) > radar_half_sector_rad:
                    continue  # Вне сектора → NaN

                # LOS-трассировка
                los_val = _trace_line_of_sight_numba(
                    dem,
                    antenna_row,
                    antenna_col,
                    antenna_abs_height,
                    row,
                    col,
                    pixel_size_m,
                    effective_earth_radius,
                    h,
                    w,
                )

                # Проверка угла места
                dist_m = dist_px * pixel_size_m
                if dist_m > 0.0:
                    # Угол места до цели при минимальной высоте видимости
                    target_elev_angle = math.atan2(los_val, dist_m)
                    if target_elev_angle < elevation_min_rad:
                        # Мёртвая воронка: цель ниже минимального угла места
                        grid_values[gr, gc] = np.inf
                        continue
                    if target_elev_angle > elevation_max_rad:
                        # Выше максимального угла места
                        grid_values[gr, gc] = np.inf
                        continue

                grid_values[gr, gc] = los_val

            else:
                # Стандартный режим 360° (НСУ)
                grid_values[gr, gc] = _trace_line_of_sight_numba(
                    dem,
                    antenna_row,
                    antenna_col,
                    antenna_abs_height,
                    row,
                    col,
                    pixel_size_m,
                    effective_earth_radius,
                    h,
                    w,
                )

    return grid_values


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

    # Вычисляем значения на грубой сетке параллельно (Numba JIT)
    grid_values = _compute_grid_values_parallel(
        dem,
        antenna_row,
        antenna_col,
        antenna_abs_height,
        pixel_size_m,
        effective_earth_radius,
        grid_step,
        h,
        w,
    )

    # Интерполируем до полного размера с помощью cv2.resize (быстрее scipy.ndimage.zoom)
    result = cv2.resize(grid_values, (w, h), interpolation=cv2.INTER_LINEAR)

    return result.astype(np.float32)


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


def get_radio_horizon_legend_params(
    max_height_m: float = RADIO_HORIZON_MAX_HEIGHT_M,
) -> tuple[float, float, str]:
    """
    Возвращает параметры для легенды радиогоризонта.

    Returns:
        (min_value, max_value, unit_label)

    """
    return RADIO_HORIZON_MIN_HEIGHT_M, max_height_m, RADIO_HORIZON_LEGEND_UNIT_LABEL


# === Обобщённые функции покрытия (для RADIO_HORIZON и RADAR_COVERAGE) ===


def compute_and_colorize_coverage(
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
    uav_height_reference: UavHeightReference = UavHeightReference.GROUND,
    cp_elevation: float | None = None,
    *,
    # --- Параметры сектора (для РЛС) ---
    sector_enabled: bool = False,
    radar_azimuth_deg: float = 0.0,
    radar_sector_width_deg: float = 360.0,
    elevation_min_deg: float = 0.0,
    elevation_max_deg: float = 90.0,
    max_range_m: float = float('inf'),
    target_size: tuple[int, int] | None = None,
    target_height_min_m: float = 0.0,
) -> Image.Image:
    """
    Обобщённая функция: вычисляет и раскрашивает покрытие (360° или сектор).

    При sector_enabled=False — работает как НСУ 360° (без фильтрации по сектору).
    При sector_enabled=True — добавляется фильтрация по сектору/дальности/углам.

    Args:
        dem: Матрица высот (numpy array, dtype=float32).
        antenna_row: Индекс строки положения антенны в DEM.
        antenna_col: Индекс столбца положения антенны в DEM.
        antenna_height_m: Высота антенны над поверхностью, м.
        pixel_size_m: Размер пикселя DEM в метрах.
        earth_radius_m: Радиус Земли в метрах.
        refraction_k: Коэффициент рефракции.
        grid_step: Шаг грубой сетки для ускорения расчёта.
        max_height_m: Максимальная высота для цветовой шкалы.
        color_ramp: Градиент цветов для раскраски.
        unreachable_color: Цвет для недостижимых пикселей.
        uav_height_reference: Режим отсчёта высоты БпЛА.
        cp_elevation: Высота контрольной точки (для CONTROL_POINT).
        sector_enabled: Включить фильтрацию по сектору.
        radar_azimuth_deg: Азимут центра сектора РЛС (градусы).
        radar_sector_width_deg: Ширина сектора РЛС (градусы).
        elevation_min_deg: Минимальный угол места (градусы).
        elevation_max_deg: Максимальный угол места (градусы).
        max_range_m: Максимальная дальность обнаружения (метры).
        target_size: (width, height) финального изображения после resize.
            Если отличается от размера DEM, угловые границы сектора
            предкомпенсируются для неравномерного масштабирования.
        target_height_min_m: Минимальная высота цели для шкалы (метры).

    """
    if color_ramp is None:
        color_ramp = RADIO_HORIZON_COLOR_RAMP

    if dem.size == 0:
        return Image.new(
            'RGB', RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX, RADIO_HORIZON_EMPTY_IMAGE_COLOR
        )

    h, w = dem.shape

    # Адаптивный шаг сетки
    total_pixels = h * w
    if total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_LARGE:
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_LARGE)
    elif total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_MEDIUM:
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_MEDIUM)
    elif total_pixels > RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL:
        grid_step = max(grid_step, RADIO_HORIZON_GRID_STEP_SMALL)

    antenna_row = max(0, min(antenna_row, h - 1))
    antenna_col = max(0, min(antenna_col, w - 1))

    antenna_abs_height = float(dem[antenna_row, antenna_col]) + antenna_height_m
    effective_earth_radius = earth_radius_m * refraction_k

    # Вычисляем LOS на грубой сетке (всегда 360° — без сектора).
    # Сектор/дальность/углы накладываются ПОСЛЕ интерполяции на полном разрешении,
    # чтобы границы были пиксельно-точными (без блочных артефактов от grid_step).
    grid_values = _compute_grid_values_parallel(
        dem,
        antenna_row,
        antenna_col,
        antenna_abs_height,
        pixel_size_m,
        effective_earth_radius,
        grid_step,
        h,
        w,
    )

    # Интерполяция до полного размера (чистая, без NaN)
    full_values = cv2.resize(
        grid_values, (w, h), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    # Пересчёт высот в зависимости от режима отсчёта.
    # Для РЛС (sector_enabled) всегда используется GROUND — высота над поверхностью.
    if not sector_enabled:
        if uav_height_reference == UavHeightReference.CONTROL_POINT:
            if cp_elevation is None:
                cp_elevation = float(dem[antenna_row, antenna_col])
            full_values = full_values + dem.astype(np.float32) - cp_elevation
        elif uav_height_reference == UavHeightReference.SEA_LEVEL:
            full_values = full_values + dem.astype(np.float32)

    # Применение маски сектора на полном разрешении (пиксельно-точные границы)
    if sector_enabled:
        radar_azimuth_rad = math.radians(radar_azimuth_deg)
        radar_half_sector_rad = math.radians(radar_sector_width_deg / 2.0)
        elevation_min_rad = math.radians(elevation_min_deg)
        elevation_max_rad = math.radians(elevation_max_deg)
        max_range_px = max_range_m / pixel_size_m if pixel_size_m > 0 else 1e9

        rows = np.arange(h, dtype=np.float32)
        cols = np.arange(w, dtype=np.float32)
        col_grid, row_grid = np.meshgrid(cols, rows)

        dy = -(row_grid - antenna_row)  # Y inverted (up = north)
        dx = col_grid - antenna_col
        dist_px = np.sqrt(dx * dx + dy * dy)

        # 1) Outside range → unreachable
        outside_range = dist_px > max_range_px

        # 2) Outside azimuth sector → unreachable
        # Pre-compensate for non-uniform resize: if the image will be
        # resized to target_size with different aspect ratio, scale dx/dy
        # so that sector boundaries appear at correct geographic angles
        # in the final image.
        if target_size and (target_size[0] != w or target_size[1] != h):
            sx = target_size[0] / w  # horizontal scale factor
            sy = target_size[1] / h  # vertical scale factor
            target_az = np.arctan2(dx * sx, dy * sy)
        else:
            target_az = np.arctan2(dx, dy)  # atan2(east, north) = azimuth from north CW
        angle_diff = target_az - radar_azimuth_rad
        angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi
        outside_sector = np.abs(angle_diff) > radar_half_sector_rad

        outside = outside_range | outside_sector
        full_values[outside] = np.nan

        # 3) Elevation angle limits → adjust minimum detection height.
        # The target must satisfy BOTH:
        #   a) H >= full_values  (LOS-visible above terrain shadow)
        #   b) tan(elev_min) <= H/dist <= tan(elev_max)  (within radar beam)
        # So the effective minimum height = max(full_values, dist * tan(elev_min)).
        # If that exceeds dist * tan(elev_max), the pixel is a true dead zone
        # (too close — all visible heights exceed the beam ceiling).
        inside_valid = ~outside & (dist_px > 1.0)
        dist_m = dist_px * pixel_size_m

        tan_min = math.tan(elevation_min_rad)
        tan_max = math.tan(elevation_max_rad)

        h_beam_min = np.where(inside_valid, dist_m * tan_min, 0.0)
        h_beam_max = np.where(inside_valid, dist_m * tan_max, 1e9)

        # Raise minimum detection height to beam floor where needed
        fv = np.where(inside_valid, full_values, 0.0)
        effective_h = np.maximum(fv, h_beam_min)

        # Dead zone: minimum detectable height exceeds beam ceiling
        dead_zone = inside_valid & (effective_h > h_beam_max)
        full_values[dead_zone] = np.inf

        # Update non-dead pixels with effective height (may be raised by beam floor)
        beam_raised = inside_valid & ~dead_zone & (h_beam_min > fv)
        full_values[beam_raised] = h_beam_min[beam_raised]

    # Строим LUT и раскрашиваем
    lut_size = RADIO_HORIZON_LUT_SIZE
    lut = _build_color_lut(color_ramp, unreachable_color, lut_size)
    unreachable_idx = lut_size

    if sector_enabled and target_height_min_m > 0:
        # Radar mode: scale from min to max height
        h_range = max_height_m - target_height_min_m
        inv_range = (lut_size - 1) / h_range if h_range > 0 else 0.0
        clamped = np.clip(
            np.nan_to_num(full_values, nan=0.0, posinf=max_height_m + 1, neginf=0.0),
            target_height_min_m,
            max_height_m + 1,
        )
        indices = ((clamped - target_height_min_m) * inv_range).astype(np.int32)
        np.clip(indices, 0, lut_size - 1, out=indices)
    else:
        # НСУ: шкала [0, h_max] (без изменений)
        inv_max = (lut_size - 1) / max_height_m if max_height_m > 0 else 0.0
        indices = (
            np.nan_to_num(full_values, nan=0.0, posinf=max_height_m + 1, neginf=0.0)
            * inv_max
        ).astype(np.int32)
        np.clip(indices, 0, lut_size - 1, out=indices)

    # Недостижимые (>max_height, NaN, inf)
    unreachable_mask = (full_values > max_height_m) | ~np.isfinite(full_values)
    indices[unreachable_mask] = unreachable_idx

    rgb = lut[indices]
    return Image.fromarray(rgb)


def recompute_coverage_fast(
    dem: np.ndarray,
    new_antenna_row: int,
    new_antenna_col: int,
    antenna_height_m: float,
    pixel_size_m: float,
    topo_base: Image.Image,
    overlay_alpha: float,
    max_height_m: float = RADIO_HORIZON_MAX_HEIGHT_M,
    color_ramp: list[tuple[float, tuple[int, int, int]]] | None = None,
    unreachable_color: tuple[int, int, int] = RADIO_HORIZON_UNREACHABLE_COLOR,
    uav_height_reference: UavHeightReference = UavHeightReference.GROUND,
    earth_radius_m: float = EARTH_RADIUS_M,
    refraction_k: float = RADIO_HORIZON_REFRACTION_K,
    grid_step: int = RADIO_HORIZON_GRID_STEP,
    final_size: tuple[int, int] | None = None,
    crop_size: tuple[int, int] | None = None,
    rotation_deg: float = 0.0,
    *,
    # --- Параметры сектора ---
    sector_enabled: bool = False,
    radar_azimuth_deg: float = 0.0,
    radar_sector_width_deg: float = 360.0,
    elevation_min_deg: float = 0.0,
    elevation_max_deg: float = 90.0,
    max_range_m: float = float('inf'),
    target_height_min_m: float = 0.0,
) -> Image.Image | tuple[Image.Image, Image.Image]:
    """
    Быстрое перестроение покрытия с новой позицией (общая для НСУ и РЛС).

    Args:
        dem: Матрица высот (numpy array, dtype=float32).
        new_antenna_row: Индекс строки новой позиции антенны.
        new_antenna_col: Индекс столбца новой позиции антенны.
        antenna_height_m: Высота антенны над поверхностью, м.
        pixel_size_m: Размер пикселя DEM в метрах.
        topo_base: Базовое топографическое изображение для наложения.
        overlay_alpha: Прозрачность наложения покрытия.
        max_height_m: Максимальная высота для цветовой шкалы.
        color_ramp: Градиент цветов для раскраски.
        unreachable_color: Цвет для недостижимых пикселей.
        uav_height_reference: Режим отсчёта высоты БпЛА.
        earth_radius_m: Радиус Земли в метрах.
        refraction_k: Коэффициент рефракции.
        grid_step: Шаг грубой сетки для ускорения расчёта.
        final_size: (w, h) итогового изображения.
        crop_size: (w, h) оригинального DEM до даунсэмплинга (= crop_rect px).
        rotation_deg: Угол поворота карты (градусы, против часовой).
            Если задан вместе с crop_size, результат реплицирует пайплайн
            первого построения: resize -> rotate -> center crop, что обеспечивает
            совпадение системы координат с overlay_layer первого построения.
        sector_enabled: Включить фильтрацию по сектору.
        radar_azimuth_deg: Азимут центра сектора РЛС (градусы).
        radar_sector_width_deg: Ширина сектора РЛС (градусы).
        elevation_min_deg: Минимальный угол места (градусы).
        elevation_max_deg: Максимальный угол места (градусы).
        max_range_m: Максимальная дальность обнаружения (метры).
        target_height_min_m: Минимальная высота цели для шкалы (метры).

    """
    _logger = logging.getLogger(__name__)

    if color_ramp is None:
        color_ramp = RADIO_HORIZON_COLOR_RAMP

    h, w = dem.shape
    new_antenna_row = max(0, min(new_antenna_row, h - 1))
    new_antenna_col = max(0, min(new_antenna_col, w - 1))
    cp_elevation = float(dem[new_antenna_row, new_antenna_col])

    _logger.info(
        '    └─ DEM=(%d,%d), topo=%s, final=%s, crop=%s',
        w,
        h,
        topo_base.size,
        final_size,
        crop_size,
    )

    # When crop_size is available, skip target_size pre-compensation:
    # we'll crop the center + uniform resize instead, which preserves
    # circular sector boundaries and correct antenna position.
    use_crop = bool(crop_size and final_size)
    compute_target = None if use_crop else final_size

    step_start = time.monotonic()
    result = compute_and_colorize_coverage(
        dem=dem,
        antenna_row=new_antenna_row,
        antenna_col=new_antenna_col,
        antenna_height_m=antenna_height_m,
        pixel_size_m=pixel_size_m,
        earth_radius_m=earth_radius_m,
        refraction_k=refraction_k,
        grid_step=grid_step,
        max_height_m=max_height_m,
        color_ramp=color_ramp,
        unreachable_color=unreachable_color,
        uav_height_reference=uav_height_reference,
        cp_elevation=cp_elevation,
        sector_enabled=sector_enabled,
        radar_azimuth_deg=radar_azimuth_deg,
        radar_sector_width_deg=radar_sector_width_deg,
        elevation_min_deg=elevation_min_deg,
        elevation_max_deg=elevation_max_deg,
        max_range_m=max_range_m,
        target_size=compute_target,
        target_height_min_m=target_height_min_m,
    )
    step_elapsed = time.monotonic() - step_start
    _logger.info('    └─ compute_and_colorize_coverage: %.3f sec', step_elapsed)

    use_rotation = use_crop and abs(rotation_deg) > ROTATION_EPSILON

    if use_rotation and crop_size is not None and final_size is not None:
        # Replicate first-build pipeline: resize to crop_size → rotate → center crop.
        # This keeps the result in the same rotated coordinate system as the first
        # build, so overlay_layer, _rh_click_pos, and _dem_grid all remain aligned.
        step_start = time.monotonic()
        crop_w, crop_h = crop_size
        fw, fh = final_size

        # 1. Resize to crop_size (= original DEM before downsampling = crop_rect px)
        if result.size != (crop_w, crop_h):
            result = result.resize((crop_w, crop_h), Image.Resampling.BILINEAR)
        if topo_base.size != (crop_w, crop_h):
            topo_base = topo_base.resize((crop_w, crop_h), Image.Resampling.BILINEAR)

        step_elapsed = time.monotonic() - step_start
        _logger.info(
            '    └─ Resize to crop_size (%d×%d): %.3f sec', crop_w, crop_h, step_elapsed
        )

        # Blend and rotate+crop are done below (after converting to RGBA)
        result = result.convert('RGBA')
        blend_alpha = 1.0 - overlay_alpha
        blended = Image.blend(topo_base, result, blend_alpha)

        # 2. Rotate keeping same canvas size (same as rotate_keep_size in first build)
        step_start = time.monotonic()
        blended = rotate_keep_size(blended, rotation_deg, fill=(128, 128, 128))
        coverage_rotated = rotate_keep_size(result, rotation_deg, fill=(0, 0, 0, 0))
        step_elapsed = time.monotonic() - step_start
        _logger.info('    └─ Rotate by %.1f°: %.3f sec', rotation_deg, step_elapsed)

        # 3. Center crop to final_size
        step_start = time.monotonic()
        blended = center_crop(blended, fw, fh)
        coverage_rotated = center_crop(coverage_rotated, fw, fh)
        step_elapsed = time.monotonic() - step_start
        _logger.info(
            '    └─ Center crop to final (%d×%d): %.3f sec', fw, fh, step_elapsed
        )

        return blended, coverage_rotated

    if use_crop and crop_size is not None and final_size is not None:
        # No rotation but crop_size available: crop center + uniform resize.
        step_start = time.monotonic()
        crop_w, crop_h = crop_size
        fw, fh = final_size
        vis_w = round(fw * w / crop_w)
        vis_h = round(fh * h / crop_h)
        left = (w - vis_w) // 2
        top = (h - vis_h) // 2
        left = max(0, left)
        top = max(0, top)
        right = min(w, left + vis_w)
        bottom = min(h, top + vis_h)

        result = result.crop((left, top, right, bottom))
        if result.size != final_size:
            result = result.resize(final_size, Image.Resampling.BILINEAR)

        if topo_base.size == (w, h):
            topo_base = topo_base.crop((left, top, right, bottom))
            if topo_base.size != final_size:
                topo_base = topo_base.resize(final_size, Image.Resampling.BILINEAR)
        elif topo_base.size != final_size:
            topo_base = topo_base.resize(final_size, Image.Resampling.BILINEAR)

        step_elapsed = time.monotonic() - step_start
        _logger.info('    └─ Crop center + resize to final: %.3f sec', step_elapsed)
    else:
        if final_size and result.size != final_size:
            step_start = time.monotonic()
            result = result.resize(final_size, Image.Resampling.BILINEAR)
            step_elapsed = time.monotonic() - step_start
            _logger.info('    └─ Resize result to final size: %.3f sec', step_elapsed)

        if final_size and topo_base.size != final_size:
            step_start = time.monotonic()
            topo_base = topo_base.resize(final_size, Image.Resampling.BILINEAR)
            step_elapsed = time.monotonic() - step_start
            _logger.info(
                '    └─ Resize topo_base to final size: %.3f sec', step_elapsed
            )

    result = result.convert('RGBA')

    step_start = time.monotonic()
    blend_alpha = 1.0 - overlay_alpha
    blended = Image.blend(topo_base, result, blend_alpha)
    step_elapsed = time.monotonic() - step_start
    _logger.info('    └─ Blend with topo base: %.3f sec', step_elapsed)

    return blended, result


# === Функции совместимости для работы со старым кодом (list[list[float]]) ===


def _list_to_numpy(dem_list: list[list[float]]) -> np.ndarray:
    """Конвертирует list[list[float]] в numpy array."""
    return np.array(dem_list, dtype=np.float32)


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
    return _bilinear_interpolate_numba(dem, row, col, h, w)


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
    return _trace_line_of_sight_numba(
        dem,
        antenna_row,
        antenna_col,
        antenna_abs_height,
        target_row,
        target_col,
        pixel_size_m,
        effective_earth_radius,
        h,
        w,
    )

"""
Модуль построения изолиний (контуров).

Оптимизированная версия с использованием OpenCV для быстрого построения контуров
и субпиксельной интерполяцией для точности. Поддерживает параллельную обработку уровней.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from contours.seeds import simple_smooth_polyline
from shared.constants import CONTOUR_PARALLEL_WORKERS, CONTOUR_SEED_SMOOTHING

CONTOUR_POINTS_NDIM = 2
MIN_REFINEMENT_POINTS = 2
MIN_POLYLINE_POINTS = 3
GRAD_MAG_EPS = 1e-10


def _refine_contour_subpixel(
    contour: np.ndarray,
    dem: np.ndarray,
    level: float,
) -> list[tuple[float, float]]:
    """
    Уточняет позиции точек контура с субпиксельной точностью.

    Использует билинейную интерполяцию для нахождения точного положения
    изолинии между пикселями. Векторизованная реализация через NumPy.
    """
    if len(contour) < MIN_REFINEMENT_POINTS:
        return []

    h, w = dem.shape

    # Векторизованная обработка всех точек
    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)

    # Целочисленные индексы
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)

    # Маска точек на границе (не уточняем)
    boundary_mask = (x0 >= w - 1) | (y0 >= h - 1)

    # Ограничиваем индексы для безопасного доступа к массиву
    x0_safe = np.clip(x0, 0, w - 1)
    y0_safe = np.clip(y0, 0, h - 1)
    x1 = np.minimum(x0_safe + 1, w - 1)
    y1 = np.minimum(y0_safe + 1, h - 1)

    # Значения в углах ячеек (векторный доступ)
    v00 = dem[y0_safe, x0_safe]
    v10 = dem[y0_safe, x1]
    v01 = dem[y1, x0_safe]
    v11 = dem[y1, x1]

    # Субпиксельная коррекция по градиенту
    dx = (v10 - v00 + v11 - v01) * 0.5
    dy = (v01 - v00 + v11 - v10) * 0.5

    # Дробные части координат
    fx = x - x0
    fy = y - y0

    # Билинейная интерполяция
    v_interp = (
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )

    # Квадрат магнитуды градиента
    grad_mag_sq = dx * dx + dy * dy

    # Маска точек с достаточным градиентом
    valid_grad = grad_mag_sq > GRAD_MAG_EPS

    # Коррекция позиции (только где градиент достаточен)
    # Используем безопасное деление, чтобы избежать RuntimeWarning
    safe_grad_mag_sq = np.where(valid_grad, grad_mag_sq, 1.0)
    delta = np.where(valid_grad, (level - v_interp) / safe_grad_mag_sq, 0.0)
    x_new = x + delta * dx
    y_new = y + delta * dy

    # Ограничиваем коррекцию в пределах ячейки
    x_new = np.clip(x_new, x0, x1)
    y_new = np.clip(y_new, y0, y1)

    # Для точек на границе или с малым градиентом — оставляем исходные
    x_result = np.where(boundary_mask | ~valid_grad, x, x_new)
    y_result = np.where(boundary_mask | ~valid_grad, y, y_new)

    # Конвертируем в список кортежей
    return list(zip(x_result.tolist(), y_result.tolist(), strict=False))


def _build_contours_for_level(
    dem: np.ndarray,
    level: float,
    *,
    use_subpixel: bool = True,
) -> list[list[tuple[float, float]]]:
    """
    Строит контуры для одного уровня высоты с использованием OpenCV.

    Args:
        dem: Матрица высот (numpy array)
        level: Уровень высоты для изолинии
        use_subpixel: Использовать субпиксельную интерполяцию

    Returns:
        Список полилиний, каждая — список точек (x, y)

    """
    # Бинаризация по уровню
    binary = (dem >= level).astype(np.uint8) * 255

    # Поиск контуров через OpenCV (C++ реализация, очень быстро)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    polylines: list[list[tuple[float, float]]] = []

    for contour in contours:
        if len(contour) < MIN_POLYLINE_POINTS:
            continue

        # Убираем лишнее измерение: (N, 1, 2) -> (N, 2)
        pts = contour.squeeze()
        if pts.ndim != CONTOUR_POINTS_NDIM:
            continue

        if use_subpixel:
            # Субпиксельное уточнение позиций
            refined = _refine_contour_subpixel(pts, dem, level)
            if len(refined) >= MIN_POLYLINE_POINTS:
                if CONTOUR_SEED_SMOOTHING:
                    refined = simple_smooth_polyline(refined)
                if len(refined) >= MIN_POLYLINE_POINTS:
                    polylines.append(refined)
        else:
            # Просто конвертируем в список кортежей
            poly = [(float(p[0]), float(p[1])) for p in pts]
            if CONTOUR_SEED_SMOOTHING:
                poly = simple_smooth_polyline(poly)
            if len(poly) >= MIN_POLYLINE_POINTS:
                polylines.append(poly)

    return polylines


def build_seed_polylines(
    seed_dem: list[list[float]] | np.ndarray,
    levels: list[float],
    seed_h: int,
    seed_w: int,
) -> dict[int, list[list[tuple[float, float]]]]:
    """
    Строит полилинии изолиний с использованием OpenCV и параллельной обработки.

    Оптимизированная версия:
    - OpenCV cv2.findContours для быстрого построения (C++ реализация)
    - Субпиксельная интерполяция для точности
    - Параллельная обработка уровней через ThreadPoolExecutor

    Args:
        seed_dem: Матрица высот (уменьшенная для seed)
        levels: Список уровней высот для изолиний
        seed_h: Высота seed-матрицы
        seed_w: Ширина seed-матрицы

    Returns:
        Словарь {индекс_уровня: список_полилиний}, где каждая полилиния —
        список точек (x, y) в координатах seed-сетки.

    """
    _ = (seed_h, seed_w)
    # Конвертируем в numpy array если нужно
    if isinstance(seed_dem, list):
        dem_array = np.array(seed_dem, dtype=np.float32)
    else:
        dem_array = seed_dem.astype(np.float32)

    polylines_by_level: dict[int, list[list[tuple[float, float]]]] = {}

    # Определяем количество воркеров
    num_workers = min(
        CONTOUR_PARALLEL_WORKERS, max(1, os.cpu_count() or 1), len(levels)
    )

    if num_workers > 1 and len(levels) > 1:
        # Параллельная обработка уровней
        def process_level(
            li_level: tuple[int, float],
        ) -> tuple[int, list[list[tuple[float, float]]]]:
            li, level = li_level
            polys = _build_contours_for_level(dem_array, level, use_subpixel=True)
            return li, polys

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(process_level, enumerate(levels))
            polylines_by_level = dict(results)
    else:
        # Последовательная обработка для малого числа уровней
        for li, level in enumerate(levels):
            polylines_by_level[li] = _build_contours_for_level(
                dem_array, level, use_subpixel=True
            )

    return polylines_by_level

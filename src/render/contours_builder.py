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
    изолинии между пикселями.
    """
    if len(contour) < MIN_REFINEMENT_POINTS:
        return []

    h, w = dem.shape
    refined: list[tuple[float, float]] = []

    for pt in contour:
        x, y = float(pt[0]), float(pt[1])

        # Границы для интерполяции
        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        if x0 >= w - 1 or y0 >= h - 1:
            refined.append((x, y))
            continue

        # Значения в углах ячейки
        v00 = dem[y0, x0]
        v10 = dem[y0, x1]
        v01 = dem[y1, x0]
        v11 = dem[y1, x1]

        # Субпиксельная коррекция по градиенту
        dx = (v10 - v00 + v11 - v01) * 0.5
        dy = (v01 - v00 + v11 - v10) * 0.5

        # Текущее интерполированное значение
        fx = x - x0
        fy = y - y0
        v_interp = (
            v00 * (1 - fx) * (1 - fy)
            + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy
            + v11 * fx * fy
        )

        # Коррекция позиции
        grad_mag_sq = dx * dx + dy * dy
        if grad_mag_sq > GRAD_MAG_EPS:
            delta = (level - v_interp) / grad_mag_sq
            x_new = x + delta * dx
            y_new = y + delta * dy
            # Ограничиваем коррекцию в пределах ячейки
            x_new = max(x0, min(x1, x_new))
            y_new = max(y0, min(y1, y_new))
            refined.append((x_new, y_new))
        else:
            refined.append((x, y))

    return refined


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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import splev, splprep

from shared.constants import (
    CONTOUR_SMOOTHING_FACTOR,
    CONTOUR_SMOOTHING_ITERATIONS,
    CONTOUR_SMOOTHING_STRENGTH,
    MARCHING_SQUARES_CENTER_WEIGHT,
    MIN_GRID_SIZE,
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
    SEED_POLYLINE_QUANT_FACTOR,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class MSParams:
    center_weight: float = MARCHING_SQUARES_CENTER_WEIGHT


def _interp(p0: float, p1: float, v0: float, v1: float, level: float) -> float:
    if v1 == v0:
        return p0
    t = (level - v0) / (v1 - v0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return p0 + (p1 - p0) * t


def _mask_value(v00: float, v10: float, v11: float, v01: float, level: float) -> int:
    return (
        (1 if v00 >= level else 0)
        | ((1 if v10 >= level else 0) << 1)
        | ((1 if v11 >= level else 0) << 2)
        | ((1 if v01 >= level else 0) << 3)
    )


def _simple_segment(
    mask: int,
    x: int,
    y: int,
    xt: float,
    yl: float,
    yr: float,
    xb: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    candidates = (
        (MS_CONNECT_TOP_LEFT, ((xt, y), (x, yl))),
        (MS_CONNECT_TOP_RIGHT, ((xt, y), (x + 1, yr))),
        (MS_CONNECT_LEFT_RIGHT, ((x, yl), (x + 1, yr))),
        (MS_CONNECT_RIGHT_BOTTOM, ((x + 1, yr), (xb, y + 1))),
        (MS_CONNECT_TOP_BOTTOM, ((xt, y), (xb, y + 1))),
        (MS_CONNECT_LEFT_BOTTOM, ((x, yl), (xb, y + 1))),
    )
    for masks, segment in candidates:
        if mask in masks:
            return segment
    return None


def _ambiguous_segments(
    mask: int,
    v00: float,
    v10: float,
    v11: float,
    v01: float,
    x: int,
    y: int,
    xt: float,
    yl: float,
    yr: float,
    xb: float,
    level: float,
    center_weight: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    center = (v00 + v10 + v11 + v01) * center_weight
    choose_diag = center >= level
    if choose_diag:
        if mask == MS_MASK_TL_BR:
            return [((xt, y), (xb, y + 1))]
        return [((x, yl), (x + 1, yr))]
    if mask == MS_MASK_TL_BR:
        return [((x, yl), (x + 1, yr))]
    return [((xt, y), (xb, y + 1))]


def _cell_segments(
    v00: float,
    v10: float,
    v11: float,
    v01: float,
    x: int,
    y: int,
    level: float,
    center_weight: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    mask = _mask_value(v00, v10, v11, v01, level)
    if mask in MS_NO_CONTOUR_CASES:
        return []

    yl = _interp(y, y + 1, v00, v01, level)
    yr = _interp(y, y + 1, v10, v11, level)
    xt = _interp(x, x + 1, v00, v10, level)
    xb = _interp(x, x + 1, v01, v11, level)
    simple_segment = _simple_segment(mask, x, y, xt, yl, yr, xb)
    if simple_segment is not None:
        return [simple_segment]
    if mask in MS_AMBIGUOUS_CASES:
        return _ambiguous_segments(
            mask,
            v00,
            v10,
            v11,
            v01,
            x,
            y,
            xt,
            yl,
            yr,
            xb,
            level,
            center_weight,
        )
    return []


def _quant_key(p: tuple[float, float], quant_factor: float) -> tuple[int, int]:
    return round(p[0] * quant_factor), round(p[1] * quant_factor)


def _build_buckets(
    segs: list[tuple[tuple[float, float], tuple[float, float]]],
    quant_factor: float,
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], dict[int, bool]]:
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    unused: dict[int, bool] = {}
    for idx, (a, b) in enumerate(segs):
        unused[idx] = True
        buckets[_quant_key(a, quant_factor)].append((idx, 0))
        buckets[_quant_key(b, quant_factor)].append((idx, 1))
    return buckets, unused


def _pop_next_segment(
    segs: list[tuple[tuple[float, float], tuple[float, float]]],
    buckets: dict[tuple[int, int], list[tuple[int, int]]],
    unused: dict[int, bool],
    end: tuple[float, float],
    quant_factor: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    for idx, endpos in buckets.get(_quant_key(end, quant_factor), []):
        if idx in unused:
            a, b = segs[idx]
            unused.pop(idx, None)
            return (b, b) if endpos == 0 else (a, a)
    return None


def _chain_segments(
    segs: list[tuple[tuple[float, float], tuple[float, float]]],
    quant_factor: float,
) -> list[list[tuple[float, float]]]:
    if not segs:
        return []
    buckets, unused = _build_buckets(segs, quant_factor)
    polylines: list[list[tuple[float, float]]] = []
    for si in range(len(segs)):
        if si not in unused:
            continue
        unused.pop(si, None)
        a, b = segs[si]
        poly = [a, b]
        end = b
        while True:
            next_seg = _pop_next_segment(segs, buckets, unused, end, quant_factor)
            if next_seg is None:
                break
            end, point = next_seg
            poly.append(point)
        polylines.append(poly)
    return polylines


def _build_segments_for_level(
    dem_list: list[list[float]],
    seed_h: int,
    seed_w: int,
    level: float,
    ms: MSParams,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if seed_h < MIN_GRID_SIZE or seed_w < MIN_GRID_SIZE:
        return []
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for j in range(seed_h - 1):
        row0 = dem_list[j]
        row1 = dem_list[j + 1]
        for i in range(seed_w - 1):
            segs.extend(
                _cell_segments(
                    row0[i],
                    row0[i + 1],
                    row1[i + 1],
                    row1[i],
                    i,
                    j,
                    level,
                    ms.center_weight,
                )
            )
    return segs


def build_seed_polylines(
    dem: list[list[float]] | Iterable[Iterable[float]],
    levels: list[float],
    *,
    ms: MSParams | None = None,
    quant_factor: float = SEED_POLYLINE_QUANT_FACTOR,
) -> dict[int, list[list[tuple[float, float]]]]:
    """
    Build seed polylines using a simple marching-squares over a DEM grid.

    Args:
        dem: 2D grid of elevation values (row-major, dem[y][x]).
        levels: Contour levels (same order will be used for indexing in result).
        ms: Optional marching-squares parameters.
        quant_factor: Quantization for endpoint bucketing when chaining segments.

    Returns:
        Mapping: level_index -> list of polylines (each polyline is a list of
            (x,y) in DEM cell coords).

    """
    if ms is None:
        ms = MSParams()

    # Normalize dem to a list of lists for indexing
    dem_list: list[list[float]] = [list(row) for row in dem]

    seed_h = len(dem_list)
    seed_w = len(dem_list[0]) if seed_h else 0

    polylines_by_level: dict[int, list[list[tuple[float, float]]]] = {}

    for li, level in enumerate(levels):
        segs = _build_segments_for_level(dem_list, seed_h, seed_w, level, ms)
        polylines_by_level[li] = _chain_segments(segs, quant_factor)
    return polylines_by_level


def smooth_polyline(
    points: list[tuple[float, float]],
    smoothing_factor: int | None = None,
    smoothing_strength: float | None = None,
) -> list[tuple[float, float]]:
    """
    Сглаживание полилинии с помощью B-spline интерполяции.

    Args:
        points: Исходные точки полилинии
        smoothing_factor: Фактор увеличения количества точек
            (None = из constants)
        smoothing_strength: Параметр сглаживания s (None = из constants)

    Returns:
        Сглаженная полилиния с большим количеством точек

    """
    if len(points) < MIN_POINTS_FOR_SMOOTHING:
        return points

    if smoothing_factor is None:
        smoothing_factor = CONTOUR_SMOOTHING_FACTOR
    if smoothing_strength is None:
        smoothing_strength = CONTOUR_SMOOTHING_STRENGTH

    # Разделяем x и y координаты
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    # Параметр s контролирует агрессивность сглаживания:
    # s=0 → точная интерполяция через все точки
    # s>0 → аппроксимация с допустимым отклонением
    s_param = len(points) * smoothing_strength
    tck, u = splprep([x, y], s=s_param, k=min(3, len(points) - 1))

    # Генерация новых точек
    u_new = np.linspace(0, 1, len(points) * smoothing_factor)
    x_new, y_new = splev(u_new, tck)

    return list(zip(x_new, y_new, strict=False))


def simple_smooth_polyline(
    points: list[tuple[float, float]],
    iterations: int | None = None,
) -> list[tuple[float, float]]:
    """
    Простое сглаживание методом скользящего среднего (не требует scipy).

    Векторизованная реализация через NumPy для ускорения.
    """
    if len(points) < MIN_POINTS_FOR_SMOOTHING:
        return points

    if iterations is None:
        iterations = CONTOUR_SMOOTHING_ITERATIONS

    # Векторизованная реализация через NumPy
    arr = np.array(points, dtype=np.float64)
    for _ in range(iterations):
        # Скользящее среднее по 3 точкам (центральные точки)
        smoothed_middle = (arr[:-2] + arr[1:-1] + arr[2:]) / 3.0
        # Собираем результат: первая точка + сглаженные + последняя точка
        arr = np.vstack([arr[0:1], smoothed_middle, arr[-1:]])
    return [(float(x), float(y)) for x, y in arr]

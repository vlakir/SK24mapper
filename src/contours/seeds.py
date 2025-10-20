from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from constants import (
    MARCHING_SQUARES_CENTER_WEIGHT,
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

# Импорт с дефолтами для обратной совместимости
try:
    from constants import (
        CONTOUR_SMOOTHING_FACTOR,
        CONTOUR_SMOOTHING_ITERATIONS,
        CONTOUR_SMOOTHING_STRENGTH,
    )
except ImportError:
    CONTOUR_SMOOTHING_FACTOR = 5
    CONTOUR_SMOOTHING_STRENGTH = 2.0
    CONTOUR_SMOOTHING_ITERATIONS = 2

if TYPE_CHECKING:
    from collections.abc import Iterable

MIN_GRID_SIZE = 2


@dataclass(frozen=True)
class MSParams:
    center_weight: float = MARCHING_SQUARES_CENTER_WEIGHT


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
        Mapping: level_index -> list of polylines (each polyline is a list of (x,y) in DEM cell coords).

    """
    if ms is None:
        ms = MSParams()

    # Normalize dem to a list of lists for indexing
    dem_list: list[list[float]] = [list(row) for row in dem]

    seed_h = len(dem_list)
    seed_w = len(dem_list[0]) if seed_h else 0

    def interp(p0: float, p1: float, v0: float, v1: float, level: float) -> float:
        if v1 == v0:
            return p0
        t = (level - v0) / (v1 - v0)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        return p0 + (p1 - p0) * t

    polylines_by_level: dict[int, list[list[tuple[float, float]]]] = {}

    for li, level in enumerate(levels):
        segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
        if seed_h >= MIN_GRID_SIZE and seed_w >= MIN_GRID_SIZE:
            for j in range(seed_h - 1):
                row0 = dem_list[j]
                row1 = dem_list[j + 1]
                for i in range(seed_w - 1):
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
                        center = (v00 + v10 + v11 + v01) * ms.center_weight
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
        polylines: list[list[tuple[float, float]]] = []
        if segs:
            buckets = defaultdict(list)

            def key(p: tuple[float, float]) -> tuple[int, int]:
                qx = round(p[0] * quant_factor)
                qy = round(p[1] * quant_factor)
                return qx, qy

            unused: dict[int, bool] = {}
            for idx, (a, b) in enumerate(segs):
                unused[idx] = True
                buckets[key(a)].append((idx, 0))
                buckets[key(b)].append((idx, 1))
            for si in range(len(segs)):
                if si not in unused:
                    continue
                unused.pop(si, None)
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


def smooth_polyline(
    points: list[tuple[float, float]],
    smoothing_factor: int | None = None,
    smoothing_strength: float | None = None,
) -> list[tuple[float, float]]:
    """
    Сглаживание полилинии с помощью B-spline интерполяции.

    Args:
        points: Исходные точки полилинии
        smoothing_factor: Фактор увеличения количества точек (None = из constants)
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

    try:
        import numpy as np
        from scipy.interpolate import splev, splprep

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
    except Exception:
        # Fallback: если scipy недоступна, используем простое усреднение
        return simple_smooth_polyline(points, iterations=CONTOUR_SMOOTHING_ITERATIONS)


def simple_smooth_polyline(
    points: list[tuple[float, float]], iterations: int | None = None
) -> list[tuple[float, float]]:
    """Простое сглаживание методом скользящего среднего (не требует scipy)."""
    if len(points) < MIN_POINTS_FOR_SMOOTHING:
        return points

    if iterations is None:
        iterations = CONTOUR_SMOOTHING_ITERATIONS

    smoothed = points[:]
    for _ in range(iterations):
        new_points = [smoothed[0]]  # Сохраняем первую точку
        for i in range(1, len(smoothed) - 1):
            # Усредняем с соседними точками
            x = (smoothed[i - 1][0] + smoothed[i][0] + smoothed[i + 1][0]) / 3.0
            y = (smoothed[i - 1][1] + smoothed[i][1] + smoothed[i + 1][1]) / 3.0
            new_points.append((x, y))
        new_points.append(smoothed[-1])  # Сохраняем последнюю точку
        smoothed = new_points

    return smoothed

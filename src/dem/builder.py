"""
Модуль сборки DEM (Digital Elevation Model) из тайлов.

Содержит функции для сшивки DEM-тайлов и подготовки матрицы высот.
"""

from __future__ import annotations

import numpy as np

from geo.topography import assemble_dem, cache_dem_tile, get_cached_dem_tile


def build_dem_from_tiles(
    tiles_data: list[np.ndarray],
    tiles_x: int,
    tiles_y: int,
    eff_tile_px: int,
    crop_rect: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Собирает DEM из списка тайлов.

    Args:
        tiles_data: Список DEM-тайлов (numpy arrays)
        tiles_x: Количество тайлов по горизонтали
        tiles_y: Количество тайлов по вертикали
        eff_tile_px: Эффективный размер тайла в пикселях
        crop_rect: Прямоугольник обрезки (x, y, w, h)

    Returns:
        Собранная и обрезанная матрица высот (numpy array, float32)

    """
    return assemble_dem(tiles_data, tiles_x, tiles_y, eff_tile_px, crop_rect)


def downsample_dem_for_seed(
    dem: np.ndarray,
    downsample_factor: int,
) -> tuple[list[list[float]], int, int]:
    """
    Уменьшает DEM для построения seed-изолиний.

    Args:
        dem: Исходная матрица высот
        downsample_factor: Коэффициент уменьшения

    Returns:
        (seed_dem как list[list[float]], seed_h, seed_w)

    """
    h, w = dem.shape
    seed_h = (h + downsample_factor - 1) // downsample_factor
    seed_w = (w + downsample_factor - 1) // downsample_factor

    # Создаём уменьшенную версию
    seed_dem: list[list[float]] = []
    for j in range(seed_h):
        row: list[float] = []
        src_j = min(j * downsample_factor, h - 1)
        for i in range(seed_w):
            src_i = min(i * downsample_factor, w - 1)
            row.append(float(dem[src_j, src_i]))
        seed_dem.append(row)

    return seed_dem, seed_h, seed_w


def compute_elevation_levels(
    dem: np.ndarray,
    interval: float,
) -> tuple[list[float], float, float]:
    """
    Вычисляет уровни изолиний для DEM.

    Args:
        dem: Матрица высот
        interval: Интервал между изолиниями (метры)

    Returns:
        (список уровней, min высота, max высота)

    """
    import math

    # Находим min/max
    valid_mask = np.isfinite(dem)
    if not np.any(valid_mask):
        return [], 0.0, 0.0

    mn = float(np.min(dem[valid_mask]))
    mx = float(np.max(dem[valid_mask]))

    # Генерируем уровни
    start = math.floor(mn / interval) * interval
    end = math.ceil(mx / interval) * interval

    levels: list[float] = []
    k = 0
    v = start
    while v <= end:
        levels.append(v)
        k += 1
        v = start + k * interval

    return levels, mn, mx


class DEMCache:
    """Обёртка над кэшем DEM-тайлов."""

    @staticmethod
    def get(z: int, x: int, y: int) -> np.ndarray | None:
        """Получает тайл из кэша."""
        return get_cached_dem_tile(z, x, y)

    @staticmethod
    def put(z: int, x: int, y: int, dem: np.ndarray) -> None:
        """Добавляет тайл в кэш."""
        cache_dem_tile(z, x, y, dem)

    @staticmethod
    def get_or_fetch(
        z: int,
        x: int,
        y: int,
        fetch_func,
    ) -> np.ndarray:
        """
        Получает тайл из кэша или загружает через fetch_func.

        Args:
            z: Координата тайла по оси Z
            x: Координата тайла по оси X
            y: Координата тайла по оси Y
            fetch_func: Функция загрузки (должна вернуть np.ndarray)

        Returns:
            DEM-тайл

        """
        cached = get_cached_dem_tile(z, x, y)
        if cached is not None:
            return cached

        dem = fetch_func()
        cache_dem_tile(z, x, y, dem)
        return dem

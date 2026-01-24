"""
Модуль загрузки тайлов карт и DEM.

Содержит функции для асинхронной загрузки тайлов из различных источников.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from geo.topography import (
    async_fetch_terrain_rgb_tile,
    async_fetch_xyz_tile,
    decode_terrain_rgb_to_elevation_m,
)
from shared.constants import (
    DOWNLOAD_CONCURRENCY,
    ELEVATION_USE_RETINA,
    RADIO_HORIZON_USE_RETINA,
    TILE_SIZE,
    XYZ_TILE_SIZE,
    XYZ_USE_RETINA,
    MapType,
    map_type_to_style_id,
)

if TYPE_CHECKING:
    import aiohttp
    import numpy as np
    from PIL import Image

logger = logging.getLogger(__name__)


def get_effective_tile_size(map_type: MapType, use_retina: bool) -> int:
    """Возвращает эффективный размер тайла с учётом retina."""
    if map_type in (MapType.ELEVATION, MapType.RADIO_HORIZON):
        return TILE_SIZE * 2 if use_retina else TILE_SIZE
    return XYZ_TILE_SIZE * 2 if use_retina else XYZ_TILE_SIZE


def get_retina_flag(map_type: MapType) -> bool:
    """Возвращает флаг использования retina для типа карты."""
    if map_type == MapType.ELEVATION:
        return ELEVATION_USE_RETINA
    if map_type == MapType.RADIO_HORIZON:
        return RADIO_HORIZON_USE_RETINA
    return XYZ_USE_RETINA


def get_style_id(map_type: MapType) -> str:
    """Возвращает style_id для типа карты."""
    return map_type_to_style_id(map_type)


async def fetch_map_tile(
    client: aiohttp.ClientSession,
    api_key: str,
    z: int,
    x: int,
    y: int,
    map_type: MapType,
    use_retina: bool,
) -> Image.Image:
    """
    Загружает тайл карты указанного типа.

    Args:
        client: HTTP-клиент
        api_key: API-ключ Mapbox
        z: Координата тайла по оси Z
        x: Координата тайла по оси X
        y: Координата тайла по оси Y
        map_type: Тип карты
        use_retina: Использовать retina-разрешение

    Returns:
        PIL Image тайла

    """
    style_id = get_style_id(map_type)
    return await async_fetch_xyz_tile(
        client, api_key, z, x, y, style_id=style_id, use_retina=use_retina
    )


async def fetch_dem_tile(
    client: aiohttp.ClientSession,
    api_key: str,
    z: int,
    x: int,
    y: int,
    use_retina: bool,
) -> np.ndarray:
    """
    Загружает и декодирует DEM-тайл.

    Args:
        client: HTTP-клиент
        api_key: API-ключ Mapbox
        z: Координата тайла по оси Z
        x: Координата тайла по оси X
        y: Координата тайла по оси Y
        use_retina: Использовать retina-разрешение

    Returns:
        numpy array с высотами (float32)

    """
    img = await async_fetch_terrain_rgb_tile(
        client, api_key, z, x, y, use_retina=use_retina
    )
    return decode_terrain_rgb_to_elevation_m(img)


class TileFetcher:
    """Класс для пакетной загрузки тайлов с ограничением параллелизма."""

    def __init__(
        self,
        client: aiohttp.ClientSession,
        api_key: str,
        zoom: int,
        map_type: MapType,
        use_retina: bool,
        concurrency: int = DOWNLOAD_CONCURRENCY,
    ):
        self.client = client
        self.api_key = api_key
        self.zoom = zoom
        self.map_type = map_type
        self.use_retina = use_retina
        self.semaphore = asyncio.Semaphore(concurrency)

    async def fetch_tile(self, x: int, y: int) -> Image.Image:
        """Загружает один тайл карты."""
        async with self.semaphore:
            return await fetch_map_tile(
                self.client,
                self.api_key,
                self.zoom,
                x,
                y,
                self.map_type,
                self.use_retina,
            )

    async def fetch_dem(self, x: int, y: int) -> np.ndarray:
        """Загружает один DEM-тайл."""
        async with self.semaphore:
            return await fetch_dem_tile(
                self.client,
                self.api_key,
                self.zoom,
                x,
                y,
                self.use_retina,
            )

    async def fetch_tiles_batch(
        self,
        tile_coords: list[tuple[int, int]],
    ) -> list[Image.Image]:
        """Загружает пакет тайлов карты."""
        tasks = [self.fetch_tile(x, y) for x, y in tile_coords]
        return await asyncio.gather(*tasks)

    async def fetch_dem_batch(
        self,
        tile_coords: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        """Загружает пакет DEM-тайлов."""
        tasks = [self.fetch_dem(x, y) for x, y in tile_coords]
        return await asyncio.gather(*tasks)

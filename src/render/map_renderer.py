"""
Модуль рендеринга карты.

Содержит функции для отрисовки различных элементов карты.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from topography import colorize_dem_to_image

if TYPE_CHECKING:
    pass


def render_elevation_map(
    dem: np.ndarray,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
    min_range_m: float = 50.0,
) -> Image.Image:
    """
    Рендерит карту высот из DEM.

    Args:
        dem: Матрица высот (numpy array)
        p_lo: Нижний перцентиль для нормализации
        p_hi: Верхний перцентиль для нормализации
        min_range_m: Минимальный диапазон высот

    Returns:
        PIL Image с раскрашенной картой высот
    """
    return colorize_dem_to_image(dem, p_lo, p_hi, min_range_m)


def blend_images(
    base: Image.Image,
    overlay: Image.Image,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Смешивает два изображения с заданной прозрачностью.

    Args:
        base: Базовое изображение
        overlay: Накладываемое изображение
        alpha: Прозрачность overlay (0.0 - 1.0)

    Returns:
        Смешанное изображение
    """
    if base.size != overlay.size:
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)

    if base.mode != 'RGBA':
        base = base.convert('RGBA')
    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')

    return Image.blend(base, overlay, alpha)


def composite_with_mask(
    base: Image.Image,
    overlay: Image.Image,
    mask: Image.Image | None = None,
) -> Image.Image:
    """
    Накладывает изображение с маской.

    Args:
        base: Базовое изображение
        overlay: Накладываемое изображение
        mask: Маска прозрачности (L mode)

    Returns:
        Результат композиции
    """
    if base.size != overlay.size:
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)

    if mask is not None and mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.LANCZOS)

    result = base.copy()
    result.paste(overlay, mask=mask)
    return result


class MapRenderer:
    """Класс для рендеринга карты с различными слоями."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.layers: list[Image.Image] = []

    def add_layer(self, image: Image.Image) -> None:
        """Добавляет слой."""
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        self.layers.append(image)

    def render(self) -> Image.Image:
        """Рендерит все слои в одно изображение."""
        if not self.layers:
            return Image.new('RGB', (self.width, self.height), (255, 255, 255))

        result = self.layers[0].copy()
        if result.mode != 'RGBA':
            result = result.convert('RGBA')

        for layer in self.layers[1:]:
            if layer.mode != 'RGBA':
                layer = layer.convert('RGBA')
            result = Image.alpha_composite(result, layer)

        return result.convert('RGB')

    def clear(self) -> None:
        """Очищает все слои."""
        self.layers.clear()

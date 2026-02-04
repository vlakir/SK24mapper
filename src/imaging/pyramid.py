"""Пирамидальное превью для больших изображений.

Создаёт многоуровневую пирамиду изображения для эффективного
отображения в GUI без загрузки полного изображения в память.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from shared.constants import (
    PYRAMID_BASE_MAX_SIZE,
    PYRAMID_MIN_LEVEL_SIZE,
    PYRAMID_SCALE_FACTOR,
    PYRAMID_THRESHOLD_PIXELS,
    STREAMING_STRIP_HEIGHT,
)

if TYPE_CHECKING:
    from imaging.streaming import StreamingImage

logger = logging.getLogger(__name__)


class ImagePyramid:
    """
    Пирамида изображения для эффективного отображения больших карт.

    Хранит несколько уровней детализации изображения, от самого маленького
    (для обзора) до базового уровня (максимальная детализация для превью).

    Attributes:
        levels: Список PIL.Image от самого детального до самого маленького
        original_size: Оригинальный размер изображения (width, height)
        scale_to_original: Масштаб от базового уровня к оригиналу

    """

    def __init__(
        self,
        levels: list[Image.Image],
        original_size: tuple[int, int],
        scale_to_original: float,
    ) -> None:
        """
        Инициализирует пирамиду.

        Args:
            levels: Список уровней от детального к обзорному
            original_size: Оригинальный размер (width, height)
            scale_to_original: Масштаб базового уровня к оригиналу

        """
        self.levels = levels
        self.original_size = original_size
        self.scale_to_original = scale_to_original

    @property
    def base_level(self) -> Image.Image:
        """Возвращает базовый (самый детальный) уровень."""
        return self.levels[0] if self.levels else None

    @property
    def overview_level(self) -> Image.Image:
        """Возвращает обзорный (самый маленький) уровень."""
        return self.levels[-1] if self.levels else None

    @property
    def num_levels(self) -> int:
        """Количество уровней в пирамиде."""
        return len(self.levels)

    def get_level_for_scale(self, view_scale: float) -> tuple[Image.Image, float]:
        """
        Возвращает оптимальный уровень для заданного масштаба отображения.

        Args:
            view_scale: Текущий масштаб отображения (1.0 = 100%)

        Returns:
            Кортеж (изображение уровня, масштаб уровня относительно оригинала)

        """
        if not self.levels:
            return None, 1.0

        # Вычисляем эффективный масштаб относительно оригинала
        effective_scale = view_scale * self.scale_to_original

        # Находим подходящий уровень
        current_scale = self.scale_to_original
        for i, level in enumerate(self.levels):
            # Если текущий уровень достаточно детальный для отображения
            if current_scale <= effective_scale * 2:
                return level, current_scale
            # Переходим к следующему (менее детальному) уровню
            if i + 1 < len(self.levels):
                current_scale /= PYRAMID_SCALE_FACTOR

        # Возвращаем самый маленький уровень
        return self.levels[-1], current_scale

    def close(self) -> None:
        """Освобождает ресурсы всех уровней."""
        for level in self.levels:
            try:
                level.close()
            except Exception:
                pass
        self.levels.clear()


def build_pyramid_from_streaming(
    streaming_img: StreamingImage,
    max_base_size: int = PYRAMID_BASE_MAX_SIZE,
    min_level_size: int = PYRAMID_MIN_LEVEL_SIZE,
) -> ImagePyramid:
    """
    Строит пирамиду изображения из StreamingImage потоково.

    Алгоритм:
    1. Вычисляем масштаб для базового уровня
    2. Создаём базовый уровень, читая полосы из StreamingImage
    3. Строим остальные уровни уменьшением базового

    Args:
        streaming_img: Исходное изображение (StreamingImage)
        max_base_size: Максимальный размер базового уровня
        min_level_size: Минимальный размер уровня

    Returns:
        ImagePyramid с уровнями детализации

    """
    orig_w, orig_h = streaming_img.width, streaming_img.height
    logger.info(
        'Building pyramid from StreamingImage %dx%d, max_base=%d',
        orig_w,
        orig_h,
        max_base_size,
    )

    # Вычисляем масштаб для базового уровня
    max_dim = max(orig_w, orig_h)
    if max_dim <= max_base_size:
        # Изображение достаточно маленькое, масштаб 1:1
        scale = 1.0
        base_w, base_h = orig_w, orig_h
    else:
        scale = max_base_size / max_dim
        base_w = int(orig_w * scale)
        base_h = int(orig_h * scale)

    logger.info(
        'Pyramid base level: %dx%d (scale %.4f from %dx%d)',
        base_w,
        base_h,
        scale,
        orig_w,
        orig_h,
    )

    # Создаём базовый уровень потоково
    base_level = _create_base_level_streaming(streaming_img, base_w, base_h)

    # Строим остальные уровни пирамиды
    levels = [base_level]
    current_img = base_level
    current_w, current_h = base_w, base_h

    while min(current_w, current_h) > min_level_size:
        new_w = max(1, current_w // PYRAMID_SCALE_FACTOR)
        new_h = max(1, current_h // PYRAMID_SCALE_FACTOR)

        # Уменьшаем текущий уровень
        new_level = current_img.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )
        levels.append(new_level)

        current_img = new_level
        current_w, current_h = new_w, new_h

    logger.info(
        'Pyramid built: %d levels, sizes: %s',
        len(levels),
        [f'{lvl.width}x{lvl.height}' for lvl in levels],
    )

    return ImagePyramid(
        levels=levels,
        original_size=(orig_w, orig_h),
        scale_to_original=scale,
    )


def _create_base_level_streaming(
    streaming_img: StreamingImage,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """
    Создаёт базовый уровень пирамиды потоковым чтением.

    Читает изображение полосами и уменьшает каждую полосу,
    затем собирает результат.

    Args:
        streaming_img: Исходное изображение
        target_w: Целевая ширина
        target_h: Целевая высота

    Returns:
        PIL.Image базового уровня

    """
    orig_w, orig_h = streaming_img.width, streaming_img.height

    # Если масштаб 1:1, просто копируем
    if target_w == orig_w and target_h == orig_h:
        result = np.empty((orig_h, orig_w, 3), dtype=np.uint8)
        strip_h = STREAMING_STRIP_HEIGHT
        for y in range(0, orig_h, strip_h):
            strip = streaming_img.get_strip(y, strip_h)
            y_end = min(y + strip_h, orig_h)
            result[y:y_end] = strip
            del strip
        return Image.fromarray(result)

    # Вычисляем масштаб
    scale_y = target_h / orig_h
    scale_x = target_w / orig_w

    # Создаём результирующий массив
    result = np.empty((target_h, target_w, 3), dtype=np.uint8)

    # Размер полосы в исходном изображении
    src_strip_h = STREAMING_STRIP_HEIGHT
    # Размер полосы в результате
    dst_strip_h = max(1, int(src_strip_h * scale_y))

    # Обрабатываем полосами
    src_y = 0
    dst_y = 0

    while src_y < orig_h and dst_y < target_h:
        # Читаем полосу из исходного изображения
        actual_src_h = min(src_strip_h, orig_h - src_y)
        strip = streaming_img.get_strip(src_y, actual_src_h)

        # Вычисляем размер результирующей полосы
        dst_y_end = min(int((src_y + actual_src_h) * scale_y), target_h)
        actual_dst_h = dst_y_end - dst_y

        if actual_dst_h > 0:
            # Уменьшаем полосу
            resized_strip = cv2.resize(
                strip,
                (target_w, actual_dst_h),
                interpolation=cv2.INTER_AREA,
            )
            result[dst_y:dst_y_end] = resized_strip
            del resized_strip

        del strip
        src_y += actual_src_h
        dst_y = dst_y_end

    return Image.fromarray(result)


def build_pyramid_from_pil(
    pil_image: Image.Image,
    max_base_size: int = PYRAMID_BASE_MAX_SIZE,
    min_level_size: int = PYRAMID_MIN_LEVEL_SIZE,
) -> ImagePyramid:
    """
    Строит пирамиду из PIL.Image (для небольших изображений).

    Args:
        pil_image: Исходное изображение
        max_base_size: Максимальный размер базового уровня
        min_level_size: Минимальный размер уровня

    Returns:
        ImagePyramid с уровнями детализации

    """
    orig_w, orig_h = pil_image.size

    # Вычисляем масштаб для базового уровня
    max_dim = max(orig_w, orig_h)
    if max_dim <= max_base_size:
        scale = 1.0
        base_level = pil_image.copy()
    else:
        scale = max_base_size / max_dim
        base_w = int(orig_w * scale)
        base_h = int(orig_h * scale)
        base_level = pil_image.resize((base_w, base_h), Image.Resampling.LANCZOS)

    # Строим остальные уровни
    levels = [base_level]
    current_img = base_level
    current_w, current_h = base_level.size

    while min(current_w, current_h) > min_level_size:
        new_w = max(1, current_w // PYRAMID_SCALE_FACTOR)
        new_h = max(1, current_h // PYRAMID_SCALE_FACTOR)
        new_level = current_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        levels.append(new_level)
        current_img = new_level
        current_w, current_h = new_w, new_h

    return ImagePyramid(
        levels=levels,
        original_size=(orig_w, orig_h),
        scale_to_original=scale,
    )


def should_use_pyramid(width: int, height: int) -> bool:
    """
    Определяет, нужно ли использовать пирамиду для изображения.

    Args:
        width: Ширина изображения
        height: Высота изображения

    Returns:
        True если изображение достаточно большое для пирамиды

    """
    return width * height > PYRAMID_THRESHOLD_PIXELS

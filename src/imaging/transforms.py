"""Image transformation utilities - rotation, cropping, masking."""

import cv2
import numpy as np
from PIL import Image

ROTATE_ANGLE_EPS = 1e-6


def rotate_keep_size(
    img: Image.Image,
    angle_deg: float,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Поворачивает изображение на заданный угол (против часовой стрелки).

    Сохраняет исходный размер (обрезая углы).
    Использует OpenCV для ускорения операции.

    Args:
        img: Исходное изображение.
        angle_deg: Угол поворота в градусах (положительный — против часовой).
        fill: Цвет заливки для углов (по умолчанию белый).

    Returns:
        Повёрнутое изображение того же размера.

    """
    if abs(angle_deg) < ROTATE_ANGLE_EPS:
        return img.copy()

    arr = np.array(img)
    h, w = arr.shape[:2]
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # OpenCV использует BGR, но fill передаётся как RGB — для borderValue
    # порядок не важен, так как мы конвертируем обратно в тот же формат
    rotated = cv2.warpAffine(
        arr,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )

    return Image.fromarray(rotated)


def center_crop(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Обрезает изображение по центру до заданного размера."""
    w, h = img.size
    left = (w - out_w) // 2
    top = (h - out_h) // 2
    return img.crop((left, top, left + out_w, top + out_h))


def apply_white_mask(img: Image.Image, opacity: float) -> Image.Image:
    """
    Накладывает полупрозрачную белую маску на изображение.

    Args:
        img: Исходное изображение (RGB).
        opacity: Непрозрачность маски (0.0 — без маски, 1.0 — полностью белое).

    Returns:
        Изображение с наложенной маской.

    """
    if opacity <= 0:
        return img
    white = Image.new('RGB', img.size, (255, 255, 255))
    return Image.blend(img, white, opacity)

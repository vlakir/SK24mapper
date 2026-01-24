"""Image transformation utilities - rotation, cropping, masking."""
from PIL import Image


def rotate_keep_size(
    img: Image.Image,
    angle_deg: float,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Поворачивает изображение на заданный угол (против часовой стрелки),
    сохраняя исходный размер (обрезая углы).

    Args:
        img: Исходное изображение.
        angle_deg: Угол поворота в градусах (положительный — против часовой).
        fill: Цвет заливки для углов (по умолчанию белый).

    Returns:
        Повёрнутое изображение того же размера.
    """
    if abs(angle_deg) < 1e-6:
        return img.copy()

    rotated = img.rotate(
        angle_deg,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )
    return rotated


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

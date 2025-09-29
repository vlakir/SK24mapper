from __future__ import annotations

from typing import TYPE_CHECKING, Any

from image import rotate_keep_size
from image_io import build_save_kwargs as _build_save_kwargs
from image_io import save_jpeg as _save_jpeg

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image


def compose_final_image(
    base_img: Image.Image,
    *,
    rotate_deg: float | None = None,
    center_cross=None,
    grid=None,
) -> Image.Image:
    """
    Thin facade for final composition steps; currently only rotation is applied here.
    Other overlays are still applied in service.py using existing helpers.
    """
    img = base_img
    if rotate_deg is not None:
        img = rotate_keep_size(img, rotate_deg, fill=(255, 255, 255))
    return img


def save_image(
    img: Image.Image, path: Path, *, save_kwargs: dict[str, Any] | None = None
) -> None:
    if save_kwargs is None:
        # Fallback: try to infer default save kwargs
        from profiles import MapSettings  # type: ignore

        try:
            save_kwargs = _build_save_kwargs(path, MapSettings())  # type: ignore[arg-type]
        except Exception:
            save_kwargs = {}
    _save_jpeg(img, path, save_kwargs)

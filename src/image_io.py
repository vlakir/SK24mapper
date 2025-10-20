from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image


def build_save_kwargs(out_path: Path, quality: int = 95) -> dict[str, Any]:
    """Build PIL.Image.save kwargs for JPEG based on quality value."""
    q = max(10, min(100, int(quality)))
    return {
        'format': 'JPEG',
        'quality': q,
        'subsampling': 0,
        'optimize': True,
        'progressive': True,
        'exif': b'',
    }


def save_jpeg(img: Image.Image, out_path: Path, save_kwargs: dict[str, Any]) -> None:
    """Save an image to JPEG path and fsync to ensure data is written."""
    # Use temporary RGB image for saving and close it afterwards
    tmp_rgb = img.convert('RGB') if img.mode != 'RGB' else img.copy()
    try:
        tmp_rgb.save(out_path, **save_kwargs)
    finally:
        with contextlib.suppress(Exception):
            tmp_rgb.close()
    # Ensure data is written to disk
    fd = os.open(out_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

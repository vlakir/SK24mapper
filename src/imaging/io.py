from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from PIL import Image


def build_save_kwargs(_out_path: Path, quality: int = 95) -> dict[str, Any]:
    """
    Build PIL.Image.save kwargs for JPEG based on quality value.

    Lean defaults: stock subsampling (4:2:0), no `optimize`, no
    `progressive`. The previous production set (`subsampling=0` for full
    chroma, plus optimize+progressive) cost ~1 s per save on 8404² and
    only shrank the file by ~10-15 %. With default chroma subsampling
    and a single pass we go from ~1033 ms → ~125 ms on the same image
    (libjpeg-turbo via PIL is plenty fast); the size delta is invisible
    at q=95 on map imagery where chroma is already smooth.
    """
    q = max(10, min(100, int(quality)))
    return {
        'format': 'JPEG',
        'quality': q,
        'exif': b'',
    }


def save_jpeg(img: Image.Image, out_path: Path, save_kwargs: dict[str, Any]) -> None:
    """Save an image to JPEG path and fsync to ensure data is written.

    For an RGB image we save it directly — PIL.Image.save() reads pixels
    but does not mutate the source, so the previous defensive `.copy()`
    was pure waste (~600 ms memcpy at 16k² SATELLITE/HYBRID). Only
    convert to a temporary if the source is in a non-RGB mode.
    """
    if img.mode == 'RGB':
        img.save(out_path, **save_kwargs)
    else:
        tmp_rgb = img.convert('RGB')
        try:
            tmp_rgb.save(out_path, **save_kwargs)
        finally:
            with contextlib.suppress(Exception):
                tmp_rgb.close()
    # Ensure data is written to disk (may fail on some Windows configurations)
    try:
        with out_path.open('rb') as f:
            os.fsync(f.fileno())
    except OSError:
        pass  # fsync not supported on this file system

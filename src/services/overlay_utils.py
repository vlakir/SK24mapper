"""
Overlay utilities for contour and elevation map compositing.

This module contains helper functions for creating and compositing
overlay layers on top of base map images.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from PIL import Image as PILImage

if TYPE_CHECKING:
    from PIL import Image

from PIL import ImageDraw

logger = logging.getLogger(__name__)


def composite_overlay_on_base(
    base: Image.Image,
    overlay: Image.Image,
    *,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    """
    Composite an RGBA overlay onto a base image.

    Args:
        base: Base image (RGB or RGBA)
        overlay: Overlay image (RGBA with transparency)
        target_size: Optional target size to resize overlay before compositing

    Returns:
        Composited RGB image

    """
    # Resize overlay if needed
    if target_size and overlay.size != target_size:
        logger.info(
            'Масштабирование overlay до размера базы: %s -> %s',
            overlay.size,
            target_size,
        )
        overlay = overlay.resize(target_size, PILImage.Resampling.BICUBIC)

    # Convert base to RGBA for compositing
    base_rgba = base.convert('RGBA')

    # Apply alpha composite
    base_rgba.alpha_composite(overlay)

    # Convert back to RGB
    result = base_rgba.convert('RGB')

    # Cleanup
    with contextlib.suppress(Exception):
        base_rgba.close()

    return result


def blend_with_grayscale_base(
    base: Image.Image,
    color_overlay: Image.Image,
    alpha: float = 0.7,
) -> Image.Image:
    """
    Blend a color overlay with a grayscale version of the base image.

    Used for radio horizon maps where the topographic base is converted
    to grayscale before blending with the colored horizon map.

    Args:
        base: Base image (will be converted to grayscale)
        color_overlay: Color overlay image
        alpha: Blend factor (0.0 = only base, 1.0 = only overlay)

    Returns:
        Blended RGBA image

    """
    # Ensure same size
    if base.size != color_overlay.size:
        base = base.resize(color_overlay.size, PILImage.Resampling.BILINEAR)

    # Convert base to grayscale then RGBA
    topo_gray = base.convert('L').convert('RGBA')
    overlay_rgba = color_overlay.convert('RGBA')

    # Blend
    result = PILImage.blend(topo_gray, overlay_rgba, alpha)

    # Cleanup
    with contextlib.suppress(Exception):
        topo_gray.close()
    with contextlib.suppress(Exception):
        overlay_rgba.close()

    return result


def create_contour_gap_at_labels(
    overlay: Image.Image,
    label_bboxes: list[tuple[int, int, int, int]],
    gap_padding: int,
) -> None:
    """
    Create gaps in contour lines at label positions.

    Modifies the overlay image in-place by making areas around labels transparent.

    Args:
        overlay: RGBA overlay image to modify
        label_bboxes: List of (x0, y0, x1, y1) bounding boxes for labels
        gap_padding: Padding around each label bbox in pixels

    """
    if not label_bboxes:
        return

    draw = ImageDraw.Draw(overlay)
    for bbox in label_bboxes:
        x0, y0, x1, y1 = bbox
        gap_area = (
            max(0, x0 - gap_padding),
            max(0, y0 - gap_padding),
            min(overlay.width, x1 + gap_padding),
            min(overlay.height, y1 + gap_padding),
        )
        # Make area transparent
        draw.rectangle(gap_area, fill=(0, 0, 0, 0))

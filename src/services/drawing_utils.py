"""
Drawing utilities for map markers and annotations.

This module contains helper functions for drawing control points,
markers, and labels on map images.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def draw_control_point_marker(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    size_px: int,
    color: tuple[int, int, int],
    outline_color: tuple[int, int, int] = (0, 0, 0),
) -> int:
    """
    Draw a triangular control point marker.

    Args:
        draw: PIL ImageDraw object
        center: (x, y) center position in pixels
        size_px: Size of the triangle in pixels
        color: Fill color (R, G, B)
        outline_color: Outline color (R, G, B)

    Returns:
        Y coordinate of the marker bottom (for label positioning)
    """
    cx, cy = center
    h_tri = size_px
    half_base = size_px // 2

    # Triangle vertices: top vertex, bottom-left, bottom-right
    p1 = (cx, cy - h_tri // 2)  # top
    p2 = (cx - half_base, cy + h_tri // 2)  # bottom-left
    p3 = (cx + half_base, cy + h_tri // 2)  # bottom-right

    draw.polygon([p1, p2, p3], fill=color, outline=outline_color)

    return cy + h_tri // 2


def draw_center_cross(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    length_px: int,
    width_px: int,
    color: tuple[int, int, int],
) -> None:
    """
    Draw a cross marker at the center of the map.

    Args:
        draw: PIL ImageDraw object
        center: (x, y) center position in pixels
        length_px: Length of each arm of the cross
        width_px: Line width in pixels
        color: Line color (R, G, B)
    """
    cx, cy = center
    half_len = length_px // 2

    # Horizontal line
    draw.line(
        [(cx - half_len, cy), (cx + half_len, cy)],
        fill=color,
        width=width_px,
    )
    # Vertical line
    draw.line(
        [(cx, cy - half_len), (cx, cy + half_len)],
        fill=color,
        width=width_px,
    )


def compute_rotated_position(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    rotation_deg: float,
) -> tuple[float, float]:
    """
    Compute position after rotation around a center point.

    Args:
        x: Original X coordinate
        y: Original Y coordinate
        center_x: Rotation center X
        center_y: Rotation center Y
        rotation_deg: Rotation angle in degrees (negative = clockwise)

    Returns:
        (new_x, new_y) rotated coordinates
    """
    rotation_rad = math.radians(-rotation_deg)
    cos_rot = math.cos(rotation_rad)
    sin_rot = math.sin(rotation_rad)

    dx = x - center_x
    dy = y - center_y

    new_x = center_x + dx * cos_rot - dy * sin_rot
    new_y = center_y + dx * sin_rot + dy * cos_rot

    return new_x, new_y

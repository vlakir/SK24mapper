"""Post-processing utilities for map images (grid, legend, markers, rotation)."""

import math
from collections.abc import Callable

from PIL import Image, ImageDraw

from shared.constants import (
    CENTER_CROSS_LENGTH_M,
    CENTER_CROSS_LINE_WIDTH_M,
    CONTROL_POINT_COLOR,
    CONTROL_POINT_SIZE_M,
)


def draw_center_cross_on_image(
    img: Image.Image,
    meters_per_px: float,
) -> None:
    """
    Draw a red cross marker at the center of the image.

    Args:
        img: PIL Image to draw on (modified in place)
        meters_per_px: Meters per pixel at image center

    """
    if meters_per_px <= 0:
        return

    ppm = 1.0 / meters_per_px
    cx = img.width // 2
    cy = img.height // 2
    length_px = max(1, round(CENTER_CROSS_LENGTH_M * ppm))
    half = max(1, length_px // 2)
    line_w = max(1, round(CENTER_CROSS_LINE_WIDTH_M * ppm))

    draw = ImageDraw.Draw(img)
    # Red color for center cross for consistency with informer and CP markers
    color = (255, 0, 0)
    draw.line([(cx, cy - half), (cx, cy + half)], fill=color, width=line_w)
    draw.line([(cx - half, cy), (cx + half, cy)], fill=color, width=line_w)


def draw_control_point_triangle(
    img: Image.Image,
    cx_img: float,
    cy_img: float,
    meters_per_px: float,
    rotation_deg: float = 0.0,
    size_m: float | None = None,
) -> None:
    """
    Draw a triangular control point marker at specified position.

    Args:
        img: PIL Image to draw on (modified in place)
        cx_img: X coordinate in image pixels
        cy_img: Y coordinate in image pixels
        meters_per_px: Meters per pixel
        rotation_deg: Image rotation in degrees (for marker orientation)
        size_m: Marker size in meters (defaults to CONTROL_POINT_SIZE_M)

    """
    if meters_per_px <= 0:
        return

    ppm = 1.0 / meters_per_px
    effective_size_m = size_m if size_m is not None else CONTROL_POINT_SIZE_M
    size_px = max(5, round(effective_size_m * ppm))
    half = size_px / 2.0

    # Triangle pointing up (before rotation)
    # Vertices: top, bottom-left, bottom-right
    pts_local = [
        (0, -half),  # top
        (-half, half),  # bottom-left
        (half, half),  # bottom-right
    ]

    # Rotate triangle to align with map orientation
    rotation_rad = math.radians(-rotation_deg)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)

    pts_rotated = []
    for px, py in pts_local:
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        pts_rotated.append((cx_img + rx, cy_img + ry))

    draw = ImageDraw.Draw(img)
    color = tuple(CONTROL_POINT_COLOR)
    draw.polygon(pts_rotated, fill=color, outline=color)


def compute_control_point_image_coords(
    cp_lat_wgs: float,
    cp_lng_wgs: float,
    center_lat_wgs: float,
    center_lng_wgs: float,
    zoom: int,
    eff_scale: int,
    img_width: int,
    img_height: int,
    rotation_deg: float,
    latlng_to_pixel_xy_func: Callable[[float, float, int], tuple[float, float]],
) -> tuple[float, float]:
    """
    Compute control point position in image coordinates.

    Args:
        cp_lat_wgs: Control point latitude (WGS84)
        cp_lng_wgs: Control point longitude (WGS84)
        center_lat_wgs: Map center latitude (WGS84)
        center_lng_wgs: Map center longitude (WGS84)
        zoom: Zoom level
        eff_scale: Effective scale (1 or 2 for retina)
        img_width: Image width in pixels
        img_height: Image height in pixels
        rotation_deg: Map rotation in degrees
        latlng_to_pixel_xy_func: Function to convert lat/lng to pixel coords

    Returns:
        Tuple of (x, y) in image coordinates

    """
    # Get world pixel coordinates
    cx_world, cy_world = latlng_to_pixel_xy_func(center_lat_wgs, center_lng_wgs, zoom)
    cp_x_world, cp_y_world = latlng_to_pixel_xy_func(cp_lat_wgs, cp_lng_wgs, zoom)

    # Convert to image coordinates (relative to center, before rotation)
    img_cx = img_width / 2.0
    img_cy = img_height / 2.0
    x_pre = img_cx + (cp_x_world - cx_world) * eff_scale
    y_pre = img_cy + (cp_y_world - cy_world) * eff_scale

    # Apply rotation around image center
    rotation_rad = math.radians(-rotation_deg)
    cos_rot = math.cos(rotation_rad)
    sin_rot = math.sin(rotation_rad)
    dx = x_pre - img_cx
    dy = y_pre - img_cy
    cx_img = img_cx + dx * cos_rot - dy * sin_rot
    cy_img = img_cy + dx * sin_rot + dy * cos_rot

    return cx_img, cy_img

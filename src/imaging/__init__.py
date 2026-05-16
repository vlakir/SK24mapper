"""Imaging package - image processing utilities for map generation."""

from imaging.composer import assemble_and_crop
from imaging.grid import draw_axis_aligned_km_grid
from imaging.legend import draw_elevation_legend
from imaging.text import (
    draw_label_with_bg,
    draw_label_with_subscript_bg,
    draw_text_with_outline,
    load_grid_font,
)
from imaging.transforms import (
    apply_white_mask,
    center_crop,
    center_crop_np,
    rotate_keep_size,
    rotate_keep_size_np,
    rotate_then_center_crop,
)

__all__ = [
    'apply_white_mask',
    'assemble_and_crop',
    'center_crop',
    'center_crop_np',
    'draw_axis_aligned_km_grid',
    'draw_elevation_legend',
    'draw_label_with_bg',
    'draw_label_with_subscript_bg',
    'draw_text_with_outline',
    'load_grid_font',
    'rotate_keep_size',
    'rotate_keep_size_np',
    'rotate_then_center_crop',
]

"""Imaging package - image processing utilities for map generation."""

from imaging.composer import assemble_and_crop
from imaging.grid import draw_axis_aligned_km_grid
from imaging.grid_streaming import (
    draw_control_point_streaming,
    draw_cross_streaming,
    draw_grid_streaming,
    draw_labels_streaming,
    draw_legend_streaming,
    draw_polylines_streaming,
)
from imaging.legend import draw_elevation_legend
from imaging.pyramid import (
    ImagePyramid,
    build_pyramid_from_pil,
    build_pyramid_from_streaming,
    should_use_pyramid,
)
from imaging.streaming import (
    StreamingImage,
    assemble_tiles_streaming,
    crop_streaming,
    rotate_streaming,
    save_streaming_image,
    save_streaming_jpeg,
    save_streaming_tiff,
)
from imaging.text import (
    calculate_adaptive_grid_font_size,
    draw_label_with_bg,
    draw_label_with_subscript_bg,
    draw_text_with_outline,
    load_grid_font,
)
from imaging.transforms import apply_white_mask, center_crop, rotate_keep_size

__all__ = [
    'ImagePyramid',
    'StreamingImage',
    'apply_white_mask',
    'assemble_and_crop',
    'assemble_tiles_streaming',
    'build_pyramid_from_pil',
    'build_pyramid_from_streaming',
    'calculate_adaptive_grid_font_size',
    'center_crop',
    'crop_streaming',
    'draw_axis_aligned_km_grid',
    'draw_control_point_streaming',
    'draw_cross_streaming',
    'draw_elevation_legend',
    'draw_grid_streaming',
    'draw_label_with_bg',
    'draw_label_with_subscript_bg',
    'draw_labels_streaming',
    'draw_legend_streaming',
    'draw_polylines_streaming',
    'draw_text_with_outline',
    'load_grid_font',
    'rotate_keep_size',
    'rotate_streaming',
    'save_streaming_image',
    'save_streaming_jpeg',
    'save_streaming_tiff',
    'should_use_pyramid',
]

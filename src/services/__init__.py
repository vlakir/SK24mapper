"""Services package - map generation and processing services."""

from services.color_utils import ColorMapper, build_color_lut, color_at_lut, lerp
from services.coordinate_transformer import (
    CoordinateResult,
    CoordinateTransformer,
    validate_control_point_bounds,
)
from services.dem_colorizer import colorize_dem_overlap, colorize_dem_tile_numpy
from services.drawing_utils import (
    compute_rotated_position,
    draw_center_cross,
    draw_control_point_marker,
)
from services.map_context import MapDownloadContext
from services.map_download_service import MapDownloadService, download_map
from services.map_postprocessing import (
    compute_control_point_image_coords,
    draw_center_cross_on_image,
    draw_control_point_triangle,
)
from services.overlay_utils import (
    blend_with_grayscale_base,
    composite_overlay_on_base,
    create_contour_gap_at_labels,
)
from services.processors import (
    process_elevation_color,
    process_elevation_contours,
    process_radio_horizon,
    process_xyz_tiles,
)
from services.tile_coverage import TileCoverage, compute_tile_coverage
from services.tile_fetcher import fetch_xyz_tiles_batch

__all__ = [
    'ColorMapper',
    'CoordinateResult',
    'CoordinateTransformer',
    'MapDownloadContext',
    'MapDownloadService',
    'TileCoverage',
    'blend_with_grayscale_base',
    'build_color_lut',
    'color_at_lut',
    'colorize_dem_overlap',
    'colorize_dem_tile_numpy',
    'composite_overlay_on_base',
    'compute_control_point_image_coords',
    'compute_rotated_position',
    'compute_tile_coverage',
    'create_contour_gap_at_labels',
    'download_map',
    'draw_center_cross',
    'draw_center_cross_on_image',
    'draw_control_point_marker',
    'draw_control_point_triangle',
    'fetch_xyz_tiles_batch',
    'lerp',
    'process_elevation_color',
    'process_elevation_contours',
    'process_radio_horizon',
    'process_xyz_tiles',
    'validate_control_point_bounds',
]

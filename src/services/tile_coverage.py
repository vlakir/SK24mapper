"""Tile coverage calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.constants import ROTATION_PAD_MIN_PX, ROTATION_PAD_RATIO
from geo.topography import compute_xyz_coverage, estimate_crop_size_px


@dataclass
class TileCoverage:
    """Result of tile coverage calculation."""
    
    tiles: list[tuple[int, int]]
    tiles_x: int
    tiles_y: int
    crop_rect: tuple[int, int, int, int]
    map_params: dict[str, Any]
    target_w_px: int
    target_h_px: int
    pad_px: int


def compute_tile_coverage(
    center_lat_wgs: float,
    center_lng_wgs: float,
    width_m: float,
    height_m: float,
    zoom: int,
    eff_scale: int,
) -> TileCoverage:
    """Compute tile coverage for given map parameters.
    
    Args:
        center_lat_wgs: Center latitude in WGS84
        center_lng_wgs: Center longitude in WGS84
        width_m: Map width in meters
        height_m: Map height in meters
        zoom: Zoom level
        eff_scale: Effective scale factor (1 or 2 for retina)
        
    Returns:
        TileCoverage with all computed values
    """
    # Estimate target size
    target_w_px, target_h_px, _ = estimate_crop_size_px(
        center_lat_wgs,
        width_m,
        height_m,
        zoom,
        eff_scale,
    )
    
    # Calculate padding for rotation
    base_pad = round(min(target_w_px, target_h_px) * ROTATION_PAD_RATIO)
    pad_px = max(base_pad, ROTATION_PAD_MIN_PX)
    
    # Compute XYZ tile coverage
    tiles, (tiles_x, tiles_y), crop_rect, map_params = compute_xyz_coverage(
        center_lat=center_lat_wgs,
        center_lng=center_lng_wgs,
        width_m=width_m,
        height_m=height_m,
        zoom=zoom,
        eff_scale=eff_scale,
        pad_px=pad_px,
    )
    
    return TileCoverage(
        tiles=tiles,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        crop_rect=crop_rect,
        map_params=map_params,
        target_w_px=target_w_px,
        target_h_px=target_h_px,
        pad_px=pad_px,
    )

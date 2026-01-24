"""Elevation module - DEM processing and elevation data providers."""

from .dem_builder import (
    DEMCache,
    build_dem_from_tiles,
    compute_elevation_levels,
    downsample_dem_for_seed,
)
from .provider import ElevationTileProvider, TileKey
from .stats import compute_elevation_range, sample_elevation_percentiles

__all__ = [
    'DEMCache',
    'ElevationTileProvider',
    'TileKey',
    'build_dem_from_tiles',
    'compute_elevation_levels',
    'compute_elevation_range',
    'downsample_dem_for_seed',
    'sample_elevation_percentiles',
]

"""Elevation module - DEM processing and elevation data providers."""

from .provider import ElevationTileProvider, TileKey
from .stats import compute_elevation_range, sample_elevation_percentiles

__all__ = [
    'ElevationTileProvider',
    'TileKey',
    'compute_elevation_range',
    'sample_elevation_percentiles',
]

"""Map processors package."""

from services.processors.elevation_color import process_elevation_color
from services.processors.elevation_contours import process_elevation_contours
from services.processors.radio_horizon import process_radio_horizon
from services.processors.xyz_tiles import process_xyz_tiles

__all__ = [
    'process_elevation_color',
    'process_elevation_contours',
    'process_radio_horizon',
    'process_xyz_tiles',
]

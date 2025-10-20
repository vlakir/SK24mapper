#!/usr/bin/env python3
"""Test script to verify elevation legend is added to elevation color maps."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from constants import MapType
from domen import MapSettings
from service import download_satellite_rectangle


async def test_elevation_legend():
    """Test that elevation legend appears on elevation color maps."""
    print('Testing elevation legend on elevation color map...')

    # Get API key from environment
    api_key = os.getenv('MAPBOX_API_KEY')
    if not api_key:
        print('ERROR: MAPBOX_API_KEY environment variable not set')
        return False

    # Test settings for a small elevation color map
    # Using coordinates from one of the profile files (Moscow area)
    settings = MapSettings(
        from_x_high=54,
        from_y_high=74,
        to_x_high=54,
        to_y_high=74,
        from_x_low=14,
        from_y_low=43,
        to_x_low=23,
        to_y_low=49,
        output_path='test_legend_output.jpg',
        grid_width_px=4,
        grid_font_size=86,
        grid_text_margin=43,
        grid_label_bg_padding=6,
        mask_opacity=0.0,
        map_type=MapType.ELEVATION_COLOR,
    )

    # Center coordinates in SK-42 GK (Moscow area, zone 7)
    # X (northing) = 6174000, Y (easting) = 7437000
    center_x = 7437000.0
    center_y = 6174000.0

    # Small area: 2km x 2km
    width_m = 2000.0
    height_m = 2000.0

    try:
        output_path = await download_satellite_rectangle(
            center_x_sk42_gk=center_x,
            center_y_sk42_gk=center_y,
            width_m=width_m,
            height_m=height_m,
            api_key=api_key,
            output_path='maps/test_elevation_legend.jpg',
            max_zoom=15,
            settings=settings,
        )

        print(f'\nSUCCESS: Elevation color map generated: {output_path}')
        print('Please check the bottom-right corner for the elevation legend.')
        return True

    except Exception as e:
        print(f'\nERROR: Failed to generate map: {e}')
        import traceback

        traceback.print_exc()
        return False


if __name__ == '__main__':
    result = asyncio.run(test_elevation_legend())
    sys.exit(0 if result else 1)

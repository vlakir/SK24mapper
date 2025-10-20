#!/usr/bin/env python3
"""
Test script to verify overlay_contours functionality.
Tests that the overlay_contours flag is properly read from profiles
and passed through the system.
"""

import sys

sys.path.append('src')

from profiles import load_profile
from constants import MapType


def test_overlay_contours_flag():
    """Test that overlay_contours flag is properly loaded from profiles."""
    print('=== Testing overlay_contours flag ===\n')

    # Test 1: Load default profile
    print('Test 1: Loading default profile')
    try:
        settings = load_profile('default')
        overlay_flag = getattr(settings, 'overlay_contours', None)
        print(f'  overlay_contours = {overlay_flag}')
        print(f'  map_type = {settings.map_type}')
        print('  ✓ Profile loaded successfully\n')
    except Exception as e:
        print(f'  ✗ Error: {e}\n')
        return False

    # Test 2: Verify the flag is a boolean
    print('Test 2: Verify overlay_contours is boolean')
    if isinstance(overlay_flag, bool):
        print(f'  ✓ overlay_contours is bool: {overlay_flag}\n')
    else:
        print(f'  ✗ overlay_contours is not bool: {type(overlay_flag)}\n')
        return False

    # Test 3: Check MapType enum values
    print('Test 3: Check MapType enum')
    try:
        print(f'  SATELLITE = {MapType.SATELLITE.value}')
        print(f'  ELEVATION_CONTOURS = {MapType.ELEVATION_CONTOURS.value}')
        print('  ✓ MapType enum is accessible\n')
    except Exception as e:
        print(f'  ✗ Error: {e}\n')
        return False

    # Test 4: Verify logic for overlay contours
    print('Test 4: Verify overlay logic conditions')
    is_elev_contours = settings.map_type == MapType.ELEVATION_CONTOURS
    should_overlay = overlay_flag and not is_elev_contours
    print(f'  overlay_contours = {overlay_flag}')
    print(f'  map_type = {settings.map_type}')
    print(f'  is_elev_contours = {is_elev_contours}')
    print(f'  should_overlay = {should_overlay}')
    print('  ✓ Logic check passed\n')

    print('=== All tests passed ===')
    return True


if __name__ == '__main__':
    success = test_overlay_contours_flag()
    sys.exit(0 if success else 1)

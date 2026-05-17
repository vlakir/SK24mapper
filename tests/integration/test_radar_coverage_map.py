"""Integration tests for radar coverage map generation."""

import pytest

from shared.constants import MapType

pytestmark = pytest.mark.integration


async def test_radar_coverage_basic(
    run_map, make_settings, assert_valid_map_output
):
    """Radar sector coverage with DEM-based LOS → overlay → save."""
    settings = make_settings(
        map_type=MapType.RADAR_COVERAGE,
        control_point_enabled=True,
        control_point_x=5418500,
        control_point_y=7446500,
        control_point_name='РЛС',
        radar_azimuth_deg=223.0,
        radar_sector_width_deg=90.0,
        radar_elevation_min_deg=1.0,
        radar_elevation_max_deg=30.0,
        radar_max_range_km=15.0,
        radar_target_height_min_m=30.0,
        radar_target_height_max_m=500.0,
    )
    result_path, metadata, img = await run_map(settings=settings)

    assert_valid_map_output(
        result_path, metadata, img, expected_map_type=MapType.RADAR_COVERAGE
    )

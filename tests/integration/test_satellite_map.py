"""Integration tests for satellite map generation."""

import pytest

from shared.constants import MapType

pytestmark = pytest.mark.integration


async def test_satellite_basic(run_map, make_settings, assert_valid_map_output):
    """Full satellite pipeline: XYZ tiles → stitch → grid → save."""
    settings = make_settings(map_type=MapType.SATELLITE)
    result_path, metadata, img = await run_map(settings=settings)

    assert_valid_map_output(
        result_path, metadata, img, expected_map_type=MapType.SATELLITE
    )


async def test_satellite_with_grid_and_control_point(
    run_map, make_settings, assert_valid_map_output
):
    """Satellite map with grid enabled and control point marker."""
    settings = make_settings(
        map_type=MapType.SATELLITE,
        display_grid=True,
        control_point_enabled=True,
        control_point_x=5418500,
        control_point_y=7446500,
        control_point_name='Test',
    )
    result_path, metadata, img = await run_map(settings=settings)

    assert_valid_map_output(
        result_path, metadata, img, expected_map_type=MapType.SATELLITE
    )
    assert metadata.control_point_enabled is True

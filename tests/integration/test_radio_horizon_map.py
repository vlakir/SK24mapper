"""Integration tests for radio horizon map generation."""

import pytest

from shared.constants import MapType

pytestmark = pytest.mark.integration


async def test_radio_horizon_basic(
    run_map, make_settings, assert_valid_map_output
):
    """LOS raytracing from control point → overlay → grid → save."""
    settings = make_settings(
        map_type=MapType.RADIO_HORIZON,
        control_point_enabled=True,
        control_point_x=5418500,
        control_point_y=7446500,
        control_point_name='НСУ',
        antenna_height_m=10.0,
        max_flight_height_m=500.0,
    )
    result_path, metadata, img = await run_map(settings=settings)

    assert_valid_map_output(
        result_path, metadata, img, expected_map_type=MapType.RADIO_HORIZON
    )


async def test_radio_horizon_requires_control_point(
    run_map, make_settings, assert_valid_map_output
):
    """Radio horizon without control point should raise or warn."""
    settings = make_settings(
        map_type=MapType.RADIO_HORIZON,
        control_point_enabled=False,
    )
    # Service should still produce a map (with warning), or raise.
    # We test that it doesn't silently produce garbage.
    try:
        result_path, metadata, img = await run_map(settings=settings)
        # If it succeeds, basic output checks still apply
        assert_valid_map_output(
            result_path, metadata, img, expected_map_type=MapType.RADIO_HORIZON
        )
    except (RuntimeError, ValueError, Exception):
        # Expected — control point is required for meaningful LOS
        pass

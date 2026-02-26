"""Integration tests for overlay contours on top of a base map."""

import pytest

from shared.constants import MapType

pytestmark = pytest.mark.integration


async def test_satellite_with_overlay_contours(
    run_map, make_settings, assert_valid_map_output
):
    """Satellite base + overlay_contours=True → contour lines on top."""
    settings = make_settings(
        map_type=MapType.SATELLITE,
        overlay_contours=True,
    )
    result_path, metadata, img = await run_map(settings=settings)

    assert_valid_map_output(
        result_path, metadata, img, expected_map_type=MapType.SATELLITE
    )

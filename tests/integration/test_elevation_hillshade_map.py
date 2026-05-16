"""Integration tests for elevation hillshade map generation."""

import pytest

from shared.constants import MapType

pytestmark = pytest.mark.integration


async def test_elevation_hillshade_basic(
    run_map, make_settings, assert_valid_map_output
):
    """DEM z14 → hillshade computation → blend → grid → save."""
    settings = make_settings(map_type=MapType.ELEVATION_HILLSHADE)
    result_path, metadata, img = await run_map(settings=settings)

    assert_valid_map_output(
        result_path, metadata, img, expected_map_type=MapType.ELEVATION_HILLSHADE
    )

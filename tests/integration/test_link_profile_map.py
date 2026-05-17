"""Integration tests for link profile map generation."""

import pytest

from shared.constants import MapType

pytestmark = pytest.mark.integration


async def test_link_profile_basic(
    run_map, make_settings, assert_valid_map_output
):
    """Terrain profile A↔B with LOS/Fresnel analysis + inset diagram."""
    settings = make_settings(
        map_type=MapType.LINK_PROFILE,
        link_point_a_x=5418500,
        link_point_a_y=7446500,
        link_point_a_name='Лютик',
        link_point_b_x=5419000,
        link_point_b_y=7447000,
        link_point_b_name='Одуванчик',
        link_freq_mhz=900.0,
        link_antenna_a_m=10.0,
        link_antenna_b_m=10.0,
    )
    result_path, metadata, img = await run_map(settings=settings)

    assert_valid_map_output(
        result_path, metadata, img, expected_map_type=MapType.LINK_PROFILE
    )

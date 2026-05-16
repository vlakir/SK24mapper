"""Tests for coords_sk42 module."""

import pytest

from geo.coords_sk42 import build_sk42_gk_crs, determine_zone, validate_sk42_bounds


class TestDetermineZone:
    """Tests for determine_zone function."""

    def test_zone_from_x_coordinate_zone_7(self):
        """X coordinate 7_500_000 should return zone 7."""
        assert determine_zone(7_500_000) == 7

    def test_zone_from_x_coordinate_zone_12(self):
        """X coordinate 12_300_000 should return zone 12."""
        assert determine_zone(12_300_000) == 12

    def test_zone_from_x_coordinate_zone_1(self):
        """X coordinate 1_500_000 should return zone 1."""
        assert determine_zone(1_500_000) == 1

    def test_zone_from_x_coordinate_zone_60(self):
        """X coordinate 60_500_000 should return zone 60."""
        assert determine_zone(60_500_000) == 60

    def test_zone_minimum_boundary(self):
        """Zone should be at least 1."""
        # Very small X should clamp to zone 1
        result = determine_zone(100_000)
        assert result >= 1


class TestBuildSk42GkCrs:
    """Tests for build_sk42_gk_crs function."""

    def test_build_crs_zone_7(self):
        """Build CRS for zone 7 should return valid CRS."""
        crs = build_sk42_gk_crs(7)
        assert crs is not None
        # EPSG:28407 is SK-42 GK zone 7
        assert '28407' in str(crs.to_epsg()) or crs.is_projected

    def test_build_crs_zone_12(self):
        """Build CRS for zone 12 should return valid CRS."""
        crs = build_sk42_gk_crs(12)
        assert crs is not None
        assert crs.is_projected

    def test_build_crs_zone_1(self):
        """Build CRS for zone 1 should return valid CRS."""
        crs = build_sk42_gk_crs(1)
        assert crs is not None

    def test_crs_is_projected(self):
        """CRS should be a projected coordinate system."""
        crs = build_sk42_gk_crs(10)
        assert crs.is_projected


class TestValidateSk42Bounds:
    """Tests for validate_sk42_bounds function."""

    def test_valid_coordinates_center(self):
        """Valid coordinates in center of range should not raise."""
        # Should not raise any exception
        validate_sk42_bounds(lng=50.0, lat=55.0)

    def test_valid_coordinates_at_boundaries(self):
        """Coordinates at valid boundaries should not raise."""
        validate_sk42_bounds(lng=19.0, lat=35.0)  # min boundaries
        validate_sk42_bounds(lng=190.0, lat=85.0)  # max boundaries

    def test_invalid_longitude_below_min(self):
        """Longitude below minimum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(lng=18.0, lat=55.0)

    def test_invalid_longitude_above_max(self):
        """Longitude above maximum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(lng=191.0, lat=55.0)

    def test_invalid_latitude_below_min(self):
        """Latitude below minimum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(lng=50.0, lat=34.0)

    def test_invalid_latitude_above_max(self):
        """Latitude above maximum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(lng=50.0, lat=86.0)

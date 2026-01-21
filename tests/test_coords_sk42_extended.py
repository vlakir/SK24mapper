"""Extended tests for coords_sk42 module."""

import pytest
from pyproj import CRS

from coords_sk42 import build_sk42_gk_crs, determine_zone, validate_sk42_bounds


class TestDetermineZone:
    """Tests for determine_zone function."""

    def test_zone_from_prefixed_x(self):
        """Should extract zone from prefixed X coordinate."""
        # X = 7_500_000 means zone 7
        zone = determine_zone(7_500_000.0)
        assert zone == 7

    def test_zone_10(self):
        """Should handle zone 10."""
        zone = determine_zone(10_500_000.0)
        assert zone == 10

    def test_zone_1(self):
        """Should handle zone 1."""
        zone = determine_zone(1_500_000.0)
        assert zone == 1

    def test_zone_clamp_low(self):
        """Should clamp to minimum zone 1."""
        zone = determine_zone(100_000.0)
        assert zone >= 1

    def test_zone_clamp_high(self):
        """Should clamp to maximum zone."""
        zone = determine_zone(99_500_000.0)
        assert zone <= 60  # MAX_GK_ZONE

    def test_typical_moscow_coords(self):
        """Moscow area should be in zone 7."""
        # Moscow is around X=7_400_000 in SK-42 GK
        zone = determine_zone(7_400_000.0)
        assert zone == 7


class TestBuildSk42GkCrs:
    """Tests for build_sk42_gk_crs function."""

    def test_returns_crs(self):
        """Should return a CRS object."""
        crs = build_sk42_gk_crs(7)
        assert isinstance(crs, CRS)

    def test_zone_7(self):
        """Should build CRS for zone 7."""
        crs = build_sk42_gk_crs(7)
        assert crs is not None

    def test_zone_10(self):
        """Should build CRS for zone 10."""
        crs = build_sk42_gk_crs(10)
        assert crs is not None

    def test_different_zones_different_crs(self):
        """Different zones should produce different CRS."""
        crs7 = build_sk42_gk_crs(7)
        crs8 = build_sk42_gk_crs(8)
        assert crs7.to_wkt() != crs8.to_wkt()


class TestValidateSk42Bounds:
    """Tests for validate_sk42_bounds function."""

    def test_valid_moscow(self):
        """Moscow coordinates should be valid."""
        # Should not raise
        validate_sk42_bounds(37.62, 55.75)

    def test_valid_novosibirsk(self):
        """Novosibirsk coordinates should be valid."""
        validate_sk42_bounds(82.92, 55.03)

    def test_valid_boundary_min(self):
        """Minimum boundary should be valid."""
        validate_sk42_bounds(19.0, 35.0)

    def test_valid_boundary_max(self):
        """Maximum boundary should be valid."""
        validate_sk42_bounds(190.0, 85.0)

    def test_invalid_longitude_low(self):
        """Longitude below minimum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(10.0, 55.0)

    def test_invalid_longitude_high(self):
        """Longitude above maximum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(200.0, 55.0)

    def test_invalid_latitude_low(self):
        """Latitude below minimum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(37.0, 30.0)

    def test_invalid_latitude_high(self):
        """Latitude above maximum should raise SystemExit."""
        with pytest.raises(SystemExit):
            validate_sk42_bounds(37.0, 90.0)

"""Tests for services.coordinate_transformer module."""

import logging

import pytest

from services.coordinate_transformer import (
    CoordinateTransformer,
    gk_to_sk42_raw,
    is_point_within_bounds,
    validate_control_point_bounds,
)

# Zone 7: lon_0 = 7*6 - 3 = 39 degrees (Kostroma / Moscow region)
# X (easting) = zone_prefix * 1_000_000 + false_easting(500_000) + offset
# Y (northing) ~ 6_180_000 for lat ~55.75
ZONE7_CENTER_X = 7_500_000.0
ZONE7_CENTER_Y = 6_180_000.0

# Helmert params representative of European Russia region
# (dx, dy, dz, rx, ry, rz, ds)
HELMERT_PARAMS = (23.57, -140.95, -79.8, 0.0, -0.35, -0.79, -0.22)


@pytest.fixture()
def transformer_no_helmert():
    """CoordinateTransformer for zone 7 without Helmert parameters."""
    return CoordinateTransformer(ZONE7_CENTER_X, ZONE7_CENTER_Y)


@pytest.fixture()
def transformer_with_helmert():
    """CoordinateTransformer for zone 7 with custom Helmert parameters."""
    return CoordinateTransformer(
        ZONE7_CENTER_X, ZONE7_CENTER_Y, helmert_params=HELMERT_PARAMS
    )


class TestCoordinateTransformerInit:
    """Tests for CoordinateTransformer initialization and zone detection."""

    def test_zone_detected_correctly(self, transformer_no_helmert):
        """Zone 7 must be detected from X coordinate prefix 7_xxx_xxx."""
        assert transformer_no_helmert.zone == 7

    def test_sk42_center_latitude_range(self, transformer_no_helmert):
        """SK-42 geographic latitude should be close to 55.7 degrees."""
        lat, _ = transformer_no_helmert.get_sk42_center()
        assert 55.0 < lat < 56.5

    def test_sk42_center_longitude_range(self, transformer_no_helmert):
        """SK-42 geographic longitude should be close to 39 degrees."""
        _, lng = transformer_no_helmert.get_sk42_center()
        assert 38.0 < lng < 40.0


class TestGetWgs84Center:
    """Tests for get_wgs84_center method (line 138)."""

    def test_returns_tuple(self, transformer_no_helmert):
        """get_wgs84_center must return a (lat, lng) tuple."""
        result = transformer_no_helmert.get_wgs84_center()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_wgs84_latitude_range(self, transformer_no_helmert):
        """WGS84 latitude should be close to 55.7 degrees."""
        lat, _ = transformer_no_helmert.get_wgs84_center()
        assert 55.0 < lat < 56.5

    def test_wgs84_longitude_range(self, transformer_no_helmert):
        """WGS84 longitude should be close to 39 degrees."""
        _, lng = transformer_no_helmert.get_wgs84_center()
        assert 38.0 < lng < 40.0


class TestGetSk42Center:
    """Tests for get_sk42_center method (line 148)."""

    def test_returns_tuple(self, transformer_no_helmert):
        """get_sk42_center must return a (lat, lng) tuple."""
        result = transformer_no_helmert.get_sk42_center()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_sk42_values_match_attributes(self, transformer_no_helmert):
        """Returned values must match the internal attributes."""
        lat, lng = transformer_no_helmert.get_sk42_center()
        assert lat == transformer_no_helmert.center_lat_sk42
        assert lng == transformer_no_helmert.center_lng_sk42


class TestGkToWgs84:
    """Tests for gk_to_wgs84 method (lines 162-164)."""

    def test_center_close_to_init_wgs84(self, transformer_no_helmert):
        """Converting the initialization GK point should match the WGS84 center."""
        lat, lng = transformer_no_helmert.gk_to_wgs84(ZONE7_CENTER_X, ZONE7_CENTER_Y)
        wgs_lat, wgs_lng = transformer_no_helmert.get_wgs84_center()
        assert lat == pytest.approx(wgs_lat, abs=0.0001)
        assert lng == pytest.approx(wgs_lng, abs=0.0001)

    def test_returns_valid_geographic_coords(self, transformer_no_helmert):
        """Result must be within valid geographic bounds."""
        lat, lng = transformer_no_helmert.gk_to_wgs84(ZONE7_CENTER_X, ZONE7_CENTER_Y)
        assert -90 <= lat <= 90
        assert -180 <= lng <= 180

    def test_offset_point(self, transformer_no_helmert):
        """Converting a point shifted 1 km east should change longitude."""
        lat_c, lng_c = transformer_no_helmert.gk_to_wgs84(
            ZONE7_CENTER_X, ZONE7_CENTER_Y
        )
        lat_e, lng_e = transformer_no_helmert.gk_to_wgs84(
            ZONE7_CENTER_X + 1000, ZONE7_CENTER_Y
        )
        # Eastward shift increases longitude
        assert lng_e > lng_c
        # Latitude should remain approximately the same
        assert lat_e == pytest.approx(lat_c, abs=0.02)


class TestWgs84ToGk:
    """Tests for wgs84_to_gk method (lines 178-180)."""

    def test_center_roundtrip(self, transformer_no_helmert):
        """Converting WGS84 center back to GK should match the original GK point."""
        wgs_lat, wgs_lng = transformer_no_helmert.get_wgs84_center()
        x_gk, y_gk = transformer_no_helmert.wgs84_to_gk(wgs_lat, wgs_lng)
        assert x_gk == pytest.approx(ZONE7_CENTER_X, abs=1.0)
        assert y_gk == pytest.approx(ZONE7_CENTER_Y, abs=1.0)

    def test_returns_valid_gk_coords(self, transformer_no_helmert):
        """Result X should have zone 7 prefix, Y should be reasonable northing."""
        wgs_lat, wgs_lng = transformer_no_helmert.get_wgs84_center()
        x_gk, y_gk = transformer_no_helmert.wgs84_to_gk(wgs_lat, wgs_lng)
        assert 7_000_000 < x_gk < 8_000_000
        assert 5_000_000 < y_gk < 7_000_000


class TestRoundTrip:
    """Round-trip conversion: gk_to_wgs84 then wgs84_to_gk must be close to original."""

    def test_roundtrip_no_helmert(self, transformer_no_helmert):
        """GK -> WGS84 -> GK round-trip without Helmert."""
        original_x = ZONE7_CENTER_X + 2000
        original_y = ZONE7_CENTER_Y + 3000

        lat, lng = transformer_no_helmert.gk_to_wgs84(original_x, original_y)
        recovered_x, recovered_y = transformer_no_helmert.wgs84_to_gk(lat, lng)

        assert recovered_x == pytest.approx(original_x, abs=1.0)
        assert recovered_y == pytest.approx(original_y, abs=1.0)

    def test_roundtrip_with_helmert(self, transformer_with_helmert):
        """GK -> WGS84 -> GK round-trip with custom Helmert."""
        original_x = ZONE7_CENTER_X - 500
        original_y = ZONE7_CENTER_Y + 1500

        lat, lng = transformer_with_helmert.gk_to_wgs84(original_x, original_y)
        recovered_x, recovered_y = transformer_with_helmert.wgs84_to_gk(lat, lng)

        assert recovered_x == pytest.approx(original_x, abs=1.0)
        assert recovered_y == pytest.approx(original_y, abs=1.0)


class TestHelmertLogging:
    """Tests for Helmert params logging (line 97)."""

    def test_no_helmert_logs_warning(self, caplog):
        """Without Helmert params a warning about potential shift is logged."""
        with caplog.at_level(logging.WARNING):
            CoordinateTransformer(ZONE7_CENTER_X, ZONE7_CENTER_Y)
        assert any(
            "без явных региональных" in record.message for record in caplog.records
        )

    def test_custom_helmert_logs_info(self, caplog):
        """With custom Helmert params an info message listing them is logged."""
        with caplog.at_level(logging.INFO):
            CoordinateTransformer(
                ZONE7_CENTER_X, ZONE7_CENTER_Y, helmert_params=HELMERT_PARAMS
            )
        assert any(
            "пользовательские параметры Helmert" in record.message
            for record in caplog.records
        )


class TestValidateControlPointBounds:
    """Tests for validate_control_point_bounds function (lines 256-273)."""

    def test_point_inside_does_not_raise(self):
        """A control point strictly inside the map must not raise."""
        validate_control_point_bounds(
            control_x_gk=7_500_000,
            control_y_gk=6_180_000,
            center_x_gk=7_500_000,
            center_y_gk=6_180_000,
            width_m=10_000,
            height_m=10_000,
        )

    def test_point_on_boundary_does_not_raise(self):
        """A control point exactly on the map edge is still valid."""
        validate_control_point_bounds(
            control_x_gk=7_505_000,
            control_y_gk=6_185_000,
            center_x_gk=7_500_000,
            center_y_gk=6_180_000,
            width_m=10_000,
            height_m=10_000,
        )

    def test_point_outside_east_raises(self):
        """A control point east of the map must raise ValueError."""
        with pytest.raises(ValueError, match="выходит за пределы карты"):
            validate_control_point_bounds(
                control_x_gk=7_510_000,
                control_y_gk=6_180_000,
                center_x_gk=7_500_000,
                center_y_gk=6_180_000,
                width_m=10_000,
                height_m=10_000,
            )

    def test_point_outside_north_raises(self):
        """A control point north of the map must raise ValueError."""
        with pytest.raises(ValueError, match="выходит за пределы карты"):
            validate_control_point_bounds(
                control_x_gk=7_500_000,
                control_y_gk=6_190_000,
                center_x_gk=7_500_000,
                center_y_gk=6_180_000,
                width_m=10_000,
                height_m=10_000,
            )

    def test_point_outside_west_raises(self):
        """A control point west of the map must raise ValueError."""
        with pytest.raises(ValueError, match="выходит за пределы карты"):
            validate_control_point_bounds(
                control_x_gk=7_490_000,
                control_y_gk=6_180_000,
                center_x_gk=7_500_000,
                center_y_gk=6_180_000,
                width_m=10_000,
                height_m=10_000,
            )

    def test_point_outside_south_raises(self):
        """A control point south of the map must raise ValueError."""
        with pytest.raises(ValueError, match="выходит за пределы карты"):
            validate_control_point_bounds(
                control_x_gk=7_500_000,
                control_y_gk=6_170_000,
                center_x_gk=7_500_000,
                center_y_gk=6_180_000,
                width_m=10_000,
                height_m=10_000,
            )

    def test_error_message_contains_coordinates(self):
        """ValueError message must include the out-of-bounds coordinates."""
        with pytest.raises(ValueError, match="7510000") as exc_info:
            validate_control_point_bounds(
                control_x_gk=7_510_000,
                control_y_gk=6_180_000,
                center_x_gk=7_500_000,
                center_y_gk=6_180_000,
                width_m=10_000,
                height_m=10_000,
            )
        assert "6180000" in str(exc_info.value)


class TestIsPointWithinBounds:
    """Tests for is_point_within_bounds function."""

    def test_center_point_inside(self):
        assert is_point_within_bounds(
            7_500_000, 6_180_000, 7_500_000, 6_180_000, 10_000, 10_000
        )

    def test_boundary_point_inside(self):
        assert is_point_within_bounds(
            7_505_000, 6_185_000, 7_500_000, 6_180_000, 10_000, 10_000
        )

    def test_point_outside_east(self):
        assert not is_point_within_bounds(
            7_510_000, 6_180_000, 7_500_000, 6_180_000, 10_000, 10_000
        )

    def test_point_outside_north(self):
        assert not is_point_within_bounds(
            7_500_000, 6_190_000, 7_500_000, 6_180_000, 10_000, 10_000
        )

    def test_point_outside_west(self):
        assert not is_point_within_bounds(
            7_490_000, 6_180_000, 7_500_000, 6_180_000, 10_000, 10_000
        )

    def test_point_outside_south(self):
        assert not is_point_within_bounds(
            7_500_000, 6_170_000, 7_500_000, 6_180_000, 10_000, 10_000
        )


class TestGkToSk42Raw:
    """Tests for gk_to_sk42_raw — inverse of MapSettings.*_sk42_gk properties."""

    def test_roundtrip_default_control_point(self):
        """Default control_point_x=5415000, control_point_y=7440000."""
        # Forward: control_point_y=7440000 → x_gk = 1e3*40 + 1e5*74 = 7440000
        # Forward: control_point_x=5415000 → y_gk = 1e3*15 + 1e5*54 = 5415000
        raw_x, raw_y = gk_to_sk42_raw(7_440_000.0, 5_415_000.0)
        assert raw_x == 5415000
        assert raw_y == 7440000

    def test_roundtrip_offset_point(self):
        """Verify round-trip for a different GK coordinate pair."""
        # raw_x=6015000 → y_gk = 1e3*15 + 1e5*60 = 6015000
        # raw_y=7500000 → x_gk = 1e3*0 + 1e5*75 = 7500000
        raw_x, raw_y = gk_to_sk42_raw(7_500_000.0, 6_015_000.0)
        assert raw_x == 6015000
        assert raw_y == 7500000

    def test_fractional_meters_rounded(self):
        """GK coordinates with fractional meters are rounded to nearest int."""
        raw_x, raw_y = gk_to_sk42_raw(7_440_500.7, 5_415_250.3)
        # x_gk=7440500.7 → y_high=74, y_low_km=40.5007 → raw_y = 74*1e5 + round(40500.7) = 7440501
        assert raw_y == 7440501
        # y_gk=5415250.3 → x_high=54, x_low_km=15.2503 → raw_x = 54*1e5 + round(15250.3) = 5415250
        assert raw_x == 5415250

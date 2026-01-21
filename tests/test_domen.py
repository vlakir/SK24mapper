"""Tests for domen module."""

import pytest

from domen import MapSettings


def create_base_settings(**overrides):
    """Create MapSettings with default values and optional overrides."""
    defaults = {
        'from_x_high': 54,
        'from_y_high': 74,
        'to_x_high': 54,
        'to_y_high': 74,
        'from_x_low': 12,
        'from_y_low': 35,
        'to_x_low': 22,
        'to_y_low': 49,
        'output_path': 'test.jpg',
        'grid_width_m': 5.0,
        'grid_font_size_m': 100.0,
        'grid_text_margin_m': 50.0,
        'grid_label_bg_padding_m': 10.0,
        'mask_opacity': 0.35,
    }
    defaults.update(overrides)
    return MapSettings(**defaults)


class TestMapSettingsValidators:
    """Tests for MapSettings validators."""

    def test_mask_opacity_valid(self):
        """Valid mask_opacity values should be accepted."""
        settings = create_base_settings(mask_opacity=0.5)
        assert settings.mask_opacity == 0.5

    def test_mask_opacity_zero(self):
        """mask_opacity=0.0 should be valid."""
        settings = create_base_settings(mask_opacity=0.0)
        assert settings.mask_opacity == 0.0

    def test_mask_opacity_one(self):
        """mask_opacity=1.0 should be valid."""
        settings = create_base_settings(mask_opacity=1.0)
        assert settings.mask_opacity == 1.0

    def test_mask_opacity_invalid_below_zero(self):
        """mask_opacity below 0 should raise ValueError."""
        with pytest.raises(ValueError):
            create_base_settings(mask_opacity=-0.1)

    def test_mask_opacity_invalid_above_one(self):
        """mask_opacity above 1 should raise ValueError."""
        with pytest.raises(ValueError):
            create_base_settings(mask_opacity=1.5)

    def test_brightness_clamped_to_max(self):
        """Brightness above 4.0 should be clamped to 4.0."""
        settings = create_base_settings(brightness=5.0)
        assert settings.brightness == 4.0

    def test_brightness_clamped_to_min(self):
        """Brightness below 0 should be clamped to 0."""
        settings = create_base_settings(brightness=-1.0)
        assert settings.brightness == 0.0

    def test_contrast_clamped_to_max(self):
        """Contrast above 2.0 should be clamped to 2.0."""
        settings = create_base_settings(contrast=3.0)
        assert settings.contrast == 2.0

    def test_saturation_clamped_to_max(self):
        """Saturation above 2.0 should be clamped to 2.0."""
        settings = create_base_settings(saturation=3.0)
        assert settings.saturation == 2.0


class TestMapSettingsProperties:
    """Tests for MapSettings computed properties."""

    def test_bottom_left_x_sk42_gk(self):
        """Test bottom_left_x_sk42_gk calculation."""
        # ADDITIVE_RATIO = 0.3
        # from_y_low = 35, from_y_high = 74
        # Result: 1e3 * (35 - 0.3) + 1e5 * 74 = 34700 + 7400000 = 7434700
        settings = create_base_settings()
        assert settings.bottom_left_x_sk42_gk == 7434700.0

    def test_bottom_left_y_sk42_gk(self):
        """Test bottom_left_y_sk42_gk calculation."""
        # from_x_low = 12, from_x_high = 54
        # Result: 1e3 * (12 - 0.3) + 1e5 * 54 = 11700 + 5400000 = 5411700
        settings = create_base_settings()
        assert settings.bottom_left_y_sk42_gk == 5411700.0

    def test_top_right_x_sk42_gk(self):
        """Test top_right_x_sk42_gk calculation."""
        # to_y_low = 49, to_y_high = 74
        # Result: 1e3 * (49 + 0.3) + 1e5 * 74 = 49300 + 7400000 = 7449300
        settings = create_base_settings()
        assert settings.top_right_x_sk42_gk == 7449300.0

    def test_top_right_y_sk42_gk(self):
        """Test top_right_y_sk42_gk calculation."""
        # to_x_low = 22, to_x_high = 54
        # Result: 1e3 * (22 + 0.3) + 1e5 * 54 = 22300 + 5400000 = 5422300
        settings = create_base_settings()
        assert settings.top_right_y_sk42_gk == 5422300.0

    def test_control_point_coordinates(self):
        """Test control point coordinate conversion."""
        settings = create_base_settings(
            control_point_enabled=True,
            control_point_x=5415000,
            control_point_y=7440000,
        )
        # control_point_y = 7440000 -> y_high=74, y_low_km=40.0
        # control_point_x_sk42_gk = 1e3 * 40.0 + 1e5 * 74 = 40000 + 7400000 = 7440000
        assert settings.control_point_x_sk42_gk == 7440000.0
        # control_point_x = 5415000 -> x_high=54, x_low_km=15.0
        # control_point_y_sk42_gk = 1e3 * 15.0 + 1e5 * 54 = 15000 + 5400000 = 5415000
        assert settings.control_point_y_sk42_gk == 5415000.0


class TestMapSettingsHelmert:
    """Tests for Helmert parameters."""

    def test_custom_helmert_all_set(self):
        """All Helmert parameters set should return tuple."""
        settings = create_base_settings(
            helmert_dx=-50.957,
            helmert_dy=-39.724,
            helmert_dz=-76.877,
            helmert_rx_as=2.33295,
            helmert_ry_as=2.13987,
            helmert_rz_as=-2.03005,
            helmert_ds_ppm=-1.43065,
        )
        helmert = settings.custom_helmert
        assert helmert is not None
        assert len(helmert) == 7
        assert helmert[0] == -50.957  # dx
        assert helmert[6] == -1.43065  # ds_ppm

    def test_custom_helmert_none_when_partial(self):
        """Partial Helmert parameters should return None."""
        settings = create_base_settings(
            helmert_dx=-50.957,
            helmert_dy=-39.724,
            # Other parameters not set (None by default)
        )
        assert settings.custom_helmert is None

    def test_custom_helmert_none_by_default(self):
        """Default settings should have None custom_helmert."""
        settings = create_base_settings()
        assert settings.custom_helmert is None


class TestMapSettingsExtraFields:
    """Tests for extra field handling."""

    def test_extra_fields_ignored(self):
        """Extra fields in input should be ignored."""
        # Model config has extra='ignore'
        settings = MapSettings(
            from_x_high=54,
            from_y_high=74,
            to_x_high=54,
            to_y_high=74,
            from_x_low=12,
            from_y_low=35,
            to_x_low=22,
            to_y_low=49,
            output_path='test.jpg',
            grid_width_m=5.0,
            grid_font_size_m=100.0,
            grid_text_margin_m=50.0,
            grid_label_bg_padding_m=10.0,
            mask_opacity=0.35,
            unknown_field='should_be_ignored',  # Extra field
        )
        assert not hasattr(settings, 'unknown_field')

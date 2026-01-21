"""Extended tests for domen module."""

import pytest

from domen import MapSettings


def create_settings(**overrides):
    """Create MapSettings with default values."""
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


class TestMapSettingsGridParams:
    """Tests for grid parameters."""

    def test_grid_width_m_default(self):
        """Default grid width should be set."""
        settings = create_settings()
        assert settings.grid_width_m == 5.0

    def test_grid_font_size_m(self):
        """Grid font size should be set."""
        settings = create_settings(grid_font_size_m=150.0)
        assert settings.grid_font_size_m == 150.0

    def test_grid_text_margin_m(self):
        """Grid text margin should be set."""
        settings = create_settings(grid_text_margin_m=75.0)
        assert settings.grid_text_margin_m == 75.0

    def test_grid_label_bg_padding_m(self):
        """Grid label bg padding should be set."""
        settings = create_settings(grid_label_bg_padding_m=15.0)
        assert settings.grid_label_bg_padding_m == 15.0


class TestMapSettingsImageParams:
    """Tests for image parameters."""

    def test_brightness_default(self):
        """Default brightness should be 1.0."""
        settings = create_settings()
        assert settings.brightness == 1.0

    def test_contrast_default(self):
        """Default contrast should be 1.0."""
        settings = create_settings()
        assert settings.contrast == 1.0

    def test_saturation_default(self):
        """Default saturation should be 1.0."""
        settings = create_settings()
        assert settings.saturation == 1.0

    def test_brightness_custom(self):
        """Custom brightness should be set."""
        settings = create_settings(brightness=1.5)
        assert settings.brightness == 1.5

    def test_contrast_custom(self):
        """Custom contrast should be set."""
        settings = create_settings(contrast=1.5)
        assert settings.contrast == 1.5

    def test_saturation_custom(self):
        """Custom saturation should be set."""
        settings = create_settings(saturation=1.5)
        assert settings.saturation == 1.5


class TestMapSettingsControlPoint:
    """Tests for control point settings."""

    def test_control_point_disabled_by_default(self):
        """Control point should be disabled by default."""
        settings = create_settings()
        assert settings.control_point_enabled is False

    def test_control_point_enabled(self):
        """Control point can be enabled."""
        settings = create_settings(control_point_enabled=True, control_point_x=5415000, control_point_y=7440000)
        assert settings.control_point_enabled is True

    def test_control_point_x_has_default(self):
        """Control point X has default value."""
        settings = create_settings()
        assert settings.control_point_x == 5415000

    def test_control_point_y_has_default(self):
        """Control point Y has default value."""
        settings = create_settings()
        assert settings.control_point_y == 7440000

    def test_control_point_x_sk42_gk(self):
        """Control point X SK42 GK should be calculated."""
        settings = create_settings(control_point_x=5415000, control_point_y=7440000)
        assert settings.control_point_x_sk42_gk == 7440000.0

    def test_control_point_y_sk42_gk(self):
        """Control point Y SK42 GK should be calculated."""
        settings = create_settings(control_point_x=5415000, control_point_y=7440000)
        assert settings.control_point_y_sk42_gk == 5415000.0


class TestMapSettingsMapType:
    """Tests for map type settings."""

    def test_map_type_default(self):
        """Default map type should be SATELLITE."""
        settings = create_settings()
        assert settings.map_type == 'SATELLITE'

    def test_map_type_hybrid(self):
        """Map type can be set to HYBRID."""
        settings = create_settings(map_type='HYBRID')
        assert settings.map_type == 'HYBRID'

    def test_map_type_streets(self):
        """Map type can be set to STREETS."""
        settings = create_settings(map_type='STREETS')
        assert settings.map_type == 'STREETS'


class TestMapSettingsDisplayOptions:
    """Tests for display options."""

    def test_display_grid_default(self):
        """Display grid should be True by default."""
        settings = create_settings()
        assert settings.display_grid is True

    def test_display_grid_disabled(self):
        """Display grid can be disabled."""
        settings = create_settings(display_grid=False)
        assert settings.display_grid is False

    def test_overlay_contours_default(self):
        """Overlay contours should be False by default."""
        settings = create_settings()
        assert settings.overlay_contours is False

    def test_overlay_contours_enabled(self):
        """Overlay contours can be enabled."""
        settings = create_settings(overlay_contours=True)
        assert settings.overlay_contours is True


class TestMapSettingsHelmertExtended:
    """Extended tests for Helmert parameters."""

    def test_helmert_dx_default(self):
        """Helmert dx should be None by default."""
        settings = create_settings()
        assert settings.helmert_dx is None

    def test_helmert_all_params(self):
        """All Helmert params can be set."""
        settings = create_settings(
            helmert_dx=1.0,
            helmert_dy=2.0,
            helmert_dz=3.0,
            helmert_rx_as=0.1,
            helmert_ry_as=0.2,
            helmert_rz_as=0.3,
            helmert_ds_ppm=0.01,
        )
        assert settings.helmert_dx == 1.0
        assert settings.helmert_dy == 2.0
        assert settings.helmert_dz == 3.0
        assert settings.helmert_rx_as == 0.1
        assert settings.helmert_ry_as == 0.2
        assert settings.helmert_rz_as == 0.3
        assert settings.helmert_ds_ppm == 0.01

    def test_custom_helmert_returns_tuple(self):
        """custom_helmert should return tuple when all set."""
        settings = create_settings(
            helmert_dx=1.0,
            helmert_dy=2.0,
            helmert_dz=3.0,
            helmert_rx_as=0.1,
            helmert_ry_as=0.2,
            helmert_rz_as=0.3,
            helmert_ds_ppm=0.01,
        )
        helmert = settings.custom_helmert
        assert helmert is not None
        assert len(helmert) == 7


class TestMapSettingsCoordinates:
    """Tests for coordinate properties."""

    def test_bottom_left_x_sk42_gk(self):
        """Bottom left X should be calculated."""
        settings = create_settings()
        assert settings.bottom_left_x_sk42_gk > 0

    def test_bottom_left_y_sk42_gk(self):
        """Bottom left Y should be calculated."""
        settings = create_settings()
        assert settings.bottom_left_y_sk42_gk > 0

    def test_top_right_x_sk42_gk(self):
        """Top right X should be calculated."""
        settings = create_settings()
        assert settings.top_right_x_sk42_gk > 0

    def test_top_right_y_sk42_gk(self):
        """Top right Y should be calculated."""
        settings = create_settings()
        assert settings.top_right_y_sk42_gk > 0

    def test_top_right_greater_than_bottom_left(self):
        """Top right should be greater than bottom left."""
        settings = create_settings()
        assert settings.top_right_x_sk42_gk > settings.bottom_left_x_sk42_gk
        assert settings.top_right_y_sk42_gk > settings.bottom_left_y_sk42_gk


class TestMapSettingsOutputPath:
    """Tests for output path settings."""

    def test_output_path_default(self):
        """Output path should be set."""
        settings = create_settings()
        assert settings.output_path == 'test.jpg'

    def test_output_path_custom(self):
        """Custom output path should be set."""
        settings = create_settings(output_path='custom/path/map.jpg')
        assert settings.output_path == 'custom/path/map.jpg'


class TestMapSettingsFromToCoords:
    """Tests for from/to coordinate fields."""

    def test_from_x_high(self):
        """from_x_high should be set."""
        settings = create_settings(from_x_high=55)
        assert settings.from_x_high == 55

    def test_from_y_high(self):
        """from_y_high should be set."""
        settings = create_settings(from_y_high=75)
        assert settings.from_y_high == 75

    def test_to_x_high(self):
        """to_x_high should be set."""
        settings = create_settings(to_x_high=56)
        assert settings.to_x_high == 56

    def test_to_y_high(self):
        """to_y_high should be set."""
        settings = create_settings(to_y_high=76)
        assert settings.to_y_high == 76

    def test_from_x_low(self):
        """from_x_low should be set."""
        settings = create_settings(from_x_low=15)
        assert settings.from_x_low == 15

    def test_from_y_low(self):
        """from_y_low should be set."""
        settings = create_settings(from_y_low=40)
        assert settings.from_y_low == 40

    def test_to_x_low(self):
        """to_x_low should be set."""
        settings = create_settings(to_x_low=25)
        assert settings.to_x_low == 25

    def test_to_y_low(self):
        """to_y_low should be set."""
        settings = create_settings(to_y_low=55)
        assert settings.to_y_low == 55


class TestMapSettingsMaskOpacity:
    """Tests for mask opacity."""

    def test_mask_opacity_set(self):
        """mask_opacity should be set."""
        settings = create_settings(mask_opacity=0.5)
        assert settings.mask_opacity == 0.5

    def test_mask_opacity_zero(self):
        """mask_opacity zero should work."""
        settings = create_settings(mask_opacity=0.0)
        assert settings.mask_opacity == 0.0

    def test_mask_opacity_one(self):
        """mask_opacity one should work."""
        settings = create_settings(mask_opacity=1.0)
        assert settings.mask_opacity == 1.0

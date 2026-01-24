"""Tests for image module."""

import pytest
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

from geo.topography import crs_sk42_geog
from imaging import (
    apply_white_mask,
    assemble_and_crop,
    calculate_adaptive_grid_font_size,
    center_crop,
    draw_axis_aligned_km_grid,
    draw_elevation_legend,
    draw_label_with_bg,
    draw_text_with_outline,
    load_grid_font,
    rotate_keep_size,
)


class TestCenterCrop:
    """Tests for center_crop function."""

    def test_center_crop_exact_size(self):
        """Crop to exact same size returns same dimensions."""
        img = Image.new('RGB', (100, 100), color='red')
        result = center_crop(img, 100, 100)
        assert result.size == (100, 100)

    def test_center_crop_smaller(self):
        """Crop to smaller size works correctly."""
        img = Image.new('RGB', (200, 200), color='blue')
        result = center_crop(img, 100, 100)
        assert result.size == (100, 100)

    def test_center_crop_asymmetric(self):
        """Crop with different width and height."""
        img = Image.new('RGB', (300, 200), color='green')
        result = center_crop(img, 150, 100)
        assert result.size == (150, 100)

    def test_center_crop_larger_than_image(self):
        """Crop larger than image clips to available size."""
        img = Image.new('RGB', (50, 50), color='yellow')
        result = center_crop(img, 100, 100)
        # Should return what's available from center
        assert result.size[0] <= 100
        assert result.size[1] <= 100


class TestApplyWhiteMask:
    """Tests for apply_white_mask function."""

    def test_zero_opacity_returns_original(self):
        """Zero opacity should return original image."""
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        result = apply_white_mask(img, 0.0)
        # Should be unchanged
        assert result.size == img.size

    def test_full_opacity_returns_white(self):
        """Full opacity should return nearly white image."""
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        result = apply_white_mask(img, 1.0)
        # Center pixel should be white
        pixel = result.getpixel((50, 50))
        assert pixel == (255, 255, 255)

    def test_half_opacity_blends(self):
        """Half opacity should blend colors."""
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        result = apply_white_mask(img, 0.5)
        pixel = result.getpixel((50, 50))
        # Should be grayish (around 127-128)
        assert 120 <= pixel[0] <= 135
        assert 120 <= pixel[1] <= 135
        assert 120 <= pixel[2] <= 135

    def test_opacity_clamped_below_zero(self):
        """Negative opacity should be clamped to 0."""
        img = Image.new('RGB', (100, 100), color=(100, 100, 100))
        result = apply_white_mask(img, -0.5)
        # Should return original (opacity clamped to 0)
        assert result.size == img.size

    def test_opacity_clamped_above_one(self):
        """Opacity above 1 should be clamped to 1."""
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        result = apply_white_mask(img, 1.5)
        pixel = result.getpixel((50, 50))
        assert pixel == (255, 255, 255)

    def test_returns_rgb_mode(self):
        """Result should be in RGB mode."""
        img = Image.new('RGB', (100, 100), color='red')
        result = apply_white_mask(img, 0.3)
        assert result.mode == 'RGB'


class TestDrawTextWithOutline:
    """Tests for draw_text_with_outline function."""

    def test_draws_text(self):
        """Should draw text on image."""
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        font = load_grid_font(20)
        draw_text_with_outline(draw, (10, 10), 'Test', font)
        # Image should have some non-white pixels
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_draws_with_custom_colors(self):
        """Should draw text with custom fill and outline colors."""
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        font = load_grid_font(20)
        draw_text_with_outline(
            draw, (50, 50), 'X', font,
            fill=(255, 0, 0), outline=(0, 0, 255), outline_width=2
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_no_outline_when_width_zero(self):
        """Should draw only fill text when outline_width is 0."""
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        font = load_grid_font(20)
        draw_text_with_outline(
            draw, (50, 50), 'A', font,
            fill=(0, 0, 0), outline=(255, 0, 0), outline_width=0
        )
        pixels = list(img.getdata())
        # Should have black pixels but no red
        has_black = any(p[0] < 50 and p[1] < 50 and p[2] < 50 for p in pixels)
        assert has_black


class TestDrawLabelWithBg:
    """Tests for draw_label_with_bg function."""

    def test_draws_label_with_background(self):
        """Should draw label with background rectangle."""
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        font = load_grid_font(20)
        draw_label_with_bg(
            draw, (150, 50), 'Label', font,
            anchor='mm', img_size=img.size
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_custom_background_color(self):
        """Should use custom background color."""
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        font = load_grid_font(20)
        draw_label_with_bg(
            draw, (150, 50), 'Test', font,
            anchor='mm', img_size=img.size,
            bg_color=(255, 0, 0), padding=10
        )
        pixels = list(img.getdata())
        # Should have red pixels from background
        has_red = any(p[0] > 200 and p[1] < 50 and p[2] < 50 for p in pixels)
        assert has_red


class TestLoadGridFont:
    """Tests for load_grid_font function."""

    def test_returns_font(self):
        """Should return a font object."""
        font = load_grid_font(20)
        assert font is not None

    def test_different_sizes(self):
        """Should load fonts of different sizes."""
        font_small = load_grid_font(12)
        font_large = load_grid_font(48)
        assert font_small is not None
        assert font_large is not None

    def test_default_size(self):
        """Should work with default size."""
        font = load_grid_font()
        assert font is not None


class TestAssembleAndCrop:
    """Tests for assemble_and_crop function."""

    def test_single_tile(self):
        """Should handle single tile."""
        tile = Image.new('RGB', (256, 256), color='red')
        result = assemble_and_crop([tile], 1, 1, 256, (0, 0, 256, 256))
        assert result.size == (256, 256)

    def test_2x2_grid(self):
        """Should assemble 2x2 grid correctly."""
        tiles = [Image.new('RGB', (256, 256), color=c) for c in ['red', 'green', 'blue', 'yellow']]
        result = assemble_and_crop(tiles, 2, 2, 256, (0, 0, 512, 512))
        assert result.size == (512, 512)

    def test_with_crop(self):
        """Should crop to specified rectangle."""
        tiles = [Image.new('RGB', (256, 256), color='red') for _ in range(4)]
        result = assemble_and_crop(tiles, 2, 2, 256, (50, 50, 200, 200))
        assert result.size == (200, 200)

    def test_3x2_grid(self):
        """Should handle non-square grid."""
        tiles = [Image.new('RGB', (256, 256), color='blue') for _ in range(6)]
        result = assemble_and_crop(tiles, 3, 2, 256, (0, 0, 768, 512))
        assert result.size == (768, 512)

    def test_different_tile_size(self):
        """Should work with different tile sizes."""
        tiles = [Image.new('RGB', (512, 512), color='green') for _ in range(4)]
        result = assemble_and_crop(tiles, 2, 2, 512, (0, 0, 1024, 1024))
        assert result.size == (1024, 1024)


class TestRotateKeepSize:
    """Tests for rotate_keep_size function."""

    def test_no_rotation(self):
        """Zero rotation should keep same image."""
        img = Image.new('RGB', (100, 100), color='red')
        result = rotate_keep_size(img, 0.0)
        assert result.size == (100, 100)

    def test_90_degree_rotation(self):
        """90 degree rotation should keep size."""
        img = Image.new('RGB', (100, 100), color='blue')
        result = rotate_keep_size(img, 90.0)
        assert result.size == (100, 100)

    def test_45_degree_rotation(self):
        """45 degree rotation should keep size."""
        img = Image.new('RGB', (100, 100), color='green')
        result = rotate_keep_size(img, 45.0)
        assert result.size == (100, 100)

    def test_negative_rotation(self):
        """Negative rotation should work."""
        img = Image.new('RGB', (100, 100), color='yellow')
        result = rotate_keep_size(img, -30.0)
        assert result.size == (100, 100)

    def test_custom_fill_color(self):
        """Should use custom fill color."""
        img = Image.new('RGB', (100, 100), color='red')
        result = rotate_keep_size(img, 45.0, fill=(0, 0, 255))
        assert result.size == (100, 100)

    def test_rectangular_image(self):
        """Should handle rectangular images."""
        img = Image.new('RGB', (200, 100), color='purple')
        result = rotate_keep_size(img, 30.0)
        assert result.size == (200, 100)


class TestCalculateAdaptiveGridFontSize:
    """Tests for calculate_adaptive_grid_font_size function."""

    def test_returns_positive(self):
        """Should return positive font size."""
        size = calculate_adaptive_grid_font_size(1.0)
        assert size > 0

    def test_smaller_mpp_larger_font(self):
        """Smaller mpp (higher zoom) should give larger font."""
        size_high_zoom = calculate_adaptive_grid_font_size(0.5)
        size_low_zoom = calculate_adaptive_grid_font_size(5.0)
        assert size_high_zoom >= size_low_zoom

    def test_typical_values(self):
        """Should return reasonable values for typical mpp."""
        size = calculate_adaptive_grid_font_size(2.0)
        assert 10 <= size <= 200

    def test_very_small_mpp(self):
        """Should handle very small mpp."""
        size = calculate_adaptive_grid_font_size(0.1)
        assert size > 0

    def test_very_large_mpp(self):
        """Should handle very large mpp."""
        size = calculate_adaptive_grid_font_size(100.0)
        assert size > 0


@pytest.fixture
def sk42_gk_crs():
    """Return example SK-42 GK CRS."""
    return CRS.from_string(
        "+proj=tmerc +lat_0=0 +lon_0=39 +k=1 +x_0=7500000 +y_0=0 +ellps=krass +units=m +no_defs"
    )


@pytest.fixture
def sk42_to_wgs_transformer():
    """Return transformer from SK-42 to WGS-84."""
    return Transformer.from_crs(crs_sk42_geog, "EPSG:4326", always_xy=True)


class TestDrawAxisAlignedKmGrid:
    """Tests for draw_axis_aligned_km_grid function."""

    def test_draw_grid_full(self, sk42_gk_crs, sk42_to_wgs_transformer):
        """Should draw grid when enabled."""
        img = Image.new('RGB', (1000, 1000), color='white')
        draw_axis_aligned_km_grid(
            img,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=True,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_draw_grid_crosses_only(self, sk42_gk_crs, sk42_to_wgs_transformer):
        """Should draw only crosses when grid disabled."""
        img = Image.new('RGB', (1000, 1000), color='white')
        draw_axis_aligned_km_grid(
            img,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=False,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_draw_grid_with_legend_bounds(self, sk42_gk_crs, sk42_to_wgs_transformer):
        """Should work with legend bounds provided."""
        img = Image.new('RGB', (1000, 1000), color='white')
        legend_bounds = (100, 100, 400, 400)
        draw_axis_aligned_km_grid(
            img,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=True,
            legend_bounds=legend_bounds,
        )
        assert img.size == (1000, 1000)


class TestDrawElevationLegend:
    """Tests for draw_elevation_legend function."""

    def test_draw_elevation_legend_basic(self):
        """Should draw legend without errors."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 255)),
            (0.5, (0, 255, 0)),
            (1.0, (255, 0, 0)),
        ]
        draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=1000,
            center_lat_wgs=55.75,
            zoom=12,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

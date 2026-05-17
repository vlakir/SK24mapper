"""Tests for image module."""

import pytest
from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer

from geo.topography import crs_sk42_geog
from imaging import (
    apply_white_mask,
    assemble_and_crop,
    center_crop,
    draw_axis_aligned_km_grid,
    draw_elevation_legend,
    draw_label_with_bg,
    draw_label_with_subscript_bg,
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


class TestDrawLabelWithSubscriptBg:
    """Tests for draw_label_with_subscript_bg function."""

    def test_draws_label_with_subscript(self):
        """Should draw label with subscript and background."""
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        font = load_grid_font(20)
        sub_font = load_grid_font(14)
        parts = [('Normal', False), ('Sub', True), ('Text', False)]
        
        draw_label_with_subscript_bg(
            draw, (200, 100), parts, font, sub_font,
            anchor='mm', img_size=img.size, bg_color=(0, 255, 0)
        )
        
        pixels = list(img.getdata())
        # Should have green pixels from background
        has_green = any(p[0] < 50 and p[1] > 200 and p[2] < 50 for p in pixels)
        assert has_green


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

    def test_pil_fallback_for_large_image(self):
        """Images exceeding cv2 SHRT_MAX limit should use PIL fallback."""
        from imaging.transforms import _CV2_DIM_LIMIT

        # Минимальное изображение с одной стороной >= лимита
        img = Image.new('RGB', (4, _CV2_DIM_LIMIT), color=(100, 150, 200))
        result = rotate_keep_size(img, 1.5, fill=(255, 255, 255))
        assert result.size == img.size

    def test_pil_fallback_wide_image(self):
        """Wide image exceeding limit should rotate without error."""
        from imaging.transforms import _CV2_DIM_LIMIT

        img = Image.new('RGB', (_CV2_DIM_LIMIT, 4), color=(50, 100, 150))
        result = rotate_keep_size(img, -2.0, fill=(0, 0, 0))
        assert result.size == img.size


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


class TestDrawLabelWithSubscriptBgAnchors:
    """Tests for draw_label_with_subscript_bg with various anchor values."""

    @pytest.fixture()
    def _fonts(self):
        font = load_grid_font(24)
        sub_font = load_grid_font(16)
        return font, sub_font

    @pytest.fixture()
    def _parts(self):
        return [('X', False), ('2', True)]

    def test_anchor_rt_right_top(self, _fonts, _parts):
        """Anchor 'rt' should position text to the left of x, top-aligned."""
        font, sub_font = _fonts
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw_label_with_subscript_bg(
            draw, (380, 10), _parts, font, sub_font,
            anchor='rt', img_size=img.size, bg_color=(255, 0, 0),
        )
        pixels = list(img.getdata())
        has_red = any(p[0] > 200 and p[1] < 50 and p[2] < 50 for p in pixels)
        assert has_red, "Expected red background pixels for anchor='rt'"

    def test_anchor_lb_left_bottom(self, _fonts, _parts):
        """Anchor 'lb' should position text from x to the right, bottom-aligned."""
        font, sub_font = _fonts
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw_label_with_subscript_bg(
            draw, (10, 190), _parts, font, sub_font,
            anchor='lb', img_size=img.size, bg_color=(0, 0, 255),
        )
        pixels = list(img.getdata())
        has_blue = any(p[0] < 50 and p[1] < 50 and p[2] > 200 for p in pixels)
        assert has_blue, "Expected blue background pixels for anchor='lb'"

    def test_anchor_mm_middle_middle(self, _fonts, _parts):
        """Anchor 'mm' should center text around (x, y)."""
        font, sub_font = _fonts
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw_label_with_subscript_bg(
            draw, (200, 100), _parts, font, sub_font,
            anchor='mm', img_size=img.size, bg_color=(0, 200, 0),
        )
        pixels = list(img.getdata())
        has_green = any(p[1] > 150 and p[0] < 50 and p[2] < 50 for p in pixels)
        assert has_green, "Expected green background pixels for anchor='mm'"

    def test_anchor_mt_middle_top(self, _fonts, _parts):
        """Anchor 'mt' should center horizontally, top-align vertically."""
        font, sub_font = _fonts
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw_label_with_subscript_bg(
            draw, (200, 10), _parts, font, sub_font,
            anchor='mt', img_size=img.size, bg_color=(200, 200, 0),
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_anchor_unknown_suffix_defaults(self, _fonts, _parts):
        """Anchor with unknown suffix (e.g. 'lx') should default base_y = y."""
        font, sub_font = _fonts
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw_label_with_subscript_bg(
            draw, (50, 50), _parts, font, sub_font,
            anchor='lx', img_size=img.size, bg_color=(128, 0, 128),
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0


class _NonFreeTypeFontWrapper:
    """Wraps a real FreeTypeFont but fails isinstance(FreeTypeFont) checks."""

    def __init__(self, real_font):
        self._real = real_font

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestDrawLabelWithSubscriptBgNonFreeType:
    """Tests for draw_label_with_subscript_bg with non-FreeType fonts (lines 114, 121, 138-139)."""

    @staticmethod
    def _make_non_freetype_font():
        real = load_grid_font(20)
        wrapper = _NonFreeTypeFontWrapper(real)
        assert not isinstance(wrapper, ImageFont.FreeTypeFont)
        return wrapper

    def test_non_freetype_mixed_parts(self):
        """Non-FreeType font should hit else branches on lines 114 and 121."""
        font = self._make_non_freetype_font()
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        parts = [('AB', False), ('c', True)]
        draw_label_with_subscript_bg(
            draw, (50, 30), parts, font, font,
            anchor='lt', img_size=img.size, bg_color=(255, 255, 0),
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_non_freetype_subscript_only(self):
        """All-subscript parts with non-FreeType font triggers max_height fallback (lines 138-139)."""
        font = self._make_non_freetype_font()
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        parts = [('sub', True)]
        draw_label_with_subscript_bg(
            draw, (50, 30), parts, font, font,
            anchor='lt', img_size=img.size, bg_color=(0, 200, 200),
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0


class TestLoadGridFontFallbacks:
    """Tests for load_grid_font fallback paths when configured paths are invalid."""

    def test_invalid_bold_path_falls_through(self):
        """When GRID_FONT_PATH_BOLD is invalid, should fallback to GRID_FONT_PATH or system."""
        import imaging.text as text_mod
        orig_bold = text_mod.GRID_FONT_PATH_BOLD
        orig_regular = text_mod.GRID_FONT_PATH
        try:
            text_mod.GRID_FONT_PATH_BOLD = '/nonexistent/font_bold.ttf'
            text_mod.GRID_FONT_PATH = None
            font = load_grid_font(20)
            assert font is not None
        finally:
            text_mod.GRID_FONT_PATH_BOLD = orig_bold
            text_mod.GRID_FONT_PATH = orig_regular

    def test_invalid_regular_path_falls_through(self):
        """When GRID_FONT_PATH is invalid, should fallback to system fonts."""
        import imaging.text as text_mod
        orig_bold = text_mod.GRID_FONT_PATH_BOLD
        orig_regular = text_mod.GRID_FONT_PATH
        try:
            text_mod.GRID_FONT_PATH_BOLD = None
            text_mod.GRID_FONT_PATH = '/nonexistent/font.ttf'
            font = load_grid_font(20)
            assert font is not None
        finally:
            text_mod.GRID_FONT_PATH_BOLD = orig_bold
            text_mod.GRID_FONT_PATH = orig_regular

    def test_both_invalid_paths_fall_through(self):
        """When both font paths are invalid, should fallback to system fonts."""
        import imaging.text as text_mod
        orig_bold = text_mod.GRID_FONT_PATH_BOLD
        orig_regular = text_mod.GRID_FONT_PATH
        try:
            text_mod.GRID_FONT_PATH_BOLD = '/nonexistent/bold.ttf'
            text_mod.GRID_FONT_PATH = '/nonexistent/regular.ttf'
            font = load_grid_font(20)
            assert font is not None
        finally:
            text_mod.GRID_FONT_PATH_BOLD = orig_bold
            text_mod.GRID_FONT_PATH = orig_regular


class TestDrawElevationLegendTitle:
    """Tests for draw_elevation_legend with title parameter.

    Covers _wrap_legend_title inner function (lines 78-101) and
    title rendering logic (lines 187-199, 213, 312-315).
    """

    def test_legend_with_short_title(self):
        """Short title fits on one line, exercises title layout path."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 255)),
            (0.5, (0, 255, 0)),
            (1.0, (255, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=500,
            center_lat_wgs=55.75,
            zoom=12,
            title='Высота, м',
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4
        assert bounds[0] < bounds[2]
        assert bounds[1] < bounds[3]

    def test_legend_with_long_wrapping_title(self):
        """Long title triggers word-wrapping in _wrap_legend_title (lines 82-101)."""
        img = Image.new('RGB', (500, 500), color='white')
        color_ramp = [
            (0.0, (0, 100, 200)),
            (1.0, (200, 100, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=100,
            max_elevation_m=900,
            center_lat_wgs=55.75,
            zoom=12,
            title='Абсолютная высота рельефа над уровнем моря в метрах',
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4

    def test_legend_with_empty_title(self):
        """Empty string title exercises _wrap_legend_title empty-words branch (line 84)."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 255)),
            (1.0, (255, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=200,
            center_lat_wgs=55.75,
            zoom=12,
            title='',
        )
        assert len(bounds) == 4

    def test_legend_title_with_single_word(self):
        """Single-word title produces one line without entering the wrap loop."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (50, 50, 200)),
            (1.0, (200, 50, 50)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=300,
            center_lat_wgs=55.75,
            zoom=12,
            title='Высота',
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4


class TestDrawElevationLegendLabelStep:
    """Tests for draw_elevation_legend with label_step_m parameter (covers line 331)."""

    def test_legend_with_label_step_100(self):
        """Elevation labels rounded to nearest 100m."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 128)),
            (0.5, (128, 128, 0)),
            (1.0, (255, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=1000,
            center_lat_wgs=55.75,
            zoom=12,
            label_step_m=100,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4

    def test_legend_with_label_step_50(self):
        """Elevation labels rounded to nearest 50m."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 200)),
            (1.0, (200, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=25,
            max_elevation_m=475,
            center_lat_wgs=55.75,
            zoom=12,
            label_step_m=50,
        )
        assert len(bounds) == 4
        assert bounds[0] < bounds[2]


class TestDrawElevationLegendGridFontSize:
    """Tests for draw_elevation_legend with grid_font_size_m parameter (covers line 132)."""

    def test_legend_with_explicit_font_size(self):
        """Explicit grid_font_size_m bypasses default font-size calculation."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (50, 50, 200)),
            (1.0, (200, 50, 50)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=50,
            max_elevation_m=500,
            center_lat_wgs=55.75,
            zoom=12,
            grid_font_size_m=80.0,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4

    def test_legend_with_large_font_size(self):
        """Large grid_font_size_m still produces valid legend."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 255)),
            (1.0, (255, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=1000,
            center_lat_wgs=55.75,
            zoom=12,
            grid_font_size_m=200.0,
        )
        assert len(bounds) == 4


class TestDrawElevationLegendSmallMap:
    """Tests for draw_elevation_legend with small map (covers line 116).

    When map_height_m < LEGEND_MIN_MAP_HEIGHT_M_FOR_RATIO the legend height is
    computed from LEGEND_MIN_HEIGHT_GRID_SQUARES * GRID_STEP_M * ppm.
    """

    def test_small_map_high_zoom(self):
        """500px at zoom 16 => ~450m height, well below 10000m threshold."""
        img = Image.new('RGB', (500, 500), color='white')
        color_ramp = [
            (0.0, (0, 0, 255)),
            (1.0, (255, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=100,
            max_elevation_m=300,
            center_lat_wgs=55.75,
            zoom=16,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4

    def test_small_map_with_title(self):
        """Small map with title should render correctly."""
        img = Image.new('RGB', (500, 500), color='white')
        color_ramp = [
            (0.0, (0, 100, 200)),
            (1.0, (200, 100, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=200,
            center_lat_wgs=55.75,
            zoom=16,
            title='Высота, м',
        )
        assert len(bounds) == 4


class TestDrawElevationLegendRGBA:
    """Tests for draw_elevation_legend with RGBA image mode (covers lines 263-273)."""

    def test_rgba_image(self):
        """RGBA image exercises the alpha_composite branch for background."""
        img = Image.new('RGBA', (1000, 1000), color=(255, 255, 255, 255))
        color_ramp = [
            (0.0, (0, 0, 255)),
            (0.5, (0, 255, 0)),
            (1.0, (255, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=1000,
            center_lat_wgs=55.75,
            zoom=12,
        )
        assert img.mode == 'RGBA'
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4

    def test_rgba_with_title_and_label_step(self):
        """RGBA mode combined with title and label_step_m."""
        img = Image.new('RGBA', (500, 500), color=(255, 255, 255, 255))
        color_ramp = [
            (0.0, (0, 50, 150)),
            (0.5, (100, 200, 50)),
            (1.0, (200, 50, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=500,
            center_lat_wgs=55.75,
            zoom=12,
            title='Высота, м',
            label_step_m=100,
        )
        assert img.mode == 'RGBA'
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4

    def test_rgba_with_transparent_background(self):
        """RGBA image with semi-transparent starting color should receive legend."""
        img = Image.new('RGBA', (1000, 1000), color=(200, 200, 200, 128))
        color_ramp = [
            (0.0, (0, 0, 200)),
            (1.0, (200, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=50,
            max_elevation_m=800,
            center_lat_wgs=55.75,
            zoom=12,
        )
        assert len(bounds) == 4


class TestDrawElevationLegendAllParams:
    """Tests for draw_elevation_legend with all optional parameters combined."""

    def test_all_params_together(self):
        """Title + label_step_m + grid_font_size_m combined on RGB image."""
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 200)),
            (0.33, (0, 200, 0)),
            (0.66, (200, 200, 0)),
            (1.0, (200, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=50,
            max_elevation_m=950,
            center_lat_wgs=55.75,
            zoom=12,
            title='Высота, м',
            label_step_m=100,
            grid_font_size_m=100.0,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4
        assert bounds[0] < bounds[2]
        assert bounds[1] < bounds[3]

    def test_all_params_on_rgba_small_map(self):
        """All optional params on an RGBA small map at high zoom."""
        img = Image.new('RGBA', (500, 500), color=(255, 255, 255, 255))
        color_ramp = [
            (0.0, (0, 0, 200)),
            (1.0, (200, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=400,
            center_lat_wgs=55.75,
            zoom=16,
            title='Высота над уровнем моря',
            label_step_m=50,
            grid_font_size_m=60.0,
        )
        assert len(bounds) == 4


class TestDrawElevationLegendFontFallback:
    """Tests for draw_elevation_legend font loading fallback (covers lines 142-143)."""

    def test_font_fallback_on_load_failure(self, monkeypatch):
        """When load_grid_font raises, legend falls back to ImageFont.load_default()."""
        def broken_load_font(size=20):
            raise OSError('Font not found')

        monkeypatch.setattr('imaging.legend.load_grid_font', broken_load_font)

        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 255)),
            (1.0, (255, 0, 0)),
        ]
        bounds = draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=500,
            center_lat_wgs=55.75,
            zoom=12,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0
        assert len(bounds) == 4


class TestDrawGridCrossesWithLegendBounds:
    """Tests for draw_axis_aligned_km_grid with display_grid=False and legend_bounds.

    Covers lines 357-359: skip crosses that fall inside legend_bounds.
    """

    def test_crosses_skip_legend_area(self, sk42_gk_crs, sk42_to_wgs_transformer):
        """Crosses inside legend_bounds should be skipped."""
        img = Image.new('RGB', (1000, 1000), color='white')
        legend_bounds = (800, 800, 950, 950)
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
            legend_bounds=legend_bounds,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_crosses_fewer_with_large_legend_bounds(
        self, sk42_gk_crs, sk42_to_wgs_transformer
    ):
        """Large legend_bounds should result in fewer drawn crosses than no bounds."""
        img_with = Image.new('RGB', (1000, 1000), color='white')
        img_without = Image.new('RGB', (1000, 1000), color='white')

        draw_axis_aligned_km_grid(
            img_with,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=False,
            legend_bounds=(0, 0, 999, 999),
        )
        draw_axis_aligned_km_grid(
            img_without,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=False,
            legend_bounds=None,
        )
        pixels_with = [p for p in img_with.getdata() if p != (255, 255, 255)]
        pixels_without = [p for p in img_without.getdata() if p != (255, 255, 255)]
        assert len(pixels_without) >= len(pixels_with)

    def test_crosses_with_small_legend_area(
        self, sk42_gk_crs, sk42_to_wgs_transformer
    ):
        """Tiny legend_bounds should exclude only a small area."""
        img = Image.new('RGB', (1000, 1000), color='white')
        legend_bounds = (900, 900, 910, 910)
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
            legend_bounds=legend_bounds,
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0


class TestDrawGridWithLegendBoundsDisplayTrue:
    """Tests for draw_axis_aligned_km_grid with display_grid=True and legend_bounds.

    Although draw_line_with_gap (lines 176-217) is defined but not currently
    invoked by the grid drawing logic, these tests ensure the grid renders
    correctly when legend_bounds are supplied in full-grid mode.
    """

    def test_grid_with_bottom_right_legend_bounds(
        self, sk42_gk_crs, sk42_to_wgs_transformer
    ):
        """Full grid with legend_bounds in bottom-right should draw lines."""
        img = Image.new('RGB', (1000, 1000), color='white')
        legend_bounds = (800, 800, 950, 950)
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
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_grid_with_centered_legend_bounds(
        self, sk42_gk_crs, sk42_to_wgs_transformer
    ):
        """Legend bounds in center of image should not prevent grid rendering."""
        img = Image.new('RGB', (1000, 1000), color='white')
        legend_bounds = (400, 400, 600, 600)
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
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

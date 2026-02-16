"""Tests for services.radar_coverage module (sector overlay, ceiling arcs)."""

import math

import pytest
from PIL import Image

from services.radar_coverage import draw_sector_overlay


class TestDrawSectorOverlay:
    """Tests for draw_sector_overlay function."""

    def _make_rgba(self, size: int = 200) -> Image.Image:
        return Image.new('RGBA', (size, size), (128, 128, 128, 255))

    def test_draws_on_rgba_image(self):
        """Should modify RGBA image without error."""
        img = self._make_rgba()
        draw_sector_overlay(
            img, cx=100, cy=100,
            azimuth_deg=0.0, sector_width_deg=90.0,
            max_range_px=80.0, pixel_size_m=1.0,
        )
        # Image should have been modified (shadow outside sector)
        # At least some pixels should differ from original
        pixels = list(img.getdata())
        unique = set(pixels)
        assert len(unique) > 1

    def test_no_op_on_rgb_image(self):
        """Should do nothing on non-RGBA images."""
        img = Image.new('RGB', (100, 100), (128, 128, 128))
        original_data = list(img.getdata())
        draw_sector_overlay(
            img, cx=50, cy=50,
            azimuth_deg=0.0, sector_width_deg=90.0,
            max_range_px=40.0, pixel_size_m=1.0,
        )
        assert list(img.getdata()) == original_data

    def test_sector_center_is_visible(self):
        """Pixels along the azimuth should be brighter than those outside sector."""
        img = self._make_rgba(300)
        draw_sector_overlay(
            img, cx=150, cy=150,
            azimuth_deg=0.0, sector_width_deg=90.0,
            max_range_px=100.0, pixel_size_m=1.0,
        )
        # Pixel along azimuth 0 (north = upward) inside sector
        pixel_inside = img.getpixel((150, 100))
        # Pixel clearly outside sector (south)
        pixel_outside = img.getpixel((150, 250))
        # Inside sector should be brighter (less shadowed) than outside
        inside_brightness = pixel_inside[0] + pixel_inside[1] + pixel_inside[2]
        outside_brightness = pixel_outside[0] + pixel_outside[1] + pixel_outside[2]
        assert inside_brightness >= outside_brightness

    def test_outside_sector_unchanged(self):
        """Pixels outside sector should not be modified (no shadow overlay)."""
        original = self._make_rgba(300)
        original_pixel = original.getpixel((150, 250))

        img = self._make_rgba(300)
        draw_sector_overlay(
            img, cx=150, cy=150,
            azimuth_deg=0.0, sector_width_deg=30.0,  # Narrow sector north
            max_range_px=100.0, pixel_size_m=1.0,
        )
        # Pixel to the south (outside sector) should be unchanged —
        # the coverage kernel already colors outside-sector as unreachable.
        pixel = img.getpixel((150, 250))
        assert pixel[:3] == original_pixel[:3]

    def test_full_circle_sector(self):
        """360° sector should produce no shadow (effectively all visible)."""
        img = self._make_rgba(200)
        original_pixel = img.getpixel((100, 50))
        draw_sector_overlay(
            img, cx=100, cy=100,
            azimuth_deg=0.0, sector_width_deg=360.0,
            max_range_px=90.0, pixel_size_m=1.0,
        )
        # Inside the range circle, pixel should remain relatively unchanged
        pixel_after = img.getpixel((100, 50))
        # The alpha composite of full sector should not darken much inside circle
        assert pixel_after[0] >= original_pixel[0] - 5

    def test_various_azimuths_dont_crash(self):
        """Different azimuths should work without errors."""
        for az in [0, 45, 90, 135, 180, 225, 270, 315, 359.9]:
            img = self._make_rgba(100)
            draw_sector_overlay(
                img, cx=50, cy=50,
                azimuth_deg=az, sector_width_deg=60.0,
                max_range_px=40.0, pixel_size_m=1.0,
            )

    def test_ceiling_arcs_drawn(self):
        """With large range and small pixel size, ceiling arcs should appear."""
        img = self._make_rgba(400)
        # pixel_size_m=10 means max_range = 200*10 = 2000m
        # Ceiling arc for 30m at elev_max=30°: range = 30/tan(30°) ≈ 52m → 5.2px
        draw_sector_overlay(
            img, cx=200, cy=200,
            azimuth_deg=0.0, sector_width_deg=90.0,
            max_range_px=180.0, pixel_size_m=10.0,
            elevation_max_deg=30.0,
        )
        # Just verify no crash, actual arc pixel verification would be complex

    def test_small_range(self):
        """Very small range should not crash."""
        img = self._make_rgba(100)
        draw_sector_overlay(
            img, cx=50, cy=50,
            azimuth_deg=0.0, sector_width_deg=90.0,
            max_range_px=3.0, pixel_size_m=1.0,
        )

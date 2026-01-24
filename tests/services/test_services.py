"""Tests for services package utilities."""

import pytest
from PIL import Image, ImageDraw

from services.drawing_utils import (
    compute_rotated_position,
    draw_center_cross,
    draw_control_point_marker,
)
from services.overlay_utils import (
    blend_with_grayscale_base,
    composite_overlay_on_base,
    create_contour_gap_at_labels,
)


class TestDrawingUtils:
    """Tests for drawing_utils module."""

    def test_draw_control_point_marker_returns_bottom_y(self):
        """Test that marker returns correct bottom Y coordinate."""
        img = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        bottom_y = draw_control_point_marker(
            draw,
            center=(50, 50),
            size_px=20,
            color=(255, 0, 0),
        )
        
        assert bottom_y == 60  # 50 + 20//2
        img.close()

    def test_draw_control_point_marker_draws_triangle(self):
        """Test that marker actually draws something."""
        img = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        draw_control_point_marker(
            draw,
            center=(50, 50),
            size_px=20,
            color=(255, 0, 0),
        )
        
        # Check that center area has red color
        pixel = img.getpixel((50, 50))
        assert pixel == (255, 0, 0)
        img.close()

    def test_draw_center_cross(self):
        """Test that cross is drawn at center."""
        img = Image.new('RGB', (100, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        draw_center_cross(
            draw,
            center=(50, 50),
            length_px=20,
            width_px=2,
            color=(0, 0, 255),
        )
        
        # Check that center has blue color
        pixel = img.getpixel((50, 50))
        assert pixel == (0, 0, 255)
        img.close()

    def test_compute_rotated_position_no_rotation(self):
        """Test rotation with 0 degrees."""
        x, y = compute_rotated_position(10, 20, 0, 0, 0)
        assert x == pytest.approx(10, abs=0.001)
        assert y == pytest.approx(20, abs=0.001)

    def test_compute_rotated_position_90_degrees(self):
        """Test rotation with 90 degrees."""
        x, y = compute_rotated_position(10, 0, 0, 0, 90)
        assert x == pytest.approx(0, abs=0.001)
        assert y == pytest.approx(-10, abs=0.001)

    def test_compute_rotated_position_180_degrees(self):
        """Test rotation with 180 degrees."""
        x, y = compute_rotated_position(10, 0, 0, 0, 180)
        assert x == pytest.approx(-10, abs=0.001)
        assert y == pytest.approx(0, abs=0.001)


class TestOverlayUtils:
    """Tests for overlay_utils module."""

    def test_composite_overlay_on_base(self):
        """Test basic overlay compositing."""
        base = Image.new('RGB', (100, 100), (255, 255, 255))
        overlay = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        
        result = composite_overlay_on_base(base, overlay)
        
        assert result.mode == 'RGB'
        assert result.size == (100, 100)
        # Result should be pinkish (blend of white and red)
        pixel = result.getpixel((50, 50))
        assert pixel[0] == 255  # Red channel stays 255
        assert pixel[1] < 255  # Green reduced
        assert pixel[2] < 255  # Blue reduced
        
        base.close()
        overlay.close()
        result.close()

    def test_composite_overlay_with_resize(self):
        """Test overlay compositing with resize."""
        base = Image.new('RGB', (100, 100), (255, 255, 255))
        overlay = Image.new('RGBA', (50, 50), (255, 0, 0, 128))
        
        result = composite_overlay_on_base(base, overlay, target_size=(100, 100))
        
        assert result.size == (100, 100)
        
        base.close()
        overlay.close()
        result.close()

    def test_blend_with_grayscale_base(self):
        """Test blending with grayscale conversion."""
        base = Image.new('RGB', (100, 100), (100, 150, 200))
        overlay = Image.new('RGB', (100, 100), (255, 0, 0))
        
        result = blend_with_grayscale_base(base, overlay, alpha=0.5)
        
        assert result.mode == 'RGBA'
        assert result.size == (100, 100)
        
        base.close()
        overlay.close()
        result.close()

    def test_create_contour_gap_at_labels(self):
        """Test gap creation at label positions."""
        overlay = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        
        bboxes = [(40, 40, 60, 60)]
        create_contour_gap_at_labels(overlay, bboxes, gap_padding=5)
        
        # Center should be transparent now
        pixel = overlay.getpixel((50, 50))
        assert pixel[3] == 0  # Alpha = 0 (transparent)
        
        # Corner should still be red
        pixel = overlay.getpixel((10, 10))
        assert pixel == (255, 0, 0, 255)
        
        overlay.close()

    def test_create_contour_gap_empty_bboxes(self):
        """Test gap creation with empty bbox list."""
        overlay = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        
        create_contour_gap_at_labels(overlay, [], gap_padding=5)
        
        # Should remain unchanged
        pixel = overlay.getpixel((50, 50))
        assert pixel == (255, 0, 0, 255)
        
        overlay.close()

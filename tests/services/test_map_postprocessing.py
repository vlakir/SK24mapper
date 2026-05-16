"""Tests for services.map_postprocessing module."""

import math

import pytest
from PIL import Image

from services.map_postprocessing import (
    compute_control_point_image_coords,
    draw_center_cross_on_image,
    draw_control_point_triangle,
    draw_radar_marker,
)


class TestDrawCenterCrossOnImage:
    """Tests for draw_center_cross_on_image function."""

    def test_draws_cross_at_center(self):
        """Should draw cross at image center."""
        # Use black background since CENTER_CROSS_COLOR is white (255, 255, 255)
        img = Image.new('RGB', (100, 100), color=(0, 0, 0))
        # CENTER_CROSS_LENGTH_M=40, so at meters_per_px=2.0, cross is 20px (fits in 100x100)
        draw_center_cross_on_image(img, meters_per_px=2.0)
        
        # Check that center pixel is not black (cross was drawn)
        center_pixel = img.getpixel((50, 50))
        assert center_pixel != (0, 0, 0)

    def test_handles_zero_meters_per_px(self):
        """Should not crash with zero meters_per_px."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        # Should not raise
        draw_center_cross_on_image(img, meters_per_px=0.0)

    def test_handles_negative_meters_per_px(self):
        """Should not crash with negative meters_per_px."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        # Should not raise
        draw_center_cross_on_image(img, meters_per_px=-1.0)


class TestDrawControlPointTriangle:
    """Tests for draw_control_point_triangle function."""

    def test_draws_triangle(self):
        """Should draw triangle at specified position."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw_control_point_triangle(img, cx_img=50.0, cy_img=50.0, meters_per_px=1.0)
        
        # Check that some pixels near center are not white
        center_pixel = img.getpixel((50, 50))
        assert center_pixel != (255, 255, 255)

    def test_handles_rotation(self):
        """Should handle rotation parameter."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        # Should not raise with rotation
        draw_control_point_triangle(
            img, cx_img=50.0, cy_img=50.0, meters_per_px=1.0, rotation_deg=45.0
        )

    def test_handles_zero_meters_per_px(self):
        """Should not crash with zero meters_per_px."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw_control_point_triangle(img, cx_img=50.0, cy_img=50.0, meters_per_px=0.0)


class TestComputeControlPointImageCoords:
    """Tests for compute_control_point_image_coords function."""

    def test_center_point_at_center(self):
        """Control point at map center should be at image center."""
        def mock_latlng_to_pixel(lat, lng, zoom):
            # Simple mock: 1 degree = 1000 pixels
            return lng * 1000, lat * 1000
        
        cx, cy = compute_control_point_image_coords(
            cp_lat_wgs=50.0,
            cp_lng_wgs=30.0,
            center_lat_wgs=50.0,
            center_lng_wgs=30.0,
            zoom=14,
            eff_scale=1,
            img_width=1000,
            img_height=1000,
            rotation_deg=0.0,
            latlng_to_pixel_xy_func=mock_latlng_to_pixel,
        )
        
        assert cx == pytest.approx(500.0)
        assert cy == pytest.approx(500.0)

    def test_offset_point(self):
        """Control point offset from center should be offset in image."""
        def mock_latlng_to_pixel(lat, lng, zoom):
            return lng * 1000, lat * 1000
        
        cx, cy = compute_control_point_image_coords(
            cp_lat_wgs=50.0,
            cp_lng_wgs=30.1,  # 0.1 degree east
            center_lat_wgs=50.0,
            center_lng_wgs=30.0,
            zoom=14,
            eff_scale=1,
            img_width=1000,
            img_height=1000,
            rotation_deg=0.0,
            latlng_to_pixel_xy_func=mock_latlng_to_pixel,
        )
        
        # Should be offset to the right (east)
        assert cx > 500.0

    def test_rotation_applied(self):
        """Rotation should affect control point position."""
        def mock_latlng_to_pixel(lat, lng, zoom):
            return lng * 1000, lat * 1000
        
        # Point to the east of center
        cx_no_rot, cy_no_rot = compute_control_point_image_coords(
            cp_lat_wgs=50.0,
            cp_lng_wgs=30.1,
            center_lat_wgs=50.0,
            center_lng_wgs=30.0,
            zoom=14,
            eff_scale=1,
            img_width=1000,
            img_height=1000,
            rotation_deg=0.0,
            latlng_to_pixel_xy_func=mock_latlng_to_pixel,
        )
        
        cx_rot, cy_rot = compute_control_point_image_coords(
            cp_lat_wgs=50.0,
            cp_lng_wgs=30.1,
            center_lat_wgs=50.0,
            center_lng_wgs=30.0,
            zoom=14,
            eff_scale=1,
            img_width=1000,
            img_height=1000,
            rotation_deg=90.0,
            latlng_to_pixel_xy_func=mock_latlng_to_pixel,
        )
        
        # With 90 degree rotation, x offset should become y offset
        assert cx_rot != cx_no_rot or cy_rot != cy_no_rot

    def test_eff_scale_multiplier(self):
        """eff_scale should multiply the offset."""
        def mock_latlng_to_pixel(lat, lng, zoom):
            return lng * 1000, lat * 1000
        
        cx_scale1, _ = compute_control_point_image_coords(
            cp_lat_wgs=50.0,
            cp_lng_wgs=30.1,
            center_lat_wgs=50.0,
            center_lng_wgs=30.0,
            zoom=14,
            eff_scale=1,
            img_width=1000,
            img_height=1000,
            rotation_deg=0.0,
            latlng_to_pixel_xy_func=mock_latlng_to_pixel,
        )
        
        cx_scale2, _ = compute_control_point_image_coords(
            cp_lat_wgs=50.0,
            cp_lng_wgs=30.1,
            center_lat_wgs=50.0,
            center_lng_wgs=30.0,
            zoom=14,
            eff_scale=2,
            img_width=1000,
            img_height=1000,
            rotation_deg=0.0,
            latlng_to_pixel_xy_func=mock_latlng_to_pixel,
        )
        
        # Offset from center should be doubled
        offset1 = cx_scale1 - 500.0
        offset2 = cx_scale2 - 500.0
        assert offset2 == pytest.approx(offset1 * 2)


class TestDrawRadarMarker:
    """Tests for draw_radar_marker function."""

    def test_draws_marker(self):
        """Should draw diamond marker at specified position."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw_radar_marker(img, cx_img=50.0, cy_img=50.0, meters_per_px=1.0)
        # Check that some pixels near center are not white (marker drawn)
        center_pixel = img.getpixel((50, 50))
        assert center_pixel != (255, 255, 255)

    def test_handles_azimuth(self):
        """Should handle azimuth parameter for direction line."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw_radar_marker(
            img, cx_img=50.0, cy_img=50.0, meters_per_px=1.0,
            azimuth_deg=90.0,
        )
        # Should not raise

    def test_handles_rotation(self):
        """Should handle rotation parameter."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw_radar_marker(
            img, cx_img=50.0, cy_img=50.0, meters_per_px=1.0,
            azimuth_deg=45.0, rotation_deg=30.0,
        )

    def test_handles_zero_meters_per_px(self):
        """Should not crash with zero meters_per_px."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw_radar_marker(img, cx_img=50.0, cy_img=50.0, meters_per_px=0.0)

    def test_handles_negative_meters_per_px(self):
        """Should not crash with negative meters_per_px."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw_radar_marker(img, cx_img=50.0, cy_img=50.0, meters_per_px=-1.0)

    def test_draws_on_rgba(self):
        """Should work on RGBA images."""
        img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 255))
        draw_radar_marker(img, cx_img=50.0, cy_img=50.0, meters_per_px=1.0)
        center_pixel = img.getpixel((50, 50))
        assert center_pixel != (255, 255, 255, 255)

"""Tests for services.dem_colorizer module."""

import numpy as np
import pytest
from PIL import Image

from services.color_utils import ColorMapper
from services.dem_colorizer import colorize_dem_overlap, colorize_dem_tile_numpy


class TestColorizeDemTileNumpy:
    """Tests for colorize_dem_tile_numpy function."""

    def test_basic_colorization(self):
        """Should colorize DEM array correctly."""
        # Create a simple color ramp (green to red)
        color_ramp = [
            (0.0, (0, 255, 0)),    # green at low
            (1.0, (255, 0, 0)),    # red at high
        ]
        mapper = ColorMapper(color_ramp)
        
        # Create a 2x2 DEM tile with known elevation values
        # terrain-rgb formula: elevation = -10000 + (R*256*256 + G*256 + B) * 0.1
        # For elevation 0m: R*65536 + G*256 + B = 100000 -> R=1, G=134, B=160
        # For elevation 100m: R*65536 + G*256 + B = 101000 -> R=1, G=138, B=136
        dem_rgb = np.array([
            [[1, 134, 160], [1, 138, 136]],  # 0m, 100m
            [[1, 142, 112], [1, 146, 88]],   # 200m, 300m
        ], dtype=np.uint8)
        
        result = colorize_dem_tile_numpy(dem_rgb, 0.0, 300.0, mapper)
        
        assert result.shape == (2, 2, 3)
        assert result.dtype == np.uint8

    def test_clamping_out_of_range(self):
        """Should clamp values outside elevation range."""
        color_ramp = [
            (0.0, (0, 0, 0)),
            (1.0, (255, 255, 255)),
        ]
        mapper = ColorMapper(color_ramp)
        
        # Very low elevation (below range)
        dem_rgb = np.array([[[0, 0, 0]]], dtype=np.uint8)  # -10000m
        result = colorize_dem_tile_numpy(dem_rgb, 0.0, 100.0, mapper)
        
        # Should be clamped to black (low end)
        assert result[0, 0, 0] == 0

    def test_equal_bounds(self):
        """Should handle equal lo and hi bounds."""
        color_ramp = [(0.0, (128, 128, 128)), (1.0, (128, 128, 128))]
        mapper = ColorMapper(color_ramp)
        
        dem_rgb = np.array([[[1, 134, 160]]], dtype=np.uint8)
        # Should not crash with division by zero
        result = colorize_dem_tile_numpy(dem_rgb, 100.0, 100.0, mapper)
        assert result.shape == (1, 1, 3)


class TestColorizeDemOverlap:
    """Tests for colorize_dem_overlap function."""

    def test_overlap_extraction(self):
        """Should extract and colorize overlap region correctly."""
        color_ramp = [(0.0, (0, 255, 0)), (1.0, (255, 0, 0))]
        mapper = ColorMapper(color_ramp)
        
        # Create a 4x4 DEM tile image
        dem_data = np.zeros((4, 4, 3), dtype=np.uint8)
        dem_data[:, :] = [1, 134, 160]  # ~0m elevation
        dem_img = Image.fromarray(dem_data, mode='RGB')
        
        # Extract 2x2 overlap from center
        overlap = (1, 1, 3, 3)  # x0, y0, x1, y1
        tile_base_x = 0
        tile_base_y = 0
        
        result, dest_x, dest_y = colorize_dem_overlap(
            dem_img, overlap, tile_base_x, tile_base_y, 0.0, 100.0, mapper
        )
        
        assert result.shape == (2, 2, 3)
        assert dest_x == 1
        assert dest_y == 1

    def test_offset_tile_base(self):
        """Should handle non-zero tile base coordinates."""
        color_ramp = [(0.0, (100, 100, 100)), (1.0, (200, 200, 200))]
        mapper = ColorMapper(color_ramp)
        
        dem_data = np.ones((4, 4, 3), dtype=np.uint8) * 128
        dem_img = Image.fromarray(dem_data, mode='RGB')
        
        # Tile starts at (100, 100) in global coords
        overlap = (101, 101, 103, 103)
        tile_base_x = 100
        tile_base_y = 100
        
        result, dest_x, dest_y = colorize_dem_overlap(
            dem_img, overlap, tile_base_x, tile_base_y, 0.0, 100.0, mapper
        )
        
        assert result.shape == (2, 2, 3)
        assert dest_x == 101
        assert dest_y == 101

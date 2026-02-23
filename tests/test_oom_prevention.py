"""Tests for OOM prevention: assemble_dem direct-crop and memory estimation."""

from unittest.mock import patch

import numpy as np
import pytest

from geo.topography import assemble_dem
from shared.memory_estimation import (
    choose_safe_zoom,
    estimate_map_memory_mb,
    get_available_memory_mb,
)


class TestAssembleDemDirectCrop:
    """Tests for the direct-crop assemble_dem implementation."""

    def test_basic_assembly(self):
        """Full canvas equals crop: all tiles contribute."""
        tile_px = 4
        tiles_x, tiles_y = 2, 2
        tiles = [
            np.full((tile_px, tile_px), float(i), dtype=np.float32)
            for i in range(tiles_x * tiles_y)
        ]
        crop_rect = (0, 0, tile_px * tiles_x, tile_px * tiles_y)
        result = assemble_dem(tiles, tiles_x, tiles_y, tile_px, crop_rect)

        assert result.shape == (tile_px * tiles_y, tile_px * tiles_x)
        # Top-left tile = 0, top-right = 1, bottom-left = 2, bottom-right = 3
        assert result[0, 0] == 0.0
        assert result[0, tile_px] == 1.0
        assert result[tile_px, 0] == 2.0
        assert result[tile_px, tile_px] == 3.0

    def test_crop_subset(self):
        """Crop extracts only a portion of the tiled area."""
        tile_px = 4
        tiles_x, tiles_y = 3, 3
        tiles = [
            np.full((tile_px, tile_px), float(i), dtype=np.float32)
            for i in range(tiles_x * tiles_y)
        ]
        # Crop center region
        crop_rect = (2, 2, 4, 4)
        result = assemble_dem(tiles, tiles_x, tiles_y, tile_px, crop_rect)

        assert result.shape == (4, 4)
        assert result.dtype == np.float32

    def test_tiles_freed_after_assembly(self):
        """Tiles should be set to None after processing."""
        tile_px = 4
        tiles_x, tiles_y = 2, 2
        tiles = [
            np.full((tile_px, tile_px), 1.0, dtype=np.float32)
            for _ in range(tiles_x * tiles_y)
        ]
        crop_rect = (0, 0, tile_px * tiles_x, tile_px * tiles_y)
        assemble_dem(tiles, tiles_x, tiles_y, tile_px, crop_rect)

        for tile in tiles:
            assert tile is None, 'Tile should be freed (set to None) after assembly'

    def test_non_intersecting_tiles_ignored(self):
        """Tiles outside crop_rect should not affect result."""
        tile_px = 4
        tiles_x, tiles_y = 3, 1
        tiles = [
            np.full((tile_px, tile_px), float(i + 1), dtype=np.float32)
            for i in range(3)
        ]
        # Crop only the middle tile
        crop_rect = (tile_px, 0, tile_px, tile_px)
        result = assemble_dem(tiles, tiles_x, tiles_y, tile_px, crop_rect)

        assert result.shape == (tile_px, tile_px)
        np.testing.assert_array_equal(result, 2.0)  # middle tile value

    def test_partial_tile_intersection(self):
        """When crop_rect partially overlaps a tile."""
        tile_px = 4
        tiles_x, tiles_y = 1, 1
        tile = np.arange(tile_px * tile_px, dtype=np.float32).reshape(tile_px, tile_px)
        tiles = [tile]
        # Crop bottom-right 2x2
        crop_rect = (2, 2, 2, 2)
        result = assemble_dem(tiles, tiles_x, tiles_y, tile_px, crop_rect)

        assert result.shape == (2, 2)
        expected = tile[2:4, 2:4]
        np.testing.assert_array_equal(result, expected)

    def test_ramp_values(self):
        """Verify exact values with a ramp DEM."""
        tile_px = 4
        tiles_x, tiles_y = 2, 1
        tile_a = np.arange(16, dtype=np.float32).reshape(4, 4)
        tile_b = np.arange(16, 32, dtype=np.float32).reshape(4, 4)
        tiles = [tile_a, tile_b]
        crop_rect = (0, 0, 8, 4)
        result = assemble_dem(tiles, tiles_x, tiles_y, tile_px, crop_rect)

        np.testing.assert_array_equal(result[:, :4], tile_a)
        np.testing.assert_array_equal(result[:, 4:], tile_b)


class TestEstimateMapMemory:
    """Tests for estimate_map_memory_mb."""

    def test_basic_estimation(self):
        """Check that estimation returns positive values."""
        est = estimate_map_memory_mb(
            tiles_count=100,
            eff_tile_px=256,
            crop_w=2000,
            crop_h=2000,
        )
        assert est['tiles_mb'] > 0
        assert est['canvas_mb'] > 0
        assert est['peak_mb'] > 0
        assert est['dem_mb'] == 0  # no DEM

    def test_dem_estimation(self):
        """DEM mode adds extra memory."""
        est_no_dem = estimate_map_memory_mb(
            tiles_count=100, eff_tile_px=256,
            crop_w=2000, crop_h=2000, is_dem=False,
        )
        est_dem = estimate_map_memory_mb(
            tiles_count=100, eff_tile_px=256,
            crop_w=2000, crop_h=2000, is_dem=True,
        )
        assert est_dem['dem_mb'] > 0
        assert est_dem['peak_mb'] > est_no_dem['peak_mb']

    def test_contours_no_extra_memory(self):
        """Contours via PIL ImageDraw don't add extra memory (no numpy copy)."""
        est_no = estimate_map_memory_mb(
            tiles_count=100, eff_tile_px=256,
            crop_w=2000, crop_h=2000, has_contours=False,
        )
        est_yes = estimate_map_memory_mb(
            tiles_count=100, eff_tile_px=256,
            crop_w=2000, crop_h=2000, has_contours=True,
        )
        assert est_yes['peak_mb'] == est_no['peak_mb']
        assert est_yes['contour_mb'] == 0.0

    def test_known_values(self):
        """Verify calculation for known input."""
        mb = 1024 * 1024
        # 100 tiles, 256px, 1000x1000 crop
        est = estimate_map_memory_mb(
            tiles_count=100, eff_tile_px=256,
            crop_w=1000, crop_h=1000,
        )
        # tiles: 100 * 256*256 * 3 / MB
        expected_tiles = 100 * 256 * 256 * 3 / mb
        assert abs(est['tiles_mb'] - round(expected_tiles, 1)) < 0.2


class TestChooseSafeZoom:
    """Tests for choose_safe_zoom."""

    def test_no_reduction_with_plenty_ram(self):
        """With 32 GB available, zoom should not be reduced for small maps."""
        with patch('shared.memory_estimation.get_available_memory_mb', return_value=32000):
            zoom, info = choose_safe_zoom(
                center_lat=55.0,
                width_m=5000, height_m=5000,
                desired_zoom=15, eff_scale=1,
                max_pixels=500_000_000,
            )
        assert zoom == 15
        assert not info['zoom_reduced']

    def test_reduction_with_low_ram(self):
        """With very low RAM, zoom should be reduced."""
        with patch('shared.memory_estimation.get_available_memory_mb', return_value=500):
            zoom, info = choose_safe_zoom(
                center_lat=55.0,
                width_m=50000, height_m=50000,
                desired_zoom=18, eff_scale=2,
                max_pixels=500_000_000,
                is_dem=True,
            )
        assert zoom < 18
        assert info['zoom_reduced']

    def test_pixel_limit_respected(self):
        """Pixel limit should still be respected."""
        with patch('shared.memory_estimation.get_available_memory_mb', return_value=32000):
            zoom, info = choose_safe_zoom(
                center_lat=55.0,
                width_m=100000, height_m=100000,
                desired_zoom=20, eff_scale=2,
                max_pixels=100_000_000,
            )
        # Should be reduced by pixel limit
        assert zoom < 20

    def test_psutil_unavailable(self):
        """When psutil unavailable (returns 0), skip memory check."""
        with patch('shared.memory_estimation.get_available_memory_mb', return_value=0):
            zoom, info = choose_safe_zoom(
                center_lat=55.0,
                width_m=5000, height_m=5000,
                desired_zoom=15, eff_scale=1,
                max_pixels=500_000_000,
            )
        assert zoom == 15

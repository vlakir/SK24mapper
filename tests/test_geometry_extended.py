"""Extended tests for geometry module."""

import pytest

from geo.geometry import tile_overlap_rect_common


class TestTileOverlapRectCommon:
    """Tests for tile_overlap_rect_common function."""

    def test_full_overlap(self):
        """Tile fully inside crop rect should return tile bounds."""
        crop_rect = (0, 0, 1000, 1000)
        result = tile_overlap_rect_common(1, 1, crop_rect, 256)
        assert result == (256, 256, 512, 512)

    def test_partial_overlap_right(self):
        """Tile partially overlapping on right should return intersection."""
        crop_rect = (0, 0, 300, 1000)
        result = tile_overlap_rect_common(1, 0, crop_rect, 256)
        # Tile at (256, 0) to (512, 256), crop ends at x=300
        assert result == (256, 0, 300, 256)

    def test_partial_overlap_bottom(self):
        """Tile partially overlapping on bottom should return intersection."""
        crop_rect = (0, 0, 1000, 300)
        result = tile_overlap_rect_common(0, 1, crop_rect, 256)
        # Tile at (0, 256) to (256, 512), crop ends at y=300
        assert result == (0, 256, 256, 300)

    def test_no_overlap_outside_right(self):
        """Tile completely outside on right should return None."""
        crop_rect = (0, 0, 200, 200)
        result = tile_overlap_rect_common(1, 0, crop_rect, 256)
        assert result is None

    def test_no_overlap_outside_bottom(self):
        """Tile completely outside on bottom should return None."""
        crop_rect = (0, 0, 200, 200)
        result = tile_overlap_rect_common(0, 1, crop_rect, 256)
        assert result is None

    def test_edge_touch_no_overlap(self):
        """Tiles touching at edge should return None."""
        crop_rect = (0, 0, 256, 256)
        result = tile_overlap_rect_common(1, 0, crop_rect, 256)
        assert result is None

    def test_first_tile_full(self):
        """First tile (0, 0) fully inside should return full tile."""
        crop_rect = (0, 0, 512, 512)
        result = tile_overlap_rect_common(0, 0, crop_rect, 256)
        assert result == (0, 0, 256, 256)

    def test_crop_rect_offset(self):
        """Crop rect with offset should work correctly."""
        crop_rect = (100, 100, 400, 400)  # x=100, y=100, w=400, h=400
        result = tile_overlap_rect_common(0, 0, crop_rect, 256)
        # Tile at (0, 0) to (256, 256), crop from (100, 100) to (500, 500)
        assert result == (100, 100, 256, 256)

    def test_large_tile_size(self):
        """Should work with larger tile sizes."""
        crop_rect = (0, 0, 2000, 2000)
        result = tile_overlap_rect_common(1, 1, crop_rect, 512)
        assert result == (512, 512, 1024, 1024)

    def test_small_crop_rect(self):
        """Small crop rect inside tile should return crop rect bounds."""
        crop_rect = (100, 100, 50, 50)  # Small rect inside first tile
        result = tile_overlap_rect_common(0, 0, crop_rect, 256)
        assert result == (100, 100, 150, 150)

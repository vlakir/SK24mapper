"""Tests for geometry module."""

from geo.geometry import tile_overlap_rect_common


class TestTileOverlapRectCommon:
    """Tests for tile_overlap_rect_common function."""

    def test_full_overlap(self):
        """Tile fully inside crop rect should return tile bounds."""
        # Tile at (1, 1) with size 256, crop rect covers larger area
        result = tile_overlap_rect_common(
            tx=1, ty=1, crop_rect=(0, 0, 1000, 1000), full_eff_tile_px=256
        )
        assert result == (256, 256, 512, 512)

    def test_partial_overlap_right(self):
        """Tile partially overlapping crop rect on right side."""
        # Tile at (0, 0) with size 256, crop rect starts at 128
        result = tile_overlap_rect_common(
            tx=0, ty=0, crop_rect=(128, 0, 256, 256), full_eff_tile_px=256
        )
        assert result == (128, 0, 256, 256)

    def test_partial_overlap_bottom(self):
        """Tile partially overlapping crop rect on bottom side."""
        result = tile_overlap_rect_common(
            tx=0, ty=0, crop_rect=(0, 128, 256, 256), full_eff_tile_px=256
        )
        assert result == (0, 128, 256, 256)

    def test_no_overlap_tile_left_of_crop(self):
        """Tile completely to the left of crop rect returns None."""
        result = tile_overlap_rect_common(
            tx=0, ty=0, crop_rect=(512, 0, 256, 256), full_eff_tile_px=256
        )
        assert result is None

    def test_no_overlap_tile_above_crop(self):
        """Tile completely above crop rect returns None."""
        result = tile_overlap_rect_common(
            tx=0, ty=0, crop_rect=(0, 512, 256, 256), full_eff_tile_px=256
        )
        assert result is None

    def test_no_overlap_tile_right_of_crop(self):
        """Tile completely to the right of crop rect returns None."""
        result = tile_overlap_rect_common(
            tx=5, ty=0, crop_rect=(0, 0, 256, 256), full_eff_tile_px=256
        )
        assert result is None

    def test_no_overlap_tile_below_crop(self):
        """Tile completely below crop rect returns None."""
        result = tile_overlap_rect_common(
            tx=0, ty=5, crop_rect=(0, 0, 256, 256), full_eff_tile_px=256
        )
        assert result is None

    def test_edge_touching_no_overlap(self):
        """Tiles touching at edge but not overlapping returns None."""
        # Tile ends at x=256, crop starts at x=256
        result = tile_overlap_rect_common(
            tx=0, ty=0, crop_rect=(256, 0, 256, 256), full_eff_tile_px=256
        )
        assert result is None

    def test_single_pixel_overlap(self):
        """Minimal overlap of 1 pixel should return valid rect."""
        result = tile_overlap_rect_common(
            tx=0, ty=0, crop_rect=(255, 255, 10, 10), full_eff_tile_px=256
        )
        assert result == (255, 255, 256, 256)

    def test_different_tile_size(self):
        """Test with different tile size (512)."""
        result = tile_overlap_rect_common(
            tx=1, ty=1, crop_rect=(0, 0, 2000, 2000), full_eff_tile_px=512
        )
        assert result == (512, 512, 1024, 1024)

    def test_crop_rect_smaller_than_tile(self):
        """Crop rect smaller than tile and fully inside tile."""
        result = tile_overlap_rect_common(
            tx=0, ty=0, crop_rect=(50, 50, 100, 100), full_eff_tile_px=256
        )
        assert result == (50, 50, 150, 150)

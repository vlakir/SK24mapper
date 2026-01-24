"""Tests for tiles.coverage module."""

import pytest

from tiles.coverage import iter_overlapping_tiles


class TestIterOverlappingTiles:
    """Tests for iter_overlapping_tiles function."""

    def test_all_tiles_overlap(self):
        """All tiles should be yielded when crop rect covers all."""
        tiles = [(i, (i * 100, i * 100)) for i in range(4)]
        tiles_x = 2
        crop_rect = (0, 0, 1000, 1000)
        tile_px = 256

        result = list(iter_overlapping_tiles(tiles, tiles_x, crop_rect, tile_px=tile_px))
        assert len(result) == 4

    def test_no_tiles_overlap(self):
        """No tiles should be yielded when crop rect is outside."""
        tiles = [(i, (i * 100, i * 100)) for i in range(4)]
        tiles_x = 2
        crop_rect = (10000, 10000, 100, 100)  # Far away
        tile_px = 256

        result = list(iter_overlapping_tiles(tiles, tiles_x, crop_rect, tile_px=tile_px))
        assert len(result) == 0

    def test_partial_overlap(self):
        """Only overlapping tiles should be yielded."""
        # 2x2 grid of tiles
        tiles = [(i, (i * 100, i * 100)) for i in range(4)]
        tiles_x = 2
        # Crop rect only covers first tile (0,0)
        crop_rect = (0, 0, 100, 100)
        tile_px = 256

        result = list(iter_overlapping_tiles(tiles, tiles_x, crop_rect, tile_px=tile_px))
        assert len(result) == 1
        assert result[0][0] == 0  # First tile

    def test_empty_tiles_list(self):
        """Empty tiles list should yield nothing."""
        tiles = []
        tiles_x = 2
        crop_rect = (0, 0, 1000, 1000)
        tile_px = 256

        result = list(iter_overlapping_tiles(tiles, tiles_x, crop_rect, tile_px=tile_px))
        assert result == []

    def test_preserves_tile_data(self):
        """Should preserve original tile data in output."""
        tiles = [(0, (500, 600)), (1, (700, 800))]
        tiles_x = 2
        crop_rect = (0, 0, 1000, 1000)
        tile_px = 256

        result = list(iter_overlapping_tiles(tiles, tiles_x, crop_rect, tile_px=tile_px))
        assert len(result) == 2
        assert result[0] == (0, (500, 600))
        assert result[1] == (1, (700, 800))

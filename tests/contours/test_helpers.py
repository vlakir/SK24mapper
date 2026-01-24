"""Tests for contours_helpers module."""

import pytest

from contours.helpers import TileOverlapParams, tile_overlap_rect, tx_ty_from_index


class TestTileOverlapParams:
    """Tests for TileOverlapParams dataclass."""

    def test_create_params(self):
        """Should create params with all fields."""
        params = TileOverlapParams(
            full_eff_tile_px=512,
            cx0=100,
            cy0=100,
            cx1=1000,
            cy1=1000,
        )
        assert params.full_eff_tile_px == 512
        assert params.cx0 == 100
        assert params.cy0 == 100
        assert params.cx1 == 1000
        assert params.cy1 == 1000
        assert params.crop_rect is None

    def test_create_params_with_crop_rect(self):
        """Should create params with crop_rect."""
        params = TileOverlapParams(
            full_eff_tile_px=256,
            cx0=0,
            cy0=0,
            cx1=500,
            cy1=500,
            crop_rect=(10, 10, 480, 480),
        )
        assert params.crop_rect == (10, 10, 480, 480)


class TestTileOverlapRect:
    """Tests for tile_overlap_rect function."""

    def test_full_overlap(self):
        """Tile fully inside working rect should return tile bounds."""
        params = TileOverlapParams(
            full_eff_tile_px=256,
            cx0=0,
            cy0=0,
            cx1=1000,
            cy1=1000,
        )
        result = tile_overlap_rect(1, 1, params)
        assert result == (256, 256, 512, 512)

    def test_partial_overlap(self):
        """Tile partially overlapping should return intersection."""
        params = TileOverlapParams(
            full_eff_tile_px=256,
            cx0=300,
            cy0=300,
            cx1=600,
            cy1=600,
        )
        result = tile_overlap_rect(1, 1, params)
        # Tile 1,1 is at (256, 256) to (512, 512)
        # Intersection with (300, 300, 600, 600) is (300, 300, 512, 512)
        assert result == (300, 300, 512, 512)

    def test_no_overlap(self):
        """Tile outside working rect should return None."""
        params = TileOverlapParams(
            full_eff_tile_px=256,
            cx0=0,
            cy0=0,
            cx1=100,
            cy1=100,
        )
        result = tile_overlap_rect(1, 1, params)
        assert result is None

    def test_edge_touch_no_overlap(self):
        """Tiles touching at edge should return None."""
        params = TileOverlapParams(
            full_eff_tile_px=256,
            cx0=0,
            cy0=0,
            cx1=256,
            cy1=256,
        )
        result = tile_overlap_rect(1, 0, params)
        assert result is None

    def test_first_tile(self):
        """First tile (0, 0) should work correctly."""
        params = TileOverlapParams(
            full_eff_tile_px=512,
            cx0=0,
            cy0=0,
            cx1=1024,
            cy1=1024,
        )
        result = tile_overlap_rect(0, 0, params)
        assert result == (0, 0, 512, 512)


class TestTxTyFromIndex:
    """Tests for tx_ty_from_index function."""

    def test_first_index(self):
        """Index 0 should return (0, 0)."""
        tx, ty = tx_ty_from_index(0, 10)
        assert tx == 0
        assert ty == 0

    def test_first_row(self):
        """Indices in first row should have ty=0."""
        for i in range(5):
            tx, ty = tx_ty_from_index(i, 5)
            assert tx == i
            assert ty == 0

    def test_second_row(self):
        """Indices in second row should have ty=1."""
        tx, ty = tx_ty_from_index(5, 5)
        assert tx == 0
        assert ty == 1

    def test_arbitrary_index(self):
        """Arbitrary index should convert correctly."""
        # Index 17 with 5 columns: 17 // 5 = 3 (ty), 17 % 5 = 2 (tx)
        tx, ty = tx_ty_from_index(17, 5)
        assert tx == 2
        assert ty == 3

    def test_single_column(self):
        """Single column grid should work."""
        tx, ty = tx_ty_from_index(5, 1)
        assert tx == 0
        assert ty == 5

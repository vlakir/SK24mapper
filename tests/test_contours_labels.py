"""Tests for contours_labels module."""

import pytest
from PIL import Image

from contours.labels import draw_contour_labels


class TestDrawContourLabels:
    """Tests for draw_contour_labels function."""

    def test_empty_polylines(self):
        """Should handle empty polylines dict."""
        img = Image.new('RGBA', (500, 500), color='white')
        result = draw_contour_labels(
            img,
            seed_polylines={},
            levels=[],
            crop_rect=None,
            seed_ds=1,
            mpp=1.0,
        )
        assert isinstance(result, list)
        assert result == []

    def test_dry_run_mode(self):
        """Dry run should not modify image."""
        img = Image.new('RGBA', (500, 500), color='white')
        result = draw_contour_labels(
            img,
            seed_polylines={0: [[(100.0, 100.0), (200.0, 100.0), (300.0, 100.0), (400.0, 100.0)]]},
            levels=[100.0],
            crop_rect=None,
            seed_ds=1,
            mpp=1.0,
            dry_run=True,
        )
        assert isinstance(result, list)

    def test_with_crop_rect(self):
        """Should handle crop_rect parameter."""
        img = Image.new('RGBA', (500, 500), color='white')
        result = draw_contour_labels(
            img,
            seed_polylines={},
            levels=[50.0],
            crop_rect=(10, 10, 480, 480),
            seed_ds=1,
            mpp=1.0,
        )
        assert isinstance(result, list)

    def test_single_level_single_polyline(self):
        """Should handle single level with single polyline."""
        img = Image.new('RGBA', (600, 400), color='white')
        polylines = {
            0: [[(50.0, 200.0), (150.0, 200.0), (250.0, 200.0), (350.0, 200.0), (450.0, 200.0), (550.0, 200.0)]]
        }
        result = draw_contour_labels(
            img,
            seed_polylines=polylines,
            levels=[100.0],
            crop_rect=None,
            seed_ds=1,
            mpp=1.0,
        )
        assert isinstance(result, list)

    def test_multiple_polylines_per_level(self):
        """Should handle multiple polylines per level."""
        img = Image.new('RGBA', (600, 600), color='white')
        polylines = {
            0: [
                [(50.0, 100.0), (200.0, 100.0), (350.0, 100.0), (500.0, 100.0)],
                [(50.0, 300.0), (200.0, 300.0), (350.0, 300.0), (500.0, 300.0)],
            ]
        }
        result = draw_contour_labels(
            img,
            seed_polylines=polylines,
            levels=[100.0],
            crop_rect=None,
            seed_ds=1,
            mpp=1.0,
        )
        assert isinstance(result, list)

    def test_different_mpp_values(self):
        """Should handle different mpp values."""
        img = Image.new('RGBA', (500, 500), color='white')
        for mpp in [0.5, 1.0, 2.0, 5.0]:
            result = draw_contour_labels(
                img.copy(),
                seed_polylines={},
                levels=[100.0],
                crop_rect=None,
                seed_ds=1,
                mpp=mpp,
            )
            assert isinstance(result, list)

    def test_different_seed_ds_values(self):
        """Should handle different seed_ds values."""
        img = Image.new('RGBA', (500, 500), color='white')
        for seed_ds in [1, 2, 4, 8]:
            result = draw_contour_labels(
                img.copy(),
                seed_polylines={},
                levels=[100.0],
                crop_rect=None,
                seed_ds=seed_ds,
                mpp=1.0,
            )
            assert isinstance(result, list)

    def test_short_polyline(self):
        """Should handle short polylines gracefully."""
        img = Image.new('RGBA', (500, 500), color='white')
        polylines = {
            0: [[(100.0, 100.0), (150.0, 100.0)]]  # Only 2 points
        }
        result = draw_contour_labels(
            img,
            seed_polylines=polylines,
            levels=[100.0],
            crop_rect=None,
            seed_ds=1,
            mpp=1.0,
        )
        assert isinstance(result, list)

    def test_returns_bounding_boxes(self):
        """Result should be list of bounding box tuples."""
        img = Image.new('RGBA', (800, 600), color='white')
        polylines = {
            0: [[(50.0, 300.0), (150.0, 300.0), (250.0, 300.0), (350.0, 300.0), (450.0, 300.0), (550.0, 300.0), (650.0, 300.0)]]
        }
        result = draw_contour_labels(
            img,
            seed_polylines=polylines,
            levels=[100.0],
            crop_rect=None,
            seed_ds=1,
            mpp=1.0,
        )
        assert isinstance(result, list)
        for box in result:
            assert isinstance(box, tuple)
            assert len(box) == 4

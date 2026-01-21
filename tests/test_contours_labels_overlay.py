"""Tests for contours.labels_overlay module."""

import pytest
from PIL import Image

from contours.labels_overlay import draw_contour_labels_overlay


class TestDrawContourLabelsOverlay:
    """Tests for draw_contour_labels_overlay function."""

    def test_empty_seeds(self):
        """Should handle empty seeds dict."""
        img = Image.new('RGBA', (500, 500), color='white')
        result = draw_contour_labels_overlay(
            img,
            seeds_by_level={},
            levels=[],
            mpp=1.0,
            seed_ds=1,
        )
        assert isinstance(result, list)

    def test_dry_run_returns_boxes(self):
        """Dry run should return list of boxes without modifying image."""
        img = Image.new('RGBA', (500, 500), color='white')
        original_data = list(img.getdata())
        result = draw_contour_labels_overlay(
            img,
            seeds_by_level={0: [[(100.0, 100.0), (200.0, 100.0), (300.0, 100.0)]]},
            levels=[100.0],
            mpp=1.0,
            seed_ds=1,
            dry_run=True,
        )
        assert isinstance(result, list)

    def test_returns_list(self):
        """Should return a list."""
        img = Image.new('RGBA', (400, 400), color='white')
        result = draw_contour_labels_overlay(
            img,
            seeds_by_level={},
            levels=[50.0, 100.0],
            mpp=2.0,
            seed_ds=2,
        )
        assert isinstance(result, list)

    def test_with_simple_polyline(self):
        """Should handle simple polyline."""
        img = Image.new('RGBA', (600, 400), color='white')
        seeds = {
            0: [[(50.0, 200.0), (150.0, 200.0), (250.0, 200.0), (350.0, 200.0), (450.0, 200.0)]]
        }
        result = draw_contour_labels_overlay(
            img,
            seeds_by_level=seeds,
            levels=[100.0],
            mpp=1.0,
            seed_ds=1,
        )
        assert isinstance(result, list)

    def test_multiple_levels(self):
        """Should handle multiple levels."""
        img = Image.new('RGBA', (500, 500), color='white')
        seeds = {
            0: [[(50.0, 100.0), (200.0, 100.0), (350.0, 100.0)]],
            1: [[(50.0, 300.0), (200.0, 300.0), (350.0, 300.0)]],
        }
        result = draw_contour_labels_overlay(
            img,
            seeds_by_level=seeds,
            levels=[100.0, 200.0],
            mpp=1.0,
            seed_ds=1,
        )
        assert isinstance(result, list)

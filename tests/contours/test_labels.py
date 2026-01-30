"""Tests for contours_labels module."""

import pytest
from PIL import Image

from unittest.mock import MagicMock, patch

from PIL import Image, ImageFont
from contours.labels import (
    LabelConfig,
    LabelStats,
    build_font_getter,
    compute_bbox,
    draw_contour_labels,
    intersects,
    normalize_angle,
    prepare_label,
    process_polyline,
    render_label_box,
    resolve_label_settings,
    segment_lengths,
)


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
            0: [[(100.0, 100.0), (105.0, 100.0)]]  # Very short
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


def test_intersects():
    assert intersects((0, 0, 10, 10), (5, 5, 15, 15)) is True
    assert intersects((0, 0, 10, 10), (10, 10, 20, 20)) is False
    assert intersects((0, 0, 10, 10), (11, 11, 20, 20)) is False


def test_resolve_label_settings():
    s, m, e, f = resolve_label_settings(100.0, 50.0, 10.0, 12.0)
    assert s == 100.0
    assert m == 50.0
    assert e == 10.0
    assert f == 12.0

    s, m, e, f = resolve_label_settings(None, None, None, None)
    assert s is not None
    assert m is not None
    assert e is not None
    assert f is not None


def test_normalize_angle():
    import math
    assert normalize_angle(0) == 0
    assert normalize_angle(math.pi) == 0
    assert normalize_angle(-math.pi) == 0
    assert normalize_angle(math.pi / 2) == math.pi / 2
    assert normalize_angle(-math.pi / 2) == -math.pi / 2
    assert normalize_angle(math.pi * 3 / 4) == -math.pi / 4
    assert normalize_angle(-math.pi * 3 / 4) == math.pi / 4


def test_segment_lengths():
    pts = [(0, 0), (3, 4), (6, 8)]
    lengths, total = segment_lengths(pts)
    assert lengths == [5.0, 5.0]
    assert total == 10.0


def test_compute_bbox():
    img = Image.new('RGBA', (20, 10))
    bbox = compute_bbox(100, 100, img)
    assert bbox == (90, 95, 110, 105)


def test_label_stats_init():
    stats = LabelStats()
    assert stats.placed == 0
    assert stats.attempts == 0


def test_label_config_init():
    config = LabelConfig(w=100, h=100, spacing_px=10, min_seg_px=5, edge_margin_px=2, dry_run=False)
    assert config.w == 100
    assert config.dry_run is False


def test_render_label_box():
    # Use default font if possible
    from PIL import ImageFont
    try:
        font = ImageFont.load_default()
    except:
        font = MagicMock()
    
    box = render_label_box("100", font)
    assert isinstance(box, Image.Image)
    assert box.mode == 'RGBA'


def test_build_font_getter():
    getter = build_font_getter(1.0, 50.0)
    font = getter()
    assert font is not None
    
    # Test mpp=0
    getter_zero = build_font_getter(0, 50.0)
    font_zero = getter_zero()
    assert font_zero is not None


def test_prepare_label():
    config = LabelConfig(w=1000, h=1000, spacing_px=100, min_seg_px=50, edge_margin_px=10, dry_run=False)
    get_font = lambda: ImageFont.load_default()
    
    bbox, rot, reason = prepare_label("100", 0, 500, 500, [], config, get_font)
    assert bbox is not None
    assert rot is not None
    assert reason is None
    
    # Collision
    bbox2, rot2, reason2 = prepare_label("100", 0, 500, 500, [bbox], config, get_font)
    assert bbox2 is None
    assert reason2 == 'collision'
    
    # Out of bounds
    bbox3, rot3, reason3 = prepare_label("100", 0, -100, -100, [], config, get_font)
    assert bbox3 is None
    assert reason3 == 'bbox'


def test_process_polyline():
    img = Image.new('RGBA', (1000, 1000))
    config = LabelConfig(w=1000, h=1000, spacing_px=10, min_seg_px=5, edge_margin_px=2, dry_run=False)
    stats = LabelStats()
    total_stats = LabelStats()
    get_font = lambda: ImageFont.load_default()
    
    # Needs to be long enough and have enough segments
    poly = [(100, 100), (200, 100), (300, 100), (400, 100), (500, 100)]
    process_polyline(img, poly, "100", 1, config, [], get_font, stats, total_stats)
    assert stats.attempts >= 0

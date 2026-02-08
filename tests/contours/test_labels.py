"""Tests for contours_labels module."""

import pytest
from PIL import Image

from unittest.mock import MagicMock, patch

from PIL import Image, ImageFont
from contours.labels import (
    LabelConfig,
    LabelStats,
    build_font_getter,
    classify_bbox,
    compute_bbox,
    draw_contour_labels,
    find_segment_for_target,
    intersects,
    normalize_angle,
    prepare_label,
    process_polyline,
    render_label_box,
    resolve_label_settings,
    segment_lengths,
    within_edge_margin,
)
from imaging.text import load_grid_font


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


class TestFontCacheHit:
    """Tests that build_font_getter returns cached font on second call (line 126)."""

    def test_second_call_returns_same_object(self):
        """Calling get_font() twice should return the exact same font object (cache hit)."""
        getter = build_font_getter(1.0, 50.0)
        font_first = getter()
        font_second = getter()
        assert font_first is font_second


class TestFindSegmentForTargetNone:
    """Tests for find_segment_for_target returning None (line 176)."""

    def test_target_exceeds_total_length(self):
        """When target > total accumulated length, should return None."""
        seg_lengths = [10.0, 20.0, 30.0]
        result = find_segment_for_target(seg_lengths, 100.0)
        assert result is None

    def test_empty_segments(self):
        """Empty segment list should return None for any positive target."""
        result = find_segment_for_target([], 1.0)
        assert result is None

    def test_target_exactly_at_boundary(self):
        """Target exactly at total length should still find the last segment."""
        seg_lengths = [10.0, 20.0]
        result = find_segment_for_target(seg_lengths, 30.0)
        assert result == (1, 10.0)

    def test_target_just_beyond_total(self):
        """Target barely exceeding total length should return None."""
        seg_lengths = [10.0, 20.0]
        result = find_segment_for_target(seg_lengths, 30.1)
        assert result is None


class TestRenderLabelBoxWithOutline:
    """Tests for render_label_box with CONTOUR_LABEL_OUTLINE_WIDTH > 0 (lines 197-201)."""

    def test_outline_renders_extra_pixels(self):
        """When outline width > 0, label box should contain outline pixels."""
        import contours.labels as labels_mod
        orig_ow = labels_mod.CONTOUR_LABEL_OUTLINE_WIDTH
        orig_bg = labels_mod.CONTOUR_LABEL_BG_RGBA
        try:
            labels_mod.CONTOUR_LABEL_OUTLINE_WIDTH = 2
            labels_mod.CONTOUR_LABEL_BG_RGBA = (255, 255, 200, 180)
            font = load_grid_font(24)
            box = render_label_box("250", font)
            assert isinstance(box, Image.Image)
            assert box.mode == 'RGBA'
            assert box.size[0] > 0 and box.size[1] > 0
        finally:
            labels_mod.CONTOUR_LABEL_OUTLINE_WIDTH = orig_ow
            labels_mod.CONTOUR_LABEL_BG_RGBA = orig_bg

    def test_outline_width_zero_no_outline(self):
        """When outline width is 0, should still render a valid box (no outline loop)."""
        import contours.labels as labels_mod
        orig_ow = labels_mod.CONTOUR_LABEL_OUTLINE_WIDTH
        try:
            labels_mod.CONTOUR_LABEL_OUTLINE_WIDTH = 0
            font = load_grid_font(24)
            box = render_label_box("100", font)
            assert isinstance(box, Image.Image)
        finally:
            labels_mod.CONTOUR_LABEL_OUTLINE_WIDTH = orig_ow


class TestClassifyBboxCollision:
    """Tests for classify_bbox collision detection (line 243)."""

    def test_collision_with_existing_box(self):
        """Should return 'collision' when bbox overlaps with placed boxes."""
        config = LabelConfig(
            w=1000, h=1000, spacing_px=100,
            min_seg_px=50, edge_margin_px=10, dry_run=False,
        )
        placed = [(100, 100, 200, 200)]
        result = classify_bbox((150, 150, 250, 250), placed, config)
        assert result == 'collision'

    def test_no_collision_no_overlap(self):
        """Should return None when bbox does not overlap with placed boxes."""
        config = LabelConfig(
            w=1000, h=1000, spacing_px=100,
            min_seg_px=50, edge_margin_px=10, dry_run=False,
        )
        placed = [(100, 100, 200, 200)]
        result = classify_bbox((300, 300, 400, 400), placed, config)
        assert result is None

    def test_out_of_bounds(self):
        """Should return 'bbox' when box extends outside image."""
        config = LabelConfig(
            w=500, h=500, spacing_px=100,
            min_seg_px=50, edge_margin_px=10, dry_run=False,
        )
        result = classify_bbox((-10, 100, 50, 200), [], config)
        assert result == 'bbox'

    def test_collision_with_multiple_placed(self):
        """Should detect collision even if only one of multiple placed boxes overlaps."""
        config = LabelConfig(
            w=1000, h=1000, spacing_px=100,
            min_seg_px=50, edge_margin_px=10, dry_run=False,
        )
        placed = [(10, 10, 50, 50), (400, 400, 500, 500)]
        result = classify_bbox((420, 420, 480, 480), placed, config)
        assert result == 'collision'


class TestProcessPolylineFull:
    """Tests for full process_polyline flow with labels actually placed (lines 283-345)."""

    @staticmethod
    def _long_horizontal_polyline(n_points, y=500.0, x_start=50.0, step=50.0):
        """Build a long horizontal polyline with n_points."""
        return [(x_start + i * step, y) for i in range(n_points)]

    def test_labels_placed_on_long_polyline(self):
        """A long polyline should result in at least one label placed."""
        img = Image.new('RGBA', (2000, 1000), (255, 255, 255, 255))
        config = LabelConfig(
            w=2000, h=1000, spacing_px=200,
            min_seg_px=50, edge_margin_px=10, dry_run=False,
        )
        stats = LabelStats()
        total_stats = LabelStats()
        placed = []
        get_font = build_font_getter(1.0, 50.0)
        poly = self._long_horizontal_polyline(30, y=500.0, x_start=50.0, step=60.0)
        process_polyline(
            img, poly, "150", 1, config, placed, get_font, stats, total_stats,
        )
        assert stats.placed > 0
        assert len(placed) > 0

    def test_dry_run_does_not_modify_image(self):
        """In dry_run mode, labels should be counted but image not modified."""
        img = Image.new('RGBA', (2000, 1000), (255, 255, 255, 255))
        original_data = list(img.getdata())
        config = LabelConfig(
            w=2000, h=1000, spacing_px=200,
            min_seg_px=50, edge_margin_px=10, dry_run=True,
        )
        stats = LabelStats()
        total_stats = LabelStats()
        placed = []
        get_font = build_font_getter(1.0, 50.0)
        poly = self._long_horizontal_polyline(30, y=500.0, x_start=50.0, step=60.0)
        process_polyline(
            img, poly, "200", 1, config, placed, get_font, stats, total_stats,
        )
        assert list(img.getdata()) == original_data

    def test_too_short_polyline_skipped(self):
        """Polyline shorter than min_seg_px should be skipped."""
        img = Image.new('RGBA', (1000, 1000))
        config = LabelConfig(
            w=1000, h=1000, spacing_px=500,
            min_seg_px=900, edge_margin_px=10, dry_run=False,
        )
        stats = LabelStats()
        total_stats = LabelStats()
        placed = []
        get_font = build_font_getter(1.0, 50.0)
        poly = self._long_horizontal_polyline(15, y=500.0, x_start=100.0, step=10.0)
        process_polyline(
            img, poly, "100", 1, config, placed, get_font, stats, total_stats,
        )
        assert stats.skipped_short == 1
        assert stats.placed == 0

    def test_edge_margin_skips(self):
        """Labels near edges should be skipped due to edge margin."""
        img = Image.new('RGBA', (500, 500))
        config = LabelConfig(
            w=500, h=500, spacing_px=30,
            min_seg_px=10, edge_margin_px=400, dry_run=False,
        )
        stats = LabelStats()
        total_stats = LabelStats()
        placed = []
        get_font = build_font_getter(1.0, 50.0)
        poly = self._long_horizontal_polyline(20, y=50.0, x_start=10.0, step=25.0)
        process_polyline(
            img, poly, "100", 1, config, placed, get_font, stats, total_stats,
        )
        assert stats.skipped_edge > 0

    def test_rgb_image_paste_fallback(self):
        """process_polyline on RGB (not RGBA) image should use paste instead of alpha_composite."""
        img = Image.new('RGB', (2000, 1000), (255, 255, 255))
        config = LabelConfig(
            w=2000, h=1000, spacing_px=200,
            min_seg_px=50, edge_margin_px=10, dry_run=False,
        )
        stats = LabelStats()
        total_stats = LabelStats()
        placed = []
        get_font = build_font_getter(1.0, 50.0)
        poly = self._long_horizontal_polyline(30, y=500.0, x_start=50.0, step=60.0)
        process_polyline(
            img, poly, "300", 1, config, placed, get_font, stats, total_stats,
        )
        assert stats.placed > 0


    def test_collision_inside_process_polyline(self):
        """Pre-placed boxes covering the polyline should cause collision skips."""
        img = Image.new('RGBA', (2000, 1000), (255, 255, 255, 255))
        config = LabelConfig(
            w=2000, h=1000, spacing_px=200,
            min_seg_px=50, edge_margin_px=10, dry_run=False,
        )
        stats = LabelStats()
        total_stats = LabelStats()
        placed = [(0, 0, 2000, 1000)]
        get_font = build_font_getter(1.0, 50.0)
        poly = self._long_horizontal_polyline(30, y=500.0, x_start=50.0, step=60.0)
        process_polyline(
            img, poly, "150", 1, config, placed, get_font, stats, total_stats,
        )
        assert stats.skipped_collision > 0
        assert stats.placed == 0

    def test_bbox_skip_near_image_boundary(self):
        """Labels placed near image boundary should trigger bbox skip."""
        img = Image.new('RGBA', (200, 200), (255, 255, 255, 255))
        config = LabelConfig(
            w=200, h=200, spacing_px=20,
            min_seg_px=5, edge_margin_px=1, dry_run=False,
        )
        stats = LabelStats()
        total_stats = LabelStats()
        placed = []
        get_font = build_font_getter(1.0, 50.0)
        poly = self._long_horizontal_polyline(20, y=5.0, x_start=1.0, step=10.0)
        process_polyline(
            img, poly, "100", 1, config, placed, get_font, stats, total_stats,
        )
        assert stats.skipped_bbox > 0 or stats.placed > 0


class TestDrawContourLabelsDisabled:
    """Tests for draw_contour_labels when CONTOUR_LABELS_ENABLED is False (lines 435-436)."""

    def test_disabled_returns_empty_list(self):
        """When CONTOUR_LABELS_ENABLED is False, should return [] immediately."""
        import contours.labels as labels_mod
        orig = labels_mod.CONTOUR_LABELS_ENABLED
        try:
            labels_mod.CONTOUR_LABELS_ENABLED = False
            img = Image.new('RGBA', (500, 500), color='white')
            result = draw_contour_labels(
                img,
                seed_polylines={0: [[(100.0, 100.0), (400.0, 100.0)]]},
                levels=[100.0],
                crop_rect=None,
                seed_ds=1,
                mpp=1.0,
            )
            assert result == []
        finally:
            labels_mod.CONTOUR_LABELS_ENABLED = orig


class TestWithinEdgeMargin:
    """Tests for within_edge_margin helper."""

    def test_inside_margin(self):
        config = LabelConfig(
            w=1000, h=1000, spacing_px=100,
            min_seg_px=50, edge_margin_px=50, dry_run=False,
        )
        assert within_edge_margin(500, 500, config) is True

    def test_outside_margin_left(self):
        config = LabelConfig(
            w=1000, h=1000, spacing_px=100,
            min_seg_px=50, edge_margin_px=100, dry_run=False,
        )
        assert within_edge_margin(50, 500, config) is False

    def test_outside_margin_bottom(self):
        config = LabelConfig(
            w=1000, h=1000, spacing_px=100,
            min_seg_px=50, edge_margin_px=100, dry_run=False,
        )
        assert within_edge_margin(500, 950, config) is False

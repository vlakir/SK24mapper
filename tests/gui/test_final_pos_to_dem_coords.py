"""Unit tests for MainWindow._final_pos_to_dem_coords().

Uses inspect.getsource pattern (no QApplication needed) for structural tests,
and direct method testing via types.MethodType binding on a mock object.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest


def _make_stub(**cache_overrides: object) -> object:
    """Create a minimal stub with _rh_cache for binding _final_pos_to_dem_coords."""
    from gui.view import MainWindow  # noqa: PLC0415

    stub = types.SimpleNamespace()
    stub._rh_cache = {}
    stub._rh_cache.update(cache_overrides)
    # Bind the real method to our stub
    stub._final_pos_to_dem_coords = types.MethodType(
        MainWindow._final_pos_to_dem_coords, stub
    )
    return stub


# ---------------------------------------------------------------------------
# 1. Returns None when DEM is missing from cache
# ---------------------------------------------------------------------------

class TestNoCacheReturnsNone:
    def test_empty_cache(self) -> None:
        stub = _make_stub()
        assert stub._final_pos_to_dem_coords(100.0, 200.0) is None

    def test_no_dem(self) -> None:
        stub = _make_stub(final_size=(800, 600))
        assert stub._final_pos_to_dem_coords(100.0, 200.0) is None


# ---------------------------------------------------------------------------
# 2. No rotation, no crop offset (simplest path)
# ---------------------------------------------------------------------------

class TestNoRotation:
    def test_identity_when_crop_equals_final(self) -> None:
        """When crop == final == DEM, coords should pass through 1:1."""
        dem = np.zeros((100, 200), dtype=np.float32)
        stub = _make_stub(
            dem=dem,
            final_size=(200, 100),
            rotation_deg=0.0,
        )
        result = stub._final_pos_to_dem_coords(50.0, 25.0)
        assert result == (25, 50)  # (row, col)

    def test_with_downsampled_dem(self) -> None:
        """DEM is smaller than crop → coords should be scaled down."""
        dem = np.zeros((50, 100), dtype=np.float32)
        dem_full = np.zeros((200, 400), dtype=np.float32)
        stub = _make_stub(
            dem=dem,
            dem_full=dem_full,
            final_size=(400, 200),
            rotation_deg=0.0,
        )
        # px=200, py=100 → center of image → center of DEM
        result = stub._final_pos_to_dem_coords(200.0, 100.0)
        assert result == (25, 50)  # (row=100*50/200, col=200*100/400)

    def test_with_crop_size_no_dem_full(self) -> None:
        """NSU cache has crop_size but no dem_full — must use crop_size."""
        dem = np.zeros((50, 100), dtype=np.float32)  # downsampled
        # crop_size = (width=400, height=200) — original pre-downsampled
        stub = _make_stub(
            dem=dem,
            crop_size=(400, 200),
            final_size=(400, 200),
            rotation_deg=0.0,
        )
        # px=200, py=100 → center → DEM center (25, 50)
        result = stub._final_pos_to_dem_coords(200.0, 100.0)
        assert result == (25, 50)

    def test_with_crop_larger_than_final(self) -> None:
        """Crop is larger than final → center crop offset applied."""
        dem = np.zeros((100, 200), dtype=np.float32)
        dem_full = np.zeros((100, 200), dtype=np.float32)
        # crop=200x100, final=180x80 → left_crop=10, top_crop=10
        stub = _make_stub(
            dem=dem,
            dem_full=dem_full,
            final_size=(180, 80),
            rotation_deg=0.0,
        )
        # px=0, py=0 → after undo crop: (10, 10) → scale 1:1 → DEM(10, 10)
        result = stub._final_pos_to_dem_coords(0.0, 0.0)
        assert result == (10, 10)


# ---------------------------------------------------------------------------
# 3. With rotation
# ---------------------------------------------------------------------------

class TestWithRotation:
    def test_rotation_90_center_stays(self) -> None:
        """Center of image should stay at center regardless of rotation."""
        dem = np.zeros((100, 100), dtype=np.float32)
        stub = _make_stub(
            dem=dem,
            final_size=(100, 100),
            rotation_deg=90.0,
        )
        result = stub._final_pos_to_dem_coords(50.0, 50.0)
        assert result == (50, 50)

    def test_rotation_small_angle(self) -> None:
        """With small rotation, coords should be slightly shifted."""
        dem = np.zeros((100, 200), dtype=np.float32)
        stub = _make_stub(
            dem=dem,
            final_size=(200, 100),
            rotation_deg=5.0,
        )
        # Just verify it returns a valid result (not None) with reasonable values
        result = stub._final_pos_to_dem_coords(100.0, 50.0)
        assert result is not None
        row, col = result
        # Center point shouldn't move much with small rotation
        assert abs(row - 50) < 5
        assert abs(col - 100) < 5


# ---------------------------------------------------------------------------
# 4. Edge clamping
# ---------------------------------------------------------------------------

class TestEdgeClamping:
    def test_negative_coords_clamped_to_zero(self) -> None:
        """Negative final coords should clamp to (0, 0) in DEM."""
        dem = np.zeros((100, 200), dtype=np.float32)
        stub = _make_stub(
            dem=dem,
            final_size=(200, 100),
            rotation_deg=0.0,
        )
        result = stub._final_pos_to_dem_coords(-50.0, -30.0)
        assert result is not None
        row, col = result
        assert row >= 0
        assert col >= 0

    def test_large_coords_clamped_to_max(self) -> None:
        """Coords beyond DEM size should clamp to (dem_h-1, dem_w-1)."""
        dem = np.zeros((100, 200), dtype=np.float32)
        stub = _make_stub(
            dem=dem,
            final_size=(200, 100),
            rotation_deg=0.0,
        )
        result = stub._final_pos_to_dem_coords(999.0, 999.0)
        assert result is not None
        row, col = result
        assert row <= 99
        assert col <= 199


# ---------------------------------------------------------------------------
# 5. Fallback branch (no final_size)
# ---------------------------------------------------------------------------

class TestNoFinalSize:
    def test_direct_coords_used(self) -> None:
        """Without final_size, px/py used directly as DEM col/row."""
        dem = np.zeros((100, 200), dtype=np.float32)
        stub = _make_stub(dem=dem)
        result = stub._final_pos_to_dem_coords(42.0, 37.0)
        assert result == (37, 42)  # (row, col) = (int(py), int(px))


# ---------------------------------------------------------------------------
# 6. Structural regression: methods use _final_pos_to_dem_coords
# ---------------------------------------------------------------------------

class TestStructuralRegression:
    """Verify that _recompute_coverage_at_click and _trigger_nsu_recompute
    call _final_pos_to_dem_coords instead of duplicating the transform."""

    def _get_source(self, method_name: str) -> str:
        import inspect

        from gui.view import MainWindow  # noqa: PLC0415

        return inspect.getsource(getattr(MainWindow, method_name))

    def test_recompute_coverage_uses_shared_method(self) -> None:
        src = self._get_source('_recompute_coverage_at_click')
        assert '_final_pos_to_dem_coords' in src, (
            '_recompute_coverage_at_click must call _final_pos_to_dem_coords'
        )

    def test_trigger_nsu_uses_shared_method(self) -> None:
        src = self._get_source('_trigger_nsu_recompute')
        assert '_final_pos_to_dem_coords' in src, (
            '_trigger_nsu_recompute must call _final_pos_to_dem_coords'
        )

    def test_no_inline_inverse_rotation_in_recompute(self) -> None:
        """The inline rotation math should no longer be in _recompute_coverage_at_click."""
        src = self._get_source('_recompute_coverage_at_click')
        # The old inline code had "dx * cos_a - dy * sin_a" — should be gone
        assert 'cos_a - dy * sin_a' not in src, (
            'Inline rotation math should be replaced by _final_pos_to_dem_coords call'
        )

    def test_no_inline_inverse_rotation_in_nsu(self) -> None:
        """The inline rotation math should no longer be in _trigger_nsu_recompute."""
        src = self._get_source('_trigger_nsu_recompute')
        assert 'cos_a - dy * sin_a' not in src, (
            'Inline rotation math should be replaced by _final_pos_to_dem_coords call'
        )

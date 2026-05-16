"""Unit tests for MainWindow._nsu_reset_overlay().

Verifies that clearing all NSU points properly resets the UI:
- coverage_layer removed from cache
- topo_base restored to display
- draggable points removed
- NSU markers cleared
"""

from __future__ import annotations

import inspect
import types
from unittest.mock import MagicMock, call

import numpy as np
from PIL import Image


def _get_source(method_name: str) -> str:
    from gui.view import MainWindow  # noqa: PLC0415

    return inspect.getsource(getattr(MainWindow, method_name))


# ---------------------------------------------------------------------------
# Structural: _nsu_reset_overlay exists
# ---------------------------------------------------------------------------

class TestStructural:
    def test_method_exists(self) -> None:
        from gui.view import MainWindow  # noqa: PLC0415

        assert hasattr(MainWindow, '_nsu_reset_overlay'), (
            'MainWindow must have _nsu_reset_overlay method'
        )

    def test_trigger_nsu_calls_reset_on_empty_points(self) -> None:
        """When points list is empty, _trigger_nsu_recompute must call _nsu_reset_overlay."""
        src = _get_source('_trigger_nsu_recompute')
        assert '_nsu_reset_overlay' in src, (
            '_trigger_nsu_recompute must call _nsu_reset_overlay when points are empty'
        )

    def test_trigger_nsu_no_bare_return_on_empty(self) -> None:
        """The old pattern of bare 'return' after 'if not points:' should be replaced."""
        src = _get_source('_trigger_nsu_recompute')
        # Find the "if not points:" block — it should NOT have a bare return
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'if not points:' in line:
                # Next non-blank line should contain _nsu_reset_overlay, not bare return
                for j in range(i + 1, min(i + 3, len(lines))):
                    stripped = lines[j].strip()
                    if stripped:
                        assert stripped != 'return', (
                            'Bare return after "if not points:" must be replaced '
                            'with _nsu_reset_overlay() call'
                        )
                        break
                break


# ---------------------------------------------------------------------------
# Structural: _apply_interactive_alpha restores NSU points after set_image
# ---------------------------------------------------------------------------

class TestAlphaRestoresNsu:
    def _get_source(self, method_name: str) -> str:
        from gui.view import MainWindow  # noqa: PLC0415

        return inspect.getsource(getattr(MainWindow, method_name))

    def test_alpha_draws_nsu_markers(self) -> None:
        src = self._get_source('_apply_interactive_alpha')
        assert '_nsu_draw_markers_on_image' in src, (
            '_apply_interactive_alpha must draw NSU markers on blended image'
        )

    def test_alpha_restores_draggable_points(self) -> None:
        src = self._get_source('_apply_interactive_alpha')
        assert '_nsu_register_draggable_points' in src, (
            '_apply_interactive_alpha must restore NSU draggable points after set_image'
        )


# ---------------------------------------------------------------------------
# Behavioral: _nsu_reset_overlay cleans up properly
# ---------------------------------------------------------------------------

class TestBehavior:
    def _make_stub(self) -> object:
        """Create a stub with mocked dependencies for _nsu_reset_overlay."""
        from gui.view import MainWindow  # noqa: PLC0415

        stub = types.SimpleNamespace()

        # _rh_cache with coverage data
        topo_base = Image.new('RGBA', (200, 100), (128, 128, 128, 255))
        coverage = Image.new('RGBA', (200, 100), (255, 0, 0, 128))
        overlay = Image.new('RGBA', (200, 100), (0, 0, 0, 0))

        stub._rh_cache = {
            'topo_base': topo_base,
            'coverage_layer': coverage,
            'overlay_layer': overlay,
            'final_size': (200, 100),
        }

        # Mock preview area
        stub._preview_area = MagicMock()

        # Bind the real method
        stub._nsu_reset_overlay = types.MethodType(
            MainWindow._nsu_reset_overlay, stub
        )

        return stub

    def test_coverage_layer_removed(self) -> None:
        stub = self._make_stub()
        stub._nsu_reset_overlay()
        assert 'coverage_layer' not in stub._rh_cache

    def test_draggable_points_removed(self) -> None:
        stub = self._make_stub()
        stub._nsu_reset_overlay()
        # Should call remove_draggable_point for T0..T9
        calls = stub._preview_area.remove_draggable_point.call_args_list
        expected_ids = {f'T{i}' for i in range(10)}
        actual_ids = {c.args[0] for c in calls}
        assert expected_ids == actual_ids

    def test_nsu_markers_cleared(self) -> None:
        stub = self._make_stub()
        stub._nsu_reset_overlay()
        stub._preview_area.clear_nsu_markers.assert_called_once()

    def test_image_updated_with_clean_topo(self) -> None:
        stub = self._make_stub()
        stub._nsu_reset_overlay()
        # set_image should be called (to display clean map)
        stub._preview_area.set_image.assert_called_once()

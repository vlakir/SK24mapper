"""Structural regression tests for interactive recompute flow in view.py.

Uses inspect.getsource to verify call ordering contracts without creating
a QApplication (no Qt event loop needed).

Prevents:
  BUG2 — label drawn before model is synced (stale antenna_height_m in label).
"""

from __future__ import annotations

import inspect
import re


def _get_source(method_name: str) -> str:
    """Import MainWindow and return source of the named method."""
    # Import lazily to avoid instantiating QApplication
    from gui.view import MainWindow  # noqa: PLC0415

    method = getattr(MainWindow, method_name)
    return inspect.getsource(method)


def _first_occurrence(source: str, pattern: str) -> int:
    """Return the character offset of the first match, or -1 if not found."""
    m = re.search(pattern, source)
    return m.start() if m else -1


# ---------------------------------------------------------------------------
# BUG2: _sync_ui_to_model_now must be called BEFORE _draw_rh_control_point_label
# ---------------------------------------------------------------------------

def test_BUG2_sync_before_label_draw():
    """In _on_radio_horizon_recompute_finished, _sync_ui_to_model_now must
    appear before _draw_rh_control_point_label so that label reads current
    antenna_height_m from model.settings (not stale value)."""
    src = _get_source('_on_radio_horizon_recompute_finished')

    pos_sync = _first_occurrence(src, r'_sync_ui_to_model_now')
    pos_label = _first_occurrence(src, r'_draw_rh_control_point_label')

    assert pos_sync != -1, '_sync_ui_to_model_now not found in method'
    assert pos_label != -1, '_draw_rh_control_point_label not found in method'
    assert pos_sync < pos_label, (
        'BUG2 regression: _sync_ui_to_model_now must be called '
        'BEFORE _draw_rh_control_point_label'
    )


def test_BUG2_sync_before_label_draw_in_interactive_alpha():
    """Document that _apply_interactive_alpha does NOT call
    _sync_ui_to_model_now (alpha slider syncs separately on release).

    If _draw_rh_control_point_label is present, either _sync_ui_to_model_now
    must precede it, or it must be absent (current design: alpha slider syncs
    on release via _on_alpha_slider_released → _sync_ui_to_model_now).
    """
    src = _get_source('_apply_interactive_alpha')

    has_sync = _first_occurrence(src, r'_sync_ui_to_model_now') != -1
    has_label = _first_occurrence(src, r'_draw_rh_control_point_label') != -1

    if has_label and has_sync:
        pos_sync = _first_occurrence(src, r'_sync_ui_to_model_now')
        pos_label = _first_occurrence(src, r'_draw_rh_control_point_label')
        assert pos_sync < pos_label, (
            'If both present, sync must come before label draw'
        )
    # If _draw_rh_control_point_label is present without sync, that's the
    # current accepted design: alpha slider reads already-synced model.settings.
    # This test documents the pattern.


# ---------------------------------------------------------------------------
# CP marker restoration: _update_cp_marker_from_settings after set_image
# ---------------------------------------------------------------------------

def test_recompute_finished_restores_cp_marker():
    """After set_image (which clears QGraphicsScene), the method must call
    _update_cp_marker_from_settings to restore the control point marker."""
    src = _get_source('_on_radio_horizon_recompute_finished')

    pos_set_image = _first_occurrence(src, r'set_image\(')
    pos_restore = _first_occurrence(src, r'_update_cp_marker_from_settings')

    assert pos_set_image != -1, 'set_image not found in method'
    assert pos_restore != -1, '_update_cp_marker_from_settings not found in method'
    assert pos_set_image < pos_restore, (
        '_update_cp_marker_from_settings must be called AFTER set_image '
        '(set_image clears scene, marker must be restored after)'
    )

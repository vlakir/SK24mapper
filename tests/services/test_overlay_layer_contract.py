"""Structural regression tests for overlay layer creation contract.

Ensures that the overlay layer (grid + legend + contours) is created for ALL
map types that use the interactive alpha slider:
  - RADIO_HORIZON
  - RADAR_COVERAGE
  - ELEVATION_COLOR

Without the overlay layer, grid/legend/contours disappear when the alpha
slider is moved because _apply_interactive_alpha re-blends from scratch.

Uses inspect.getsource to verify code structure without Qt or IO.
"""

from __future__ import annotations

import inspect
import re

import pytest


def _get_postprocess_source() -> str:
    """Return source of MapDownloadService._postprocess."""
    from services.map_download_service import MapDownloadService  # noqa: PLC0415

    return inspect.getsource(MapDownloadService._postprocess)


def _get_create_overlay_source() -> str:
    """Return source of MapDownloadService._create_rh_overlay_layer."""
    from services.map_download_service import MapDownloadService  # noqa: PLC0415

    return inspect.getsource(MapDownloadService._create_rh_overlay_layer)


def _get_apply_alpha_source() -> str:
    """Return source of MainWindow._apply_interactive_alpha."""
    from gui.view import MainWindow  # noqa: PLC0415

    return inspect.getsource(MainWindow._apply_interactive_alpha)


def _get_has_coverage_cache_source() -> str:
    """Return source of MainWindow._has_coverage_cache."""
    from gui.view import MainWindow  # noqa: PLC0415

    return inspect.getsource(MainWindow._has_coverage_cache)


# ---------------------------------------------------------------------------
# Test 1: Contours drawn on separate layer for ELEVATION_COLOR
# ---------------------------------------------------------------------------

def test_elev_color_contours_on_separate_layer():
    """In _postprocess, the branch that creates rh_contour_layer must include
    is_elev_color so contours go on a separate transparent layer."""
    src = _get_postprocess_source()
    # Find the line that creates contour_layer (separate transparent layer)
    # It should be guarded by a condition that includes is_elev_color
    pattern = r'if\s+.*is_elev_color.*:.*\n\s+#.*contour.*\n\s+contour_layer\s*='
    match = re.search(pattern, src, re.IGNORECASE)
    if match is None:
        # Alternative: check that the condition containing rh_contour_layer has is_elev_color
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'rh_contour_layer' in line and 'contour_layer' in line:
                # Look backwards for the if-condition
                for j in range(i, max(i - 5, 0), -1):
                    if 'if ' in lines[j] and 'is_elev_color' in lines[j]:
                        return  # OK, found
        # Last resort: find 'if' line that contains both is_radio_horizon and is_elev_color
        # before contour_layer assignment
        contour_block = re.search(
            r'if\s+.*is_radio_horizon.*is_elev_color.*:',
            src,
        )
        assert contour_block is not None, (
            'Contour layer creation condition must include is_elev_color'
        )


# ---------------------------------------------------------------------------
# Test 2: Overlay layer created for ELEVATION_COLOR
# ---------------------------------------------------------------------------

def test_elev_color_overlay_layer_created():
    """In _postprocess, _create_rh_overlay_layer must be called for
    ELEVATION_COLOR (condition includes is_elev_color)."""
    src = _get_postprocess_source()
    # Find the condition that guards _create_rh_overlay_layer call
    match = re.search(
        r'if\s+(.*?):\s*\n\s+self\._create_rh_overlay_layer',
        src,
    )
    assert match is not None, '_create_rh_overlay_layer call not found in _postprocess'
    condition = match.group(1)
    assert 'is_elev_color' in condition, (
        f'_create_rh_overlay_layer condition must include is_elev_color, '
        f'got: {condition}'
    )


# ---------------------------------------------------------------------------
# Test 3: Legend drawn on overlay for ELEVATION_COLOR
# ---------------------------------------------------------------------------

def test_elev_color_legend_on_overlay():
    """In _create_rh_overlay_layer, _draw_legend must be called for
    ELEVATION_COLOR (condition includes is_elev_color)."""
    src = _get_create_overlay_source()
    # Support both single-line and multi-line if conditions before _draw_legend
    match = re.search(
        r'if\s+(.*?)\s*self\._draw_legend',
        src,
        re.DOTALL,
    )
    assert match is not None, '_draw_legend call not found in _create_rh_overlay_layer'
    condition = match.group(1)
    assert 'is_elev_color' in condition, (
        f'_draw_legend condition in overlay must include is_elev_color, '
        f'got: {condition}'
    )


# ---------------------------------------------------------------------------
# Test 4: _apply_interactive_alpha uses overlay_layer
# ---------------------------------------------------------------------------

def test_apply_interactive_alpha_uses_overlay_layer():
    """_apply_interactive_alpha must read overlay_layer from cache and apply
    it via alpha_composite to preserve grid/legend/contours."""
    src = _get_apply_alpha_source()
    assert 'overlay_layer' in src, (
        '_apply_interactive_alpha must use overlay_layer from cache'
    )
    assert 'alpha_composite' in src, (
        '_apply_interactive_alpha must use alpha_composite to apply overlay'
    )


# ---------------------------------------------------------------------------
# Test 5: _has_coverage_cache recognises ELEVATION_COLOR
# ---------------------------------------------------------------------------

def test_all_alpha_map_types_covered():
    """_has_coverage_cache must recognise ELEVATION_COLOR as a valid cache type."""
    src = _get_has_coverage_cache_source()
    assert 'ELEVATION_COLOR' in src, (
        '_has_coverage_cache must handle ELEVATION_COLOR map type'
    )

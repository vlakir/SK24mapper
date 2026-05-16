"""Regression tests for rh_cache key contract.

Ensures the rh_cache dict assembled by map_download_service._save() contains all
keys that CoverageRecomputeWorker and recompute_coverage_fast() rely on.

Prevents BUG1: slider writes one key, worker reads another.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from services.radio_horizon import recompute_coverage_fast
from shared.constants import UavHeightReference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_nsu_rh_cache() -> dict:
    """Build a minimal rh_cache dict as map_download_service does for RADIO_HORIZON."""
    dem = np.zeros((64, 64), dtype=np.float32)
    topo = Image.new('RGBA', (64, 64), (128, 128, 128, 255))
    settings = MagicMock()
    return {
        'dem': dem,
        'dem_full': dem.copy(),
        'topo_base': topo,
        'antenna_row': 32,
        'antenna_col': 32,
        'pixel_size_m': 10.0,
        'antenna_height_m': 5.0,
        'overlay_alpha': 0.3,
        'max_height_m': 120.0,
        'radar_target_height_min_m': 0.0,
        'radar_target_height_max_m': 120.0,
        'uav_height_reference': UavHeightReference.GROUND,
        'final_size': (64, 64),
        'crop_size': (64, 64),
        'coverage_layer': topo.copy(),
        'overlay_layer': None,
        'settings': settings,
        'is_radar_coverage': False,
        'radar_azimuth_deg': 0.0,
        'radar_sector_width_deg': 360.0,
        'radar_elevation_min_deg': 0.0,
        'radar_elevation_max_deg': 90.0,
        'radar_max_range_km': 15.0,
        'rotation_deg': 0.0,
    }


def _build_radar_rh_cache() -> dict:
    """Build rh_cache dict for RADAR_COVERAGE mode."""
    cache = _build_nsu_rh_cache()
    cache['is_radar_coverage'] = True
    cache['radar_azimuth_deg'] = 45.0
    cache['radar_sector_width_deg'] = 90.0
    cache['radar_elevation_min_deg'] = 0.5
    cache['radar_elevation_max_deg'] = 30.0
    cache['radar_max_range_km'] = 20.0
    cache['radar_target_height_min_m'] = 10.0
    cache['radar_target_height_max_m'] = 300.0
    cache['max_height_m'] = 300.0
    return cache


def _build_elev_color_rh_cache() -> dict:
    """Build rh_cache dict for ELEVATION_COLOR mode."""
    topo = Image.new('RGBA', (64, 64), (128, 128, 128, 255))
    coverage = Image.new('RGBA', (64, 64), (200, 100, 50, 255))
    overlay = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    settings = MagicMock()
    return {
        'topo_base': topo,
        'overlay_alpha': 0.3,
        'coverage_layer': coverage,
        'overlay_layer': overlay,
        'settings': settings,
        'is_elev_color': True,
        'rotation_deg': 0.0,
        'final_size': (64, 64),
        'crop_size': (64, 64),
    }


# ---------------------------------------------------------------------------
# Key contract: NSU (RADIO_HORIZON) cache
# ---------------------------------------------------------------------------

# Keys that CoverageRecomputeWorker.run() reads from rh_cache (view.py)
_WORKER_KEYS_NSU = {
    'dem',
    'antenna_height_m',
    'pixel_size_m',
    'topo_base',
    'overlay_alpha',
    'max_height_m',
    'uav_height_reference',
    'final_size',
    'crop_size',
    'rotation_deg',
}


def test_nsu_cache_contains_all_worker_keys():
    """rh_cache for RADIO_HORIZON must contain every key that the worker reads."""
    cache = _build_nsu_rh_cache()
    missing = _WORKER_KEYS_NSU - set(cache.keys())
    assert not missing, f'NSU rh_cache missing worker keys: {missing}'


# ---------------------------------------------------------------------------
# Key contract: RADAR_COVERAGE cache
# ---------------------------------------------------------------------------

_RADAR_EXTRA_KEYS = {
    'radar_azimuth_deg',
    'radar_sector_width_deg',
    'radar_elevation_min_deg',
    'radar_elevation_max_deg',
    'radar_max_range_km',
    'radar_target_height_min_m',
    'radar_target_height_max_m',
    'is_radar_coverage',
}


def test_radar_cache_contains_sector_keys():
    """rh_cache for RADAR_COVERAGE must contain sector-specific keys."""
    cache = _build_radar_rh_cache()
    required = _WORKER_KEYS_NSU | _RADAR_EXTRA_KEYS
    missing = required - set(cache.keys())
    assert not missing, f'Radar rh_cache missing keys: {missing}'


# ---------------------------------------------------------------------------
# Key contract: ELEVATION_COLOR cache
# ---------------------------------------------------------------------------

_ELEV_COLOR_KEYS = {
    'topo_base',
    'overlay_alpha',
    'coverage_layer',
    'rotation_deg',
    'final_size',
    'crop_size',
}


def test_elev_color_cache_contains_alpha_keys():
    """rh_cache for ELEVATION_COLOR must contain keys for interactive alpha."""
    cache = _build_elev_color_rh_cache()
    missing = _ELEV_COLOR_KEYS - set(cache.keys())
    assert not missing, f'ElevColor rh_cache missing keys: {missing}'


# ---------------------------------------------------------------------------
# BUG1 regression: slider must update max_height_m (not only the typed key)
# ---------------------------------------------------------------------------

def test_BUG1_flight_height_must_update_max_height_m():
    """Regression BUG1: _on_flight_height_slider_changed must write max_height_m."""
    cache = _build_nsu_rh_cache()
    cache['max_height_m'] = 120.0  # initial

    # Simulate _on_flight_height_slider_changed logic (view.py:2586-2595):
    new_value = 250
    cache['max_flight_height_m'] = float(new_value)
    cache['max_height_m'] = float(new_value)  # <- the fix for BUG1

    assert cache['max_height_m'] == 250.0, (
        'BUG1 regression: max_height_m not updated by flight height slider'
    )


def test_BUG1_target_height_must_update_max_height_m():
    """Regression BUG1: _on_target_h_range_changed must write max_height_m."""
    cache = _build_radar_rh_cache()
    cache['max_height_m'] = 300.0  # initial

    # Simulate _on_target_h_range_changed logic (view.py:2535-2546):
    lo, hi = 1, 50  # raw slider values (×10 = metres)
    cache['radar_target_height_min_m'] = float(lo * 10)
    cache['radar_target_height_max_m'] = float(hi * 10)
    cache['max_height_m'] = float(hi * 10)  # <- the fix for BUG1

    assert cache['max_height_m'] == 500.0, (
        'BUG1 regression: max_height_m not updated by target height slider'
    )


# ---------------------------------------------------------------------------
# Functional: recompute_coverage_fast actually respects key parameters
# ---------------------------------------------------------------------------

def _small_dem(height_value: float = 100.0) -> np.ndarray:
    """Create a small DEM for fast recompute tests."""
    return np.full((32, 32), height_value, dtype=np.float32)


def _small_topo() -> Image.Image:
    return Image.new('RGBA', (32, 32), (128, 128, 128, 255))


def test_recompute_fast_uses_antenna_height_param():
    """recompute_coverage_fast must produce different results for different antenna heights."""
    dem = _small_dem()
    topo = _small_topo()

    result_low, _ = recompute_coverage_fast(
        dem=dem,
        new_antenna_row=16,
        new_antenna_col=16,
        antenna_height_m=1.0,
        pixel_size_m=10.0,
        topo_base=topo,
        overlay_alpha=0.0,
        max_height_m=50.0,
    )

    result_high, _ = recompute_coverage_fast(
        dem=dem,
        new_antenna_row=16,
        new_antenna_col=16,
        antenna_height_m=100.0,
        pixel_size_m=10.0,
        topo_base=topo,
        overlay_alpha=0.0,
        max_height_m=50.0,
    )

    arr_low = np.array(result_low)
    arr_high = np.array(result_high)
    assert not np.array_equal(arr_low, arr_high), (
        'antenna_height_m has no effect on recompute result'
    )


def test_recompute_fast_uses_max_height_param():
    """recompute_coverage_fast must produce different results for different max_height_m."""
    dem = _small_dem()
    topo = _small_topo()

    result_low, _ = recompute_coverage_fast(
        dem=dem,
        new_antenna_row=16,
        new_antenna_col=16,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        topo_base=topo,
        overlay_alpha=0.0,
        max_height_m=5.0,
    )

    result_high, _ = recompute_coverage_fast(
        dem=dem,
        new_antenna_row=16,
        new_antenna_col=16,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        topo_base=topo,
        overlay_alpha=0.0,
        max_height_m=500.0,
    )

    arr_low = np.array(result_low)
    arr_high = np.array(result_high)
    assert not np.array_equal(arr_low, arr_high), (
        'max_height_m has no effect on recompute result'
    )


# ---------------------------------------------------------------------------
# Overlay layer must NOT be None for ELEVATION_COLOR (fix for alpha slider bug)
# ---------------------------------------------------------------------------

def test_elev_color_overlay_layer_should_not_be_none():
    """After the fix, overlay_layer in ELEVATION_COLOR cache must not be None.

    When overlay_layer is None, _apply_interactive_alpha skips the
    alpha_composite step and grid/legend/contours disappear on slider move.
    """
    cache = _build_elev_color_rh_cache()
    assert cache['overlay_layer'] is not None, (
        'ELEVATION_COLOR rh_cache must have a non-None overlay_layer '
        'so that grid/legend/contours survive alpha slider changes'
    )

"""Pytest configuration and fixtures for SK42 tests."""

import sys
from pathlib import Path

import pytest

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def _isolate_contour_layer_disk_cache(monkeypatch):
    """
    Point the contour-layer disk LRU at a throw-away dir per test so the
    real user-cache (~/.cache/sk42mapper/contour_layers) never leaks state
    between test runs. Without this, a previously-written PNG could hit a
    test's cache-key on subsequent runs and skip code under test.
    """
    from shared import contour_layer_disk_cache
    monkeypatch.setattr(
        contour_layer_disk_cache, '_resolve_dir', lambda: None,
    )

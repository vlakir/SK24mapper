"""Tests for dem.builder module."""

import numpy as np

import dem.builder as builder
from dem.builder import compute_elevation_levels, downsample_dem_for_seed


def test_build_dem_from_tiles_delegates(monkeypatch):
    """build_dem_from_tiles should delegate to assemble_dem."""
    dem = np.zeros((1, 1), dtype=float)
    called = {}

    def fake_assemble(tiles, tiles_x, tiles_y, eff_tile_px, crop_rect):
        called["args"] = (tiles, tiles_x, tiles_y, eff_tile_px, crop_rect)
        return dem

    monkeypatch.setattr(builder, "assemble_dem", fake_assemble)

    result = builder.build_dem_from_tiles([dem], 1, 2, 256, (0, 0, 1, 1))

    assert result is dem
    assert called["args"] == ([dem], 1, 2, 256, (0, 0, 1, 1))


def test_downsample_dem_for_seed_samples_points():
    """Downsample should pick samples with expected stride."""
    dem = np.arange(16, dtype=float).reshape((4, 4))

    seed_dem, seed_h, seed_w = downsample_dem_for_seed(dem, downsample_factor=3)

    assert (seed_h, seed_w) == (2, 2)
    assert seed_dem == [[0.0, 3.0], [12.0, 15.0]]


def test_compute_elevation_levels_handles_nan():
    """All-NaN DEM should return empty levels and zeros."""
    dem = np.array([[float("nan")]], dtype=float)

    levels, mn, mx = compute_elevation_levels(dem, interval=5.0)

    assert levels == []
    assert (mn, mx) == (0.0, 0.0)
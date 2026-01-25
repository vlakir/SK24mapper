"""Tests for elevation.dem_builder module."""

import numpy as np

import elevation.dem_builder as dem_builder
from elevation.dem_builder import (
    DEMCache,
    compute_elevation_levels,
    downsample_dem_for_seed,
)


class TestDownsampleDemForSeed:
    """Tests for downsample_dem_for_seed."""

    def test_downsample_basic_grid(self):
        """Downsample should pick source points with correct step."""
        dem = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=float,
        )

        seed_dem, seed_h, seed_w = downsample_dem_for_seed(dem, downsample_factor=2)

        assert (seed_h, seed_w) == (2, 2)
        assert seed_dem == [[1.0, 3.0], [7.0, 9.0]]


class TestComputeElevationLevels:
    """Tests for compute_elevation_levels."""

    def test_levels_from_range(self):
        """Should build levels spanning min/max with interval rounding."""
        dem = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

        levels, mn, mx = compute_elevation_levels(dem, interval=2.0)

        assert (mn, mx) == (1.0, 4.0)
        assert levels == [0.0, 2.0, 4.0]

    def test_all_nan_returns_empty(self):
        """All-NaN DEM should return empty levels and zeros."""
        dem = np.array([[float('nan')]], dtype=float)

        levels, mn, mx = compute_elevation_levels(dem, interval=10.0)

        assert levels == []
        assert (mn, mx) == (0.0, 0.0)


class TestDEMCache:
    """Tests for DEMCache wrapper."""

    def test_get_or_fetch_uses_cache(self, monkeypatch):
        """Cached tiles should bypass fetch and cache writes."""
        cached = np.array([[1.0]], dtype=float)
        cache_calls: list[tuple[int, int, int]] = []

        monkeypatch.setattr(dem_builder, "get_cached_dem_tile", lambda z, x, y: cached)
        monkeypatch.setattr(
            dem_builder,
            "cache_dem_tile",
            lambda z, x, y, dem: cache_calls.append((z, x, y)),
        )

        result = DEMCache.get_or_fetch(1, 2, 3, fetch_func=lambda: np.zeros((1, 1)))

        assert result is cached
        assert cache_calls == []

    def test_get_or_fetch_fetches_and_caches(self, monkeypatch):
        """Missing tiles should be fetched and stored in cache."""
        cache_calls: list[tuple[int, int, int]] = []
        fetched = np.array([[5.0]], dtype=float)

        monkeypatch.setattr(dem_builder, "get_cached_dem_tile", lambda z, x, y: None)
        monkeypatch.setattr(
            dem_builder,
            "cache_dem_tile",
            lambda z, x, y, dem: cache_calls.append((z, x, y)),
        )

        result = DEMCache.get_or_fetch(4, 5, 6, fetch_func=lambda: fetched)

        assert result is fetched
        assert cache_calls == [(4, 5, 6)]
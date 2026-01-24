"""Tests for contours_builder module."""

import pytest

from contours.builder import (
    BuildOpts,
    ContourLevels,
    CoordMap,
    Sampling,
    SeedGrid,
    build_seed_polylines,
)


class TestSeedGrid:
    """Tests for SeedGrid dataclass."""

    def test_create(self):
        """Should create SeedGrid with all fields."""
        grid = SeedGrid(w=100, h=200, step=10)
        assert grid.w == 100
        assert grid.h == 200
        assert grid.step == 10


class TestContourLevels:
    """Tests for ContourLevels dataclass."""

    def test_create(self):
        """Should create ContourLevels with all fields."""
        levels = ContourLevels(base=0.0, interval=10.0, index_every=5)
        assert levels.base == 0.0
        assert levels.interval == 10.0
        assert levels.index_every == 5


class TestSampling:
    """Tests for Sampling dataclass."""

    def test_create(self):
        """Should create Sampling with callable."""
        def get_val(x, y):
            return float(x + y)
        sampling = Sampling(get_value=get_val)
        assert sampling.get_value(1, 2) == 3.0


class TestCoordMap:
    """Tests for CoordMap dataclass."""

    def test_create(self):
        """Should create CoordMap with callable."""
        def to_px(x, y):
            return (float(x * 10), float(y * 10))
        coord = CoordMap(to_px=to_px)
        assert coord.to_px(5, 3) == (50.0, 30.0)


class TestBuildOpts:
    """Tests for BuildOpts dataclass."""

    def test_create(self):
        """Should create BuildOpts with quant."""
        opts = BuildOpts(quant=0.5)
        assert opts.quant == 0.5


class TestBuildSeedPolylines:
    """Tests for build_seed_polylines function."""

    def test_not_implemented(self):
        """Should raise NotImplementedError."""
        grid = SeedGrid(w=10, h=10, step=1)
        levels = ContourLevels(base=0.0, interval=10.0, index_every=5)
        sampling = Sampling(get_value=lambda x, y: 0.0)
        coord = CoordMap(to_px=lambda x, y: (float(x), float(y)))
        opts = BuildOpts(quant=0.5)
        
        with pytest.raises(NotImplementedError):
            build_seed_polylines(grid, levels, sampling, coord, opts)

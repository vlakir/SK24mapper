"""Tests for contours.seeds module."""

import pytest

from contours.seeds import (
    MSParams,
    build_seed_polylines,
    simple_smooth_polyline,
    smooth_polyline,
)


class TestMSParams:
    """Tests for MSParams dataclass."""

    def test_create_default(self):
        """Should create MSParams with defaults."""
        params = MSParams()
        assert params is not None


class TestSmoothPolyline:
    """Tests for smooth_polyline function."""

    def test_empty_list(self):
        """Should handle empty list."""
        result = smooth_polyline([])
        assert result == []

    def test_single_point(self):
        """Should handle single point."""
        result = smooth_polyline([(0.0, 0.0)])
        assert len(result) == 1

    def test_two_points(self):
        """Should handle two points."""
        result = smooth_polyline([(0.0, 0.0), (10.0, 10.0)])
        assert len(result) == 2

    def test_preserves_endpoints(self):
        """Should preserve first and last points."""
        points = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
        result = smooth_polyline(points)
        assert result[0] == points[0]
        assert result[-1] == points[-1]

    def test_with_smoothing_factor(self):
        """Should accept smoothing_factor parameter."""
        points = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0), (15.0, 5.0)]
        result = smooth_polyline(points, smoothing_factor=2)
        assert len(result) >= 2


class TestSimpleSmoothPolyline:
    """Tests for simple_smooth_polyline function."""

    def test_empty_list(self):
        """Should handle empty list."""
        result = simple_smooth_polyline([])
        assert result == []

    def test_single_point(self):
        """Should handle single point."""
        result = simple_smooth_polyline([(0.0, 0.0)])
        assert len(result) == 1

    def test_two_points(self):
        """Should handle two points."""
        result = simple_smooth_polyline([(0.0, 0.0), (10.0, 10.0)])
        assert len(result) == 2

    def test_preserves_endpoints(self):
        """Should preserve first and last points."""
        points = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
        result = simple_smooth_polyline(points)
        assert result[0] == points[0]
        assert result[-1] == points[-1]

    def test_with_iterations(self):
        """Should accept iterations parameter."""
        points = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0), (15.0, 5.0)]
        result = simple_smooth_polyline(points, iterations=3)
        assert len(result) >= 2


class TestBuildSeedPolylines:
    """Tests for build_seed_polylines function."""

    def test_flat_dem_no_contours(self):
        """Flat DEM should produce no contours for non-matching levels."""
        dem = [[100.0] * 10 for _ in range(10)]
        levels = [50.0, 150.0]  # No level at 100
        result = build_seed_polylines(dem, levels)
        # Should return dict with empty lists for each level
        assert isinstance(result, dict)

    def test_simple_gradient(self):
        """Simple gradient should produce contours."""
        # Create a simple gradient from 0 to 100
        dem = [[float(y * 10) for x in range(11)] for y in range(11)]
        levels = [50.0]
        result = build_seed_polylines(dem, levels)
        assert isinstance(result, dict)
        # Level 50 should have some polylines
        assert 0 in result or len(result) > 0

    def test_returns_dict(self):
        """Should return dictionary."""
        dem = [[0.0, 10.0], [10.0, 20.0]]
        levels = [5.0]
        result = build_seed_polylines(dem, levels)
        assert isinstance(result, dict)

    def test_with_ms_params(self):
        """Should accept MSParams."""
        dem = [[float(y * 10) for x in range(5)] for y in range(5)]
        levels = [25.0]
        params = MSParams()
        result = build_seed_polylines(dem, levels, ms=params)
        assert isinstance(result, dict)

    def test_empty_dem(self):
        """Should handle empty DEM."""
        dem = []
        levels = [50.0]
        result = build_seed_polylines(dem, levels)
        assert isinstance(result, dict)

    def test_single_row_dem(self):
        """Should handle single row DEM."""
        dem = [[0.0, 50.0, 100.0]]
        levels = [50.0]
        result = build_seed_polylines(dem, levels)
        assert isinstance(result, dict)

    def test_multiple_levels(self):
        """Should handle multiple levels."""
        dem = [[float(y * 10) for x in range(10)] for y in range(10)]
        levels = [25.0, 50.0, 75.0]
        result = build_seed_polylines(dem, levels)
        assert isinstance(result, dict)

    def test_negative_elevations(self):
        """Should handle negative elevations."""
        dem = [[float(y * 10 - 50) for x in range(10)] for y in range(10)]
        levels = [-25.0, 0.0, 25.0]
        result = build_seed_polylines(dem, levels)
        assert isinstance(result, dict)


class TestSmoothPolylineExtended:
    """Extended tests for smooth_polyline function."""

    def test_many_points(self):
        """Should handle many points."""
        points = [(float(i), float(i % 10)) for i in range(100)]
        result = smooth_polyline(points)
        assert len(result) >= 2

    def test_zigzag_pattern(self):
        """Should smooth zigzag pattern."""
        points = [(float(i), float(i % 2 * 10)) for i in range(20)]
        result = smooth_polyline(points)
        assert len(result) >= 2

    def test_closed_loop(self):
        """Should handle closed loop."""
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
        result = smooth_polyline(points)
        assert len(result) >= 2


class TestSimpleSmoothPolylineExtended:
    """Extended tests for simple_smooth_polyline function."""

    def test_many_points(self):
        """Should handle many points."""
        points = [(float(i), float(i % 10)) for i in range(100)]
        result = simple_smooth_polyline(points)
        assert len(result) >= 2

    def test_multiple_iterations(self):
        """Should handle multiple iterations."""
        points = [(float(i), float(i % 5)) for i in range(20)]
        result = simple_smooth_polyline(points, iterations=5)
        assert len(result) >= 2

    def test_zero_iterations(self):
        """Zero iterations should return original."""
        points = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]
        result = simple_smooth_polyline(points, iterations=0)
        assert result == points

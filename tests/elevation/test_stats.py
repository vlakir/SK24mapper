"""Tests for elevation.stats module."""

import pytest

from elevation.stats import compute_elevation_range


class TestComputeElevationRange:
    """Tests for compute_elevation_range function."""

    def test_empty_samples(self):
        """Empty samples should return (0.0, 0.0)."""
        result = compute_elevation_range([], p_lo=5.0, p_hi=95.0, min_range_m=10.0)
        assert result == (0.0, 0.0)

    def test_normal_range(self):
        """Normal samples should return percentile-based range."""
        samples = list(range(0, 101))  # 0 to 100
        lo, hi = compute_elevation_range(samples, p_lo=10.0, p_hi=90.0, min_range_m=1.0)
        assert lo >= 0
        assert hi <= 100
        assert lo < hi

    def test_min_range_enforced(self):
        """Minimum range should be enforced for narrow data."""
        samples = [50.0, 50.0, 50.0, 50.0, 50.0]
        lo, hi = compute_elevation_range(samples, p_lo=5.0, p_hi=95.0, min_range_m=20.0)
        assert hi - lo >= 20.0

    def test_single_value(self):
        """Single value should use min_range around that value."""
        samples = [100.0]
        lo, hi = compute_elevation_range(samples, p_lo=5.0, p_hi=95.0, min_range_m=10.0)
        assert hi - lo >= 10.0
        assert (lo + hi) / 2 == 100.0

    def test_negative_elevations(self):
        """Should handle negative elevations (below sea level)."""
        samples = [-50.0, -30.0, -10.0, 0.0, 10.0]
        lo, hi = compute_elevation_range(samples, p_lo=10.0, p_hi=90.0, min_range_m=1.0)
        assert lo < hi

    def test_large_range(self):
        """Should handle large elevation ranges."""
        samples = list(range(0, 5000, 10))  # 0 to 4990
        lo, hi = compute_elevation_range(samples, p_lo=5.0, p_hi=95.0, min_range_m=10.0)
        assert lo >= 0
        assert hi <= 5000
        assert hi - lo > 100

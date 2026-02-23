"""Tests for link profile processor: terrain extraction, LOS/Fresnel analysis."""

import math

import numpy as np
import pytest

from services.processors.link_profile import (
    compute_link_analysis,
    extract_terrain_profile,
)
from shared.constants import EARTH_RADIUS_M, SPEED_OF_LIGHT_MPS


class TestExtractTerrainProfile:
    """Tests for extract_terrain_profile()."""

    def test_flat_dem_constant_elevation(self):
        """Flat DEM should produce constant elevations along the profile."""
        dem = np.full((100, 100), 200.0)
        result = extract_terrain_profile(
            dem,
            point_a_px=(10, 50),
            point_b_px=(90, 50),
            pixel_size_m=10.0,
            num_samples=50,
        )
        assert len(result['distances_m']) == 50
        assert len(result['elevations_m']) == 50
        np.testing.assert_allclose(result['elevations_m'], 200.0, atol=0.1)
        assert result['total_distance_m'] == pytest.approx(800.0, rel=0.01)

    def test_ramp_dem_linear_elevation(self):
        """Linear ramp DEM should produce linearly increasing elevations."""
        dem = np.zeros((100, 200))
        for col in range(200):
            dem[:, col] = col * 1.0  # elevation = column index

        result = extract_terrain_profile(
            dem,
            point_a_px=(0, 50),
            point_b_px=(199, 50),
            pixel_size_m=5.0,
            num_samples=200,
        )
        elevations = result['elevations_m']
        # Should be monotonically increasing
        assert elevations[-1] > elevations[0]
        # First point near 0, last near 199
        assert elevations[0] == pytest.approx(0.0, abs=1.0)
        assert elevations[-1] == pytest.approx(199.0, abs=1.0)

    def test_total_distance_diagonal(self):
        """Diagonal distance should be computed correctly."""
        dem = np.full((100, 100), 100.0)
        result = extract_terrain_profile(
            dem,
            point_a_px=(0, 0),
            point_b_px=(30, 40),
            pixel_size_m=10.0,
            num_samples=100,
        )
        expected = math.sqrt(300**2 + 400**2)
        assert result['total_distance_m'] == pytest.approx(expected, rel=0.01)

    def test_zero_distance(self):
        """Same point A and B should give zero distance."""
        dem = np.full((50, 50), 150.0)
        result = extract_terrain_profile(
            dem,
            point_a_px=(25, 25),
            point_b_px=(25, 25),
            pixel_size_m=10.0,
            num_samples=10,
        )
        assert result['total_distance_m'] == pytest.approx(0.0)


class TestComputeLinkAnalysis:
    """Tests for compute_link_analysis()."""

    def test_fresnel_zero_at_endpoints(self):
        """Fresnel radius should be zero at endpoints."""
        profile = {
            'distances_m': np.linspace(0, 10000, 500),
            'elevations_m': np.full(500, 100.0),
            'total_distance_m': 10000.0,
        }
        result = compute_link_analysis(
            profile,
            antenna_a_m=10.0,
            antenna_b_m=10.0,
            freq_mhz=900.0,
        )
        assert result['fresnel_radius_m'][0] == pytest.approx(0.0, abs=0.01)
        assert result['fresnel_radius_m'][-1] == pytest.approx(0.0, abs=0.01)

    def test_fresnel_max_at_midpoint(self):
        """Fresnel radius should be maximum near the midpoint."""
        profile = {
            'distances_m': np.linspace(0, 10000, 501),
            'elevations_m': np.full(501, 100.0),
            'total_distance_m': 10000.0,
        }
        result = compute_link_analysis(
            profile,
            antenna_a_m=10.0,
            antenna_b_m=10.0,
            freq_mhz=900.0,
        )
        fresnel = result['fresnel_radius_m']
        mid_idx = len(fresnel) // 2
        # Midpoint should have the maximum Fresnel radius
        assert fresnel[mid_idx] == pytest.approx(np.max(fresnel), rel=0.01)

        # Verify Fresnel radius formula at midpoint: r = sqrt(λ * D/4)
        wavelength = SPEED_OF_LIGHT_MPS / (900.0 * 1e6)
        expected_r = math.sqrt(wavelength * 10000 / 4)
        assert fresnel[mid_idx] == pytest.approx(expected_r, rel=0.02)

    def test_earth_curvature_10km(self):
        """Earth curvature correction at 10km midpoint ≈ 1.96m (K=4/3)."""
        d = 10000.0
        n = 501
        profile = {
            'distances_m': np.linspace(0, d, n),
            'elevations_m': np.full(n, 100.0),
            'total_distance_m': d,
        }
        k = 4.0 / 3.0
        result = compute_link_analysis(
            profile,
            antenna_a_m=10.0,
            antenna_b_m=10.0,
            freq_mhz=900.0,
            k=k,
        )
        correction = result['earth_correction_m']
        mid_correction = correction[n // 2]
        # At midpoint: d/2 * d/2 / (2*K*R) = d²/(8*K*R)
        expected = (d**2) / (8 * k * EARTH_RADIUS_M)
        assert mid_correction == pytest.approx(expected, rel=0.01)

    def test_flat_terrain_no_obstruction(self):
        """Flat terrain with antennas should have no obstruction."""
        profile = {
            'distances_m': np.linspace(0, 5000, 200),
            'elevations_m': np.full(200, 100.0),
            'total_distance_m': 5000.0,
        }
        result = compute_link_analysis(
            profile,
            antenna_a_m=10.0,
            antenna_b_m=10.0,
            freq_mhz=900.0,
        )
        assert not result['has_obstruction']
        assert result['worst_clearance_m'] > 0

    def test_hill_obstruction(self):
        """A hill in the middle should cause obstruction."""
        n = 500
        distances = np.linspace(0, 10000, n)
        elevations = np.full(n, 100.0)
        # Add a 200m hill in the middle
        mid = n // 2
        for i in range(mid - 20, mid + 20):
            elevations[i] = 300.0

        profile = {
            'distances_m': distances,
            'elevations_m': elevations,
            'total_distance_m': 10000.0,
        }
        result = compute_link_analysis(
            profile,
            antenna_a_m=10.0,
            antenna_b_m=10.0,
            freq_mhz=900.0,
        )
        # LOS at midpoint: 100+10 = 110m, hill = 300m → obstruction
        assert result['has_obstruction']
        assert result['worst_clearance_m'] < 0

    def test_los_straight_line(self):
        """LOS heights should form a straight line from A antenna tip to B antenna tip."""
        n = 100
        profile = {
            'distances_m': np.linspace(0, 5000, n),
            'elevations_m': np.linspace(100, 200, n),
            'total_distance_m': 5000.0,
        }
        result = compute_link_analysis(
            profile,
            antenna_a_m=15.0,
            antenna_b_m=20.0,
            freq_mhz=900.0,
        )
        los = result['los_heights_m']
        # First point: 100 + 15 = 115
        assert los[0] == pytest.approx(115.0, abs=0.1)
        # Last point: 200 + 20 = 220
        assert los[-1] == pytest.approx(220.0, abs=0.1)
        # Midpoint: (115 + 220) / 2 = 167.5
        assert los[n // 2] == pytest.approx(167.5, rel=0.02)

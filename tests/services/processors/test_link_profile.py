"""Tests for link profile processor: terrain extraction, LOS/Fresnel analysis."""

import math

import numpy as np
import pytest
from PIL import Image

from services.processors.link_profile import (
    _haversine_distance,
    compute_link_analysis,
    extract_terrain_profile,
    render_profile_inset,
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

    def test_zero_distance_no_crash(self):
        """Zero-distance profile should not crash compute_link_analysis."""
        profile = {
            'distances_m': np.zeros(10),
            'elevations_m': np.full(10, 100.0),
            'total_distance_m': 0.0,
        }
        result = compute_link_analysis(
            profile,
            antenna_a_m=10.0,
            antenna_b_m=10.0,
            freq_mhz=900.0,
        )
        # LOS should be constant at 110 (elev 100 + antenna 10)
        np.testing.assert_allclose(result['los_heights_m'], 110.0)
        assert not result['has_obstruction']

    def test_asymmetric_antennas(self):
        """Different antenna heights should produce tilted LOS."""
        n = 200
        profile = {
            'distances_m': np.linspace(0, 5000, n),
            'elevations_m': np.full(n, 100.0),
            'total_distance_m': 5000.0,
        }
        result = compute_link_analysis(
            profile, antenna_a_m=5.0, antenna_b_m=50.0, freq_mhz=900.0,
        )
        los = result['los_heights_m']
        assert los[0] == pytest.approx(105.0, abs=0.1)
        assert los[-1] == pytest.approx(150.0, abs=0.1)
        # LOS should be monotonically increasing
        assert all(los[i] <= los[i + 1] + 0.01 for i in range(n - 1))

    def test_earth_correction_zero_at_endpoints(self):
        """Earth curvature correction should be zero at both endpoints."""
        profile = {
            'distances_m': np.linspace(0, 10000, 500),
            'elevations_m': np.full(500, 100.0),
            'total_distance_m': 10000.0,
        }
        result = compute_link_analysis(
            profile, antenna_a_m=10.0, antenna_b_m=10.0, freq_mhz=900.0,
        )
        assert result['earth_correction_m'][0] == pytest.approx(0.0, abs=1e-6)
        assert result['earth_correction_m'][-1] == pytest.approx(0.0, abs=1e-6)

    def test_fresnel_clearance_pct_clear_link(self):
        """Clear LOS over flat terrain should give positive Fresnel clearance %."""
        profile = {
            'distances_m': np.linspace(0, 5000, 200),
            'elevations_m': np.full(200, 0.0),
            'total_distance_m': 5000.0,
        }
        result = compute_link_analysis(
            profile, antenna_a_m=50.0, antenna_b_m=50.0, freq_mhz=900.0,
        )
        assert result['fresnel_clearance_pct'] > 0
        assert not result['has_obstruction']

    def test_output_keys_complete(self):
        """compute_link_analysis should return all expected keys."""
        profile = {
            'distances_m': np.linspace(0, 1000, 50),
            'elevations_m': np.full(50, 100.0),
            'total_distance_m': 1000.0,
        }
        result = compute_link_analysis(
            profile, antenna_a_m=10.0, antenna_b_m=10.0, freq_mhz=900.0,
        )
        expected_keys = {
            'los_heights_m', 'earth_correction_m', 'fresnel_radius_m',
            'clearance_m', 'has_obstruction', 'worst_clearance_m',
            'fresnel_clearance_pct', 'antenna_a_m', 'antenna_b_m',
        }
        assert set(result.keys()) == expected_keys


class TestHaversineDistance:
    """Tests for _haversine_distance()."""

    def test_same_point_zero(self):
        """Same point should give zero distance."""
        assert _haversine_distance(55.0, 37.0, 55.0, 37.0) == pytest.approx(0.0)

    def test_known_distance_moscow_spb(self):
        """Moscow→SPb: ~634 km great-circle distance."""
        d = _haversine_distance(55.7558, 37.6173, 59.9343, 30.3351)
        assert 630_000 < d < 640_000

    def test_equator_one_degree(self):
        """1 degree of longitude at equator ≈ 111.32 km."""
        d = _haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert d == pytest.approx(111_320, rel=0.01)

    def test_symmetry(self):
        """Distance A→B should equal distance B→A."""
        d1 = _haversine_distance(55.0, 37.0, 60.0, 40.0)
        d2 = _haversine_distance(60.0, 40.0, 55.0, 37.0)
        assert d1 == pytest.approx(d2, rel=1e-10)

    def test_antipodal_points(self):
        """Antipodal points should be ~half circumference of Earth."""
        d = _haversine_distance(0.0, 0.0, 0.0, 180.0)
        half_circumference = math.pi * EARTH_RADIUS_M
        assert d == pytest.approx(half_circumference, rel=0.01)


class TestExtractTerrainProfileEdgeCases:
    """Edge case tests for extract_terrain_profile."""

    def test_points_outside_dem_clamped(self):
        """Points outside DEM bounds should be clamped."""
        dem = np.full((10, 10), 50.0)
        # Point B far outside bounds
        result = extract_terrain_profile(
            dem,
            point_a_px=(5, 5),
            point_b_px=(100, 100),
            pixel_size_m=1.0,
            num_samples=20,
        )
        # Should not crash, elevations clamped to edge values
        assert len(result['elevations_m']) == 20
        np.testing.assert_allclose(result['elevations_m'], 50.0, atol=0.1)

    def test_single_sample(self):
        """Single sample should return a valid result."""
        dem = np.full((50, 50), 300.0)
        result = extract_terrain_profile(
            dem,
            point_a_px=(10, 10),
            point_b_px=(40, 40),
            pixel_size_m=10.0,
            num_samples=1,
        )
        assert len(result['distances_m']) == 1
        assert result['distances_m'][0] == 0.0  # single point at t=0

    def test_vertical_line(self):
        """Vertical line (same column) should have correct distance."""
        dem = np.full((100, 100), 200.0)
        result = extract_terrain_profile(
            dem,
            point_a_px=(50, 10),
            point_b_px=(50, 90),
            pixel_size_m=5.0,
            num_samples=50,
        )
        expected_dist = 80 * 5.0  # 80 rows * 5 m/px
        assert result['total_distance_m'] == pytest.approx(expected_dist, rel=0.01)


class TestRenderProfileInset:
    """Tests for render_profile_inset() — ensures it produces valid images."""

    @staticmethod
    def _make_link_data(total_d=5000.0, n=200, elev=100.0, **overrides):
        distances = np.linspace(0, total_d, n)
        elevations = np.full(n, elev)
        profile = {
            'distances_m': distances,
            'elevations_m': elevations,
            'total_distance_m': total_d,
        }
        analysis = compute_link_analysis(
            profile,
            antenna_a_m=overrides.get('antenna_a_m', 10.0),
            antenna_b_m=overrides.get('antenna_b_m', 10.0),
            freq_mhz=overrides.get('freq_mhz', 900.0),
        )
        data = {**profile, **analysis}
        data['point_a_name'] = overrides.get('point_a_name', 'A')
        data['point_b_name'] = overrides.get('point_b_name', 'B')
        data['freq_mhz'] = overrides.get('freq_mhz', 900.0)
        return data

    def test_returns_rgba_image(self):
        """Should return RGBA PIL image."""
        data = self._make_link_data()
        img = render_profile_inset(data, (800, 600))
        assert isinstance(img, Image.Image)
        assert img.mode == 'RGBA'

    def test_image_dimensions(self):
        """Image width should match map width."""
        data = self._make_link_data()
        img = render_profile_inset(data, (1024, 768))
        assert img.width == 1024

    def test_minimum_height(self):
        """Inset should have at least 120px height."""
        data = self._make_link_data()
        img = render_profile_inset(data, (200, 50))
        assert img.height >= 120

    def test_small_map_no_crash(self):
        """Very small map size should not crash."""
        data = self._make_link_data()
        img = render_profile_inset(data, (100, 100))
        assert isinstance(img, Image.Image)

    def test_long_distance_renders(self):
        """Long distance (50km) link should render."""
        data = self._make_link_data(total_d=50000.0, n=500)
        img = render_profile_inset(data, (1200, 800))
        assert img.width == 1200

    def test_short_distance_renders(self):
        """Short distance (100m) link with meter labels."""
        data = self._make_link_data(total_d=100.0, n=50)
        img = render_profile_inset(data, (800, 600))
        assert isinstance(img, Image.Image)

    def test_cyrillic_names(self):
        """Cyrillic point names should not crash rendering."""
        data = self._make_link_data(
            point_a_name='Узел-1', point_b_name='Узел-2',
        )
        img = render_profile_inset(data, (800, 600))
        assert isinstance(img, Image.Image)

    def test_obstructed_link_renders(self):
        """Obstructed link (red Fresnel) should render without error."""
        n = 200
        distances = np.linspace(0, 10000, n)
        elevations = np.full(n, 100.0)
        elevations[n // 2 - 10: n // 2 + 10] = 300.0  # Hill
        profile = {
            'distances_m': distances,
            'elevations_m': elevations,
            'total_distance_m': 10000.0,
        }
        analysis = compute_link_analysis(
            profile, antenna_a_m=10.0, antenna_b_m=10.0, freq_mhz=900.0,
        )
        data = {**profile, **analysis}
        data['point_a_name'] = 'A'
        data['point_b_name'] = 'B'
        data['freq_mhz'] = 900.0
        assert data['has_obstruction']
        img = render_profile_inset(data, (800, 600))
        assert isinstance(img, Image.Image)

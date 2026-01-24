"""Tests for topography module."""

import math

import pytest
from PIL import Image

from geo.topography import (
    choose_zoom_with_limit,
    compute_grid,
    compute_percentiles,
    compute_xyz_coverage,
    decode_terrain_rgb_to_elevation_m,
    effective_scale_for_xyz,
    estimate_crop_size_px,
    latlng_to_pixel_xy,
    meters_per_pixel,
    pixel_xy_to_latlng,
)


class TestMetersPerPixel:
    """Tests for meters_per_pixel function."""

    def test_equator_zoom_0(self):
        """At equator, zoom 0, should return large value."""
        mpp = meters_per_pixel(0.0, 0)
        assert mpp > 50000  # Very coarse at zoom 0

    def test_equator_zoom_20(self):
        """At equator, zoom 20, should return small value."""
        mpp = meters_per_pixel(0.0, 20)
        assert mpp < 1  # Very fine at zoom 20

    def test_higher_latitude_smaller_mpp(self):
        """Higher latitude should have smaller meters per pixel."""
        mpp_equator = meters_per_pixel(0.0, 10)
        mpp_moscow = meters_per_pixel(55.75, 10)
        assert mpp_moscow < mpp_equator

    def test_higher_zoom_smaller_mpp(self):
        """Higher zoom should have smaller meters per pixel."""
        mpp_z10 = meters_per_pixel(45.0, 10)
        mpp_z15 = meters_per_pixel(45.0, 15)
        assert mpp_z15 < mpp_z10


class TestLatlngToPixelXy:
    """Tests for latlng_to_pixel_xy function."""

    def test_origin(self):
        """(0, 0) should be at center of world."""
        x, y = latlng_to_pixel_xy(0.0, 0.0, 0)
        assert x == 128  # Half of 256 (tile size)
        assert abs(y - 128) < 1  # Should be near center

    def test_positive_longitude(self):
        """Positive longitude should increase x."""
        x1, _ = latlng_to_pixel_xy(0.0, 0.0, 10)
        x2, _ = latlng_to_pixel_xy(0.0, 10.0, 10)
        assert x2 > x1

    def test_positive_latitude(self):
        """Positive latitude should decrease y (north is up)."""
        _, y1 = latlng_to_pixel_xy(0.0, 0.0, 10)
        _, y2 = latlng_to_pixel_xy(45.0, 0.0, 10)
        assert y2 < y1


class TestPixelXyToLatlng:
    """Tests for pixel_xy_to_latlng function."""

    def test_roundtrip(self):
        """Converting to pixel and back should return original coords."""
        lat, lng = 55.75, 37.62  # Moscow
        x, y = latlng_to_pixel_xy(lat, lng, 15)
        lat2, lng2 = pixel_xy_to_latlng(x, y, 15)
        assert abs(lat2 - lat) < 0.0001
        assert abs(lng2 - lng) < 0.0001

    def test_center_of_world(self):
        """Center pixel should return (0, 0) approximately."""
        lat, lng = pixel_xy_to_latlng(128, 128, 0)
        assert abs(lat) < 1
        assert abs(lng) < 1


class TestEstimateCropSizePx:
    """Tests for estimate_crop_size_px function."""

    def test_returns_positive_values(self):
        """Should return positive width, height, and total."""
        w, h, total = estimate_crop_size_px(55.0, 1000.0, 1000.0, 15)
        assert w > 0
        assert h > 0
        assert total == w * h

    def test_larger_area_more_pixels(self):
        """Larger area should require more pixels."""
        w1, h1, _ = estimate_crop_size_px(55.0, 1000.0, 1000.0, 15)
        w2, h2, _ = estimate_crop_size_px(55.0, 2000.0, 2000.0, 15)
        assert w2 > w1
        assert h2 > h1

    def test_higher_zoom_more_pixels(self):
        """Higher zoom should produce more pixels."""
        _, _, total1 = estimate_crop_size_px(55.0, 1000.0, 1000.0, 14)
        _, _, total2 = estimate_crop_size_px(55.0, 1000.0, 1000.0, 16)
        assert total2 > total1


class TestComputePercentiles:
    """Tests for compute_percentiles function."""

    def test_simple_range(self):
        """Simple range should return correct percentiles."""
        values = list(range(0, 101))  # 0 to 100
        lo, hi = compute_percentiles(values, 10.0, 90.0)
        assert abs(lo - 10) < 2
        assert abs(hi - 90) < 2

    def test_single_value(self):
        """Single value should return that value for both."""
        values = [50.0]
        lo, hi = compute_percentiles(values, 5.0, 95.0)
        assert lo == 50.0
        assert hi == 50.0

    def test_two_values(self):
        """Two values should work correctly."""
        values = [10.0, 90.0]
        lo, hi = compute_percentiles(values, 0.0, 100.0)
        assert lo == 10.0
        assert hi == 90.0

    def test_percentile_0_and_100(self):
        """0th and 100th percentiles should be min and max."""
        values = [5.0, 10.0, 15.0, 20.0, 25.0]
        lo, hi = compute_percentiles(values, 0.0, 100.0)
        assert lo == 5.0
        assert hi == 25.0


class TestChooseZoomWithLimit:
    """Tests for choose_zoom_with_limit function."""

    def test_returns_desired_zoom_when_under_limit(self):
        """Should return desired zoom when pixels under limit."""
        zoom = choose_zoom_with_limit(55.0, 1000.0, 1000.0, 15, 1, 100_000_000)
        assert zoom == 15

    def test_reduces_zoom_when_over_limit(self):
        """Should reduce zoom when pixels exceed limit."""
        zoom = choose_zoom_with_limit(55.0, 10000.0, 10000.0, 20, 1, 1_000_000)
        assert zoom < 20

    def test_returns_zero_for_impossible_limit(self):
        """Should return 0 for very small pixel limit."""
        zoom = choose_zoom_with_limit(55.0, 100000.0, 100000.0, 20, 1, 1)
        assert zoom == 0


class TestComputeGrid:
    """Tests for compute_grid function."""

    def test_returns_correct_structure(self):
        """Should return tuple with correct structure."""
        centers, tiles_xy, grid_size, crop_rect, map_params = compute_grid(
            55.75, 37.62, 5000.0, 5000.0, 15
        )
        assert isinstance(centers, list)
        assert len(tiles_xy) == 2
        assert len(grid_size) == 2
        assert len(crop_rect) == 4
        assert len(map_params) == 7

    def test_centers_count_matches_tiles(self):
        """Number of centers should match tiles_x * tiles_y."""
        centers, (tiles_x, tiles_y), _, _, _ = compute_grid(
            55.75, 37.62, 5000.0, 5000.0, 15
        )
        assert len(centers) == tiles_x * tiles_y

    def test_with_padding(self):
        """Should handle padding correctly."""
        _, tiles_no_pad, _, _, _ = compute_grid(55.75, 37.62, 5000.0, 5000.0, 15, pad_px=0)
        _, tiles_with_pad, _, _, _ = compute_grid(55.75, 37.62, 5000.0, 5000.0, 15, pad_px=500)
        # With padding, may need more tiles
        assert tiles_with_pad[0] >= tiles_no_pad[0]
        assert tiles_with_pad[1] >= tiles_no_pad[1]


class TestEffectiveScaleForXyz:
    """Tests for effective_scale_for_xyz function."""

    def test_tile_256_no_retina(self):
        """256 tile without retina should return 1."""
        scale = effective_scale_for_xyz(256, use_retina=False)
        assert scale == 1

    def test_tile_256_with_retina(self):
        """256 tile with retina should return 2."""
        scale = effective_scale_for_xyz(256, use_retina=True)
        assert scale == 2

    def test_tile_512_no_retina(self):
        """512 tile without retina should return 2."""
        scale = effective_scale_for_xyz(512, use_retina=False)
        assert scale == 2

    def test_tile_512_with_retina(self):
        """512 tile with retina should return 4."""
        scale = effective_scale_for_xyz(512, use_retina=True)
        assert scale == 4


class TestDecodeTerrainRgbToElevationM:
    """Tests for decode_terrain_rgb_to_elevation_m function."""

    def test_sea_level(self):
        """RGB encoding for ~0m elevation."""
        # Mapbox Terrain-RGB: elevation = -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
        # For 0m: R*256*256 + G*256 + B = 100000 -> R=1, G=134, B=160
        img = Image.new('RGB', (1, 1), color=(1, 134, 160))
        dem = decode_terrain_rgb_to_elevation_m(img)
        assert len(dem) == 1
        assert len(dem[0]) == 1
        assert abs(dem[0][0]) < 1  # Should be close to 0

    def test_positive_elevation(self):
        """Should decode positive elevation correctly."""
        # For 1000m: value = 110000 -> R=1, G=173, B=112
        img = Image.new('RGB', (1, 1), color=(1, 173, 112))
        dem = decode_terrain_rgb_to_elevation_m(img)
        assert abs(dem[0][0] - 1000) < 10  # Allow some tolerance

    def test_image_dimensions(self):
        """Output should match input dimensions."""
        img = Image.new('RGB', (10, 20), color=(1, 134, 160))
        dem = decode_terrain_rgb_to_elevation_m(img)
        assert len(dem) == 20  # height
        assert len(dem[0]) == 10  # width


class TestComputeXyzCoverage:
    """Tests for compute_xyz_coverage function."""

    def test_returns_correct_structure(self):
        """Should return tuple with correct structure."""
        tiles, counts, crop_rect, map_params = compute_xyz_coverage(
            55.75, 37.62, 5000.0, 5000.0, 15, 2, 0
        )
        assert isinstance(tiles, list)
        assert len(counts) == 2
        assert len(crop_rect) == 4
        assert len(map_params) == 7

    def test_tiles_count_matches(self):
        """Number of tiles should match counts."""
        tiles, (count_x, count_y), _, _ = compute_xyz_coverage(
            55.75, 37.62, 5000.0, 5000.0, 15, 2, 0
        )
        assert len(tiles) == count_x * count_y

    def test_with_padding(self):
        """Should handle padding correctly."""
        _, counts_no_pad, _, _ = compute_xyz_coverage(55.75, 37.62, 5000.0, 5000.0, 15, 2, 0)
        _, counts_with_pad, _, _ = compute_xyz_coverage(55.75, 37.62, 5000.0, 5000.0, 15, 2, 500)
        # With padding, may need more tiles
        assert counts_with_pad[0] >= counts_no_pad[0]
        assert counts_with_pad[1] >= counts_no_pad[1]

    def test_crop_rect_positive(self):
        """Crop rect dimensions should be positive."""
        _, _, (cx, cy, cw, ch), _ = compute_xyz_coverage(
            55.75, 37.62, 5000.0, 5000.0, 15, 2, 0
        )
        assert cw > 0
        assert ch > 0

    def test_different_zoom_levels(self):
        """Different zoom levels should produce different tile counts."""
        _, counts_z10, _, _ = compute_xyz_coverage(55.75, 37.62, 10000.0, 10000.0, 10, 2, 0)
        _, counts_z15, _, _ = compute_xyz_coverage(55.75, 37.62, 10000.0, 10000.0, 15, 2, 0)
        # Higher zoom should have more tiles
        assert counts_z15[0] * counts_z15[1] > counts_z10[0] * counts_z10[1]

    def test_tiles_have_valid_coords(self):
        """All tiles should have valid x, y coordinates."""
        tiles, _, _, _ = compute_xyz_coverage(55.75, 37.62, 5000.0, 5000.0, 15, 2, 0)
        max_tile = 2 ** 15
        for x, y in tiles:
            assert 0 <= x < max_tile
            assert 0 <= y < max_tile


class TestMetersPerPixelExtended:
    """Extended tests for meters_per_pixel function."""

    def test_higher_zoom_smaller_mpp(self):
        """Higher zoom should give smaller meters per pixel."""
        mpp_z10 = meters_per_pixel(55.0, 10)
        mpp_z15 = meters_per_pixel(55.0, 15)
        assert mpp_z15 < mpp_z10

    def test_equator_vs_high_latitude(self):
        """Equator should have larger mpp than high latitude."""
        mpp_equator = meters_per_pixel(0.0, 15)
        mpp_high = meters_per_pixel(70.0, 15)
        assert mpp_equator > mpp_high

    def test_scale_factor(self):
        """Scale factor should affect mpp."""
        mpp_s1 = meters_per_pixel(55.0, 15, scale=1)
        mpp_s2 = meters_per_pixel(55.0, 15, scale=2)
        assert mpp_s1 > mpp_s2


class TestLatlngPixelConversionExtended:
    """Extended tests for coordinate conversions."""

    def test_roundtrip_conversion(self):
        """Converting lat/lng to pixel and back should give same coords."""
        lat, lng = 55.75, 37.62
        zoom = 15
        x, y = latlng_to_pixel_xy(lat, lng, zoom)
        lat2, lng2 = pixel_xy_to_latlng(x, y, zoom)
        assert abs(lat - lat2) < 0.0001
        assert abs(lng - lng2) < 0.0001

    def test_different_zoom_levels(self):
        """Different zoom levels should give different pixel coords."""
        lat, lng = 55.75, 37.62
        x10, y10 = latlng_to_pixel_xy(lat, lng, 10)
        x15, y15 = latlng_to_pixel_xy(lat, lng, 15)
        assert x15 > x10
        assert y15 > y10

    def test_negative_longitude(self):
        """Should handle negative longitude."""
        x, y = latlng_to_pixel_xy(40.0, -74.0, 15)
        assert x > 0
        assert y > 0

    def test_southern_hemisphere(self):
        """Should handle southern hemisphere."""
        x, y = latlng_to_pixel_xy(-33.9, 18.4, 15)
        assert x > 0
        assert y > 0


class TestEstimateCropSizeExtended:
    """Extended tests for estimate_crop_size_px."""

    def test_larger_area_more_pixels(self):
        """Larger area should need more pixels."""
        w1, h1, _ = estimate_crop_size_px(55.0, 1000.0, 1000.0, 15)
        w2, h2, _ = estimate_crop_size_px(55.0, 5000.0, 5000.0, 15)
        assert w2 > w1
        assert h2 > h1

    def test_higher_zoom_more_pixels(self):
        """Higher zoom should give more pixels."""
        _, _, total_z10 = estimate_crop_size_px(55.0, 5000.0, 5000.0, 10)
        _, _, total_z15 = estimate_crop_size_px(55.0, 5000.0, 5000.0, 15)
        assert total_z15 > total_z10

    def test_asymmetric_dimensions(self):
        """Should handle asymmetric width/height."""
        w, h, _ = estimate_crop_size_px(55.0, 10000.0, 5000.0, 15)
        assert w > h


class TestComputeGridExtended:
    """Extended tests for compute_grid."""

    def test_small_area(self):
        """Should handle small area."""
        centers, (tx, ty), _, _, _ = compute_grid(55.75, 37.62, 500.0, 500.0, 15)
        assert tx >= 1
        assert ty >= 1

    def test_large_area(self):
        """Should handle large area."""
        centers, (tx, ty), _, _, _ = compute_grid(55.75, 37.62, 50000.0, 50000.0, 12)
        assert tx > 1
        assert ty > 1

    def test_map_params_structure(self):
        """Map params should have correct structure."""
        _, _, _, _, map_params = compute_grid(55.75, 37.62, 5000.0, 5000.0, 15)
        assert len(map_params) == 7
        # Check first element (mpp) is positive
        assert map_params[0] > 0

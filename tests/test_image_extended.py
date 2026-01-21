
import pytest
from PIL import Image
from pyproj import CRS, Transformer
from image import draw_axis_aligned_km_grid, draw_elevation_legend
from topography import (
    build_transformers_sk42, crs_sk42_geog, compute_grid,
    compute_xyz_coverage, decode_terrain_rgb_to_elevation_m,
    colorize_dem_to_image
)

@pytest.fixture
def sk42_gk_crs():
    # Zone 7 for example
    return CRS.from_string("+proj=tmerc +lat_0=0 +lon_0=39 +k=1 +x_0=7500000 +y_0=0 +ellps=krass +units=m +no_defs")

@pytest.fixture
def sk42_to_wgs_transformer():
    return Transformer.from_crs(crs_sk42_geog, "EPSG:4326", always_xy=True)

class TestDrawAxisAlignedKmGrid:
    def test_draw_grid_full(self, sk42_gk_crs, sk42_to_wgs_transformer):
        img = Image.new('RGB', (1000, 1000), color='white')
        draw_axis_aligned_km_grid(
            img,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=True
        )
        # Check if something was drawn
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_draw_grid_crosses_only(self, sk42_gk_crs, sk42_to_wgs_transformer):
        img = Image.new('RGB', (1000, 1000), color='white')
        draw_axis_aligned_km_grid(
            img,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=False
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

    def test_draw_grid_with_legend_bounds(self, sk42_gk_crs, sk42_to_wgs_transformer):
        img = Image.new('RGB', (1000, 1000), color='white')
        legend_bounds = (100, 100, 400, 400)
        draw_axis_aligned_km_grid(
            img,
            center_lat_sk42=55.75,
            center_lng_sk42=37.62,
            center_lat_wgs=55.75,
            center_lng_wgs=37.62,
            zoom=12,
            crs_sk42_gk=sk42_gk_crs,
            t_sk42_to_wgs=sk42_to_wgs_transformer,
            display_grid=True,
            legend_bounds=legend_bounds
        )
        # Just ensure it runs without error
        assert img.size == (1000, 1000)

class TestDrawElevationLegend:
    def test_draw_elevation_legend_basic(self):
        img = Image.new('RGB', (1000, 1000), color='white')
        color_ramp = [
            (0.0, (0, 0, 255)),
            (0.5, (0, 255, 0)),
            (1.0, (255, 0, 0))
        ]
        draw_elevation_legend(
            img,
            color_ramp=color_ramp,
            min_elevation_m=0,
            max_elevation_m=1000,
            center_lat_wgs=55.75,
            zoom=12
        )
        pixels = list(img.getdata())
        non_white = [p for p in pixels if p != (255, 255, 255)]
        assert len(non_white) > 0

class TestTopographyExtended:
    def test_build_transformers_sk42_with_helmert(self):
        helmert = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        t_sk42_to_wgs, t_wgs_to_sk42 = build_transformers_sk42(helmert)
        assert t_sk42_to_wgs is not None
        assert t_wgs_to_sk42 is not None

    def test_compute_grid(self):
        # center_lat, center_lng, width_m, height_m, zoom
        grid = compute_grid(55.75, 37.62, 1000, 1000, 12)
        # grid: tuple of (tiles, tiles_grid, assembled_size, crop_rect, map_params)
        assert len(grid) == 5
        assert isinstance(grid[0], list)

    def test_compute_xyz_coverage(self):
        coverage = compute_xyz_coverage(55.75, 37.62, 1000, 1000, 12, 1, 0)
        # coverage: tuple of (tiles, tiles_grid, crop_rect, map_params)
        assert len(coverage) == 4
        assert isinstance(coverage[0], list)

    def test_decode_terrain_rgb_to_elevation_m(self):
        # Create a terrain RGB image
        # R=1, G=2, B=3 -> height = -10000 + (R*256*256 + G*256 + B)*0.1
        # height = -10000 + (65536 + 512 + 3)*0.1 = -10000 + 6605.1 = -3394.9
        img = Image.new('RGB', (10, 10), color=(1, 2, 3))
        elev = decode_terrain_rgb_to_elevation_m(img)
        assert elev[0][0] == pytest.approx(-3394.9, rel=1e-3)

    def test_colorize_dem_to_image(self):
        dem = [[0.0, 500.0], [1000.0, 1500.0]]
        img = colorize_dem_to_image(dem)
        assert isinstance(img, Image.Image)
        assert img.size == (2, 2)

"""Tests for NSU optimizer computation kernel and processor."""

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from PIL import Image

from services.map_context import MapDownloadContext
from services.radio_horizon import (
    compute_and_colorize_nsu_optimizer,
    recompute_nsu_optimizer_fast,
)


# ---------------------------------------------------------------------------
# Tests for compute_and_colorize_nsu_optimizer
# ---------------------------------------------------------------------------

def test_compute_nsu_optimizer_returns_image():
    """Basic test: flat DEM with one target should produce a valid RGB image."""
    dem = np.ones((64, 64), dtype=np.float32) * 100.0  # Flat terrain at 100m
    target_rows = np.array([32], dtype=np.int32)
    target_cols = np.array([32], dtype=np.int32)

    result = compute_and_colorize_nsu_optimizer(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        max_height_m=500.0,
        grid_step=4,
    )

    assert isinstance(result, Image.Image)
    assert result.mode == 'RGB'
    assert result.size == (64, 64)


def test_compute_nsu_optimizer_multiple_targets():
    """Multiple targets should produce a valid result."""
    dem = np.ones((64, 64), dtype=np.float32) * 100.0
    target_rows = np.array([10, 30, 50], dtype=np.int32)
    target_cols = np.array([10, 30, 50], dtype=np.int32)

    result = compute_and_colorize_nsu_optimizer(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        max_height_m=500.0,
        grid_step=4,
    )

    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)


def test_compute_nsu_optimizer_empty_dem():
    """Empty DEM should return a default-sized image."""
    dem = np.array([], dtype=np.float32).reshape(0, 0)
    target_rows = np.array([], dtype=np.int32)
    target_cols = np.array([], dtype=np.int32)

    result = compute_and_colorize_nsu_optimizer(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
    )

    assert isinstance(result, Image.Image)


def test_compute_nsu_optimizer_empty_targets():
    """DEM with no targets should return a default-sized image."""
    dem = np.ones((64, 64), dtype=np.float32) * 100.0
    target_rows = np.array([], dtype=np.int32)
    target_cols = np.array([], dtype=np.int32)

    result = compute_and_colorize_nsu_optimizer(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
    )

    assert isinstance(result, Image.Image)


def test_compute_nsu_optimizer_flat_terrain_low_heights():
    """On flat terrain, LOS is always clear so required height should be ~0."""
    dem = np.zeros((32, 32), dtype=np.float32)  # Perfectly flat at sea level
    target_rows = np.array([16], dtype=np.int32)
    target_cols = np.array([16], dtype=np.int32)

    result = compute_and_colorize_nsu_optimizer(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=100.0,
        max_height_m=500.0,
        grid_step=2,
    )

    # The image should have mostly green-ish tones (low heights)
    arr = np.array(result)
    # Check that most pixels are not the unreachable gray color (128, 128, 128)
    gray_mask = (arr[:, :, 0] == 128) & (arr[:, :, 1] == 128) & (arr[:, :, 2] == 128)
    gray_fraction = gray_mask.sum() / (32 * 32)
    assert gray_fraction < 0.5, f'Too many unreachable pixels on flat terrain: {gray_fraction:.2%}'


# ---------------------------------------------------------------------------
# Tests for recompute_nsu_optimizer_fast
# ---------------------------------------------------------------------------

def test_recompute_nsu_optimizer_fast_basic():
    """recompute_fast should return (blended_image, coverage_layer) tuple."""
    dem = np.ones((32, 32), dtype=np.float32) * 100.0
    topo_base = Image.new('RGBA', (32, 32), (128, 128, 128, 255))
    target_rows = np.array([16], dtype=np.int32)
    target_cols = np.array([16], dtype=np.int32)

    blended, coverage = recompute_nsu_optimizer_fast(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        topo_base=topo_base,
        overlay_alpha=0.3,
        max_height_m=500.0,
        grid_step=2,
    )

    assert isinstance(blended, Image.Image)
    assert isinstance(coverage, Image.Image)


def test_recompute_nsu_optimizer_fast_with_rotation():
    """recompute_fast with rotation should resize, rotate, and crop."""
    dem = np.ones((64, 64), dtype=np.float32) * 100.0
    topo_base = Image.new('RGBA', (64, 64), (128, 128, 128, 255))
    target_rows = np.array([32], dtype=np.int32)
    target_cols = np.array([32], dtype=np.int32)

    blended, coverage = recompute_nsu_optimizer_fast(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        topo_base=topo_base,
        overlay_alpha=0.3,
        max_height_m=500.0,
        grid_step=4,
        crop_size=(64, 64),
        final_size=(50, 50),
        rotation_deg=5.0,
    )

    assert isinstance(blended, Image.Image)
    # Blended stays at DEM-proportional size (not final_size)
    assert blended.size[0] <= 64
    assert blended.size[1] <= 64


# ---------------------------------------------------------------------------
# Tests for memory-efficient recompute (coverage should stay at DEM size)
# ---------------------------------------------------------------------------

def test_recompute_nsu_no_image_exceeds_dem_size_with_rotation():
    """Neither blended nor coverage should be upscaled to final_size.
    Both must stay at DEM-proportional size to avoid OOM on large maps."""
    dem_size = 64
    final_w, final_h = 512, 512
    crop_w, crop_h = 600, 600

    dem = np.ones((dem_size, dem_size), dtype=np.float32) * 100.0
    topo_base = Image.new('RGBA', (dem_size, dem_size), (128, 128, 128, 255))
    target_rows = np.array([32], dtype=np.int32)
    target_cols = np.array([32], dtype=np.int32)

    blended, coverage = recompute_nsu_optimizer_fast(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        topo_base=topo_base,
        overlay_alpha=0.3,
        max_height_m=500.0,
        grid_step=4,
        crop_size=(crop_w, crop_h),
        final_size=(final_w, final_h),
        rotation_deg=5.0,
    )

    # Neither image should be at final_size — both stay compact
    assert blended.size[0] <= dem_size, (
        f'blended width {blended.size[0]} should be <= DEM size {dem_size}, '
        f'not upscaled to final_size {final_w}'
    )
    assert coverage.size[0] <= dem_size, (
        f'coverage width {coverage.size[0]} should be <= DEM size {dem_size}, '
        f'not upscaled to final_size {final_w}'
    )


def test_recompute_nsu_no_image_exceeds_dem_size_no_rotation():
    """Even without rotation, no returned image should be at final_size."""
    dem_size = 64
    final_w, final_h = 512, 512
    crop_w, crop_h = 600, 600

    dem = np.ones((dem_size, dem_size), dtype=np.float32) * 100.0
    topo_base = Image.new('RGBA', (dem_size, dem_size), (128, 128, 128, 255))
    target_rows = np.array([32], dtype=np.int32)
    target_cols = np.array([32], dtype=np.int32)

    blended, coverage = recompute_nsu_optimizer_fast(
        dem=dem,
        target_rows=target_rows,
        target_cols=target_cols,
        antenna_height_m=10.0,
        pixel_size_m=10.0,
        topo_base=topo_base,
        overlay_alpha=0.3,
        max_height_m=500.0,
        grid_step=4,
        crop_size=(crop_w, crop_h),
        final_size=(final_w, final_h),
        rotation_deg=0.0,
    )

    assert blended.size[0] <= dem_size
    assert coverage.size[0] <= dem_size


# ---------------------------------------------------------------------------
# Tests for process_nsu_optimizer (processor)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_nsu_optimizer_no_points():
    """With no target points, processor should return clean topo base."""
    ctx = MapDownloadContext(
        center_x_sk42_gk=1000.0,
        center_y_sk42_gk=1000.0,
        width_m=500.0,
        height_m=500.0,
        api_key='test_key',
        output_path='test.png',
        max_zoom=10,
        settings=AsyncMock(),
    )
    ctx.settings.nsu_target_points = []
    ctx.settings.nsu_antenna_height_m = 10.0
    ctx.settings.nsu_max_flight_height_m = 500.0
    ctx.settings.nsu_overlay_alpha = 0.3
    ctx.settings.overlay_contours = False

    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()
    ctx.target_w_px = 256
    ctx.target_h_px = 256
    ctx.full_eff_tile_px = 512
    ctx.center_lat_wgs = 50.0
    ctx.eff_scale = 1

    with patch('services.processors.radio_horizon.ElevationTileProvider') as mock_prov_cls, \
         patch('services.processors.radio_horizon.async_fetch_xyz_tile', new_callable=AsyncMock) as mock_fetch, \
         patch('services.processors.radio_horizon.meters_per_pixel') as mock_mpp, \
         patch('services.processors.radio_horizon.assemble_dem') as mock_assemble:

        mock_prov = mock_prov_cls.return_value
        mock_prov.get_tile_image = AsyncMock(return_value=Image.new('RGB', (256, 256)))
        mock_prov.get_tile_dem = AsyncMock(return_value=np.zeros((256, 256), dtype=np.float32))

        mock_fetch.return_value = Image.new('RGB', (512, 512))
        mock_mpp.return_value = 1.0
        mock_assemble.return_value = np.zeros((256, 256), dtype=np.float32)

        from services.processors.nsu_optimizer import process_nsu_optimizer
        result = await process_nsu_optimizer(ctx)

        assert isinstance(result, Image.Image)
        # No target points → clean topo base returned
        assert ctx.nsu_cache_dem is not None


@pytest.mark.asyncio
async def test_process_nsu_optimizer_with_points():
    """With target points, processor should compute coverage and return blended image."""
    ctx = MapDownloadContext(
        center_x_sk42_gk=1000.0,
        center_y_sk42_gk=1000.0,
        width_m=500.0,
        height_m=500.0,
        api_key='test_key',
        output_path='test.png',
        max_zoom=10,
        settings=AsyncMock(),
    )
    ctx.settings.nsu_target_points = [(5415000, 7440000)]
    ctx.settings.nsu_antenna_height_m = 10.0
    ctx.settings.nsu_max_flight_height_m = 500.0
    ctx.settings.nsu_overlay_alpha = 0.3
    ctx.settings.overlay_contours = False

    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()
    ctx.target_w_px = 256
    ctx.target_h_px = 256
    ctx.full_eff_tile_px = 512
    ctx.center_lat_wgs = 50.0
    ctx.eff_scale = 1

    ctx.t_sk42_from_gk = AsyncMock()
    ctx.t_sk42_from_gk.transform = lambda x, y: (50.0, 30.0)
    ctx.t_sk42_to_wgs = AsyncMock()
    ctx.t_sk42_to_wgs.transform = lambda x, y: (50.0, 30.0)

    with patch('services.processors.radio_horizon.ElevationTileProvider') as mock_prov_cls, \
         patch('services.processors.radio_horizon.async_fetch_xyz_tile', new_callable=AsyncMock) as mock_fetch, \
         patch('services.processors.nsu_optimizer.meters_per_pixel') as mock_mpp_nsu, \
         patch('services.processors.radio_horizon.meters_per_pixel') as mock_mpp, \
         patch('services.processors.radio_horizon.assemble_dem') as mock_assemble, \
         patch('services.processors.nsu_optimizer.compute_and_colorize_nsu_optimizer') as mock_compute:

        mock_prov = mock_prov_cls.return_value
        mock_prov.get_tile_image = AsyncMock(return_value=Image.new('RGB', (256, 256)))
        mock_prov.get_tile_dem = AsyncMock(return_value=np.zeros((256, 256), dtype=np.float32))

        mock_fetch.return_value = Image.new('RGB', (512, 512))
        mock_mpp.return_value = 1.0
        mock_mpp_nsu.return_value = 1.0
        mock_assemble.return_value = np.zeros((256, 256), dtype=np.float32)
        mock_compute.return_value = Image.new('RGB', (256, 256), (0, 255, 0))

        from services.processors.nsu_optimizer import process_nsu_optimizer
        result = await process_nsu_optimizer(ctx)

        assert isinstance(result, Image.Image)
        mock_compute.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for MapSettings NSU fields
# ---------------------------------------------------------------------------

_SETTINGS_DEFAULTS = dict(
    from_x_high=54, from_y_high=74, to_x_high=54, to_y_high=74,
    from_x_low=18, from_y_low=46, to_x_low=19, to_y_low=47,
    grid_width_m=5.0, grid_font_size_m=100.0, grid_text_margin_m=50.0,
    grid_label_bg_padding_m=10.0, mask_opacity=0.35,
)


def _make_settings(**overrides):
    from domain.models import MapSettings
    kw = {**_SETTINGS_DEFAULTS, **overrides}
    return MapSettings(**kw)


def test_map_settings_nsu_target_points_property():
    """nsu_target_points property should parse JSON correctly."""
    s = _make_settings()
    assert s.nsu_target_points == []

    s.nsu_target_points_json = '[[5415000, 7440000], [5420000, 7445000]]'
    pts = s.nsu_target_points
    assert len(pts) == 2
    assert pts[0] == (5415000, 7440000)
    assert pts[1] == (5420000, 7445000)


def test_map_settings_nsu_target_points_invalid_json():
    """Invalid JSON should return empty list."""
    s = _make_settings(nsu_target_points_json='not json')
    assert s.nsu_target_points == []


def test_map_settings_nsu_defaults():
    """Default NSU field values should be sensible."""
    s = _make_settings()
    assert s.nsu_antenna_height_m == 10.0
    assert s.nsu_max_flight_height_m == 500.0
    assert 0.0 <= s.nsu_overlay_alpha <= 1.0


# ---------------------------------------------------------------------------
# Tests for MapType constants
# ---------------------------------------------------------------------------

def test_nsu_optimizer_map_type_exists():
    """NSU_OPTIMIZER should be a valid MapType."""
    from shared.constants import MAP_TYPE_LABELS_RU, MapType

    assert hasattr(MapType, 'NSU_OPTIMIZER')
    assert MapType.NSU_OPTIMIZER in MAP_TYPE_LABELS_RU


def test_nsu_optimizer_constants():
    """NSU optimizer constants should be defined."""
    from shared.constants import (
        NSU_OPTIMIZER_MAX_POINTS,
        NSU_OPTIMIZER_POINT_COLORS,
        NSU_OPTIMIZER_USE_RETINA,
    )

    assert isinstance(NSU_OPTIMIZER_MAX_POINTS, int)
    assert NSU_OPTIMIZER_MAX_POINTS > 0
    assert isinstance(NSU_OPTIMIZER_POINT_COLORS, list)
    assert len(NSU_OPTIMIZER_POINT_COLORS) >= 1
    assert isinstance(NSU_OPTIMIZER_USE_RETINA, bool)

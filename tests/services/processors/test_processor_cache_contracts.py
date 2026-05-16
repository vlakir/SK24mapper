"""Regression tests: processors must save cache layers on ctx BEFORE blend.

Prevents:
  BUG3 — processor does not save cache layer before blend (no rh_cache_coverage
         for interactive alpha).
  BUG4 — cache layer sizes do not match (topo_base vs coverage_layer).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from services.map_context import MapDownloadContext
from services.processors.elevation_color import process_elevation_color
from services.processors.radar_coverage import process_radar_coverage
from services.processors.radio_horizon import process_radio_horizon


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SIZE = (256, 256)
_DEM = np.random.default_rng(42).uniform(100.0, 500.0, (256, 256)).astype(np.float32)


def _make_settings(**overrides: object) -> MagicMock:
    settings = MagicMock()
    settings.radio_horizon_overlay_alpha = 0.3
    settings.antenna_height_m = 10.0
    settings.max_flight_height_m = 120.0
    settings.uav_height_reference = 'ground'
    settings.control_point_x_sk42_gk = 1000.0
    settings.control_point_y_sk42_gk = 1000.0
    # Radar-specific
    settings.radar_azimuth_deg = 0.0
    settings.radar_sector_width_deg = 90.0
    settings.radar_elevation_min_deg = 0.5
    settings.radar_elevation_max_deg = 30.0
    settings.radar_max_range_km = 15.0
    settings.radar_target_height_min_m = 0.0
    settings.radar_target_height_max_m = 300.0
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_ctx(settings: MagicMock | None = None) -> MapDownloadContext:
    if settings is None:
        settings = _make_settings()
    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=100.0,
        height_m=100.0,
        api_key='test_key',
        output_path='test.png',
        max_zoom=10,
        settings=settings,
    )
    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.target_w_px = 256
    ctx.target_h_px = 256
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()
    ctx.center_lat_wgs = 50.0
    ctx.center_lng_wgs = 30.0
    ctx.eff_scale = 1
    # Transformers (non-async, return dummy coords)
    ctx.t_sk42_from_gk = MagicMock()
    ctx.t_sk42_from_gk.transform = lambda x, y: (50.0, 30.0)
    ctx.t_sk42_to_wgs = MagicMock()
    ctx.t_sk42_to_wgs.transform = lambda x, y: (50.0, 30.0)
    ctx.t_gk_from_sk42 = MagicMock()
    ctx.t_gk_from_sk42.transform = lambda x, y: (1000.0, 1000.0)
    return ctx


def _topo_image(size: tuple[int, int] = _SIZE) -> Image.Image:
    return Image.new('RGB', size, (128, 128, 128))


# ---------------------------------------------------------------------------
# Patchers for radio_horizon / radar_coverage processors
# ---------------------------------------------------------------------------

def _patch_rh_loaders():
    """Patch _load_dem_and_topo for radio_horizon and radar_coverage processors."""
    dem = _DEM.copy()
    topo = _topo_image()
    antenna_row, antenna_col = 128, 128
    pixel_size_m = 1.0
    cp_elevation = 200.0
    ds_factor = 1

    async def fake_load_dem_and_topo(ctx, **kwargs):
        ctx.raw_dem_for_cursor = dem.copy()
        ctx.rh_cache_crop_size = (dem.shape[1], dem.shape[0])
        ctx.rh_cache_topo_base = topo.convert('L').convert('RGBA')
        ctx.rh_cache_dem_full = dem.copy()
        ctx.rh_cache_dem = dem.copy()
        ctx.rh_cache_antenna_row = antenna_row
        ctx.rh_cache_antenna_col = antenna_col
        ctx.rh_cache_pixel_size_m = pixel_size_m
        return dem, topo, antenna_row, antenna_col, pixel_size_m, cp_elevation, ds_factor

    return patch(
        'services.processors.radio_horizon._load_dem_and_topo',
        side_effect=fake_load_dem_and_topo,
    )


def _patch_compute_coverage(result_size: tuple[int, int] = _SIZE):
    """Patch compute_and_colorize_coverage to return a fixed RGBA image."""
    result = Image.new('RGBA', result_size, (255, 0, 0, 200))
    return patch(
        'services.processors.radio_horizon.compute_and_colorize_coverage',
        return_value=result,
    )


def _patch_radar_compute_coverage(result_size: tuple[int, int] = _SIZE):
    """Patch compute_and_colorize_coverage in radar_coverage module."""
    result = Image.new('RGBA', result_size, (0, 0, 255, 200))
    return patch(
        'services.processors.radar_coverage.compute_and_colorize_coverage',
        return_value=result,
    )


def _patch_radar_loaders():
    """Patch _load_dem_and_topo used by radar_coverage processor."""
    dem = _DEM.copy()
    topo = _topo_image()
    antenna_row, antenna_col = 128, 128
    pixel_size_m = 1.0
    cp_elevation = 200.0
    ds_factor = 1

    async def fake_load_dem_and_topo(ctx, **kwargs):
        ctx.raw_dem_for_cursor = dem.copy()
        ctx.rh_cache_crop_size = (dem.shape[1], dem.shape[0])
        ctx.rh_cache_topo_base = topo.convert('L').convert('RGBA')
        ctx.rh_cache_dem_full = dem.copy()
        ctx.rh_cache_dem = dem.copy()
        ctx.rh_cache_antenna_row = antenna_row
        ctx.rh_cache_antenna_col = antenna_col
        ctx.rh_cache_pixel_size_m = pixel_size_m
        return dem, topo, antenna_row, antenna_col, pixel_size_m, cp_elevation, ds_factor

    # radar_coverage imports _load_dem_and_topo from radio_horizon
    return patch(
        'services.processors.radar_coverage._load_dem_and_topo',
        side_effect=fake_load_dem_and_topo,
    )


# Patchers for elevation_color
def _patch_elev_loaders(dem: np.ndarray | None = None, topo_size: tuple[int, int] = _SIZE):
    topo = _topo_image(topo_size)
    if dem is None:
        dem = _DEM.copy()
    dem_patch = patch(
        'services.processors.elevation_color._load_dem',
        new_callable=AsyncMock,
        return_value=(dem, 1),
    )
    topo_patch = patch(
        'services.processors.elevation_color._load_topo',
        new_callable=AsyncMock,
        return_value=topo,
    )
    return dem_patch, topo_patch


# ---------------------------------------------------------------------------
# BUG3: processor must save rh_cache_coverage before blend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_BUG3_radio_horizon_saves_coverage_cache():
    """process_radio_horizon must set ctx.rh_cache_coverage (RGBA) before blend."""
    ctx = _make_ctx()
    with _patch_rh_loaders(), _patch_compute_coverage():
        await process_radio_horizon(ctx)

    assert ctx.rh_cache_coverage is not None, (
        'BUG3: process_radio_horizon did not save rh_cache_coverage'
    )
    assert isinstance(ctx.rh_cache_coverage, Image.Image)
    assert ctx.rh_cache_coverage.mode == 'RGBA'


@pytest.mark.asyncio
async def test_BUG3_radar_coverage_saves_coverage_cache():
    """process_radar_coverage must set ctx.rh_cache_coverage before blend."""
    ctx = _make_ctx()
    ctx.is_radar_coverage = True
    with _patch_radar_loaders(), _patch_radar_compute_coverage():
        await process_radar_coverage(ctx)

    assert ctx.rh_cache_coverage is not None, (
        'BUG3: process_radar_coverage did not save rh_cache_coverage'
    )
    assert isinstance(ctx.rh_cache_coverage, Image.Image)


@pytest.mark.asyncio
async def test_BUG3_elevation_color_saves_coverage_and_topo():
    """process_elevation_color must set both rh_cache_coverage AND rh_cache_topo_base."""
    ctx = _make_ctx()
    dem_p, topo_p = _patch_elev_loaders()
    with dem_p, topo_p:
        await process_elevation_color(ctx)

    assert ctx.rh_cache_coverage is not None, (
        'BUG3: process_elevation_color did not save rh_cache_coverage'
    )
    assert ctx.rh_cache_topo_base is not None, (
        'BUG3: process_elevation_color did not save rh_cache_topo_base'
    )


# ---------------------------------------------------------------------------
# BUG4: cache layer sizes must be compatible
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_BUG4_coverage_cache_size_matches_crop_rect():
    """rh_cache_coverage.size must match crop_rect dimensions."""
    ctx = _make_ctx()
    with _patch_rh_loaders(), _patch_compute_coverage():
        await process_radio_horizon(ctx)

    expected_size = (ctx.crop_rect[2], ctx.crop_rect[3])
    assert ctx.rh_cache_coverage.size == expected_size, (
        f'BUG4: coverage cache size {ctx.rh_cache_coverage.size} != '
        f'crop_rect size {expected_size}'
    )


@pytest.mark.asyncio
async def test_BUG4_topo_base_cache_size_matches_coverage():
    """For elevation_color: rh_cache_topo_base.size must match rh_cache_coverage.size."""
    ctx = _make_ctx()
    dem_p, topo_p = _patch_elev_loaders()
    with dem_p, topo_p:
        await process_elevation_color(ctx)

    assert ctx.rh_cache_topo_base.size == ctx.rh_cache_coverage.size, (
        f'BUG4: topo_base size {ctx.rh_cache_topo_base.size} != '
        f'coverage size {ctx.rh_cache_coverage.size}'
    )

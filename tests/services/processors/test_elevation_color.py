import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from services.map_context import MapDownloadContext
from services.processors.elevation_color import process_elevation_color


def _make_ctx(overlay_alpha=0.3):
    """Helper to create a MapDownloadContext with sensible defaults."""
    settings = MagicMock()
    settings.radio_horizon_overlay_alpha = overlay_alpha
    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=100.0,
        height_m=100.0,
        api_key="test_key",
        output_path="test.png",
        max_zoom=10,
        settings=settings,
    )
    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()
    return ctx


def _topo_image(size=(256, 256)):
    """Return a simple RGB topo base image for mocking."""
    return Image.new('RGB', size, (128, 128, 128))


def _patch_loaders(dem, ds_factor=1, topo_size=(256, 256)):
    """Return patch context managers for both _load_dem and _load_topo."""
    topo = _topo_image(topo_size)
    dem_patch = patch(
        'services.processors.elevation_color._load_dem',
        new_callable=AsyncMock,
        return_value=(dem, ds_factor),
    )
    topo_patch = patch(
        'services.processors.elevation_color._load_topo',
        new_callable=AsyncMock,
        return_value=topo,
    )
    return dem_patch, topo_patch


@pytest.mark.asyncio
async def test_process_elevation_color_basic():
    ctx = _make_ctx()
    dem = np.random.default_rng(42).uniform(100.0, 500.0, (256, 256)).astype(np.float32)
    dem_p, topo_p = _patch_loaders(dem)

    with dem_p, topo_p:
        result = await process_elevation_color(ctx)

        assert isinstance(result, Image.Image)
        assert result.size == (256, 256)
        assert result.mode == 'RGBA'
        assert ctx.elev_min_m is not None
        assert ctx.elev_max_m is not None
        assert ctx.elev_min_m < ctx.elev_max_m


@pytest.mark.asyncio
async def test_process_elevation_color_with_downsample():
    ctx = _make_ctx()
    dem = np.random.default_rng(42).uniform(100.0, 500.0, (128, 128)).astype(np.float32)
    dem_p, topo_p = _patch_loaders(dem, ds_factor=2)

    with dem_p, topo_p:
        result = await process_elevation_color(ctx)

        assert isinstance(result, Image.Image)
        assert result.size == (256, 256)


@pytest.mark.asyncio
async def test_process_elevation_color_error_handling():
    ctx = _make_ctx()

    with patch(
        'services.processors.elevation_color._load_dem',
        new_callable=AsyncMock,
        side_effect=Exception("Failed to fetch"),
    ):
        with pytest.raises(Exception, match="Failed to fetch"):
            await process_elevation_color(ctx)


@pytest.mark.asyncio
async def test_process_elevation_color_flat_terrain():
    """Test with completely flat terrain (all same elevation)."""
    ctx = _make_ctx()
    dem = np.full((256, 256), 200.0, dtype=np.float32)
    dem_p, topo_p = _patch_loaders(dem)

    with dem_p, topo_p:
        result = await process_elevation_color(ctx)

        assert isinstance(result, Image.Image)
        assert result.size == (256, 256)
        assert ctx.elev_max_m > ctx.elev_min_m


@pytest.mark.asyncio
async def test_process_elevation_color_blend_alpha():
    """Test that blend respects overlay alpha setting."""
    dem = np.random.default_rng(42).uniform(100.0, 500.0, (256, 256)).astype(np.float32)

    ctx0 = _make_ctx(overlay_alpha=0.0)
    dem_p0, topo_p0 = _patch_loaders(dem)
    with dem_p0, topo_p0:
        result_alpha0 = await process_elevation_color(ctx0)

    ctx1 = _make_ctx(overlay_alpha=1.0)
    dem_p1, topo_p1 = _patch_loaders(dem)
    with dem_p1, topo_p1:
        result_alpha1 = await process_elevation_color(ctx1)

    # Results should differ when alpha changes
    arr0 = np.array(result_alpha0)
    arr1 = np.array(result_alpha1)
    assert not np.array_equal(arr0, arr1)

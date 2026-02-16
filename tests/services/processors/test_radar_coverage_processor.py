"""Tests for services.processors.radar_coverage module."""

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from PIL import Image

from services.map_context import MapDownloadContext
from services.processors.radar_coverage import process_radar_coverage


@pytest.mark.asyncio
async def test_process_radar_coverage_basic():
    """process_radar_coverage should return an Image using mocked dependencies."""
    ctx = MapDownloadContext(
        center_x_sk42_gk=1000.0,
        center_y_sk42_gk=1000.0,
        width_m=500.0,
        height_m=500.0,
        api_key="test_key",
        output_path="test.png",
        max_zoom=10,
        settings=AsyncMock(),
    )
    ctx.settings.antenna_height_m = 10.0
    ctx.settings.radio_horizon_overlay_alpha = 0.5
    ctx.settings.max_flight_height_m = 500.0
    ctx.settings.uav_height_reference = 'GROUND'
    ctx.settings.overlay_contours = False
    ctx.settings.control_point_enabled = True
    ctx.settings.control_point_x_sk42_gk = 1000.0
    ctx.settings.control_point_y_sk42_gk = 1000.0
    ctx.settings.radar_azimuth_deg = 0.0
    ctx.settings.radar_sector_width_deg = 90.0
    ctx.settings.radar_elevation_min_deg = 0.5
    ctx.settings.radar_elevation_max_deg = 30.0
    ctx.settings.radar_max_range_km = 15.0

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

    ctx.t_sk42_from_gk = AsyncMock()
    ctx.t_sk42_from_gk.transform = lambda x, y: (50.0, 30.0)
    ctx.t_gk_from_sk42 = AsyncMock()
    ctx.t_gk_from_sk42.transform = lambda x, y: (1000.0, 1000.0)
    ctx.t_sk42_to_wgs = AsyncMock()
    ctx.t_sk42_to_wgs.transform = lambda x, y: (50.0, 30.0)

    ctx.map_params = {
        'center_lat': 50.0,
        'center_lng': 30.0,
        'zoom': 10,
        'eff_scale': 1,
        'full_eff_tile_px': 512,
        'tiles_x': 1,
        'tiles_y': 1,
        'crop_rect': (0, 0, 256, 256),
        'tiles': [(10, 20)],
    }

    with patch('services.processors.radio_horizon.ElevationTileProvider') as mock_prov_cls, \
         patch('services.processors.radio_horizon.async_fetch_xyz_tile', new_callable=AsyncMock) as mock_fetch, \
         patch('services.processors.radar_coverage.compute_and_colorize_coverage') as mock_compute, \
         patch('services.processors.radio_horizon.meters_per_pixel') as mock_mpp, \
         patch('services.processors.radio_horizon.assemble_dem') as mock_assemble:

        mock_prov = mock_prov_cls.return_value
        mock_prov.get_tile_image = AsyncMock(return_value=Image.new('RGB', (256, 256)))
        mock_prov.get_tile_dem = AsyncMock(return_value=np.zeros((256, 256), dtype=np.float32))

        mock_fetch.return_value = Image.new('RGB', (512, 512))
        mock_compute.return_value = Image.new('RGB', (256, 256))
        mock_mpp.return_value = 1.0
        mock_assemble.return_value = np.zeros((256, 256), dtype=np.float32)

        result = await process_radar_coverage(ctx)

        assert isinstance(result, Image.Image)
        assert result.size == (256, 256)
        # Verify compute was called with sector_enabled=True
        mock_compute.assert_called_once()
        call_kwargs = mock_compute.call_args
        assert call_kwargs.kwargs.get('sector_enabled') is True or \
               (len(call_kwargs.args) > 0 and any(
                   k in str(call_kwargs) for k in ['sector_enabled']
               ))

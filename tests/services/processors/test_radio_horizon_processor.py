import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from PIL import Image
import numpy as np
from services.map_context import MapDownloadContext
from services.processors.radio_horizon import process_radio_horizon

@pytest.mark.asyncio
async def test_process_radio_horizon_basic():
    ctx = MapDownloadContext(
        center_x_sk42_gk=1000.0,
        center_y_sk42_gk=1000.0,
        width_m=500.0,
        height_m=500.0,
        api_key="test_key",
        output_path="test.png",
        max_zoom=10,
        settings=AsyncMock()
    )
    ctx.settings.radio_antenna_height_m = 10.0
    ctx.settings.radio_max_height_m = 50.0
    ctx.settings.radio_refraction_k = 1.33
    ctx.settings.radio_unreachable_color = (0, 0, 0)
    
    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()
    ctx.target_w_px = 256
    ctx.target_h_px = 256
    ctx.crop_rect = (0, 0, 256, 256)
    
    # Mocking transforms
    ctx.t_sk42_from_gk = AsyncMock()
    ctx.t_sk42_from_gk.transform.side_effect = None
    ctx.t_sk42_from_gk.transform.return_value = (50.0, 30.0)
    
    ctx.t_gk_from_sk42 = AsyncMock()
    ctx.t_gk_from_sk42.transform.side_effect = None
    ctx.t_gk_from_sk42.transform.return_value = (1000.0, 1000.0)

    # We need to make them not async
    ctx.t_sk42_from_gk.transform = lambda x, y: (50.0, 30.0)
    ctx.t_gk_from_sk42.transform = lambda x, y: (1000.0, 1000.0)
    
    ctx.t_sk42_to_wgs = AsyncMock()
    ctx.t_sk42_to_wgs.transform = lambda x, y: (50.0, 30.0)
    ctx.t_wgs_to_sk42 = AsyncMock()
    ctx.t_wgs_to_sk42.transform = lambda x, y: (50.0, 30.0)

    # Mocking compute_xyz_coverage
    ctx.map_params = {
        'center_lat': 50.0,
        'center_lng': 30.0,
        'zoom': 10,
        'eff_scale': 1,
        'full_eff_tile_px': 512,
        'tiles_x': 1,
        'tiles_y': 1,
        'crop_rect': (0, 0, 256, 256),
        'tiles': [(10, 20)]
    }

    # Mocking ElevationTileProvider and topography
    with patch('services.processors.radio_horizon.ElevationTileProvider') as mock_provider_cls, \
         patch('services.processors.radio_horizon.async_fetch_xyz_tile', new_callable=AsyncMock) as mock_fetch_xyz, \
         patch('services.processors.radio_horizon.compute_and_colorize_coverage') as mock_compute, \
         patch('services.processors.radio_horizon.meters_per_pixel') as mock_mpp:
        
        mock_provider = mock_provider_cls.return_value
        mock_provider.get_tile_image = AsyncMock(return_value=Image.new('RGB', (256, 256)))
        mock_provider.get_tile_dem = AsyncMock(return_value=np.zeros((256, 256), dtype=np.float32))
        
        mock_fetch_xyz.return_value = Image.new('RGB', (512, 512))
        mock_compute.return_value = Image.new('RGB', (256, 256))
        mock_mpp.return_value = 1.0
        
        # We need to mock assemble_dem too
        with patch('services.processors.radio_horizon.assemble_dem') as mock_assemble:
            mock_assemble.return_value = np.zeros((256, 256), dtype=np.float32)
            
            result = await process_radio_horizon(ctx)
            
            assert isinstance(result, Image.Image)
            assert result.size == (256, 256)

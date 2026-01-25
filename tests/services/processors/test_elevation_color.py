import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from PIL import Image
import numpy as np
from services.map_context import MapDownloadContext
from services.processors.elevation_color import process_elevation_color

@pytest.mark.asyncio
async def test_process_elevation_color_basic():
    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=100.0,
        height_m=100.0,
        api_key="test_key",
        output_path="test.png",
        max_zoom=10,
        settings=None
    )
    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()

    # Mocking ElevationTileProvider
    mock_img = Image.new('RGB', (256, 256), color=(0, 0, 0)) # Elevation 0
    
    with patch('services.processors.elevation_color.ElevationTileProvider') as mock_provider_cls:
        mock_provider = mock_provider_cls.return_value
        mock_provider.get_tile_image = AsyncMock(side_effect=lambda *args: Image.new('RGB', (256, 256), color=(0, 0, 0)))
        
        # We need to make sure decode_terrain_rgb_to_elevation_m returns something
        with patch('elevation.stats.decode_terrain_rgb_to_elevation_m') as mock_decode:
            mock_decode.return_value = np.zeros((256, 256), dtype=np.float32)
            
            result = await process_elevation_color(ctx)
            
            assert isinstance(result, Image.Image)
            assert result.size == (256, 256)
            assert ctx.elev_min_m is not None
            assert ctx.elev_max_m is not None

@pytest.mark.asyncio
async def test_process_elevation_color_error_handling():
    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=100.0,
        height_m=100.0,
        api_key="test_key",
        output_path="test.png",
        max_zoom=10,
        settings=None
    )
    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    
    with patch('services.processors.elevation_color.ElevationTileProvider') as mock_provider_cls:
        mock_provider = mock_provider_cls.return_value
        # Fail Pass A
        mock_provider.get_tile_image = AsyncMock(side_effect=Exception("Failed to fetch"))
        
        with pytest.raises(Exception, match="Failed to fetch"):
            await process_elevation_color(ctx)

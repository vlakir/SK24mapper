import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from PIL import Image
from services.map_context import MapDownloadContext
from services.processors.xyz_tiles import process_xyz_tiles

@pytest.mark.asyncio
async def test_process_xyz_tiles_basic():
    # Mocking context
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
    ctx.tiles = [(10, 20), (11, 20)]
    ctx.tiles_x = 2
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 512, 256)
    ctx.style_id = "satellite"
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()

    # Mocking async_fetch_xyz_tile
    def get_mock_img(*args, **kwargs):
        return Image.new('RGB', (256, 256), color='red')
    
    with patch('services.processors.xyz_tiles.async_fetch_xyz_tile', side_effect=get_mock_img) as mock_fetch:
        
        result = await process_xyz_tiles(ctx)
        
        assert isinstance(result, Image.Image)
        assert result.size == (512, 256)
        assert mock_fetch.call_count == 2
        
        # Verify call arguments
        args, kwargs = mock_fetch.call_args
        assert kwargs['z'] == 10
        assert kwargs['api_key'] == "test_key"
        assert kwargs['style_id'] == "satellite"

@pytest.mark.asyncio
async def test_process_xyz_tiles_empty():
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
    ctx.tiles = []
    ctx.tiles_x = 0
    ctx.tiles_y = 0
    ctx.crop_rect = (0, 0, 0, 0)
    ctx.client = AsyncMock()
    
    # assemble_and_crop might fail with empty tiles, but let's see
    with patch('services.processors.xyz_tiles.assemble_and_crop') as mock_assemble:
        mock_assemble.return_value = Image.new('RGB', (1, 1))
        result = await process_xyz_tiles(ctx)
        assert result.size == (1, 1)

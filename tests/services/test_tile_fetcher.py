import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from PIL import Image
from services.tile_fetcher import fetch_xyz_tiles_batch

@pytest.mark.asyncio
async def test_fetch_xyz_tiles_batch_basic():
    client = AsyncMock()
    tiles = [(1, 2), (3, 4)]
    api_key = "test_key"
    style_id = "test_style"
    zoom = 10
    tile_size = 256
    use_retina = False
    semaphore = asyncio.Semaphore(1)
    
    mock_progress = AsyncMock()
    
    with patch('services.tile_fetcher.async_fetch_xyz_tile', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = Image.new('RGB', (256, 256))
        
        results = await fetch_xyz_tiles_batch(
            client, tiles, api_key=api_key, style_id=style_id,
            zoom=zoom, tile_size=tile_size, use_retina=use_retina,
            semaphore=semaphore, on_progress=mock_progress
        )
        
        assert len(results) == 2
        assert results[0][0] == 0
        assert results[1][0] == 1
        assert isinstance(results[0][1], Image.Image)
        assert mock_fetch.call_count == 2
        assert mock_progress.call_count == 2

@pytest.mark.asyncio
async def test_fetch_xyz_tiles_batch_empty():
    client = AsyncMock()
    results = await fetch_xyz_tiles_batch(
        client, [], api_key="k", style_id="s",
        zoom=10, tile_size=256, use_retina=False,
        semaphore=asyncio.Semaphore(1)
    )
    assert results == []

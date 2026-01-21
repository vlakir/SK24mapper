
import pytest
import asyncio
from pathlib import Path
from PIL import Image
from io import BytesIO
from unittest.mock import MagicMock, AsyncMock, patch
from elevation_provider import ElevationTileProvider, TileKey

@pytest.fixture
def mock_client():
    return AsyncMock()

@pytest.fixture
def temp_cache_dir(tmp_path):
    return tmp_path / "cache"

class TestElevationTileProvider:
    @pytest.mark.asyncio
    async def test_get_tile_image_cached_mem(self, mock_client, temp_cache_dir):
        provider = ElevationTileProvider(mock_client, "api_key", use_retina=False, cache_root=temp_cache_dir)
        key = provider._key(10, 500, 500)
        
        # Manually seed memory cache
        img = Image.new('RGB', (256, 256), color=(1, 2, 3))
        buf = BytesIO()
        img.save(buf, format='PNG')
        data = buf.getvalue()
        provider._remember_raw(key, data)
        
        result_img = await provider.get_tile_image(10, 500, 500)
        assert result_img.size == (256, 256)
        assert result_img.getpixel((0, 0)) == (1, 2, 3)
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_tile_image_cached_disk(self, mock_client, temp_cache_dir):
        provider = ElevationTileProvider(mock_client, "api_key", use_retina=False, cache_root=temp_cache_dir)
        key = provider._key(10, 500, 500)
        
        # Manually seed disk cache
        img = Image.new('RGB', (256, 256), color=(4, 5, 6))
        buf = BytesIO()
        img.save(buf, format='PNG')
        data = buf.getvalue()
        
        disk_path = provider._disk_path(key)
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_bytes(data)
        
        result_img = await provider.get_tile_image(10, 500, 500)
        assert result_img.getpixel((0, 0)) == (4, 5, 6)
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_tile_dem(self, mock_client, temp_cache_dir):
        provider = ElevationTileProvider(mock_client, "api_key", use_retina=False, cache_root=temp_cache_dir)
        
        # Seed memory cache with a known image
        # R=1, G=2, B=3 -> elevation approx -3394.9
        img = Image.new('RGB', (2, 2), color=(1, 2, 3))
        buf = BytesIO()
        img.save(buf, format='PNG')
        provider._remember_raw(provider._key(10, 500, 500), buf.getvalue())
        
        dem = await provider.get_tile_dem(10, 500, 500)
        assert len(dem) == 2
        assert dem[0][0] == pytest.approx(-3394.9, rel=1e-3)
        
        # Second call should use dem cache
        dem2 = await provider.get_tile_dem(10, 500, 500)
        assert dem2 is dem

    @pytest.mark.asyncio
    async def test_fetch_raw_network(self, mock_client, temp_cache_dir):
        provider = ElevationTileProvider(mock_client, "api_key", use_retina=False, cache_root=temp_cache_dir)
        
        mock_img = Image.new('RGB', (256, 256), color=(10, 20, 30))
        
        with patch('elevation_provider.async_fetch_terrain_rgb_tile', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_img
            
            raw = await provider._fetch_raw(provider._key(10, 1, 1))
            assert len(raw) > 0
            
            # Verify it's saved to disk
            assert provider._disk_path(provider._key(10, 1, 1)).exists()

    @pytest.mark.asyncio
    async def test_lru_eviction(self, mock_client, temp_cache_dir):
        # max_mem_tiles is at least 16 in __init__
        provider = ElevationTileProvider(mock_client, "api_key", use_retina=False, cache_root=temp_cache_dir, max_mem_tiles=2)
        
        keys = [provider._key(i, i, i) for i in range(20)]
        
        for i, k in enumerate(keys):
            provider._remember_raw(k, b"data")
            
        # First key should be evicted as we added 20 and max is 16
        assert keys[0] not in provider._mem_raw
        assert keys[19] in provider._mem_raw


import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
from tiles.coverage import compute_coverage, iter_overlapping_tiles
from tiles.fetcher import TileFetcher, stream_tiles

class TestTiles:
    @pytest.mark.asyncio
    async def test_tile_fetcher(self):
        mock_get = AsyncMock()
        mock_get.return_value = Image.new('RGB', (10, 10))
        
        fetcher = TileFetcher(mock_get, concurrency=2)
        tiles = [(0, (100, 200)), (1, (101, 200))]
        
        res = await fetcher.fetch_many(tiles, zoom=10)
        assert len(res) == 2
        assert (10, 100, 200) in res
        assert (10, 101, 200) in res

    @pytest.mark.asyncio
    async def test_stream_tiles(self):
        mock_get = AsyncMock()
        mock_get.return_value = Image.new('RGB', (10, 10))
        
        tiles = [(0, (100, 200)), (1, (101, 200))]
        count = 0
        async for key, img in stream_tiles(tiles, mock_get, zoom=10):
            count += 1
            assert img.size == (10, 10)
        assert count == 2

    def test_compute_coverage(self):
        # mock topography.compute_xyz_coverage
        with patch('tiles.coverage._compute_xyz_coverage') as mock_comp:
            mock_comp.return_value = ([], (0, 0), (0, 0, 0, 0), {})
            res = compute_coverage(0, 0, 0, 0, 10, 1, 0)
            assert len(res) == 4

    def test_iter_overlapping_tiles(self):
        # tiles: list[tuple[int, tuple[int, int]]]
        # (idx, (xw, yw))
        tiles = [
            (0, (100, 100)), # tx=0, ty=0
            (1, (101, 100)), # tx=1, ty=0
        ]
        tiles_x = 2
        crop_rect = (0, 0, 100, 100) # Only first tile overlaps
        tile_px = 256
        
        res = list(iter_overlapping_tiles(tiles, tiles_x, crop_rect, tile_px=tile_px))
        assert len(res) == 1
        assert res[0][0] == 0

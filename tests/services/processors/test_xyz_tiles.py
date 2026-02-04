import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image
from services.map_context import MapDownloadContext
from services.processors.xyz_tiles import process_xyz_tiles
from imaging.streaming import StreamingImage


@pytest.fixture
def mock_tile_fetcher():
    """Create a mock TileFetcher that returns red tiles."""
    fetcher = MagicMock()

    async def mock_fetch_xyz(*args, **kwargs):
        return Image.new('RGB', (256, 256), color='red')

    fetcher.fetch_xyz = AsyncMock(side_effect=mock_fetch_xyz)
    return fetcher


@pytest.mark.asyncio
async def test_process_xyz_tiles_basic(mock_tile_fetcher):
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
    ctx.tile_fetcher = mock_tile_fetcher

    # Mock constants to use eff_tile_px = 256 (XYZ_TILE_SIZE=256, XYZ_USE_RETINA=False)
    with patch('services.processors.xyz_tiles.XYZ_TILE_SIZE', 256), \
         patch('services.processors.xyz_tiles.XYZ_USE_RETINA', False):

        result = await process_xyz_tiles(ctx)

        assert isinstance(result, StreamingImage)
        assert result.size == (512, 256)
        assert mock_tile_fetcher.fetch_xyz.call_count == 2

        # Verify call arguments
        call_args = mock_tile_fetcher.fetch_xyz.call_args_list[0]
        kwargs = call_args.kwargs
        assert kwargs['z'] == 10
        assert kwargs['style_id'] == "satellite"
        result.close()


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
    ctx.crop_rect = (0, 0, 1, 1)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()
    ctx.tile_fetcher = MagicMock()

    # With empty tiles, assemble_tiles_streaming creates an empty image
    result = await process_xyz_tiles(ctx)
    assert isinstance(result, StreamingImage)
    assert result.size == (1, 1)
    result.close()

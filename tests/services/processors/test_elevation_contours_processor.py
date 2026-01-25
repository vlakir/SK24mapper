import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from PIL import Image
import numpy as np
from services.map_context import MapDownloadContext
from services.processors.elevation_contours import process_elevation_contours

@pytest.mark.asyncio
async def test_process_elevation_contours_basic():
    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=100.0,
        height_m=100.0,
        api_key="test_key",
        output_path="test.png",
        max_zoom=10,
        settings=AsyncMock()
    )
    ctx.settings.contour_interval_m = 10.0
    ctx.settings.contour_label_font_size = 12
    ctx.settings.contour_label_font_color = (0, 0, 0)
    ctx.settings.contour_line_color = (0, 0, 0)
    ctx.settings.contour_line_width = 1
    ctx.settings.contour_use_subpixel = True
    ctx.settings.contour_min_length_px = 10
    ctx.settings.contour_smooth_sigma = 0.5
    ctx.settings.contour_label_step_px = 500
    
    ctx.tiles = [(10, 20)]
    ctx.tiles_x = 1
    ctx.tiles_y = 1
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.client = AsyncMock()

    with patch('services.processors.elevation_contours.ElevationTileProvider') as mock_provider_cls, \
         patch('services.processors.elevation_contours.build_seed_polylines') as mock_build_seeds, \
         patch('geo.topography.assemble_dem') as mock_assemble:
        
        mock_provider = mock_provider_cls.return_value
        mock_provider.get_tile_image = AsyncMock(return_value=Image.new('RGB', (256, 256)))
        
        mock_assemble.return_value = np.zeros((256, 256), dtype=np.float32)
        mock_build_seeds.return_value = {0: []}
        
        # We need to mock more things inside the processor because it's very long
        # But even this should cover the beginning
        mock_img = Image.new('RGBA', (256, 256))
        with patch('services.processors.elevation_contours.Image.new', return_value=mock_img):
            result = await process_elevation_contours(ctx)
            assert isinstance(result, Image.Image)

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
    ctx.center_lat_wgs = 55.0

    with patch('services.processors.elevation_contours.ElevationTileProvider') as mock_provider_cls, \
         patch('services.processors.elevation_contours.build_seed_polylines') as mock_build_seeds, \
         patch('services.processors.elevation_contours.decode_terrain_rgb_to_elevation_m') as mock_decode, \
         patch('services.processors.elevation_contours.ConsoleProgress') as mock_progress, \
         patch('services.processors.elevation_contours.LiveSpinner') as mock_spinner, \
         patch('services.processors.elevation_contours.compute_contour_adaptive_params') as mock_compute_params:
        
        mock_provider = mock_provider_cls.return_value
        mock_provider.get_tile_image = AsyncMock(return_value=Image.new('RGB', (256, 256)))
        
        mock_decode.return_value = np.zeros((256, 256), dtype=np.float32)
        mock_build_seeds.return_value = {0: [[(10.0, 10.0), (20.0, 20.0)]]}
        
        from contours.adaptive import ContourAdaptiveParams
        mock_compute_params.return_value = ContourAdaptiveParams(
            interval_m=10.0, 
            scale=1.0,
            label_spacing_m=100.0,
            label_min_seg_len_m=50.0,
            label_edge_margin_m=10.0,
            label_font_m=12.0,
            label_gap_padding_m=5.0
        )
        
        result = await process_elevation_contours(ctx)
        assert isinstance(result, Image.Image)


@pytest.mark.asyncio
async def test_process_elevation_contours_no_tiles():
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
    ctx.tiles = []
    ctx.tiles_x = 0
    ctx.tiles_y = 0
    ctx.zoom = 10
    ctx.crop_rect = (0, 0, 256, 256)
    ctx.semaphore = asyncio.Semaphore(1)
    ctx.center_lat_wgs = 55.0
    
    with patch('services.processors.elevation_contours.ElevationTileProvider'), \
         patch('services.processors.elevation_contours.build_seed_polylines', return_value={}), \
         patch('services.processors.elevation_contours.ConsoleProgress'), \
         patch('services.processors.elevation_contours.LiveSpinner'):
        result = await process_elevation_contours(ctx)
        assert isinstance(result, Image.Image)


@patch('services.processors.elevation_contours.draw_contour_labels')
def test_add_contour_labels(mock_draw):
    from services.processors.elevation_contours import _add_contour_labels
    img = Image.new('RGB', (100, 100))
    seed_polylines = {0: []}
    levels = [100.0]
    crop_rect = (0, 0, 100, 100)
    
    mock_draw.return_value = []
    
    result = _add_contour_labels(img, seed_polylines, levels, crop_rect, 1, 55.0, 10)
    assert isinstance(result, Image.Image)
    assert mock_draw.called

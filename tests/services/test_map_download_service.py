"""Tests for services.map_download_service module."""

import sys
import asyncio
from types import SimpleNamespace

import pytest
from PIL import Image
from unittest.mock import patch

import services.map_download_service as map_download_service
from services.map_context import MapDownloadContext
from services.map_download_service import MapDownloadService
from shared.constants import MapType


@pytest.mark.asyncio
async def test_determine_map_type_xyz(monkeypatch):
    """XYZ map types should validate style API and return style id."""
    calls = {}

    async def fake_validate(api_key, style_id):
        calls["style"] = (api_key, style_id)

    async def fake_validate_terrain(_api_key):
        calls["terrain"] = True

    monkeypatch.setattr(map_download_service, "_validate_api_and_connectivity", fake_validate)
    monkeypatch.setattr(map_download_service, "_validate_terrain_api", fake_validate_terrain)

    service = MapDownloadService("token")
    settings = SimpleNamespace(control_point_enabled=True, antenna_height_m=10)

    style_id, is_color, is_contours, is_radio = await service._determine_map_type(
        MapType.STREETS, settings
    )

    assert style_id is not None
    assert (is_color, is_contours, is_radio) == (False, False, False)
    assert calls["style"][0] == "token"
    assert "terrain" not in calls


@pytest.mark.asyncio
async def test_determine_map_type_elevation(monkeypatch):
    """Elevation maps should validate Terrain-RGB API."""
    calls = []

    async def fake_validate(_api_key):
        calls.append("terrain")

    monkeypatch.setattr(map_download_service, "_validate_terrain_api", fake_validate)
    monkeypatch.setattr(
        map_download_service,
        "_validate_api_and_connectivity",
        lambda *_args, **_kwargs: None,
    )

    service = MapDownloadService("token")
    settings = SimpleNamespace(control_point_enabled=True, antenna_height_m=10)

    style_id, is_color, is_contours, is_radio = await service._determine_map_type(
        MapType.ELEVATION_COLOR, settings
    )

    assert style_id is None
    assert (is_color, is_contours, is_radio) == (True, False, False)
    assert calls == ["terrain"]


@pytest.mark.asyncio
async def test_determine_map_type_radio_requires_control_point(monkeypatch):
    """Radio horizon maps require control point enabled."""
    service = MapDownloadService("token")
    settings = SimpleNamespace(control_point_enabled=False, antenna_height_m=10)

    with pytest.raises(ValueError):
        await service._determine_map_type(MapType.RADIO_HORIZON, settings)


@pytest.mark.asyncio
async def test_run_processor_branches(monkeypatch):
    """_run_processor should choose correct processor module."""
    async def fake_return(_ctx):
        return Image.new("RGB", (1, 1))

    sys.modules["services.processors.xyz_tiles"] = SimpleNamespace(
        process_xyz_tiles=fake_return
    )
    sys.modules["services.processors.elevation_color"] = SimpleNamespace(
        process_elevation_color=fake_return
    )
    sys.modules["services.processors.elevation_contours"] = SimpleNamespace(
        process_elevation_contours=fake_return
    )
    sys.modules["services.processors.radio_horizon"] = SimpleNamespace(
        process_radio_horizon=fake_return
    )

    service = MapDownloadService("token")
    ctx = SimpleNamespace(is_elev_color=False, is_elev_contours=False, is_radio_horizon=False)

    result = await service._run_processor(ctx)
    assert isinstance(result, Image.Image)

    ctx.is_elev_color = True
    assert await service._run_processor(ctx)

    ctx.is_elev_color = False
    ctx.is_elev_contours = True
    assert await service._run_processor(ctx)

    ctx.is_elev_contours = False
    ctx.is_radio_horizon = True
    assert await service._run_processor(ctx)


@pytest.mark.asyncio
async def test_postprocess_flow(monkeypatch):
    """_postprocess should run overlay, rotation, crop, and draw steps."""
    from imaging.streaming import StreamingImage
    
    class DummySpinner:
        def __init__(self, _label):
            pass

        def start(self):
            return None

        def stop(self, _msg):
            return None

    monkeypatch.setattr(map_download_service, "LiveSpinner", DummySpinner)
    monkeypatch.setattr(map_download_service, "log_memory_usage", lambda *_args, **_kwargs: None)
    
    # Mock rotate_streaming and crop_streaming to return the same image
    def fake_rotate(src, angle, fill=(255,255,255), temp_dir=None):
        return src
    
    def fake_crop(src, w, h, temp_dir=None):
        return src
    
    monkeypatch.setattr(map_download_service, "rotate_streaming", fake_rotate)
    monkeypatch.setattr(map_download_service, "crop_streaming", fake_crop)

    overlay_calls = []

    async def fake_overlay(_self, _ctx, img):
        overlay_calls.append(True)
        return img

    monkeypatch.setattr(MapDownloadService, "_apply_overlay_contours", fake_overlay)
    monkeypatch.setattr(MapDownloadService, "_draw_overlays", lambda self, ctx, result: result)

    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=1.0,
        height_m=1.0,
        api_key="token",
        output_path="out.png",
        max_zoom=1,
        settings=SimpleNamespace(control_point_enabled=True, display_grid=True, max_flight_height_m=100.0, map_type=MapType.STREETS),
    )
    # Create a StreamingImage instead of PIL.Image
    ctx.result = StreamingImage(10, 10)
    ctx.overlay_contours = True
    ctx.is_elev_contours = False
    ctx.is_elev_color = True
    ctx.is_radio_horizon = False
    ctx.rotation_deg = 0.0
    ctx.target_w_px = 10
    ctx.target_h_px = 10

    service = MapDownloadService("token")
    await service._postprocess(ctx)

    assert overlay_calls == [True]
    assert ctx.result.size == (10, 10)
    ctx.result.close()


@pytest.mark.asyncio
async def test_draw_methods(monkeypatch, tmp_path):
    """Test various draw methods of MapDownloadService (streaming versions)."""
    from imaging.streaming import StreamingImage
    from imaging.grid import GridElements

    service = MapDownloadService("token")
    ctx = MapDownloadContext(
        center_x_sk42_gk=0, center_y_sk42_gk=0, width_m=100, height_m=100,
        api_key="token", output_path="out.png", max_zoom=10,
        settings=SimpleNamespace(
            display_grid=True, grid_step_m=100,
            control_point_enabled=True,
            control_point_x_sk42_gk=0.0, control_point_y_sk42_gk=0.0,
            control_point_name="Test", display_center_cross=True,
            legend_enabled=True, antenna_height_m=10.0, max_flight_height_m=100.0,
            map_type=MapType.STREETS,
            grid_font_size_m=100.0, grid_label_bg_padding_m=10.0,
            uav_height_reference="AGL",
        )
    )
    # Use StreamingImage instead of PIL.Image for streaming methods
    ctx.result = StreamingImage(100, 100, temp_dir=str(tmp_path))
    ctx.mpp_eff = 1.0
    ctx.crop_rect = (0, 0, 100, 100)
    ctx.center_lat_wgs = 55.0
    ctx.zoom = 10
    ctx.eff_scale = 1
    ctx.rotation_deg = 0.0
    ctx.image_was_rotated = False
    ctx.is_elev_color = False
    ctx.is_radio_horizon = False
    ctx.coord_result = SimpleNamespace(
        center_lat_sk42=55.0, center_lng_sk42=37.0, crs_sk42_gk=None
    )

    # Mock the grid elements computation to return empty grid
    mock_grid_elements = GridElements()
    monkeypatch.setattr(
        "services.map_download_service.compute_km_grid_elements",
        lambda *args, **kwargs: mock_grid_elements
    )
    # Mock streaming drawing functions
    monkeypatch.setattr(
        "services.map_download_service.draw_polylines_streaming",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "services.map_download_service.draw_labels_streaming",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "services.map_download_service.draw_cross_streaming",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "services.map_download_service.draw_control_point_streaming",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "services.map_download_service.draw_elevation_legend",
        lambda *args, **kwargs: None
    )

    with patch("services.map_download_service.CoordinateTransformer") as mock_transformer_cls:
        mock_transformer = mock_transformer_cls.return_value
        mock_transformer.lat_lon_to_sk42_gk.return_value = (10, 10)
        mock_transformer.sk42_gk_to_image_px.return_value = (50, 50)

        # Test streaming methods
        service._draw_grid_streaming(ctx, ctx.result)
        service._draw_center_cross_streaming(ctx, ctx.result)

    assert ctx.result is not None
    ctx.result.close()


@pytest.mark.asyncio
async def test_cleanup_session(tmp_path):
    service = MapDownloadService("token")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    class FakeSession:
        async def close(self):
            pass
            
    # Mock _cleanup_sqlite_cache as it's the one that is called
    with patch('services.map_download_service._cleanup_sqlite_cache') as mock_cleanup:
        # We need to make sure Path(cache_dir).resolve().exists() is True
        # Path is imported in map_download_service
        service._cleanup_session(FakeSession(), str(cache_dir))
        assert mock_cleanup.called


@pytest.mark.asyncio
async def test_save(tmp_path):
    from imaging.streaming import StreamingImage
    
    service = MapDownloadService("token")
    out_path = tmp_path / "out.jpg"
    ctx = MapDownloadContext(
        center_x_sk42_gk=0, center_y_sk42_gk=0, width_m=100, height_m=100,
        api_key="token", output_path=str(out_path), max_zoom=10,
        settings=SimpleNamespace()
    )
    # Use StreamingImage instead of PIL.Image
    ctx.result = StreamingImage(10, 10)
    
    with patch("services.map_download_service.LiveSpinner"), \
         patch("services.map_download_service.publish_preview_image", return_value=False), \
         patch("services.map_download_service.save_streaming_image", return_value=str(out_path)) as mock_save:
        await service._save(ctx)
        assert mock_save.called
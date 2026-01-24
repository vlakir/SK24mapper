"""Tests for services.map_download_service module."""

import sys
from types import SimpleNamespace

import pytest
from PIL import Image

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
    class DummySpinner:
        def __init__(self, _label):
            pass

        def start(self):
            return None

        def stop(self, _msg):
            return None

    monkeypatch.setattr(map_download_service, "LiveSpinner", DummySpinner)
    monkeypatch.setattr(map_download_service, "rotate_keep_size", lambda img, *_args, **_kwargs: img)
    monkeypatch.setattr(map_download_service, "center_crop", lambda img, *_args, **_kwargs: img)
    monkeypatch.setattr(map_download_service, "log_memory_usage", lambda *_args, **_kwargs: None)

    overlay_calls = []

    async def fake_overlay(_self, _ctx, img):
        overlay_calls.append(True)
        return img

    monkeypatch.setattr(MapDownloadService, "_apply_overlay_contours", fake_overlay)
    monkeypatch.setattr(MapDownloadService, "_draw_grid", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MapDownloadService, "_draw_legend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MapDownloadService, "_draw_center_cross", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MapDownloadService, "_draw_control_point", lambda *_args, **_kwargs: None)

    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=1.0,
        height_m=1.0,
        api_key="token",
        output_path="out.png",
        max_zoom=1,
        settings=SimpleNamespace(control_point_enabled=True, display_grid=True, max_flight_height_m=100.0),
    )
    ctx.result = Image.new("RGB", (10, 10), color=(255, 255, 255))
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
"""Tests for services.map_context module."""

from geo import topography
from services.map_context import MapDownloadContext


def test_map_context_pixel_helpers(monkeypatch):
    """MapDownloadContext should use meters_per_pixel helper."""
    monkeypatch.setattr(topography, "meters_per_pixel", lambda *_args, **_kwargs: 2.0)

    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=1.0,
        height_m=1.0,
        api_key="key",
        output_path="out.png",
        max_zoom=1,
        settings=object(),
    )
    ctx.center_lat_wgs = 10.0
    ctx.zoom = 5
    ctx.eff_scale = 2

    assert ctx.get_meters_per_pixel() == 2.0
    assert ctx.get_pixels_per_meter() == 0.5


def test_map_context_radar_coverage_flag():
    """MapDownloadContext should have is_radar_coverage flag defaulting to False."""
    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=1.0,
        height_m=1.0,
        api_key="key",
        output_path="out.png",
        max_zoom=1,
        settings=object(),
    )
    assert ctx.is_radar_coverage is False

    ctx.is_radar_coverage = True
    assert ctx.is_radar_coverage is True
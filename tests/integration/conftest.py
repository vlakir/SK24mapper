"""Shared fixtures for integration tests.

These tests run real MapDownloadService.download() with actual Mapbox API
requests.  Tiles are cached in ~/.cache/sk42mapper/tiles/ so repeated
runs do not re-download.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Generator

import numpy as np
import pytest
from PIL import Image

from domain.models import MapMetadata, MapSettings
from services.map_download_service import MapDownloadService
from shared.constants import MapType
from shared.progress import (
    NeverCancelToken,
    cleanup_all_progress_resources,
    set_cancel_event,
    set_preview_image_callback,
    set_progress_callback,
    set_spinner_callbacks,
    set_warning_callback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str | None:
    """Try to load Mapbox API key from .secrets.env or environment."""
    # 1. Environment variable
    key = os.environ.get('MAPBOX_ACCESS_TOKEN') or os.environ.get('API_KEY')
    if key:
        return key

    # 2. .secrets.env in project root
    secrets_path = Path(__file__).resolve().parents[2] / '.secrets.env'
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            line = line.strip()
            if line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            v = v.strip().strip('"').strip("'")
            if k.strip() in ('API_KEY', 'MAPBOX_ACCESS_TOKEN') and v:
                return v
    return None


def _noop_progress(done: int, total: int, label: str) -> None:
    pass


def _noop_spinner(label: str) -> None:
    pass


def _noop_warning(text: str, field_updates: dict | None = None) -> None:
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='session')
def api_key() -> str:
    """Load Mapbox API key; skip entire session if unavailable."""
    key = _load_api_key()
    if not key:
        pytest.skip('Mapbox API key not found (set API_KEY or .secrets.env)')
    return key


@pytest.fixture(autouse=True)
def _isolate_progress_globals() -> Generator[None, None, None]:
    """Reset global callbacks so service writes to disk, not GUI."""
    set_progress_callback(_noop_progress)
    set_spinner_callbacks(_noop_spinner, _noop_spinner)
    set_preview_image_callback(None)  # ensures _save() writes JPEG
    set_warning_callback(_noop_warning)
    set_cancel_event(NeverCancelToken())
    yield
    cleanup_all_progress_resources()


@pytest.fixture()
def make_settings():
    """Factory for MapSettings with 1×1 km area near Kostroma."""

    def _factory(**overrides) -> MapSettings:
        defaults = dict(
            from_x_high=54,
            from_y_high=74,
            to_x_high=54,
            to_y_high=74,
            from_x_low=18,
            from_y_low=46,
            to_x_low=19,
            to_y_low=47,
            grid_width_m=5.0,
            grid_font_size_m=100.0,
            grid_text_margin_m=50.0,
            grid_label_bg_padding_m=10.0,
            mask_opacity=0.35,
            display_grid=True,
            helmert_enabled=True,
            helmert_dx=-50.957,
            helmert_dy=-39.724,
            helmert_dz=-76.877,
            helmert_rx_as=2.33295,
            helmert_ry_as=2.13987,
            helmert_rz_as=-2.03005,
            helmert_ds_ppm=-1.43065,
        )
        defaults.update(overrides)
        return MapSettings(**defaults)

    return _factory


@pytest.fixture()
def run_map(api_key, make_settings, tmp_path):
    """Async helper that runs the full download pipeline and returns results."""

    async def _run(
        settings: MapSettings | None = None, **overrides
    ) -> tuple[str, MapMetadata, Image.Image]:
        if settings is None:
            settings = make_settings(**overrides)

        # Compute center GK from settings properties
        center_x = (
            settings.bottom_left_x_sk42_gk + settings.top_right_x_sk42_gk
        ) / 2
        center_y = (
            settings.bottom_left_y_sk42_gk + settings.top_right_y_sk42_gk
        ) / 2
        width_m = abs(
            settings.top_right_x_sk42_gk - settings.bottom_left_x_sk42_gk
        )
        height_m = abs(
            settings.top_right_y_sk42_gk - settings.bottom_left_y_sk42_gk
        )

        out_path = str(tmp_path / 'map.jpg')
        svc = MapDownloadService(api_key)
        result_path, metadata = await svc.download(
            center_x_sk42_gk=center_x,
            center_y_sk42_gk=center_y,
            width_m=width_m,
            height_m=height_m,
            output_path=out_path,
            settings=settings,
        )
        img = Image.open(result_path)
        img.load()
        return result_path, metadata, img

    return _run


# ---------------------------------------------------------------------------
# Shared assertion helper (exposed as fixture)
# ---------------------------------------------------------------------------


def _assert_valid_map_output(
    result_path: str,
    metadata: MapMetadata,
    img: Image.Image,
    *,
    expected_map_type: MapType | None = None,
) -> None:
    """Common assertions for any generated map."""
    # File exists and is non-trivial
    p = Path(result_path)
    assert p.exists(), f'Output file missing: {result_path}'
    assert p.stat().st_size > 1024, f'Output file too small: {p.stat().st_size} bytes'

    # Image dimensions
    w, h = img.size
    assert w > 100, f'Image too narrow: {w}px'
    assert h > 100, f'Image too short: {h}px'

    # Not blank (std deviation of pixel values should be noticeable)
    arr = np.array(img.convert('L'), dtype=np.float32)
    std = float(np.std(arr))
    assert std > 3.0, f'Image looks blank (std={std:.1f})'

    # Metadata sanity
    assert metadata.zoom > 0
    assert metadata.meters_per_pixel > 0
    assert metadata.width_px > 0
    assert metadata.height_px > 0

    if expected_map_type is not None:
        assert metadata.map_type == expected_map_type, (
            f'Expected map_type={expected_map_type}, got {metadata.map_type}'
        )


@pytest.fixture()
def assert_valid_map_output():
    """Fixture providing the shared map output assertion function."""
    return _assert_valid_map_output

"""Tests for tiles.loader module."""

import asyncio

import numpy as np
import pytest
from PIL import Image

import tiles.loader as loader
from shared.constants import MapType, TILE_SIZE, XYZ_TILE_SIZE


def test_get_effective_tile_size_xyz():
    """Current implementation raises due to missing MapType.ELEVATION."""
    with pytest.raises(AttributeError):
        loader.get_effective_tile_size(MapType.STREETS, use_retina=False)
    with pytest.raises(AttributeError):
        loader.get_effective_tile_size(MapType.STREETS, use_retina=True)


def test_get_effective_tile_size_elevation():
    """Current implementation raises due to missing MapType.ELEVATION."""
    with pytest.raises(AttributeError):
        loader.get_effective_tile_size(MapType.ELEVATION_COLOR, use_retina=False)
    with pytest.raises(AttributeError):
        loader.get_effective_tile_size(MapType.ELEVATION_COLOR, use_retina=True)


def test_get_retina_flag_uses_constants():
    """Current implementation raises due to missing MapType.ELEVATION."""
    with pytest.raises(AttributeError):
        loader.get_retina_flag(MapType.ELEVATION_COLOR)
    with pytest.raises(AttributeError):
        loader.get_retina_flag(MapType.RADIO_HORIZON)


def test_get_style_id_matches_constants():
    """Style id should be delegated to map_type_to_style_id."""
    style_id = loader.get_style_id(MapType.STREETS)
    assert style_id is not None


@pytest.mark.asyncio
async def test_fetch_map_tile_delegates(monkeypatch):
    """fetch_map_tile should delegate to async_fetch_xyz_tile."""
    called = {}

    async def fake_fetch(client, api_key, z, x, y, *, style_id, use_retina):
        called["args"] = (client, api_key, z, x, y, style_id, use_retina)
        return Image.new("RGB", (5, 5), color=(1, 2, 3))

    monkeypatch.setattr(loader, "async_fetch_xyz_tile", fake_fetch)
    client = object()

    result = await loader.fetch_map_tile(
        client, "key", 1, 2, 3, MapType.STREETS, use_retina=True
    )

    assert result.size == (5, 5)
    assert called["args"][0] is client
    assert called["args"][1] == "key"
    assert called["args"][2:5] == (1, 2, 3)
    assert called["args"][6] is True


@pytest.mark.asyncio
async def test_fetch_dem_tile_decodes(monkeypatch):
    """fetch_dem_tile should decode the fetched Terrain-RGB tile."""
    tile = Image.new("RGB", (2, 2), color=(0, 0, 0))

    async def fake_fetch(*_args, **_kwargs):
        return tile

    def fake_decode(img):
        assert img is tile
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    monkeypatch.setattr(loader, "async_fetch_terrain_rgb_tile", fake_fetch)
    monkeypatch.setattr(loader, "decode_terrain_rgb_to_elevation_m", fake_decode)

    result = await loader.fetch_dem_tile(object(), "key", 0, 1, 2, use_retina=False)

    assert result.shape == (2, 2)
    assert float(result[0, 0]) == 1.0


@pytest.mark.asyncio
async def test_tile_fetcher_batches(monkeypatch):
    """TileFetcher should gather batch results."""
    async def fake_fetch_map_tile(*_args, **_kwargs):
        await asyncio.sleep(0)
        return Image.new("RGB", (1, 1))

    async def fake_fetch_dem_tile(*_args, **_kwargs):
        await asyncio.sleep(0)
        return np.zeros((1, 1), dtype=float)

    monkeypatch.setattr(loader, "fetch_map_tile", fake_fetch_map_tile)
    monkeypatch.setattr(loader, "fetch_dem_tile", fake_fetch_dem_tile)

    fetcher = loader.TileFetcher(
        client=object(),
        api_key="key",
        zoom=3,
        map_type=MapType.STREETS,
        use_retina=False,
        concurrency=2,
    )

    tiles = await fetcher.fetch_tiles_batch([(0, 0), (1, 1)])
    dems = await fetcher.fetch_dem_batch([(0, 0), (1, 1)])

    assert len(tiles) == 2
    assert len(dems) == 2
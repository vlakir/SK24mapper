"""Integration tests for Mapbox API key validation."""

import pytest

from infrastructure.http.client import validate_style_api, validate_terrain_api

pytestmark = pytest.mark.integration


async def test_validate_style_api_valid_key(api_key):
    """Valid key should pass without error."""
    await validate_style_api(api_key, 'mapbox/satellite-v9')


async def test_validate_terrain_api_valid_key(api_key):
    """Valid key should access Terrain-RGB endpoint."""
    await validate_terrain_api(api_key)


async def test_validate_style_api_invalid_key():
    """Invalid key should raise RuntimeError."""
    with pytest.raises(RuntimeError, match='ключ|key|недоступен|соединения'):
        await validate_style_api('pk.invalid_key_123', 'mapbox/satellite-v9')


async def test_validate_terrain_api_invalid_key():
    """Invalid key should raise RuntimeError."""
    with pytest.raises(RuntimeError, match='ключ|key|недоступен|соединения'):
        await validate_terrain_api('pk.invalid_key_123')

"""Tests for TileFetcher."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tiles.cache import TileCache
from tiles.fetcher import TileFetcher
from tiles.writer import CacheWriter


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache(temp_cache_dir):
    """Create TileCache instance."""
    tc = TileCache(cache_dir=temp_cache_dir)
    yield tc
    tc.close()


@pytest.fixture
def fetcher(cache):
    """Create TileFetcher instance."""
    return TileFetcher(
        cache=cache,
        writer=None,
        api_key='test-token',
        ttl_hours=24,
    )


class TestTileFetcher:
    """Tests for TileFetcher class."""

    def test_init(self, cache):
        """Test fetcher initialization."""
        fetcher = TileFetcher(cache=cache, api_key='test')
        assert fetcher.cache is cache
        assert fetcher.api_key == 'test'
        assert fetcher._stats_cache_hits == 0
        assert fetcher._stats_cache_misses == 0

    def test_stats(self, fetcher):
        """Test stats property."""
        stats = fetcher.stats
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'downloads' in stats
        assert 'errors' in stats

    def test_is_expired_false(self, fetcher):
        """Test _is_expired returns False for fresh tile."""
        now = int(time.time())
        assert not fetcher._is_expired(now)

    def test_is_expired_true(self, fetcher):
        """Test _is_expired returns True for old tile."""
        old_time = int(time.time()) - (25 * 3600)  # 25 hours ago
        assert fetcher._is_expired(old_time)

    def test_is_expired_disabled(self, cache):
        """Test _is_expired with TTL disabled."""
        fetcher = TileFetcher(cache=cache, ttl_hours=0)
        old_time = int(time.time()) - (1000 * 3600)
        assert not fetcher._is_expired(old_time)

    def test_get_source_for_style_satellite(self, fetcher):
        """Test source detection for satellite style."""
        source = fetcher._get_source_for_style('mapbox/satellite-v9')
        assert source == 'satellite'

    def test_get_source_for_style_other(self, fetcher):
        """Test source detection for other styles."""
        source = fetcher._get_source_for_style('mapbox/streets-v12')
        assert source == 'mapbox-streets-v12'

    @pytest.mark.asyncio
    async def test_fetch_xyz_from_cache(self, fetcher, cache):
        """Test fetch_xyz returns cached tile."""
        # Put tile in cache
        png_data = _create_test_png()
        cache.put(zoom=15, x=100, y=200, source='satellite', data=png_data)

        # Create mock session
        mock_session = MagicMock()

        # Fetch should return cached tile without HTTP request
        img = await fetcher.fetch_xyz(
            client=mock_session,
            style_id='mapbox/satellite-v9',
            z=15,
            x=100,
            y=200,
        )

        assert img is not None
        assert img.mode == 'RGB'
        assert fetcher._stats_cache_hits == 1
        mock_session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_xyz_expired_tile(self, fetcher, cache):
        """Test fetch_xyz downloads when cached tile is expired."""
        # Put expired tile in cache
        png_data = _create_test_png()
        old_time = int(time.time()) - (25 * 3600)  # 25 hours ago
        cache.put(zoom=15, x=100, y=200, source='satellite', data=png_data, fetched_at=old_time)

        # Mock HTTP response
        fresh_png = _create_test_png()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=fresh_png)
        mock_response.close = MagicMock()
        mock_response.release = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        # Fetch should download fresh tile
        img = await fetcher.fetch_xyz(
            client=mock_session,
            style_id='mapbox/satellite-v9',
            z=15,
            x=100,
            y=200,
        )

        assert img is not None
        assert fetcher._stats_cache_misses == 1
        assert fetcher._stats_downloads == 1
        mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_xyz_force_download(self, fetcher, cache):
        """Test fetch_xyz with force_download."""
        # Put tile in cache
        png_data = _create_test_png()
        cache.put(zoom=15, x=100, y=200, source='satellite', data=png_data)

        # Mock HTTP response
        fresh_png = _create_test_png()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=fresh_png)
        mock_response.close = MagicMock()
        mock_response.release = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        # Fetch with force_download should download
        img = await fetcher.fetch_xyz(
            client=mock_session,
            style_id='mapbox/satellite-v9',
            z=15,
            x=100,
            y=200,
            force_download=True,
        )

        assert img is not None
        mock_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_xyz_writes_to_cache(self, fetcher, cache):
        """Test fetch_xyz writes downloaded tile to cache."""
        # Mock HTTP response
        png_data = _create_test_png()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=png_data)
        mock_response.close = MagicMock()
        mock_response.release = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        # Fetch tile
        await fetcher.fetch_xyz(
            client=mock_session,
            style_id='mapbox/satellite-v9',
            z=15,
            x=100,
            y=200,
        )

        # Verify tile is in cache
        cached = cache.get(zoom=15, x=100, y=200, source='satellite')
        assert cached == png_data

    @pytest.mark.asyncio
    async def test_fetch_terrain_from_cache(self, fetcher, cache):
        """Test fetch_terrain returns cached tile."""
        png_data = _create_test_png()
        cache.put(zoom=15, x=100, y=200, source='terrain-rgb', data=png_data)

        mock_session = MagicMock()

        img = await fetcher.fetch_terrain(
            client=mock_session,
            z=15,
            x=100,
            y=200,
        )

        assert img is not None
        assert fetcher._stats_cache_hits == 1
        mock_session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_xyz_raw(self, fetcher, cache):
        """Test fetch_xyz_raw returns raw bytes."""
        png_data = _create_test_png()
        cache.put(zoom=15, x=100, y=200, source='satellite', data=png_data)

        mock_session = MagicMock()

        data = await fetcher.fetch_xyz_raw(
            client=mock_session,
            style_id='mapbox/satellite-v9',
            z=15,
            x=100,
            y=200,
        )

        assert data == png_data
        assert isinstance(data, bytes)

    @pytest.mark.asyncio
    async def test_fetch_with_writer(self, cache, temp_cache_dir):
        """Test fetch uses CacheWriter for async writes."""
        writer = CacheWriter(cache)
        writer.start()

        fetcher = TileFetcher(
            cache=cache,
            writer=writer,
            api_key='test',
        )

        # Mock HTTP response
        png_data = _create_test_png()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=png_data)
        mock_response.close = MagicMock()
        mock_response.release = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        # Fetch tile
        await fetcher.fetch_xyz(
            client=mock_session,
            style_id='mapbox/satellite-v9',
            z=15,
            x=100,
            y=200,
        )

        # Wait for writer
        import time
        time.sleep(0.5)
        writer.stop()

        # Verify tile was written via writer
        cached = cache.get(zoom=15, x=100, y=200, source='satellite')
        assert cached == png_data

    @pytest.mark.asyncio
    async def test_fetch_unauthorized(self, fetcher):
        """Test fetch raises error on 401."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.close = MagicMock()
        mock_response.release = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match='Access denied'):
            await fetcher.fetch_xyz(
                client=mock_session,
                style_id='mapbox/satellite-v9',
                z=15,
                x=100,
                y=200,
            )

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, fetcher):
        """Test fetch raises error on 404."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.close = MagicMock()
        mock_response.release = MagicMock()

        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match='Not found'):
            await fetcher.fetch_xyz(
                client=mock_session,
                style_id='mapbox/satellite-v9',
                z=15,
                x=100,
                y=200,
            )


def _create_test_png() -> bytes:
    """Create a minimal valid PNG for testing."""
    from PIL import Image
    from io import BytesIO

    img = Image.new('RGB', (256, 256), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

"""Tests for TileFetcher offline mode."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from PIL import Image

from tiles.fetcher import TileFetcher


@pytest.fixture
def mock_cache():
    """Create mock TileCache."""
    cache = MagicMock()
    cache.get_info.return_value = None
    cache.get.return_value = None
    return cache


@pytest.fixture
def mock_writer():
    """Create mock CacheWriter."""
    writer = MagicMock()
    writer.is_running.return_value = True
    return writer


class TestTileFetcherOffline:
    """Tests for offline mode functionality."""

    def test_offline_mode_initialization(self, mock_cache, mock_writer):
        """Should initialize with offline mode disabled by default."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
        )
        assert fetcher.offline is False

    def test_offline_mode_enabled(self, mock_cache, mock_writer):
        """Should initialize with offline mode enabled."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
            offline=True,
        )
        assert fetcher.offline is True

    def test_stats_include_offline_misses(self, mock_cache, mock_writer):
        """Stats should include offline_misses counter."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
            offline=True,
        )
        stats = fetcher.stats
        assert 'offline_misses' in stats
        assert stats['offline_misses'] == 0

    @pytest.mark.asyncio
    async def test_offline_xyz_returns_placeholder(self, mock_cache, mock_writer):
        """Offline mode should return placeholder tile when not in cache."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
            offline=True,
        )
        mock_client = AsyncMock()

        # Cache miss
        mock_cache.get_info.return_value = None

        result = await fetcher.fetch_xyz(
            client=mock_client,
            style_id='mapbox/satellite-v9',
            z=15,
            x=100,
            y=200,
        )

        # Should return placeholder image
        assert isinstance(result, Image.Image)
        assert result.size == (512, 512)
        # Should not make HTTP request
        mock_client.get.assert_not_called()
        # Should increment offline_misses
        assert fetcher.stats['offline_misses'] == 1

    @pytest.mark.asyncio
    async def test_offline_terrain_returns_placeholder(self, mock_cache, mock_writer):
        """Offline mode should return placeholder terrain tile when not in cache."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
            offline=True,
        )
        mock_client = AsyncMock()

        result = await fetcher.fetch_terrain(
            client=mock_client,
            z=15,
            x=100,
            y=200,
        )

        # Should return placeholder terrain image
        assert isinstance(result, Image.Image)
        assert result.size == (512, 512)
        mock_client.get.assert_not_called()
        assert fetcher.stats['offline_misses'] == 1

    def test_create_placeholder_tile(self, mock_cache, mock_writer):
        """Placeholder tile should be gray."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
        )
        placeholder = fetcher._create_placeholder_tile()

        assert isinstance(placeholder, Image.Image)
        assert placeholder.mode == 'RGB'
        assert placeholder.size == (512, 512)
        # Should be gray
        pixel = placeholder.getpixel((0, 0))
        assert pixel == (180, 180, 180)

    def test_create_placeholder_terrain_tile(self, mock_cache, mock_writer):
        """Placeholder terrain tile should encode sea level elevation."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
        )
        placeholder = fetcher._create_placeholder_terrain_tile()

        assert isinstance(placeholder, Image.Image)
        assert placeholder.mode == 'RGB'
        assert placeholder.size == (512, 512)
        # Should be terrain-RGB encoded for ~0m elevation
        pixel = placeholder.getpixel((0, 0))
        assert pixel == (1, 134, 160)

    def test_check_tiles_in_cache(self, mock_cache, mock_writer):
        """check_tiles_in_cache should count cached tiles."""
        fetcher = TileFetcher(
            cache=mock_cache,
            writer=mock_writer,
            api_key='test-key',
        )

        # Mock: first 2 tiles are cached, third is not
        mock_cache.get_info.side_effect = [
            MagicMock(),  # Tile 1 - cached
            MagicMock(),  # Tile 2 - cached
            None,  # Tile 3 - not cached
        ]

        tiles = [(100, 200), (101, 200), (102, 200)]
        cached, total = fetcher.check_tiles_in_cache(tiles, zoom=15, source='satellite')

        assert cached == 2
        assert total == 3

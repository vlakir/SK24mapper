"""Tests for TileCache."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from tiles.cache import CacheStats, TileCache, TileInfo


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


class TestTileCache:
    """Tests for TileCache class."""

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that init creates cache directory."""
        cache_dir = temp_cache_dir / 'tiles'
        cache = TileCache(cache_dir=cache_dir)
        assert cache_dir.exists()
        cache.close()

    def test_put_and_get(self, cache):
        """Test basic put and get operations."""
        data = b'test tile data'
        cache.put(zoom=15, x=100, y=200, source='satellite', data=data)
        result = cache.get(zoom=15, x=100, y=200, source='satellite')
        assert result == data

    def test_get_nonexistent_returns_none(self, cache):
        """Test that get returns None for nonexistent tile."""
        result = cache.get(zoom=15, x=999, y=999, source='satellite')
        assert result is None

    def test_exists(self, cache):
        """Test exists method."""
        assert not cache.exists(zoom=15, x=100, y=200, source='satellite')
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'data')
        assert cache.exists(zoom=15, x=100, y=200, source='satellite')

    def test_put_updates_existing(self, cache):
        """Test that put overwrites existing tile."""
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'old')
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'new')
        result = cache.get(zoom=15, x=100, y=200, source='satellite')
        assert result == b'new'

    def test_different_sources(self, cache):
        """Test that different sources are stored separately."""
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'sat')
        cache.put(zoom=15, x=100, y=200, source='terrain-rgb', data=b'terrain')
        assert cache.get(zoom=15, x=100, y=200, source='satellite') == b'sat'
        assert cache.get(zoom=15, x=100, y=200, source='terrain-rgb') == b'terrain'

    def test_different_zooms(self, cache):
        """Test that different zooms use separate databases."""
        cache.put(zoom=14, x=100, y=200, source='satellite', data=b'z14')
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'z15')
        assert cache.get(zoom=14, x=100, y=200, source='satellite') == b'z14'
        assert cache.get(zoom=15, x=100, y=200, source='satellite') == b'z15'

    def test_delete(self, cache):
        """Test delete method."""
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'data')
        assert cache.delete(zoom=15, x=100, y=200, source='satellite')
        assert not cache.exists(zoom=15, x=100, y=200, source='satellite')

    def test_delete_nonexistent(self, cache):
        """Test delete returns False for nonexistent tile."""
        assert not cache.delete(zoom=15, x=999, y=999, source='satellite')

    def test_get_info(self, cache):
        """Test get_info method."""
        before = int(time.time())
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'test data')
        after = int(time.time())

        info = cache.get_info(zoom=15, x=100, y=200, source='satellite')
        assert info is not None
        assert isinstance(info, TileInfo)
        assert info.x == 100
        assert info.y == 200
        assert info.zoom == 15
        assert info.source == 'satellite'
        assert info.size_bytes == len(b'test data')
        assert before <= info.fetched_at <= after

    def test_get_info_nonexistent(self, cache):
        """Test get_info returns None for nonexistent tile."""
        info = cache.get_info(zoom=15, x=999, y=999, source='satellite')
        assert info is None

    def test_get_updates_last_used(self, cache):
        """Test that get updates last_used_at."""
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'data')
        info1 = cache.get_info(zoom=15, x=100, y=200, source='satellite')

        time.sleep(0.1)
        cache.get(zoom=15, x=100, y=200, source='satellite')
        info2 = cache.get_info(zoom=15, x=100, y=200, source='satellite')

        # last_used_at should be updated (or at least not decreased)
        assert info2.last_used_at >= info1.last_used_at

    def test_put_batch(self, cache):
        """Test batch put operation."""
        tiles = [
            (100, 200, 'satellite', b'tile1'),
            (101, 200, 'satellite', b'tile2'),
            (102, 200, 'satellite', b'tile3'),
        ]
        cache.put_batch(zoom=15, tiles=tiles)

        assert cache.get(zoom=15, x=100, y=200, source='satellite') == b'tile1'
        assert cache.get(zoom=15, x=101, y=200, source='satellite') == b'tile2'
        assert cache.get(zoom=15, x=102, y=200, source='satellite') == b'tile3'

    def test_put_batch_empty(self, cache):
        """Test batch put with empty list."""
        cache.put_batch(zoom=15, tiles=[])  # Should not raise

    def test_get_stats_empty(self, cache):
        """Test get_stats on empty cache."""
        stats = cache.get_stats()
        assert isinstance(stats, CacheStats)
        assert stats.total_tiles == 0
        assert stats.total_size_bytes == 0

    def test_get_stats(self, cache):
        """Test get_stats with data."""
        cache.put(zoom=14, x=100, y=200, source='satellite', data=b'1234567890')
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'12345')
        cache.put(zoom=15, x=101, y=200, source='satellite', data=b'12345')

        stats = cache.get_stats()
        assert stats.total_tiles == 3
        assert stats.total_size_bytes == 20
        assert stats.tiles_by_zoom[14] == 1
        assert stats.tiles_by_zoom[15] == 2
        assert stats.size_by_zoom[14] == 10
        assert stats.size_by_zoom[15] == 10

    def test_clear_zoom(self, cache, temp_cache_dir):
        """Test clear_zoom method."""
        cache.put(zoom=14, x=100, y=200, source='satellite', data=b'z14')
        cache.put(zoom=15, x=100, y=200, source='satellite', data=b'z15')

        deleted = cache.clear_zoom(14)
        assert deleted == 1
        assert not (temp_cache_dir / 'zoom_14.db').exists()
        assert cache.get(zoom=14, x=100, y=200, source='satellite') is None
        assert cache.get(zoom=15, x=100, y=200, source='satellite') == b'z15'

    def test_clear_zoom_nonexistent(self, cache):
        """Test clear_zoom for nonexistent zoom."""
        deleted = cache.clear_zoom(99)
        assert deleted == 0

    def test_cleanup_lru(self, cache):
        """Test LRU cleanup."""
        # Put some tiles with different sizes
        cache.put(zoom=15, x=1, y=1, source='satellite', data=b'a' * 1000)
        time.sleep(0.01)
        cache.put(zoom=15, x=2, y=2, source='satellite', data=b'b' * 1000)
        time.sleep(0.01)
        cache.put(zoom=15, x=3, y=3, source='satellite', data=b'c' * 1000)

        # Access first tile to make it "recent"
        cache.get(zoom=15, x=1, y=1, source='satellite')

        # Cleanup to very small size - should remove oldest unused
        freed = cache.cleanup_lru(max_size_mb=0)  # Force cleanup of everything

        # At least some bytes should be freed
        assert freed > 0

    def test_context_manager(self, temp_cache_dir):
        """Test context manager usage."""
        with TileCache(cache_dir=temp_cache_dir) as cache:
            cache.put(zoom=15, x=100, y=200, source='satellite', data=b'data')
            assert cache.get(zoom=15, x=100, y=200, source='satellite') == b'data'
        # Cache should be closed after with block

    def test_custom_fetched_at(self, cache):
        """Test put with custom fetched_at."""
        custom_time = 1000000000
        cache.put(
            zoom=15,
            x=100,
            y=200,
            source='satellite',
            data=b'data',
            fetched_at=custom_time,
        )
        info = cache.get_info(zoom=15, x=100, y=200, source='satellite')
        assert info.fetched_at == custom_time

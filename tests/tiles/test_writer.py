"""Tests for CacheWriter."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from tiles.cache import TileCache
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


class TestCacheWriter:
    """Tests for CacheWriter class."""

    def test_start_stop(self, cache):
        """Test basic start and stop."""
        writer = CacheWriter(cache)
        assert not writer.is_running()

        writer.start()
        assert writer.is_running()

        writer.stop()
        assert not writer.is_running()

    def test_context_manager(self, cache):
        """Test context manager usage."""
        with CacheWriter(cache) as writer:
            assert writer.is_running()
        assert not writer.is_running()

    def test_put_and_write(self, cache):
        """Test that put queues tiles for writing."""
        with CacheWriter(cache) as writer:
            result = writer.put(
                zoom=15,
                x=100,
                y=200,
                source='satellite',
                data=b'test data',
            )
            assert result is True

            # Wait for write to complete
            time.sleep(0.5)

        # Verify tile was written
        assert cache.get(zoom=15, x=100, y=200, source='satellite') == b'test data'

    def test_multiple_puts(self, cache):
        """Test multiple puts."""
        with CacheWriter(cache) as writer:
            for i in range(10):
                writer.put(
                    zoom=15,
                    x=i,
                    y=100,
                    source='satellite',
                    data=f'tile{i}'.encode(),
                )

            # Wait for writes
            time.sleep(1.0)

        # Verify all tiles
        for i in range(10):
            assert cache.get(zoom=15, x=i, y=100, source='satellite') == f'tile{i}'.encode()

    def test_batch_write(self, cache):
        """Test that tiles are written in batches."""
        writer = CacheWriter(cache, max_queue_size=100)
        writer.BATCH_SIZE = 5  # Small batch for testing
        writer.start()

        # Put enough tiles for one batch
        for i in range(5):
            writer.put(
                zoom=15,
                x=i,
                y=100,
                source='satellite',
                data=f'tile{i}'.encode(),
            )

        # Wait for batch write
        time.sleep(0.5)
        writer.stop()

        # Verify all tiles
        for i in range(5):
            assert cache.get(zoom=15, x=i, y=100, source='satellite') is not None

    def test_stats(self, cache):
        """Test stats property."""
        with CacheWriter(cache) as writer:
            writer.put(
                zoom=15,
                x=100,
                y=200,
                source='satellite',
                data=b'data',
            )

            time.sleep(0.5)
            stats = writer.stats

            assert 'written' in stats
            assert 'dropped' in stats
            assert 'queue_size' in stats
            assert 'running' in stats
            assert stats['running'] is True

    def test_queue_size(self, cache):
        """Test queue_size method."""
        writer = CacheWriter(cache, max_queue_size=100)
        # Don't start writer so queue accumulates

        for i in range(5):
            writer._queue.put_nowait(
                type('Req', (), {'zoom': 15, 'x': i, 'y': 100, 'source': 's', 'data': b'd'})()
            )

        assert writer.queue_size() == 5

    def test_different_zooms(self, cache):
        """Test writing to different zoom levels."""
        with CacheWriter(cache) as writer:
            writer.put(zoom=14, x=100, y=200, source='satellite', data=b'z14')
            writer.put(zoom=15, x=100, y=200, source='satellite', data=b'z15')
            writer.put(zoom=16, x=100, y=200, source='satellite', data=b'z16')

            time.sleep(0.5)

        assert cache.get(zoom=14, x=100, y=200, source='satellite') == b'z14'
        assert cache.get(zoom=15, x=100, y=200, source='satellite') == b'z15'
        assert cache.get(zoom=16, x=100, y=200, source='satellite') == b'z16'

    def test_double_start(self, cache):
        """Test that double start is safe."""
        writer = CacheWriter(cache)
        writer.start()
        writer.start()  # Should not raise
        assert writer.is_running()
        writer.stop()

    def test_double_stop(self, cache):
        """Test that double stop is safe."""
        writer = CacheWriter(cache)
        writer.start()
        writer.stop()
        writer.stop()  # Should not raise
        assert not writer.is_running()

    def test_stop_without_start(self, cache):
        """Test that stop without start is safe."""
        writer = CacheWriter(cache)
        writer.stop()  # Should not raise

    def test_timeout_flush(self, cache):
        """Test that incomplete batch is flushed after timeout."""
        writer = CacheWriter(cache)
        writer.BATCH_SIZE = 100  # Large batch
        writer.BATCH_TIMEOUT = 0.2  # Short timeout
        writer.start()

        # Put fewer tiles than batch size
        writer.put(zoom=15, x=1, y=1, source='satellite', data=b'data1')
        writer.put(zoom=15, x=2, y=2, source='satellite', data=b'data2')

        # Wait for timeout flush
        time.sleep(0.5)
        writer.stop()

        # Tiles should be written even though batch wasn't full
        assert cache.get(zoom=15, x=1, y=1, source='satellite') == b'data1'
        assert cache.get(zoom=15, x=2, y=2, source='satellite') == b'data2'

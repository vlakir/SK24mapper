"""Tests for tiles.fetcher module."""

import asyncio

import pytest
from PIL import Image

from tiles.fetcher import TileFetcher, stream_tiles


class TestTileFetcher:
    """Tests for TileFetcher class."""

    @pytest.mark.asyncio
    async def test_fetch_many_empty(self):
        """Should handle empty tiles list."""
        async def mock_get(x, y, z):
            return Image.new('RGB', (256, 256))

        fetcher = TileFetcher(mock_get, concurrency=4)
        result = await fetcher.fetch_many([], zoom=15)
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetch_many_single_tile(self):
        """Should fetch single tile."""
        async def mock_get(x, y, z):
            return Image.new('RGB', (256, 256), color=(x % 256, y % 256, z % 256))

        fetcher = TileFetcher(mock_get, concurrency=4)
        tiles = [(0, (100, 200))]
        result = await fetcher.fetch_many(tiles, zoom=15)
        assert len(result) == 1
        assert (15, 100, 200) in result

    @pytest.mark.asyncio
    async def test_fetch_many_multiple_tiles(self):
        """Should fetch multiple tiles."""
        call_count = 0

        async def mock_get(x, y, z):
            nonlocal call_count
            call_count += 1
            return Image.new('RGB', (256, 256))

        fetcher = TileFetcher(mock_get, concurrency=4)
        tiles = [(i, (i * 10, i * 20)) for i in range(5)]
        result = await fetcher.fetch_many(tiles, zoom=10)
        assert len(result) == 5
        assert call_count == 5

    @pytest.mark.asyncio
    async def test_fetch_many_with_progress(self):
        """Should call progress callback."""
        progress_calls = []

        async def mock_get(x, y, z):
            return Image.new('RGB', (256, 256))

        async def on_progress(n):
            progress_calls.append(n)

        fetcher = TileFetcher(mock_get, concurrency=2)
        tiles = [(i, (i, i)) for i in range(3)]
        await fetcher.fetch_many(tiles, zoom=15, on_progress=on_progress)
        assert len(progress_calls) == 3

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Should respect concurrency limit."""
        concurrent = 0
        max_concurrent = 0

        async def mock_get(x, y, z):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.01)
            concurrent -= 1
            return Image.new('RGB', (256, 256))

        fetcher = TileFetcher(mock_get, concurrency=2)
        tiles = [(i, (i, i)) for i in range(10)]
        await fetcher.fetch_many(tiles, zoom=15)
        assert max_concurrent <= 2


class TestStreamTiles:
    """Tests for stream_tiles function."""

    @pytest.mark.asyncio
    async def test_stream_empty(self):
        """Should handle empty tiles list."""
        async def mock_get(x, y, z):
            return Image.new('RGB', (256, 256))

        results = []
        async for key, img in stream_tiles([], mock_get, zoom=15):
            results.append((key, img))
        assert results == []

    @pytest.mark.asyncio
    async def test_stream_single_tile(self):
        """Should stream single tile."""
        async def mock_get(x, y, z):
            return Image.new('RGB', (256, 256))

        tiles = [(0, (100, 200))]
        results = []
        async for key, img in stream_tiles(tiles, mock_get, zoom=15):
            results.append(key)
        assert len(results) == 1
        assert (15, 100, 200) in results

    @pytest.mark.asyncio
    async def test_stream_multiple_tiles(self):
        """Should stream multiple tiles."""
        async def mock_get(x, y, z):
            return Image.new('RGB', (256, 256))

        tiles = [(i, (i * 10, i * 20)) for i in range(5)]
        results = []
        async for key, img in stream_tiles(tiles, mock_get, zoom=10, concurrency=2):
            results.append(key)
        assert len(results) == 5

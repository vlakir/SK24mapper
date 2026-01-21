"""Tests for tiles_executor module."""

import asyncio

import pytest

from tiles_executor import run_tiles


class TestRunTiles:
    """Tests for run_tiles function."""

    @pytest.mark.asyncio
    async def test_empty_tiles(self):
        """Should handle empty tiles list."""
        processed = []

        async def process(idx, tx, ty):
            processed.append((idx, tx, ty))

        await run_tiles(
            [],
            tile_cols=2,
            process_tile=process,
            concurrency=4,
        )
        assert processed == []

    @pytest.mark.asyncio
    async def test_processes_all_tiles(self):
        """Should process all tiles."""
        processed = []

        async def process(idx, tx, ty):
            processed.append((idx, tx, ty))

        tiles = [(0, 0), (1, 0), (0, 1), (1, 1)]
        await run_tiles(
            tiles,
            tile_cols=2,
            process_tile=process,
            concurrency=4,
        )
        assert len(processed) == 4

    @pytest.mark.asyncio
    async def test_with_progress_callback(self):
        """Should call progress callback."""
        progress_calls = []

        async def process(idx, tx, ty):
            pass

        async def on_progress(n):
            progress_calls.append(n)

        tiles = [(0, 0), (1, 0), (2, 0)]
        await run_tiles(
            tiles,
            tile_cols=3,
            process_tile=process,
            concurrency=2,
            progress_step=on_progress,
        )
        assert len(progress_calls) == 3

    @pytest.mark.asyncio
    async def test_with_skip_function(self):
        """Should skip tiles when should_skip returns True."""
        processed = []

        async def process(idx, tx, ty):
            processed.append((tx, ty))

        def should_skip(tx, ty):
            return tx == 1  # Skip all tiles with tx=1

        tiles = [(0, 0), (1, 0), (0, 1), (1, 1)]
        await run_tiles(
            tiles,
            tile_cols=2,
            process_tile=process,
            concurrency=4,
            should_skip=should_skip,
        )
        # Only tiles with tx != 1 should be processed
        assert (1, 0) not in processed
        assert (1, 1) not in processed

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Should respect concurrency limit."""
        concurrent = 0
        max_concurrent = 0

        async def process(idx, tx, ty):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.01)
            concurrent -= 1

        tiles = [(i, i) for i in range(10)]
        await run_tiles(
            tiles,
            tile_cols=10,
            process_tile=process,
            concurrency=2,
        )
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_skip_with_progress(self):
        """Skipped tiles should still call progress."""
        progress_calls = []

        async def process(idx, tx, ty):
            pass

        async def on_progress(n):
            progress_calls.append(n)

        def should_skip(tx, ty):
            return True  # Skip all

        tiles = [(0, 0), (1, 0)]
        await run_tiles(
            tiles,
            tile_cols=2,
            process_tile=process,
            concurrency=2,
            progress_step=on_progress,
            should_skip=should_skip,
        )
        assert len(progress_calls) == 2

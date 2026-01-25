from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable


async def run_tiles(
    tiles: Iterable[tuple[int, int]],
    *,
    tile_cols: int,
    process_tile: Callable[[int, int, int], Awaitable[None]],
    concurrency: int,
    progress_step: Callable[[int], Awaitable[None]] | None = None,
    should_skip: Callable[[int, int], bool] | None = None,
) -> None:
    """
    Placeholder async executor for tiles.

    The detailed implementation will be migrated in a later iteration to avoid
    regressions. For now, this function is not used by service.py yet.
    """
    _ = tile_cols
    sem = asyncio.Semaphore(concurrency)

    async def worker(idx: int, tx: int, ty: int) -> None:
        if should_skip and should_skip(tx, ty):
            if progress_step:
                await progress_step(1)
            return
        async with sem:
            await process_tile(idx, tx, ty)
        if progress_step:
            await progress_step(1)

    await asyncio.gather(
        *(worker(i, x, y) for i, (x, y) in enumerate(tiles)),
        return_exceptions=True,
    )

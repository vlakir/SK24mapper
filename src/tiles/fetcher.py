from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable

    from PIL import Image


class TileFetcher:
    def __init__(
        self,
        get_tile_image: Callable[[int, int, int], Awaitable[Image.Image]],
        *,
        concurrency: int = 8,
    ):
        self._get = get_tile_image
        self._sem = asyncio.Semaphore(concurrency)

    async def fetch_many(
        self,
        tiles: Iterable[tuple[int, tuple[int, int]]],
        *,
        zoom: int,
        on_progress: Callable[[int], Awaitable[None]] | None = None,
    ) -> dict[tuple[int, int, int], Image.Image]:
        """Fetch many tiles concurrently, returning a dict keyed by (zoom,x,y)."""
        out: dict[tuple[int, int, int], Image.Image] = {}

        async def _worker(idx_xy: tuple[int, tuple[int, int]]) -> None:
            idx, (xw, yw) = idx_xy
            async with self._sem:
                img = await self._get(xw, yw, zoom)
            out[(zoom, xw, yw)] = img
            if on_progress is not None:
                with contextlib.suppress(Exception):
                    await on_progress(1)

        await asyncio.gather(*(_worker(t) for t in tiles))
        return out


async def stream_tiles(
    tiles: Iterable[tuple[int, tuple[int, int]]],
    get_tile_image: Callable[[int, int, int], Awaitable[Image.Image]],
    *,
    zoom: int,
    max_queue: int = 64,
    concurrency: int = 8,
) -> AsyncIterator[tuple[tuple[int, int, int], Image.Image]]:
    """Stream tiles using a bounded queue to limit memory."""
    sem = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue[tuple[tuple[int, int, int], Image.Image | None]] = (
        asyncio.Queue(maxsize=max_queue)
    )

    async def _produce() -> None:
        async def _fetch_one(idx_xy: tuple[int, tuple[int, int]]) -> None:
            idx, (xw, yw) = idx_xy
            async with sem:
                img = await get_tile_image(xw, yw, zoom)
            await queue.put(((zoom, xw, yw), img))

        await asyncio.gather(*(_fetch_one(t) for t in tiles))
        await queue.put(((zoom, -1, -1), None))  # sentinel

    prod_task = asyncio.create_task(_produce())
    try:
        while True:
            key, img = await queue.get()
            if img is None and key[1] == -1:
                break
            assert img is not None
            yield key, img
    finally:
        with contextlib.suppress(Exception):
            prod_task.cancel()

"""
Streaming tile fetch + assemble.

Instead of the gather-all-then-assemble pattern (which holds
`len(tiles) × tile_size` bytes in memory at peak — 4.5 GB for HYBRID-grade
1024² tiles at z=17), this module fetches tiles concurrently and assembles
them into the final image AS THEY ARRIVE, freeing each tile immediately
after its pixels are copied into the result.

Peak memory: roughly `concurrency × tile_size` (e.g. 40 × 1024² × 3 = 120 MB
for the XYZ topo path). This makes it feasible to share the HYBRID retina
cache for elev_color / RH / Radar / NSU at z=17 — without it the auto-zoom-
downgrade fires immediately when peak estimate hits the budget ceiling.

Two flavours mirror the existing `assemble_and_crop` / `assemble_dem`
intersect-and-paste logic (row-major tile index → mosaic offset), but
operate per-tile inside an `asyncio.as_completed` loop so order of arrival
doesn't matter.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

import numpy as np
from PIL import Image

from shared.progress import ConsoleProgress

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def stream_fetch_assemble_xyz(
    ctx: MapDownloadContext,
    *,
    eff_tile_px: int,
    fetch_one: Callable[[int, int], Awaitable[Image.Image]],
    label: str = 'Загрузка тайлов',
) -> Image.Image:
    """
    Fetch XYZ-style tiles concurrently, assemble into result image on the fly.

    Args:
        ctx: download context (uses ctx.tiles, tiles_x/y, crop_rect, semaphore).
        eff_tile_px: pixel size of one assembled tile (256, 512, 1024…).
        fetch_one: async callable `(x, y) -> PIL.Image` (caller supplies the
            cache-aware fetcher, e.g. `async_fetch_xyz_tile` with style_id
            and retina baked in).
        label: progress bar label.

    Returns:
        PIL.Image of size `(crop_w, crop_h)` from ctx.crop_rect, mode RGB.
    """
    tiles_x = ctx.tiles_x
    tiles_y = ctx.tiles_y
    crop_x, crop_y, crop_w, crop_h = ctx.crop_rect
    result = Image.new('RGB', (crop_w, crop_h))

    progress = ConsoleProgress(total=len(ctx.tiles), label=label)

    async def _fetch_with_idx(
        idx: int, xy: tuple[int, int]
    ) -> tuple[int, Image.Image]:
        x, y = xy
        async with ctx.semaphore:
            img = await fetch_one(x, y)
        await progress.step(1)
        return idx, img

    tasks = [
        asyncio.create_task(_fetch_with_idx(idx, xy))
        for idx, xy in enumerate(ctx.tiles)
    ]
    try:
        for coro in asyncio.as_completed(tasks):
            idx, img = await coro
            _paste_xyz_tile_into_result(
                result=result,
                img=img,
                idx=idx,
                tiles_x=tiles_x,
                eff_tile_px=eff_tile_px,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_w=crop_w,
                crop_h=crop_h,
            )
            with contextlib.suppress(Exception):
                img.close()
    except BaseException:
        for t in tasks:
            if not t.done():
                t.cancel()
        progress.close()
        raise

    progress.close()
    return result


def _paste_xyz_tile_into_result(
    *,
    result: Image.Image,
    img: Image.Image,
    idx: int,
    tiles_x: int,
    eff_tile_px: int,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
) -> None:
    """Same intersect-and-paste logic as imaging.composer.assemble_and_crop."""
    j = idx // tiles_x
    i = idx % tiles_x

    if img.size != (eff_tile_px, eff_tile_px):
        resized = img.resize(
            (eff_tile_px, eff_tile_px), Image.Resampling.LANCZOS
        )
        with contextlib.suppress(Exception):
            img.close()
        img = resized

    tile_x0 = i * eff_tile_px
    tile_y0 = j * eff_tile_px
    tile_x1 = tile_x0 + eff_tile_px
    tile_y1 = tile_y0 + eff_tile_px

    inter_x0 = max(tile_x0, crop_x)
    inter_y0 = max(tile_y0, crop_y)
    inter_x1 = min(tile_x1, crop_x + crop_w)
    inter_y1 = min(tile_y1, crop_y + crop_h)
    if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
        return

    src_x0 = inter_x0 - tile_x0
    src_y0 = inter_y0 - tile_y0
    src_x1 = inter_x1 - tile_x0
    src_y1 = inter_y1 - tile_y0
    dst_x = inter_x0 - crop_x
    dst_y = inter_y0 - crop_y

    tw, th = img.size
    if src_x0 == 0 and src_y0 == 0 and src_x1 == tw and src_y1 == th:
        result.paste(img, (dst_x, dst_y))
    else:
        tile_crop = img.crop((src_x0, src_y0, src_x1, src_y1))
        result.paste(tile_crop, (dst_x, dst_y))
        tile_crop.close()


async def stream_fetch_assemble_dem(
    ctx: MapDownloadContext,
    *,
    eff_tile_px: int,
    fetch_one: Callable[[int, int], Awaitable[np.ndarray]],
    label: str = 'Загрузка DEM',
    on_tile_done: Callable[[int], None] | None = None,
) -> np.ndarray:
    """
    Fetch Terrain-RGB DEM tiles concurrently, write into result on the fly.

    Args:
        ctx: download context.
        eff_tile_px: tile pixel size (Terrain-RGB is 256 or 512@2x).
        fetch_one: async callable `(x, y) -> np.ndarray (float32 elevation)`.
        label: progress bar label.
        on_tile_done: optional callback `(tile_count_done)` after each paste.

    Returns:
        np.ndarray float32 of shape `(crop_h, crop_w)` from ctx.crop_rect.
    """
    tiles_x = ctx.tiles_x
    cx, cy, cw, ch = ctx.crop_rect
    result = np.zeros((ch, cw), dtype=np.float32)

    progress = ConsoleProgress(total=len(ctx.tiles), label=label)
    tile_count = 0

    async def _fetch_with_idx(
        idx: int, xy: tuple[int, int]
    ) -> tuple[int, np.ndarray]:
        x, y = xy
        async with ctx.semaphore:
            tile = await fetch_one(x, y)
        await progress.step(1)
        return idx, tile

    tasks = [
        asyncio.create_task(_fetch_with_idx(idx, xy))
        for idx, xy in enumerate(ctx.tiles)
    ]
    try:
        for coro in asyncio.as_completed(tasks):
            idx, tile = await coro
            _paste_dem_tile_into_result(
                result=result,
                tile=tile,
                idx=idx,
                tiles_x=tiles_x,
                eff_tile_px=eff_tile_px,
                cx=cx,
                cy=cy,
                cw=cw,
                ch=ch,
            )
            # Drop the tile ref immediately — numpy releases since we hold
            # no other reference after the slice copy above.
            del tile
            tile_count += 1
            if on_tile_done is not None:
                on_tile_done(tile_count)
    except BaseException:
        for t in tasks:
            if not t.done():
                t.cancel()
        progress.close()
        raise

    progress.close()
    return result


def _paste_dem_tile_into_result(
    *,
    result: np.ndarray,
    tile: np.ndarray,
    idx: int,
    tiles_x: int,
    eff_tile_px: int,
    cx: int,
    cy: int,
    cw: int,
    ch: int,
) -> None:
    """Same intersect-and-paste logic as geo.topography.assemble_dem."""
    ty = idx // tiles_x
    tx = idx % tiles_x
    tile_h, tile_w = tile.shape

    tile_x0 = tx * eff_tile_px
    tile_y0 = ty * eff_tile_px
    tile_x1 = tile_x0 + min(eff_tile_px, tile_w)
    tile_y1 = tile_y0 + min(eff_tile_px, tile_h)

    inter_x0 = max(tile_x0, cx)
    inter_y0 = max(tile_y0, cy)
    inter_x1 = min(tile_x1, cx + cw)
    inter_y1 = min(tile_y1, cy + ch)
    if inter_x0 >= inter_x1 or inter_y0 >= inter_y1:
        return

    src_x0 = inter_x0 - tile_x0
    src_y0 = inter_y0 - tile_y0
    src_x1 = inter_x1 - tile_x0
    src_y1 = inter_y1 - tile_y0
    dst_x = inter_x0 - cx
    dst_y = inter_y0 - cy
    dst_w = src_x1 - src_x0
    dst_h = src_y1 - src_y0

    result[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w] = tile[
        src_y0:src_y1, src_x0:src_x1
    ]

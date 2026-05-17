"""
In-memory + disk LRU for the heaviest two artefacts of the run_processor
stage: the assembled DEM (numpy float32) and the assembled topo base
(numpy uint8 RGB). Together they account for ~3 s of warm rebuild time
on elev_color at z=17 (1444 tile fetches + decodes + assembly).

Two layers:
  - In-memory: 1-entry LRU per kind (DEM, topo). Holds ~640 MB peak
    (367 DEM + 275 topo for a z=17 retina elev_color area). Keyed on
    area + zoom + retina + style — enough to make the contents
    bit-identical when it hits.
  - Disk: .npy via memmap. ~640 MB on disk per entry. mtime-based LRU
    with DEM_TOPO_DISK_CACHE_DAYS TTL.

API mirrors contour_layer_disk_cache:
  load(key) -> ndarray | None
  save(key, arr) -> None (sync write; saves are small relative to read)
  wait_for_pending_saves(timeout=None)

In-memory access is via get_inmem(key) / set_inmem(key, arr).

The PIL topo image is stored as a numpy view (Image.fromarray-able) to
avoid a second PIL <-> numpy round trip on the cache path.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from infrastructure.http.client import resolve_cache_dir
from shared.constants import (
    DEM_TOPO_DISK_CACHE_DAYS,
    DEM_TOPO_DISK_CACHE_ENABLED,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CLEANUP_THROTTLE_SECONDS = 24 * 3600
_MEM_LRU_MAX = 1  # one entry per kind — each is up to ~640 MB

# Per-kind LRU dicts (insertion-order = LRU).
_inmem: dict[str, OrderedDict[tuple, np.ndarray]] = {
    'dem': OrderedDict(),
    'topo': OrderedDict(),
}

_pending_saves: list[threading.Thread] = []
_pending_lock = threading.Lock()


def _hash_key(key: tuple) -> str:
    return hashlib.sha256(repr(key).encode('utf-8')).hexdigest()


def _resolve_dir(kind: str) -> Path | None:
    if not DEM_TOPO_DISK_CACHE_ENABLED:
        return None
    tiles_dir = resolve_cache_dir()
    if tiles_dir is None:
        return None
    # tiles_dir = <cache_root>/tiles  →  sibling <cache_root>/<kind>_layers
    return tiles_dir.parent / f'{kind}_layers'


# ---------------------------------------------------------------------------
# In-memory layer
# ---------------------------------------------------------------------------


def get_inmem(kind: str, key: tuple) -> np.ndarray | None:
    bucket = _inmem.get(kind)
    if bucket is None:
        return None
    arr = bucket.get(key)
    if arr is not None:
        # LRU bump
        bucket.move_to_end(key)
    return arr


def set_inmem(kind: str, key: tuple, arr: np.ndarray) -> None:
    bucket = _inmem.setdefault(kind, OrderedDict())
    if key in bucket:
        bucket.move_to_end(key)
        bucket[key] = arr
        return
    bucket[key] = arr
    while len(bucket) > _MEM_LRU_MAX:
        evicted_key, _evicted = bucket.popitem(last=False)
        logger.info(
            'dem_topo_cache[%s] in-mem evict (LRU full at %d)', kind, _MEM_LRU_MAX,
        )


def get(kind: str, key: tuple) -> np.ndarray | None:
    """Two-layer lookup: in-mem first, then disk (promotes hit to in-mem)."""
    arr = get_inmem(kind, key)
    if arr is not None:
        return arr
    arr = load(kind, key)
    if arr is not None:
        set_inmem(kind, key, arr)
    return arr


def put(kind: str, key: tuple, arr: np.ndarray) -> None:
    """Two-layer write: in-mem now, disk in background."""
    set_inmem(kind, key, arr)
    save(kind, key, arr)


# ---------------------------------------------------------------------------
# Disk layer
# ---------------------------------------------------------------------------


def load(kind: str, key: tuple) -> np.ndarray | None:
    """Load a ndarray from disk; returns None on miss / error.

    Uses np.load with mmap_mode='r' for instant access then .copy() so the
    returned array isn't tied to a file handle. Memmap read of ~640 MB
    .npy on NVMe is ~100-200 ms (cache miss in the OS page cache); a
    second call within the page-cache window is essentially free.
    """
    cache_dir = _resolve_dir(kind)
    if cache_dir is None:
        return None
    path = cache_dir / f'{_hash_key(key)}.npy'
    if not path.exists():
        return None
    t0 = time.monotonic()
    try:
        mm = np.load(path, mmap_mode='r')
        arr = np.array(mm)  # one memcpy out of the mapping
        del mm
        with contextlib.suppress(OSError):
            path.touch()
        logger.info(
            'dem_topo_cache[%s] disk HIT %s shape=%s (%.0f MB, %.2f s)',
            kind, path.name, arr.shape, arr.nbytes / 1e6,
            time.monotonic() - t0,
        )
    except Exception:
        logger.warning(
            'dem_topo_cache[%s] disk load failed for %s',
            kind, path, exc_info=True,
        )
        return None
    else:
        return arr


def save(kind: str, key: tuple, arr: np.ndarray) -> None:
    """Schedule background np.save to disk; never blocks the caller."""
    cache_dir = _resolve_dir(kind)
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(
            'dem_topo_cache[%s] cannot create %s', kind, cache_dir,
            exc_info=True,
        )
        return

    # Snapshot: take a contiguous copy so the saver thread isn't racing
    # with caller mutations / GCs of the original buffer.
    try:
        snapshot = np.ascontiguousarray(arr)
        if snapshot is arr:
            snapshot = arr.copy()
    except Exception:
        logger.warning('dem_topo_cache[%s] snapshot failed', kind, exc_info=True)
        return

    path = cache_dir / f'{_hash_key(key)}.npy'
    t = threading.Thread(
        target=_save_worker,
        args=(kind, snapshot, path, cache_dir),
        name=f'{kind}-disk-save',
        daemon=True,
    )
    with _pending_lock:
        _pending_saves[:] = [p for p in _pending_saves if p.is_alive()]
        _pending_saves.append(t)
    t.start()


def _save_worker(
    kind: str, snapshot: np.ndarray, path: Path, cache_dir: Path,
) -> None:
    tmp = path.with_suffix('.npy.tmp')
    t0 = time.monotonic()
    try:
        # File-object form prevents np.save from auto-appending '.npy' to
        # the path (which would write to '<name>.npy.tmp.npy' and leave
        # the rename below with no source).
        with tmp.open('wb') as f:
            np.save(f, snapshot, allow_pickle=False)
        tmp.replace(path)
        size_mb = path.stat().st_size / 1e6
        logger.info(
            'dem_topo_cache[%s] disk stored %s (%.0f MB, %.2f s, bg)',
            kind, path.name, size_mb, time.monotonic() - t0,
        )
    except Exception:
        logger.warning(
            'dem_topo_cache[%s] disk save failed for %s',
            kind, path, exc_info=True,
        )
        with contextlib.suppress(OSError):
            tmp.unlink()
        return

    _maybe_cleanup_expired(kind, cache_dir)


def _maybe_cleanup_expired(kind: str, cache_dir: Path) -> None:
    marker = cache_dir / '.last_cleanup'
    now = time.time()
    try:
        last = marker.stat().st_mtime
    except OSError:
        last = 0.0
    if now - last < _CLEANUP_THROTTLE_SECONDS:
        return
    cutoff = now - DEM_TOPO_DISK_CACHE_DAYS * 86400
    removed = 0
    total = 0
    for entry in cache_dir.glob('*.npy'):
        total += 1
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            logger.debug(
                'dem_topo_cache[%s] cleanup failed for %s', kind, entry,
                exc_info=True,
            )
    with contextlib.suppress(OSError):
        marker.touch()
    if removed:
        logger.info(
            'dem_topo_cache[%s] pruned %d/%d entries older than %d days',
            kind, removed, total, DEM_TOPO_DISK_CACHE_DAYS,
        )


def wait_for_pending_saves(timeout: float | None = None) -> None:
    with _pending_lock:
        threads = list(_pending_saves)
        _pending_saves.clear()
    deadline = (
        time.monotonic() + timeout if timeout is not None else None
    )
    for t in threads:
        if t.is_alive():
            remaining = (
                None if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            t.join(timeout=remaining)


# ---------------------------------------------------------------------------
# High-level get_or_build
# ---------------------------------------------------------------------------


def make_area_key(
    *,
    kind_tag: str,
    zoom: int,
    eff_scale: int,
    use_retina: bool,
    center_lat_wgs: float,
    center_lng_wgs: float,
    width_m: float,
    height_m: float,
    style_id: str | None = None,
    tile_size: int | None = None,
    extras: tuple = (),
) -> tuple:
    """Canonical cache key for an assembled DEM / topo over an area."""
    return (
        kind_tag,
        zoom,
        eff_scale,
        bool(use_retina),
        round(center_lat_wgs, 6),
        round(center_lng_wgs, 6),
        round(width_m, 2),
        round(height_m, 2),
        style_id,
        tile_size,
        *extras,
    )

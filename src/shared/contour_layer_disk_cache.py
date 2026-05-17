"""
On-disk LRU for pre-rotation contour overlay layers.

Survives across worker processes and app restarts: the first build after
a fresh launch on a familiar area no longer pays the marching-squares +
label-placement + line-drawing cost (~1.5 s on elev_color at z=17). Sits
behind the in-memory _contour_layer_cache in map_download_service so the
hot in-process path stays fast, and only cold restarts touch the disk.

Eviction:
  - mtime-based: entries older than CONTOUR_LAYER_DISK_CACHE_DAYS are
    pruned. Reads bump mtime via Path.touch() so frequently-used entries
    stay warm.
  - Throttled to once per 24 h via a `.last_cleanup` marker in the cache
    directory so the cleanup never lands on the hot path.

Format: PNG with `compress_level=1` (fast). Most of a contour layer's
pixels are alpha=0, so PNG with even low compression gets a 10-30×
ratio. A failed read/write logs at WARNING and degrades to a miss; the
in-memory cache and on-the-fly rebuild still work.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from infrastructure.http.client import resolve_cache_dir
from shared.constants import (
    CONTOUR_LAYER_DISK_CACHE_DAYS,
    CONTOUR_LAYER_DISK_CACHE_ENABLED,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

_CLEANUP_THROTTLE_SECONDS = 24 * 3600

# Track outstanding background save threads so tests / benchmarks can
# join them before the process exits. In production the persistent
# worker lives long enough that daemon threads always finish between
# builds; this is purely here for synchronous tear-downs.
_pending_saves: list[threading.Thread] = []
_pending_lock = threading.Lock()


def wait_for_pending_saves(timeout: float | None = None) -> None:
    """Join all outstanding background save threads.

    Useful for benchmarks / shutdown — in normal worker operation save
    daemon threads finish between builds and this is a no-op.
    """
    with _pending_lock:
        threads = list(_pending_saves)
        _pending_saves.clear()
    deadline = (time.monotonic() + timeout) if timeout is not None else None
    for t in threads:
        if t.is_alive():
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            t.join(timeout=remaining)


def _hash_key(key: tuple) -> str:
    # tuple repr() is deterministic for the built-in / enum types this key
    # uses (int, float, str, bool, None, MapType etc.), so sha256 over repr
    # gives a stable filename across processes.
    return hashlib.sha256(repr(key).encode('utf-8')).hexdigest()


def _resolve_dir() -> Path | None:
    if not CONTOUR_LAYER_DISK_CACHE_ENABLED:
        return None
    tiles_dir = resolve_cache_dir()
    if tiles_dir is None:
        return None
    # tiles_dir = <cache_root>/tiles  →  sibling <cache_root>/contour_layers
    return tiles_dir.parent / 'contour_layers'


def load(key: tuple) -> 'Image.Image | None':
    """Return a PIL Image from disk, or None on miss / error."""
    from PIL import Image  # local import keeps cold-start light

    cache_dir = _resolve_dir()
    if cache_dir is None:
        return None
    hash_name = _hash_key(key)
    # WebP-lossless is the current format (commit replacing PNG). Older
    # `.png` files left over from a previous version are simply ignored
    # — they expire under the regular TTL or get cleaned by the
    # background sweep below.
    path = cache_dir / f'{hash_name}.webp'
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        img.load()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        # Bump mtime so LRU keeps frequently-touched entries fresh.
        with contextlib.suppress(OSError):
            path.touch()
        logger.info(
            'contour_layer disk cache HIT: %s (%d×%d)',
            path.name, img.width, img.height,
        )
    except Exception:
        logger.warning(
            'contour_layer disk cache: load failed for %s', path,
            exc_info=True,
        )
        return None
    else:
        return img


def save(key: tuple, img: 'Image.Image') -> None:
    """
    Schedule a background write of `img` to disk; never blocks the caller.

    PNG encoding at 8404² RGBA takes ~2 s even at compress_level=1 — far
    too expensive for the postprocess hot path. We snapshot the bitmap
    (a single ~282 MB RGBA copy, ~80 ms) so the caller is free to
    rotate/close the original immediately, then push the encode into a
    daemon thread. Failures are logged inside the thread and never
    surface back to the caller.
    """
    cache_dir = _resolve_dir()
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning(
            'contour_layer disk cache: cannot create %s', cache_dir,
            exc_info=True,
        )
        return

    # Snapshot: detach from the caller's lifecycle. _postprocess закрывает
    # contour_layer (rotate_then_center_crop с close_input=True). Без
    # этой копии save-thread читал бы закрытый PIL.Image и WebPEncode
    # валился с ValueError: encoding error 1.
    try:
        snapshot = img if img.mode == 'RGBA' else img.convert('RGBA')
        snapshot = snapshot.copy()
    except Exception:
        logger.warning(
            'contour_layer disk cache: snapshot failed', exc_info=True,
        )
        return

    path = cache_dir / f'{_hash_key(key)}.webp'
    t = threading.Thread(
        target=_save_worker,
        args=(snapshot, path, cache_dir),
        name='contour-disk-save',
        daemon=True,
    )
    with _pending_lock:
        # Garbage-collect already-finished threads so the list doesn't
        # grow without bound over the worker's lifetime.
        _pending_saves[:] = [p for p in _pending_saves if p.is_alive()]
        _pending_saves.append(t)
    t.start()


def _save_worker(snapshot: 'Image.Image', path: Path, cache_dir: Path) -> None:
    tmp = path.with_suffix('.webp.tmp')
    t0 = time.monotonic()
    try:
        # WebP lossless (libwebp via Pillow). Bench on a real 9580²
        # sparse-RGBA contour layer:
        #   PNG level=1: encode 1.96 s → 9.2 MB, decode 0.71 s
        #   WebP q=9   : encode 1.43 s → 2.1 MB, decode 0.48 s
        # ~4× smaller on disk, ~30 % faster encode, ~30 % faster decode.
        # `method=6` is the highest-quality encoder pass available; Pillow
        # exposes the 0–9 axis via `quality=` even for lossless (it maps
        # to libwebp's `method`).
        snapshot.save(
            tmp, format='WebP', lossless=True, quality=9, method=6,
        )
        tmp.replace(path)
        size_mb = path.stat().st_size / 1e6
        logger.info(
            'contour_layer disk cache: stored %s (%.1f MB, %.2f s, bg)',
            path.name, size_mb, time.monotonic() - t0,
        )
    except Exception:
        logger.warning(
            'contour_layer disk cache: save failed for %s', path,
            exc_info=True,
        )
        with contextlib.suppress(OSError):
            tmp.unlink()
        return
    finally:
        with contextlib.suppress(Exception):
            snapshot.close()

    _maybe_cleanup_expired(cache_dir)


def _maybe_cleanup_expired(cache_dir: Path) -> None:
    """Cleanup runs at most once per _CLEANUP_THROTTLE_SECONDS."""
    marker = cache_dir / '.last_cleanup'
    now = time.time()
    try:
        last = marker.stat().st_mtime
    except OSError:
        last = 0.0
    if now - last < _CLEANUP_THROTTLE_SECONDS:
        return
    cutoff = now - CONTOUR_LAYER_DISK_CACHE_DAYS * 86400
    removed = 0
    total = 0
    # Sweep both formats: `.webp` is the current format, legacy `.png`
    # entries (from before the WebP migration) are unconditionally
    # eligible for pruning since the load() path no longer reads them.
    candidates: list[tuple[Path, bool]] = []
    for entry in cache_dir.glob('*.webp'):
        candidates.append((entry, False))
    for entry in cache_dir.glob('*.png'):
        candidates.append((entry, True))
    for entry, is_legacy in candidates:
        total += 1
        try:
            if is_legacy or entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            logger.debug(
                'contour_layer disk cache: cleanup failed for %s', entry,
                exc_info=True,
            )
    with contextlib.suppress(OSError):
        marker.touch()
    if removed:
        logger.info(
            'contour_layer disk cache: pruned %d/%d entries older than %d days',
            removed, total, CONTOUR_LAYER_DISK_CACHE_DAYS,
        )

"""Async writer thread for non-blocking cache writes.

This module provides CacheWriter class that handles tile writes
in a background thread to avoid blocking the main download loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiles.cache import TileCache

from shared.constants import TILE_WRITE_QUEUE_SIZE

logger = logging.getLogger(__name__)


@dataclass
class TileWriteRequest:
    """Request to write a tile to cache."""

    zoom: int
    x: int
    y: int
    source: str
    data: bytes
    fetched_at: int | None = None


class CacheWriter:
    """Background writer thread for tile cache.

    Collects write requests in a queue and processes them in batches
    in a background thread. This allows the main download loop to
    continue without waiting for SQLite writes.

    Features:
    - Non-blocking put() method
    - Batch writes for better performance
    - Graceful shutdown with flush
    - Queue size monitoring

    Usage:
        cache = TileCache()
        writer = CacheWriter(cache)
        writer.start()

        # Non-blocking writes
        writer.put(zoom=15, x=100, y=200, source='satellite', data=tile_bytes)

        # Shutdown
        writer.stop()  # Waits for queue to drain
    """

    BATCH_SIZE = 50  # Number of tiles to write per transaction
    BATCH_TIMEOUT = 1.0  # Seconds to wait before writing incomplete batch

    def __init__(
        self,
        cache: TileCache,
        max_queue_size: int | None = None,
    ) -> None:
        """Initialize cache writer.

        Args:
            cache: TileCache instance to write to.
            max_queue_size: Maximum queue size. Defaults to TILE_WRITE_QUEUE_SIZE.
        """
        self.cache = cache
        self.max_queue_size = max_queue_size or TILE_WRITE_QUEUE_SIZE
        self._queue: queue.Queue[TileWriteRequest | None] = queue.Queue(
            maxsize=self.max_queue_size
        )
        self._thread: threading.Thread | None = None
        self._running = False
        self._stats_written = 0
        self._stats_dropped = 0

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        logger.info('CacheWriter started')

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the writer thread and wait for queue to drain.

        Args:
            timeout: Maximum time to wait for queue to drain.
        """
        if not self._running:
            return
        self._running = False

        # Signal thread to stop
        try:
            self._queue.put(None, block=False)
        except queue.Full:
            pass

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning('CacheWriter thread did not stop within timeout')

        logger.info(
            'CacheWriter stopped: %d tiles written, %d dropped',
            self._stats_written,
            self._stats_dropped,
        )

    def put(
        self,
        zoom: int,
        x: int,
        y: int,
        source: str,
        data: bytes,
        fetched_at: int | None = None,
        block: bool = False,
    ) -> bool:
        """Queue a tile for writing.

        Args:
            zoom: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            source: Tile source.
            data: Tile data.
            fetched_at: Timestamp when tile was fetched.
            block: If True, block until queue has space.

        Returns:
            True if tile was queued, False if queue was full.
        """
        request = TileWriteRequest(
            zoom=zoom,
            x=x,
            y=y,
            source=source,
            data=data,
            fetched_at=fetched_at,
        )
        try:
            self._queue.put(request, block=block, timeout=1.0 if block else None)
            return True
        except queue.Full:
            self._stats_dropped += 1
            logger.warning(
                'Write queue full, dropping tile z%d/%d/%d (%s)',
                zoom,
                x,
                y,
                source,
            )
            return False

    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def is_running(self) -> bool:
        """Check if writer thread is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict:
        """Get writer statistics."""
        return {
            'written': self._stats_written,
            'dropped': self._stats_dropped,
            'queue_size': self.queue_size(),
            'running': self.is_running(),
        }

    def _writer_loop(self) -> None:
        """Background thread loop that processes write requests."""
        batch: list[TileWriteRequest] = []
        last_write_time = time.time()

        while True:
            try:
                # Wait for request with timeout
                request = self._queue.get(timeout=self.BATCH_TIMEOUT)

                # None signals shutdown
                if request is None:
                    # Write remaining batch
                    if batch:
                        self._write_batch(batch)
                    break

                batch.append(request)

                # Write batch if full or timeout reached
                current_time = time.time()
                if (
                    len(batch) >= self.BATCH_SIZE
                    or (current_time - last_write_time) >= self.BATCH_TIMEOUT
                ):
                    self._write_batch(batch)
                    batch = []
                    last_write_time = current_time

            except queue.Empty:
                # Timeout - write any pending tiles
                if batch:
                    self._write_batch(batch)
                    batch = []
                    last_write_time = time.time()

                # Check if we should stop
                if not self._running:
                    break

    def _write_batch(self, batch: list[TileWriteRequest]) -> None:
        """Write a batch of tiles to cache.

        Args:
            batch: List of tile write requests.
        """
        if not batch:
            return

        # Group by zoom for efficient batch writes
        by_zoom: dict[int, list[tuple[int, int, str, bytes]]] = {}
        for req in batch:
            if req.zoom not in by_zoom:
                by_zoom[req.zoom] = []
            by_zoom[req.zoom].append((req.x, req.y, req.source, req.data))

        # Write each zoom level batch
        for zoom, tiles in by_zoom.items():
            try:
                self.cache.put_batch(zoom, tiles)
                self._stats_written += len(tiles)
            except Exception:
                logger.exception('Error writing batch to zoom %d', zoom)

    def __enter__(self) -> CacheWriter:
        """Context manager entry - starts the writer."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops the writer."""
        self.stop()

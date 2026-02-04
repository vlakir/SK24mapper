"""SQLite-based tile cache with LRU eviction.

This module provides TileCache class for storing and retrieving map tiles
in SQLite databases organized by zoom level.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from shared.constants import TILE_CACHE_DIR, TILE_CACHE_MAX_SIZE_MB

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Information about a cached tile."""

    x: int
    y: int
    zoom: int
    source: str
    size_bytes: int
    fetched_at: int
    last_used_at: int


@dataclass
class CacheStats:
    """Statistics about the tile cache."""

    total_tiles: int
    total_size_bytes: int
    tiles_by_zoom: dict[int, int]
    size_by_zoom: dict[int, int]
    oldest_tile: int | None
    newest_tile: int | None


class TileCache:
    """SQLite-based tile cache with separate databases per zoom level.

    Features:
    - Separate SQLite database for each zoom level
    - WAL mode for concurrent reads
    - LRU eviction based on total cache size
    - Automatic last_used_at update on reads

    Usage:
        cache = TileCache()
        cache.put(zoom=15, x=100, y=200, source='satellite', data=tile_bytes)
        tile_data = cache.get(zoom=15, x=100, y=200, source='satellite')
        cache.close()
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize tile cache.

        Args:
            cache_dir: Directory for cache files. Defaults to TILE_CACHE_DIR.
        """
        self.cache_dir = Path(cache_dir or TILE_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._connections: dict[int, sqlite3.Connection] = {}
        logger.info('TileCache initialized at %s', self.cache_dir)

    def _get_db_path(self, zoom: int) -> Path:
        """Get database file path for a zoom level."""
        return self.cache_dir / f'zoom_{zoom}.db'

    def _get_connection(self, zoom: int) -> sqlite3.Connection:
        """Get or create a connection for the given zoom level."""
        if zoom not in self._connections:
            db_path = self._get_db_path(zoom)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.execute('PRAGMA cache_size=-64000')  # 64MB cache
            self._init_schema(conn)
            self._connections[zoom] = conn
        return self._connections[zoom]

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema."""
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS tiles (
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                source TEXT NOT NULL,
                tile_data BLOB NOT NULL,
                fetched_at INTEGER NOT NULL,
                last_used_at INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY (x, y, source)
            );

            CREATE INDEX IF NOT EXISTS idx_tiles_last_used ON tiles(last_used_at);
            CREATE INDEX IF NOT EXISTS idx_tiles_source ON tiles(source);
        ''')
        conn.commit()

    def get(self, zoom: int, x: int, y: int, source: str) -> bytes | None:
        """Get tile data from cache.

        Updates last_used_at timestamp for LRU tracking.

        Args:
            zoom: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            source: Tile source (e.g., 'satellite', 'terrain-rgb').

        Returns:
            Tile data as bytes, or None if not found.
        """
        conn = self._get_connection(zoom)
        cursor = conn.execute(
            'SELECT tile_data FROM tiles WHERE x = ? AND y = ? AND source = ?',
            (x, y, source),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        # Update last_used_at for LRU
        now = int(time.time())
        conn.execute(
            'UPDATE tiles SET last_used_at = ? WHERE x = ? AND y = ? AND source = ?',
            (now, x, y, source),
        )
        conn.commit()
        return row[0]

    def get_info(self, zoom: int, x: int, y: int, source: str) -> TileInfo | None:
        """Get tile metadata without updating last_used_at.

        Args:
            zoom: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            source: Tile source.

        Returns:
            TileInfo or None if not found.
        """
        conn = self._get_connection(zoom)
        cursor = conn.execute(
            '''SELECT x, y, source, size_bytes, fetched_at, last_used_at
               FROM tiles WHERE x = ? AND y = ? AND source = ?''',
            (x, y, source),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return TileInfo(
            x=row[0],
            y=row[1],
            zoom=zoom,
            source=row[2],
            size_bytes=row[3],
            fetched_at=row[4],
            last_used_at=row[5],
        )

    def exists(self, zoom: int, x: int, y: int, source: str) -> bool:
        """Check if tile exists in cache.

        Args:
            zoom: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            source: Tile source.

        Returns:
            True if tile exists.
        """
        conn = self._get_connection(zoom)
        cursor = conn.execute(
            'SELECT 1 FROM tiles WHERE x = ? AND y = ? AND source = ?',
            (x, y, source),
        )
        return cursor.fetchone() is not None

    def put(
        self,
        zoom: int,
        x: int,
        y: int,
        source: str,
        data: bytes,
        fetched_at: int | None = None,
    ) -> None:
        """Store tile in cache.

        Args:
            zoom: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            source: Tile source.
            data: Tile data as bytes.
            fetched_at: Timestamp when tile was fetched. Defaults to now.
        """
        now = int(time.time())
        fetched_at = fetched_at or now
        conn = self._get_connection(zoom)
        conn.execute(
            '''INSERT OR REPLACE INTO tiles
               (x, y, source, tile_data, fetched_at, last_used_at, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (x, y, source, data, fetched_at, now, len(data)),
        )
        conn.commit()

    def put_batch(
        self,
        zoom: int,
        tiles: Sequence[tuple[int, int, str, bytes]],
        fetched_at: int | None = None,
    ) -> None:
        """Store multiple tiles in a single transaction.

        Args:
            zoom: Zoom level.
            tiles: Sequence of (x, y, source, data) tuples.
            fetched_at: Timestamp when tiles were fetched. Defaults to now.
        """
        if not tiles:
            return
        now = int(time.time())
        fetched_at = fetched_at or now
        conn = self._get_connection(zoom)
        conn.executemany(
            '''INSERT OR REPLACE INTO tiles
               (x, y, source, tile_data, fetched_at, last_used_at, size_bytes)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            [(x, y, source, data, fetched_at, now, len(data)) for x, y, source, data in tiles],
        )
        conn.commit()

    def delete(self, zoom: int, x: int, y: int, source: str) -> bool:
        """Delete a tile from cache.

        Args:
            zoom: Zoom level.
            x: Tile X coordinate.
            y: Tile Y coordinate.
            source: Tile source.

        Returns:
            True if tile was deleted.
        """
        conn = self._get_connection(zoom)
        cursor = conn.execute(
            'DELETE FROM tiles WHERE x = ? AND y = ? AND source = ?',
            (x, y, source),
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics across all zoom levels.

        Returns:
            CacheStats with totals and per-zoom breakdown.
        """
        total_tiles = 0
        total_size = 0
        tiles_by_zoom: dict[int, int] = {}
        size_by_zoom: dict[int, int] = {}
        oldest_tile: int | None = None
        newest_tile: int | None = None

        # Scan for all zoom databases
        for db_file in self.cache_dir.glob('zoom_*.db'):
            try:
                zoom = int(db_file.stem.split('_')[1])
            except (IndexError, ValueError):
                continue

            conn = self._get_connection(zoom)
            cursor = conn.execute(
                'SELECT COUNT(*), COALESCE(SUM(size_bytes), 0), MIN(fetched_at), MAX(fetched_at) FROM tiles'
            )
            row = cursor.fetchone()
            if row:
                count, size, oldest, newest = row
                tiles_by_zoom[zoom] = count
                size_by_zoom[zoom] = size
                total_tiles += count
                total_size += size
                if oldest is not None:
                    if oldest_tile is None or oldest < oldest_tile:
                        oldest_tile = oldest
                if newest is not None:
                    if newest_tile is None or newest > newest_tile:
                        newest_tile = newest

        return CacheStats(
            total_tiles=total_tiles,
            total_size_bytes=total_size,
            tiles_by_zoom=tiles_by_zoom,
            size_by_zoom=size_by_zoom,
            oldest_tile=oldest_tile,
            newest_tile=newest_tile,
        )

    def cleanup_lru(self, max_size_mb: int | None = None) -> int:
        """Remove least recently used tiles to stay under size limit.

        Args:
            max_size_mb: Maximum cache size in MB. Defaults to TILE_CACHE_MAX_SIZE_MB.

        Returns:
            Number of bytes freed.
        """
        if max_size_mb is None:
            max_size_mb = TILE_CACHE_MAX_SIZE_MB
        max_size_bytes = max_size_mb * 1024 * 1024
        stats = self.get_stats()

        if stats.total_size_bytes <= max_size_bytes:
            return 0

        bytes_to_free = stats.total_size_bytes - max_size_bytes
        bytes_freed = 0

        # Collect all tiles with their last_used_at across all zooms
        all_tiles: list[tuple[int, int, int, str, int, int]] = []  # (zoom, x, y, source, size, last_used)

        for db_file in self.cache_dir.glob('zoom_*.db'):
            try:
                zoom = int(db_file.stem.split('_')[1])
            except (IndexError, ValueError):
                continue

            conn = self._get_connection(zoom)
            cursor = conn.execute(
                'SELECT x, y, source, size_bytes, last_used_at FROM tiles'
            )
            for row in cursor:
                all_tiles.append((zoom, row[0], row[1], row[2], row[3], row[4]))

        # Sort by last_used_at (oldest first)
        all_tiles.sort(key=lambda t: t[5])

        # Delete oldest tiles until we've freed enough space
        for zoom, x, y, source, size, _ in all_tiles:
            if bytes_freed >= bytes_to_free:
                break
            if self.delete(zoom, x, y, source):
                bytes_freed += size

        logger.info(
            'LRU cleanup: freed %.1f MB (target: %.1f MB)',
            bytes_freed / 1024 / 1024,
            bytes_to_free / 1024 / 1024,
        )
        return bytes_freed

    def clear_zoom(self, zoom: int) -> int:
        """Delete all tiles for a specific zoom level.

        Args:
            zoom: Zoom level to clear.

        Returns:
            Number of tiles deleted.
        """
        db_path = self._get_db_path(zoom)
        if not db_path.exists():
            return 0

        # Close connection if open
        if zoom in self._connections:
            self._connections[zoom].close()
            del self._connections[zoom]

        # Get count before deletion
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute('SELECT COUNT(*) FROM tiles')
        count = cursor.fetchone()[0]
        conn.close()

        # Delete the database file
        db_path.unlink()
        logger.info('Cleared zoom %d: %d tiles deleted', zoom, count)
        return count

    def close(self) -> None:
        """Close all database connections."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
        logger.info('TileCache closed')

    def __enter__(self) -> TileCache:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

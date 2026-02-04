"""Tests for http_client module."""

import gc
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.http.client import (
    cleanup_sqlite_cache,
    make_http_session,
    resolve_cache_dir,
    validate_style_api,
    validate_terrain_api,
)


class TestResolveCacheDir:
    """Tests for resolve_cache_dir function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = resolve_cache_dir()
        assert isinstance(result, Path)

    def test_with_localappdata_env(self):
        """Should return absolute path from TILE_CACHE_DIR."""
        result = resolve_cache_dir()
        assert result is not None
        # TILE_CACHE_DIR is now absolute path in home directory
        assert '.sk24mapper' in str(result) or 'tiles' in str(result)

    def test_fallback_to_home(self):
        """Should fallback to home directory when LOCALAPPDATA not set."""
        with patch.dict('os.environ', {'LOCALAPPDATA': ''}, clear=False):
            with patch('os.getenv', return_value=None):
                result = resolve_cache_dir()
                assert result is not None


class TestCleanupSqliteCache:
    """Tests for cleanup_sqlite_cache function.

    Note: cleanup_sqlite_cache is now a no-op for backward compatibility.
    Tile cache cleanup is handled by TileCache.cleanup_lru().
    """

    def test_cleanup_nonexistent_cache(self):
        """Should handle non-existent cache directory gracefully (no-op)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'nonexistent'
            # Should not raise (no-op function)
            cleanup_sqlite_cache(cache_dir)

    def test_cleanup_existing_cache(self):
        """Should be a no-op even when cache file exists."""
        import sqlite3
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / 'http_cache.sqlite'
            # Create a test SQLite file
            conn = sqlite3.connect(cache_file)
            conn.execute('CREATE TABLE test (id INTEGER)')
            conn.close()
            # Should not raise (no-op function)
            cleanup_sqlite_cache(cache_dir)
            # File should still exist (function is no-op)
            assert cache_file.exists()


class TestMakeHttpSession:
    """Tests for make_http_session function."""

    @pytest.mark.asyncio
    async def test_creates_session_without_cache(self):
        """Should create session without cache when cache_dir is None."""
        session = make_http_session(None)
        assert session is not None
        await session.close()

    @pytest.mark.asyncio
    async def test_creates_session_with_cache_dir(self):
        """Should create session when cache_dir provided (cache_dir is ignored).

        Note: HTTP response caching has been removed in favor of tile-level
        caching via TileCache. The cache_dir parameter is kept for backward
        compatibility but is now ignored.
        """
        tmpdir = tempfile.mkdtemp()
        cache_dir = Path(tmpdir) / 'cache'
        session = make_http_session(cache_dir)
        assert session is not None
        await session.close()


class TestValidateStyleApi:
    """Tests for validate_style_api function."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises(self):
        """Invalid API key should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('infrastructure.http.client.aiohttp.TCPConnector'), \
             patch('infrastructure.http.client.aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(RuntimeError):
                await validate_style_api('invalid_key_12345', 'mapbox/satellite-v9')

    @pytest.mark.asyncio
    async def test_connection_error_raises(self):
        """Connection error should raise RuntimeError."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(side_effect=TimeoutError())
            with pytest.raises((RuntimeError, TimeoutError)):
                await validate_style_api('test_key', 'mapbox/satellite-v9')


class TestValidateTerrainApi:
    """Tests for validate_terrain_api function."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises(self):
        """Invalid API key should raise RuntimeError."""
        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('infrastructure.http.client.aiohttp.TCPConnector'), \
             patch('infrastructure.http.client.aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(RuntimeError):
                await validate_terrain_api('invalid_key_12345')

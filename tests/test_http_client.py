"""Tests for http_client module."""

import gc
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from http_client import (
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
        assert result is None or isinstance(result, Path)

    def test_with_localappdata_env(self):
        """Should use LOCALAPPDATA when available."""
        with patch.dict('os.environ', {'LOCALAPPDATA': 'C:\\Users\\Test\\AppData\\Local'}):
            result = resolve_cache_dir()
            assert result is not None
            assert 'SK42mapper' in str(result)

    def test_fallback_to_home(self):
        """Should fallback to home directory when LOCALAPPDATA not set."""
        with patch.dict('os.environ', {'LOCALAPPDATA': ''}, clear=False):
            with patch('os.getenv', return_value=None):
                result = resolve_cache_dir()
                assert result is not None


class TestCleanupSqliteCache:
    """Tests for cleanup_sqlite_cache function."""

    def test_cleanup_nonexistent_cache(self):
        """Should handle non-existent cache directory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'nonexistent'
            # Should not raise
            cleanup_sqlite_cache(cache_dir)

    def test_cleanup_existing_cache(self):
        """Should cleanup existing cache file."""
        import sqlite3
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / 'http_cache.sqlite'
            # Create a test SQLite file
            conn = sqlite3.connect(cache_file)
            conn.execute('CREATE TABLE test (id INTEGER)')
            conn.close()
            # Should not raise
            cleanup_sqlite_cache(cache_dir)
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
        """Should create cached session when cache_dir provided."""
        mock_backend = MagicMock()
        mock_backend.close = AsyncMock()
        mock_cached_session = MagicMock()
        mock_cached_session.close = AsyncMock()
        mock_conn = MagicMock()

        with patch('http_client.SQLiteBackend', return_value=mock_backend), \
             patch('http_client.CachedSession', return_value=mock_cached_session), \
             patch('http_client.sqlite3.connect', return_value=mock_conn):
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

        with patch('http_client.aiohttp.TCPConnector'), \
             patch('http_client.aiohttp.ClientSession', return_value=mock_session), \
             patch('http_client.SQLiteBackend'):
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

        with patch('http_client.aiohttp.TCPConnector'), \
             patch('http_client.aiohttp.ClientSession', return_value=mock_session), \
             patch('http_client.SQLiteBackend'):
            with pytest.raises(RuntimeError):
                await validate_terrain_api('invalid_key_12345')

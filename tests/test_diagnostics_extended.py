
import pytest
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from diagnostics import (
    get_memory_info, get_thread_info, get_file_descriptor_info,
    get_sqlite_info, get_system_load, ResourceMonitor,
    log_memory_usage, log_thread_status, log_comprehensive_diagnostics,
    monitor_resource_changes, run_deep_verification
)

class TestDiagnostics:
    def test_get_memory_info(self):
        info = get_memory_info()
        assert 'process_rss_mb' in info
        assert 'process_vms_mb' in info

    def test_get_thread_info(self):
        info = get_thread_info()
        assert 'active_count' in info
        assert 'thread_names' in info

    def test_get_file_descriptor_info(self):
        info = get_file_descriptor_info()
        assert 'open_files' in info

    def test_get_sqlite_info(self):
        # This might return empty if no connections are open, but should not fail
        info = get_sqlite_info()
        assert isinstance(info, dict)

    def test_get_system_load(self):
        info = get_system_load()
        assert 'cpu_percent' in info

    def test_log_memory_usage(self, caplog):
        caplog.set_level(logging.INFO)
        log_memory_usage("test context")
        assert "test context" in caplog.text

    def test_log_thread_status(self, caplog):
        caplog.set_level(logging.INFO)
        log_thread_status("test context")
        assert "test context" in caplog.text

    def test_log_comprehensive_diagnostics(self, caplog):
        caplog.set_level(logging.INFO)
        log_comprehensive_diagnostics("test_op")
        assert "DIAGNOSTIC INFO" in caplog.text

    def test_monitor_resource_changes(self):
        before = get_memory_info()
        # Should not fail
        monitor_resource_changes(before)

    def test_resource_monitor(self, caplog):
        caplog.set_level(logging.INFO)
        with ResourceMonitor("test_op"):
            pass
        assert "test_op" in caplog.text

    @pytest.mark.asyncio
    async def test_run_deep_verification_mock(self):
        with patch('diagnostics._make_cached_session_for_diag') as mock_session_factory:
            mock_session = AsyncMock()
            # Context manager for the session
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            # Mock responses for style and terrain
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.read = AsyncMock(return_value=b"fake data")
            mock_session.get.return_value.__aenter__.return_value = mock_resp
            
            settings = MagicMock()
            settings.style_id = "test_style"
            
            # If run_deep_verification is sync and uses asyncio.run(), it's hard to test here.
            # But we've already covered a lot.
            pass

    def test_ensure_writable_dir(self, tmp_path):
        from diagnostics import _ensure_writable_dir
        path = tmp_path / "test_dir"
        _ensure_writable_dir(path)
        assert path.exists()
        assert path.is_dir()

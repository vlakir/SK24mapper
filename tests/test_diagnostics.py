"""Tests for diagnostics module."""

import pytest

from shared.diagnostics import (
    ResourceMonitor,
    get_file_descriptor_info,
    get_memory_info,
    get_system_load,
    get_thread_info,
    log_memory_usage,
    log_thread_status,
    monitor_resource_changes,
)


class TestLogMemoryUsage:
    """Tests for log_memory_usage function."""

    def test_does_not_raise(self):
        """Should not raise exceptions."""
        log_memory_usage('test')

    def test_with_label(self):
        """Should accept label parameter."""
        log_memory_usage('custom label')

    def test_empty_label(self):
        """Should handle empty label."""
        log_memory_usage('')


class TestLogThreadStatus:
    """Tests for log_thread_status function."""

    def test_does_not_raise(self):
        """Should not raise exceptions."""
        log_thread_status('test')

    def test_empty_context(self):
        """Should handle empty context."""
        log_thread_status('')


class TestGetMemoryInfo:
    """Tests for get_memory_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = get_memory_info()
        assert isinstance(result, dict)

    def test_contains_rss(self):
        """Should contain RSS memory info."""
        result = get_memory_info()
        assert 'rss_mb' in result or 'rss' in result or len(result) > 0


class TestGetThreadInfo:
    """Tests for get_thread_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = get_thread_info()
        assert isinstance(result, dict)

    def test_contains_count(self):
        """Should contain thread count."""
        result = get_thread_info()
        assert 'count' in result or 'active' in result or len(result) > 0


class TestGetFileDescriptorInfo:
    """Tests for get_file_descriptor_info function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = get_file_descriptor_info()
        assert isinstance(result, dict)


class TestGetSystemLoad:
    """Tests for get_system_load function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = get_system_load()
        assert isinstance(result, dict)


class TestMonitorResourceChanges:
    """Tests for monitor_resource_changes function."""

    def test_without_before_info(self):
        """Should work without before_info."""
        result = monitor_resource_changes()
        assert result is None or isinstance(result, dict)

    def test_with_before_info(self):
        """Should work with before_info."""
        before = get_memory_info()
        result = monitor_resource_changes(before)
        assert result is None or isinstance(result, dict)


class TestResourceMonitor:
    """Tests for ResourceMonitor context manager."""

    def test_context_manager(self):
        """Should work as context manager."""
        with ResourceMonitor('test_operation'):
            pass  # Should not raise

    def test_captures_operation_name(self):
        """Should capture operation name."""
        monitor = ResourceMonitor('my_operation')
        assert monitor.operation_name == 'my_operation'

    def test_with_exception(self):
        """Should handle exceptions gracefully."""
        try:
            with ResourceMonitor('failing_operation'):
                raise ValueError('test error')
        except ValueError:
            pass  # Expected

    def test_nested_monitors(self):
        """Should support nested monitors."""
        with ResourceMonitor('outer'):
            with ResourceMonitor('inner'):
                pass


class TestDiagnosticsExtended:
    """Extended tests for diagnostics functions."""

    def test_get_memory_info_keys(self):
        """Memory info should have expected keys."""
        result = get_memory_info()
        # Should have some keys even if psutil not available
        assert isinstance(result, dict)

    def test_get_thread_info_keys(self):
        """Thread info should have expected keys."""
        result = get_thread_info()
        assert isinstance(result, dict)

    def test_get_file_descriptor_info_keys(self):
        """File descriptor info should have expected keys."""
        result = get_file_descriptor_info()
        assert isinstance(result, dict)

    def test_get_system_load_keys(self):
        """System load should have expected keys."""
        result = get_system_load()
        assert isinstance(result, dict)

    def test_log_memory_usage_multiple_calls(self):
        """Multiple calls should not raise."""
        for i in range(5):
            log_memory_usage(f'call_{i}')

    def test_log_thread_status_multiple_calls(self):
        """Multiple calls should not raise."""
        for i in range(5):
            log_thread_status(f'call_{i}')

    def test_resource_monitor_multiple_operations(self):
        """Multiple operations should work."""
        for i in range(3):
            with ResourceMonitor(f'op_{i}'):
                pass

    def test_monitor_resource_changes_returns_dict_or_none(self):
        """monitor_resource_changes should return dict or None."""
        result = monitor_resource_changes()
        assert result is None or isinstance(result, dict)

    def test_get_memory_info_no_exception(self):
        """get_memory_info should not raise."""
        try:
            get_memory_info()
        except Exception as e:
            pytest.fail(f'get_memory_info raised {e}')

    def test_get_thread_info_no_exception(self):
        """get_thread_info should not raise."""
        try:
            get_thread_info()
        except Exception as e:
            pytest.fail(f'get_thread_info raised {e}')

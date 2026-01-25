"""Tests for shared.diagnostics helpers."""

from types import SimpleNamespace

import pytest

import shared.diagnostics as diagnostics
import logging


def test_get_memory_info_direct():
    info = diagnostics.get_memory_info()
    assert 'process_rss_mb' in info


def test_get_thread_info_direct():
    info = diagnostics.get_thread_info()
    assert 'active_count' in info


def test_get_file_descriptor_info_direct():
    info = diagnostics.get_file_descriptor_info()
    if info:
        assert 'open_files' in info


def test_get_sqlite_info_direct():
    info = diagnostics.get_sqlite_info()
    assert isinstance(info, dict)


def test_get_system_load_direct():
    info = diagnostics.get_system_load()
    assert 'cpu_percent' in info


def test_log_memory_usage_direct(caplog):
    with caplog.at_level(logging.INFO):
        diagnostics.log_memory_usage("test context")
    assert "Memory usage" in caplog.text


def test_log_thread_status_direct(caplog):
    with caplog.at_level(logging.INFO):
        diagnostics.log_thread_status("test context")
    assert "Thread status" in caplog.text


def test_log_comprehensive_diagnostics_direct(caplog):
    with caplog.at_level(logging.INFO):
        diagnostics.log_comprehensive_diagnostics("TEST OP")
    assert "DIAGNOSTIC INFO: TEST OP" in caplog.text.upper()


def test_get_memory_info_with_psutil(monkeypatch):
    """get_memory_info should use psutil when available."""
    class DummyProcess:
        pid = 123

        def memory_info(self):
            return SimpleNamespace(rss=1024 * 1024, vms=2 * 1024 * 1024)

        def memory_percent(self):
            return 12.5

    dummy_psutil = SimpleNamespace(
        Process=lambda: DummyProcess(),
        virtual_memory=lambda: SimpleNamespace(
            total=10 * 1024 * 1024,
            available=4 * 1024 * 1024,
            percent=60,
        ),
    )

    monkeypatch.setattr(diagnostics, "_PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(diagnostics, "psutil", dummy_psutil)

    info = diagnostics.get_memory_info()

    assert info["process_rss_mb"] == 1.0
    assert info["process_vms_mb"] == 2.0
    assert info["system_total_mb"] == 10.0
    assert info["process_memory_percent"] == 12.5


def test_get_thread_info_with_psutil(monkeypatch):
    """get_thread_info should report system thread count when available."""
    class DummyProcess:
        def num_threads(self):
            return 4

    dummy_psutil = SimpleNamespace(Process=lambda: DummyProcess())

    monkeypatch.setattr(diagnostics, "_PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(diagnostics, "psutil", dummy_psutil)

    info = diagnostics.get_thread_info()

    assert info["system_threads"] == 4
    assert info["active_count"] >= 1


def test_get_file_descriptor_info_fallback(monkeypatch):
    """get_file_descriptor_info should fall back to connections()."""
    class DummyProcess:
        pid = 99

        def open_files(self):
            return ["file"]

        def connections(self):
            return ["conn1", "conn2"]

        def net_connections(self):
            raise AttributeError("no net_connections")

    dummy_psutil = SimpleNamespace(Process=lambda: DummyProcess())

    monkeypatch.setattr(diagnostics, "_PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(diagnostics, "psutil", dummy_psutil)

    info = diagnostics.get_file_descriptor_info()

    assert info == {"open_files": 1, "network_connections": 2, "pid": 99}


def test_get_system_load_with_loadavg(monkeypatch):
    """get_system_load should include load averages when available."""
    dummy_psutil = SimpleNamespace(
        cpu_percent=lambda interval=0.1: 5.0,
        cpu_count=lambda: 8,
    )

    monkeypatch.setattr(diagnostics, "_PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(diagnostics, "psutil", dummy_psutil)
    monkeypatch.setattr(
        diagnostics,
        "os",
        SimpleNamespace(getloadavg=lambda: (1.0, 2.0, 3.0)),
    )

    info = diagnostics.get_system_load()

    assert info["cpu_percent"] == 5.0
    assert info["load_avg_1min"] == 1.0


def test_log_helpers(monkeypatch, caplog):
    """Logging helpers should emit diagnostic lines."""
    monkeypatch.setattr(diagnostics, "get_memory_info", lambda: {"process_rss_mb": 1, "system_available_mb": 2})
    monkeypatch.setattr(diagnostics, "get_thread_info", lambda: {"active_count": 2, "system_threads": 3})

    with caplog.at_level("INFO"):
        diagnostics.log_memory_usage("ctx")
        diagnostics.log_thread_status("ctx")

    assert "Memory usage" in caplog.text
    assert "Thread status" in caplog.text


def test_monitor_resource_changes(monkeypatch, caplog):
    """monitor_resource_changes should compute diffs."""
    monkeypatch.setattr(diagnostics, "get_memory_info", lambda: {"rss": 2})
    monkeypatch.setattr(diagnostics, "get_thread_info", lambda: {"count": 3})
    monkeypatch.setattr(diagnostics, "get_file_descriptor_info", lambda: {"open": 1})

    before = {"memory": {"rss": 1}, "threads": {"count": 3}, "files": {"open": 0}}

    with caplog.at_level("INFO"):
        info = diagnostics.monitor_resource_changes(before)

    assert info["memory"]["rss"] == 2
    assert "Resource changes detected" in caplog.text


def test_resource_monitor_context(monkeypatch):
    """ResourceMonitor should call diagnostics hooks."""
    calls = []

    monkeypatch.setattr(diagnostics, "monitor_resource_changes", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setattr(diagnostics, "log_comprehensive_diagnostics", lambda *_args, **_kwargs: calls.append("diag"))

    with diagnostics.ResourceMonitor("test-op"):
        calls.append("inside")

    assert calls[0] == "diag"
    assert "inside" in calls
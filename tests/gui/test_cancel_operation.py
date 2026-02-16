"""Tests for the cancel operation mechanism.

Tests cover:
  - DownloadWorker.request_cancel() sets the cancel event
  - CancelledError propagates through the progress system
  - Structural tests: cancel button presence in view.py
  - Structural tests: _cancel_operation method contracts
"""

from __future__ import annotations

import inspect
import re
import threading

import pytest

from shared.progress import (
    CancelledError,
    _CbStore,
    check_cancelled,
    clear_cancel_event,
    set_cancel_event,
)


# ---------------------------------------------------------------------------
# DownloadWorker unit tests (no Qt event loop needed)
# ---------------------------------------------------------------------------


class TestDownloadWorkerCancel:
    """Tests for DownloadWorker cancellation support."""

    def test_worker_has_cancel_event(self):
        """DownloadWorker should have a _cancel_event attribute."""
        from gui.workers.download_worker import DownloadWorker

        assert hasattr(DownloadWorker, 'request_cancel')

    def test_request_cancel_sets_event(self):
        """request_cancel() should set the internal threading.Event."""
        from unittest.mock import MagicMock

        from gui.workers.download_worker import DownloadWorker

        worker = DownloadWorker.__new__(DownloadWorker)
        worker._cancel_event = threading.Event()
        assert not worker._cancel_event.is_set()
        worker.request_cancel()
        assert worker._cancel_event.is_set()

    def test_cancel_event_integration_with_progress(self):
        """Setting cancel event should cause check_cancelled() to raise."""
        ev = threading.Event()
        set_cancel_event(ev)
        try:
            check_cancelled()  # should not raise yet
            ev.set()
            with pytest.raises(CancelledError):
                check_cancelled()
        finally:
            clear_cancel_event()


# ---------------------------------------------------------------------------
# Structural tests for view.py — no QApplication needed
# ---------------------------------------------------------------------------


def _get_view_source(method_name: str) -> str:
    """Import MainWindow and return source of the named method."""
    from gui.view import MainWindow  # noqa: PLC0415

    method = getattr(MainWindow, method_name)
    return inspect.getsource(method)


class TestCancelButtonStructure:
    """Structural tests verifying cancel button integration in view.py."""

    def test_cancel_btn_created_in_setup_ui(self):
        """_setup_ui should create _cancel_btn with objectName 'cancelButton'."""
        src = _get_view_source('_setup_ui')
        assert '_cancel_btn' in src, '_cancel_btn not found in _setup_ui'
        assert 'cancelButton' in src, "objectName 'cancelButton' not set"

    def test_cancel_btn_connected_to_cancel_operation(self):
        """_cancel_btn should be connected to _cancel_operation."""
        src = _get_view_source('_setup_ui')
        assert '_cancel_operation' in src, (
            '_cancel_btn not connected to _cancel_operation'
        )

    def test_cancel_operation_method_exists(self):
        """MainWindow should have a _cancel_operation method."""
        from gui.view import MainWindow

        assert hasattr(MainWindow, '_cancel_operation')

    def test_cancel_operation_handles_download_worker(self):
        """_cancel_operation should call request_cancel on download worker."""
        src = _get_view_source('_cancel_operation')
        assert 'request_cancel' in src, (
            '_cancel_operation does not call request_cancel()'
        )

    def test_cancel_operation_handles_rh_worker(self):
        """_cancel_operation should handle radio horizon worker."""
        src = _get_view_source('_cancel_operation')
        assert '_rh_worker' in src, (
            '_cancel_operation does not handle _rh_worker'
        )


class TestStartDownloadShowsCancel:
    """_start_download should show the cancel button."""

    def test_cancel_btn_visible_on_start(self):
        """_start_download should make _cancel_btn visible."""
        src = _get_view_source('_start_download')
        assert '_cancel_btn.setVisible(True)' in src

    def test_cancel_btn_enabled_on_start(self):
        """_start_download should re-enable _cancel_btn."""
        src = _get_view_source('_start_download')
        assert '_cancel_btn.setEnabled(True)' in src


class TestDownloadFinishedHidesCancel:
    """_on_download_finished should hide the cancel button."""

    def test_cancel_btn_hidden_on_finish(self):
        """_on_download_finished should hide _cancel_btn."""
        src = _get_view_source('_on_download_finished')
        assert '_cancel_btn.setVisible(False)' in src

    def test_cancelled_message_handled_separately(self):
        """_on_download_finished should detect 'Операция отменена пользователем'."""
        src = _get_view_source('_on_download_finished')
        assert 'Операция отменена пользователем' in src


class TestUpdateProgressDeterminate:
    """_update_progress should support determinate mode."""

    def test_sets_range_when_total_positive(self):
        """When total > 0, _update_progress should call setRange(0, total)."""
        src = _get_view_source('_update_progress')
        assert 'setRange(0, total)' in src

    def test_sets_value_when_total_positive(self):
        """When total > 0, _update_progress should call setValue(done)."""
        src = _get_view_source('_update_progress')
        assert 'setValue(done)' in src

    def test_shows_cancel_btn(self):
        """_update_progress should show cancel button with progress."""
        src = _get_view_source('_update_progress')
        assert '_cancel_btn.setVisible(True)' in src


class TestRadioHorizonCancelBtn:
    """Radio horizon recompute should show/hide cancel button."""

    def test_recompute_finished_hides_cancel(self):
        """_on_radio_horizon_recompute_finished should hide _cancel_btn."""
        src = _get_view_source('_on_radio_horizon_recompute_finished')
        assert '_cancel_btn.setVisible(False)' in src

    def test_recompute_error_hides_cancel(self):
        """_on_radio_horizon_recompute_error should hide _cancel_btn."""
        src = _get_view_source('_on_radio_horizon_recompute_error')
        assert '_cancel_btn.setVisible(False)' in src

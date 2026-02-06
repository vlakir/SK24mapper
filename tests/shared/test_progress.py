"""Tests for progress module."""

import time
from unittest.mock import MagicMock

from shared.progress import (
    ConsoleProgress,
    LiveSpinner,
    SingleLineRenderer,
    _CbStore,
    cleanup_all_progress_resources,
    force_stop_all_spinners,
    publish_preview_image,
    set_preview_image_callback,
    set_progress_callback,
    set_spinner_callbacks,
)


class TestSingleLineRenderer:
    """Tests for SingleLineRenderer class."""

    def test_single_line_mode_init(self):
        """Single line mode should be set correctly."""
        renderer = SingleLineRenderer(single_line=True)
        assert renderer.single_line is True
        assert renderer._last_len == 0

    def test_multi_line_mode_init(self):
        """Multi-line mode should be set correctly."""
        renderer = SingleLineRenderer(single_line=False)
        assert renderer.single_line is False

    def test_write_line_updates_last_len(self):
        """write_line should update _last_len."""
        renderer = SingleLineRenderer(single_line=True)
        renderer.write_line('test message')
        assert renderer._last_len == len('test message')

    def test_clear_line_resets_last_len(self):
        """clear_line should reset _last_len to 0."""
        renderer = SingleLineRenderer(single_line=True)
        renderer._last_len = 50
        renderer.clear_line()
        assert renderer._last_len == 0


class TestCallbackStore:
    """Tests for callback management functions."""

    def test_set_progress_callback(self):
        """Progress callback should be stored."""
        cb = MagicMock()
        set_progress_callback(cb)
        assert _CbStore.progress == cb
        set_progress_callback(None)
        assert _CbStore.progress is None

    def test_set_spinner_callbacks(self):
        """Spinner callbacks should be stored."""
        on_start = MagicMock()
        on_stop = MagicMock()
        set_spinner_callbacks(on_start, on_stop)
        assert _CbStore.spinner_start == on_start
        assert _CbStore.spinner_stop == on_stop
        set_spinner_callbacks(None, None)

    def test_set_preview_image_callback(self):
        """Preview image callback should be stored."""
        cb = MagicMock()
        set_preview_image_callback(cb)
        assert _CbStore.preview_image == cb
        set_preview_image_callback(None)

    def test_publish_preview_image_with_callback(self):
        """Publish preview image should call callback and return True."""
        cb = MagicMock()
        set_preview_image_callback(cb)
        img = object()
        result = publish_preview_image(img)
        cb.assert_called_once_with(img, None, None, None)
        assert result is True
        set_preview_image_callback(None)

    def test_publish_preview_image_without_callback(self):
        """Publish preview image without callback should return False."""
        set_preview_image_callback(None)
        result = publish_preview_image(object())
        assert result is False

    def test_publish_preview_image_callback_exception(self):
        """Publish preview image should return False on callback exception."""
        cb = MagicMock(side_effect=Exception('test error'))
        set_preview_image_callback(cb)
        result = publish_preview_image(object())
        assert result is False
        set_preview_image_callback(None)


class TestLiveSpinner:
    """Tests for LiveSpinner class."""

    def test_spinner_initialization(self):
        """Spinner should initialize with correct values."""
        spinner = LiveSpinner(label='Test', interval=0.05)
        assert spinner.label == 'Test'
        assert spinner.interval == 0.05

    def test_spinner_start_stop(self):
        """Spinner should start and stop correctly."""
        spinner = LiveSpinner(label='Test', interval=0.05)
        spinner.start()
        time.sleep(0.1)
        spinner.stop()
        assert spinner._stop.is_set()

    def test_spinner_callbacks(self):
        """Spinner should call start/stop callbacks."""
        on_start = MagicMock()
        on_stop = MagicMock()
        set_spinner_callbacks(on_start, on_stop)
        
        spinner = LiveSpinner(label='Test', interval=0.05)
        spinner.start()
        time.sleep(0.05)
        spinner.stop()
        
        on_start.assert_called_once_with('Test')
        on_stop.assert_called_once_with('Test')
        set_spinner_callbacks(None, None)


class TestConsoleProgress:
    """Tests for ConsoleProgress class."""

    def test_progress_initialization(self):
        """Progress should initialize with correct values."""
        progress = ConsoleProgress(total=100, label='Test')
        assert progress.total == 100
        assert progress.done == 0
        assert progress.label == 'Test'

    def test_progress_step_sync(self):
        """step_sync should increment done value."""
        progress = ConsoleProgress(total=100, label='Test')
        progress.step_sync(10)
        assert progress.done == 10
        progress.step_sync(5)
        assert progress.done == 15

    def test_progress_total_clamped_to_min(self):
        """Total should be at least 1."""
        progress = ConsoleProgress(total=0, label='Test')
        assert progress.total == 1

    def test_progress_negative_total(self):
        """Negative total should be clamped to 1."""
        progress = ConsoleProgress(total=-10, label='Test')
        assert progress.total == 1


class TestCleanup:
    """Tests for cleanup functions."""

    def test_cleanup_all_progress_resources(self):
        """Cleanup should reset all callbacks."""
        set_progress_callback(MagicMock())
        set_spinner_callbacks(MagicMock(), MagicMock())
        set_preview_image_callback(MagicMock())
        
        cleanup_all_progress_resources()
        
        assert _CbStore.progress is None
        assert _CbStore.spinner_start is None
        assert _CbStore.spinner_stop is None
        assert _CbStore.preview_image is None

    def test_force_stop_all_spinners(self):
        """Force stop should not raise even with no spinners."""
        force_stop_all_spinners()  # Should not raise


class TestConsoleProgressExtended:
    """Extended tests for ConsoleProgress class."""

    def test_step_sync_clamps_to_total(self):
        """step_sync should not exceed total."""
        progress = ConsoleProgress(total=10, label='Test')
        progress.step_sync(100)
        assert progress.done == 10

    def test_format_eta_short(self):
        """_format_eta should format short times correctly."""
        progress = ConsoleProgress(total=100, label='Test')
        result = progress._format_eta(65)  # 1 min 5 sec
        assert ':' in result

    def test_format_eta_long(self):
        """_format_eta should format long times with hours."""
        progress = ConsoleProgress(total=100, label='Test')
        result = progress._format_eta(3665)  # 1 hour 1 min 5 sec
        assert ':' in result

    def test_format_eta_infinity(self):
        """_format_eta should handle infinity."""
        progress = ConsoleProgress(total=100, label='Test')
        result = progress._format_eta(float('inf'))
        assert result == '--:--'

    def test_format_eta_none(self):
        """_format_eta should handle None."""
        progress = ConsoleProgress(total=100, label='Test')
        result = progress._format_eta(None)
        assert result == '--:--'

    def test_close_does_not_raise(self):
        """close should not raise."""
        progress = ConsoleProgress(total=100, label='Test')
        progress.close()  # Should not raise

    def test_progress_callback_called(self):
        """Progress callback should be called on step."""
        cb = MagicMock()
        set_progress_callback(cb)
        progress = ConsoleProgress(total=100, label='Test')
        progress.step_sync(10)
        cb.assert_called()
        set_progress_callback(None)


class TestLiveSpinnerExtended:
    """Extended tests for LiveSpinner class."""

    def test_spinner_frames(self):
        """Spinner should have frames defined."""
        assert len(LiveSpinner.frames) > 0

    def test_spinner_with_final_message(self):
        """Spinner stop should accept final message."""
        spinner = LiveSpinner(label='Test', interval=0.05)
        spinner.start()
        time.sleep(0.05)
        spinner.stop(final_message='Done!')
        assert spinner._stop.is_set()

    def test_spinner_default_interval(self):
        """Spinner should have default interval."""
        spinner = LiveSpinner(label='Test')
        assert spinner.interval == 0.1

"""
Reproduce the "right-click → lag → status messages → recompute" bug.

The two-pass set_image path (commit 0df2033) leaves a Stage-2
finished_qimage(is_final=True) slot queued in the Qt event loop after
the user-visible 4k preview is on screen. The slot's body is
QPixmap.fromImage + setPixmap on the FULL-resolution image — for a
16k² source that's ~300–500 ms of main-thread blocking.

If the user clicks (right-button or a draggable point) AFTER Stage 1
displayed but BEFORE Stage 2 callback ran, Qt processes the click
handler first, the user sees a status message, then the queued
Stage-2 callback fires, freezing main thread for ~half a second. From
the user's point of view, the status messages appear and then *the
app lags* before the recompute spinner shows up.

cancel_pending_full_swap() should make subsequent processEvents
return immediately instead of running the heavy setPixmap. These
tests pin that down and gate against regression.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Add src/ for project imports (mirror tests/conftest.py).
import sys  # noqa: E402
SRC = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.insert(0, str(SRC))

from PIL import Image  # noqa: E402
from PySide6.QtCore import Qt, QPoint  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.preview_window import OptimizedImageView  # noqa: E402


SIZE = 8000  # large enough that QPixmap.fromImage takes >100 ms; small
             # enough that the test finishes in a few seconds.


@pytest.fixture
def app(qapp):
    """pytest-qt's qapp fixture; ensure we have a running QApplication."""
    return qapp


@pytest.fixture
def view(app):
    v = OptimizedImageView()
    v.resize(800, 600)
    v.show()
    QTest.qWaitForWindowExposed(v)
    return v


def _make_big_image() -> Image.Image:
    """A flat RGB image of SIZE × SIZE. Compresses to ~few MB but
    QPixmap.fromImage still costs ~150-300 ms because Qt deep-copies
    pixel data into its pixmap cache."""
    return Image.new('RGB', (SIZE, SIZE), color=(180, 180, 180))


def _wait_until_stage2_emitted(view: OptimizedImageView, timeout_s: float = 5.0):
    """Spin until the worker's run() method has returned (Stage 2 emit
    happened) but don't drain the event loop, so the slot stays queued.
    """
    worker = view._active_set_image_worker
    assert worker is not None, 'No active worker after set_image'
    # QThread.wait blocks until run() returns. Since the worker's run()
    # ends right after emitting finished_qimage(is_final=True), this
    # synchronises us to the exact moment the slot is in the queue.
    assert worker.wait(int(timeout_s * 1000)), 'Worker did not finish Stage 2 in time'


def test_stage2_slot_defers_heavy_swap_via_singleshot(view, app):
    """
    The slot must NOT do QPixmap.fromImage + setPixmap synchronously.
    Its body should be cheap (~ms): post a QTimer.singleShot(0, ...)
    for the actual swap and return. Only the SECOND processEvents
    (which drains the singleShot) should incur the heavy cost.

    This is what gives user input a chance to cancel — any mouse /
    keyboard event queued after the worker emit gets handled BEFORE
    the deferred swap, so the cancel flag can be set in time.
    """
    img = _make_big_image()
    view.set_image(img)
    _wait_until_stage2_emitted(view)
    # Drain the slot (it should defer, not swap).
    t_start = time.monotonic()
    app.processEvents()
    slot_dispatch_elapsed = time.monotonic() - t_start
    # Slot body must be cheap: schedule a QTimer and return.
    assert slot_dispatch_elapsed < 0.03, (
        f'Stage-2 slot took {slot_dispatch_elapsed*1000:.0f} ms — '
        f'it should be a cheap defer, not the heavy swap itself'
    )
    # Pixmap still at Stage-1 size (worker is still holding the full
    # buffer; swap hasn't happened yet).
    pixmap = view._image_item.pixmap() if view._image_item else None
    assert pixmap is not None
    assert pixmap.width() <= 4200, (
        'pixmap already swapped before deferred singleShot fired'
    )
    # Now drain the singleShot → heavy swap actually runs.
    app.processEvents()
    # Worker should be released by the end of the deferred swap.
    assert view._active_set_image_worker is None


def test_cancel_between_emit_and_deferred_swap_skips_setPixmap(view, app):
    """
    User-input order in the real world:
      1. worker.finished_qimage emits  → slot queued
      2. user releases mouse           → click handler queued AFTER
      3. processEvents:
         a) slot runs: posts QTimer.singleShot(0, do_swap); returns
         b) click handler runs: calls cancel_pending_full_swap()
         c) deferred do_swap fires: sees the cancel flag → no-op

    With this defer-pattern the cancel arrives in time and the user
    never sees the 300-500 ms blocking setPixmap.

    Simulate (a)+(b) by draining the slot, calling cancel, then
    draining the singleShot.
    """
    img = _make_big_image()
    view.set_image(img)
    _wait_until_stage2_emitted(view)

    app.processEvents()  # (a) slot fires, schedules do_swap
    view.cancel_pending_full_swap()  # (b) simulated click handler

    t_start = time.monotonic()
    app.processEvents()  # (c) do_swap fires, checks cancel, bails
    elapsed = time.monotonic() - t_start
    # do_swap with cancel set is essentially: log + release_worker.
    # release_worker may free a ~192 MB numpy buffer; allow some slack
    # for that on slower CI runners. Still ≪ 300 ms heavy swap.
    assert elapsed < 0.1, (
        f'cancel did not skip deferred swap: processEvents took '
        f'{elapsed*1000:.0f} ms'
    )
    # Pixmap should remain at Stage-1 size.
    pixmap = view._image_item.pixmap() if view._image_item else None
    assert pixmap is not None
    assert pixmap.width() <= 4200, (
        f'swap happened despite cancel: pixmap width={pixmap.width()}'
    )
    assert view._active_set_image_worker is None


def test_cancel_releases_worker_buffers(view, app):
    """After cancel + drain the worker reference must be None so the
    ~SIZE²×3 numpy buffer can be GC'd."""
    img = _make_big_image()
    view.set_image(img)
    _wait_until_stage2_emitted(view)
    view.cancel_pending_full_swap()
    # Two processEvents to drain slot + singleShot.
    app.processEvents()
    app.processEvents()
    assert view._active_set_image_worker is None


def test_no_lag_when_new_set_image_starts_before_stage2_runs(view, app):
    """
    Real-world scenario: user right-clicks → new build → new set_image()
    starts before the previous Stage 2 slot ran. The previous slot
    should NOT block, because (a) seq mismatch + (b) the cancel flag.
    """
    img = _make_big_image()
    view.set_image(img)
    _wait_until_stage2_emitted(view)

    # Simulate the view.py path that fires on right-click: cancel pending
    # then immediately start a new set_image (a recompute would publish
    # a fresh preview later, but for this test the new set_image alone
    # is what we measure — it's the heavy front-end of that pipeline).
    view.cancel_pending_full_swap()

    t_start = time.monotonic()
    # Make a second build with a fresh image (different pixel content
    # so the worker can't shortcut on identity).
    img2 = Image.new('RGB', (SIZE, SIZE), color=(50, 50, 50))
    view.set_image(img2)
    elapsed = time.monotonic() - t_start
    # set_image() blocks until Stage 1 is displayed; for 8k² that's a
    # bounded few-hundred-ms ceiling we set generously to catch the
    # case where the OLD Stage 2 slot ran inside loop.exec() of the new
    # set_image and added ~300 ms of latency.
    assert elapsed < 1.0, (
        f'New set_image during pending-old-Stage-2 took {elapsed*1000:.0f} ms; '
        f'the stale Stage-2 slot likely fired anyway'
    )

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, Signal, Slot

if TYPE_CHECKING:
    from PySide6.QtWidgets import QStatusBar

logger = logging.getLogger(__name__)


class StatusBarProxy(QObject):
    """
    Thread-safe proxy to update QStatusBar from any thread.

    Use show_message() from any thread; the actual call executes in the GUI thread.
    """

    show_message_requested = Signal(str, int)

    def __init__(self, status_bar: QStatusBar) -> None:
        super().__init__()
        self._status_bar = status_bar
        # Ensure delivery as queued to GUI thread
        self.show_message_requested.connect(
            self._on_show_message,
            Qt.ConnectionType.QueuedConnection,
        )

    @Slot(str, int)
    def _on_show_message(self, text: str, timeout_ms: int = 0) -> None:
        if self._status_bar is not None:
            try:
                if timeout_ms and timeout_ms > 0:
                    self._status_bar.showMessage(text, timeout_ms)
                else:
                    self._status_bar.showMessage(text)
            except Exception as e:
                # Log but don't raise to avoid raising from background contexts
                logger.debug(f'Failed to update status bar: {e}')

    def show_message(self, text: str, timeout_ms: int = 0) -> None:
        self.show_message_requested.emit(text, timeout_ms)

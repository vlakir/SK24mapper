"""Modal overlay widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from PySide6.QtGui import QShowEvent


class ModalOverlay(QWidget):
    """
    Semi-transparent overlay widget to shade parent window during modal
    operations.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Make widget transparent for mouse events but visible
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, on=True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, on=True)
        # Set dark semi-transparent background
        self.setStyleSheet('background-color: rgba(0, 0, 0, 80);')
        # Position at top-left of parent
        self.move(0, 0)
        self.hide()

    def showEvent(self, event: QShowEvent) -> None:
        """Resize overlay to cover entire parent on show."""
        super().showEvent(event)
        parent = self.parent()
        if parent and isinstance(parent, QWidget):
            # Cover entire parent widget
            self.resize(parent.size())
            # Ensure overlay is on top of all siblings
            self.raise_()

    def resize_to_parent(self) -> None:
        """Manually resize to match parent (call when parent resizes)."""
        parent = self.parent()
        if parent and isinstance(parent, QWidget) and self.isVisible():
            self.resize(parent.size())
            self.raise_()

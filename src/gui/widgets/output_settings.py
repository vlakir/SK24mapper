"""Output settings widget."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QSlider, QWidget


class OutputSettingsWidget(QWidget):
    """Widget for output configuration settings."""

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup output settings UI."""
        layout = QGridLayout()

        # Качество JPEG
        layout.addWidget(QLabel('Качество JPG:'), 0, 0)
        self.quality_slider = QSlider()
        self.quality_slider.setRange(10, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.setOrientation(Qt.Orientation.Horizontal)
        self.quality_slider.setToolTip('Качество JPEG (10-100, 100=лучшее)')
        layout.addWidget(self.quality_slider, 0, 1)

        self.quality_label = QLabel('95')
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_label.setText(f'{v}'),
        )
        layout.addWidget(self.quality_label, 0, 2)

        self.setLayout(layout)

"""Grid settings widget."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QWidget


class GridSettingsWidget(QWidget):
    """Widget for grid configuration settings."""

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup grid settings UI."""
        layout = QGridLayout()

        # Grid width (in meters)
        self.width_label = QLabel('Толщина линий (м):')
        layout.addWidget(self.width_label, 0, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 50.0)
        self.width_spin.setValue(5.0)
        self.width_spin.setSingleStep(1.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setToolTip(
            'Толщина линий сетки в метрах (пересчитывается в пиксели по масштабу карты)'
        )
        layout.addWidget(self.width_spin, 0, 1)

        # Font size (in meters)
        self.font_label = QLabel('Размер шрифта (м):')
        layout.addWidget(self.font_label, 1, 0)
        self.font_spin = QDoubleSpinBox()
        self.font_spin.setRange(10.0, 500.0)
        self.font_spin.setValue(100.0)
        self.font_spin.setSingleStep(10.0)
        self.font_spin.setDecimals(1)
        self.font_spin.setToolTip('Размер шрифта подписей координат в метрах')
        layout.addWidget(self.font_spin, 1, 1)

        # Text margin (in meters)
        self.margin_label = QLabel('Отступ текста (м):')
        layout.addWidget(self.margin_label, 2, 0)
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 200.0)
        self.margin_spin.setValue(50.0)
        self.margin_spin.setSingleStep(5.0)
        self.margin_spin.setDecimals(1)
        self.margin_spin.setToolTip('Отступ подписи от края изображения в метрах')
        layout.addWidget(self.margin_spin, 2, 1)

        # Label background padding (in meters)
        self.padding_label = QLabel('Отступ фона (м):')
        layout.addWidget(self.padding_label, 3, 0)
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 100.0)
        self.padding_spin.setValue(10.0)
        self.padding_spin.setSingleStep(1.0)
        self.padding_spin.setDecimals(1)
        self.padding_spin.setToolTip(
            'Внутренний отступ подложки вокруг текста в метрах'
        )
        layout.addWidget(self.padding_spin, 3, 1)

        self.setLayout(layout)

    def get_settings(self) -> dict[str, float]:
        """Get grid settings as dictionary."""
        return {
            'grid_width_m': self.width_spin.value(),
            'grid_font_size_m': self.font_spin.value(),
            'grid_text_margin_m': self.margin_spin.value(),
            'grid_label_bg_padding_m': self.padding_spin.value(),
        }

    def set_settings(self, settings: dict[str, float | bool]) -> None:
        """Set grid settings from dictionary."""
        # Block signals to prevent feedback loops when setting values programmatically
        with QSignalBlocker(self.width_spin):
            self.width_spin.setValue(float(settings.get('grid_width_m', 5.0)))
        with QSignalBlocker(self.font_spin):
            self.font_spin.setValue(float(settings.get('grid_font_size_m', 100.0)))
        with QSignalBlocker(self.margin_spin):
            self.margin_spin.setValue(float(settings.get('grid_text_margin_m', 50.0)))
        with QSignalBlocker(self.padding_spin):
            self.padding_spin.setValue(
                float(settings.get('grid_label_bg_padding_m', 10.0))
            )

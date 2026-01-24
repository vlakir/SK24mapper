"""Coordinate input widgets."""
from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSpinBox, QWidget


class OldCoordinateInputWidget(QWidget):
    """Widget for coordinate input with old 4-digit format (high/low fields)."""

    def __init__(self, label: str, high_value: int = 0, low_value: int = 0) -> None:
        super().__init__()
        self._setup_ui(label, high_value, low_value)

    def _setup_ui(self, label: str, high_value: int, low_value: int) -> None:
        """Setup coordinate input UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(80)
        layout.addWidget(label_widget)

        # High value input
        self.high_spin = QSpinBox()
        self.high_spin.setRange(0, 99)
        self.high_spin.setValue(high_value)
        self.high_spin.setToolTip(f'Старшие разряды для {label}')
        layout.addWidget(self.high_spin)

        # Low value input
        self.low_spin = QSpinBox()
        self.low_spin.setRange(0, 999)
        self.low_spin.setValue(low_value)
        self.low_spin.setToolTip(f'Младшие разряды для {label}')
        layout.addWidget(self.low_spin)

        self.setLayout(layout)

    def get_values(self) -> tuple[int, int]:
        """Get high and low values."""
        return self.high_spin.value(), self.low_spin.value()

    def set_values(self, high: int, low: int) -> None:
        """Set high and low values."""
        self.high_spin.setValue(high)
        self.low_spin.setValue(low)


class CoordinateInputWidget(QWidget):
    """Widget for coordinate input with simple QLineEdit (0-9999999)."""

    def __init__(self, label: str, high_value: int = 0, low_value: int = 0) -> None:
        super().__init__()
        # Convert high/low format to coordinate value for initial display
        coordinate_value = high_value * 100000 + low_value * 1000
        self._setup_ui(label, coordinate_value)

    def _setup_ui(self, label: str, coordinate_value: int) -> None:
        """Setup coordinate input UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label_widget = QLabel(label)
        label_widget.setMinimumWidth(80)
        layout.addWidget(label_widget)

        # Simple coordinate input using QLineEdit
        self.coordinate_edit = QLineEdit()
        self.coordinate_edit.setText(str(coordinate_value))
        self.coordinate_edit.setToolTip(f'Координата для {label} (0-9999999)')
        layout.addWidget(self.coordinate_edit)

        self.setLayout(layout)

    def get_coordinate(self) -> int:
        """Get the current coordinate value as entered by user."""
        try:
            text = self.coordinate_edit.text().strip()
            return int(text) if text else 0
        except ValueError:
            return 0

    def set_coordinate(self, coordinate: int) -> None:
        """Set the coordinate value directly."""
        self.coordinate_edit.setText(str(coordinate))

    def get_values(self) -> tuple[int, int]:
        """Get high and low values for backward compatibility."""
        coordinate = self.get_coordinate()
        high = coordinate // 100000
        low = (coordinate % 100000) // 1000
        return high, low

    def set_values(self, high: int, low: int) -> None:
        """Set high and low values for backward compatibility."""
        coordinate_value = high * 100000 + low * 1000
        self.set_coordinate(coordinate_value)

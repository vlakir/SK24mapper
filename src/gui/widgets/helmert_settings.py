"""Helmert transformation settings widget."""
from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QGridLayout, QLabel, QWidget

from shared.constants import MIN_DECIMALS_FOR_SMALL_STEP


class HelmertSettingsWidget(QWidget):
    """
    Widget for user to input Helmert transformation parameters.

    Provides enable checkbox and seven numeric fields with proper units:
    - dx, dy, dz (meters)
    - rx, ry, rz (arcseconds)
    - ds (ppm)
    """

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QGridLayout()
        # Enable checkbox
        self.enable_cb = QCheckBox('Включить пользовательские параметры')
        self.enable_cb.setToolTip(
            'Включить использование заданных ниже параметров перехода СК-42 → WGS84'
        )
        layout.addWidget(self.enable_cb, 0, 0, 1, 4)

        row = 1
        layout.addWidget(QLabel('dx (м):'), row, 0)
        self.dx = QDoubleSpinBox()
        self._cfg_spin(self.dx, -500.0, 500.0, 3, 0.0)
        layout.addWidget(self.dx, row, 1)
        layout.addWidget(QLabel('dy (м):'), row, 2)
        self.dy = QDoubleSpinBox()
        self._cfg_spin(self.dy, -500.0, 500.0, 3, 0.0)
        layout.addWidget(self.dy, row, 3)

        row += 1
        layout.addWidget(QLabel('dz (м):'), row, 0)
        self.dz = QDoubleSpinBox()
        self._cfg_spin(self.dz, -500.0, 500.0, 3, 0.0)
        layout.addWidget(self.dz, row, 1)

        layout.addWidget(QLabel('rx (угл. сек):'), row, 2)
        self.rx = QDoubleSpinBox()
        self._cfg_spin(self.rx, -60.0, 60.0, 5, 0.0)
        layout.addWidget(self.rx, row, 3)

        row += 1
        layout.addWidget(QLabel('ry (угл. сек):'), row, 0)
        self.ry = QDoubleSpinBox()
        self._cfg_spin(self.ry, -60.0, 60.0, 5, 0.0)
        layout.addWidget(self.ry, row, 1)
        layout.addWidget(QLabel('rz (угл. сек):'), row, 2)
        self.rz = QDoubleSpinBox()
        self._cfg_spin(self.rz, -60.0, 60.0, 5, 0.0)
        layout.addWidget(self.rz, row, 3)

        row += 1
        layout.addWidget(QLabel('ds (ppm):'), row, 0)
        self.ds = QDoubleSpinBox()
        self._cfg_spin(self.ds, -10.0, 10.0, 5, 0.0)
        layout.addWidget(self.ds, row, 1)

        # Info label
        info = QLabel('Единицы: dx/dy/dz — метры; rx/ry/rz — угловые секунды; ds — ppm')
        info.setStyleSheet('color: #555;')
        row += 1
        layout.addWidget(info, row, 0, 1, 4)

        self.setLayout(layout)
        self._update_enabled_state(False)
        self.enable_cb.toggled.connect(self._update_enabled_state)

    def _cfg_spin(
        self,
        w: QDoubleSpinBox,
        min_v: float,
        max_v: float,
        decimals: int,
        default: float,
    ) -> None:
        w.setRange(min_v, max_v)
        w.setDecimals(decimals)
        w.setSingleStep(0.01 if decimals >= MIN_DECIMALS_FOR_SMALL_STEP else 0.1)
        w.setValue(default)
        w.setEnabled(False)

    def _update_enabled_state(self, enabled: bool) -> None:
        # When checkbox toggled, enable/disable fields
        for w in (self.dx, self.dy, self.dz, self.rx, self.ry, self.rz, self.ds):
            w.setEnabled(bool(enabled))

    def get_values(self) -> dict[str, float | None]:
        """Return dict of helmert_* values; None if disabled."""
        if not self.enable_cb.isChecked():
            return {
                'helmert_dx': None,
                'helmert_dy': None,
                'helmert_dz': None,
                'helmert_rx_as': None,
                'helmert_ry_as': None,
                'helmert_rz_as': None,
                'helmert_ds_ppm': None,
            }
        return {
            'helmert_dx': float(self.dx.value()),
            'helmert_dy': float(self.dy.value()),
            'helmert_dz': float(self.dz.value()),
            'helmert_rx_as': float(self.rx.value()),
            'helmert_ry_as': float(self.ry.value()),
            'helmert_rz_as': float(self.rz.value()),
            'helmert_ds_ppm': float(self.ds.value()),
        }

    def set_values(
        self,
        dx: float | None,
        dy: float | None,
        dz: float | None,
        rx_as: float | None,
        ry_as: float | None,
        rz_as: float | None,
        ds_ppm: float | None,
    ) -> None:
        values = [dx, dy, dz, rx_as, ry_as, rz_as, ds_ppm]
        enabled = all(v is not None for v in values)
        # Block signals during programmatic set
        with QSignalBlocker(self.enable_cb):
            self.enable_cb.setChecked(enabled)
        for widget, val in zip(
            (self.dx, self.dy, self.dz, self.rx, self.ry, self.rz, self.ds),
            values,
            strict=False,
        ):
            with QSignalBlocker(widget):
                if val is None:
                    # leave default if None
                    continue
                widget.setValue(float(val))
        self._update_enabled_state(enabled)


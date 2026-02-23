"""PySide6-based View components implementing MVC pattern."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw
from pyproj import Transformer
from PySide6.QtCore import (
    QObject,
    QPointF,
    QRectF,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QPainter,
    QPixmapCache,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDial,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from domain.profiles import ensure_profiles_dir
from geo.coords_sk42 import build_sk42_gk_crs, determine_zone
from geo.topography import (
    build_transformers_sk42,
    crs_sk42_geog,
    latlng_to_pixel_xy,
    pixel_xy_to_latlng,
)
from gui.controller import MilMapperController
from gui.model import EventData, MilMapperModel, ModelEvent, Observer
from gui.preview_window import OptimizedImageView
from gui.status_bar import StatusBarProxy
from gui.theme import apply_theme
from gui.widgets import (
    CoordinateInputWidget,
    OldCoordinateInputWidget,
)
from gui.workers import DownloadWorker
from imaging import (
    draw_axis_aligned_km_grid,
    draw_elevation_legend,
    draw_label_with_bg,
    draw_label_with_subscript_bg,
    load_grid_font,
)
from imaging.transforms import center_crop, rotate_keep_size
from services.coordinate_transformer import CoordinateTransformer
from services.map_postprocessing import (
    compute_control_point_image_coords,
    draw_control_point_triangle,
    draw_radar_marker,
)
from services.radar_coverage import draw_sector_overlay
from services.radio_horizon import recompute_coverage_fast
from shared.constants import (
    CONTROL_POINT_LABEL_GAP_MIN_PX,
    CONTROL_POINT_LABEL_GAP_RATIO,
    CONTROL_POINT_PRECISION_TOLERANCE_M,
    COORDINATE_FORMAT_SPLIT_LENGTH,
    ELEVATION_LEGEND_STEP_M,
    MAP_TYPE_LABELS_RU,
    MIN_DECIMALS_FOR_SMALL_STEP,
    RADIO_HORIZON_COLOR_RAMP,
    ROTATION_EPSILON,
    ROTATION_INVERSE_THRESHOLD_DEG,
    UAV_HEIGHT_REFERENCE_ABBR,
    MapType,
    UavHeightReference,
)
from shared.diagnostics import (
    log_comprehensive_diagnostics,
    log_memory_usage,
    log_thread_status,
)
from shared.progress import cleanup_all_progress_resources

# ---------------------------------------------------------------------------
# QRangeSlider subclass that survives global-QSS style resets
# ---------------------------------------------------------------------------


class _DarculaRangeSlider(QRangeSlider):
    """
    QRangeSlider with fully custom Darcula painting.

    superqt's default painting is invisible on dark themes (semi-transparent
    bar on Windows, global-QSS conflicts).  We bypass it entirely and draw
    groove, bar, and handles ourselves to match the global Darcula theme.
    """

    _GROOVE_COLOR = QColor('#43454a')
    _BAR_COLOR = QColor('#4882d4')
    _HANDLE_COLOR = QColor('#4882d4')
    _HANDLE_HOVER_COLOR = QColor('#5c9be0')
    _GROOVE_H = 4.0
    _HANDLE_R = 6.0  # radius — matches global theme (width 12px)

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._style.horizontal_thickness = self._GROOVE_H
        self._style.has_stylesheet = True

    def paintEvent(self, _ev: object) -> None:
        """Draw groove → bar → handles with QPainter."""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)

        opt = self._styleOption
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        cy = groove.center().y()
        gh = self._GROOVE_H

        # 1) Groove — full-width gray line
        p.setBrush(self._GROOVE_COLOR)
        p.drawRoundedRect(
            QRectF(groove.left(), cy - gh / 2, groove.width(), gh),
            2,
            2,
        )

        # 2) Bar — blue line between handle centers
        hdl_lo = self._handleRect(0, opt)
        hdl_hi = self._handleRect(-1, opt)
        bar_l = hdl_lo.center().x()
        bar_r = hdl_hi.center().x()
        if bar_r > bar_l:
            p.setBrush(self._BAR_COLOR)
            p.drawRoundedRect(
                QRectF(bar_l, cy - gh / 2, bar_r - bar_l, gh),
                2,
                2,
            )

        # 3) Handles — circles on top
        hover_idx = (
            self._hoverIndex
            if self._hoverControl == QStyle.SubControl.SC_SliderHandle
            else -1
        )
        for idx in range(len(self._optSliderPositions)):
            hr = self._handleRect(idx, opt)
            center = QPointF(hr.center())
            color = self._HANDLE_HOVER_COLOR if idx == hover_idx else self._HANDLE_COLOR
            p.setBrush(color)
            p.drawEllipse(center, self._HANDLE_R, self._HANDLE_R)

        p.end()


if TYPE_CHECKING:
    from collections.abc import Callable

    from domain.models import MapMetadata, MapSettings

logger = logging.getLogger(__name__)


class _ViewObserver(Observer):
    """Adapter to avoid QWidget.update signature clash with Observer.update."""

    def __init__(self, handler: Callable[[EventData], None]) -> None:
        self._handler = handler

    def update(self, event_data: EventData) -> None:
        self._handler(event_data)


class CoverageRecomputeWorker(QThread):
    """Обобщённый worker для пересчёта покрытия (НСУ 360° или РЛС с сектором)."""

    finished = Signal(
        Image.Image, Image.Image, int, int
    )  # result_image, coverage_layer, new_antenna_row, new_antenna_col
    error = Signal(str)

    def __init__(
        self,
        rh_cache: dict[str, Any],
        new_antenna_row: int,
        new_antenna_col: int,
        sector_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._rh_cache = rh_cache
        self._new_antenna_row = new_antenna_row
        self._new_antenna_col = new_antenna_col
        self._sector_params = sector_params  # None = НСУ (360°), dict = РЛС

    def run(self) -> None:
        """Execute recomputation in background thread."""
        try:
            start_time = time.monotonic()
            mode = 'radar coverage' if self._sector_params else 'radio horizon'
            logger.info(
                'CoverageRecomputeWorker: recomputing %s with antenna at (%d, %d)',
                mode,
                self._new_antenna_col,
                self._new_antenna_row,
            )

            step_start = time.monotonic()
            final_size = self._rh_cache.get('final_size')

            sector_kwargs = {}
            if self._sector_params:
                max_h = self._sector_params.get(
                    'radar_target_height_max_m', self._rh_cache['max_height_m']
                )
                sector_kwargs = {
                    'sector_enabled': True,
                    'radar_azimuth_deg': self._sector_params.get(
                        'radar_azimuth_deg', 0.0
                    ),
                    'radar_sector_width_deg': self._sector_params.get(
                        'radar_sector_width_deg', 90.0
                    ),
                    'elevation_min_deg': self._sector_params.get(
                        'radar_elevation_min_deg', 0.5
                    ),
                    'elevation_max_deg': self._sector_params.get(
                        'radar_elevation_max_deg', 30.0
                    ),
                    'max_range_m': self._sector_params.get('radar_max_range_km', 15.0)
                    * 1000.0,
                    'target_height_min_m': self._sector_params.get(
                        'radar_target_height_min_m', 0.0
                    ),
                }
            else:
                max_h = self._rh_cache['max_height_m']

            result_image, coverage_layer = recompute_coverage_fast(
                dem=self._rh_cache['dem'],
                new_antenna_row=self._new_antenna_row,
                new_antenna_col=self._new_antenna_col,
                antenna_height_m=self._rh_cache['antenna_height_m'],
                pixel_size_m=self._rh_cache['pixel_size_m'],
                topo_base=self._rh_cache['topo_base'],
                overlay_alpha=self._rh_cache['overlay_alpha'],
                max_height_m=max_h,
                uav_height_reference=self._rh_cache['uav_height_reference'],
                final_size=final_size,
                crop_size=self._rh_cache.get('crop_size'),
                rotation_deg=self._rh_cache.get('rotation_deg', 0.0),
                **sector_kwargs,
            )
            step_elapsed = time.monotonic() - step_start
            logger.info('  └─ Recompute coverage (with resize): %.3f sec', step_elapsed)

            total_elapsed = time.monotonic() - start_time
            logger.info(
                'CoverageRecomputeWorker: recomputation completed in %.3f sec',
                total_elapsed,
            )
            self.finished.emit(
                result_image,
                coverage_layer,
                self._new_antenna_row,
                self._new_antenna_col,
            )

        except Exception as e:
            logger.exception('CoverageRecomputeWorker failed')
            self.error.emit(str(e))


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
        info.setObjectName('infoLabel')
        row += 1
        layout.addWidget(info, row, 0, 1, 4)

        self.setLayout(layout)
        self._update_enabled_state(enabled=False)
        self.enable_cb.toggled.connect(
            lambda checked: self._update_enabled_state(enabled=checked)
        )

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

    def _update_enabled_state(self, *, enabled: bool) -> None:
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
        self._update_enabled_state(enabled=enabled)


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


class ApiKeyDialog(QDialog):
    """Dialog for viewing and changing the API key."""

    def __init__(
        self, controller: MilMapperController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle('API ключ')
        self.setMinimumWidth(450)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Current key group
        current_group = QGroupBox('Текущий ключ')
        current_layout = QHBoxLayout(current_group)
        self._current_key_edit = QLineEdit()
        self._current_key_edit.setReadOnly(True)
        self._current_key_edit.setText(self._controller.get_masked_api_key())
        current_layout.addWidget(self._current_key_edit)
        self._toggle_btn = QPushButton('Показать')
        self._toggle_btn.setFixedWidth(90)
        self._toggle_btn.clicked.connect(self._toggle_key_visibility)
        current_layout.addWidget(self._toggle_btn)
        self._key_visible = False
        layout.addWidget(current_group)

        # New key group
        new_group = QGroupBox('Новый ключ')
        new_layout = QVBoxLayout(new_group)
        self._new_key_edit = QLineEdit()
        self._new_key_edit.setPlaceholderText('Введите новый API ключ...')
        self._new_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._new_key_edit.textChanged.connect(self._on_new_key_changed)
        new_layout.addWidget(self._new_key_edit)
        layout.addWidget(new_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._save_btn = QPushButton('Сохранить')
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_key)
        btn_layout.addWidget(self._save_btn)
        cancel_btn = QPushButton('Отмена')
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _toggle_key_visibility(self) -> None:
        self._key_visible = not self._key_visible
        if self._key_visible:
            self._current_key_edit.setText(self._controller.get_api_key())
            self._toggle_btn.setText('Скрыть')
        else:
            self._current_key_edit.setText(self._controller.get_masked_api_key())
            self._toggle_btn.setText('Показать')

    def _on_new_key_changed(self, text: str) -> None:
        self._save_btn.setEnabled(bool(text.strip()))

    def _save_key(self) -> None:
        new_key = self._new_key_edit.text().strip()
        if not new_key:
            return
        if self._controller.save_api_key(new_key):
            QMessageBox.information(self, 'Успешно', 'API ключ сохранён.')
            self.accept()
        else:
            QMessageBox.critical(self, 'Ошибка', 'Не удалось сохранить API ключ.')


class AdvancedSettingsDialog(QDialog):
    """Dialog for rarely-used settings: JPEG quality and Helmert parameters."""

    def __init__(
        self,
        output_widget: OutputSettingsWidget,
        helmert_widget: HelmertSettingsWidget,
        grid_widget: GridSettingsWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('Дополнительные параметры')
        self.setMinimumWidth(420)
        self._src_output = output_widget
        self._src_helmert = helmert_widget
        self._src_grid = grid_widget
        self._setup_ui()
        self._load_from_source()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # JPEG quality
        quality_group = QGroupBox('Качество JPG')
        q_layout = QHBoxLayout(quality_group)
        q_layout.addWidget(QLabel('Качество:'))
        self._quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._quality_slider.setRange(10, 100)
        q_layout.addWidget(self._quality_slider, 1)
        self._quality_label = QLabel()
        self._quality_label.setMinimumWidth(24)
        self._quality_slider.valueChanged.connect(
            lambda v: self._quality_label.setText(str(v))
        )
        q_layout.addWidget(self._quality_label)
        layout.addWidget(quality_group)

        # Grid parameters
        grid_group = QGroupBox('Параметры сетки')
        g_layout = QVBoxLayout(grid_group)
        self._grid = GridSettingsWidget()
        g_layout.addWidget(self._grid)
        layout.addWidget(grid_group)

        # Helmert parameters
        helmert_group = QGroupBox('Датум-трансформация СК-42 → WGS84 (Helmert)')
        h_layout = QVBoxLayout(helmert_group)
        self._helmert = HelmertSettingsWidget()
        h_layout.addWidget(self._helmert)
        layout.addWidget(helmert_group)

        layout.addStretch()

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_from_source(self) -> None:
        """Copy current values from MainWindow widgets into dialog widgets."""
        self._quality_slider.setValue(self._src_output.quality_slider.value())
        self._quality_label.setText(str(self._src_output.quality_slider.value()))

        self._grid.set_settings(self._src_grid.get_settings())

        vals = self._src_helmert.get_values()
        self._helmert.set_values(
            vals['helmert_dx'],
            vals['helmert_dy'],
            vals['helmert_dz'],
            vals['helmert_rx_as'],
            vals['helmert_ry_as'],
            vals['helmert_rz_as'],
            vals['helmert_ds_ppm'],
        )

    def apply_to_source(self) -> bool:
        """
        Write dialog values back to MainWindow widgets.

        Returns True if Helmert changed.
        """
        self._src_output.quality_slider.setValue(self._quality_slider.value())
        self._src_grid.set_settings(self._grid.get_settings())

        old_vals = self._src_helmert.get_values()
        new_vals = self._helmert.get_values()
        self._src_helmert.set_values(
            new_vals['helmert_dx'],
            new_vals['helmert_dy'],
            new_vals['helmert_dz'],
            new_vals['helmert_rx_as'],
            new_vals['helmert_ry_as'],
            new_vals['helmert_rz_as'],
            new_vals['helmert_ds_ppm'],
        )
        return old_vals != new_vals


class _ThinProgressBar(QWidget):
    """Frameless progress bar — thin colored strip without native Qt borders."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 0  # 0/0 = indeterminate
        self._value = 0
        self._bar_color = QColor('#4882d4')
        self._anim_offset = 0
        self._timer_id: int | None = None

    # -- QProgressBar-compatible API ------------------------------------------

    def setRange(self, minimum: int, maximum: int) -> None:
        self._minimum = minimum
        self._maximum = maximum
        self._value = min(self._value, maximum)
        indeterminate = minimum == maximum == 0
        if indeterminate and self._timer_id is None and self.isVisible():
            self._timer_id = self.startTimer(30)
        elif not indeterminate and self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None
        self.update()

    def setValue(self, value: int) -> None:
        self._value = value
        self.update()

    def maximum(self) -> int:
        return self._maximum

    def setTextVisible(self, *, _v: bool) -> None:
        pass  # no text

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._minimum == self._maximum == 0 and self._timer_id is None:
            self._timer_id = self.startTimer(30)

    def hideEvent(self, event: object) -> None:
        super().hideEvent(event)  # type: ignore[arg-type]
        if self._timer_id is not None:
            self.killTimer(self._timer_id)
            self._timer_id = None

    def timerEvent(self, _event: object) -> None:
        self._anim_offset = (self._anim_offset + 2) % max(self.width(), 1)
        self.update()

    def paintEvent(self, _event: object) -> None:
        p = QPainter(self)
        antialiasing = False
        p.setRenderHint(QPainter.RenderHint.Antialiasing, antialiasing)
        w, h = self.width(), self.height()
        if self._minimum == self._maximum == 0:
            # Indeterminate: bouncing 30%-width chunk
            chunk_w = max(w * 3 // 10, 4)
            period = w + chunk_w
            x = (self._anim_offset % period) - chunk_w
            p.fillRect(max(x, 0), 0, min(chunk_w, w - max(x, 0)), h, self._bar_color)
        elif self._maximum > self._minimum:
            frac = (self._value - self._minimum) / (self._maximum - self._minimum)
            fill_w = int(w * frac)
            if fill_w > 0:
                p.fillRect(0, 0, fill_w, h, self._bar_color)
        p.end()


class MainWindow(QMainWindow):
    """Main application window implementing Observer pattern."""

    def __init__(self, model: MilMapperModel, controller: MilMapperController) -> None:
        super().__init__()
        self._model = model
        self._controller = controller
        self._download_worker: DownloadWorker | None = None
        self._busy_dialog: QProgressDialog | None = None
        self._modal_overlay: ModalOverlay | None = None  # Will be created in _setup_ui
        self._current_image: Any = None  # Store current image for saving
        self._save_thread: Any = None
        self._save_worker: Any = None

        # Register as observer via adapter to avoid QWidget.update signature clash
        self._observer_adapter = _ViewObserver(self._handle_model_event)
        self._model.add_observer(self._observer_adapter)

        # Image state
        self._base_image: Image.Image | None = None
        # DEM grid for cursor elevation display
        self._dem_grid: Any = None

        # Radio horizon cache for interactive rebuilding
        self._rh_cache: dict[str, Any] = {}
        self._rh_worker: QThread | None = None
        self._rh_click_pos: tuple[float, float] | None = (
            None  # Store click position for marker
        )
        # Pending recompute: if worker is busy when debounce fires,
        # store params and trigger recompute when current worker finishes.
        self._pending_recompute_pos: tuple[float, float] | None = None

        # True when azimuth was changed and recompute hasn't fired yet
        self._azimuth_needs_recompute: bool = False
        # True when parameter was changed interactively and recompute hasn't fired yet
        self._range_needs_recompute: bool = False
        self._sector_needs_recompute: bool = False
        self._antenna_needs_recompute: bool = False
        self._alpha_needs_recompute: bool = False
        self._elev_needs_recompute: bool = False
        self._flight_height_needs_recompute: bool = False
        self._target_h_needs_recompute: bool = False

        # Guard flag to prevent model updates while UI is being populated
        # programmatically
        self._ui_sync_in_progress: bool = False

        self._setup_ui()
        self._setup_connections()

        self._load_initial_data()
        logger.info('MainWindow initialized')

    def _cleanup_download_worker(self) -> None:
        """Disconnect and delete the download worker to break back-references."""
        try:
            if self._download_worker is None:
                return
            # Disconnect all signals from worker to UI
            self._download_worker.finished.disconnect()
            self._download_worker.progress_update.disconnect()
            self._download_worker.preview_ready.disconnect()
            # Остановить дочерний процесс
            self._download_worker.stop_and_join(timeout_ms=2000)
            # Delete later and drop reference
            self._download_worker.deleteLater()
        finally:
            self._download_worker = None

    def _setup_ui(self) -> None:
        """Setup the main window UI."""
        self.setWindowTitle('SK42')
        # Используем минимальный размер и возможность свободно менять размер окна
        self.setMinimumSize(900, 500)
        # Адаптивный стартовый размер: 90% экрана, но не более 1500x950
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(int(avail.width() * 0.9), 1500)
            h = min(int(avail.height() * 0.9), 950)
            self.resize(max(w, 900), max(h, 500))
        else:
            self.resize(1500, 950)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create modal overlay as child of central widget for proper stacking
        self._modal_overlay = ModalOverlay(central_widget)

        # Главный горизонтальный лейаут
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Левая колонка (контролы)
        left_container = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_container)

        # Правая колонка (превью)
        right_container = QVBoxLayout()
        right_container.setContentsMargins(0, 0, 0, 0)
        right_widget = QWidget()
        right_widget.setLayout(right_container)

        # Меню
        self._create_menu()

        # Статус-бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._status_proxy = StatusBarProxy(self.status_bar)

        # Прогресс-бар в левом нижнем углу (в статус-баре): текст → полоска → ✕
        # Фиксированная ширина метки — прогресс-бар не прыгает при смене текста
        self._progress_label = QLabel()
        self._progress_label.setObjectName('progressLabel')
        self._progress_label.setFixedWidth(280)
        self._progress_label.setVisible(False)
        self._progress_bar = _ThinProgressBar()
        self._progress_bar.setFixedWidth(120)
        self._progress_bar.setFixedHeight(3)
        self._progress_bar.setVisible(False)
        self._cancel_btn = QPushButton('\u2715')
        self._cancel_btn.setObjectName('cancelButton')
        self._cancel_btn.setFixedSize(16, 16)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setToolTip('Отменить операцию')
        self._cancel_btn.clicked.connect(self._cancel_operation)
        # Добавляем виджеты прогресса в левую часть статус-бара
        self.status_bar.addWidget(self._progress_label)
        self.status_bar.addWidget(self._progress_bar)
        self.status_bar.addWidget(self._cancel_btn)

        # Метка для координат СК-42 (в правую часть статус-бара)
        self._coords_label = QLabel()
        self._coords_label.setObjectName('coordsLabel')
        self.status_bar.addPermanentWidget(self._coords_label)

        # Блок профилей
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel('Профиль:'))

        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip('Выберите профиль настроек')
        profile_layout.addWidget(self.profile_combo)

        self.save_profile_btn = QPushButton('Сохранить')
        self.save_profile_btn.setObjectName('accentButtonSmall')
        self.save_profile_btn.setToolTip('Сохранить текущие настройки в профиль')
        profile_layout.addWidget(self.save_profile_btn)

        self.save_profile_as_btn = QPushButton('Сохранить как...')
        self.save_profile_as_btn.setObjectName('accentButtonSmall')
        self.save_profile_as_btn.setToolTip(
            'Сохранить текущие настройки в новый профиль',
        )
        profile_layout.addWidget(self.save_profile_as_btn)

        profile_layout.addStretch()
        left_container.addLayout(profile_layout)

        # Координаты
        coords_frame = QFrame()
        coords_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        coords_layout = QVBoxLayout()
        coords_frame.setLayout(coords_layout)

        coords_layout.addWidget(QLabel('Координаты области (СК-42):'))

        self.from_x_widget = OldCoordinateInputWidget('X (вертикаль):', 54, 14)
        self.from_y_widget = OldCoordinateInputWidget('Y (горизонталь):', 74, 43)
        self.to_x_widget = OldCoordinateInputWidget('X (вертикаль):', 54, 23)
        self.to_y_widget = OldCoordinateInputWidget('Y (горизонталь):', 74, 49)

        panels_layout = QHBoxLayout()

        from_group = QFrame()
        from_group.setFrameStyle(QFrame.Shape.StyledPanel)
        from_layout = QVBoxLayout()
        from_title = QLabel('Левый нижний угол')
        from_title.setContentsMargins(5, 5, 5, 5)
        from_layout.addWidget(from_title)
        from_layout.addWidget(self.from_x_widget)
        from_layout.addWidget(self.from_y_widget)
        from_group.setLayout(from_layout)
        panels_layout.addWidget(from_group)

        to_group = QFrame()
        to_group.setFrameStyle(QFrame.Shape.StyledPanel)
        to_layout = QVBoxLayout()
        to_title = QLabel('Правый верхний угол')
        to_title.setContentsMargins(5, 5, 5, 5)
        to_layout.addWidget(to_title)
        to_layout.addWidget(self.to_x_widget)
        to_layout.addWidget(self.to_y_widget)
        to_group.setLayout(to_layout)
        panels_layout.addWidget(to_group)

        coords_layout.addLayout(panels_layout)

        # Контрольная точка
        control_point_group = QFrame()
        control_point_group.setFrameStyle(QFrame.Shape.StyledPanel)
        control_point_layout = QVBoxLayout()

        self.control_point_checkbox = QCheckBox('Контрольная точка')
        self.control_point_checkbox.setToolTip(
            'Включить отображение контрольной точки на карте'
        )

        self.control_point_x_widget = CoordinateInputWidget('X (вертикаль):', 54, 15)
        self.control_point_y_widget = CoordinateInputWidget('Y (горизонталь):', 74, 40)

        # Название контрольной точки
        self.control_point_name_label = QLabel('Название:')
        self.control_point_name_edit = QLineEdit()
        self.control_point_name_edit.setPlaceholderText('Название точки')
        self.control_point_name_edit.setToolTip(
            'Название контрольной точки для отображения на карте'
        )

        # Высота антенны (для карты радиогоризонта) — виджеты,
        # вставляются в grid ниже (row 5)
        self.antenna_height_label = QLabel('Высота антенны (м):')
        self.antenna_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.antenna_height_slider.setRange(0, 50)
        self.antenna_height_slider.setValue(10)
        self.antenna_height_slider.setToolTip(
            'Высота антенны над поверхностью земли (для карты радиогоризонта)'
        )
        self.antenna_height_value_label = QLabel('10')
        self.antenna_height_value_label.setMinimumWidth(24)

        # Максимальная высота полёта (для карты радиогоризонта)
        self.max_flight_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_flight_height_slider.setRange(10, 5000)
        self.max_flight_height_slider.setSingleStep(10)
        self.max_flight_height_slider.setPageStep(10)
        self.max_flight_height_slider.setValue(500)
        self.max_flight_height_slider.setToolTip(
            'Максимальная высота полёта для цветовой шкалы радиогоризонта.\n'
            'Значения выше будут отображаться серым цветом.'
        )
        self.max_flight_height_value_label = QLabel('500')
        self.max_flight_height_value_label.setMinimumWidth(30)

        name_row = QHBoxLayout()
        name_row.addWidget(self.control_point_checkbox)
        name_row.addWidget(self.control_point_name_label)
        name_row.addWidget(self.control_point_name_edit)

        coords_row = QHBoxLayout()
        coords_row.addWidget(self.control_point_x_widget)
        coords_row.addWidget(self.control_point_y_widget)

        # Режим отсчёта высоты БпЛА (для карты радиогоризонта)
        self.height_ref_label = QLabel('Отсчёт высоты:')
        self.height_ref_group = QButtonGroup(self)

        self.height_ref_cp_radio = QRadioButton('От уровня КТ (AGL)')
        self.height_ref_cp_radio.setToolTip(
            'Высота БпЛА отсчитывается от высоты контрольной точки'
        )
        self.height_ref_ground_radio = QRadioButton('От земной поверхности (RA)')
        self.height_ref_ground_radio.setToolTip(
            'Высота БпЛА отсчитывается от земли под ним'
        )
        self.height_ref_sea_radio = QRadioButton('От уровня моря (AMSL)')
        self.height_ref_sea_radio.setToolTip('Абсолютная высота БпЛА над уровнем моря')

        self.height_ref_cp_radio.setChecked(True)  # По умолчанию

        self.height_ref_group.addButton(self.height_ref_cp_radio, 0)
        self.height_ref_group.addButton(self.height_ref_ground_radio, 1)
        self.height_ref_group.addButton(self.height_ref_sea_radio, 2)

        control_point_layout.addLayout(name_row)
        control_point_layout.addLayout(coords_row)
        control_point_group.setLayout(control_point_layout)

        # По умолчанию контролы координат отключены
        self.control_point_x_widget.setEnabled(False)
        self.control_point_y_widget.setEnabled(False)
        self.control_point_name_label.setEnabled(False)
        self.control_point_name_edit.setEnabled(False)

        coords_layout.addWidget(control_point_group)
        left_container.addWidget(coords_frame)

        # Настройки с вкладками
        settings_container = QFrame()
        settings_container.setFrameStyle(QFrame.Shape.StyledPanel)
        settings_main_layout = QVBoxLayout()

        # Тип карты и чекбокс изолиний (над вкладками, всегда видны)
        maptype_row = QHBoxLayout()
        maptype_label = QLabel('<b>Тип карты:</b>')
        self.map_type_combo = QComboBox()
        # Styling handled by global theme (theme.py)

        self._maptype_order = [
            MapType.SATELLITE,
            MapType.HYBRID,
            # MapType.STREETS,
            MapType.OUTDOORS,
            MapType.ELEVATION_COLOR,
            # MapType.ELEVATION_HILLSHADE,  # скрыт — IN PROGRESS
            MapType.RADIO_HORIZON,
            MapType.RADAR_COVERAGE,
            MapType.LINK_PROFILE,
        ]
        for mt in self._maptype_order:
            self.map_type_combo.addItem(MAP_TYPE_LABELS_RU[mt], userData=mt.value)
        # По умолчанию «Спутник»
        self.map_type_combo.setCurrentIndex(0)
        # Чекбокс "Изолинии"
        self.contours_checkbox = QCheckBox('Изолинии')
        self.contours_checkbox.setToolTip(
            'Наложить изолинии поверх выбранного типа карты'
        )
        # Гарантируем кликабельность: резервируем место и фиксируем размер ползунка
        self.contours_checkbox.setEnabled(True)
        self.contours_checkbox.setMinimumWidth(110)
        self.contours_checkbox.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        # Чекбокс "Сетка" (раньше был внутри GridSettingsWidget)
        self.display_grid_cb = QCheckBox('Сетка')
        self.display_grid_cb.setChecked(True)
        self.display_grid_cb.setToolTip(
            'Если включено: рисуются линии сетки и подписи.\n'
            'Если выключено: рисуются только крестики в точках пересечения '
            'без подписей.'
        )

        maptype_row.addWidget(maptype_label)
        maptype_row.addWidget(self.map_type_combo, 1)
        maptype_row.addSpacing(8)
        maptype_row.addWidget(self.display_grid_cb)
        maptype_row.addWidget(self.contours_checkbox)

        maptype_frame = QFrame()
        maptype_frame.setLayout(maptype_row)
        maptype_frame.setObjectName('maptypeFrame')

        # Вставить maptype_frame в coords_layout перед control_point_group
        coords_layout.insertWidget(coords_layout.count() - 1, maptype_frame)

        # GridSettingsWidget создаётся, но не добавляется в layout —
        # используется только в AdvancedSettingsDialog
        self.grid_widget = GridSettingsWidget()

        # Виджеты для «Дополнительных параметров» (не в layout, только хранятся)
        self.output_widget = OutputSettingsWidget()
        self.quality_slider = self.output_widget.quality_slider

        # Настройки радиогоризонта (единая панель для НСУ и РЛС)
        radio_horizon_group = QGroupBox('Радиогоризонт')
        radio_horizon_layout = QVBoxLayout()

        # === Общие установки (создаём виджеты, добавим в grid ниже) ===

        # Прозрачность слоя (виджеты — добавляются в grid позже)
        self.radio_horizon_alpha_label = QLabel('30%')
        self.radio_horizon_alpha_label.setMinimumWidth(30)
        self.radio_horizon_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.radio_horizon_alpha_slider.setRange(0, 100)  # проценты
        self.radio_horizon_alpha_slider.setSingleStep(5)
        self.radio_horizon_alpha_slider.setPageStep(5)
        self.radio_horizon_alpha_slider.setValue(30)  # 30%
        self.radio_horizon_alpha_slider.setToolTip(
            '0% = топооснова не видна, 100% = чистая топооснова'
        )

        # === НСУ-специфичные установки (скрываются при РЛС) ===

        # Режим отсчёта высоты БпЛА — обёрнут в QWidget для управления видимостью
        self._nsu_height_ref_widget = QWidget()
        nsu_ref_layout = QHBoxLayout()
        nsu_ref_layout.setContentsMargins(0, 0, 0, 0)
        nsu_ref_layout.addWidget(self.height_ref_label)
        nsu_ref_layout.addWidget(self.height_ref_cp_radio)
        nsu_ref_layout.addWidget(self.height_ref_ground_radio)
        nsu_ref_layout.addWidget(self.height_ref_sea_radio)
        nsu_ref_layout.addStretch()
        self._nsu_height_ref_widget.setLayout(nsu_ref_layout)
        radio_horizon_layout.addWidget(self._nsu_height_ref_widget)

        # === РЛС/НСУ-специфичные установки в общей сетке ===

        # Все РЛС-строки обёрнуты в один QWidget для единого управления видимостью
        self._radar_settings_widget = QWidget()
        radar_hbox = QHBoxLayout()
        radar_hbox.setContentsMargins(0, 0, 0, 0)
        radar_hbox.setSpacing(16)

        # Азимут: лейбл + крутилка + значение — отцентрированы по вертикали
        self._azimuth_top_label = QLabel('Азимут')
        self._azimuth_top_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.radar_azimuth_dial = QDial()
        self.radar_azimuth_dial.setRange(0, 359)
        self.radar_azimuth_dial.setWrapping(True)
        self.radar_azimuth_dial.setNotchesVisible(True)
        self.radar_azimuth_dial.setInvertedAppearance(True)
        self.radar_azimuth_dial.setFixedSize(110, 110)
        self.radar_azimuth_dial.setValue(180)  # 0° azimuth
        self.radar_azimuth_dial.setToolTip(
            'Азимут направления РЛС (0°=север, по часовой)'
        )

        self.radar_azimuth_label = QLabel('0°')
        self.radar_azimuth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.radar_azimuth_label.setToolTip(
            'Азимут направления РЛС (0°=север, по часовой)'
        )

        self._azimuth_widget = QWidget()
        azimuth_col = QVBoxLayout(self._azimuth_widget)
        azimuth_col.setContentsMargins(0, 0, 0, 0)
        azimuth_col.setSpacing(0)
        azimuth_col.addStretch()
        azimuth_col.addWidget(self._azimuth_top_label)
        azimuth_col.addWidget(self.radar_azimuth_dial, 0, Qt.AlignmentFlag.AlignCenter)
        azimuth_col.addWidget(self.radar_azimuth_label)
        azimuth_col.addStretch()
        radar_hbox.addWidget(self._azimuth_widget)

        # Слайдеры справа от крутилки
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        grid.setHorizontalSpacing(2)

        # Row 0: Азимутальный угол
        self.radar_sector_slider = QSlider(Qt.Orientation.Horizontal)
        self.radar_sector_slider.setRange(30, 360)
        self.radar_sector_slider.setValue(90)
        self.radar_sector_slider.setToolTip('Ширина сектора обзора РЛС (градусы)')
        self.radar_sector_label = QLabel('90°')
        self.radar_sector_label.setMinimumWidth(30)
        sector_row = QVBoxLayout()
        sector_row.setContentsMargins(0, 0, 0, 0)
        sector_row.setSpacing(0)
        self.radar_sector_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        sector_row.addWidget(self.radar_sector_label)
        sector_row.addWidget(self.radar_sector_slider)
        self._sector_label = QLabel('Азимут. угол (°):')
        grid.addWidget(self._sector_label, 0, 0)
        grid.addLayout(sector_row, 0, 1)

        # Row 1: Угол места — range slider + label
        self.radar_elev_slider = _DarculaRangeSlider(Qt.Orientation.Horizontal)
        self.radar_elev_slider.setRange(0, 90)
        self.radar_elev_slider.setValue((1, 30))
        self.radar_elev_slider.setToolTip('Диапазон углов места РЛС (°)')
        self.radar_elev_label = QLabel('1—30°')
        self.radar_elev_label.setMinimumWidth(42)
        elev_row = QVBoxLayout()
        elev_row.setContentsMargins(0, 0, 0, 0)
        elev_row.setSpacing(0)
        self.radar_elev_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        elev_row.addWidget(self.radar_elev_label)
        elev_row.addWidget(self.radar_elev_slider)
        self._elev_label = QLabel('Угол места (°):')
        grid.addWidget(self._elev_label, 1, 0)
        grid.addLayout(elev_row, 1, 1)

        # Row 2: Дальность — slider + label
        self.radar_range_slider = QSlider(Qt.Orientation.Horizontal)
        self.radar_range_slider.setRange(1, 100)  # 1–100 км
        self.radar_range_slider.setValue(15)
        self.radar_range_slider.setToolTip(
            'Максимальная дальность обнаружения РЛС (км)'
        )
        self.radar_range_label = QLabel('15')
        self.radar_range_label.setMinimumWidth(24)
        range_row = QVBoxLayout()
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.setSpacing(0)
        self.radar_range_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        range_row.addWidget(self.radar_range_label)
        range_row.addWidget(self.radar_range_slider)
        self._range_label = QLabel('Дальность (км):')
        grid.addWidget(self._range_label, 2, 0)
        grid.addLayout(range_row, 2, 1)

        # Row 3: Высота целей — range slider + label
        self.radar_target_h_slider = _DarculaRangeSlider(Qt.Orientation.Horizontal)
        self.radar_target_h_slider.setRange(2, 500)  # ×10 → 20–5000 м
        self.radar_target_h_slider.setValue((3, 500))
        self.radar_target_h_slider.setToolTip('Диапазон высот целей (м)')
        self.radar_target_h_label = QLabel('30—5000')
        self.radar_target_h_label.setMinimumWidth(55)
        target_h_row = QVBoxLayout()
        target_h_row.setContentsMargins(0, 0, 0, 0)
        target_h_row.setSpacing(0)
        self.radar_target_h_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        target_h_row.addWidget(self.radar_target_h_label)
        target_h_row.addWidget(self.radar_target_h_slider)
        self._target_h_label = QLabel('Высота целей (м):')
        grid.addWidget(self._target_h_label, 3, 0)
        grid.addLayout(target_h_row, 3, 1)

        # Row 4: Высота антенны — slider + label (общая для НСУ и РЛС)
        # Совмещаем с лейблом азимута "0°" (col 0) на той же строке
        antenna_h_row = QVBoxLayout()
        antenna_h_row.setContentsMargins(0, 0, 0, 0)
        antenna_h_row.setSpacing(0)
        self.antenna_height_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        antenna_h_row.addWidget(self.antenna_height_value_label)
        antenna_h_row.addWidget(self.antenna_height_slider)
        grid.addWidget(self.antenna_height_label, 4, 0)
        grid.addLayout(antenna_h_row, 4, 1)

        # Row 5: Практический потолок БпЛА (НСУ-only)
        flight_h_row = QVBoxLayout()
        flight_h_row.setContentsMargins(0, 0, 0, 0)
        flight_h_row.setSpacing(0)
        self.max_flight_height_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        flight_h_row.addWidget(self.max_flight_height_value_label)
        flight_h_row.addWidget(self.max_flight_height_slider)
        self._flight_height_label = QLabel('Потолок БпЛА (м):')
        grid.addWidget(self._flight_height_label, 5, 0)
        grid.addLayout(flight_h_row, 5, 1)

        # Row 6: Прозрачность топоосновы (общая для НСУ и РЛС)
        alpha_col_row = QVBoxLayout()
        alpha_col_row.setContentsMargins(0, 0, 0, 0)
        alpha_col_row.setSpacing(0)
        self.radio_horizon_alpha_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        alpha_col_row.addWidget(self.radio_horizon_alpha_label)
        alpha_col_row.addWidget(self.radio_horizon_alpha_slider)
        self._alpha_label = QLabel('Прозрачность:')
        grid.addWidget(self._alpha_label, 6, 0)
        grid.addLayout(alpha_col_row, 6, 1)

        # Виджеты, видимые только в режиме РЛС (строки 0-3 сетки)
        self._radar_only_widgets = [
            self._azimuth_widget,
            self._sector_label,
            self.radar_sector_slider,
            self.radar_sector_label,
            self._elev_label,
            self.radar_elev_slider,
            self.radar_elev_label,
            self._range_label,
            self.radar_range_slider,
            self.radar_range_label,
            self._target_h_label,
            self.radar_target_h_slider,
            self.radar_target_h_label,
        ]

        # Виджеты, видимые только в режиме НСУ (строка 5 сетки)
        self._nsu_only_widgets = [
            self._flight_height_label,
            self.max_flight_height_slider,
            self.max_flight_height_value_label,
        ]

        # Виджеты антенны — видны для НСУ/РЛС, скрыты для ELEVATION_COLOR
        self._antenna_widgets = [
            self.antenna_height_label,
            self.antenna_height_slider,
            self.antenna_height_value_label,
        ]

        radar_hbox.addLayout(grid, 1)
        self._radar_settings_widget.setLayout(radar_hbox)
        self._radar_settings_widget.setVisible(False)
        radio_horizon_layout.addWidget(self._radar_settings_widget)

        radio_horizon_group.setLayout(radio_horizon_layout)
        radio_horizon_group.setVisible(False)  # Видна только при НСУ/РЛС
        self._radio_horizon_group = radio_horizon_group
        settings_main_layout.addWidget(radio_horizon_group)

        # --- Панель «Профиль радиолинии» ---
        link_profile_group = QGroupBox('Профиль радиолинии')
        link_layout = QVBoxLayout()

        # Координаты точки A
        link_a_label = QLabel('<b>Точка A:</b>')
        self.link_a_x_widget = CoordinateInputWidget('X (вертикаль):', 54, 15)
        self.link_a_y_widget = CoordinateInputWidget('Y (горизонталь):', 74, 40)

        link_a_coords_row = QHBoxLayout()
        link_a_coords_row.addWidget(self.link_a_x_widget)
        link_a_coords_row.addWidget(self.link_a_y_widget)

        # Имя точки A
        self.link_a_name_label = QLabel('Название A:')
        self.link_a_name_edit = QLineEdit()
        self.link_a_name_edit.setPlaceholderText('A')
        self.link_a_name_edit.setText('A')
        self.link_a_name_edit.setMaximumWidth(100)

        link_a_name_row = QHBoxLayout()
        link_a_name_row.addWidget(link_a_label)
        link_a_name_row.addWidget(self.link_a_name_edit)
        link_a_name_row.addStretch()

        link_layout.addLayout(link_a_name_row)
        link_layout.addLayout(link_a_coords_row)

        # Координаты точки B
        link_b_label = QLabel('<b>Точка B:</b>')
        self.link_b_x_widget = CoordinateInputWidget('X (вертикаль):', 54, 20)
        self.link_b_y_widget = CoordinateInputWidget('Y (горизонталь):', 74, 45)

        link_b_coords_row = QHBoxLayout()
        link_b_coords_row.addWidget(self.link_b_x_widget)
        link_b_coords_row.addWidget(self.link_b_y_widget)

        # Имя точки B
        self.link_b_name_label = QLabel('Название B:')
        self.link_b_name_edit = QLineEdit()
        self.link_b_name_edit.setPlaceholderText('B')
        self.link_b_name_edit.setText('B')
        self.link_b_name_edit.setMaximumWidth(100)

        link_b_name_row = QHBoxLayout()
        link_b_name_row.addWidget(link_b_label)
        link_b_name_row.addWidget(self.link_b_name_edit)
        link_b_name_row.addStretch()

        link_layout.addLayout(link_b_name_row)
        link_layout.addLayout(link_b_coords_row)

        # Параметры радиолинии — grid
        link_grid = QGridLayout()
        link_grid.setContentsMargins(0, 0, 0, 0)
        link_grid.setVerticalSpacing(2)
        link_grid.setHorizontalSpacing(4)

        # Частота
        link_grid.addWidget(QLabel('Частота (МГц):'), 0, 0)
        self.link_freq_spin = QDoubleSpinBox()
        self.link_freq_spin.setRange(1.0, 100000.0)
        self.link_freq_spin.setDecimals(1)
        self.link_freq_spin.setValue(900.0)
        self.link_freq_spin.setToolTip('Рабочая частота радиолинии (МГц)')
        link_grid.addWidget(self.link_freq_spin, 0, 1)

        # Высота антенны A
        link_grid.addWidget(QLabel('Высота ант. A (м):'), 1, 0)
        self.link_antenna_a_spin = QDoubleSpinBox()
        self.link_antenna_a_spin.setRange(0.0, 200.0)
        self.link_antenna_a_spin.setDecimals(1)
        self.link_antenna_a_spin.setValue(10.0)
        self.link_antenna_a_spin.setToolTip('Высота антенны в точке A (м)')
        link_grid.addWidget(self.link_antenna_a_spin, 1, 1)

        # Высота антенны B
        link_grid.addWidget(QLabel('Высота ант. B (м):'), 2, 0)
        self.link_antenna_b_spin = QDoubleSpinBox()
        self.link_antenna_b_spin.setRange(0.0, 200.0)
        self.link_antenna_b_spin.setDecimals(1)
        self.link_antenna_b_spin.setValue(10.0)
        self.link_antenna_b_spin.setToolTip('Высота антенны в точке B (м)')
        link_grid.addWidget(self.link_antenna_b_spin, 2, 1)

        link_layout.addLayout(link_grid)

        link_profile_group.setLayout(link_layout)
        link_profile_group.setVisible(False)
        self._link_profile_group = link_profile_group
        settings_main_layout.addWidget(link_profile_group)

        # Helmert widget (не в layout, хранится для диалога и профилей)
        self.helmert_widget = HelmertSettingsWidget()

        settings_main_layout.addStretch()
        settings_container.setLayout(settings_main_layout)
        left_container.addWidget(settings_container)

        # Оборачиваем левую колонку в QScrollArea для предотвращения обрезания контента
        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._left_scroll.setWidget(left_widget)
        self._left_content_widget = left_widget

        # Кнопка "Создать карту" — вне скролла, всегда видна
        self.download_btn = QPushButton('Создать карту')
        self.download_btn.setToolTip('Начать создание карты')
        self.download_btn.setObjectName('accentButton')
        self.download_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        # Обёртка: скролл и кнопка
        self._left_panel = QWidget()
        left_panel_layout = QVBoxLayout()
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.addWidget(self._left_scroll, 1)
        left_panel_layout.addWidget(self.download_btn)
        self._left_panel.setLayout(left_panel_layout)

        # Превью справа
        self._preview_frame = QFrame()
        self._preview_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout()
        self._preview_frame.setLayout(preview_layout)

        preview_layout.addWidget(QLabel('Предпросмотр карты:'))

        self._preview_area = OptimizedImageView()
        self._preview_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._preview_area.setMinimumHeight(220)
        self._preview_area.setMinimumWidth(300)
        preview_layout.addWidget(self._preview_area, 1)

        right_container.addWidget(self._preview_frame, 1)

        # Кнопка "Сохранить карту"
        self.save_map_btn = QPushButton('Сохранить карту')
        self.save_map_btn.setObjectName('accentButton')
        self.save_map_btn.setToolTip('Сохранить карту в файл')
        self.save_map_btn.setEnabled(False)
        self.save_map_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        right_container.addWidget(self.save_map_btn)

        self._set_sliders_enabled(enabled=False)

        # Отложенное создание BusyDialog
        self._busy_dialog = None

        # Разделитель колонок для настраиваемых ширин
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._left_panel)
        self._splitter.addWidget(right_widget)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(1)
        # Левая панель: ширина по содержимому, не растягивается
        left_preferred = self._left_content_widget.sizeHint().width() + 4
        self._left_panel.setMinimumWidth(left_preferred)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([left_preferred, 600])

        # Добавляем splitter вместо двух виджетов
        main_layout.addWidget(self._splitter, 1)

    def _create_menu(self) -> None:
        """Create application menu."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('Файл')

        new_action = QAction('Новый профиль', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self._new_profile)
        file_menu.addAction(new_action)

        open_action = QAction('Открыть профиль...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self._open_profile)
        file_menu.addAction(open_action)

        save_action = QAction('Сохранить профиль', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self._save_current_profile)
        file_menu.addAction(save_action)

        save_as_action = QAction('Сохранить профиль как...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(self._save_profile_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        api_key_action = QAction('API ключ...', self)
        api_key_action.triggered.connect(self._show_api_key_dialog)
        file_menu.addAction(api_key_action)

        file_menu.addSeparator()

        exit_action = QAction('Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Operations menu
        operations_menu = menubar.addMenu('Операции')

        create_map_action = QAction('Создать карту', self)
        create_map_action.triggered.connect(self._start_download)
        operations_menu.addAction(create_map_action)

        self.save_map_action = QAction('Сохранить карту', self)
        self.save_map_action.triggered.connect(self._save_map)
        self.save_map_action.setEnabled(False)  # Initially disabled like the button
        operations_menu.addAction(self.save_map_action)

        operations_menu.addSeparator()

        advanced_action = QAction('Дополнительные параметры…', self)
        advanced_action.triggered.connect(self._show_advanced_settings)
        operations_menu.addAction(advanced_action)

        # Help menu
        help_menu = menubar.addMenu('Справка')

        about_action = QAction('О программе', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_connections(self) -> None:
        """Setup signal connections."""
        # Preview mouse tracking
        self._preview_area.mouse_moved_on_map.connect(self._on_mouse_moved_on_map)
        self._preview_area.map_right_clicked.connect(self._on_map_right_clicked)
        self._preview_area.shift_wheel_rotated.connect(self._rotate_radar_azimuth)
        self._preview_area.shift_key_released.connect(self._on_shift_released_recompute)

        # Profile management
        # Подключение сигнала выбора профиля произойдет после первичной инициализации
        self.save_profile_btn.clicked.connect(self._save_current_profile)
        self.save_profile_as_btn.clicked.connect(self._save_profile_as)

        # Download
        self.download_btn.clicked.connect(self._start_download)

        # Save map
        self.save_map_btn.clicked.connect(self._save_map)

        # Settings change tracking
        self._connect_setting_changes()
        # Map type change triggers settings update and clears preview immediately
        self.map_type_combo.currentIndexChanged.connect(self._on_map_type_changed)
        # Overlay contours toggle
        self.contours_checkbox.toggled.connect(self._on_settings_changed)

    def _set_profile_selection_safely(
        self, *, name: str | None = None, index: int | None = None
    ) -> None:
        """Set profile combo selection with signals blocked to avoid recursion."""
        blocker = QSignalBlocker(self.profile_combo)
        try:
            if name is not None:
                self.profile_combo.setCurrentText(name)
            elif index is not None:
                self.profile_combo.setCurrentIndex(index)
        finally:
            del blocker

    def _connect_setting_changes(self) -> None:
        """Connect all setting change signals."""
        # Coordinates (old format with spinboxes)
        for widget in [
            self.from_x_widget,
            self.from_y_widget,
            self.to_x_widget,
            self.to_y_widget,
        ]:
            widget.high_edit.editingFinished.connect(self._on_settings_changed)
            widget.low_edit.editingFinished.connect(self._on_settings_changed)

        # Control point
        self.control_point_checkbox.stateChanged.connect(self._on_control_point_toggled)
        self.control_point_checkbox.stateChanged.connect(self._on_settings_changed)
        # Use editingFinished instead of textChanged
        # to avoid cyclic updates during typing
        self.control_point_x_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )
        self.control_point_y_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )
        # Control point name
        self.control_point_name_edit.editingFinished.connect(self._on_settings_changed)
        # Antenna height for radio horizon
        self.antenna_height_slider.valueChanged.connect(self._on_antenna_slider_changed)
        self.antenna_height_slider.sliderReleased.connect(
            self._on_antenna_slider_released
        )
        # Max flight height for radio horizon
        self.max_flight_height_slider.valueChanged.connect(
            self._on_flight_height_slider_changed
        )
        self.max_flight_height_slider.sliderReleased.connect(
            self._on_flight_height_slider_released
        )
        # UAV height reference radios
        self.height_ref_group.buttonClicked.connect(self._on_height_ref_changed)

        # Overlay alpha
        self.radio_horizon_alpha_slider.valueChanged.connect(
            self._on_alpha_slider_changed
        )
        self.radio_horizon_alpha_slider.sliderReleased.connect(
            self._on_alpha_slider_released
        )

        # Radar coverage settings
        self.radar_azimuth_dial.valueChanged.connect(self._on_azimuth_dial_changed)
        self.radar_azimuth_dial.sliderReleased.connect(self._on_azimuth_dial_released)
        self.radar_sector_slider.valueChanged.connect(self._on_sector_slider_changed)
        self.radar_sector_slider.sliderReleased.connect(self._on_sector_slider_released)
        self.radar_elev_slider.valueChanged.connect(self._on_elev_range_changed)
        self.radar_elev_slider.sliderReleased.connect(self._on_elev_slider_released)
        self.radar_range_slider.valueChanged.connect(self._on_range_slider_changed)
        self.radar_range_slider.sliderReleased.connect(self._on_range_slider_released)
        self.radar_target_h_slider.valueChanged.connect(self._on_target_h_range_changed)
        self.radar_target_h_slider.sliderReleased.connect(
            self._on_target_h_slider_released
        )

        # Grid settings
        self.grid_widget.width_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.font_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.margin_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.padding_spin.valueChanged.connect(self._on_settings_changed)
        self.display_grid_cb.stateChanged.connect(self._on_settings_changed)

        # Link profile settings
        self.link_a_x_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )
        self.link_a_y_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )
        self.link_a_name_edit.editingFinished.connect(self._on_settings_changed)
        self.link_b_x_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )
        self.link_b_y_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )
        self.link_b_name_edit.editingFinished.connect(self._on_settings_changed)
        self.link_freq_spin.valueChanged.connect(self._on_settings_changed)
        self.link_antenna_a_spin.valueChanged.connect(self._on_settings_changed)
        self.link_antenna_b_spin.valueChanged.connect(self._on_settings_changed)

        # Helmert settings — changed only via AdvancedSettingsDialog

    def _load_initial_data(self) -> None:
        """Load initial application data."""
        # Load available profiles
        profiles = self._controller.get_available_profiles()

        # Блокируем сигналы на время программных изменений
        block_signals = True
        self.profile_combo.blockSignals(block_signals)
        try:
            self.profile_combo.clear()
            self.profile_combo.addItems(profiles)
            if 'default' in profiles:
                self.profile_combo.setCurrentText('default')
            elif profiles:
                self.profile_combo.setCurrentIndex(0)
        finally:
            block_signals = False
            self.profile_combo.blockSignals(block_signals)

        # Явно загружаем профиль ровно один раз
        if profiles:
            self._load_selected_profile(-1)

        # Теперь подключаем обработчик изменения выбора
        self.profile_combo.currentIndexChanged.connect(self._load_selected_profile)

    @Slot()
    def _on_settings_changed(self) -> None:
        """Handle settings change from UI (bulk update without intermediate events)."""
        if getattr(self, '_ui_sync_in_progress', False):
            logging.getLogger(__name__).debug(
                'Settings change ignored: UI sync in progress'
            )
            return

        # Set flag to prevent feedback loop when model emits SETTINGS_CHANGED
        self._ui_sync_in_progress = True
        try:
            # Clear any existing preview to avoid showing outdated imagery
            # when coordinates or contours change
            self._clear_preview_ui()
            # Use the same collection logic as forced sync
            self._sync_ui_to_model_now()
        finally:
            self._ui_sync_in_progress = False

    def _on_control_point_toggled(self) -> None:
        """Хендлер изменения состояния чекбокса контрольной точки."""
        enabled = self.control_point_checkbox.isChecked()
        self.control_point_x_widget.setEnabled(enabled)
        self.control_point_y_widget.setEnabled(enabled)
        self.control_point_name_label.setEnabled(enabled)
        self.control_point_name_edit.setEnabled(enabled)

    @Slot()
    def _on_map_type_changed(self) -> None:
        """Clear preview immediately when map type changes and propagate setting."""
        # Clear any existing preview to avoid showing outdated imagery for another
        # map type
        self._clear_preview_ui()

        # При выборе "Радиогоризонт" автоматически включаем контрольную точку
        # и блокируем чекбокс
        try:
            idx = max(0, self.map_type_combo.currentIndex())
            map_type_value = self.map_type_combo.itemData(idx)
            is_radio_horizon = map_type_value == MapType.RADIO_HORIZON.value
            is_radar_coverage = map_type_value == MapType.RADAR_COVERAGE.value
            is_link_profile = map_type_value == MapType.LINK_PROFILE.value
            is_elev_color = map_type_value in (
                MapType.ELEVATION_COLOR.value,
                MapType.ELEVATION_HILLSHADE.value,
            )
        except Exception:
            is_radio_horizon = False
            is_radar_coverage = False
            is_link_profile = False
            is_elev_color = False

        # Панель видна для НСУ/РЛС/Карта высот (слайдер прозрачности)
        self._radio_horizon_group.setVisible(
            is_radio_horizon or is_radar_coverage or is_elev_color
        )

        # Панель профиля радиолинии
        self._link_profile_group.setVisible(is_link_profile)

        # Заголовок панели зависит от режима
        if is_elev_color:
            self._radio_horizon_group.setTitle('')
        else:
            self._radio_horizon_group.setTitle('')

        # Управление видимостью НСУ/РЛС-специфичных виджетов внутри единой панели
        nsu_visible = is_radio_horizon and not is_radar_coverage
        self._nsu_height_ref_widget.setVisible(nsu_visible)
        # Для ELEVATION_COLOR показываем только слайдер альфы
        self._radar_settings_widget.setVisible(
            is_radio_horizon or is_radar_coverage or is_elev_color
        )
        for w in self._radar_only_widgets:
            w.setVisible(is_radar_coverage)
        for w in self._nsu_only_widgets:
            w.setVisible(nsu_visible)
        # Виджеты антенны скрыты для ELEVATION_COLOR
        antenna_visible = is_radio_horizon or is_radar_coverage
        for w in self._antenna_widgets:
            w.setVisible(antenna_visible)

        if is_link_profile:
            # Для профиля радиолинии: контрольная точка заблокирована в выключенном
            # состоянии — точки A и B задаются в собственной панели
            self.control_point_checkbox.setChecked(False)
            self.control_point_checkbox.setEnabled(False)
            self.control_point_x_widget.setEnabled(False)
            self.control_point_y_widget.setEnabled(False)
            self.control_point_name_label.setEnabled(False)
            self.control_point_name_edit.setEnabled(False)
        elif is_radio_horizon or is_radar_coverage:
            # Принудительно включаем контрольную точку
            # и блокируем возможность отключения
            self.control_point_checkbox.setChecked(True)
            self.control_point_checkbox.setEnabled(False)
            # Включаем поля ввода координат и названия
            self.control_point_x_widget.setEnabled(True)
            self.control_point_y_widget.setEnabled(True)
            self.control_point_name_label.setEnabled(True)
            self.control_point_name_edit.setEnabled(True)
        else:
            # Разблокируем чекбокс для других типов карт
            self.control_point_checkbox.setEnabled(True)
            # Обновляем состояние полей согласно текущему состоянию чекбокса
            self._on_control_point_toggled()

        # Подстроить ширину левой панели под новое содержимое
        self._adjust_left_panel_width()

        # Delegate to the common settings handler to store the new map type in the model
        self._on_settings_changed()

    def _adjust_left_panel_width(self) -> None:
        """Resize left splitter pane to fit its content after visibility changes."""

        def _do_adjust() -> None:
            # Ширина реального содержимого внутри QScrollArea
            content_w = self._left_content_widget.sizeHint().width()
            # Плюс возможный вертикальный скроллбар
            sb = self._left_scroll.verticalScrollBar()
            if sb and sb.isVisible():
                content_w += sb.width()
            # Небольшой запас на рамки layout
            needed = content_w + 4
            current = self._splitter.sizes()
            if current and needed > current[0]:
                self._left_panel.setMinimumWidth(needed)
                self._splitter.setSizes(
                    [needed, max(current[1] - (needed - current[0]), 200)]
                )

        # Отложить на 0 мс — чтобы Qt успел пересчитать layout после setVisible
        QTimer.singleShot(0, _do_adjust)

    def _sync_ui_to_model_now(self) -> None:
        """
        Force-collect current UI settings and push them to the model without guards.

        Does not clear preview or check _ui_sync_in_progress to avoid losing changes
        during Save/Save As.
        """
        # Collect all current settings
        coords = self._get_current_coordinates()
        grid_settings = self.grid_widget.get_settings()
        grid_settings['display_grid'] = self.display_grid_cb.isChecked()
        helmert_settings = self.helmert_widget.get_values()

        # Map type from combo (stored as enum value string)
        try:
            idx = max(0, self.map_type_combo.currentIndex())
            map_type_value = self.map_type_combo.itemData(idx)
        except Exception:
            map_type_value = MapType.SATELLITE.value

        overlay_checked = bool(self.contours_checkbox.isChecked())

        payload: dict[str, Any] = {}
        payload.update(coords)
        payload.update(grid_settings)
        payload.update(helmert_settings)
        payload['map_type'] = map_type_value
        payload['overlay_contours'] = overlay_checked
        payload['radio_horizon_overlay_alpha'] = (
            self.radio_horizon_alpha_slider.value() / 100.0
        )
        # Radar coverage parameters
        payload['radar_azimuth_deg'] = float(
            (540 - self.radar_azimuth_dial.value()) % 360
        )
        payload['radar_sector_width_deg'] = float(self.radar_sector_slider.value())
        elev_lo: float
        elev_hi: float
        elev_lo, elev_hi = self.radar_elev_slider.value()
        payload['radar_elevation_min_deg'] = float(elev_lo)
        payload['radar_elevation_max_deg'] = float(elev_hi)
        payload['radar_max_range_km'] = float(self.radar_range_slider.value())
        target_lo: float
        target_hi: float
        target_lo, target_hi = self.radar_target_h_slider.value()
        payload['radar_target_height_min_m'] = float(target_lo * 10)
        payload['radar_target_height_max_m'] = float(target_hi * 10)
        # Link profile parameters
        payload['link_point_a_x'] = self.link_a_x_widget.get_coordinate()
        payload['link_point_a_y'] = self.link_a_y_widget.get_coordinate()
        payload['link_point_a_name'] = self.link_a_name_edit.text() or 'A'
        payload['link_point_b_x'] = self.link_b_x_widget.get_coordinate()
        payload['link_point_b_y'] = self.link_b_y_widget.get_coordinate()
        payload['link_point_b_name'] = self.link_b_name_edit.text() or 'B'
        payload['link_freq_mhz'] = self.link_freq_spin.value()
        payload['link_antenna_a_m'] = self.link_antenna_a_spin.value()
        payload['link_antenna_b_m'] = self.link_antenna_b_spin.value()
        self._controller.update_settings_bulk(**payload)

    def _get_current_coordinates(self) -> dict[str, int | bool | float | str]:
        """Get current coordinate values from UI."""
        from_x_high, from_x_low = self.from_x_widget.get_values()
        from_y_high, from_y_low = self.from_y_widget.get_values()
        to_x_high, to_x_low = self.to_x_widget.get_values()
        to_y_high, to_y_low = self.to_y_widget.get_values()

        # Get control point coordinates directly to preserve user input
        control_point_x = self.control_point_x_widget.get_coordinate()
        control_point_y = self.control_point_y_widget.get_coordinate()

        return {
            'from_x_high': from_x_high,
            'from_x_low': from_x_low,
            'from_y_high': from_y_high,
            'from_y_low': from_y_low,
            'to_x_high': to_x_high,
            'to_x_low': to_x_low,
            'to_y_high': to_y_high,
            'to_y_low': to_y_low,
            'control_point_enabled': self.control_point_checkbox.isChecked(),
            'control_point_x': control_point_x,
            'control_point_y': control_point_y,
            'control_point_name': self.control_point_name_edit.text(),
            'antenna_height_m': self.antenna_height_slider.value(),
            'max_flight_height_m': float(self.max_flight_height_slider.value()),
            'uav_height_reference': self._get_uav_height_reference(),
        }

    @Slot(int)
    def _load_selected_profile(self, index: int) -> None:
        """Load the selected profile when selection changes (guarded, non-reentrant)."""
        _ = index
        # Guard against re-entrant calls
        if getattr(self, '_profile_loading', False):
            logger.debug('Profile load skipped due to guard re-entry')
            return
        if not hasattr(self, '_profile_loading'):
            self._profile_loading = False

        self._profile_loading = True
        try:
            # Ignore index value; use currentText for robustness
            profile_name = self.profile_combo.currentText()
            if not profile_name:
                return
            logger.debug(f'Loading profile: {profile_name}')
            self._controller.load_profile_by_name(profile_name)
        finally:
            self._profile_loading = False

    @Slot()
    def _save_current_profile(self) -> None:
        """Save current settings to profile."""
        # 1) Ensure pending UI edits are committed (e.g., QLineEdit editingFinished)
        w = QApplication.focusWidget()
        if w is not None:
            w.clearFocus()
        QApplication.processEvents()
        # 2) Force bulk sync UI -> Model before saving (bypass guards)
        self._sync_ui_to_model_now()

        profile_name = self.profile_combo.currentText()
        if profile_name:
            self._controller.save_current_profile(profile_name)

    @Slot()
    def _save_profile_as(self) -> None:
        """Save current settings to a new profile file."""
        # 1) Ensure pending UI edits are committed (e.g., QLineEdit editingFinished)
        w = QApplication.focusWidget()
        if w is not None:
            w.clearFocus()
        QApplication.processEvents()
        # 2) Force bulk sync UI -> Model before opening the dialog (bypass guards)
        self._sync_ui_to_model_now()

        # Get default directory (always user/resolved profiles dir)
        default_dir = ensure_profiles_dir()

        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Сохранить профиль как...',
            str(default_dir / 'новый_профиль'),
            'Файлы профилей (*.toml);;Все файлы (*)',
        )

        if file_path:
            # 3) Just in case user changed something via shortcuts while dialog open
            w = QApplication.focusWidget()
            if w is not None:
                w.clearFocus()
            QApplication.processEvents()
            self._sync_ui_to_model_now()

            # Check if profile already exists and show warning
            profile_path = Path(file_path)
            if profile_path.suffix.lower() != '.toml':
                profile_path = profile_path.with_suffix('.toml')

            if profile_path.exists():
                msg_box = QMessageBox(
                    QMessageBox.Icon.Question,
                    'Предупреждение',
                    f"Профиль с именем '{profile_path.stem}' уже существует.\n"
                    f'Перезаписать существующий профиль?',
                    parent=self,
                )
                msg_box.addButton('Да', QMessageBox.ButtonRole.YesRole)
                no_button = msg_box.addButton('Нет', QMessageBox.ButtonRole.NoRole)
                msg_box.setDefaultButton(no_button)

                msg_box.exec()

                if msg_box.clickedButton() == no_button:
                    return  # User canceled the operation

            # Call controller method
            saved_profile_name = self._controller.save_current_profile_as(file_path)
            if saved_profile_name:
                # Update profile combo if the file was saved in profiles directory
                if Path(file_path).with_suffix('.toml').parent == ensure_profiles_dir():
                    # Refresh profile list
                    self._load_initial_data()
                    # Select the newly saved profile safely (no extra load)
                    index = self.profile_combo.findText(saved_profile_name)
                    if index >= 0:
                        self._set_profile_selection_safely(index=index)
                        # Explicitly load the newly saved profile to avoid reverting
                        # to default
                        try:
                            logger.info(
                                'After Save As -> selecting and loading profile: %s',
                                saved_profile_name,
                            )
                            self._controller.load_profile_by_name(saved_profile_name)
                        except Exception:
                            logger.exception(
                                'Failed to load newly saved profile: %s',
                                saved_profile_name,
                            )

                self._status_proxy.show_message(
                    f'Профиль сохранён как: {saved_profile_name}',
                    3000,
                )

    @Slot()
    def _start_download(self) -> None:
        """Start map download process."""
        # Clear status bar message and informer message immediately
        self.status_bar.clearMessage()
        self._progress_label.setText('')

        if self._download_worker and self._download_worker.isRunning():
            QMessageBox.information(self, 'Информация', 'Загрузка уже выполняется')
            return

        # Force sync UI -> Model before starting download to ensure all settings
        # (including control_point_enabled) are up-to-date
        self._sync_ui_to_model_now()

        # Подготовка параметров (валидация, deep verification)
        try:
            params = self._controller.prepare_download_params()
        except RuntimeError as e:
            QMessageBox.critical(self, 'Ошибка', str(e))
            return

        # Clear previous preview and pixmap cache between runs
        self._clear_preview_ui()
        self._preview_area.start_loading()

        # Cleanup any stale worker from previous run
        self._cleanup_download_worker()

        self._download_worker = DownloadWorker(params)
        self._download_worker.finished.connect(
            lambda success, error_msg: self._on_download_finished(
                success=success,
                error_msg=error_msg,
            )
        )
        self._download_worker.progress_update.connect(self._update_progress)
        self._download_worker.preview_ready.connect(self._show_preview_in_main_window)
        self._download_worker.start()

        # Update UI state
        # Очищаем временное сообщение статус-бара, иначе оно скроет виджеты
        self.status_bar.clearMessage()
        # Show progress bar in status bar (will be updated by progress callbacks)
        self._progress_label.setVisible(True)
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._progress_label.setText('Подготовка…')
        self._progress_bar.setRange(0, 0)
        # Disable all UI controls during download
        self._set_controls_enabled(enabled=False)

    @Slot()
    def _cancel_operation(self) -> None:
        """Cancel the currently running operation."""
        if self._download_worker and self._download_worker.isRunning():
            logger.info('User requested download cancellation')
            self._download_worker.request_cancel()
            self._progress_label.setText('Отмена…')
            self._cancel_btn.setEnabled(False)
        elif self._rh_worker is not None and self._rh_worker.isRunning():
            logger.info('User requested radio horizon recompute cancellation')
            self._rh_worker.terminate()
            self._rh_worker.wait(500)
            self._rh_worker.deleteLater()
            self._rh_worker = None
            self._progress_bar.setVisible(False)
            self._progress_label.setVisible(False)
            self._cancel_btn.setVisible(False)
            self._set_controls_enabled(enabled=True)
            QApplication.restoreOverrideCursor()
            self._status_proxy.show_message('Операция отменена', 3000)

    @Slot(object, object)
    def _on_mouse_moved_on_map(self, px: float | None, py: float | None) -> None:
        """Handle mouse movement over preview to show SK-42 coords and elevation."""
        coords = self._calculate_sk42_from_scene_pos(px, py)
        if coords is None:
            self._coords_label.setText('')
            return

        x_val, y_val = coords

        def format_coord(val: int) -> str:
            s = str(abs(val))
            if len(s) > COORDINATE_FORMAT_SPLIT_LENGTH:
                return (
                    f'{s[:-COORDINATE_FORMAT_SPLIT_LENGTH]} '
                    f'{s[-COORDINATE_FORMAT_SPLIT_LENGTH:]}'
                )
            return s

        # Get elevation from DEM if available
        elevation_str = ''
        if self._dem_grid is not None and px is not None and py is not None:
            try:
                row = int(py)
                col = int(px)
                dem_h, dem_w = self._dem_grid.shape
                if 0 <= row < dem_h and 0 <= col < dem_w:
                    elevation = self._dem_grid[row, col]
                    elevation_str = f'  H: {int(elevation)} м'
            except Exception:
                logger.debug('Elevation lookup failed at px=%s, py=%s', px, py)

        self._coords_label.setText(
            f'X: {format_coord(x_val)}  Y: {format_coord(y_val)}{elevation_str}'
        )

    def _calculate_sk42_from_scene_pos(
        self, px: float | None, py: float | None
    ) -> tuple[int, int] | None:
        """Calculate SK-42 GK coordinates from scene pixel coordinates."""
        if px is None or py is None:
            return None

        metadata = self._model.state.last_map_metadata
        if not metadata:
            return None

        try:
            # 1. Смещение от центра изображения в пикселях
            cx_px = metadata.width_px / 2.0
            cy_px = metadata.height_px / 2.0

            dx_px = px - cx_px
            dy_px = py - cy_px

            # 2. Учет поворота (обратное преобразование)
            rotation_rad = math.radians(-metadata.rotation_deg)
            cos_rot = math.cos(rotation_rad)
            sin_rot = math.sin(rotation_rad)

            dx = dx_px * cos_rot + dy_px * sin_rot
            dy = -dx_px * sin_rot + dy_px * cos_rot

            # 3. Переход к "мировым" пикселям Web Mercator
            cx_world, cy_world = latlng_to_pixel_xy(
                metadata.center_lat_wgs, metadata.center_lng_wgs, metadata.zoom
            )

            x_world = dx / metadata.scale + cx_world
            y_world = dy / metadata.scale + cy_world

            # 4. Обратная проекция: Мировые пиксели -> WGS-84 -> SK-42 Geog -> SK-42 GK
            lat_wgs, lng_wgs = pixel_xy_to_latlng(x_world, y_world, metadata.zoom)

            t_sk42_to_wgs, t_wgs_to_sk42 = build_transformers_sk42(
                custom_helmert=metadata.helmert_params
            )
            lng_sk42, lat_sk42 = t_wgs_to_sk42.transform(lng_wgs, lat_wgs)

            zone = determine_zone(metadata.center_x_gk)
            crs_sk42_gk = build_sk42_gk_crs(zone)

            t_sk42gk_from_sk42 = Transformer.from_crs(
                crs_sk42_geog, crs_sk42_gk, always_xy=True
            )

            # Получаем (easting, northing) для GK
            y_val_m, x_val_m = t_sk42gk_from_sk42.transform(lng_sk42, lat_sk42)

            return round(x_val_m), round(y_val_m)

        except Exception:
            logger.exception('Failed to calculate SK-42 coordinates from pixel')
            return None

    @Slot(float, float)
    def _on_map_right_clicked(self, px: float, py: float) -> None:
        """Transfer right-click to control point or rebuild radio horizon."""
        metadata = self._model.state.last_map_metadata
        if not metadata:
            return

        # Check if this is a coverage map with cache available
        is_coverage_map = (
            metadata.map_type in (MapType.RADIO_HORIZON, MapType.RADAR_COVERAGE)
            and self._rh_cache
            and 'dem' in self._rh_cache
        )

        if is_coverage_map:
            # Interactive coverage rebuilding (НСУ or РЛС)
            self._recompute_coverage_at_click(px, py)
        elif metadata.control_point_enabled:
            # Standard behavior: update control point coordinates
            coords = self._calculate_sk42_from_scene_pos(px, py)
            if coords is None:
                return

            x_val, y_val = coords

            self.control_point_x_widget.set_coordinate(x_val)
            self.control_point_y_widget.set_coordinate(y_val)

            # Enable control point if it was disabled
            if not self.control_point_checkbox.isChecked():
                self.control_point_checkbox.setChecked(True)

            # Sync to model to ensure it's saved/propagated
            self._sync_ui_to_model_now()

            # Update markers on preview
            self._update_cp_marker_from_settings(self._model.settings)

            self._status_proxy.show_message(
                f'Координаты КТ обновлены: X={x_val}, Y={y_val}', 3000
            )

    def _recompute_radio_horizon_at_click(self, px: float, py: float) -> None:
        """Legacy wrapper — delegates to _recompute_coverage_at_click."""
        self._recompute_coverage_at_click(px, py)

    def _recompute_coverage_at_click(
        self,
        px: float,
        py: float,
        *,
        dem_row: int | None = None,
        dem_col: int | None = None,
    ) -> None:
        """
        Recompute coverage (НСУ or РЛС) with new position at clicked point.

        Args:
            px: X position on the final image (for overlay center).
            py: Y position on the final image (for overlay center).
            dem_row: If provided, use this DEM row directly
                instead of converting from (px, py).
            dem_col: If provided, use this DEM column directly
                instead of converting from (px, py).

        """
        # If worker is still running, defer this recompute until it finishes
        if self._rh_worker is not None and self._rh_worker.isRunning():
            logger.info('Worker busy — deferring recompute to pending queue')
            self._pending_recompute_pos = (px, py)
            return

        self._pending_recompute_pos = None

        dem = self._rh_cache.get('dem')
        final_size = self._rh_cache.get('final_size')
        if dem is None:
            logger.warning('No DEM in cache, cannot recompute')
            return

        dem_h, dem_w = dem.shape

        if dem_row is not None and dem_col is not None:
            # Direct DEM coords — no round-trip conversion needed
            new_antenna_row = dem_row
            new_antenna_col = dem_col
        elif final_size:
            final_w, final_h = final_size
            rotation_deg = self._rh_cache.get('rotation_deg', 0.0)

            # Размеры полноразмерного DEM (= crop_rect pixel size)
            dem_full = self._rh_cache.get('dem_full')
            if dem_full is not None:
                crop_h, crop_w = dem_full.shape
            else:
                crop_h, crop_w = dem_h, dem_w

            # 1. Undo center crop: final image → crop_rect (повёрнутый)
            left_crop = (crop_w - final_w) / 2.0
            top_crop = (crop_h - final_h) / 2.0
            px_rot = px + left_crop
            py_rot = py + top_crop

            # 2. Обратный поворот вокруг центра crop_rect
            # cv2 forward rotation: [[cos(a), sin(a)], [-sin(a), cos(a)]]
            # Inverse = transpose:  [[cos(a), -sin(a)], [sin(a), cos(a)]]
            # With positive angle, (dx*cos - dy*sin, dx*sin + dy*cos) is the inverse.
            if abs(rotation_deg) > ROTATION_INVERSE_THRESHOLD_DEG:
                cx, cy = crop_w / 2.0, crop_h / 2.0
                rad = math.radians(rotation_deg)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                dx, dy = px_rot - cx, py_rot - cy
                ux = dx * cos_a - dy * sin_a + cx
                uy = dx * sin_a + dy * cos_a + cy
            else:
                ux, uy = px_rot, py_rot

            # 3. Масштабирование: crop_rect → downsampled DEM
            scale_x = dem_w / crop_w
            scale_y = dem_h / crop_h
            new_antenna_col = int(ux * scale_x)
            new_antenna_row = int(uy * scale_y)
        else:
            new_antenna_col = int(px)
            new_antenna_row = int(py)

        new_antenna_row = max(0, min(new_antenna_row, dem_h - 1))
        new_antenna_col = max(0, min(new_antenna_col, dem_w - 1))

        logger.info(
            'Coverage recompute: scene (%.1f, %.1f) -> DEM (%d, %d)',
            px,
            py,
            new_antenna_col,
            new_antenna_row,
        )

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._set_controls_enabled(enabled=False)

        # Determine if this is a radar coverage map
        is_radar = self._rh_cache.get('is_radar_coverage', False)
        sector_params = None
        if is_radar:
            sector_params = {
                'radar_azimuth_deg': self._rh_cache.get('radar_azimuth_deg', 0.0),
                'radar_sector_width_deg': self._rh_cache.get(
                    'radar_sector_width_deg', 90.0
                ),
                'radar_elevation_min_deg': self._rh_cache.get(
                    'radar_elevation_min_deg', 0.5
                ),
                'radar_elevation_max_deg': self._rh_cache.get(
                    'radar_elevation_max_deg', 30.0
                ),
                'radar_max_range_km': self._rh_cache.get('radar_max_range_km', 15.0),
                'radar_target_height_min_m': self._rh_cache.get(
                    'radar_target_height_min_m', 0.0
                ),
                'radar_target_height_max_m': self._rh_cache.get(
                    'radar_target_height_max_m', 5000.0
                ),
            }
            self._progress_label.setText('Пересчет зоны обнаружения РЛС…')
        else:
            self._progress_label.setText('Пересчет радиогоризонта…')

        # Очищаем временное сообщение статус-бара, иначе оно скроет виджеты
        self.status_bar.clearMessage()
        # Индикатор прогресса (indeterminate)
        self._progress_bar.setRange(0, 0)
        self._progress_label.setVisible(True)
        self._progress_bar.setVisible(True)
        self._cancel_btn.setVisible(True)

        self._rh_click_pos = (px, py)

        # Use generalized CoverageRecomputeWorker
        self._rh_worker = CoverageRecomputeWorker(
            self._rh_cache,
            new_antenna_row,
            new_antenna_col,
            sector_params=sector_params,
        )
        self._rh_worker.finished.connect(self._on_radio_horizon_recompute_finished)
        self._rh_worker.error.connect(self._on_radio_horizon_recompute_error)
        self._rh_worker.start()

    @Slot(Image.Image, Image.Image, int, int)
    def _on_radio_horizon_recompute_finished(
        self,
        result_image: Image.Image,
        coverage_layer: Image.Image,
        new_antenna_row: int,
        new_antenna_col: int,
    ) -> None:
        """Handle radio horizon recompute completion."""
        try:
            gui_start_time = time.monotonic()
            # Sync UI → model FIRST so that label drawing reads current values
            # (e.g. antenna_height_m, max_flight_height_m from sliders)
            self._sync_ui_to_model_now()
            self._rh_cache['coverage_layer'] = coverage_layer
            logger.info('RH recompute finished - applying postprocessing')

            # Update cache with new antenna position
            self._rh_cache['antenna_row'] = new_antenna_row
            self._rh_cache['antenna_col'] = new_antenna_col
            # rotation_deg is preserved: recomputed result is now rotated
            # (same coordinate system as first build).
            rotation_deg = self._rh_cache.get('rotation_deg', 0.0)

            # Create display-sized topo for interactive alpha blending.
            # (recompute worker returns coverage_layer at final_size,
            #  so topo must also be at final_size for correct blend)
            self._prepare_rh_topo_display()

            # Apply postprocessing (grid, legend, contours)
            metadata = self._model.state.last_map_metadata
            mpp = metadata.meters_per_pixel if metadata else 0.0

            # Apply cached overlay layer, or rebuild from base + fresh legend
            step_start = time.monotonic()
            overlay_layer = self._rh_cache.get('overlay_layer')
            if overlay_layer:
                # Use cached overlay (grid + legend + contours)
                result_image = result_image.convert('RGBA')
                result_image = Image.alpha_composite(result_image, overlay_layer)
                step_elapsed = time.monotonic() - step_start
                logger.info('  └─ Apply cached overlay layer: %.3f sec', step_elapsed)
            else:
                # Rebuild overlay: base (contours + grid) + fresh legend
                overlay_base = self._rh_cache.get('overlay_base')
                if overlay_base is not None:
                    overlay_layer = overlay_base.copy()
                else:
                    overlay_layer = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
                self._draw_rh_legend(overlay_layer, mpp)
                self._rh_cache['overlay_layer'] = overlay_layer
                result_image = result_image.convert('RGBA')
                result_image = Image.alpha_composite(result_image, overlay_layer)
                step_elapsed = time.monotonic() - step_start
                logger.info(
                    '  └─ Rebuilt overlay (base + legend): %.3f sec', step_elapsed
                )

            # Draw control point triangle and label on the image (like in first build)
            step_start = time.monotonic()
            if hasattr(self, '_rh_click_pos') and self._rh_click_pos is not None:
                px, py = self._rh_click_pos

                if mpp > 0:
                    # Draw triangle (size matches font)
                    draw_control_point_triangle(
                        result_image,
                        px,
                        py,
                        mpp,
                        rotation_deg=rotation_deg,
                        size_m=self._model.settings.grid_font_size_m,
                    )

                    # Draw label (name + height)
                    self._draw_rh_control_point_label(result_image, px, py, mpp)

                # Calculate and update SK-42 coordinates
                coords = self._calculate_sk42_from_scene_pos(px, py)
                if coords:
                    x_val, y_val = coords
                    self.control_point_x_widget.set_coordinate(x_val)
                    self.control_point_y_widget.set_coordinate(y_val)

                    logger.info('Control point updated to X=%d, Y=%d', x_val, y_val)
            step_elapsed = time.monotonic() - step_start
            logger.info('  └─ Draw control point marker: %.3f sec', step_elapsed)

            # Draw radar sector overlay + marker if this is RADAR_COVERAGE
            if self._rh_cache.get('is_radar_coverage', False):
                step_start = time.monotonic()
                try:
                    result_image = result_image.convert('RGBA')
                    radar_az = self._rh_cache.get('radar_azimuth_deg', 0.0)
                    radar_sw = self._rh_cache.get('radar_sector_width_deg', 90.0)
                    radar_range_km = self._rh_cache.get('radar_max_range_km', 15.0)
                    radar_elev_max = self._rh_cache.get('radar_elevation_max_deg', 30.0)

                    # pixel_size_m in cache is DEM pixel size; overlay is drawn
                    # on the final (resized) image, so compute effective pixel size.
                    pixel_size_final = self._get_final_pixel_size_m()
                    max_range_px = (radar_range_km * 1000.0) / pixel_size_final

                    # Radar position = control point position (click pos)
                    radar_cx, radar_cy = (
                        self._rh_click_pos
                        if self._rh_click_pos
                        else (result_image.width / 2, result_image.height / 2)
                    )

                    # Font for ceiling arc labels
                    ppm = 1.0 / pixel_size_final if pixel_size_final > 0 else 1.0
                    settings = self._rh_cache.get('settings')
                    font_size_m = settings.grid_font_size_m if settings else 100.0
                    font_size_px = max(10, round(font_size_m * ppm * 0.4))
                    try:
                        arc_font = load_grid_font(font_size_px)
                    except Exception:
                        arc_font = None

                    draw_sector_overlay(
                        img=result_image,
                        cx=radar_cx,
                        cy=radar_cy,
                        azimuth_deg=radar_az,
                        sector_width_deg=radar_sw,
                        max_range_px=max_range_px,
                        pixel_size_m=pixel_size_final,
                        elevation_max_deg=radar_elev_max,
                        font=arc_font,
                        rotation_deg=rotation_deg,
                    )

                    draw_radar_marker(
                        result_image,
                        radar_cx,
                        radar_cy,
                        pixel_size_final,
                        azimuth_deg=radar_az,
                        rotation_deg=rotation_deg,
                    )

                    step_elapsed = time.monotonic() - step_start
                    logger.info(
                        '  └─ Draw radar sector overlay: %.3f sec', step_elapsed
                    )
                except Exception as e:
                    logger.warning('Failed to draw radar sector overlay: %s', e)

            # Update display
            step_start = time.monotonic()
            self._base_image = (
                result_image.convert('RGB')
                if result_image.mode != 'RGB'
                else result_image
            )
            self._current_image = self._base_image

            metadata = self._model.state.last_map_metadata
            mpp = metadata.meters_per_pixel if metadata else 0.0
            self._preview_area.set_image(self._current_image, meters_per_px=mpp)
            step_elapsed = time.monotonic() - step_start
            logger.info('  └─ Update display: %.3f sec', step_elapsed)

            # Redraw persistent azimuth indicator after set_image clears the scene
            if self._rh_cache.get('is_radar_coverage', False):
                az = self._rh_cache.get('radar_azimuth_deg', 0.0)
                self._update_azimuth_indicator(az)

            # Restore QGraphics overlay markers cleared by set_image → scene.clear()
            self._update_cp_marker_from_settings(self._model.settings)

            # Restore cursor and show success message
            QApplication.restoreOverrideCursor()

            gui_total_elapsed = time.monotonic() - gui_start_time
            logger.info('GUI postprocessing total: %.3f sec', gui_total_elapsed)
            if self._rh_cache.get('is_radar_coverage', False):
                self._status_proxy.show_message(
                    'Зона обнаружения РЛС пересчитана', 2000
                )
            else:
                self._status_proxy.show_message('Радиогоризонт пересчитан', 2000)

        except Exception as e:
            logger.exception('Failed to update preview after radio horizon recompute')
            QApplication.restoreOverrideCursor()
            self._status_proxy.show_message(f'Ошибка при обновлении превью: {e}', 5000)
        finally:
            # Скрыть прогресс-бар
            self._progress_bar.setVisible(False)
            self._progress_label.setVisible(False)
            self._cancel_btn.setVisible(False)

            # Re-enable all UI controls
            self._set_controls_enabled(enabled=True)

            # Clean up worker
            if self._rh_worker is not None:
                self._rh_worker.deleteLater()
                self._rh_worker = None

            # If a recompute was requested while worker was busy, run it now
            self._flush_pending_recompute()

    def _flush_pending_recompute(self) -> None:
        """If a recompute was deferred while worker was busy, start it now."""
        if self._pending_recompute_pos is not None:
            px, py = self._pending_recompute_pos
            self._pending_recompute_pos = None
            logger.info('Flushing pending recompute at (%.1f, %.1f)', px, py)
            self._recompute_coverage_at_click(px, py)

    def _draw_rh_grid(
        self,
        result: Image.Image,
        settings: MapSettings,
        mpp: float,
    ) -> None:
        """Draw grid on radio horizon map."""
        try:
            if mpp <= 0:
                return

            ppm = 1.0 / mpp
            grid_width_px = max(1, round(settings.grid_width_m * ppm))
            font_size_px = max(12, round(settings.grid_font_size_m * ppm))
            text_margin_px = max(5, round(settings.grid_text_margin_m * ppm))
            label_bg_padding_px = max(2, round(settings.grid_label_bg_padding_m * ppm))

            draw_axis_aligned_km_grid(
                result,
                center_x_m=settings.control_point_x_sk42_gk
                if settings.control_point_enabled
                else 0,
                center_y_m=settings.control_point_y_sk42_gk
                if settings.control_point_enabled
                else 0,
                meters_per_px=mpp,
                rotation_deg=0.0,
                grid_width_px=grid_width_px,
                font_size_px=font_size_px,
                text_margin_px=text_margin_px,
                label_bg_padding_px=label_bg_padding_px,
                display_grid=settings.display_grid,
            )
        except Exception as e:
            logger.warning('Не удалось нарисовать сетку: %s', e)

    def _draw_rh_legend(
        self,
        result: Image.Image,
        mpp: float,
    ) -> None:
        """Draw legend on radio horizon map."""
        try:
            if mpp <= 0:
                return

            settings = self._rh_cache.get('settings')
            if not settings:
                return

            metadata = self._model.state.last_map_metadata
            if not metadata:
                return

            is_radar = self._rh_cache.get('is_radar_coverage', False)
            if is_radar:
                min_elev = self._rh_cache.get(
                    'radar_target_height_min_m',
                    settings.radar_target_height_min_m,
                )
                max_elev = self._rh_cache.get(
                    'radar_target_height_max_m',
                    settings.radar_target_height_max_m,
                )
                title = 'Мин. высота обнаружения РЛС, м'
            else:
                min_elev = 0.0
                max_elev = self._rh_cache.get(
                    'max_height_m',
                    settings.max_flight_height_m,
                )
                abbr = UAV_HEIGHT_REFERENCE_ABBR.get(
                    settings.uav_height_reference,
                    '',
                )
                title = (
                    f'Минимальная высота БпЛА ({abbr}) для устойчивой радиосвязи'
                    if abbr
                    else 'Минимальная высота БпЛА'
                )

            draw_elevation_legend(
                img=result,
                color_ramp=RADIO_HORIZON_COLOR_RAMP,
                min_elevation_m=min_elev,
                max_elevation_m=max_elev,
                center_lat_wgs=metadata.center_lat_wgs,
                zoom=metadata.zoom,
                scale=metadata.scale,
                title=title,
                label_step_m=ELEVATION_LEGEND_STEP_M,
                grid_font_size_m=settings.grid_font_size_m,
            )
        except Exception as e:
            logger.warning('Не удалось нарисовать легенду: %s', e)

    def _draw_rh_control_point_label(
        self,
        result: Image.Image,
        cx_img: float,
        cy_img: float,
        mpp: float,
    ) -> None:
        """Draw control point label for radio horizon map."""
        try:
            settings = self._model.settings
            cp_name = settings.control_point_name
            antenna_h = settings.antenna_height_m

            ppm = 1.0 / mpp if mpp > 0 else 0.0
            font_size_px = max(12, round(settings.grid_font_size_m * ppm))
            label_font = load_grid_font(font_size_px)
            subscript_font = load_grid_font(max(8, font_size_px * 2 // 3))
            bg_padding_px = max(2, round(settings.grid_label_bg_padding_m * ppm))

            draw = ImageDraw.Draw(result)

            # Position below triangle (triangle size matches font size)
            tri_size_px = font_size_px
            label_x = int(cx_img)
            label_gap_px = max(
                CONTROL_POINT_LABEL_GAP_MIN_PX,
                round(tri_size_px * CONTROL_POINT_LABEL_GAP_RATIO),
            )
            current_y = int(cy_img + tri_size_px / 2 + label_gap_px + bg_padding_px)

            # Name line
            if cp_name:
                draw_label_with_bg(
                    draw,
                    (label_x, current_y),
                    cp_name,
                    font=label_font,
                    anchor='mt',
                    img_size=result.size,
                    padding=bg_padding_px,
                )
                name_bbox = draw.textbbox((0, 0), cp_name, font=label_font, anchor='lt')
                name_height = name_bbox[3] - name_bbox[1]
                current_y += name_height + bg_padding_px * 2

            # Height line with subscript
            # Get elevation from _dem_grid (same source as informer tooltip)
            cp_elev = None
            if self._dem_grid is not None:
                row, col = int(cy_img), int(cx_img)
                dh, dw = self._dem_grid.shape
                if 0 <= row < dh and 0 <= col < dw:
                    cp_elev = float(self._dem_grid[row, col])

            if cp_elev is not None:
                height_parts = [
                    ('h = ', False),
                    (f'{int(cp_elev)}', False),
                    (' + ', False),
                    (f'{int(antenna_h)} м', False),
                ]
            else:
                height_parts = [
                    ('h', False),
                    ('ант', True),
                    (f' = {int(antenna_h)} м', False),
                ]

            draw_label_with_subscript_bg(
                draw,
                (label_x, current_y),
                height_parts,
                font=label_font,
                subscript_font=subscript_font,
                anchor='mt',
                img_size=result.size,
                padding=bg_padding_px,
            )
        except Exception as e:
            logger.warning('Не удалось нарисовать подпись контрольной точки: %s', e)

    @Slot(str)
    def _on_radio_horizon_recompute_error(self, error_msg: str) -> None:
        """Handle radio horizon recompute error."""
        logger.error('Radio horizon recompute failed: %s', error_msg)
        QApplication.restoreOverrideCursor()
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._set_controls_enabled(enabled=True)
        self._status_proxy.show_message(f'Ошибка пересчета: {error_msg}', 5000)

        # Clean up worker
        if self._rh_worker is not None:
            self._rh_worker.deleteLater()
            self._rh_worker = None

        # If a recompute was deferred, run it now
        self._flush_pending_recompute()

    def _has_coverage_cache(self) -> bool:
        """
        Check if any coverage map is displayed with cache.

        Covers НСУ, РЛС and ELEVATION_COLOR map types.
        """
        metadata = self._model.state.last_map_metadata
        if metadata is None or not self._rh_cache:
            return False
        if metadata.map_type in (MapType.RADIO_HORIZON, MapType.RADAR_COVERAGE):
            return 'dem' in self._rh_cache
        if metadata.map_type == MapType.ELEVATION_COLOR:
            return 'coverage_layer' in self._rh_cache
        return False

    def _is_radar_map_active(self) -> bool:
        """Check if a radar coverage map is currently displayed with cached DEM."""
        metadata = self._model.state.last_map_metadata
        return (
            metadata is not None
            and metadata.map_type == MapType.RADAR_COVERAGE
            and bool(self._rh_cache)
            and 'dem' in self._rh_cache
        )

    def _apply_interactive_azimuth(self, new_az: float) -> None:
        """Update azimuth interactively: cache + indicator, no map clear."""
        self._rh_cache['radar_azimuth_deg'] = new_az
        self._status_proxy.show_message(f'Азимут РЛС: {new_az:.0f}°', 1000)
        self._update_azimuth_indicator(new_az)
        self._azimuth_needs_recompute = True

    @Slot(int)
    def _on_azimuth_dial_changed(self, dial_value: int) -> None:
        """Sync azimuth label when QDial is rotated."""
        az = float((540 - dial_value) % 360)
        self.radar_azimuth_label.setText(f'{int(az)}°')
        if self._is_radar_map_active():
            self._apply_interactive_azimuth(az)
        else:
            self._on_settings_changed()

    @Slot(tuple)
    def _on_elev_range_changed(self, value: tuple) -> None:
        """Update elevation range label when range slider is dragged."""
        lo, hi = value
        self.radar_elev_label.setText(f'{lo}—{hi}°')
        if self._has_coverage_cache():
            self._rh_cache['radar_elevation_min_deg'] = float(lo)
            self._rh_cache['radar_elevation_max_deg'] = float(hi)
            self._elev_needs_recompute = True
        else:
            self._on_settings_changed()

    @Slot()
    def _on_elev_slider_released(self) -> None:
        """Elevation slider released — recompute or sync model."""
        if self._elev_needs_recompute:
            self._elev_needs_recompute = False
            self._trigger_coverage_recompute()
        else:
            self._on_settings_changed()

    @Slot(tuple)
    def _on_target_h_range_changed(self, value: tuple) -> None:
        """Update label + cache when target height range slider is dragged."""
        lo, hi = value
        self.radar_target_h_label.setText(f'{lo * 10}—{hi * 10}')
        if self._is_radar_map_active():
            self._rh_cache['radar_target_height_min_m'] = float(lo * 10)
            self._rh_cache['radar_target_height_max_m'] = float(hi * 10)
            self._rh_cache['max_height_m'] = float(hi * 10)
            # Легенда встроена в overlay — сбрасываем, чтобы перерисовать
            self._rh_cache.pop('overlay_layer', None)
            self._status_proxy.show_message(
                f'Высота целей: {lo * 10}—{hi * 10} м', 1000
            )
            self._target_h_needs_recompute = True
        else:
            self._on_settings_changed()

    @Slot()
    def _on_target_h_slider_released(self) -> None:
        """Target height slider released — trigger recompute or sync model."""
        if self._target_h_needs_recompute:
            self._target_h_needs_recompute = False
            self._trigger_coverage_recompute()
        else:
            self._on_settings_changed()

    @Slot()
    def _on_azimuth_dial_released(self) -> None:
        """QDial released — trigger recompute if needed."""
        self._trigger_azimuth_recompute()

    # ── Antenna height slider slot ─────────────────────────────

    @Slot(int)
    def _on_antenna_slider_changed(self, value: int) -> None:
        """Update label + cache when slider is dragged."""
        self.antenna_height_value_label.setText(str(value))
        if self._has_coverage_cache():
            self._rh_cache['antenna_height_m'] = float(value)
            self._status_proxy.show_message(f'Высота антенны: {value} м', 1000)
            self._antenna_needs_recompute = True
        else:
            self._on_settings_changed()

    @Slot()
    def _on_antenna_slider_released(self) -> None:
        """Slider released — trigger recompute or sync model."""
        if self._antenna_needs_recompute:
            self._antenna_needs_recompute = False
            self._trigger_coverage_recompute()
        else:
            self._on_settings_changed()

    # ── Flight height slider interactive slots ──────────────

    def _on_flight_height_slider_changed(self, value: int) -> None:
        """Update label + cache when flight height slider is dragged."""
        step = self.max_flight_height_slider.singleStep()
        snapped = round(value / step) * step
        if snapped != value:
            self.max_flight_height_slider.setValue(snapped)
            return
        self.max_flight_height_value_label.setText(str(value))
        if self._has_coverage_cache():
            self._rh_cache['max_flight_height_m'] = float(value)
            self._rh_cache['max_height_m'] = float(value)
            # Легенда встроена в overlay — сбрасываем, чтобы перерисовать
            self._rh_cache.pop('overlay_layer', None)
            self._status_proxy.show_message(f'Потолок БпЛА: {value} м', 1000)
            self._flight_height_needs_recompute = True
        else:
            self._on_settings_changed()

    @Slot()
    def _on_flight_height_slider_released(self) -> None:
        """Slider released — trigger recompute or sync model."""
        if self._flight_height_needs_recompute:
            self._flight_height_needs_recompute = False
            self._trigger_coverage_recompute()
        else:
            self._on_settings_changed()

    # ── Height reference radio buttons ─────────────────────

    @Slot()
    def _on_height_ref_changed(self) -> None:
        """Radio button changed — update cache + trigger recompute."""
        ref = self._get_uav_height_reference()
        if self._has_coverage_cache():
            self._rh_cache['uav_height_reference'] = ref
            self._trigger_coverage_recompute()
        else:
            self._on_settings_changed()

    # ── Alpha slider interactive slots ───────────────────────

    def _is_coverage_map_active(self) -> bool:
        """Check if any coverage map (НСУ or РЛС) is active with cache."""
        metadata = self._model.state.last_map_metadata
        if not metadata:
            return False
        return (
            metadata.map_type in (MapType.RADIO_HORIZON, MapType.RADAR_COVERAGE)
            and bool(self._rh_cache)
            and 'coverage_layer' in self._rh_cache
        )

    def _prepare_rh_topo_display(self) -> None:
        """
        Transform topo_base to final display size (resize → rotate → crop).

        Sets rh_cache['topo_base_display'] so _apply_interactive_alpha
        blends at the correct display resolution.
        """
        if not self._rh_cache:
            return
        rotation_deg = self._rh_cache.get('rotation_deg', 0.0)
        final_size = self._rh_cache.get('final_size')
        crop_size = self._rh_cache.get('crop_size')
        if not final_size or not crop_size:
            return

        fw, fh = final_size
        cw, ch = crop_size
        has_rotation = abs(rotation_deg) > ROTATION_EPSILON

        topo = self._rh_cache.get('topo_base')
        if topo is not None:
            topo_d = topo.copy()
            if topo_d.size != (cw, ch):
                topo_d = topo_d.resize((cw, ch), Image.Resampling.BILINEAR)
            if has_rotation:
                topo_d = rotate_keep_size(topo_d, rotation_deg, fill=(128, 128, 128))
            topo_d = center_crop(topo_d, fw, fh)
            self._rh_cache['topo_base_display'] = topo_d.convert('L').convert('RGBA')

    def _prepare_rh_display_cache(self) -> None:
        """
        Transform both coverage_layer and topo_base to final display size.

        Called once after the initial build. Replicates the first-build
        pipeline (resize → rotate → center_crop) so that
        _apply_interactive_alpha works at the correct display size.
        """
        # Prepare topo
        self._prepare_rh_topo_display()

        # Prepare coverage_layer at final_size
        if not self._rh_cache:
            return
        rotation_deg = self._rh_cache.get('rotation_deg', 0.0)
        final_size = self._rh_cache.get('final_size')
        crop_size = self._rh_cache.get('crop_size')
        if not final_size or not crop_size:
            return

        fw, fh = final_size
        cw, ch = crop_size
        has_rotation = abs(rotation_deg) > ROTATION_EPSILON

        coverage = self._rh_cache.get('coverage_layer')
        if coverage is not None:
            cov_d = coverage
            if cov_d.size != (cw, ch):
                cov_d = cov_d.resize((cw, ch), Image.Resampling.BILINEAR)
            if has_rotation:
                cov_d = rotate_keep_size(cov_d, rotation_deg, fill=(0, 0, 0, 0))
            cov_d = center_crop(cov_d, fw, fh)
            self._rh_cache['coverage_layer'] = cov_d

    def _apply_interactive_alpha(self, alpha_fraction: float) -> None:
        """Re-blend coverage_layer with topo_base at new alpha (instant)."""
        coverage = self._rh_cache.get('coverage_layer')
        # Use rotated+cropped topo for display if available (matches coverage coords)
        topo_base = self._rh_cache.get('topo_base_display') or self._rh_cache.get(
            'topo_base'
        )
        if coverage is None or topo_base is None:
            return
        self._rh_cache['overlay_alpha'] = alpha_fraction
        blend_alpha = 1.0 - alpha_fraction
        rotation_deg = self._rh_cache.get('rotation_deg', 0.0)
        # Гарантируем совпадение размеров
        if coverage.size != topo_base.size:
            coverage = coverage.resize(topo_base.size, Image.Resampling.BILINEAR)
        blended = Image.blend(topo_base, coverage, blend_alpha)
        # Apply cached overlay (grid + legend)
        overlay_layer = self._rh_cache.get('overlay_layer')
        if overlay_layer:
            blended = blended.convert('RGBA')
            if overlay_layer.size != blended.size:
                overlay_layer = overlay_layer.resize(
                    blended.size, Image.Resampling.BILINEAR
                )
            blended = Image.alpha_composite(blended, overlay_layer)
        # Redraw control point triangle + label
        if hasattr(self, '_rh_click_pos') and self._rh_click_pos is not None:
            metadata = self._model.state.last_map_metadata
            mpp = metadata.meters_per_pixel if metadata else 0.0
            if mpp > 0:
                px, py = self._rh_click_pos
                draw_control_point_triangle(
                    blended,
                    px,
                    py,
                    mpp,
                    rotation_deg=rotation_deg,
                    size_m=self._model.settings.grid_font_size_m,
                )
                self._draw_rh_control_point_label(blended, px, py, mpp)
        self._preview_area.set_image(blended)
        # Restore QGraphics overlay markers cleared by set_image → scene.clear()
        self._update_cp_marker_from_settings(self._model.settings)

    @Slot(int)
    def _on_alpha_slider_changed(self, value: int) -> None:
        """Update label + status while dragging (blend deferred to release)."""
        step = self.radio_horizon_alpha_slider.singleStep()
        snapped = round(value / step) * step
        if snapped != value:
            self.radio_horizon_alpha_slider.setValue(snapped)
            return
        self.radio_horizon_alpha_label.setText(f'{value}%')
        self._status_proxy.show_message('Пересчет прозрачности...', 1000)
        self._alpha_needs_recompute = True

    @Slot()
    def _on_alpha_slider_released(self) -> None:
        """Alpha slider released — apply blend + sync model."""
        if self._alpha_needs_recompute:
            self._alpha_needs_recompute = False
            if self._has_coverage_cache() and 'coverage_layer' in self._rh_cache:
                alpha = self.radio_horizon_alpha_slider.value() / 100.0
                self._rh_cache['overlay_alpha'] = alpha
                self._set_controls_enabled(enabled=False)
                self.status_bar.clearMessage()
                self._progress_label.setText('Пересчёт…')
                self._progress_bar.setRange(0, 0)
                self._progress_label.setVisible(True)
                self._progress_bar.setVisible(True)
                self._cancel_btn.setVisible(False)  # Быстрая операция — без отмены
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                QApplication.processEvents()
                try:
                    self._apply_interactive_alpha(alpha)
                finally:
                    QApplication.restoreOverrideCursor()
                    self._progress_bar.setVisible(False)
                    self._progress_label.setVisible(False)
                    self._cancel_btn.setVisible(False)
                    self._set_controls_enabled(enabled=True)
        self._sync_ui_to_model_now()

    # ── Sector slider interactive slots ────────────────────────

    def _apply_interactive_sector(self, sector_deg: float) -> None:
        """Update sector width interactively: cache + indicator, no map clear."""
        self._rh_cache['radar_sector_width_deg'] = sector_deg
        self._status_proxy.show_message(f'Азимутальный угол: {int(sector_deg)}°', 1000)
        az = self._rh_cache.get('radar_azimuth_deg', 0.0)
        self._update_azimuth_indicator(az)
        self._sector_needs_recompute = True

    @Slot(int)
    def _on_sector_slider_changed(self, slider_value: int) -> None:
        """Update label when sector slider is dragged; interactive preview."""
        self.radar_sector_label.setText(f'{slider_value}°')
        if self._is_radar_map_active():
            self._apply_interactive_sector(float(slider_value))
        else:
            self._on_settings_changed()

    @Slot()
    def _on_sector_slider_released(self) -> None:
        """Sector slider released — trigger recompute if needed."""
        self._trigger_sector_recompute()

    def _trigger_sector_recompute(self) -> None:
        """Trigger coverage recompute after sector change."""
        if not self._sector_needs_recompute:
            return
        self._sector_needs_recompute = False
        self._trigger_coverage_recompute()

    # ── Range slider interactive slots ─────────────────────────

    def _apply_interactive_range(self, range_km: float) -> None:
        """Update range interactively: cache + indicator, no map clear."""
        self._rh_cache['radar_max_range_km'] = range_km
        self._status_proxy.show_message(f'Дальность РЛС: {range_km:.1f} км', 1000)
        az = self._rh_cache.get('radar_azimuth_deg', 0.0)
        self._update_azimuth_indicator(az)
        self._range_needs_recompute = True

    @Slot(int)
    def _on_range_slider_changed(self, slider_value: int) -> None:
        """Update label when slider is dragged; interactive preview."""
        km = float(slider_value)
        self.radar_range_label.setText(str(slider_value))
        if self._is_radar_map_active():
            self._apply_interactive_range(km)
        else:
            self._on_settings_changed()

    @Slot()
    def _on_range_slider_released(self) -> None:
        """Slider released — trigger recompute if needed."""
        self._trigger_range_recompute()

    # ── Shared coverage recompute ──────────────────────────────

    def _trigger_coverage_recompute(self) -> None:
        """
        Common recompute logic shared by azimuth, range, sector triggers.

        Antenna position doesn't change — pass DEM coords directly to avoid
        lossy round-trip (DEM→final→DEM) through rotation/crop transforms.
        The recomputed result is now rotated (same as first build), so
        _rh_click_pos (in rotated coords) remains valid.
        """
        if not self._rh_cache or 'dem' not in self._rh_cache:
            return

        ant_row = self._rh_cache.get('antenna_row', 0)
        ant_col = self._rh_cache.get('antenna_col', 0)

        # Keep _rh_click_pos as-is (rotated coords from first build).
        # If not yet set, compute from metadata.
        if self._rh_click_pos is None:
            metadata = self._model.state.last_map_metadata
            if metadata:
                self._rh_click_pos = self._compute_cp_image_pos(metadata)
        if self._rh_click_pos is None:
            return

        px, py = self._rh_click_pos
        self._recompute_coverage_at_click(px, py, dem_row=ant_row, dem_col=ant_col)

    def _dem_to_final_coords(
        self,
        dem_row: int,
        dem_col: int,
    ) -> tuple[float, float]:
        """
        Convert DEM (row, col) to final image (x, y) using crop formula.

        Accounts for center crop offset (rotation padding) in DEM.
        """
        dem = self._rh_cache.get('dem')
        final_size = self._rh_cache.get('final_size')
        crop_size = self._rh_cache.get('crop_size')
        if dem is not None and final_size:
            dem_h, dem_w = dem.shape
            final_w, final_h = final_size
            if crop_size and crop_size[0] > 0 and crop_size[1] > 0:
                cw, ch = crop_size
                fx = dem_col * cw / dem_w - (cw - final_w) / 2.0
                fy = dem_row * ch / dem_h - (ch - final_h) / 2.0
                return (fx, fy)
            return (dem_col * final_w / dem_w, dem_row * final_h / dem_h)
        return (float(dem_col), float(dem_row))

    def _trigger_range_recompute(self) -> None:
        """Trigger coverage recompute after range change."""
        if not self._range_needs_recompute:
            return
        self._range_needs_recompute = False
        self._trigger_coverage_recompute()

    @Slot(float)
    def _rotate_radar_azimuth(self, delta_deg: float) -> None:
        """
        Rotate radar azimuth visually (instant) and mark for recompute.

        Called from Shift+wheel and [ ] keys.
        - Shift+wheel: recompute fires on Shift release.
        - [ ] keys: recompute fires via debounce timer (400 ms).
        """
        metadata = self._model.state.last_map_metadata
        if not metadata or metadata.map_type != MapType.RADAR_COVERAGE:
            return
        if not self._rh_cache or 'dem' not in self._rh_cache:
            return

        # Update azimuth in cache and UI immediately
        current_az = self._rh_cache.get('radar_azimuth_deg', 0.0)
        new_az = (current_az + delta_deg) % 360.0
        self._rh_cache['radar_azimuth_deg'] = new_az

        self.radar_azimuth_label.setText(f'{int(new_az)}°')
        with QSignalBlocker(self.radar_azimuth_dial):
            self.radar_azimuth_dial.setValue(int((540 - new_az) % 360))

        self._status_proxy.show_message(f'Азимут РЛС: {new_az:.0f}°', 1000)

        # Draw azimuth indicator line instantly (lightweight scene overlay)
        self._update_azimuth_indicator(new_az)

        self._azimuth_needs_recompute = True

    @Slot()
    def _on_shift_released_recompute(self) -> None:
        """Shift key released after Shift+wheel rotation — trigger recompute."""
        if not self._azimuth_needs_recompute:
            return
        self._trigger_azimuth_recompute()

    def _trigger_azimuth_recompute(self) -> None:
        """Trigger coverage recompute after azimuth change."""
        if not self._azimuth_needs_recompute:
            return
        self._azimuth_needs_recompute = False
        self._trigger_coverage_recompute()

    def _compute_cp_image_pos(
        self,
        metadata: MapMetadata | None = None,
    ) -> tuple[float, float] | None:
        """
        Compute control point position on the final image using precise coords.

        Uses the same WGS84→pixel transform as _draw_radar_sector_overlay,
        so the azimuth line origin matches the control point marker exactly.
        Falls back to DEM→final linear scaling if metadata is unavailable.
        """
        if metadata is None:
            metadata = self._model.state.last_map_metadata
        settings = self._model.settings

        if metadata and settings and settings.control_point_enabled:
            try:
                transformer = CoordinateTransformer(
                    settings.control_point_x_sk42_gk,
                    settings.control_point_y_sk42_gk,
                    helmert_params=metadata.helmert_params,
                )
                lat_wgs, lng_wgs = transformer.get_wgs84_center()
                px, py = compute_control_point_image_coords(
                    lat_wgs,
                    lng_wgs,
                    metadata.center_lat_wgs,
                    metadata.center_lng_wgs,
                    metadata.zoom,
                    metadata.scale,
                    metadata.width_px,
                    metadata.height_px,
                    metadata.rotation_deg,
                    latlng_to_pixel_xy,
                )
            except Exception:
                logger.debug('Failed to compute CP image pos via coords, falling back')
            else:
                return (px, py)

        # Fallback: DEM→final scaling accounting for center crop
        if self._rh_cache:
            ant_row = self._rh_cache.get('antenna_row', 0)
            ant_col = self._rh_cache.get('antenna_col', 0)
            return self._dem_to_final_coords(ant_row, ant_col)
        return None

    def _get_final_pixel_size_m(self) -> float:
        """
        Get effective pixel size for the final (displayed) image.

        With crop+resize approach, the final pixel size = base_mpp
        = dem_pixel_size_m * dem_w / crop_w (= dem_px / ds_factor).
        """
        dem_px = self._rh_cache.get('pixel_size_m', 1.0)
        dem = self._rh_cache.get('dem')
        crop_size = self._rh_cache.get('crop_size')
        if dem is not None and crop_size:
            dem_w = dem.shape[1]
            crop_w = crop_size[0]
            if crop_w > 0 and dem_w > 0:
                return dem_px * dem_w / crop_w
        return dem_px

    def _update_azimuth_indicator(self, azimuth_deg: float) -> None:
        """
        Draw/update the dashed azimuth line on the preview scene.

        This is a lightweight scene overlay that provides instant visual
        feedback while rotating, before the heavy recompute finishes.
        """
        if not self._rh_cache or not self._rh_click_pos:
            return

        cx, cy = self._rh_click_pos
        pixel_size = self._get_final_pixel_size_m()
        radar_range_km = self._rh_cache.get('radar_max_range_km', 15.0)
        sector_width = self._rh_cache.get('radar_sector_width_deg', 90.0)
        rot_deg = self._rh_cache.get('rotation_deg', 0.0)

        # Length in final-image pixels
        length_px = (radar_range_km * 1000.0) / pixel_size if pixel_size > 0 else 200.0

        self._preview_area.set_azimuth_line(
            cx,
            cy,
            azimuth_deg,
            length_px,
            sector_width_deg=sector_width,
            rotation_deg=rot_deg,
        )

    @Slot(bool, str)
    def _on_download_finished(
        self,
        *,
        success: bool,
        error_msg: str,
    ) -> None:
        """Handle download completion."""
        self._preview_area.stop_loading()

        # Обновить состояние модели через контроллер
        self._controller.complete_download(success=success, error_msg=error_msg)

        # Clear coordinate informer on failure
        if not success:
            self._coords_label.setText('')

        # Hide progress widgets and re-enable controls
        try:
            self._progress_bar.setVisible(False)
            self._progress_label.setVisible(False)
            self._cancel_btn.setVisible(False)
            # Re-enable all UI controls when download is finished
            self._set_controls_enabled(enabled=True)
        except Exception as e:
            logger.debug(f'Failed to hide progress widgets: {e}')

        cancelled = error_msg == 'Операция отменена пользователем'

        if success:
            self._status_proxy.show_message(
                'Карта успешно создана. Правый клик на превью — '
                'перенос координат в КТ.',
                7000,
            )
        elif cancelled:
            self._clear_preview_ui()
            self._status_proxy.show_message('Операция отменена', 3000)
        else:
            # Clear preview and related UI on failure as per requirement
            self._clear_preview_ui()
            self._status_proxy.show_message('Ошибка при создании карты', 5000)
            QMessageBox.critical(
                self,
                'Ошибка',
                f'Не удалось создать карту:\n{error_msg}',
            )

        # Cleanup and drop references to download worker and its signal connections
        try:
            self._cleanup_download_worker()
        except Exception as e:
            logger.debug(f'Failed to cleanup download worker: {e}')

    def _handle_model_event(self, event_data: EventData) -> None:
        """Handle model events (Observer pattern)."""
        event = event_data.event
        data = event_data.data

        if event == ModelEvent.SETTINGS_CHANGED:
            self._update_ui_from_settings(data.get('settings'))
        elif event == ModelEvent.PROFILE_LOADED:
            # After loading a new profile, clear the preview and reset related UI
            try:
                self._preview_area.clear()
                self._current_image = None
                self._base_image = None
                # Disable save controls as no image is present
                self.save_map_btn.setEnabled(False)
                self.save_map_action.setEnabled(False)
            except Exception as e:
                logger.debug(f'Failed to clear preview on profile load: {e}')
            # Update the rest of the UI from profile settings
            self._update_ui_from_settings(data.get('settings'))
            self._status_proxy.show_message(
                f'Профиль загружен: {data.get("profile_name")}',
                3000,
            )
        elif event == ModelEvent.DOWNLOAD_PROGRESS:
            self._update_progress(
                data.get('done', 0),
                data.get('total', 0),
                data.get('label', ''),
            )
        elif event == ModelEvent.PREVIEW_UPDATED:
            self._show_preview_in_main_window(data.get('image'))
            # Также обновляем метаданные в модели, но MainWindow просто реагирует
            # на движение мыши, используя актуальные метаданные из состояния модели.
        elif event == ModelEvent.ERROR_OCCURRED:
            error_msg = data.get('error', 'Неизвестная ошибка')
            # Только статус-бар; модальные диалоги показываются централизованно
            # в _on_download_finished
            self._status_proxy.show_message(f'Ошибка: {error_msg}', 5000)
        elif event == ModelEvent.WARNING_OCCURRED:
            warn_msg = (
                data.get('warning')
                or data.get('message')
                or data.get('error')
                or 'Предупреждение'
            )
            # Только статус-бар; без модальных диалогов, чтобы избежать дублей
            # и вызовов не из GUI-потока
            self._status_proxy.show_message(f'Предупреждение: {warn_msg}', 5000)

    def _get_uav_height_reference(self) -> UavHeightReference:
        """Get current UAV height reference from radio buttons."""
        if self.height_ref_ground_radio.isChecked():
            return UavHeightReference.GROUND
        if self.height_ref_sea_radio.isChecked():
            return UavHeightReference.SEA_LEVEL
        return UavHeightReference.CONTROL_POINT

    def _set_uav_height_reference(self, ref: UavHeightReference) -> None:
        """Set UAV height reference radio buttons from value."""
        with QSignalBlocker(self.height_ref_group):
            if ref == UavHeightReference.GROUND:
                self.height_ref_ground_radio.setChecked(True)
            elif ref == UavHeightReference.SEA_LEVEL:
                self.height_ref_sea_radio.setChecked(True)
            else:
                self.height_ref_cp_radio.setChecked(True)

    def _update_cp_marker_from_settings(self, settings: MapSettings) -> None:
        """Update control point marker and line on preview from settings."""
        if self._current_image is None:
            return

        metadata = self._model.state.last_map_metadata
        if not metadata or not metadata.control_point_enabled:
            # We don't necessarily clear if disabled, but the triangle
            # might not be there on the final map.
            # Requirement says "При каждом обновлении координат КТ проводи... линию"
            return

        if not settings.control_point_enabled:
            # If CP is disabled in settings, clear markers from preview
            if hasattr(self._preview_area, 'clear_control_point_markers'):
                self._preview_area.clear_control_point_markers()
            else:
                # If method doesn't exist, we can at least stop drawing new ones
                pass
            return

        try:
            # 1. Check if CP matches original CP. If so, hide markers and return.
            # We use a small tolerance for float comparison, though these are
            # likely ints.
            orig_x = metadata.original_cp_x_gk or metadata.center_x_gk
            orig_y = metadata.original_cp_y_gk or metadata.center_y_gk
            dx_m = settings.control_point_x_sk42_gk - orig_x
            dy_m = settings.control_point_y_sk42_gk - orig_y
            distance_m = math.sqrt(dx_m**2 + dy_m**2)

            if distance_m < CONTROL_POINT_PRECISION_TOLERANCE_M:
                if hasattr(self._preview_area, 'clear_control_point_markers'):
                    self._preview_area.clear_control_point_markers()
                return

            # Calculate azimuth (standard geographic azimuth: North=0, East=90)
            # dx_m is Easting difference, dy_m is Northing difference
            # atan2(dx, dy) gives angle from North clockwise
            azimuth_rad = math.atan2(dx_m, dy_m)
            azimuth_deg = math.degrees(azimuth_rad) % 360.0

            # Convert control point GK to WGS84
            # We need transformer to convert from GK to WGS84

            transformer = CoordinateTransformer(
                settings.control_point_x_sk42_gk,
                settings.control_point_y_sk42_gk,
                helmert_params=metadata.helmert_params,
            )
            lat_wgs, lng_wgs = transformer.get_wgs84_center()

            px, py = compute_control_point_image_coords(
                lat_wgs,
                lng_wgs,
                metadata.center_lat_wgs,
                metadata.center_lng_wgs,
                metadata.zoom,
                metadata.scale,
                metadata.width_px,
                metadata.height_px,
                metadata.rotation_deg,
                latlng_to_pixel_xy,
            )

            # Update preview
            self._preview_area.set_control_point_marker(px, py)

            if (
                metadata.original_cp_x_gk is not None
                and metadata.original_cp_y_gk is not None
            ):
                # Convert original control point GK to WGS84
                orig_transformer = CoordinateTransformer(
                    metadata.original_cp_x_gk,
                    metadata.original_cp_y_gk,
                    helmert_params=metadata.helmert_params,
                )
                orig_lat_wgs, orig_lng_wgs = orig_transformer.get_wgs84_center()

                # Convert original WGS84 to image pixels

                orig_px, orig_py = compute_control_point_image_coords(
                    orig_lat_wgs,
                    orig_lng_wgs,
                    metadata.center_lat_wgs,
                    metadata.center_lng_wgs,
                    metadata.zoom,
                    metadata.scale,
                    metadata.width_px,
                    metadata.height_px,
                    metadata.rotation_deg,
                    latlng_to_pixel_xy,
                )
                start_x, start_y = orig_px, orig_py

            else:
                # Fallback to center
                start_x = metadata.width_px / 2.0
                start_y = metadata.height_px / 2.0

            self._preview_area.set_control_point_line(
                start_x,
                start_y,
                px,
                py,
                distance_m=distance_m,
                azimuth_deg=azimuth_deg,
                name=getattr(settings, 'control_point_name', ''),
            )
        except Exception:
            logger.exception('Failed to update control point marker from settings')

    def _update_ui_from_settings(self, settings: MapSettings) -> None:
        """Update UI controls from settings object."""
        if not settings or not hasattr(settings, 'from_x_high'):
            return

        # Block feedback to model while we populate controls programmatically
        self._ui_sync_in_progress = True

        # Update coordinates
        self.from_x_widget.set_values(settings.from_x_high, settings.from_x_low)
        self.from_y_widget.set_values(settings.from_y_high, settings.from_y_low)
        self.to_x_widget.set_values(settings.to_x_high, settings.to_x_low)
        self.to_y_widget.set_values(settings.to_y_high, settings.to_y_low)

        # Update grid settings
        grid_settings = {
            'grid_width_m': settings.grid_width_m,
            'grid_font_size_m': settings.grid_font_size_m,
            'grid_text_margin_m': settings.grid_text_margin_m,
            'grid_label_bg_padding_m': settings.grid_label_bg_padding_m,
        }
        self.grid_widget.set_settings(grid_settings)
        self.display_grid_cb.setChecked(settings.display_grid)

        # Update map type combobox and overlay checkbox
        try:
            current_mt = getattr(settings, 'map_type', MapType.SATELLITE)
            if not isinstance(current_mt, MapType):
                current_mt = MapType(str(current_mt))
        except Exception:
            current_mt = MapType.SATELLITE

        # Legacy handling: if profile had ELEVATION_CONTOURS, map to OUTDOORS
        # + overlay enabled
        overlay_flag = bool(getattr(settings, 'overlay_contours', False))
        if current_mt == MapType.ELEVATION_CONTOURS:
            current_mt = MapType.OUTDOORS
            overlay_flag = True

        # find index by userData
        target_index = 0
        for i in range(self.map_type_combo.count()):
            if self.map_type_combo.itemData(i) == current_mt.value:
                target_index = i
                break
        with QSignalBlocker(self.map_type_combo):
            self.map_type_combo.setCurrentIndex(target_index)
        with QSignalBlocker(self.contours_checkbox):
            self.contours_checkbox.setChecked(overlay_flag)

        # Проверяем режимы: Радиогоризонт, РЛС, Карта высот, Профиль радиолинии
        is_radio_horizon = current_mt == MapType.RADIO_HORIZON
        is_radar_coverage = current_mt == MapType.RADAR_COVERAGE
        is_link_profile = current_mt == MapType.LINK_PROFILE
        is_elev_color = current_mt in (
            MapType.ELEVATION_COLOR,
            MapType.ELEVATION_HILLSHADE,
        )

        # Update Helmert settings
        self.helmert_widget.set_values(
            getattr(settings, 'helmert_dx', None),
            getattr(settings, 'helmert_dy', None),
            getattr(settings, 'helmert_dz', None),
            getattr(settings, 'helmert_rx_as', None),
            getattr(settings, 'helmert_ry_as', None),
            getattr(settings, 'helmert_rz_as', None),
            getattr(settings, 'helmert_ds_ppm', None),
        )

        # Update radio horizon overlay alpha
        rh_alpha = float(getattr(settings, 'radio_horizon_overlay_alpha', 0.7))
        rh_alpha_pct = int(rh_alpha * 100)
        with QSignalBlocker(self.radio_horizon_alpha_slider):
            self.radio_horizon_alpha_slider.setValue(rh_alpha_pct)
        self.radio_horizon_alpha_label.setText(f'{rh_alpha_pct}%')

        # Update control point settings
        control_point_enabled = getattr(settings, 'control_point_enabled', False)

        # Для радиогоризонта/РЛС принудительно включаем КТ
        if is_radio_horizon or is_radar_coverage:
            control_point_enabled = True
        elif is_link_profile:
            # Для профиля радиолинии: КТ принудительно выключена и заблокирована
            control_point_enabled = False

        with QSignalBlocker(self.control_point_checkbox):
            self.control_point_checkbox.setChecked(control_point_enabled)

        # Блокируем/разблокируем чекбокс в зависимости от типа карты
        self.control_point_checkbox.setEnabled(
            not is_radio_horizon and not is_radar_coverage and not is_link_profile
        )

        # Programmatically set full control point coordinates without splitting
        # to high/low
        control_point_x = int(getattr(settings, 'control_point_x', 5415000))
        control_point_y = int(getattr(settings, 'control_point_y', 7440000))

        with QSignalBlocker(self.control_point_x_widget.coordinate_edit):
            self.control_point_x_widget.set_coordinate(control_point_x)
        with QSignalBlocker(self.control_point_y_widget.coordinate_edit):
            self.control_point_y_widget.set_coordinate(control_point_y)

        # Control point name
        control_point_name = str(getattr(settings, 'control_point_name', ''))
        with QSignalBlocker(self.control_point_name_edit):
            self.control_point_name_edit.setText(control_point_name)

        # Antenna height for radio horizon
        antenna_height = round(float(getattr(settings, 'antenna_height_m', 10.0)))
        with QSignalBlocker(self.antenna_height_slider):
            self.antenna_height_slider.setValue(antenna_height)
        self.antenna_height_value_label.setText(str(antenna_height))

        # Max flight height for radio horizon
        max_flight_height = int(float(getattr(settings, 'max_flight_height_m', 500.0)))
        with QSignalBlocker(self.max_flight_height_slider):
            self.max_flight_height_slider.setValue(max_flight_height)
        self.max_flight_height_value_label.setText(str(max_flight_height))

        # UAV height reference for radio horizon
        uav_height_ref = getattr(
            settings, 'uav_height_reference', UavHeightReference.CONTROL_POINT
        )
        self._set_uav_height_reference(uav_height_ref)

        # Radar coverage parameters
        _radar_az = float(getattr(settings, 'radar_azimuth_deg', 0.0))
        self.radar_azimuth_label.setText(f'{int(_radar_az)}°')
        with QSignalBlocker(self.radar_azimuth_dial):
            self.radar_azimuth_dial.setValue(int((540 - _radar_az) % 360))
        _sector = int(float(getattr(settings, 'radar_sector_width_deg', 90.0)))
        with QSignalBlocker(self.radar_sector_slider):
            self.radar_sector_slider.setValue(_sector)
        self.radar_sector_label.setText(f'{_sector}°')
        _elev_lo = int(float(getattr(settings, 'radar_elevation_min_deg', 1.0)))
        _elev_hi = int(float(getattr(settings, 'radar_elevation_max_deg', 30.0)))
        with QSignalBlocker(self.radar_elev_slider):
            self.radar_elev_slider.setValue((_elev_lo, _elev_hi))
        self.radar_elev_label.setText(f'{_elev_lo}—{_elev_hi}°')
        range_km = float(getattr(settings, 'radar_max_range_km', 15.0))
        with QSignalBlocker(self.radar_range_slider):
            self.radar_range_slider.setValue(int(range_km))
        self.radar_range_label.setText(str(int(range_km)))
        _th_lo = int(float(getattr(settings, 'radar_target_height_min_m', 30.0)))
        _th_hi = int(float(getattr(settings, 'radar_target_height_max_m', 5000.0)))
        with QSignalBlocker(self.radar_target_h_slider):
            self.radar_target_h_slider.setValue((_th_lo // 10, _th_hi // 10))
        self.radar_target_h_label.setText(f'{_th_lo}—{_th_hi}')

        # Link profile parameters
        _link_ax = int(getattr(settings, 'link_point_a_x', 5415000))
        _link_ay = int(getattr(settings, 'link_point_a_y', 7440000))
        with QSignalBlocker(self.link_a_x_widget.coordinate_edit):
            self.link_a_x_widget.set_coordinate(_link_ax)
        with QSignalBlocker(self.link_a_y_widget.coordinate_edit):
            self.link_a_y_widget.set_coordinate(_link_ay)
        with QSignalBlocker(self.link_a_name_edit):
            self.link_a_name_edit.setText(
                str(getattr(settings, 'link_point_a_name', 'A'))
            )
        _link_bx = int(getattr(settings, 'link_point_b_x', 5420000))
        _link_by = int(getattr(settings, 'link_point_b_y', 7445000))
        with QSignalBlocker(self.link_b_x_widget.coordinate_edit):
            self.link_b_x_widget.set_coordinate(_link_bx)
        with QSignalBlocker(self.link_b_y_widget.coordinate_edit):
            self.link_b_y_widget.set_coordinate(_link_by)
        with QSignalBlocker(self.link_b_name_edit):
            self.link_b_name_edit.setText(
                str(getattr(settings, 'link_point_b_name', 'B'))
            )
        with QSignalBlocker(self.link_freq_spin):
            self.link_freq_spin.setValue(
                float(getattr(settings, 'link_freq_mhz', 900.0))
            )
        with QSignalBlocker(self.link_antenna_a_spin):
            self.link_antenna_a_spin.setValue(
                float(getattr(settings, 'link_antenna_a_m', 10.0))
            )
        with QSignalBlocker(self.link_antenna_b_spin):
            self.link_antenna_b_spin.setValue(
                float(getattr(settings, 'link_antenna_b_m', 10.0))
            )

        # Панель видна для НСУ/РЛС/Карта высот
        self._radio_horizon_group.setVisible(
            is_radio_horizon or is_radar_coverage or is_elev_color
        )

        # Панель профиля радиолинии
        self._link_profile_group.setVisible(is_link_profile)

        # Заголовок панели зависит от режима
        if is_elev_color:
            self._radio_horizon_group.setTitle('')
        else:
            self._radio_horizon_group.setTitle('')

        # Управление видимостью НСУ/РЛС-специфичных виджетов внутри единой панели
        nsu_visible = is_radio_horizon and not is_radar_coverage
        self._nsu_height_ref_widget.setVisible(nsu_visible)
        self._radar_settings_widget.setVisible(
            is_radio_horizon or is_radar_coverage or is_elev_color
        )
        for w in self._radar_only_widgets:
            w.setVisible(is_radar_coverage)
        for w in self._nsu_only_widgets:
            w.setVisible(nsu_visible)
        # Виджеты антенны скрыты для ELEVATION_COLOR
        antenna_visible = is_radio_horizon or is_radar_coverage
        for w in self._antenna_widgets:
            w.setVisible(antenna_visible)

        # Log the values to ensure no truncation occurs during UI update
        try:
            x_text = self.control_point_x_widget.coordinate_edit.text()
            y_text = self.control_point_y_widget.coordinate_edit.text()
            logger.info(
                "UI control point set: enabled=%s, x_src=%d -> edit='%s', "
                "y_src=%d -> edit='%s', antenna_h=%d",
                control_point_enabled,
                control_point_x,
                x_text,
                control_point_y,
                y_text,
                int(antenna_height),
            )
        except Exception:
            logger.exception('Failed to log control point UI update')

        # Enable/disable coordinate inputs based on checkbox state
        self.control_point_x_widget.setEnabled(control_point_enabled)
        self.control_point_y_widget.setEnabled(control_point_enabled)
        self.control_point_name_edit.setEnabled(control_point_enabled)

        # Unblock settings propagation after UI is fully synced
        self._ui_sync_in_progress = False

        # Update markers on preview if image is loaded
        self._update_cp_marker_from_settings(settings)

    def _ensure_busy_dialog(self) -> QProgressDialog:
        """Create BusyDialog lazily to prevent it from showing at startup."""
        if self._busy_dialog is None:
            # Use keyword-only parameters to match PySide6 overload and satisfy mypy
            self._busy_dialog = QProgressDialog(self)
            self._busy_dialog.setLabelText('Подготовка…')
            self._busy_dialog.setRange(0, 0)
            self._busy_dialog.setWindowTitle('Обработка')
            self._busy_dialog.setCancelButton(None)
            # Set both window modality and explicit window flags for modal overlay
            self._busy_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            # Add Dialog flag to ensure proper modal behavior
            self._busy_dialog.setWindowFlags(
                Qt.WindowType.Dialog
                | Qt.WindowType.CustomizeWindowHint
                | Qt.WindowType.WindowTitleHint
            )
            self._busy_dialog.setMinimumDuration(0)
            self._busy_dialog.setAutoClose(False)
            self._busy_dialog.setAutoReset(False)
            self._busy_dialog.hide()
        return self._busy_dialog

    def _update_progress(self, done: int, total: int, label: str) -> None:
        """
        Update progress bar in status bar during long operations.

        When total > 0, shows a determinate progress bar (e.g. tile downloads).
        When total == 0, shows an indeterminate spinner.
        """
        # Show progress widgets if not visible
        if not self._progress_bar.isVisible():
            self._progress_bar.setVisible(True)
            self._progress_label.setVisible(True)
            self._cancel_btn.setVisible(True)
            # Disable all UI controls when showing progress
            self._set_controls_enabled(enabled=False)

        if total > 0:
            # Детерминированный прогресс (загрузка тайлов и т.д.)
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(done)
        # Indeterminate mode - show spinner
        elif self._progress_bar.maximum() != 0:
            self._progress_bar.setRange(0, 0)

        # Update progress label text
        if label:
            self._progress_label.setText(label)

    @Slot(object, object, object, object)
    def _show_preview_in_main_window(
        self,
        image: Image.Image,
        metadata: MapMetadata | None = None,
        dem_grid: object = None,
        rh_cache: dict | None = None,
    ) -> None:
        """Show preview image in the main window's integrated preview area."""
        try:
            if not isinstance(image, Image.Image):
                logger.warning('Invalid image object for preview')
                return

            # Save DEM grid for cursor elevation display
            self._dem_grid = dem_grid

            # Save radio horizon cache for interactive rebuilding
            if rh_cache is not None:
                self._rh_cache = rh_cache
                logger.info('Radio horizon cache saved for interactive rebuilding')
                # Prepare display-sized layers for interactive alpha blending
                self._prepare_rh_display_cache()
                # Compute initial click position using precise coordinate transform
                # (same as _update_cp_marker / _draw_radar_sector)
                self._rh_click_pos = self._compute_cp_image_pos(metadata)
            else:
                self._rh_cache = {}  # Clear cache for non-RH maps

            # Update model with metadata for informer
            self._model.update_preview(None, metadata)

            # Set base image (full size)
            self._base_image = image.convert('RGB') if image.mode != 'RGB' else image

            # Display image
            self._current_image = self._base_image
            mpp = metadata.meters_per_pixel if metadata else 0.0
            self._preview_area.set_image(self._current_image, meters_per_px=mpp)

            # Draw persistent azimuth indicator for RADAR_COVERAGE maps
            if self._rh_cache.get('is_radar_coverage', False):
                az = self._rh_cache.get('radar_azimuth_deg', 0.0)
                self._update_azimuth_indicator(az)

            # Update control point markers from settings (e.g. if loaded from profile)
            self._update_cp_marker_from_settings(self._model.settings)

            # Set tooltip for coordinate informer and preview area
            if metadata and metadata.map_type == MapType.RADIO_HORIZON and rh_cache:
                tooltip = 'ПКМ - перестроение радиогоризонта с новой контрольной точкой'
                self._coords_label.setToolTip(tooltip)
                self._preview_area.setToolTip(tooltip)
            elif metadata and metadata.control_point_enabled:
                tooltip = 'Правая кнопка мыши - установка контрольной точки'
                self._coords_label.setToolTip(tooltip)
                self._preview_area.setToolTip(tooltip)
            else:
                self._coords_label.setToolTip('Текущие координаты курсора')
                self._preview_area.setToolTip('')

            # Enable save button and menu action
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)

            # Hide progress bar and label when preview is ready
            try:
                if self._progress_bar.isVisible():
                    self._progress_bar.setVisible(False)
                    self._progress_label.setVisible(False)
                    self._cancel_btn.setVisible(False)
                    # Re-enable all UI controls when hiding progress
                    self._set_controls_enabled(enabled=True)
                    # Explicitly enable quality slider
                    self._set_sliders_enabled(enabled=True)
            except Exception as _e:
                logger.debug(f'Failed to hide progress widgets on preview: {_e}')

            logger.info('Preview displayed in main window')

        except Exception as e:
            error_msg = f'Ошибка при отображении предпросмотра: {e}'
            logger.exception(error_msg)
            QMessageBox.warning(self, 'Ошибка предпросмотра', error_msg)

    def _clear_preview_ui(self) -> None:
        """Clear preview image, drop pixmap cache, and disable related controls."""
        try:
            self._preview_area.clear()
            # Clear coordinates label and its tooltips
            self._coords_label.setText('')
            self._coords_label.setToolTip('')
            self._preview_area.setToolTip('')
            # Reset images
            self._current_image = None
            self._base_image = None
            # Disable save controls
            self.save_map_btn.setEnabled(False)
            self.save_map_action.setEnabled(False)
            # Disable quality slider when no image is loaded
            self._set_sliders_enabled(enabled=False)
            # Aggressively clear global QPixmap cache between runs
            QPixmapCache.clear()
        except Exception as e:
            logger.debug(f'Failed to clear preview UI: {e}')

    @Slot()
    def _save_map(self) -> None:
        """Save the current map image to file."""
        if self._current_image is None:
            QMessageBox.warning(
                self,
                'Предупреждение',
                'Нет изображения для сохранения',
            )
            return

        # Get file path from user
        maps_dir = Path(__file__).resolve().parent.parent.parent / 'maps'
        maps_dir.mkdir(exist_ok=True)  # Ensure maps directory exists
        default_path = str(maps_dir / 'map.jpg')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Сохранить карту',
            default_path,
            'JPEG files (*.jpg);;All files (*)',
            options=QFileDialog.Option.DontUseCustomDirectoryIcons
            | QFileDialog.Option.DontUseNativeDialog,
        )

        if not file_path:
            return

        try:
            # Disable save button and menu action during saving
            self.save_map_btn.setEnabled(False)
            self.save_map_action.setEnabled(False)

            out_path = Path(file_path)
            logger.info(f'Starting save operation to: {out_path}')

            # Create worker for saving in background thread
            class _SaveWorker(QObject):
                finished = Signal(bool, str)  # success, error_message

                def __init__(
                    self,
                    image: Image.Image,
                    path: Path,
                    quality: int,
                ) -> None:
                    super().__init__()
                    self.image = image
                    self.path = path
                    self.quality = quality

                def run(self) -> None:
                    """Save the image in background thread."""
                    try:
                        img = self.image
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        img.save(
                            str(self.path),
                            'JPEG',
                            quality=self.quality,
                            optimize=True,
                            subsampling=0,
                        )
                        success = True
                        self.finished.emit(success, '')
                    except Exception as e:
                        success = False
                        self.finished.emit(success, str(e))

            # Read quality directly from slider at save time
            quality = self.output_widget.quality_slider.value()
            logger.info(f'Saving map with JPEG quality: {quality}%')

            # Create and setup worker thread
            th = QThread()
            # Use base image to ensure full resolution
            base_for_save = self._base_image or self._current_image
            worker = _SaveWorker(base_for_save, out_path, quality)
            worker.moveToThread(th)

            # Store references for cleanup
            self._save_thread = th
            self._save_worker = worker

            # Setup connections
            th.started.connect(worker.run)

            def _on_save_complete(*, success: bool, err: str) -> None:
                """Handle save completion."""
                logger.info(f'[SAVE_DEBUG] _on_save_complete called: success={success}')

                # Re-enable save button and menu action
                self.save_map_btn.setEnabled(True)
                self.save_map_action.setEnabled(True)

                if success:
                    logger.info(f'Image saved to: {out_path}')
                    self._status_proxy.show_message(
                        f'Карта сохранена: {out_path.name}',
                        5000,
                    )
                    # Preview remains visible, save button remains enabled
                    # User can save again with different quality if needed
                else:
                    logger.error(f'Failed to save image: {err}')
                    err_text = str(err)
                    localized = err_text
                    lower = err_text.lower()
                    if 'already exists' in lower or 'file exists' in lower:
                        localized = 'Файл уже существует'
                    QMessageBox.critical(
                        self,
                        'Ошибка',
                        f'Не удалось сохранить изображение:\n{localized}',
                    )

                # Cleanup resources
                self._cleanup_save_resources()

            worker.finished.connect(
                lambda success, err: _on_save_complete(success=success, err=err),
                Qt.ConnectionType.QueuedConnection,
            )
            # Ensure the worker thread quits immediately after finishing to
            # release resources
            worker.finished.connect(th.quit, Qt.ConnectionType.QueuedConnection)
            th.start()

        except Exception as e:
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)
            error_msg = f'Ошибка при сохранении: {e}'
            logger.exception(error_msg)
            QMessageBox.critical(self, 'Ошибка', error_msg)

    def _cleanup_save_resources(self) -> None:
        """Clean up save operation resources."""
        try:
            # Drop heavy image reference to free memory immediately
            if self._save_worker is not None:
                if hasattr(self._save_worker, 'image'):
                    delattr(self._save_worker, 'image')
                self._save_worker.deleteLater()
                self._save_worker = None

            # Thread cleanup: since worker.finished is connected to thread.quit,
            # the thread should already be stopped when this is called
            if self._save_thread is not None:
                # Thread should already be finished due to quit() signal
                # Just schedule deletion without wait() to avoid issues
                self._save_thread.deleteLater()
                self._save_thread = None

        except Exception:
            logger.exception('Error cleaning up save resources')

    def _set_sliders_enabled(self, *, enabled: bool) -> None:
        """Enable/disable quality slider."""
        # Only quality slider remains, enable it based on parameter
        self.quality_slider.setEnabled(enabled)

    def _set_controls_enabled(self, *, enabled: bool) -> None:
        """Enable/disable all UI controls and menu when progress bar is shown/hidden."""
        # Вся левая панель (включая все QLabel) — одним вызовом
        self._left_panel.setEnabled(enabled)
        # Правая панель (надпись «Предпросмотр карты» и кнопка сохранения)
        self._preview_frame.setEnabled(enabled)
        # Меню
        self.menuBar().setEnabled(enabled)

        if enabled:
            # Восстанавливаем условные состояния после разблокировки
            try:
                idx = max(0, self.map_type_combo.currentIndex())
                map_type_value = self.map_type_combo.itemData(idx)
                is_rh = map_type_value == MapType.RADIO_HORIZON.value
            except Exception:
                is_rh = False
            if is_rh:
                self.control_point_checkbox.setEnabled(False)
            cp_enabled = self.control_point_checkbox.isChecked()
            self.control_point_x_widget.setEnabled(cp_enabled)
            self.control_point_y_widget.setEnabled(cp_enabled)
            self.control_point_name_edit.setEnabled(cp_enabled)
            # Save button depends on whether image is loaded
            has_image = self._base_image is not None
            self.save_map_btn.setEnabled(has_image)
            self.save_map_action.setEnabled(has_image)
        else:
            self.save_map_btn.setEnabled(False)
            self.save_map_action.setEnabled(False)

    @Slot()
    def _new_profile(self) -> None:
        """Create new profile (placeholder)."""
        QMessageBox.information(self, 'Информация', 'Функция создания нового профиля')

    @Slot()
    def _open_profile(self) -> None:
        """Open profile file from disk and load settings."""
        default_dir = ensure_profiles_dir()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Открыть профиль',
            str(default_dir),
            'Файлы профилей (*.toml);;Все файлы (*)',
        )
        if not file_path:
            return
        try:
            profile_name = Path(file_path).stem
            # Use controller method to load from arbitrary path
            self._controller.load_profile_from_path(file_path)
            # Update combo selection if this profile is in the list
            idx = self.profile_combo.findText(profile_name)
            # Обновляем выпадающий список без вызова автозагрузки второй раз
            if idx >= 0:
                self._set_profile_selection_safely(index=idx)
            else:
                blocker = QSignalBlocker(self.profile_combo)
                try:
                    # If it’s not in list, add it temporarily
                    self.profile_combo.addItem(profile_name)
                    self.profile_combo.setCurrentText(profile_name)
                finally:
                    del blocker
            self._status_proxy.show_message(f'Профиль загружен: {profile_name}', 3000)
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', f'Не удалось открыть профиль:\n{e}')

    @Slot()
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            'О программе',
            'SK42 v0.2\n\nПриложение для создания карт в системе Гаусса-Крюгера\n',
        )

    @Slot()
    def _show_api_key_dialog(self) -> None:
        """Show API key management dialog."""
        dialog = ApiKeyDialog(self._controller, parent=self)
        dialog.exec()

    def _show_advanced_settings(self) -> None:
        """Show advanced settings dialog (JPEG quality, Helmert parameters)."""
        dialog = AdvancedSettingsDialog(
            self.output_widget, self.helmert_widget, self.grid_widget, parent=self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            helmert_changed = dialog.apply_to_source()
            if helmert_changed:
                self._on_settings_changed()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle window resize to update overlay size."""
        super().resizeEvent(event)
        # Update overlay when central widget is resized
        if (
            hasattr(self, '_modal_overlay')
            and self._modal_overlay is not None
            and self._modal_overlay.isVisible()
        ):
            self._modal_overlay.resize_to_parent()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        logger.info('Application closing - cleaning up resources')
        log_comprehensive_diagnostics('CLEANUP_START')
        # Clear preview and drop pixmap cache to free GPU/Qt memory
        self._clear_preview_ui()
        QPixmapCache.clear()
        # Also cleanup any lingering download worker connections/callbacks
        self._cleanup_download_worker()

        # Cleanup progress system resources first
        log_memory_usage('before progress cleanup')
        log_thread_status('before progress cleanup')
        # Hide and delete busy dialog if present
        if self._busy_dialog is not None:
            try:
                if self._busy_dialog.isVisible():
                    self._busy_dialog.reset()
                    self._busy_dialog.hide()
            except Exception as e:
                logger.warning(f'Failed to hide busy dialog during cleanup: {e}')
            self._busy_dialog = None
        cleanup_all_progress_resources()
        log_memory_usage('after progress cleanup')
        log_thread_status('after progress cleanup')

        # Cleanup download worker process with timeout
        if self._download_worker and self._download_worker.isRunning():
            logger.info('Terminating download worker process')
            log_thread_status('before worker termination')
            self._download_worker.stop_and_join(timeout_ms=5000)
            log_thread_status('after worker termination')
            log_memory_usage('after worker termination')

        # Cleanup save resources
        self._cleanup_save_resources()
        log_memory_usage('after save resources cleanup')

        # Remove observer
        self._model.remove_observer(self._observer_adapter)

        log_comprehensive_diagnostics('CLEANUP_COMPLETE')
        logger.info('Resource cleanup completed')
        event.accept()


def create_application() -> tuple[
    QApplication,
    MainWindow,
    MilMapperModel,
    MilMapperController,
]:
    """Create and configure the PySide6 application."""
    app = QApplication([])
    apply_theme(app)
    app.setApplicationName('Mil Mapper')
    app.setApplicationVersion('2.0')

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Limit QPixmapCache to prevent uncontrolled memory growth (in KB)
    try:
        QPixmapCache.setCacheLimit(64 * 1024)  # 64 MB
    except Exception as e:
        logger.warning(f'Failed to set QPixmapCache limit: {e}')

    # Create MVC components
    model = MilMapperModel()
    controller = MilMapperController(model)
    window = MainWindow(model, controller)

    return app, window, model, controller

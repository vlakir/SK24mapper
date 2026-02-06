"""PySide6-based View components implementing MVC pattern."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw
from PySide6.QtCore import QObject, QSignalBlocker, Qt, QThread, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QPixmapCache,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from domain.profiles import ensure_profiles_dir
from gui.controller import MilMapperController
from gui.model import EventData, MilMapperModel, ModelEvent, Observer
from gui.preview_window import OptimizedImageView
from gui.status_bar import StatusBarProxy
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
from services.coordinate_transformer import CoordinateTransformer
from services.map_postprocessing import (
    compute_control_point_image_coords,
    draw_control_point_triangle,
)
from services.radio_horizon import recompute_radio_horizon_fast
from shared.constants import (
    CONTROL_POINT_LABEL_GAP_MIN_PX,
    CONTROL_POINT_LABEL_GAP_RATIO,
    CONTROL_POINT_SIZE_M,
    COORDINATE_FORMAT_SPLIT_LENGTH,
    ELEVATION_LEGEND_STEP_M,
    MAP_TYPE_LABELS_RU,
    MIN_DECIMALS_FOR_SMALL_STEP,
    RADIO_HORIZON_COLOR_RAMP,
    MapType,
    UavHeightReference,
)
from shared.diagnostics import (
    log_comprehensive_diagnostics,
    log_memory_usage,
    log_thread_status,
)
from shared.progress import (
    cleanup_all_progress_resources,
    set_preview_image_callback,
    set_spinner_callbacks,
)
from shared.progress import set_progress_callback as _set_prog

if TYPE_CHECKING:
    from collections.abc import Callable

    from domain.models import MapMetadata, MapSettings


import math

from pyproj import Transformer

from geo.coords_sk42 import build_sk42_gk_crs, determine_zone
from geo.topography import (
    build_transformers_sk42,
    crs_sk42_geog,
    latlng_to_pixel_xy,
    pixel_xy_to_latlng,
)
from shared.constants import CONTROL_POINT_PRECISION_TOLERANCE_M

logger = logging.getLogger(__name__)


class _ViewObserver(Observer):
    """Adapter to avoid QWidget.update signature clash with Observer.update."""

    def __init__(self, handler: Callable[[EventData], None]) -> None:
        self._handler = handler

    def update(self, event_data: EventData) -> None:
        self._handler(event_data)


class RadioHorizonRecomputeWorker(QThread):
    """Worker thread for radio horizon recomputation."""

    finished = Signal(
        Image.Image, int, int
    )  # result_image, new_antenna_row, new_antenna_col
    error = Signal(str)  # error_message

    def __init__(
        self,
        rh_cache: dict[str, Any],
        new_antenna_row: int,
        new_antenna_col: int,
    ) -> None:
        super().__init__()
        self._rh_cache = rh_cache
        self._new_antenna_row = new_antenna_row
        self._new_antenna_col = new_antenna_col

    def run(self) -> None:
        """Execute recomputation in background thread."""
        try:
            start_time = time.monotonic()
            logger.info(
                'RadioHorizonRecomputeWorker: recomputing with antenna at (%d, %d)',
                self._new_antenna_col,
                self._new_antenna_row,
            )

            # Step 1: Recompute radio horizon (with resize inside if needed)
            step_start = time.monotonic()
            final_size = self._rh_cache.get('final_size')
            result_image = recompute_radio_horizon_fast(
                dem=self._rh_cache['dem'],
                new_antenna_row=self._new_antenna_row,
                new_antenna_col=self._new_antenna_col,
                antenna_height_m=self._rh_cache['antenna_height_m'],
                pixel_size_m=self._rh_cache['pixel_size_m'],
                topo_base=self._rh_cache['topo_base'],
                overlay_alpha=self._rh_cache['overlay_alpha'],
                max_height_m=self._rh_cache['max_height_m'],
                uav_height_reference=self._rh_cache['uav_height_reference'],
                final_size=final_size,
            )
            step_elapsed = time.monotonic() - step_start
            logger.info(
                '  └─ Recompute radio horizon (with resize): %.3f sec', step_elapsed
            )

            total_elapsed = time.monotonic() - start_time
            logger.info(
                'RadioHorizonRecomputeWorker: recomputation completed in %.3f sec',
                total_elapsed,
            )
            self.finished.emit(
                result_image, self._new_antenna_row, self._new_antenna_col
            )

        except Exception as e:
            logger.exception('RadioHorizonRecomputeWorker failed')
            self.error.emit(str(e))


class GridSettingsWidget(QWidget):
    """Widget for grid configuration settings."""

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup grid settings UI."""
        layout = QGridLayout()

        # Display grid checkbox - FIRST
        self.display_grid_cb = QCheckBox('Выводить сетку')
        self.display_grid_cb.setChecked(True)
        self.display_grid_cb.setToolTip(
            'Если включено: рисуются линии сетки и подписи.\n'
            'Если выключено: рисуются только крестики в точках пересечения '
            'без подписей.'
        )
        layout.addWidget(self.display_grid_cb, 0, 0, 1, 2)  # Растянуть на 2 колонки

        # Connect checkbox to enable/disable handler
        self.display_grid_cb.toggled.connect(
            lambda checked: self._on_display_grid_toggled(checked=checked)
        )

        # Grid width (in meters)
        self.width_label = QLabel('Толщина линий (м):')
        layout.addWidget(self.width_label, 1, 0)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 50.0)
        self.width_spin.setValue(5.0)
        self.width_spin.setSingleStep(1.0)
        self.width_spin.setDecimals(1)
        self.width_spin.setToolTip(
            'Толщина линий сетки в метрах (пересчитывается в пиксели по масштабу карты)'
        )
        layout.addWidget(self.width_spin, 1, 1)

        # Font size (in meters)
        self.font_label = QLabel('Размер шрифта (м):')
        layout.addWidget(self.font_label, 2, 0)
        self.font_spin = QDoubleSpinBox()
        self.font_spin.setRange(10.0, 500.0)
        self.font_spin.setValue(100.0)
        self.font_spin.setSingleStep(10.0)
        self.font_spin.setDecimals(1)
        self.font_spin.setToolTip('Размер шрифта подписей координат в метрах')
        layout.addWidget(self.font_spin, 2, 1)

        # Text margin (in meters)
        self.margin_label = QLabel('Отступ текста (м):')
        layout.addWidget(self.margin_label, 3, 0)
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 200.0)
        self.margin_spin.setValue(50.0)
        self.margin_spin.setSingleStep(5.0)
        self.margin_spin.setDecimals(1)
        self.margin_spin.setToolTip('Отступ подписи от края изображения в метрах')
        layout.addWidget(self.margin_spin, 3, 1)

        # Label background padding (in meters)
        self.padding_label = QLabel('Отступ фона (м):')
        layout.addWidget(self.padding_label, 4, 0)
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 100.0)
        self.padding_spin.setValue(10.0)
        self.padding_spin.setSingleStep(1.0)
        self.padding_spin.setDecimals(1)
        self.padding_spin.setToolTip(
            'Внутренний отступ подложки вокруг текста в метрах'
        )
        layout.addWidget(self.padding_spin, 4, 1)

        self.setLayout(layout)

    def _on_display_grid_toggled(self, *, checked: bool) -> None:
        """Enable/disable grid parameters based on display_grid checkbox state."""
        self.width_label.setEnabled(checked)
        self.width_spin.setEnabled(checked)
        self.font_label.setEnabled(checked)
        self.font_spin.setEnabled(checked)
        self.margin_label.setEnabled(checked)
        self.margin_spin.setEnabled(checked)
        self.padding_label.setEnabled(checked)
        self.padding_spin.setEnabled(checked)

    def get_settings(self) -> dict[str, float | bool]:
        """Get grid settings as dictionary."""
        return {
            'grid_width_m': self.width_spin.value(),
            'grid_font_size_m': self.font_spin.value(),
            'grid_text_margin_m': self.margin_spin.value(),
            'grid_label_bg_padding_m': self.padding_spin.value(),
            'display_grid': self.display_grid_cb.isChecked(),
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
        with QSignalBlocker(self.display_grid_cb):
            self.display_grid_cb.setChecked(bool(settings.get('display_grid', True)))

        # Manually trigger enable/disable logic since signal was blocked
        self._on_display_grid_toggled(checked=self.display_grid_cb.isChecked())


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
        info.setStyleSheet('color: #555;')
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
            # Ensure the thread is stopped
            if self._download_worker.isRunning():
                self._download_worker.quit()
                self._download_worker.wait(1000)
            # Delete later and drop reference
            self._download_worker.deleteLater()
        finally:
            self._download_worker = None

    def _setup_ui(self) -> None:
        """Setup the main window UI."""
        self.setWindowTitle('SK42mapper')
        # Используем минимальный размер и возможность свободно менять размер окна
        self.setMinimumSize(900, 500)
        # Предпочитаемый стартовый размер (не фиксированный)
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
        right_widget = QWidget()
        right_widget.setLayout(right_container)

        # Меню
        self._create_menu()

        # Статус-бар
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._status_proxy = StatusBarProxy(self.status_bar)

        # Прогресс-бар в левом нижнем углу (в статус-баре)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setMaximumHeight(20)
        self._progress_bar.setVisible(False)
        self._progress_bar.setRange(0, 0)
        self._progress_label = QLabel()
        self._progress_label.setVisible(False)
        # Добавляем виджеты прогресса в левую часть статус-бара
        self.status_bar.addWidget(self._progress_label)
        self.status_bar.addWidget(self._progress_bar)

        # Метка для координат СК-42 (в правую часть статус-бара)
        self._coords_label = QLabel()
        self._coords_label.setStyleSheet(
            'font-family: monospace; font-weight: bold; color: red;'
        )
        self.status_bar.addPermanentWidget(self._coords_label)

        # Блок профилей
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel('Профиль:'))

        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip('Выберите профиль настроек')
        profile_layout.addWidget(self.profile_combo)

        self.save_profile_btn = QPushButton('Сохранить')
        self.save_profile_btn.setToolTip('Сохранить текущие настройки в профиль')
        profile_layout.addWidget(self.save_profile_btn)

        self.save_profile_as_btn = QPushButton('Сохранить как...')
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
        from_title.setStyleSheet('padding: 5px;')
        from_layout.addWidget(from_title)
        from_layout.addWidget(self.from_x_widget)
        from_layout.addWidget(self.from_y_widget)
        from_group.setLayout(from_layout)
        panels_layout.addWidget(from_group)

        to_group = QFrame()
        to_group.setFrameStyle(QFrame.Shape.StyledPanel)
        to_layout = QVBoxLayout()
        to_title = QLabel('Правый верхний угол')
        to_title.setStyleSheet('padding: 5px;')
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

        self.control_point_checkbox = QCheckBox('Контрольная точка (НСУ)')
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

        # Высота антенны (для карты радиогоризонта)
        antenna_row = QHBoxLayout()
        self.antenna_height_label = QLabel('Высота антенны (м):')
        self.antenna_height_spin = QDoubleSpinBox()
        self.antenna_height_spin.setRange(0.0, 500.0)
        self.antenna_height_spin.setDecimals(0)
        self.antenna_height_spin.setValue(10.0)
        self.antenna_height_spin.setSingleStep(1.0)
        self.antenna_height_spin.setToolTip(
            'Высота антенны над поверхностью земли (для карты радиогоризонта)'
        )
        antenna_row.addWidget(self.antenna_height_label)
        antenna_row.addWidget(self.antenna_height_spin)

        # Максимальная высота полёта (для карты радиогоризонта)
        max_flight_row = QHBoxLayout()
        self.max_flight_height_label = QLabel('Практический потолок БпЛА (м):')
        self.max_flight_height_spin = QDoubleSpinBox()
        self.max_flight_height_spin.setRange(10.0, 5000.0)
        self.max_flight_height_spin.setDecimals(0)
        self.max_flight_height_spin.setValue(500.0)
        self.max_flight_height_spin.setSingleStep(50.0)
        self.max_flight_height_spin.setToolTip(
            'Максимальная высота полёта для цветовой шкалы радиогоризонта.\n'
            'Значения выше будут отображаться серым цветом.'
        )
        max_flight_row.addWidget(self.max_flight_height_label)
        max_flight_row.addWidget(self.max_flight_height_spin)

        name_row = QHBoxLayout()
        name_row.addWidget(self.control_point_checkbox)
        name_row.addWidget(self.control_point_name_label)
        name_row.addWidget(self.control_point_name_edit)

        coords_row = QHBoxLayout()
        coords_row.addWidget(self.control_point_x_widget)
        coords_row.addWidget(self.control_point_y_widget)

        heights_row = QHBoxLayout()
        heights_row.addLayout(antenna_row)
        heights_row.addLayout(max_flight_row)

        # Режим отсчёта высоты БпЛА (для карты радиогоризонта)
        height_ref_row = QHBoxLayout()
        self.height_ref_label = QLabel('Отсчёт высоты БпЛА:')
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

        height_ref_row.addWidget(self.height_ref_label)
        height_ref_row.addWidget(self.height_ref_cp_radio)
        height_ref_row.addWidget(self.height_ref_ground_radio)
        height_ref_row.addWidget(self.height_ref_sea_radio)
        height_ref_row.addStretch()

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
        settings_main_layout.addWidget(QLabel('Настройки'))

        # Тип карты и чекбокс изолиний (над вкладками, всегда видны)
        maptype_row = QHBoxLayout()
        maptype_label = QLabel('Тип карты:')
        self.map_type_combo = QComboBox()

        self._maptype_order = [
            MapType.SATELLITE,
            MapType.HYBRID,
            # MapType.STREETS,
            MapType.OUTDOORS,
            MapType.ELEVATION_COLOR,
            # MapType.ELEVATION_HILLSHADE,
            MapType.RADIO_HORIZON,
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
        maptype_row.addWidget(maptype_label)
        maptype_row.addWidget(self.map_type_combo, 1)
        maptype_row.addSpacing(8)
        maptype_row.addWidget(self.contours_checkbox)
        settings_main_layout.addLayout(maptype_row)

        # Создаём вкладки
        settings_tabs = QTabWidget()

        # === Вкладка "Основные" ===
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout()

        settings_horizontal_layout = QHBoxLayout()

        grid_frame = QFrame()
        grid_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        grid_layout = QVBoxLayout()
        grid_frame.setLayout(grid_layout)
        self.grid_widget = GridSettingsWidget()
        grid_layout.addWidget(self.grid_widget)
        settings_horizontal_layout.addWidget(grid_frame)

        output_frame = QFrame()
        output_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        output_layout = QVBoxLayout()
        output_frame.setLayout(output_layout)
        self.output_widget = OutputSettingsWidget()
        output_layout.addWidget(self.output_widget)
        # Alias for quality slider
        self.quality_slider = self.output_widget.quality_slider

        settings_horizontal_layout.addWidget(output_frame)
        main_tab_layout.addLayout(settings_horizontal_layout)
        main_tab_layout.addStretch()
        main_tab.setLayout(main_tab_layout)
        settings_tabs.addTab(main_tab, 'Основные')

        # === Вкладка "Опции" ===
        options_tab = QWidget()
        options_tab_layout = QVBoxLayout()

        # Настройки радиогоризонта
        radio_horizon_group = QGroupBox('Радиогоризонт')
        radio_horizon_layout = QVBoxLayout()

        # Высота антенны и практический потолок
        radio_horizon_layout.addLayout(heights_row)

        # Режим отсчёта высоты БпЛА
        radio_horizon_layout.addLayout(height_ref_row)

        # Прозрачность слоя
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel('Прозрачность слоя радиогоризонта:'))
        self.radio_horizon_alpha_spin = QDoubleSpinBox()
        self.radio_horizon_alpha_spin.setRange(0.0, 1.0)
        self.radio_horizon_alpha_spin.setSingleStep(0.05)
        self.radio_horizon_alpha_spin.setDecimals(2)
        self.radio_horizon_alpha_spin.setValue(0.3)
        self.radio_horizon_alpha_spin.setToolTip(
            '0 = топооснова не видна, 1 = чистая топооснова'
        )
        self.radio_horizon_alpha_spin.valueChanged.connect(self._on_settings_changed)
        alpha_row.addWidget(self.radio_horizon_alpha_spin)
        alpha_row.addStretch()
        radio_horizon_layout.addLayout(alpha_row)

        radio_horizon_group.setLayout(radio_horizon_layout)
        options_tab_layout.addWidget(radio_horizon_group)

        # Датум-трансформация
        self.helmert_group = QGroupBox('Датум-трансформация СК-42 → WGS84 (Helmert)')
        helmert_group_layout = QVBoxLayout()
        self.helmert_widget = HelmertSettingsWidget()
        helmert_group_layout.addWidget(self.helmert_widget)
        self.helmert_group.setLayout(helmert_group_layout)
        options_tab_layout.addWidget(self.helmert_group)

        options_tab_layout.addStretch()
        options_tab.setLayout(options_tab_layout)
        settings_tabs.addTab(options_tab, 'Опции')

        settings_main_layout.addWidget(settings_tabs)
        settings_container.setLayout(settings_main_layout)
        left_container.addWidget(settings_container)

        # Растяжка перед кнопкой создания карты
        left_container.addStretch()

        # Кнопка "Создать карту"
        self.download_btn = QPushButton('Создать карту')
        self.download_btn.setToolTip('Начать создание карты')
        self.download_btn.setStyleSheet('QPushButton { font-weight: bold; }')
        self.download_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        left_container.addWidget(self.download_btn)

        # Оборачиваем левую колонку в QScrollArea для предотвращения обрезания контента
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left_widget)

        # Превью справа
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout()
        preview_frame.setLayout(preview_layout)

        preview_layout.addWidget(QLabel('Предпросмотр карты:'))

        self._preview_area = OptimizedImageView()
        self._preview_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._preview_area.setMinimumHeight(220)
        self._preview_area.setMinimumWidth(300)
        preview_layout.addWidget(self._preview_area, 1)

        right_container.addWidget(preview_frame, 1)

        preview_layout.addSpacing(10)

        # Кнопка "Сохранить карту"
        self.save_map_btn = QPushButton('Сохранить карту')
        self.save_map_btn.setStyleSheet('QPushButton { font-weight: bold; }')
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
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        # Prevent left panel from collapsing too small
        left_min = max(300, left_widget.sizeHint().width())
        left_scroll.setMinimumWidth(left_min)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([left_min + 100, 600])

        # Добавляем splitter вместо двух виджетов
        main_layout.addWidget(splitter, 1)

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
            widget.high_spin.valueChanged.connect(self._on_settings_changed)
            widget.low_spin.valueChanged.connect(self._on_settings_changed)

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
        self.antenna_height_spin.valueChanged.connect(self._on_settings_changed)
        # Max flight height for radio horizon
        self.max_flight_height_spin.valueChanged.connect(self._on_settings_changed)
        # UAV height reference radios
        self.height_ref_group.buttonClicked.connect(self._on_settings_changed)

        # Grid settings
        self.grid_widget.width_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.font_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.margin_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.padding_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.display_grid_cb.stateChanged.connect(self._on_settings_changed)

        # Helmert settings
        self.helmert_widget.enable_cb.toggled.connect(self._on_settings_changed)
        self.helmert_widget.dx.valueChanged.connect(self._on_settings_changed)
        self.helmert_widget.dy.valueChanged.connect(self._on_settings_changed)
        self.helmert_widget.dz.valueChanged.connect(self._on_settings_changed)
        self.helmert_widget.rx.valueChanged.connect(self._on_settings_changed)
        self.helmert_widget.ry.valueChanged.connect(self._on_settings_changed)
        self.helmert_widget.rz.valueChanged.connect(self._on_settings_changed)
        self.helmert_widget.ds.valueChanged.connect(self._on_settings_changed)

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
        except Exception:
            is_radio_horizon = False

        if is_radio_horizon:
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

        # Delegate to the common settings handler to store the new map type in the model
        self._on_settings_changed()

    def _sync_ui_to_model_now(self) -> None:
        """
        Force-collect current UI settings and push them to the model without guards.

        Does not clear preview or check _ui_sync_in_progress to avoid losing changes
        during Save/Save As.
        """
        # Collect all current settings
        coords = self._get_current_coordinates()
        grid_settings = self.grid_widget.get_settings()
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
        payload['radio_horizon_overlay_alpha'] = self.radio_horizon_alpha_spin.value()
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
            'antenna_height_m': round(self.antenna_height_spin.value()),
            'max_flight_height_m': self.max_flight_height_spin.value(),
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

        # Clear previous preview and pixmap cache between runs
        self._clear_preview_ui()

        # Cleanup any stale worker from previous run
        self._cleanup_download_worker()

        self._download_worker = DownloadWorker(self._controller)
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
        # Show progress bar in status bar (will be updated by progress callbacks)
        self._progress_bar.setVisible(True)
        self._progress_label.setVisible(True)
        self._progress_label.setText('Подготовка…')
        self._progress_bar.setRange(0, 0)
        # Disable all UI controls during download
        self._set_controls_enabled(enabled=False)

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
        """Transfer coordinates from map right-click to control point settings or rebuild radio horizon."""
        metadata = self._model.state.last_map_metadata
        if not metadata:
            return

        # Check if this is a radio horizon map with cache available
        is_radio_horizon = (
            metadata.map_type == MapType.RADIO_HORIZON
            and self._rh_cache
            and 'dem' in self._rh_cache
        )

        if is_radio_horizon:
            # Interactive radio horizon rebuilding
            self._recompute_radio_horizon_at_click(px, py)
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
        """Recompute radio horizon with new control point at clicked position."""
        # Stop any running recompute worker
        if self._rh_worker is not None and self._rh_worker.isRunning():
            logger.info('Stopping previous radio horizon recompute worker')
            self._rh_worker.quit()
            self._rh_worker.wait(1000)
            self._rh_worker = None

        # Convert pixel coordinates to DEM coordinates
        # px, py are in scene (final image) coordinates
        # We need to convert to DEM row/col considering downsampling
        dem = self._rh_cache.get('dem')
        final_size = self._rh_cache.get('final_size')
        if dem is None:
            logger.warning('No DEM in cache, cannot recompute')
            return

        dem_h, dem_w = dem.shape

        # Calculate downsampling factor from DEM size and final size
        if final_size:
            final_w, final_h = final_size
            scale_x = dem_w / final_w
            scale_y = dem_h / final_h
            # Convert final image coordinates to DEM coordinates
            new_antenna_col = int(px * scale_x)
            new_antenna_row = int(py * scale_y)
        else:
            # No downsampling, coordinates are the same
            new_antenna_col = int(px)
            new_antenna_row = int(py)

        # Clamp to DEM bounds
        new_antenna_row = max(0, min(new_antenna_row, dem_h - 1))
        new_antenna_col = max(0, min(new_antenna_col, dem_w - 1))

        logger.info(
            'Starting radio horizon recompute: scene pos (%.1f, %.1f) -> DEM pos (%d, %d)',
            px,
            py,
            new_antenna_col,
            new_antenna_row,
        )

        # Show wait cursor and status message
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._status_proxy.show_message('Пересчет радиогоризонта...', 0)

        # Store click position for later use (marker and coordinates update)
        self._rh_click_pos = (px, py)

        # Create and start worker
        self._rh_worker = RadioHorizonRecomputeWorker(
            self._rh_cache, new_antenna_row, new_antenna_col
        )
        self._rh_worker.finished.connect(self._on_radio_horizon_recompute_finished)
        self._rh_worker.error.connect(self._on_radio_horizon_recompute_error)
        self._rh_worker.start()

    @Slot(Image.Image, int, int)
    def _on_radio_horizon_recompute_finished(
        self, result_image: Image.Image, new_antenna_row: int, new_antenna_col: int
    ) -> None:
        """Handle radio horizon recompute completion."""
        try:
            gui_start_time = time.monotonic()
            logger.info(
                'Radio horizon recompute finished successfully - applying postprocessing'
            )

            # Update cache with new antenna position
            self._rh_cache['antenna_row'] = new_antenna_row
            self._rh_cache['antenna_col'] = new_antenna_col

            # Apply postprocessing (grid, legend, contours)
            metadata = self._model.state.last_map_metadata
            mpp = metadata.meters_per_pixel if metadata else 0.0

            # Apply cached overlay layer if available, otherwise draw manually
            step_start = time.monotonic()
            overlay_layer = self._rh_cache.get('overlay_layer')
            if overlay_layer:
                # Use cached overlay (grid + legend + contours)
                result_image = result_image.convert('RGBA')
                result_image = Image.alpha_composite(result_image, overlay_layer)
                step_elapsed = time.monotonic() - step_start
                logger.info('  └─ Apply cached overlay layer: %.3f sec', step_elapsed)
            else:
                # Fallback: draw grid and legend manually
                if self._rh_cache.get('settings'):
                    settings = self._rh_cache['settings']
                    self._draw_rh_grid(result_image, settings, mpp)
                self._draw_rh_legend(result_image, mpp)
                step_elapsed = time.monotonic() - step_start
                logger.info(
                    '  └─ Draw grid and legend manually: %.3f sec', step_elapsed
                )

            # Draw control point triangle and label on the image (like in first build)
            step_start = time.monotonic()
            if hasattr(self, '_rh_click_pos') and self._rh_click_pos is not None:
                px, py = self._rh_click_pos

                if mpp > 0:
                    # Draw triangle
                    draw_control_point_triangle(
                        result_image, px, py, mpp, rotation_deg=0.0
                    )

                    # Draw label (name + height)
                    self._draw_rh_control_point_label(result_image, px, py, mpp)

                # Calculate and update SK-42 coordinates
                coords = self._calculate_sk42_from_scene_pos(px, py)
                if coords:
                    x_val, y_val = coords
                    self.control_point_x_widget.set_coordinate(x_val)
                    self.control_point_y_widget.set_coordinate(y_val)

                    # Update model
                    self._sync_ui_to_model_now()

                    logger.info('Control point updated to X=%d, Y=%d', x_val, y_val)
            step_elapsed = time.monotonic() - step_start
            logger.info('  └─ Draw control point marker: %.3f sec', step_elapsed)

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

            # Restore cursor and show success message
            QApplication.restoreOverrideCursor()

            gui_total_elapsed = time.monotonic() - gui_start_time
            logger.info('GUI postprocessing total: %.3f sec', gui_total_elapsed)
            self._status_proxy.show_message('Радиогоризонт пересчитан', 2000)

        except Exception as e:
            logger.exception('Failed to update preview after radio horizon recompute')
            QApplication.restoreOverrideCursor()
            self._status_proxy.show_message(f'Ошибка при обновлении превью: {e}', 5000)
        finally:
            # Clean up worker
            if self._rh_worker is not None:
                self._rh_worker.deleteLater()
                self._rh_worker = None

    def _draw_rh_grid(
        self,
        result: Image.Image,
        settings: Any,
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

            max_height_m = settings.max_flight_height_m
            ppm = 1.0 / mpp
            font_size_px = max(12, round(settings.grid_font_size_m * ppm))
            label_bg_padding_px = max(2, round(settings.grid_label_bg_padding_m * ppm))

            draw_elevation_legend(
                result,
                min_value=0.0,
                max_value=max_height_m,
                color_ramp=RADIO_HORIZON_COLOR_RAMP,
                unit_label='м',
                step=ELEVATION_LEGEND_STEP_M,
                font_size_px=font_size_px,
                label_bg_padding_px=label_bg_padding_px,
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

            # Position below triangle
            tri_size_px = max(5, round(CONTROL_POINT_SIZE_M * ppm))
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
            # Get elevation from DEM cache
            dem = self._rh_cache.get('dem')
            cp_elev = None
            if dem is not None:
                antenna_row = self._rh_cache.get('antenna_row')
                antenna_col = self._rh_cache.get('antenna_col')
                if antenna_row is not None and antenna_col is not None:
                    dem_h, dem_w = dem.shape
                    if 0 <= antenna_row < dem_h and 0 <= antenna_col < dem_w:
                        cp_elev = float(dem[antenna_row, antenna_col])

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
        self._status_proxy.show_message(f'Ошибка пересчета: {error_msg}', 5000)

        # Clean up worker
        if self._rh_worker is not None:
            self._rh_worker.deleteLater()
            self._rh_worker = None

    @Slot(bool, str)
    def _on_download_finished(
        self,
        *,
        success: bool,
        error_msg: str,
    ) -> None:
        """Handle download completion."""
        # Clear coordinate informer on failure
        if not success:
            self._coords_label.setText('')

        # Hide progress widgets and re-enable controls
        try:
            if self._progress_bar.isVisible():
                self._progress_bar.setVisible(False)
                self._progress_label.setVisible(False)
            # Re-enable all UI controls when download is finished
            self._set_controls_enabled(enabled=True)
        except Exception as e:
            logger.debug(f'Failed to hide progress widgets: {e}')

        if success:
            self._status_proxy.show_message(
                'Карта успешно создана. Правый клик на превью — '
                'перенос координат в КТ.',
                7000,
            )
        else:
            # Clear preview and related UI on failure as per requirement
            self._clear_preview_ui()
            self._status_proxy.show_message('Ошибка при создании карты', 5000)
            QMessageBox.critical(
                self,
                'Ошибка',
                f'Не удалось создать карту:\n{error_msg}',
            )

        # Disconnect progress/preview callbacks to avoid holding images between runs
        try:
            set_preview_image_callback(None)
            set_spinner_callbacks(None, None)
            _set_prog(None)
        except Exception as e:
            logger.debug(f'Failed to reset progress callbacks: {e}')

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
            'display_grid': settings.display_grid,
        }
        self.grid_widget.set_settings(grid_settings)

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

        # Проверяем, является ли текущий тип карты "Радиогоризонт"
        is_radio_horizon = current_mt == MapType.RADIO_HORIZON

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
        with QSignalBlocker(self.radio_horizon_alpha_spin):
            self.radio_horizon_alpha_spin.setValue(rh_alpha)

        # Update control point settings
        control_point_enabled = getattr(settings, 'control_point_enabled', False)

        # Для радиогоризонта принудительно включаем контрольную точку
        if is_radio_horizon:
            control_point_enabled = True

        with QSignalBlocker(self.control_point_checkbox):
            self.control_point_checkbox.setChecked(control_point_enabled)

        # Блокируем/разблокируем чекбокс в зависимости от типа карты
        self.control_point_checkbox.setEnabled(not is_radio_horizon)

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
        antenna_height = float(getattr(settings, 'antenna_height_m', 10.0))
        antenna_height = round(antenna_height)
        with QSignalBlocker(self.antenna_height_spin):
            self.antenna_height_spin.setValue(antenna_height)

        # Max flight height for radio horizon
        max_flight_height = float(getattr(settings, 'max_flight_height_m', 500.0))
        with QSignalBlocker(self.max_flight_height_spin):
            self.max_flight_height_spin.setValue(max_flight_height)

        # UAV height reference for radio horizon
        uav_height_ref = getattr(
            settings, 'uav_height_reference', UavHeightReference.CONTROL_POINT
        )
        self._set_uav_height_reference(uav_height_ref)

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

        Shows progress bar and label in the lower left corner of the main window.
        Progress bar always operates in indeterminate (spinner) mode.
        """
        _ = (done, total)
        # Show progress widgets if not visible
        if not self._progress_bar.isVisible():
            self._progress_bar.setVisible(True)
            self._progress_label.setVisible(True)
            # Disable all UI controls when showing progress
            self._set_controls_enabled(enabled=False)

        # Always use indeterminate progress (spinner mode) - no specific progress
        # indication
        self._progress_bar.setRange(0, 0)

        # Update progress label text
        if label:
            self._progress_label.setText(label)

    @Slot(object, object, object)
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

            # Update control point markers from settings (e.g. if loaded from profile)
            self._update_cp_marker_from_settings(self._model.settings)

            # Set tooltip for coordinate informer and preview area
            if metadata and metadata.map_type == MapType.RADIO_HORIZON and rh_cache:
                tooltip = 'Правая кнопка мыши - перестроение радиогоризонта с новой контрольной точкой'
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
        # Top menu bar
        self.menuBar().setEnabled(enabled)

        # Profile controls
        self.profile_combo.setEnabled(enabled)
        self.save_profile_btn.setEnabled(enabled)
        self.save_profile_as_btn.setEnabled(enabled)

        # Coordinate input widgets
        self.from_x_widget.setEnabled(enabled)
        self.from_y_widget.setEnabled(enabled)
        self.to_x_widget.setEnabled(enabled)
        self.to_y_widget.setEnabled(enabled)

        # Control point checkbox and widgets
        # Для типа карты "Радиогоризонт" чекбокс должен оставаться заблокированным
        try:
            idx = max(0, self.map_type_combo.currentIndex())
            map_type_value = self.map_type_combo.itemData(idx)
            is_radio_horizon = map_type_value == MapType.RADIO_HORIZON.value
        except Exception:
            is_radio_horizon = False

        if is_radio_horizon:
            # Для радиогоризонта чекбокс всегда включён и заблокирован
            self.control_point_checkbox.setEnabled(False)
        else:
            self.control_point_checkbox.setEnabled(enabled)

        # Control point coordinate widgets should respect checkbox state
        if enabled:
            cp_enabled = self.control_point_checkbox.isChecked()
            self.control_point_x_widget.setEnabled(cp_enabled)
            self.control_point_y_widget.setEnabled(cp_enabled)
            self.control_point_name_edit.setEnabled(cp_enabled)
        else:
            self.control_point_x_widget.setEnabled(False)
            self.control_point_y_widget.setEnabled(False)
            self.control_point_name_edit.setEnabled(False)

        # Map type and contours
        self.map_type_combo.setEnabled(enabled)
        self.contours_checkbox.setEnabled(enabled)

        # Settings widgets
        self.helmert_widget.setEnabled(enabled)
        self.grid_widget.setEnabled(enabled)
        self.output_widget.setEnabled(enabled)

        # Main action buttons
        self.download_btn.setEnabled(enabled)
        # Save button availability depends on whether image is loaded
        if enabled and self._base_image is not None:
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)
        else:
            self.save_map_btn.setEnabled(False)
            self.save_map_action.setEnabled(False)

        # Menu actions
        if hasattr(self, 'new_profile_action'):
            self.new_profile_action.setEnabled(enabled)
        if hasattr(self, 'open_profile_action'):
            self.open_profile_action.setEnabled(enabled)
        if hasattr(self, 'save_profile_action'):
            self.save_profile_action.setEnabled(enabled)
        if hasattr(self, 'save_profile_as_action'):
            self.save_profile_as_action.setEnabled(enabled)

        if not enabled:
            self.setStyleSheet("""
                QLabel { color: grey; }
                QGroupBox { color: grey; }
                QCheckBox { color: grey; }
                QRadioButton { color: grey; }
                QTabBar::tab { color: grey; }
            """)
        else:
            self.setStyleSheet('')

        # Ensure progress label stays visible and readable (not grey if it was affected)
        if not enabled:
            # We need to make sure the progress label specifically remains readable
            self._progress_label.setStyleSheet('color: black;')
        else:
            self._progress_label.setStyleSheet('')

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
            'SK42mapper v0.2\n\nПриложение для создания карт в системе '
            'Гаусса-Крюгера\n',
        )

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

        # Cleanup download worker with timeout
        if self._download_worker and self._download_worker.isRunning():
            logger.info('Terminating download worker thread')
            log_thread_status('before worker termination')
            self._download_worker.quit()
            if not self._download_worker.wait(5000):  # 5 second timeout
                logger.warning(
                    'Download worker did not terminate gracefully, forcing termination',
                )
                self._download_worker.terminate()
                self._download_worker.wait()
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

"""PySide6-based View components implementing MVC pattern."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
from PySide6.QtCore import QObject, QSignalBlocker, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QPixmapCache
from PySide6.QtWidgets import (
    QApplication,
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
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from constants import (
    MAP_TYPE_LABELS_RU,
    MapType,
)
from diagnostics import (
    log_comprehensive_diagnostics,
    log_memory_usage,
    log_thread_status,
)
from gui.controller import MilMapperController
from gui.model import EventData, MilMapperModel, ModelEvent, Observer
from gui.preview_window import OptimizedImageView
from profiles import ensure_profiles_dir
from progress import (
    cleanup_all_progress_resources,
    set_preview_image_callback,
    set_spinner_callbacks,
)
from status_bar_proxy import StatusBarProxy

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Constant for decimal precision threshold
MIN_DECIMALS_FOR_SMALL_STEP = 2


class DownloadWorker(QThread):
    """Worker thread for map download operations."""

    finished = Signal(bool, str)  # success, error_message
    progress_update = Signal(int, int, str)  # done, total, label
    preview_ready = Signal(object)  # PIL Image object

    def __init__(self, controller: MilMapperController) -> None:
        super().__init__()
        self._controller = controller

    def run(self) -> None:
        """Execute download in background thread."""
        logger.info('DownloadWorker thread started')
        log_thread_status('worker thread start')
        log_memory_usage('worker thread start')

        try:
            # Setup thread-safe callbacks that emit signals instead of direct UI updates
            def preview_callback(img_obj: object) -> bool:
                """Handle preview image from map generation."""
                try:
                    if isinstance(img_obj, Image.Image):
                        self.preview_ready.emit(img_obj)
                        return True
                    return False
                except Exception as e:
                    logger.warning(f'Failed to process preview image: {e}')
                    return False

            # Setup progress system with thread-safe callbacks

            set_spinner_callbacks(
                lambda label: self.progress_update.emit(0, 0, label),
                lambda label: None,
            )
            set_preview_image_callback(preview_callback)

            # Import and set progress callback for ConsoleProgress updates
            from progress import set_progress_callback

            set_progress_callback(
                lambda done, total, label: self.progress_update.emit(done, total, label)
            )

            # Run the actual download
            log_memory_usage('before download sync call')
            self._controller.start_map_download_sync()
            log_memory_usage('after download sync call')

            self.finished.emit(True, '')
            logger.info('DownloadWorker thread completed successfully')

        except Exception as e:
            logger.exception(f'DownloadWorker thread failed: {e}')
            log_memory_usage('worker thread error')
            self.finished.emit(False, str(e))
        finally:
            log_thread_status('worker thread end')
            log_memory_usage('worker thread end')


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
            'Если выключено: рисуются только крестики в точках пересечения без подписей.'
        )
        layout.addWidget(self.display_grid_cb, 0, 0, 1, 2)  # Растянуть на 2 колонки

        # Connect checkbox to enable/disable handler
        self.display_grid_cb.toggled.connect(self._on_display_grid_toggled)

        # Grid width
        self.width_label = QLabel('Толщина линий (px):')
        layout.addWidget(self.width_label, 1, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20)
        self.width_spin.setValue(4)
        self.width_spin.setToolTip('Толщина линий сетки в пикселях')
        layout.addWidget(self.width_spin, 1, 1)

        # Font size
        self.font_label = QLabel('Размер шрифта (px):')
        layout.addWidget(self.font_label, 2, 0)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(10, 200)
        self.font_spin.setValue(86)
        self.font_spin.setToolTip('Размер шрифта подписей координат')
        layout.addWidget(self.font_spin, 2, 1)

        # Text margin
        self.margin_label = QLabel('Отступ текста (px):')
        layout.addWidget(self.margin_label, 3, 0)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100)
        self.margin_spin.setValue(43)
        self.margin_spin.setToolTip('Отступ подписи от края изображения')
        layout.addWidget(self.margin_spin, 3, 1)

        # Label background padding
        self.padding_label = QLabel('Отступ фона (px):')
        layout.addWidget(self.padding_label, 4, 0)
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 50)
        self.padding_spin.setValue(6)
        self.padding_spin.setToolTip('Внутренний отступ подложки вокруг текста')
        layout.addWidget(self.padding_spin, 4, 1)

        self.setLayout(layout)

    def _on_display_grid_toggled(self, checked: bool) -> None:
        """Enable/disable grid parameters based on display_grid checkbox state."""
        self.width_label.setEnabled(checked)
        self.width_spin.setEnabled(checked)
        self.font_label.setEnabled(checked)
        self.font_spin.setEnabled(checked)
        self.margin_label.setEnabled(checked)
        self.margin_spin.setEnabled(checked)
        self.padding_label.setEnabled(checked)
        self.padding_spin.setEnabled(checked)

    def get_settings(self) -> dict[str, int | bool]:
        """Get grid settings as dictionary."""
        return {
            'grid_width_px': self.width_spin.value(),
            'grid_font_size': self.font_spin.value(),
            'grid_text_margin': self.margin_spin.value(),
            'grid_label_bg_padding': self.padding_spin.value(),
            'display_grid': self.display_grid_cb.isChecked(),
        }

    def set_settings(self, settings: dict[str, int | bool]) -> None:
        """Set grid settings from dictionary."""
        # Block signals to prevent feedback loops when setting values programmatically
        with QSignalBlocker(self.width_spin):
            self.width_spin.setValue(settings.get('grid_width_px', 4))
        with QSignalBlocker(self.font_spin):
            self.font_spin.setValue(settings.get('grid_font_size', 86))
        with QSignalBlocker(self.margin_spin):
            self.margin_spin.setValue(settings.get('grid_text_margin', 43))
        with QSignalBlocker(self.padding_spin):
            self.padding_spin.setValue(settings.get('grid_label_bg_padding', 6))
        with QSignalBlocker(self.display_grid_cb):
            self.display_grid_cb.setChecked(bool(settings.get('display_grid', True)))

        # Manually trigger enable/disable logic since signal was blocked
        self._on_display_grid_toggled(self.display_grid_cb.isChecked())


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


class _ViewObserver(Observer):
    """Adapter to bridge model events to the view without name clashes with QWidget.update."""

    def __init__(self, handler: Callable[[EventData], None]) -> None:
        self._handler = handler

    def update(self, event_data: EventData) -> None:  # type: ignore[override]
        self._handler(event_data)


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


class ModalOverlay(QWidget):
    """Semi-transparent overlay widget to shade parent window during modal operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Make widget transparent for mouse events but visible
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Set dark semi-transparent background
        self.setStyleSheet('background-color: rgba(0, 0, 0, 80);')
        # Position at top-left of parent
        self.move(0, 0)
        self.hide()

    def showEvent(self, event) -> None:
        """Resize overlay to cover entire parent on show."""
        super().showEvent(event)
        parent = self.parent()
        if parent and isinstance(parent, QWidget):
            # Cover entire parent widget
            self.resize(parent.size())
            # Ensure overlay is on top of all siblings
            self.raise_()

    def resizeToParent(self) -> None:
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

        # Guard flag to prevent model updates while UI is being populated programmatically
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
        self._progress_bar.setRange(0, 0)  # Indeterminate mode by default
        self._progress_label = QLabel()
        self._progress_label.setVisible(False)
        # Добавляем виджеты прогресса в левую часть статус-бара
        self.status_bar.addWidget(self._progress_label)
        self.status_bar.addWidget(self._progress_bar)

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

        self.control_point_checkbox = QCheckBox('Контрольная точка')
        self.control_point_checkbox.setToolTip(
            'Включить отображение контрольной точки на карте'
        )
        control_point_layout.addWidget(self.control_point_checkbox)

        self.control_point_x_widget = CoordinateInputWidget('X (вертикаль):', 54, 15)
        self.control_point_y_widget = CoordinateInputWidget('Y (горизонталь):', 74, 40)

        control_point_layout.addWidget(self.control_point_x_widget)
        control_point_layout.addWidget(self.control_point_y_widget)
        control_point_group.setLayout(control_point_layout)

        # По умолчанию контролы координат отключены
        self.control_point_x_widget.setEnabled(False)
        self.control_point_y_widget.setEnabled(False)

        coords_layout.addWidget(control_point_group)
        left_container.addWidget(coords_frame)

        # Настройки
        settings_container = QFrame()
        settings_container.setFrameStyle(QFrame.Shape.StyledPanel)
        settings_vertical_layout = QVBoxLayout()

        settings_vertical_layout.addWidget(QLabel('Настройки'))

        # Тип карты и чекбокс изолиний
        maptype_row = QHBoxLayout()
        maptype_label = QLabel('Тип карты:')
        self.map_type_combo = QComboBox()
        # Заполняем пункты (исключаем «Карта высот (контуры)», теперь это опция-оверлей)
        self._maptype_order = [
            MapType.SATELLITE,
            MapType.HYBRID,
            MapType.STREETS,
            MapType.OUTDOORS,
            MapType.ELEVATION_COLOR,
            # MapType.ELEVATION_HILLSHADE,
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
        settings_vertical_layout.addLayout(maptype_row)

        self.helmert_group = QGroupBox('Датум-трансформация СК-42 → WGS84 (Helmert)')
        helmert_group_layout = QVBoxLayout()
        self.helmert_widget = HelmertSettingsWidget()
        helmert_group_layout.addWidget(self.helmert_widget)
        self.helmert_group.setLayout(helmert_group_layout)
        settings_vertical_layout.addWidget(self.helmert_group)

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
        settings_vertical_layout.addLayout(settings_horizontal_layout)
        settings_container.setLayout(settings_vertical_layout)
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

        self._set_sliders_enabled(False)

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
        splitter.setStretchFactor(0, 0)  # левая колонка — фиксированнее
        splitter.setStretchFactor(1, 1)  # правая колонка (превью) растягивается
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
        # Use editingFinished instead of textChanged to avoid cyclic updates during typing
        self.control_point_x_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )
        self.control_point_y_widget.coordinate_edit.editingFinished.connect(
            self._on_settings_changed
        )

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
        self.profile_combo.blockSignals(True)
        try:
            self.profile_combo.clear()
            self.profile_combo.addItems(profiles)
            if 'default' in profiles:
                self.profile_combo.setCurrentText('default')
            elif profiles:
                self.profile_combo.setCurrentIndex(0)
        finally:
            self.profile_combo.blockSignals(False)

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
            # Clear any existing preview to avoid showing outdated imagery when coordinates or contours change
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

    @Slot()
    def _on_map_type_changed(self) -> None:
        """Clear preview immediately when map type changes and propagate setting."""
        # Clear any existing preview to avoid showing outdated imagery for another map type
        self._clear_preview_ui()
        # Delegate to the common settings handler to store the new map type in the model
        self._on_settings_changed()

    def _sync_ui_to_model_now(self) -> None:
        """
        Force-collect current UI settings and push them to the model without guards.

        Does not clear preview or check _ui_sync_in_progress to avoid losing changes during Save/Save As.
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
        self._controller.update_settings_bulk(**payload)

    def _get_current_coordinates(self) -> dict[str, int]:
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
        }

    @Slot(int)
    def _load_selected_profile(self, index: int) -> None:
        """Load the selected profile when selection changes (guarded, non-reentrant)."""
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
                        # Explicitly load the newly saved profile to avoid reverting to default
                        try:
                            logger.info(
                                'After Save As 								-> selecting and loading profile: %s',
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
        if self._download_worker and self._download_worker.isRunning():
            QMessageBox.information(self, 'Информация', 'Загрузка уже выполняется')
            return

        # Clear previous preview and pixmap cache between runs
        self._clear_preview_ui()

        # Cleanup any stale worker from previous run
        self._cleanup_download_worker()

        self._download_worker = DownloadWorker(self._controller)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.progress_update.connect(self._update_progress)
        self._download_worker.preview_ready.connect(self._show_preview_in_main_window)
        self._download_worker.start()

        # Update UI state
        # Show progress bar in status bar (will be updated by progress callbacks)
        self._progress_bar.setVisible(True)
        self._progress_label.setVisible(True)
        self._progress_label.setText('Подготовка…')
        self._progress_bar.setRange(0, 0)  # Indeterminate mode initially
        # Disable all UI controls during download
        self._set_controls_enabled(False)

    @Slot(bool, str)
    def _on_download_finished(self, success: bool, error_msg: str) -> None:
        """Handle download completion."""
        # Hide progress widgets and re-enable controls
        try:
            if self._progress_bar.isVisible():
                self._progress_bar.setVisible(False)
                self._progress_label.setVisible(False)
            # Re-enable all UI controls when download is finished
            self._set_controls_enabled(True)
        except Exception as e:
            logger.debug(f'Failed to hide progress widgets: {e}')

        if success:
            self._status_proxy.show_message('Карта успешно создана', 5000)
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
            from progress import set_progress_callback as _set_prog

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
        elif event == ModelEvent.ERROR_OCCURRED:
            error_msg = data.get('error', 'Неизвестная ошибка')
            # Только статус-бар; модальные диалоги показываются централизованно в _on_download_finished
            self._status_proxy.show_message(f'Ошибка: {error_msg}', 5000)
        elif event == ModelEvent.WARNING_OCCURRED:
            warn_msg = (
                data.get('warning')
                or data.get('message')
                or data.get('error')
                or 'Предупреждение'
            )
            # Только статус-бар; без модальных диалогов, чтобы избежать дублей и вызовов не из GUI-потока
            self._status_proxy.show_message(f'Предупреждение: {warn_msg}', 5000)

    def _update_ui_from_settings(self, settings: Any) -> None:
        """Update UI controls from settings object."""
        if not settings:
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
            'grid_width_px': settings.grid_width_px,
            'grid_font_size': settings.grid_font_size,
            'grid_text_margin': settings.grid_text_margin,
            'grid_label_bg_padding': settings.grid_label_bg_padding,
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

        # Legacy handling: if profile had ELEVATION_CONTOURS, map to OUTDOORS + overlay enabled
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

        # Update control point settings
        control_point_enabled = getattr(settings, 'control_point_enabled', False)
        with QSignalBlocker(self.control_point_checkbox):
            self.control_point_checkbox.setChecked(control_point_enabled)

        # Programmatically set full control point coordinates without splitting to high/low
        control_point_x = int(getattr(settings, 'control_point_x', 5415000))
        control_point_y = int(getattr(settings, 'control_point_y', 7440000))

        with QSignalBlocker(self.control_point_x_widget.coordinate_edit):
            self.control_point_x_widget.set_coordinate(control_point_x)
        with QSignalBlocker(self.control_point_y_widget.coordinate_edit):
            self.control_point_y_widget.set_coordinate(control_point_y)

        # Log the values to ensure no truncation occurs during UI update
        try:
            x_text = self.control_point_x_widget.coordinate_edit.text()
            y_text = self.control_point_y_widget.coordinate_edit.text()
            logger.info(
                "UI control point set: enabled=%s, x_src=%d -> edit='%s', y_src=%d -> edit='%s'",
                control_point_enabled,
                control_point_x,
                x_text,
                control_point_y,
                y_text,
            )
        except Exception:
            logger.exception('Failed to log control point UI update')

        # Enable/disable coordinate inputs based on checkbox state
        self.control_point_x_widget.setEnabled(control_point_enabled)
        self.control_point_y_widget.setEnabled(control_point_enabled)

        # Unblock settings propagation after UI is fully synced
        self._ui_sync_in_progress = False

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
        # Show progress widgets if not visible
        if not self._progress_bar.isVisible():
            self._progress_bar.setVisible(True)
            self._progress_label.setVisible(True)
            # Disable all UI controls when showing progress
            self._set_controls_enabled(False)

        # Always use indeterminate progress (spinner mode) - no specific progress indication
        self._progress_bar.setRange(0, 0)

        # Update progress label text
        if label:
            self._progress_label.setText(label)

    def _show_preview_in_main_window(self, image: Any) -> None:
        """Show preview image in the main window's integrated preview area."""
        try:
            if not isinstance(image, Image.Image):
                logger.warning('Invalid image object for preview')
                return

            # Set base image (full size)
            self._base_image = image.convert('RGB') if image.mode != 'RGB' else image

            # Display image
            self._current_image = self._base_image
            self._preview_area.set_image(self._current_image)

            # Enable save button and menu action
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)

            # Hide progress bar and label when preview is ready
            try:
                if self._progress_bar.isVisible():
                    self._progress_bar.setVisible(False)
                    self._progress_label.setVisible(False)
                    # Re-enable all UI controls when hiding progress
                    self._set_controls_enabled(True)
                    # Explicitly enable quality slider
                    self._set_sliders_enabled(True)
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
            # Reset images
            self._current_image = None
            self._base_image = None
            # Disable save controls
            self.save_map_btn.setEnabled(False)
            self.save_map_action.setEnabled(False)
            # Disable quality slider when no image is loaded
            self._set_sliders_enabled(False)
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
                    image,
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
                        self.finished.emit(True, '')
                    except Exception as e:
                        self.finished.emit(False, str(e))

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

            def _on_save_complete(success: bool, err: str) -> None:
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
                _on_save_complete,
                Qt.ConnectionType.QueuedConnection,
            )
            # Ensure the worker thread quits immediately after finishing to release resources
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

        except Exception as e:
            logger.exception(f'Error cleaning up save resources: {e}')

    def _set_sliders_enabled(self, enabled: bool) -> None:
        """Enable/disable quality slider."""
        # Only quality slider remains, enable it based on parameter
        self.quality_slider.setEnabled(enabled)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable all UI controls when progress bar is shown/hidden."""
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
        self.control_point_checkbox.setEnabled(enabled)
        # Control point coordinate widgets should respect checkbox state
        if enabled:
            cp_enabled = self.control_point_checkbox.isChecked()
            self.control_point_x_widget.setEnabled(cp_enabled)
            self.control_point_y_widget.setEnabled(cp_enabled)
        else:
            self.control_point_x_widget.setEnabled(False)
            self.control_point_y_widget.setEnabled(False)

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

    @Slot()
    def _new_profile(self) -> None:
        """Create new profile (placeholder)."""
        QMessageBox.information(self, 'Информация', 'Функция создания нового профиля')

    @Slot()
    def _open_profile(self) -> None:
        """Open profile file from disk and load settings."""
        # Default directory: profiles dir (user or local)
        from profiles import ensure_profiles_dir

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
            'SK42mapper v0.2\n\nПриложение для создания карт в системе Гаусса-Крюгера\n',
        )

    def resizeEvent(self, event) -> None:
        """Handle window resize to update overlay size."""
        super().resizeEvent(event)
        # Update overlay when central widget is resized
        if (
            hasattr(self, '_modal_overlay')
            and self._modal_overlay is not None
            and self._modal_overlay.isVisible()
        ):
            self._modal_overlay.resizeToParent()

    def closeEvent(self, event) -> None:
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

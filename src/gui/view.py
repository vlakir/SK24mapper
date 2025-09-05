"""PySide6-based View components implementing MVC pattern."""

from __future__ import annotations

import contextlib
import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QPixmapCache
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from constants import (
    BYTES_CONVERSION_FACTOR,
    BYTES_TO_KB_THRESHOLD,
    FLOAT_COMPARISON_TOLERANCE,
    PROFILES_DIR,
)
from diagnostics import (
    log_comprehensive_diagnostics,
    log_memory_usage,
    log_thread_status,
)
from gui.controller import MilMapperController
from gui.model import EventData, MilMapperModel, ModelEvent, Observer
from gui.preview_window import OptimizedImageView
from progress import (
    cleanup_all_progress_resources,
    set_preview_image_callback,
    set_spinner_callbacks,
)
from status_bar_proxy import StatusBarProxy

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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
                lambda label: self.progress_update.emit(0, 0, label), lambda label: None
            )
            set_preview_image_callback(preview_callback)

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


class CoordinateInputWidget(QWidget):
    """Widget for coordinate input with high/low fields."""

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


class GridSettingsWidget(QWidget):
    """Widget for grid configuration settings."""

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup grid settings UI."""
        layout = QGridLayout()

        # Grid width
        layout.addWidget(QLabel('Толщина линий (px):'), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20)
        self.width_spin.setValue(4)
        self.width_spin.setToolTip('Толщина линий сетки в пикселях')
        layout.addWidget(self.width_spin, 0, 1)

        # Font size
        layout.addWidget(QLabel('Размер шрифта (px):'), 1, 0)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(10, 200)
        self.font_spin.setValue(86)
        self.font_spin.setToolTip('Размер шрифта подписей координат')
        layout.addWidget(self.font_spin, 1, 1)

        # Text margin
        layout.addWidget(QLabel('Отступ текста (px):'), 2, 0)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100)
        self.margin_spin.setValue(43)
        self.margin_spin.setToolTip('Отступ подписи от края изображения')
        layout.addWidget(self.margin_spin, 2, 1)

        # Label background padding
        layout.addWidget(QLabel('Отступ фона (px):'), 3, 0)
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 50)
        self.padding_spin.setValue(6)
        self.padding_spin.setToolTip('Внутренний отступ подложки вокруг текста')
        layout.addWidget(self.padding_spin, 3, 1)

        self.setLayout(layout)

    def get_settings(self) -> dict[str, int]:
        """Get grid settings as dictionary."""
        return {
            'grid_width_px': self.width_spin.value(),
            'grid_font_size': self.font_spin.value(),
            'grid_text_margin': self.margin_spin.value(),
            'grid_label_bg_padding': self.padding_spin.value(),
        }

    def set_settings(self, settings: dict[str, int]) -> None:
        """Set grid settings from dictionary."""
        self.width_spin.setValue(settings.get('grid_width_px', 4))
        self.font_spin.setValue(settings.get('grid_font_size', 86))
        self.margin_spin.setValue(settings.get('grid_text_margin', 43))
        self.padding_spin.setValue(settings.get('grid_label_bg_padding', 6))


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
            lambda v: self.quality_label.setText(f'{v}')
        )
        layout.addWidget(self.quality_label, 0, 2)

        # Оценка размера
        self.size_estimate_title = QLabel('Оценка размера:')
        self.size_estimate_value = QLabel('—')
        self.size_estimate_value.setToolTip(
            'Приблизительный размер итогового JPEG при текущем качестве'
        )
        layout.addWidget(self.size_estimate_title, 1, 0)
        layout.addWidget(self.size_estimate_value, 1, 1)

        # Яркость
        self.brightness_label = QLabel('Яркость: 100%')
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        layout.addWidget(self.brightness_label, 2, 0)
        layout.addWidget(self.brightness_slider, 2, 1)

        # Контраст
        self.contrast_label = QLabel('Контрастность: 100%')
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        layout.addWidget(self.contrast_label, 3, 0)
        layout.addWidget(self.contrast_slider, 3, 1)

        # Насыщенность
        self.saturation_label = QLabel('Насыщенность: 100%')
        self.saturation_slider = QSlider(Qt.Orientation.Horizontal)
        self.saturation_slider.setRange(0, 200)
        self.saturation_slider.setValue(100)
        layout.addWidget(self.saturation_label, 4, 0)
        layout.addWidget(self.saturation_slider, 4, 1)

        self.setLayout(layout)

    def get_settings(self) -> dict[str, Any]:
        """Get output settings as dictionary."""
        return {
            'jpeg_quality': self.quality_slider.value(),
            'brightness': self.brightness_slider.value() / 100.0,
            'contrast': self.contrast_slider.value() / 100.0,
            'saturation': self.saturation_slider.value() / 100.0,
        }

    def set_settings(self, settings: dict[str, Any]) -> None:
        """Set output settings from dictionary."""
        q = int(settings.get('jpeg_quality', 95))
        q = max(q, 10)
        q = min(q, 100)
        self.quality_slider.setValue(q)
        self.quality_label.setText(f'{q}')

        # Adjustments
        b = float(settings.get('brightness', 1.0))
        c = float(settings.get('contrast', 1.0))
        s = float(settings.get('saturation', 1.0))
        # Clamp within 0..2
        b = 0.0 if b < 0.0 else (min(b, 2.0))
        c = 0.0 if c < 0.0 else (min(c, 2.0))
        s = 0.0 if s < 0.0 else (min(s, 2.0))
        self.brightness_slider.setValue(round(b * 100))
        self.contrast_slider.setValue(round(c * 100))
        self.saturation_slider.setValue(round(s * 100))


class _ViewObserver(Observer):
    """Adapter to bridge model events to the view without name clashes with QWidget.update."""

    def __init__(self, handler: Callable[[EventData], None]) -> None:
        self._handler = handler

    def update(self, event_data: EventData) -> None:  # type: ignore[override]
        self._handler(event_data)


class MainWindow(QMainWindow):
    _sig_schedule_adjust = Signal()
    _sig_run_adjust_now = Signal()

    # Slots to ensure results/cleanup run on GUI thread
    @Slot(int, object, str)
    def _on_adjust_done_slot(self, gen: int, img_obj: object, err: str) -> None:
        # Always on GUI thread
        self.save_map_btn.setEnabled(True)
        self.save_map_action.setEnabled(True)
        if gen != self._adj_generation:
            return
        if err:
            logger.error(f'Adjustment error: {err}')
            return
        if isinstance(img_obj, Image.Image):
            try:
                self._preview_area.set_image(img_obj)
            except Exception as ex:
                logger.exception(f'[ADJ] set_image failed: {ex}')

    @Slot()
    def _on_adjust_thread_finished_slot(self) -> None:
        # Identify sender thread and cleanup mappings if present
        sender = self.sender()
        th = sender if isinstance(sender, QThread) else None
        if th is None:
            return
        # Remove and delete associated worker if tracked via map
        worker = (
            getattr(self, '_adjust_map', {}).pop(th, None)
            if hasattr(self, '_adjust_map')
            else None
        )
        try:
            if worker is not None and worker in self._adjust_workers:
                self._adjust_workers.remove(worker)
                worker.deleteLater()
        except Exception as e:
            logger.debug(f'Error cleaning up adjust worker: {e}')
        try:
            if th in self._adjust_threads:
                self._adjust_threads.remove(th)
            th.deleteLater()
        except Exception as e:
            logger.debug(f'Error cleaning up adjust thread: {e}')

    """Main application window implementing Observer pattern."""

    def __init__(self, model: MilMapperModel, controller: MilMapperController) -> None:
        super().__init__()
        self._model = model
        self._controller = controller
        self._download_worker: DownloadWorker | None = None
        self._busy_dialog: QProgressDialog | None = None
        self._current_image: Any = None  # Store current image for saving
        self._save_thread: Any = None
        self._save_worker: Any = None

        # Register as observer via adapter to avoid QWidget.update signature clash
        self._observer_adapter = _ViewObserver(self._handle_model_event)
        self._model.add_observer(self._observer_adapter)

        # Adjustment state
        self._base_image: Image.Image | None = None
        self._adj = {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0}
        self._adj_generation: int = 0
        # Keep strong references to all active adjustment threads/workers until they finish
        self._adjust_threads: list[QThread] = []
        self._adjust_workers: list[QObject] = []
        self._adjust_map: dict[QThread, QObject] = {}
        self._preview_update_timer = QTimer(self)
        self._preview_update_timer.setSingleShot(True)
        self._preview_update_timer.setInterval(130)

        # Size estimate debounce timer
        self._estimate_timer = QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.setInterval(400)

        # Storage for estimate workers
        self._estimate_threads: list[QThread] = []
        self._estimate_workers: list[QObject] = []

        self._setup_ui()
        self._setup_connections()

        # Connect internal signals to GUI-only slots (Queued)
        self._sig_schedule_adjust.connect(
            self._schedule_adjust_gui, Qt.ConnectionType.QueuedConnection
        )
        self._sig_run_adjust_now.connect(
            self._run_adjust_now_gui, Qt.ConnectionType.QueuedConnection
        )

        self._load_initial_data()
        logger.info('MainWindow initialized')

    def _setup_ui(self) -> None:
        """Setup the main window UI."""
        self.setWindowTitle('SK42mapper')
        self.setFixedSize(800, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Create menu
        self._create_menu()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._status_proxy = StatusBarProxy(self.status_bar)
        self._status_proxy.show_message('Готов к работе')

        # Profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel('Профиль:'))

        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip('Выберите профиль настроек')
        profile_layout.addWidget(self.profile_combo)

        self.load_profile_btn = QPushButton('Загрузить')
        self.load_profile_btn.setToolTip('Загрузить выбранный профиль')
        profile_layout.addWidget(self.load_profile_btn)

        self.save_profile_btn = QPushButton('Сохранить')
        self.save_profile_btn.setToolTip('Сохранить текущие настройки в профиль')
        profile_layout.addWidget(self.save_profile_btn)

        self.save_profile_as_btn = QPushButton('Сохранить как...')
        self.save_profile_as_btn.setToolTip(
            'Сохранить текущие настройки в новый профиль'
        )
        profile_layout.addWidget(self.save_profile_as_btn)

        profile_layout.addStretch()
        main_layout.addLayout(profile_layout)

        # Coordinates section
        coords_frame = QFrame()
        coords_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        coords_layout = QVBoxLayout()
        coords_frame.setLayout(coords_layout)

        coords_layout.addWidget(QLabel('Координаты области (СК-42):'))

        self.from_x_widget = CoordinateInputWidget('X (вертикаль):', 54, 14)
        self.from_y_widget = CoordinateInputWidget('Y (горизонталь):', 74, 43)
        self.to_x_widget = CoordinateInputWidget('X (вертикаль):', 54, 23)
        self.to_y_widget = CoordinateInputWidget('Y (горизонталь):', 74, 49)

        # Create panels for coordinate groups
        panels_layout = QHBoxLayout()

        # "From" coordinates panel
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

        # "To" coordinates panel
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

        main_layout.addWidget(coords_frame)

        # Settings sections arranged vertically with header above panels
        settings_container = QFrame()
        settings_container.setFrameStyle(QFrame.Shape.StyledPanel)
        settings_vertical_layout = QVBoxLayout()

        # Add settings header above the panels
        settings_vertical_layout.addWidget(QLabel('Настройки'))

        # Create horizontal layout for the panels
        settings_horizontal_layout = QHBoxLayout()

        # Grid settings section
        grid_frame = QFrame()
        grid_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        grid_layout = QVBoxLayout()
        grid_frame.setLayout(grid_layout)

        # grid_layout.addWidget(QLabel("Сетка:"))
        self.grid_widget = GridSettingsWidget()
        grid_layout.addWidget(self.grid_widget)

        settings_horizontal_layout.addWidget(grid_frame)

        # Output settings section
        output_frame = QFrame()
        output_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        output_layout = QVBoxLayout()
        output_frame.setLayout(output_layout)

        # output_layout.addWidget(QLabel("Файл:"))
        self.output_widget = OutputSettingsWidget()
        output_layout.addWidget(self.output_widget)
        # Alias sliders/labels from output settings to main window for event wiring
        self.quality_slider = self.output_widget.quality_slider
        self.brightness_slider = self.output_widget.brightness_slider
        self.contrast_slider = self.output_widget.contrast_slider
        self.saturation_slider = self.output_widget.saturation_slider
        self.brightness_label = self.output_widget.brightness_label
        self.contrast_label = self.output_widget.contrast_label
        self.saturation_label = self.output_widget.saturation_label

        settings_horizontal_layout.addWidget(output_frame)

        # Add the horizontal panels layout to the vertical layout
        settings_vertical_layout.addLayout(settings_horizontal_layout)

        settings_container.setLayout(settings_vertical_layout)

        main_layout.addWidget(settings_container)

        # Create map button - positioned below settings panels
        self.download_btn = QPushButton('Создать карту')
        self.download_btn.setToolTip('Начать создание карты')
        self.download_btn.setStyleSheet('QPushButton { font-weight: bold; }')
        main_layout.addWidget(self.download_btn)

        # Preview area section
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        preview_layout = QVBoxLayout()
        preview_frame.setLayout(preview_layout)

        preview_layout.addWidget(QLabel('Предпросмотр карты:'))

        self._preview_area = OptimizedImageView()
        self._preview_area.setMinimumHeight(220)
        preview_layout.addWidget(self._preview_area)

        self._set_sliders_enabled(False)

        # Add spacing before the save button
        preview_layout.addSpacing(10)

        # Save map button
        self.save_map_btn = QPushButton('Сохранить карту')
        self.save_map_btn.setStyleSheet('QPushButton { font-weight: bold; }')

        self.save_map_btn.setToolTip('Сохранить карту в файл')
        self.save_map_btn.setEnabled(False)  # Disabled until image is loaded

        main_layout.addWidget(preview_frame)
        main_layout.addWidget(self.save_map_btn)

        # BusyDialog (QProgressDialog) will be created lazily on first use to avoid
        # any chance of it appearing during application startup.
        self._busy_dialog = None
        # Ensure sliders disabled at startup
        self._set_sliders_enabled(False)

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
        self.load_profile_btn.clicked.connect(self._load_selected_profile)
        self.save_profile_btn.clicked.connect(self._save_current_profile)
        self.save_profile_as_btn.clicked.connect(self._save_profile_as)

        # Download
        self.download_btn.clicked.connect(self._start_download)

        # Save map
        self.save_map_btn.clicked.connect(self._save_map)

        # Adjustment sliders
        self.brightness_slider.valueChanged.connect(self._on_adj_slider_changed)
        self.contrast_slider.valueChanged.connect(self._on_adj_slider_changed)
        self.saturation_slider.valueChanged.connect(self._on_adj_slider_changed)
        # Immediate update on release
        self.brightness_slider.sliderReleased.connect(
            self._start_fullres_adjustment_immediate
        )
        self.contrast_slider.sliderReleased.connect(
            self._start_fullres_adjustment_immediate
        )
        self.saturation_slider.sliderReleased.connect(
            self._start_fullres_adjustment_immediate
        )
        # Debounce timer
        self._preview_update_timer.timeout.connect(self._start_fullres_adjustment)

        # Size estimate timer
        self._estimate_timer.timeout.connect(self._start_size_estimate)

        # Settings change tracking
        self._connect_setting_changes()

    def _connect_setting_changes(self) -> None:
        """Connect all setting change signals."""
        # Coordinates
        for widget in [
            self.from_x_widget,
            self.from_y_widget,
            self.to_x_widget,
            self.to_y_widget,
        ]:
            widget.high_spin.valueChanged.connect(self._on_settings_changed)
            widget.low_spin.valueChanged.connect(self._on_settings_changed)

        # Grid settings
        self.grid_widget.width_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.font_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.margin_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.padding_spin.valueChanged.connect(self._on_settings_changed)

        # Output settings
        self.output_widget.quality_slider.valueChanged.connect(
            self._on_settings_changed
        )
        # Also schedule size estimate on quality changes
        self.output_widget.quality_slider.valueChanged.connect(
            self._schedule_size_estimate
        )

    def _load_initial_data(self) -> None:
        """Load initial application data."""
        # Load available profiles
        profiles = self._controller.get_available_profiles()
        self.profile_combo.addItems(profiles)

        # Load default profile
        if 'default' in profiles:
            self.profile_combo.setCurrentText('default')
            self._controller.load_profile_by_name('default')

    @Slot()
    def _on_settings_changed(self) -> None:
        """Handle settings change from UI."""
        # Collect all current settings
        coords = self._get_current_coordinates()
        grid_settings = self.grid_widget.get_settings()
        output_settings = self.output_widget.get_settings()

        # Update model through controller
        self._controller.update_coordinates(coords)
        self._controller.update_grid_settings(grid_settings)
        self._controller.update_output_settings(output_settings)

    def _get_current_coordinates(self) -> dict[str, int]:
        """Get current coordinate values from UI."""
        from_x_high, from_x_low = self.from_x_widget.get_values()
        from_y_high, from_y_low = self.from_y_widget.get_values()
        to_x_high, to_x_low = self.to_x_widget.get_values()
        to_y_high, to_y_low = self.to_y_widget.get_values()

        return {
            'from_x_high': from_x_high,
            'from_x_low': from_x_low,
            'from_y_high': from_y_high,
            'from_y_low': from_y_low,
            'to_x_high': to_x_high,
            'to_x_low': to_x_low,
            'to_y_high': to_y_high,
            'to_y_low': to_y_low,
        }

    @Slot()
    def _load_selected_profile(self) -> None:
        """Load the selected profile."""
        profile_name = self.profile_combo.currentText()
        if profile_name:
            self._controller.load_profile_by_name(profile_name)

    @Slot()
    def _save_current_profile(self) -> None:
        """Save current settings to profile."""
        profile_name = self.profile_combo.currentText()
        if profile_name:
            self._controller.save_current_profile(profile_name)

    @Slot()
    def _save_profile_as(self) -> None:
        """Save current settings to a new profile file."""
        # Get default directory
        project_root = Path(__file__).parent.parent.parent
        default_dir = project_root / PROFILES_DIR

        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Сохранить профиль как...',
            str(default_dir / 'новый_профиль'),
            'Файлы профилей (*.toml);;Все файлы (*)',
        )

        if file_path:
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
                if Path(file_path).parent == default_dir:
                    # Refresh profile list
                    self._load_initial_data()
                    # Select the newly saved profile
                    index = self.profile_combo.findText(saved_profile_name)
                    if index >= 0:
                        self.profile_combo.setCurrentIndex(index)

                self._status_proxy.show_message(
                    f'Профиль сохранён как: {saved_profile_name}', 3000
                )

    @Slot()
    def _start_download(self) -> None:
        """Start map download process."""
        if self._download_worker and self._download_worker.isRunning():
            QMessageBox.information(self, 'Информация', 'Загрузка уже выполняется')
            return

        self._download_worker = DownloadWorker(self._controller)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.progress_update.connect(self._update_progress)
        self._download_worker.preview_ready.connect(self._show_preview_in_main_window)
        self._download_worker.start()

        # Update UI state
        self.download_btn.setEnabled(False)
        self._status_proxy.show_message('Загрузка карты...')
        # Optionally show busy dialog immediately; label will update on spinner callback
        dlg = self._ensure_busy_dialog()
        dlg.setLabelText('Подготовка…')
        dlg.setRange(0, 0)
        dlg.show()

    @Slot(bool, str)
    def _on_download_finished(self, success: bool, error_msg: str) -> None:
        """Handle download completion."""
        self.download_btn.setEnabled(True)

        # Close busy dialog (spinner) if visible
        if self._busy_dialog is not None and self._busy_dialog.isVisible():
            self._busy_dialog.reset()
            self._busy_dialog.hide()

        if success:
            self._status_proxy.show_message('Карта успешно создана', 5000)
        else:
            self._status_proxy.show_message('Ошибка при создании карты', 5000)
            QMessageBox.critical(
                self, 'Ошибка', f'Не удалось создать карту:\n{error_msg}'
            )

    def _handle_model_event(self, event_data: EventData) -> None:
        """Handle model events (Observer pattern)."""
        event = event_data.event
        data = event_data.data

        if event == ModelEvent.SETTINGS_CHANGED:
            self._update_ui_from_settings(data.get('settings'))
        elif event == ModelEvent.PROFILE_LOADED:
            # After loading a new profile, clear the preview and reset related UI
            try:
                with contextlib.suppress(Exception):
                    self._preview_area.clear()
                self._current_image = None
                self._base_image = None
                # Reset adjustments to defaults and sync labels (sliders stay disabled)
                self._adj = {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0}
                self._sync_adj_ui_from_state()
                # Disable save controls as no image is present
                self.save_map_btn.setEnabled(False)
                self.save_map_action.setEnabled(False)
                # Disable adjustment sliders when no image
                self._set_sliders_enabled(False)
                # Reset size estimate
                with contextlib.suppress(Exception):
                    self.output_widget.size_estimate_value.setText('—')
            except Exception as e:
                logger.debug(f'Failed to clear preview on profile load: {e}')
            # Update the rest of the UI from profile settings
            self._update_ui_from_settings(data.get('settings'))
            self._status_proxy.show_message(
                f'Профиль загружен: {data.get("profile_name")}', 3000
            )
        elif event == ModelEvent.DOWNLOAD_PROGRESS:
            self._update_progress(
                data.get('done', 0), data.get('total', 0), data.get('label', '')
            )
        elif event == ModelEvent.PREVIEW_UPDATED:
            self._show_preview_in_main_window(data.get('image'))
        elif event == ModelEvent.ERROR_OCCURRED:
            error_msg = data.get('error', 'Неизвестная ошибка')
            self._status_proxy.show_message(f'Ошибка: {error_msg}', 5000)
            QMessageBox.warning(self, 'Предупреждение', error_msg)

    def _update_ui_from_settings(self, settings: Any) -> None:
        """Update UI controls from settings object."""
        if not settings:
            return

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
        }
        self.grid_widget.set_settings(grid_settings)

        # Update output settings
        output_settings = {
            'output_path': settings.output_path,
            'jpeg_quality': getattr(settings, 'jpeg_quality', 95),
            'brightness': getattr(settings, 'brightness', 1.0),
            'contrast': getattr(settings, 'contrast', 1.0),
            'saturation': getattr(settings, 'saturation', 1.0),
        }
        self.output_widget.set_settings(output_settings)

    def _ensure_busy_dialog(self) -> QProgressDialog:
        """Create BusyDialog lazily to prevent it from showing at startup."""
        if self._busy_dialog is None:
            # Use keyword-only parameters to match PySide6 overload and satisfy mypy
            self._busy_dialog = QProgressDialog(self)
            self._busy_dialog.setLabelText('Подготовка…')
            self._busy_dialog.setRange(0, 0)
            self._busy_dialog.setWindowTitle('Обработка')
            self._busy_dialog.setCancelButton(None)
            self._busy_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
            self._busy_dialog.setMinimumDuration(0)
            self._busy_dialog.setAutoClose(False)
            self._busy_dialog.setAutoReset(False)
            self._busy_dialog.hide()
        return self._busy_dialog

    def _update_progress(self, done: int, total: int, label: str) -> None:
        """
        Update spinner BusyDialog text during long operations.

        The old inline progress bar is removed; we always use the spinner.
        """
        dlg = self._ensure_busy_dialog()
        if not dlg.isVisible():
            dlg.setRange(0, 0)
            dlg.show()
        if label:
            dlg.setLabelText(label)
        # Optionally mirror the current stage in the status bar for context
        if label:
            self._status_proxy.show_message(label)

    def _show_preview_in_main_window(self, image: Any) -> None:
        """Show preview image in the main window's integrated preview area."""
        try:
            if not isinstance(image, Image.Image):
                logger.warning('Invalid image object for preview')
                return

            # Set base image (full size) and keep current adjustment sliders unchanged
            self._base_image = image.convert('RGB') if image.mode != 'RGB' else image
            # Initialize adjustments from current model settings (preferred) or from sliders
            try:
                b = float(getattr(self._model.settings, 'brightness', 1.0))
                c = float(getattr(self._model.settings, 'contrast', 1.0))
                s = float(getattr(self._model.settings, 'saturation', 1.0))
            except Exception:
                b = self.brightness_slider.value() / 100.0
                c = self.contrast_slider.value() / 100.0
                s = self.saturation_slider.value() / 100.0
            self._adj = {'brightness': b, 'contrast': c, 'saturation': s}
            # Update only labels according to current sliders; do not move sliders here
            self._update_adj_labels()

            # Display original first
            self._current_image = self._base_image
            self._preview_area.set_image(self._current_image)
            # Show size estimate immediately after preview loading
            self._start_size_estimate()

            # Enable save button and menu action
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)
            # Enable adjustment sliders now that image is available
            self._set_sliders_enabled(True)

            # If a modal busy dialog is still visible, hide it to allow user interaction
            try:
                dlg = getattr(self, '_busy_dialog', None)
                if isinstance(dlg, QProgressDialog) and dlg.isVisible():
                    dlg.reset()
                    dlg.hide()
            except Exception as _e:
                logger.debug(f'Failed to hide busy dialog on preview: {_e}')

            # Apply profile adjustments immediately to the preview
            try:
                self._start_fullres_adjustment_immediate()
            except Exception as e:
                logger.debug(f'Failed to start immediate adjustment: {e}')

            logger.info('Preview displayed in main window')

        except Exception as e:
            error_msg = f'Ошибка при отображении предпросмотра: {e}'
            logger.exception(error_msg)
            QMessageBox.warning(self, 'Ошибка предпросмотра', error_msg)

    @Slot()
    def _save_map(self) -> None:
        """Save the current map image to file."""
        if self._current_image is None:
            QMessageBox.warning(
                self, 'Предупреждение', 'Нет изображения для сохранения'
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
                    self, image, path: Path, quality: int, adj: dict[str, float]
                ):
                    super().__init__()
                    self.image = image
                    self.path = path
                    self.quality = quality
                    self.adj = adj

                def run(self) -> None:
                    """Save the image in background thread."""
                    try:
                        img = self.image
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Apply same LUT adjustments
                        b = float(self.adj.get('brightness', 1.0))
                        c = float(self.adj.get('contrast', 1.0))
                        s = float(self.adj.get('saturation', 1.0))
                        if not (
                            abs(b - 1.0) < FLOAT_COMPARISON_TOLERANCE
                            and abs(c - 1.0) < FLOAT_COMPARISON_TOLERANCE
                            and abs(s - 1.0) < FLOAT_COMPARISON_TOLERANCE
                        ):

                            def clamp(v: int) -> int:
                                return 0 if v < 0 else (min(v, 255))

                            lut_y = [0] * 256
                            lut_c = [0] * 256
                            for i in range(256):
                                y = round(((i - 128) * c + 128) * b)
                                lut_y[i] = clamp(y)
                                cc = round(128 + (i - 128) * s)
                                lut_c[i] = clamp(cc)
                            ycbcr = img.convert('YCbCr')
                            y, cb, cr = ycbcr.split()
                            y = y.point(lut_y)
                            cb = cb.point(lut_c)
                            cr = cr.point(lut_c)
                            img = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')

                        img.save(
                            str(self.path),
                            'JPEG',
                            quality=self.quality,
                            optimize=True,
                            subsampling='4:4:4',
                        )
                        self.finished.emit(True, '')
                    except Exception as e:
                        self.finished.emit(False, str(e))

            # Get current quality setting
            quality = int(getattr(self._model.settings, 'jpeg_quality', 95))

            # Create and setup worker thread
            th = QThread()
            # Use base image to ensure full resolution; apply current adjustments in worker
            base_for_save = self._base_image or self._current_image
            worker = _SaveWorker(base_for_save, out_path, quality, dict(self._adj))
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
                        f'Карта сохранена: {out_path.name}', 5000
                    )
                    # Clear preview area after successful save
                    self._preview_area.clear()
                    self._current_image = None
                    self._base_image = None
                    # Reset adjustments to defaults and sync labels (sliders stay disabled)
                    self._adj = {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0}
                    self._sync_adj_ui_from_state()
                    # Disable save controls as no image is present
                    self.save_map_btn.setEnabled(False)
                    self.save_map_action.setEnabled(False)
                    # Disable adjustment sliders when no image
                    self._set_sliders_enabled(False)
                    # Reset size estimate
                    with contextlib.suppress(Exception):
                        self.output_widget.size_estimate_value.setText('—')
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
                _on_save_complete, Qt.ConnectionType.QueuedConnection
            )
            th.start()

        except Exception as e:
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)
            error_msg = f'Ошибка при сохранении: {e}'
            logger.exception(error_msg)
            QMessageBox.critical(self, 'Ошибка', error_msg)

    # ----- Adjustments (brightness/contrast/saturation) -----
    def _sync_adj_ui_from_state(self) -> None:
        self.brightness_slider.setValue(round(self._adj['brightness'] * 100))
        self.contrast_slider.setValue(round(self._adj['contrast'] * 100))
        self.saturation_slider.setValue(round(self._adj['saturation'] * 100))
        self._update_adj_labels()

    def _update_adj_labels(self) -> None:
        self.brightness_label.setText(f'Яркость: {self.brightness_slider.value()}%')
        self.contrast_label.setText(f'Контрастность: {self.contrast_slider.value()}%')
        self.saturation_label.setText(
            f'Насыщенность: {self.saturation_slider.value()}%'
        )

    def _on_adj_slider_changed(self, *args) -> None:
        # Update state from sliders
        self._adj['brightness'] = self.brightness_slider.value() / 100.0
        self._adj['contrast'] = self.contrast_slider.value() / 100.0
        self._adj['saturation'] = self.saturation_slider.value() / 100.0
        self._update_adj_labels()
        # Propagate adjustments to model settings so they are saved in profiles
        with contextlib.suppress(Exception):
            self._controller.update_output_settings(self.output_widget.get_settings())
        # Request debounced adjust strictly via GUI signal
        self._sig_schedule_adjust.emit()
        # Also schedule size estimate (debounced)
        self._schedule_size_estimate()

    def _start_fullres_adjustment_immediate(self) -> None:
        # Immediate run strictly via GUI signal
        self._sig_run_adjust_now.emit()

    @Slot()
    def _schedule_adjust_gui(self) -> None:
        try:
            logger.debug(
                f'[ADJ] schedule_adjust_gui on_gui={QThread.currentThread() is self.thread()}'
            )
            self._preview_update_timer.start()
        except Exception as e:
            logger.warning(f'Debounce timer start failed: {e}')

    @Slot()
    def _run_adjust_now_gui(self) -> None:
        try:
            logger.debug(
                f'[ADJ] run_adjust_now_gui on_gui={QThread.currentThread() is self.thread()}'
            )
            self._preview_update_timer.stop()
        except Exception as e:
            logger.debug(f'Error stopping preview update timer: {e}')
        self._start_fullres_adjustment()

    def _start_fullres_adjustment(self) -> None:
        if self._base_image is None:
            return
        # Increment generation
        self._adj_generation += 1
        generation = self._adj_generation
        adj = dict(self._adj)

        # Disable save during processing
        self.save_map_btn.setEnabled(False)
        self.save_map_action.setEnabled(False)

        # Stop previous thread if running (let it finish; we will ignore result)
        # Create worker
        class _AdjustWorker(QObject):
            finished = Signal(int, object, str)  # generation, image or None, error

            def __init__(
                self, image: Image.Image, adj: dict[str, float], gen: int
            ) -> None:
                super().__init__()
                self.image = image
                self.adj = adj
                self.gen = gen

            @Slot()
            def run(self) -> None:
                try:
                    img = self.image
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Build LUTs
                    b = float(self.adj.get('brightness', 1.0))
                    c = float(self.adj.get('contrast', 1.0))
                    s = float(self.adj.get('saturation', 1.0))

                    # Early out if identity
                    if (
                        abs(b - 1.0) < FLOAT_COMPARISON_TOLERANCE
                        and abs(c - 1.0) < FLOAT_COMPARISON_TOLERANCE
                        and abs(s - 1.0) < FLOAT_COMPARISON_TOLERANCE
                    ):
                        self.finished.emit(self.gen, img, '')
                        return

                    def clamp(v: int) -> int:
                        return 0 if v < 0 else (min(v, 255))

                    lut_y = [0] * 256
                    lut_c = [0] * 256
                    for i in range(256):
                        y = round(((i - 128) * c + 128) * b)
                        lut_y[i] = clamp(y)
                        cc = round(128 + (i - 128) * s)
                        lut_c[i] = clamp(cc)

                    ycbcr = img.convert('YCbCr')
                    y, cb, cr = ycbcr.split()
                    y = y.point(lut_y)
                    cb = cb.point(lut_c)
                    cr = cr.point(lut_c)
                    merged = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')
                    self.finished.emit(self.gen, merged, '')
                except Exception as e:
                    self.finished.emit(self.gen, None, str(e))

        # Setup thread (canonical pattern: finished->quit, thread.finished->deleteLater)
        th = QThread()
        worker = _AdjustWorker(self._base_image, adj, generation)
        worker.moveToThread(th)
        # Store strong references until finished
        self._adjust_threads.append(th)
        self._adjust_workers.append(worker)

        th.started.connect(worker.run)

        def _on_done(gen: int, img_obj: object, err: str) -> None:
            # Ensure this slot runs in GUI thread via QueuedConnection
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)
            if gen != self._adj_generation:
                return  # stale result
            if err:
                logger.error(f'Adjustment error: {err}')
                return
            if isinstance(img_obj, Image.Image):
                logger.debug(
                    f'[ADJ] Applying adjusted image gen={gen} matches current gen; size={getattr(img_obj, "size", None)}'
                )
                self._current_image = img_obj
                try:
                    self._preview_area.set_image(img_obj)
                except Exception as ex:
                    logger.exception(f'[ADJ] set_image failed: {ex}')

        # Deliver result back to UI via class slot to guarantee GUI thread
        worker.finished.connect(
            self._on_adjust_done_slot, Qt.ConnectionType.QueuedConnection
        )
        # Stop the worker thread event loop when work is finished
        worker.finished.connect(th.quit, Qt.ConnectionType.QueuedConnection)

        # Track worker by thread for cleanup
        self._adjust_map[th] = worker
        # Cleanup in GUI slot when thread finishes
        th.finished.connect(
            self._on_adjust_thread_finished_slot, Qt.ConnectionType.QueuedConnection
        )
        th.start()
        # After adjustment result, also schedule size estimate (image content changed)
        self._schedule_size_estimate()

    def _cleanup_save_resources(self) -> None:
        """Clean up save operation resources."""
        try:
            if self._save_thread is not None:
                if self._save_thread.isRunning():
                    self._save_thread.quit()

                    # Use QTimer to defer the wait call to avoid "wait on itself" error
                    def _delayed_cleanup() -> None:
                        try:
                            if self._save_thread is not None:
                                self._save_thread.wait(1000)
                                self._save_thread.deleteLater()
                                self._save_thread = None
                        except Exception as e:
                            logger.exception(f'Error in delayed thread cleanup: {e}')

                    QTimer.singleShot(100, _delayed_cleanup)
                else:
                    self._save_thread.deleteLater()
                    self._save_thread = None

            if self._save_worker is not None:
                self._save_worker.deleteLater()
                self._save_worker = None

        except Exception as e:
            logger.exception(f'Error cleaning up save resources: {e}')

    def _set_sliders_enabled(self, enabled: bool) -> None:
        """Enable/disable sliders with enhanced fade effect."""
        sliders = [
            self.brightness_slider,
            self.contrast_slider,
            self.saturation_slider,
            self.quality_slider,
        ]

        if enabled:
            # Активное состояние - оранжевые стили
            slider_style = """
            QSlider {
                background: transparent;
            }
            QSlider::groove:horizontal {
                border: 1px solid #cc6600;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8800, stop:1 #ff9933);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ff9933, stop:1 #cc6600);
                border: 1px solid #994400;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffaa44, stop:1 #dd7711);
            }
            """
        else:
            # Неактивное состояние - сильно приглушенные стили
            slider_style = """
            QSlider {
                background: transparent;
                opacity: 0.3;
            }
            QSlider::groove:horizontal {
                border: 1px solid #e0e0e0;
                height: 8px;
                background: #f5f5f5;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #e0e0e0;
                border: 1px solid #d0d0d0;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            """

        for slider in sliders:
            slider.setEnabled(enabled)
            slider.setStyleSheet(slider_style)

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
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
            else:
                # If it’s not in list, add it temporarily
                self.profile_combo.addItem(profile_name)
                self.profile_combo.setCurrentText(profile_name)
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

    # ----- Size estimate helpers -----
    def _schedule_size_estimate(self) -> None:
        try:
            self._estimate_timer.start()
        except Exception:
            # Fallback: run immediately
            self._start_size_estimate()

    @staticmethod
    def _format_bytes(num: int) -> str:
        try:
            if num < BYTES_TO_KB_THRESHOLD:
                return f'{num} Б'
            num_float = float(num)
            for unit in ['КБ', 'МБ', 'ГБ', 'ТБ']:
                num_float /= BYTES_CONVERSION_FACTOR
                if num_float < BYTES_CONVERSION_FACTOR:
                    return f'{num_float:.1f} {unit}'
            return f'{num_float:.1f} ПБ'
        except Exception:
            return '—'

    def _start_size_estimate(self) -> None:
        # Preconditions
        base = self._base_image
        if base is None:
            self.output_widget.size_estimate_value.setText('—')
            return
        try:

            class _EstimateWorker(QObject):
                finished = Signal(int, str)  # estimate_bytes, error

                def __init__(
                    self, img: Image.Image, adj: dict[str, float], quality: int
                ) -> None:
                    super().__init__()
                    self.image = img
                    self.adj = adj
                    self.quality = int(max(10, min(100, quality)))

                @Slot()
                def run(self) -> None:
                    try:
                        img = self.image
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Apply LUT same as elsewhere
                        b = float(self.adj.get('brightness', 1.0))
                        c = float(self.adj.get('contrast', 1.0))
                        s = float(self.adj.get('saturation', 1.0))
                        if not (
                            abs(b - 1.0) < FLOAT_COMPARISON_TOLERANCE
                            and abs(c - 1.0) < FLOAT_COMPARISON_TOLERANCE
                            and abs(s - 1.0) < FLOAT_COMPARISON_TOLERANCE
                        ):

                            def clamp(v: int) -> int:
                                return 0 if v < 0 else (min(v, 255))

                            lut_y = [0] * 256
                            lut_c = [0] * 256
                            for i in range(256):
                                y = round(((i - 128) * c + 128) * b)
                                lut_y[i] = clamp(y)
                                cc = round(128 + (i - 128) * s)
                                lut_c[i] = clamp(cc)
                            ycbcr = img.convert('YCbCr')
                            y, cb, cr = ycbcr.split()
                            y = y.point(lut_y)
                            cb = cb.point(lut_c)
                            cr = cr.point(lut_c)
                            img = Image.merge('YCbCr', (y, cb, cr)).convert('RGB')

                        # Downscale for fast estimation
                        max_side = 1600
                        w, h = img.size
                        scale = 1.0
                        if max(w, h) > max_side:
                            scale = max_side / float(max(w, h))
                            new_w = max(1, round(w * scale))
                            new_h = max(1, round(h * scale))
                            img_small = img.resize(
                                (new_w, new_h), Image.Resampling.LANCZOS
                            )
                        else:
                            img_small = img
                            new_w, new_h = w, h

                        buf = BytesIO()
                        img_small.save(
                            buf,
                            'JPEG',
                            quality=self.quality,
                            optimize=True,
                            subsampling='4:4:4',
                            progressive=False,
                        )
                        bytes_down = buf.tell()
                        # Scale up by pixel count ratio and add small header constant
                        k = (w * h) / float(new_w * new_h)
                        estimate = int(bytes_down * k + 2048)
                        self.finished.emit(estimate, '')
                    except Exception as e:
                        self.finished.emit(0, str(e))

            th = QThread()
            worker = _EstimateWorker(
                base,
                dict(self._adj),
                int(getattr(self._model.settings, 'jpeg_quality', 95)),
            )
            worker.moveToThread(th)
            self._estimate_threads.append(th)
            self._estimate_workers.append(worker)

            def _on_estimate_done(estimate: int, err: str) -> None:
                if err or estimate <= 0:
                    self.output_widget.size_estimate_value.setText('—')
                else:
                    self.output_widget.size_estimate_value.setText(
                        f'≈ {self._format_bytes(estimate)}'
                    )

            th.started.connect(worker.run)
            worker.finished.connect(
                _on_estimate_done, Qt.ConnectionType.QueuedConnection
            )
            worker.finished.connect(th.quit, Qt.ConnectionType.QueuedConnection)

            def _cleanup() -> None:
                try:
                    if worker in self._estimate_workers:
                        self._estimate_workers.remove(worker)
                        worker.deleteLater()
                except Exception as e:
                    logger.debug(f'Error cleaning up estimate worker: {e}')
                try:
                    if th in self._estimate_threads:
                        self._estimate_threads.remove(th)
                        th.deleteLater()
                except Exception as e:
                    logger.debug(f'Error cleaning up estimate thread: {e}')

            th.finished.connect(_cleanup, Qt.ConnectionType.QueuedConnection)
            th.start()
        except Exception:
            self.output_widget.size_estimate_value.setText('—')

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        logger.info('Application closing - cleaning up resources')
        log_comprehensive_diagnostics('CLEANUP_START')

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
                    'Download worker did not terminate gracefully, forcing termination'
                )
                self._download_worker.terminate()
                self._download_worker.wait()
            log_thread_status('after worker termination')
            log_memory_usage('after worker termination')

        # Cleanup save resources
        self._cleanup_save_resources()
        log_memory_usage('after save resources cleanup')

        # Stop and cleanup any adjustment threads
        try:
            for th in list(self._adjust_threads):
                try:
                    th.quit()
                    th.wait(5000)
                except Exception as e:
                    logger.debug(f'Error stopping adjustment thread on close: {e}')
            # deleteLater scheduled already in their cleanup, but ensure lists are cleared
            self._adjust_threads.clear()
            self._adjust_workers.clear()
        except Exception as e:
            logger.warning(f'Error cleaning adjustment threads on close: {e}')

        # Remove observer
        self._model.remove_observer(self._observer_adapter)

        log_comprehensive_diagnostics('CLEANUP_COMPLETE')
        logger.info('Resource cleanup completed')
        event.accept()


def create_application() -> tuple[
    QApplication, MainWindow, MilMapperModel, MilMapperController
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

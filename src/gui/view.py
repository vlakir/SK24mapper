"""PySide6-based View components implementing MVC pattern."""

from __future__ import annotations

import logging
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

from diagnostics import (
    log_comprehensive_diagnostics,
    log_memory_usage,
    log_thread_status,
)
from gui.controller import MilMapperController
from gui.model import EventData, MilMapperModel, ModelEvent, Observer

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

            # Import and setup progress system with thread-safe callbacks
            from progress import set_preview_image_callback, set_spinner_callbacks

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

        # Mask opacity
        layout.addWidget(QLabel('Прозрачность маски:'), 0, 0)
        self.opacity_slider = QSlider()
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(35)
        self.opacity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.opacity_slider.setToolTip('Прозрачность белой маски (0-100%)')
        layout.addWidget(self.opacity_slider, 0, 1)

        self.opacity_label = QLabel('35%')
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f'{v}%')
        )
        layout.addWidget(self.opacity_label, 0, 2)

        # Качество JPEG
        layout.addWidget(QLabel('Качество JPG:'), 1, 0)
        self.quality_slider = QSlider()
        self.quality_slider.setRange(10, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.setOrientation(Qt.Orientation.Horizontal)
        self.quality_slider.setToolTip('Качество JPEG (10-100, 100=лучшее)')
        layout.addWidget(self.quality_slider, 1, 1)

        self.quality_label = QLabel('95')
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_label.setText(f'{v}')
        )
        layout.addWidget(self.quality_label, 1, 2)

        self.setLayout(layout)

    def get_settings(self) -> dict[str, Any]:
        """Get output settings as dictionary."""
        return {
            'mask_opacity': self.opacity_slider.value() / 100.0,
            'jpeg_quality': self.quality_slider.value(),
        }

    def set_settings(self, settings: dict[str, Any]) -> None:
        """Set output settings from dictionary."""
        opacity_percent = int(settings.get('mask_opacity', 0.35) * 100)
        self.opacity_slider.setValue(opacity_percent)
        self.opacity_label.setText(f'{opacity_percent}%')
        q = int(settings.get('jpeg_quality', 95))
        q = max(q, 10)
        q = min(q, 100)
        self.quality_slider.setValue(q)
        self.quality_label.setText(f'{q}')


if TYPE_CHECKING:
    from collections.abc import Callable


class _ViewObserver(Observer):
    """Adapter to bridge model events to the view without name clashes with QWidget.update."""

    def __init__(self, handler: Callable[[EventData], None]) -> None:
        self._handler = handler

    def update(self, event_data: EventData) -> None:  # type: ignore[override]
        self._handler(event_data)


class MainWindow(QMainWindow):
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

        self._setup_ui()
        self._setup_connections()
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
        from status_bar_proxy import StatusBarProxy

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

        # Import OptimizedImageView from preview_window
        from gui.preview_window import OptimizedImageView

        self._preview_area = OptimizedImageView()
        self._preview_area.setMinimumHeight(300)
        preview_layout.addWidget(self._preview_area)

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
        self.output_widget.opacity_slider.valueChanged.connect(
            self._on_settings_changed
        )
        self.output_widget.quality_slider.valueChanged.connect(
            self._on_settings_changed
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
        from constants import PROFILES_DIR

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
            'mask_opacity': settings.mask_opacity,
            'jpeg_quality': getattr(settings, 'jpeg_quality', 95),
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

            # Store current image for saving
            self._current_image = image

            # Display image in the preview area
            self._preview_area.set_image(image)

            # Enable save button and menu action
            self.save_map_btn.setEnabled(True)
            self.save_map_action.setEnabled(True)

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
        from pathlib import Path

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

                def __init__(self, image, path: Path, quality: int):
                    super().__init__()
                    self.image = image
                    self.path = path
                    self.quality = quality

                def run(self) -> None:
                    """Save the image in background thread."""
                    try:
                        self.image.save(
                            str(self.path), 'JPEG', quality=self.quality, optimize=True
                        )
                        self.finished.emit(True, '')
                    except Exception as e:
                        self.finished.emit(False, str(e))

            # Get current quality setting
            quality = int(getattr(self._model.settings, 'jpeg_quality', 95))

            # Create and setup worker thread
            th = QThread()
            worker = _SaveWorker(self._current_image, out_path, quality)
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
                    self.save_map_btn.setEnabled(False)
                    self.save_map_action.setEnabled(False)
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

    @Slot()
    def _new_profile(self) -> None:
        """Create new profile (placeholder)."""
        QMessageBox.information(self, 'Информация', 'Функция создания нового профиля')

    @Slot()
    def _open_profile(self) -> None:
        """Open profile file (placeholder)."""
        QMessageBox.information(self, 'Информация', 'Функция открытия профиля из файла')

    @Slot()
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            'О программе',
            'SK42mapper v0.2\n\nПриложение для создания карт в системе Гаусса-Крюгера\n'
         )

    def close_event(self, event: Any) -> None:
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
        from progress import cleanup_all_progress_resources

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

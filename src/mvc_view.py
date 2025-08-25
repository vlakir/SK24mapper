"""PySide6-based View components implementing MVC pattern."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Signal, Slot, Qt
from PySide6.QtGui import QAction, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mvc_controller import MilMapperController
from mvc_model import EventData, MilMapperModel, ModelEvent, Observer
from preview_window import OptimizedPreviewWindow

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
        try:
            # Setup thread-safe callbacks that emit signals instead of direct UI updates
            def progress_callback(done: int, total: int, label: str) -> None:
                self.progress_update.emit(done, total, label)
            
            def preview_callback(img_obj: object) -> bool:
                """Handle preview image from map generation."""
                try:
                    from PIL import Image
                    if isinstance(img_obj, Image.Image):
                        self.preview_ready.emit(img_obj)
                        return True
                    return False
                except Exception:
                    return False
            
            # Import and setup progress system with thread-safe callbacks
            from progress import set_progress_callback, set_spinner_callbacks, set_preview_image_callback
            set_progress_callback(progress_callback)
            set_spinner_callbacks(
                lambda label: self.progress_update.emit(0, 0, label),
                lambda label: None
            )
            set_preview_image_callback(preview_callback)
            
            # Run the actual download
            self._controller.start_map_download_sync()
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))


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
        self.high_spin.setToolTip(f"Старшие разряды для {label}")
        layout.addWidget(self.high_spin)
        
        # Low value input
        self.low_spin = QSpinBox()
        self.low_spin.setRange(0, 999)
        self.low_spin.setValue(low_value)
        self.low_spin.setToolTip(f"Младшие разряды для {label}")
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
        layout.addWidget(QLabel("Толщина линий (px):"), 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20)
        self.width_spin.setValue(4)
        self.width_spin.setToolTip("Толщина линий сетки в пикселях")
        layout.addWidget(self.width_spin, 0, 1)
        
        # Font size
        layout.addWidget(QLabel("Размер шрифта (px):"), 1, 0)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(10, 200)
        self.font_spin.setValue(86)
        self.font_spin.setToolTip("Размер шрифта подписей координат")
        layout.addWidget(self.font_spin, 1, 1)
        
        # Text margin
        layout.addWidget(QLabel("Отступ текста (px):"), 2, 0)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100)
        self.margin_spin.setValue(43)
        self.margin_spin.setToolTip("Отступ подписи от края изображения")
        layout.addWidget(self.margin_spin, 2, 1)
        
        # Label background padding
        layout.addWidget(QLabel("Отступ фона (px):"), 3, 0)
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 50)
        self.padding_spin.setValue(6)
        self.padding_spin.setToolTip("Внутренний отступ подложки вокруг текста")
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
        
        # Output path
        layout.addWidget(QLabel("Выходной файл:"), 0, 0)
        self.path_edit = QLineEdit("../maps/map.jpg")
        self.path_edit.setToolTip("Путь к выходному файлу")
        layout.addWidget(self.path_edit, 0, 1)
        
        self.browse_btn = QPushButton("Обзор...")
        self.browse_btn.clicked.connect(self._browse_output_path)
        self.browse_btn.setToolTip("Выбрать файл для сохранения")
        layout.addWidget(self.browse_btn, 0, 2)
        
        # Mask opacity
        layout.addWidget(QLabel("Прозрачность маски:"), 1, 0)
        self.opacity_slider = QSlider()
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(35)
        self.opacity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.opacity_slider.setToolTip("Прозрачность белой маски (0-100%)")
        layout.addWidget(self.opacity_slider, 1, 1)
        
        self.opacity_label = QLabel("35%")
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{v}%")
        )
        layout.addWidget(self.opacity_label, 1, 2)
        
        # PNG compression
        layout.addWidget(QLabel("Сжатие PNG:"), 2, 0)
        self.compress_spin = QSpinBox()
        self.compress_spin.setRange(0, 9)
        self.compress_spin.setValue(6)
        self.compress_spin.setToolTip("Уровень сжатия PNG (0-9, 9=максимум)")
        layout.addWidget(self.compress_spin, 2, 1)
        
        self.setLayout(layout)
    
    @Slot()
    def _browse_output_path(self) -> None:
        """Open file dialog to select output path."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Выберите файл для сохранения",
            self.path_edit.text(),
            "Изображения (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self.path_edit.setText(file_path)
    
    def get_settings(self) -> dict[str, Any]:
        """Get output settings as dictionary."""
        return {
            'output_path': self.path_edit.text(),
            'mask_opacity': self.opacity_slider.value() / 100.0,
            'png_compress_level': self.compress_spin.value(),
        }
    
    def set_settings(self, settings: dict[str, Any]) -> None:
        """Set output settings from dictionary."""
        self.path_edit.setText(settings.get('output_path', '../maps/map.jpg'))
        opacity_percent = int(settings.get('mask_opacity', 0.35) * 100)
        self.opacity_slider.setValue(opacity_percent)
        self.opacity_label.setText(f"{opacity_percent}%")
        self.compress_spin.setValue(settings.get('png_compress_level', 6))


class MainWindow(QMainWindow, Observer):
    """Main application window implementing Observer pattern."""
    
    def __init__(self, model: MilMapperModel, controller: MilMapperController) -> None:
        super().__init__()
        self._model = model
        self._controller = controller
        self._download_worker: DownloadWorker | None = None
        self._preview_window: OptimizedPreviewWindow | None = None
        
        # Register as observer
        self._model.add_observer(self)
        
        self._setup_ui()
        self._setup_connections()
        self._load_initial_data()
        logger.info("MainWindow initialized")
    
    def _setup_ui(self) -> None:
        """Setup the main window UI."""
        self.setWindowTitle("Mil Mapper 2.0 - Создание военных карт")
        self.setMinimumSize(800, 600)
        
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
        self.status_bar.showMessage("Готов к работе")
        
        # Profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Профиль:"))
        
        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip("Выберите профиль настроек")
        profile_layout.addWidget(self.profile_combo)
        
        self.load_profile_btn = QPushButton("Загрузить")
        self.load_profile_btn.setToolTip("Загрузить выбранный профиль")
        profile_layout.addWidget(self.load_profile_btn)
        
        self.save_profile_btn = QPushButton("Сохранить")
        self.save_profile_btn.setToolTip("Сохранить текущие настройки в профиль")
        profile_layout.addWidget(self.save_profile_btn)
        
        profile_layout.addStretch()
        main_layout.addLayout(profile_layout)
        
        # Coordinates section
        coords_frame = QFrame()
        coords_frame.setFrameStyle(QFrame.StyledPanel)
        coords_layout = QVBoxLayout()
        coords_frame.setLayout(coords_layout)
        
        coords_layout.addWidget(QLabel("Координаты области (СК-42 Гаусса-Крюгера):"))
        
        self.from_x_widget = CoordinateInputWidget("От X:", 54, 14)
        self.from_y_widget = CoordinateInputWidget("От Y:", 74, 43)
        self.to_x_widget = CoordinateInputWidget("До X:", 54, 23)
        self.to_y_widget = CoordinateInputWidget("До Y:", 74, 49)
        
        coords_layout.addWidget(self.from_x_widget)
        coords_layout.addWidget(self.from_y_widget)
        coords_layout.addWidget(self.to_x_widget)
        coords_layout.addWidget(self.to_y_widget)
        
        main_layout.addWidget(coords_frame)
        
        # Grid settings section
        grid_frame = QFrame()
        grid_frame.setFrameStyle(QFrame.StyledPanel)
        grid_layout = QVBoxLayout()
        grid_frame.setLayout(grid_layout)
        
        grid_layout.addWidget(QLabel("Настройки сетки:"))
        self.grid_widget = GridSettingsWidget()
        grid_layout.addWidget(self.grid_widget)
        
        main_layout.addWidget(grid_frame)
        
        # Output settings section
        output_frame = QFrame()
        output_frame.setFrameStyle(QFrame.StyledPanel)
        output_layout = QVBoxLayout()
        output_frame.setLayout(output_layout)
        
        output_layout.addWidget(QLabel("Настройки вывода:"))
        self.output_widget = OutputSettingsWidget()
        output_layout.addWidget(self.output_widget)
        
        main_layout.addWidget(output_frame)
        
        # Progress section
        progress_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        progress_layout.addWidget(self.progress_label)
        
        main_layout.addLayout(progress_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.download_btn = QPushButton("Создать карту")
        self.download_btn.setToolTip("Начать создание карты")
        self.download_btn.setStyleSheet("QPushButton { font-weight: bold; }")
        button_layout.addWidget(self.download_btn)
        
        main_layout.addLayout(button_layout)
    
    def _create_menu(self) -> None:
        """Create application menu."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("Файл")
        
        new_action = QAction("Новый профиль", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_profile)
        file_menu.addAction(new_action)
        
        open_action = QAction("Открыть профиль...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_profile)
        file_menu.addAction(open_action)
        
        save_action = QAction("Сохранить профиль", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_current_profile)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Справка")
        
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_connections(self) -> None:
        """Setup signal connections."""
        # Profile management
        self.load_profile_btn.clicked.connect(self._load_selected_profile)
        self.save_profile_btn.clicked.connect(self._save_current_profile)
        
        # Download
        self.download_btn.clicked.connect(self._start_download)
        
        # Settings change tracking
        self._connect_setting_changes()
    
    def _connect_setting_changes(self) -> None:
        """Connect all setting change signals."""
        # Coordinates
        for widget in [self.from_x_widget, self.from_y_widget, 
                      self.to_x_widget, self.to_y_widget]:
            widget.high_spin.valueChanged.connect(self._on_settings_changed)
            widget.low_spin.valueChanged.connect(self._on_settings_changed)
        
        # Grid settings
        self.grid_widget.width_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.font_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.margin_spin.valueChanged.connect(self._on_settings_changed)
        self.grid_widget.padding_spin.valueChanged.connect(self._on_settings_changed)
        
        # Output settings
        self.output_widget.path_edit.textChanged.connect(self._on_settings_changed)
        self.output_widget.opacity_slider.valueChanged.connect(self._on_settings_changed)
        self.output_widget.compress_spin.valueChanged.connect(self._on_settings_changed)
    
    def _load_initial_data(self) -> None:
        """Load initial application data."""
        # Load available profiles
        profiles = self._controller.get_available_profiles()
        self.profile_combo.addItems(profiles)
        
        # Load default profile
        if "default" in profiles:
            self.profile_combo.setCurrentText("default")
            self._controller.load_profile_by_name("default")
    
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
            'from_x_high': from_x_high, 'from_x_low': from_x_low,
            'from_y_high': from_y_high, 'from_y_low': from_y_low,
            'to_x_high': to_x_high, 'to_x_low': to_x_low,
            'to_y_high': to_y_high, 'to_y_low': to_y_low,
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
    def _start_download(self) -> None:
        """Start map download process."""
        if self._download_worker and self._download_worker.isRunning():
            QMessageBox.information(self, "Информация", "Загрузка уже выполняется")
            return
        
        self._download_worker = DownloadWorker(self._controller)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.progress_update.connect(self._update_progress)
        self._download_worker.preview_ready.connect(self._show_preview_window)
        self._download_worker.start()
        
        # Update UI state
        self.download_btn.setEnabled(False)
        self.status_bar.showMessage("Загрузка карты...")
    
    @Slot(bool, str)
    def _on_download_finished(self, success: bool, error_msg: str) -> None:
        """Handle download completion."""
        self.download_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        
        if success:
            self.status_bar.showMessage("Карта успешно создана", 5000)
            QMessageBox.information(self, "Успех", "Карта успешно создана!")
        else:
            self.status_bar.showMessage("Ошибка при создании карты", 5000)
            QMessageBox.critical(self, "Ошибка", f"Не удалось создать карту:\n{error_msg}")
    
    def update(self, event_data: EventData) -> None:
        """Handle model events (Observer pattern)."""
        event = event_data.event
        data = event_data.data
        
        if event == ModelEvent.SETTINGS_CHANGED:
            self._update_ui_from_settings(data.get('settings'))
        elif event == ModelEvent.PROFILE_LOADED:
            self._update_ui_from_settings(data.get('settings'))
            self.status_bar.showMessage(f"Профиль загружен: {data.get('profile_name')}", 3000)
        elif event == ModelEvent.DOWNLOAD_PROGRESS:
            self._update_progress(data.get('done', 0), data.get('total', 0), data.get('label', ''))
        elif event == ModelEvent.PREVIEW_UPDATED:
            self._show_preview_window(data.get('image'))
        elif event == ModelEvent.ERROR_OCCURRED:
            error_msg = data.get('error', 'Неизвестная ошибка')
            self.status_bar.showMessage(f"Ошибка: {error_msg}", 5000)
            QMessageBox.warning(self, "Предупреждение", error_msg)
    
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
            'png_compress_level': settings.png_compress_level,
        }
        self.output_widget.set_settings(output_settings)
    
    def _update_progress(self, done: int, total: int, label: str) -> None:
        """Update progress indicators."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(done)
            self.progress_bar.setVisible(True)
        
        self.progress_label.setText(label)
    
    def _show_preview_window(self, image: Any) -> None:
        """Show preview window with the generated map image."""
        try:
            from PIL import Image
            if not isinstance(image, Image.Image):
                logger.warning("Invalid image object for preview")
                return
            
            # Create preview window if it doesn't exist or was closed
            if self._preview_window is None:
                self._preview_window = OptimizedPreviewWindow(self)
                
                # Connect save signal to update status
                def on_image_saved(file_path: str) -> None:
                    from pathlib import Path
                    self.status_bar.showMessage(f"Карта сохранена: {Path(file_path).name}", 5000)
                
                self._preview_window.saved.connect(on_image_saved)
            
            # Show the image in preview window
            self._preview_window.show_image(image)
            self._preview_window.show()
            self._preview_window.raise_()
            self._preview_window.activateWindow()
            
            logger.info("Preview window displayed")
            
        except Exception as e:
            error_msg = f"Ошибка при отображении предпросмотра: {e}"
            logger.error(error_msg)
            QMessageBox.warning(self, "Ошибка предпросмотра", error_msg)
    
    @Slot()
    def _new_profile(self) -> None:
        """Create new profile (placeholder)."""
        QMessageBox.information(self, "Информация", "Функция создания нового профиля")
    
    @Slot()
    def _open_profile(self) -> None:
        """Open profile file (placeholder)."""
        QMessageBox.information(self, "Информация", "Функция открытия профиля из файла")
    
    @Slot()
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "О программе",
            "Mil Mapper 2.0\n\nПриложение для создания военных карт\n"
            "Миграция с tkinter на PySide6\n\nВерсия 2.0"
        )
    
    def closeEvent(self, event: Any) -> None:
        """Handle window close event."""
        # Cleanup
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.quit()
            self._download_worker.wait()
        
        # Close preview window if open
        if self._preview_window:
            self._preview_window.close()
            self._preview_window = None
        
        self._model.remove_observer(self)
        event.accept()


def create_application() -> tuple[QApplication, MainWindow, MilMapperModel, MilMapperController]:
    """Create and configure the PySide6 application."""
    app = QApplication([])
    app.setApplicationName("Mil Mapper")
    app.setApplicationVersion("2.0")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create MVC components
    model = MilMapperModel()
    controller = MilMapperController(model)
    window = MainWindow(model, controller)
    
    return app, window, model, controller
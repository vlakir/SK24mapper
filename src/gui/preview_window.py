"""Optimized preview window using QGraphicsView for better performance."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from constants import PREVIEW_ROTATION_ANGLE

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class OptimizedImageView(QGraphicsView):
    """High-performance image view using QGraphicsView for zoom and pan operations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Create graphics scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Image item for display
        self._image_item: QGraphicsPixmapItem | None = None
        self._original_image: Image.Image | None = None

        # Configure view for optimal performance with proper antialiasing for thin lines
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Zoom to cursor: anchor transformations under the mouse pointer
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Solution 1: Enable antialiasing for all elements
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Solution 5: Additional render hints for improved thin line rendering
        self.setRenderHint(QPainter.RenderHint.LosslessImageRendering, True)

        # Solution 2: Remove DontAdjustForAntialiasing flag to allow proper thin line rendering
        # self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)

        # Solution 5: Note - DontClipPainter flag not available in current PySide6 version
        # self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontClipPainter, True)

        # Keep performance optimizations that don't affect thin line rendering
        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontSavePainterState, True
        )

        # Solution 5: Use FullViewportUpdate for critical thin line rendering
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Zoom limits
        self._min_zoom = 0.1  # Will be updated to fit-to-window scale
        self._max_zoom = 10.0
        self._zoom_factor = 1.15
        self._fit_to_window_scale = 1.0  # Store the fit-to-window scale as minimum

        # Enable mouse tracking for smooth interactions
        self.setMouseTracking(True)

    def set_image(self, pil_image: Image.Image) -> None:
        """Set the image to display with fixed rotation to improve thin line visibility."""
        self._original_image = pil_image

        # Convert PIL image to QPixmap efficiently
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Create QImage directly from PIL data
        width, height = pil_image.size
        image_data = pil_image.tobytes()
        qimage = QImage(
            image_data, width, height, width * 3, QImage.Format.Format_RGB888
        )

        # Create QPixmap from QImage
        qpixmap = QPixmap.fromImage(qimage)

        # Clear scene and add image
        self._scene.clear()
        self._image_item = self._scene.addPixmap(qpixmap)

        # Apply fixed rotation to improve thin line visibility
        if PREVIEW_ROTATION_ANGLE != 0:
            transform = QTransform()
            transform.rotate(PREVIEW_ROTATION_ANGLE)
            self._image_item.setTransform(transform)

            # Update scene rect to account for rotation
            rotated_rect = transform.mapRect(qpixmap.rect())
            self._scene.setSceneRect(rotated_rect)
        else:
            # Set scene rect to image bounds
            self._scene.setSceneRect(qpixmap.rect())

        # Fit image to view
        self.fit_to_window()

    def clear(self) -> None:
        """Clear the preview area."""
        self._scene.clear()
        self._image_item = None
        self._original_image = None

    def fit_to_window(self) -> None:
        """Fit image to the view window."""
        if self._image_item:
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
            # Store the fit-to-window scale as the minimum zoom limit
            self._fit_to_window_scale = self.transform().m11()
            self._min_zoom = self._fit_to_window_scale

    def zoom_in(self) -> None:
        """Zoom in by the defined zoom factor."""
        self._zoom(self._zoom_factor)

    def zoom_out(self) -> None:
        """Zoom out by the defined zoom factor."""
        self._zoom(1.0 / self._zoom_factor)

    def reset_zoom(self) -> None:
        """Reset zoom to fit the entire image."""
        self.fit_to_window()

    def _zoom(self, factor: float) -> None:
        """Apply zoom with limits and pixel-perfect alignment."""
        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        # Apply zoom limits
        if new_scale < self._min_zoom:
            factor = self._min_zoom / current_scale
        elif new_scale > self._max_zoom:
            factor = self._max_zoom / current_scale

        # Apply zoom
        self.scale(factor, factor)

        # Solution #4: Force pixel-perfect alignment for thin lines
        transform = self.transform()
        self.setTransform(
            QTransform(
                round(transform.m11() * 100) / 100,  # Round scale factors
                transform.m12(),
                transform.m21(),
                round(transform.m22() * 100) / 100,
                round(transform.dx()),  # Round translation
                round(transform.dy()),
            )
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming with zoom-to-cursor behavior."""
        if not self._image_item:
            return

        # Determine scroll amount (support angleDelta and pixelDelta for touchpads)
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            event.ignore()
            return

        zoom_in = delta > 0
        factor = self._zoom_factor if zoom_in else (1.0 / self._zoom_factor)

        # Apply zoom (anchor is under mouse, no manual centering needed)
        self._zoom(factor)
        event.accept()

    def mousePressEvent(self, event) -> None:
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        """Handle widget resize to update fit-to-window scale."""
        super().resizeEvent(event)
        if self._image_item:
            # Recalculate fit-to-window scale when widget is resized
            current_scale = self.transform().m11()
            self.resetTransform()
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fit_to_window_scale = self.transform().m11()
            self._min_zoom = self._fit_to_window_scale
            # Restore the previous scale if it was larger than fit-to-window
            if current_scale > self._fit_to_window_scale:
                factor = current_scale / self._fit_to_window_scale
                self.scale(factor, factor)


class OptimizedPreviewWindow(QDialog):
    """Optimized preview window using QGraphicsView for better performance."""

    saved = Signal(str)  # Emitted when image is saved with file path

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Предпросмотр карты (Оптимизированный)')
        self.setModal(True)
        self.resize(1000, 800)

        # Store the image for saving
        self._image: Image.Image | None = None

        self._setup_ui()
        self._setup_connections()

        logger.info('OptimizedPreviewWindow initialized')

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Create optimized image view
        self._image_view = OptimizedImageView(self)
        layout.addWidget(self._image_view)

        # Button layout
        button_layout = QHBoxLayout()

        # Zoom controls
        self._zoom_in_btn = QPushButton('Увеличить (+)')
        self._zoom_out_btn = QPushButton('Уменьшить (-)')
        self._fit_btn = QPushButton('По размеру окна')

        # Action buttons
        self._save_btn = QPushButton('Сохранить')
        self._close_btn = QPushButton('Закрыть')

        # Add buttons to layout
        button_layout.addWidget(self._zoom_in_btn)
        button_layout.addWidget(self._zoom_out_btn)
        button_layout.addWidget(self._fit_btn)
        button_layout.addStretch()
        button_layout.addWidget(self._save_btn)
        button_layout.addWidget(self._close_btn)

        layout.addLayout(button_layout)

    def _setup_connections(self) -> None:
        """Connect signals and slots."""
        self._zoom_in_btn.clicked.connect(self._image_view.zoom_in)
        self._zoom_out_btn.clicked.connect(self._image_view.zoom_out)
        self._fit_btn.clicked.connect(self._image_view.fit_to_window)
        self._save_btn.clicked.connect(self._save_image)
        self._close_btn.clicked.connect(self.reject)

    def show_image(self, pil_image: Image.Image) -> None:
        """Display the given PIL image."""
        self._image = pil_image
        self._image_view.set_image(pil_image)
        logger.info(f'Displaying image: {pil_image.size}')

    def _save_image(self) -> None:
        """Save the current image to file."""
        if not self._image:
            QMessageBox.warning(
                self, 'Предупреждение', 'Нет изображения для сохранения'
            )
            return

        # Get save file path with default directory <project root>/maps
        from pathlib import Path

        maps_dir = Path(__file__).resolve().parent.parent.parent / 'maps'
        maps_dir.mkdir(exist_ok=True)
        default_path = str(maps_dir / 'map.png')
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Сохранить карту',
            default_path,
            'PNG files (*.png);;JPEG files (*.jpg);;All files (*.*)',
        )

        if not file_path:
            return

        try:
            # Save image
            self._image.save(file_path)
            logger.info(f'Image saved to: {file_path}')

            # Emit saved signal
            self.saved.emit(file_path)

            # Show success message
            QMessageBox.information(
                self, 'Успех', f'Карта успешно сохранена:\n{file_path}'
            )

        except Exception as e:
            logger.exception(f'Failed to save image: {e}')
            QMessageBox.critical(
                self, 'Ошибка', f'Не удалось сохранить изображение:\n{e!s}'
            )

    def keyPressEvent(self, event: Any) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self._image_view.zoom_in()
        elif event.key() == Qt.Key.Key_Minus:
            self._image_view.zoom_out()
        elif event.key() == Qt.Key.Key_0:
            self._image_view.fit_to_window()
        elif (
            event.key() == Qt.Key.Key_S
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self._save_image()
        else:
            super().keyPressEvent(event)

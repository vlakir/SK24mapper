"""Optimized preview window using QGraphicsView for better performance."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from shared.constants import PREVIEW_ROTATION_ANGLE

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

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        self.setRenderHint(QPainter.RenderHint.LosslessImageRendering, True)

        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontSavePainterState,
            True,
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

        self._qimage_bytes: bytes | None = None

    def set_image(self, pil_image: Image.Image) -> None:
        """
        Set the image to display with fixed rotation to improve thin line visibility.

        Keeps current zoom/center if an image is already displayed.
        """
        # Preserve current view transform and center if already showing an image
        preserve_transform = self._image_item is not None
        current_transform = QTransform(self.transform()) if preserve_transform else None
        current_center = (
            self.mapToScene(self.viewport().rect().center())
            if preserve_transform
            else None
        )

        self._original_image = pil_image

        # Convert PIL image to QPixmap efficiently
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Create QImage directly from PIL data
        width, height = pil_image.size
        image_data = pil_image.tobytes()
        # Keep a reference to the backing bytes to prevent premature GC
        self._qimage_bytes = image_data
        qimage = QImage(
            self._qimage_bytes,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
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

        # Fit or restore transform
        if preserve_transform and current_transform is not None:
            # Restore previous transform and keep center
            self.setTransform(current_transform)
            if current_center is not None:
                self.centerOn(current_center)
        else:
            # First time: fit image to view
            self.fit_to_window()

    def clear(self) -> None:
        """Clear the preview area and release pixmap resources."""
        # If there is an existing pixmap item, replace its pixmap with an empty one
        if self._image_item is not None:
            self._image_item.setPixmap(QPixmap())
        self._scene.clear()
        self._image_item = None
        self._original_image = None
        # Allow GC of previous image bytes when clearing
        if hasattr(self, '_qimage_bytes'):
            self._qimage_bytes = None

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
            ),
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


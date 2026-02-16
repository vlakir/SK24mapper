"""Optimized preview window using QGraphicsView for better performance."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QMovie,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QLabel,
    QWidget,
)

from shared.constants import (
    CONTROL_POINT_SIZE_M,
    PREVIEW_MIN_LINE_LENGTH_FOR_LABEL,
    PREVIEW_ROTATION_ANGLE,
    PREVIEW_UPRIGHT_TEXT_ANGLE_LIMIT,
)

if TYPE_CHECKING:
    from PIL import Image

import math

logger = logging.getLogger(__name__)


class OptimizedImageView(QGraphicsView):
    """High-performance image view using QGraphicsView for zoom and pan operations."""

    mouse_moved_on_map = Signal(object, object)  # (x, y) or (None, None)
    map_right_clicked = Signal(float, float)  # (x, y)
    shift_wheel_rotated = Signal(float)  # delta_degrees (positive = CW)
    shift_key_released = Signal()  # Shift released after rotation

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Create graphics scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Image item for display
        self._image_item: QGraphicsPixmapItem | None = None
        self._original_image: Image.Image | None = None

        # Prevent concurrent wheel event processing
        self._processing_wheel_event = False
        self._updating_image = False

        # Configure view for optimal performance with proper antialiasing for thin lines
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Zoom to cursor: anchor transformations under the mouse pointer
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        render_hint_enabled = True
        self.setRenderHint(QPainter.RenderHint.Antialiasing, render_hint_enabled)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, render_hint_enabled)
        self.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            render_hint_enabled,
        )

        self.setRenderHint(
            QPainter.RenderHint.LosslessImageRendering,
            render_hint_enabled,
        )

        self.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontSavePainterState,
            enabled=True,
        )

        # Solution 5: Use FullViewportUpdate for critical thin line rendering
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Zoom limits
        self._min_zoom = 0.1  # Will be updated to fit-to-window scale
        self._max_zoom = 10.0  # Will be updated relative to fit-to-window scale
        self._zoom_factor = 1.15
        self._fit_to_window_scale = 1.0  # Store the fit-to-window scale as minimum
        self._max_zoom_multiplier = 20.0  # Allow 20x zoom from fit-to-window

        # Enable mouse tracking for smooth interactions
        self.setMouseTracking(True)

        self._qimage_bytes: bytes | None = None
        self._cp_cross_items: list[QGraphicsLineItem] = []
        self._cp_line_item: QGraphicsLineItem | None = None
        self._cp_label_item: QGraphicsTextItem | None = None
        self._meters_per_px: float = 0.0

        # Radar azimuth indicator line (dashed, shown during rotation)
        self._azimuth_line_item: QGraphicsLineItem | None = None
        self._azimuth_label_item: QGraphicsTextItem | None = None
        # Sector boundary lines (dashed, shown during rotation)
        self._sector_line_items: list[QGraphicsLineItem] = []

        # Loading overlay (QLabel + QMovie over viewport)
        self._loading_overlay = self._create_loading_overlay()

    def set_image(self, pil_image: Image.Image, meters_per_px: float = 0.0) -> None:
        """
        Set the image to display with fixed rotation to improve thin line visibility.

        Keeps current zoom/center if an image is already displayed.
        """
        self.stop_loading()
        try:
            self._updating_image = True

            # Preserve current view transform and center if already showing an image
            preserve_transform = self._image_item is not None
            current_transform = (
                QTransform(self.transform()) if preserve_transform else None
            )
            current_center = (
                self.mapToScene(self.viewport().rect().center())
                if preserve_transform
                else None
            )

            self._original_image = pil_image
            self._meters_per_px = meters_per_px

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

            # Clear scene and add image (invalidates all scene item references)
            self._scene.clear()
            self._cp_cross_items = []
            self._cp_line_item = None
            self._cp_label_item = None
            self._azimuth_line_item = None
            self._azimuth_label_item = None
            self._sector_line_items = []
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

        finally:
            self._updating_image = False

    def clear(self) -> None:
        """Clear the preview area and release pixmap resources."""
        self.stop_loading()
        # If there is an existing pixmap item, replace its pixmap with an empty one
        if self._image_item is not None:
            self._image_item.setPixmap(QPixmap())
        self._scene.clear()
        self._image_item = None
        self._original_image = None
        self._cp_cross_items = []
        self._cp_line_item = None
        self._cp_label_item = None
        self._meters_per_px = 0.0
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
            # Set max zoom relative to fit-to-window scale for consistent behavior
            self._max_zoom = self._fit_to_window_scale * self._max_zoom_multiplier

    def zoom_in(self) -> None:
        """Zoom in by the defined zoom factor."""
        self._zoom(self._zoom_factor)

    def zoom_out(self) -> None:
        """Zoom out by the defined zoom factor."""
        self._zoom(1.0 / self._zoom_factor)

    def reset_zoom(self) -> None:
        """Reset zoom to fit the entire image."""
        self.fit_to_window()

    def set_control_point_marker(self, x: float, y: float) -> None:
        """Draw a red cross marker at the specified scene coordinates."""
        if self._image_item is None or self._meters_per_px <= 0:
            return

        # Clear existing cross if any
        self.clear_control_point_markers()

        # Calculate cross size in pixels based on CONTROL_POINT_SIZE_M
        ppm = 1.0 / self._meters_per_px
        size_px = max(10, round(CONTROL_POINT_SIZE_M * ppm))
        half = size_px / 2.0

        # Create pen for the red cross
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        pen.setCosmetic(True)  # Line width stays 2px regardless of zoom

        # Create cross lines
        line1 = self._scene.addLine(x - half, y - half, x + half, y + half, pen)
        line2 = self._scene.addLine(x - half, y + half, x + half, y - half, pen)

        # Store items to manage their lifecycle
        self._cp_cross_items = [line1, line2]

    def clear_control_point_markers(self) -> None:
        """Remove control point markers (cross and line) from the scene."""
        for item in self._cp_cross_items:
            with contextlib.suppress(Exception):
                self._scene.removeItem(item)
        self._cp_cross_items.clear()

        if self._cp_line_item:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._cp_line_item)
            self._cp_line_item = None

        if self._cp_label_item:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._cp_label_item)
            self._cp_label_item = None

    def set_control_point_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        distance_m: float | None = None,
        azimuth_deg: float | None = None,
        name: str | None = None,
    ) -> None:
        """Draw a red dashed line between two points with an optional label."""
        if self._image_item is None:
            return

        # Clear existing line and label
        if self._cp_line_item:
            self._scene.removeItem(self._cp_line_item)
            self._cp_line_item = None
        if self._cp_label_item:
            self._scene.removeItem(self._cp_label_item)
            self._cp_label_item = None

        # Create pen
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)  # Fixed width matching grid cross
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)

        # Create line
        self._cp_line_item = self._scene.addLine(x1, y1, x2, y2, pen)

        # Create combined label if distance is provided
        if distance_m is not None:
            # If distance in pixels is too small, don't show label
            # to avoid "jumping" and overlapping.
            # We use scene coordinates distance.
            dist_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist_px < PREVIEW_MIN_LINE_LENGTH_FOR_LABEL:
                # Hide label if line is shorter
                return

            # Format combined text: "<name>: <azimuth> <distance>"
            # For example: "КП1: 45° 123 м"
            name_text = f'{name}: ' if name else ''
            azimuth_text = f'{round(azimuth_deg)}° ' if azimuth_deg is not None else ''
            distance_text = f'{round(distance_m)} м'
            text = f'{name_text}{azimuth_text}{distance_text}'

            self._cp_label_item = QGraphicsTextItem(text)
            self._cp_label_item.setDefaultTextColor(QColor(255, 0, 0))

            # Make font larger
            font = self._cp_label_item.font()
            font.setPointSize(12)  # Increase font size (default is usually 8 or 9)
            font.setBold(True)
            self._cp_label_item.setFont(font)

            # Position at the middle of the line
            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            self._cp_label_item.setPos(mid_x, mid_y)

            # Calculate angle of the line
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)

            # Ensure text is not upside down (readable from left to right)
            if (
                angle_deg > PREVIEW_UPRIGHT_TEXT_ANGLE_LIMIT
                or angle_deg < -PREVIEW_UPRIGHT_TEXT_ANGLE_LIMIT
            ):
                angle_deg += 180

            # Make it stay same size on screen regardless of zoom
            self._cp_label_item.setFlag(
                QGraphicsTextItem.GraphicsItemFlag.ItemIgnoresTransformations
            )

            # Center the text horizontally and place it above its position
            rect = self._cp_label_item.boundingRect()

            # Use QTransform to shift the text so its bottom-center is at (0,0) locally,
            # then rotate it.
            transform = QTransform()
            # Perform translation first to anchor bottom-center at (0,0)
            transform.translate(-rect.width() / 2.0, -rect.height())

            # Then apply rotation around that anchor point
            transform.rotate(angle_deg)
            self._cp_label_item.setTransform(transform)

            self._cp_label_item.setZValue(10)
            self._scene.addItem(self._cp_label_item)

    def set_azimuth_line(
        self,
        cx: float,
        cy: float,
        azimuth_deg: float,
        length_px: float,
        sector_width_deg: float = 0.0,
        rotation_deg: float = 0.0,
    ) -> None:
        """
        Draw a dashed azimuth indicator line from (cx, cy) in azimuth direction.

        Provides instant visual feedback during Shift+wheel rotation.
        Optionally draws sector boundary lines when sector_width_deg > 0.

        Args:
            cx: X-coordinate of the azimuth line origin (pixels).
            cy: Y-coordinate of the azimuth line origin (pixels).
            azimuth_deg: Azimuth angle in degrees (0=north, clockwise).
            length_px: Length of the indicator line in pixels.
            sector_width_deg: Sector width (degrees); 0 = no sector.
            rotation_deg: Map rotation angle to compensate (so the indicator
                aligns with the pixel-level sector mask).

        """
        if self._image_item is None:
            return

        self.clear_azimuth_line()

        # Pen for azimuth center line (bright cyan, dashed, cosmetic)
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCosmetic(True)

        # Effective azimuth in image coordinates (compensate map rotation)
        eff_az = azimuth_deg - rotation_deg

        # Azimuth: 0=north, CW → screen: dx=sin(az), dy=-cos(az)
        az_rad = math.radians(eff_az)
        ex = cx + length_px * math.sin(az_rad)
        ey = cy - length_px * math.cos(az_rad)

        self._azimuth_line_item = self._scene.addLine(cx, cy, ex, ey, pen)
        self._azimuth_line_item.setZValue(20)

        # Label with geographic azimuth angle (not compensated) — placed at
        # ~15% of the line from center so it stays visible.
        label_frac = min(0.30, 400.0 / length_px) if length_px > 0 else 0.30
        label_x = cx + length_px * label_frac * math.sin(az_rad)
        label_y = cy - length_px * label_frac * math.cos(az_rad)

        label_text = f'{azimuth_deg:.0f}°'
        self._azimuth_label_item = QGraphicsTextItem(label_text)
        self._azimuth_label_item.setDefaultTextColor(QColor(0, 255, 255))
        font = self._azimuth_label_item.font()
        font.setPointSize(14)
        font.setBold(True)
        self._azimuth_label_item.setFont(font)
        self._azimuth_label_item.setFlag(
            QGraphicsTextItem.GraphicsItemFlag.ItemIgnoresTransformations
        )
        self._azimuth_label_item.setPos(label_x, label_y)
        self._azimuth_label_item.setZValue(20)
        self._scene.addItem(self._azimuth_label_item)

        # Sector boundary lines (semi-transparent cyan, dashed)
        if sector_width_deg > 0:
            half = sector_width_deg / 2.0
            sector_pen = QPen(QColor(0, 255, 255, 100))
            sector_pen.setWidth(1)
            sector_pen.setStyle(Qt.PenStyle.DashLine)
            sector_pen.setCosmetic(True)

            for offset_deg in (-half, half):
                edge_rad = math.radians(eff_az + offset_deg)
                sx = cx + length_px * math.sin(edge_rad)
                sy = cy - length_px * math.cos(edge_rad)
                item = self._scene.addLine(cx, cy, sx, sy, sector_pen)
                item.setZValue(19)
                self._sector_line_items.append(item)

    def clear_azimuth_line(self) -> None:
        """Remove azimuth indicator line and sector boundaries from scene."""
        if self._azimuth_line_item:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._azimuth_line_item)
            self._azimuth_line_item = None

        if self._azimuth_label_item:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._azimuth_label_item)
            self._azimuth_label_item = None

        for item in self._sector_line_items:
            with contextlib.suppress(Exception):
                self._scene.removeItem(item)
        self._sector_line_items.clear()

    # ------------------------------------------------------------------
    # Loading overlay (GIF + label, parented to viewport)
    # ------------------------------------------------------------------

    def _create_loading_overlay(self) -> QWidget:
        """Build a full-viewport overlay with Matrix rain GIF and text."""
        overlay = QWidget(self.viewport())
        overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        overlay.setStyleSheet('background: black;')
        overlay.hide()

        # GIF fills the whole overlay via QMovie pre-scaling
        gif_label = QLabel(overlay)
        gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gif_label.setStyleSheet('background: black;')

        gif_path = Path(__file__).parent / 'icons' / 'loading.gif'
        movie = QMovie(str(gif_path))
        gif_label.setMovie(movie)

        self._loading_movie = movie
        self._loading_gif_label = gif_label
        return overlay

    def _layout_loading_overlay(self) -> None:
        """Position GIF inside the overlay and pre-scale movie to viewport."""
        rect = self._loading_overlay.rect()
        self._loading_gif_label.setGeometry(rect)
        self._loading_movie.setScaledSize(rect.size())

    def start_loading(self) -> None:
        """Show the loading overlay with animated GIF."""
        vp = self.viewport().rect()
        self._loading_overlay.setGeometry(vp)
        self._layout_loading_overlay()
        self._loading_movie.start()
        self._loading_overlay.show()
        self._loading_overlay.raise_()

    def stop_loading(self) -> None:
        """Hide the loading overlay."""
        self._loading_movie.stop()
        self._loading_overlay.hide()

    def _zoom(self, factor: float) -> None:
        """Apply zoom with limits and pixel-perfect alignment."""
        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        # Use tolerance to handle floating-point precision issues
        tolerance = 0.001

        # Apply zoom limits with tolerance
        if new_scale < self._min_zoom * (1.0 - tolerance):
            factor = self._min_zoom / current_scale
        elif new_scale > self._max_zoom * (1.0 + tolerance):
            factor = self._max_zoom / current_scale

        # Apply zoom
        self.scale(factor, factor)

        # Solution #4: Force pixel-perfect alignment for thin lines
        # Use finer rounding (3 decimal places) to reduce rounding errors
        transform = self.transform()
        self.setTransform(
            QTransform(
                round(transform.m11() * 1000)
                / 1000,  # Round scale factors to 3 decimals
                transform.m12(),
                transform.m21(),
                round(transform.m22() * 1000) / 1000,
                round(transform.dx()),  # Round translation
                round(transform.dy()),
            ),
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming or Shift+wheel for azimuth rotation."""
        if not self._image_item:
            event.ignore()
            return

        # Skip if image is being updated (prevents conflicts during image load)
        if self._updating_image:
            event.accept()
            return

        # Skip if already processing a wheel event (prevents event queue buildup)
        if self._processing_wheel_event:
            event.accept()
            return

        try:
            self._processing_wheel_event = True

            delta = event.angleDelta().y()
            if delta == 0:
                delta = event.pixelDelta().y()
            if delta == 0:
                event.ignore()
                return

            # Shift + wheel → rotate azimuth (for radar coverage)
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Grab focus so we receive keyReleaseEvent for Shift
                if not self.hasFocus():
                    self.setFocus(Qt.FocusReason.OtherFocusReason)
                step = 5.0 if delta > 0 else -5.0
                self.shift_wheel_rotated.emit(step)
                event.accept()
                return

            zoom_in = delta > 0
            factor = self._zoom_factor if zoom_in else (1.0 / self._zoom_factor)

            # Apply zoom (anchor is under mouse, no manual centering needed)
            self._zoom(factor)
            event.accept()

        finally:
            self._processing_wheel_event = False

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Emit shift_key_released when Shift is released (not auto-repeat)."""
        if (
            event.key() in (Qt.Key.Key_Shift, Qt.Key.Key_Shift)
            and not event.isAutoRepeat()
        ):
            self.shift_key_released.emit()
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press for panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif (
            event.button() == Qt.MouseButton.RightButton
            and self._image_item is not None
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._image_item.contains(scene_pos):
                self.map_right_clicked.emit(scene_pos.x(), scene_pos.y())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track mouse movement over the map."""
        super().mouseMoveEvent(event)

        if self._image_item is None:
            self.mouse_moved_on_map.emit(None, None)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if self._image_item.contains(scene_pos):
            self.mouse_moved_on_map.emit(scene_pos.x(), scene_pos.y())
        else:
            self.mouse_moved_on_map.emit(None, None)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize to update fit-to-window scale and overlay."""
        super().resizeEvent(event)
        if self._loading_overlay.isVisible():
            self._loading_overlay.setGeometry(self.viewport().rect())
            self._layout_loading_overlay()
        if self._image_item:
            # Recalculate fit-to-window scale when widget is resized
            current_scale = self.transform().m11()
            self.resetTransform()
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fit_to_window_scale = self.transform().m11()
            self._min_zoom = self._fit_to_window_scale
            # Update max zoom relative to new fit-to-window scale
            self._max_zoom = self._fit_to_window_scale * self._max_zoom_multiplier
            # Restore the previous scale if it was larger than fit-to-window
            if current_scale > self._fit_to_window_scale:
                factor = current_scale / self._fit_to_window_scale
                self.scale(factor, factor)

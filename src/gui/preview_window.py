"""Optimized preview window using QGraphicsView for better performance."""

from __future__ import annotations

import contextlib
import logging
import time

from PySide6.QtCore import QEventLoop, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QMouseEvent,
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
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QWidget,
)

from gui.matrix_rain import MatrixRainWidget

from shared.constants import (
    CONTROL_POINT_SIZE_M,
    LOADING_FADE_IN_MS,
    PREVIEW_MIN_LINE_LENGTH_FOR_LABEL,
    PREVIEW_ROTATION_ANGLE,
    PREVIEW_UPRIGHT_TEXT_ANGLE_LIMIT,
)

from PIL import Image

import math

logger = logging.getLogger(__name__)


class OptimizedImageView(QGraphicsView):
    """High-performance image view using QGraphicsView for zoom and pan operations."""

    mouse_moved_on_map = Signal(object, object)  # (x, y) or (None, None)
    map_right_clicked = Signal(float, float)  # (x, y)
    shift_wheel_rotated = Signal(float)  # delta_degrees (positive = CW)
    shift_key_released = Signal()  # Shift released after rotation
    point_drag_started = Signal()  # emitted when user grabs a draggable point
    point_drag_finished = Signal(str, float, float)  # (point_id, scene_x, scene_y)
    fade_in_finished = Signal()  # emitted when map fully revealed after loading

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

        # Draggable points for link profile A/B markers
        self._draggable_points: dict[str, tuple[float, float]] = {}  # id → (x, y) scene
        self._dragging_point_id: str | None = None
        self._drag_hit_radius = 20  # pixels on screen

        # Drag visual feedback (crosshair + rubber band line)
        self._drag_marker_items: list[QGraphicsLineItem] = []
        self._drag_line_item: QGraphicsLineItem | None = None
        self._drag_other_point: tuple[float, float] | None = None

        # Semi-transparent mask over stale inset during recompute
        self._inset_mask_item: QGraphicsRectItem | None = None

        # Pre-rendered QPixmap of clean base (no overlay) for instant swap on drag
        self._clean_base_pixmap: QPixmap | None = None

        # Loading overlay (Matrix rain, parented to viewport)
        self._loading_overlay = MatrixRainWidget(self.viewport())
        self._loading_overlay.hide()
        # Fade-in: black QGraphicsRectItem in the scene with decreasing opacity
        self._fade_rect: QGraphicsRectItem | None = None
        self._fade_in_opacity = 0.0
        self._fade_in_step = 0.0
        self._fade_in_timer = QTimer(self)
        self._fade_in_timer.setInterval(40)  # ~25 fps
        self._fade_in_timer.timeout.connect(self._fade_in_tick)

    def set_image(self, pil_image: Image.Image, meters_per_px: float = 0.0) -> None:
        """
        Set the image to display with fixed rotation to improve thin line visibility.

        Keeps current zoom/center if an image is already displayed.
        """
        t0 = time.monotonic()
        was_loading = self._loading_overlay.isVisible()
        if was_loading:
            self._fade_out_loading_sync()
        else:
            self.stop_loading()
        t_fade_out = time.monotonic()
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

            # Convert PIL image to QPixmap (callers guarantee RGB)
            if pil_image.mode != 'RGB':
                logger.warning('set_image: unexpected mode %s, converting to RGB', pil_image.mode)
                pil_image = pil_image.convert('RGB')

            width, height = pil_image.size
            t1 = time.monotonic()
            image_data = pil_image.tobytes()
            t2 = time.monotonic()
            # Keep a reference to the backing bytes to prevent premature GC
            self._qimage_bytes = image_data
            qimage = QImage(
                self._qimage_bytes,
                width,
                height,
                width * 3,
                QImage.Format.Format_RGB888,
            )
            t3 = time.monotonic()
            qpixmap = QPixmap.fromImage(qimage)
            t4 = time.monotonic()

            # Clear scene and add image (invalidates all scene item references)
            self._fade_in_timer.stop()
            self._scene.clear()
            self._fade_rect = None
            self._cp_cross_items = []
            self._cp_line_item = None
            self._cp_label_item = None
            self._azimuth_line_item = None
            self._azimuth_label_item = None
            self._sector_line_items = []
            self._drag_marker_items = []
            self._drag_line_item = None
            self._drag_other_point = None
            self._inset_mask_item = None
            self._image_item = self._scene.addPixmap(qpixmap)
            t5 = time.monotonic()

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
                self.setTransform(current_transform)
                if current_center is not None:
                    self.centerOn(current_center)
            else:
                self.fit_to_window()

            # Smooth fade-in when replacing the loading screen
            if was_loading:
                self._start_fade_in()

            logger.info(
                'set_image %dx%d: fade_out=%.3fs  tobytes=%.3fs  '
                'QImage=%.3fs  QPixmap=%.3fs  scene=%.3fs  TOTAL=%.3fs',
                width, height,
                t_fade_out - t0, t2 - t1, t3 - t2, t4 - t3,
                t5 - t4, time.monotonic() - t0,
            )

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
        self._draggable_points = {}
        self._dragging_point_id = None
        self._drag_marker_items = []
        self._drag_line_item = None
        self._drag_other_point = None
        self._inset_mask_item = None
        self._clean_base_pixmap = None
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
    # Draggable points (link profile A/B markers)
    # ------------------------------------------------------------------

    def set_draggable_points(self, points: dict[str, tuple[float, float]]) -> None:
        """Set draggable point positions (scene/image coordinates)."""
        self._draggable_points = dict(points)

    def _hit_test_draggable(self, event: QMouseEvent) -> str | None:
        """Return point_id if cursor is near a draggable point, else None."""
        if not self._draggable_points or self._image_item is None:
            return None
        view_pos = event.position().toPoint()
        for pid, (sx, sy) in self._draggable_points.items():
            # Convert scene point to view (screen) coordinates for distance check
            view_pt = self.mapFromScene(sx, sy)
            dx = view_pos.x() - view_pt.x()
            dy = view_pos.y() - view_pt.y()
            if dx * dx + dy * dy <= self._drag_hit_radius ** 2:
                return pid
        return None

    # ------------------------------------------------------------------
    # Drag visual feedback (crosshair + rubber band line)
    # ------------------------------------------------------------------

    def set_clean_base_pixmap(
        self,
        pil_image: Image.Image,
        full_image: Image.Image | None = None,
    ) -> None:
        """Pre-render and cache QPixmap of the clean base map for instant swap on drag.

        Args:
            pil_image: Clean base (map only, no overlay).
            full_image: Full composite (map + inset). Inset area is blended with grey.
        """
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        w, h = pil_image.size
        if full_image is not None and full_image.height > h:
            # Blend inset portion with grey for semi-transparent mask effect
            inset_region = full_image.crop((0, h, w, full_image.height)).convert('RGB')
            grey = Image.new('RGB', inset_region.size, (128, 128, 128))
            dimmed = Image.blend(inset_region, grey, alpha=0.55)
            extended = Image.new('RGB', (w, full_image.height), (128, 128, 128))
            extended.paste(pil_image, (0, 0))
            extended.paste(dimmed, (0, h))
            pil_image = extended
            h = full_image.height
        data = pil_image.tobytes()
        qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._clean_base_pixmap = QPixmap.fromImage(qimg)

    def _start_drag_feedback(self, point_id: str) -> None:
        """Swap to clean base pixmap and remember the other point for rubber band."""
        # Instantly remove yellow line + inset by swapping to clean base
        if self._clean_base_pixmap is not None and self._image_item is not None:
            self._image_item.setPixmap(self._clean_base_pixmap)

        other_id = 'B' if point_id == 'A' else 'A'
        other = self._draggable_points.get(other_id)
        self._drag_other_point = other

    def _update_drag_feedback(self, sx: float, sy: float) -> None:
        """Draw crosshair at (sx, sy) and a yellow dashed line to the other point."""
        self._clear_drag_feedback()

        # Crosshair color: blue for A, red for B
        if self._dragging_point_id == 'A':
            color = QColor(0, 0, 255)
        else:
            color = QColor(255, 0, 0)

        pen = QPen(color)
        pen.setWidth(2)
        pen.setCosmetic(True)

        arm = 12  # pixels on screen → convert to scene via current scale
        scale = self.transform().m11()
        arm_scene = arm / scale if scale > 0 else arm

        line1 = self._scene.addLine(sx - arm_scene, sy, sx + arm_scene, sy, pen)
        line2 = self._scene.addLine(sx, sy - arm_scene, sx, sy + arm_scene, pen)
        line1.setZValue(30)
        line2.setZValue(30)
        self._drag_marker_items = [line1, line2]

        # Rubber band yellow dashed line to the other point
        if self._drag_other_point is not None:
            ox, oy = self._drag_other_point
            line_pen = QPen(QColor(255, 255, 0))
            line_pen.setWidth(2)
            line_pen.setStyle(Qt.PenStyle.DashLine)
            line_pen.setCosmetic(True)
            self._drag_line_item = self._scene.addLine(sx, sy, ox, oy, line_pen)
            self._drag_line_item.setZValue(29)

    def show_inset_mask(self, y_top: float) -> None:
        """Show a semi-transparent grey rectangle over the inset area (below y_top)."""
        self._hide_inset_mask()
        if self._image_item is None:
            return
        img_rect = self._image_item.pixmap().rect()
        if y_top >= img_rect.height():
            return
        rect = QRectF(0, y_top, img_rect.width(), img_rect.height() - y_top)
        brush = QBrush(QColor(128, 128, 128, 140))
        self._inset_mask_item = self._scene.addRect(rect, QPen(Qt.PenStyle.NoPen), brush)
        self._inset_mask_item.setZValue(20)

    def _hide_inset_mask(self) -> None:
        """Remove the inset mask from the scene."""
        if self._inset_mask_item is not None:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._inset_mask_item)
            self._inset_mask_item = None

    def _clear_drag_feedback(self) -> None:
        """Remove drag feedback items from the scene."""
        for item in self._drag_marker_items:
            with contextlib.suppress(Exception):
                self._scene.removeItem(item)
        self._drag_marker_items.clear()

        if self._drag_line_item is not None:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._drag_line_item)
            self._drag_line_item = None

    # ------------------------------------------------------------------
    # Loading overlay & fade transitions
    # ------------------------------------------------------------------

    def start_loading(self) -> None:
        """Show the loading overlay with Matrix rain animation."""
        self._stop_fade_in()
        self._loading_overlay.setGeometry(self.viewport().rect())
        self._loading_overlay.start()
        self._loading_overlay.show()
        self._loading_overlay.raise_()

    def stop_loading(self) -> None:
        """Hide the loading overlay."""
        self._loading_overlay.stop()
        self._loading_overlay.hide()

    def _fade_out_loading_sync(self) -> None:
        """Fade-out the Matrix rain and block until fully black.

        Runs a local event loop so the animation keeps playing while we wait.
        """
        loop = QEventLoop()
        self._loading_overlay.faded_out.connect(loop.quit)
        self._loading_overlay.fade_out()
        loop.exec()
        self._loading_overlay.stop()
        self._loading_overlay.hide()

    def _start_fade_in(self, duration_ms: int = LOADING_FADE_IN_MS) -> None:
        """Add a black rect on top of the scene that dissolves to reveal the map."""
        self._remove_fade_rect()
        scene_rect = self._scene.sceneRect()
        self._fade_rect = self._scene.addRect(
            scene_rect, QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0)),
        )
        self._fade_rect.setZValue(10000)
        self._fade_rect.setOpacity(1.0)
        self._fade_in_opacity = 1.0
        ticks = max(1, duration_ms // 40)
        self._fade_in_step = 1.0 / ticks
        self._fade_in_timer.start()

    def _stop_fade_in(self) -> None:
        self._fade_in_timer.stop()
        self._remove_fade_rect()
        self.fade_in_finished.emit()

    def is_fading_in(self) -> bool:
        """Return True if the fade-in animation is currently running."""
        return self._fade_in_timer.isActive()

    def _remove_fade_rect(self) -> None:
        if self._fade_rect is not None:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._fade_rect)
            self._fade_rect = None

    def _fade_in_tick(self) -> None:
        self._fade_in_opacity -= self._fade_in_step
        if self._fade_in_opacity <= 0:
            self._stop_fade_in()
            return
        if self._fade_rect is not None:
            self._fade_rect.setOpacity(self._fade_in_opacity)

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
        """Handle mouse press for panning or dragging a link profile point."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if we hit a draggable point first
            hit = self._hit_test_draggable(event)
            if hit is not None:
                self._dragging_point_id = hit
                self.point_drag_started.emit()
                self._start_drag_feedback(hit)
                self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                event.accept()
                return
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
        """Handle mouse release — finish drag or end panning."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_point_id is not None:
                # Keep drag feedback visible until recompute finishes and set_image() clears the scene
                scene_pos = self.mapToScene(event.position().toPoint())
                self.point_drag_finished.emit(
                    self._dragging_point_id, scene_pos.x(), scene_pos.y()
                )
                self._dragging_point_id = None
                self._drag_other_point = None
                self.viewport().unsetCursor()
                event.accept()
                return
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Track mouse movement over the map, update cursor for draggable points."""
        if self._dragging_point_id is not None:
            # During drag — keep closed hand cursor, draw feedback, emit coordinates
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            scene_pos = self.mapToScene(event.position().toPoint())
            self._update_drag_feedback(scene_pos.x(), scene_pos.y())
            self.mouse_moved_on_map.emit(scene_pos.x(), scene_pos.y())
            event.accept()
            return

        super().mouseMoveEvent(event)

        if self._image_item is None:
            self.mouse_moved_on_map.emit(None, None)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if self._image_item.contains(scene_pos):
            self.mouse_moved_on_map.emit(scene_pos.x(), scene_pos.y())
            # Show open hand cursor when hovering over a draggable point
            if self._hit_test_draggable(event) is not None:
                self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self.viewport().unsetCursor()
        else:
            self.mouse_moved_on_map.emit(None, None)
            self.viewport().unsetCursor()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize to update fit-to-window scale and overlay."""
        super().resizeEvent(event)
        if self._loading_overlay.isVisible():
            self._loading_overlay.setGeometry(self.viewport().rect())
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

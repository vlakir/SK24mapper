"""Optimized preview window using QGraphicsView for better performance."""

from __future__ import annotations

import contextlib
import logging
import math
import time

from PIL import Image
from PySide6.QtCore import QEventLoop, QRectF, Qt, QTimer, Signal
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
    QGraphicsEllipseItem,
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

        # Draggable points (link profile A/B, control point CP, etc.)
        self._draggable_points: dict[str, tuple[float, float]] = {}  # id → (x, y) scene
        self._drag_colors: dict[str, QColor] = {}  # id → crosshair colour
        self._drag_anchors: dict[
            str, tuple[float, float]
        ] = {}  # id → rubber band anchor
        self._dragging_point_id: str | None = None
        self._drag_hit_radius = 20  # pixels on screen

        # Drag visual feedback (crosshair + rubber band line)
        self._drag_marker_items: list[QGraphicsLineItem] = []
        self._drag_line_item: QGraphicsLineItem | None = None
        self._drag_other_point: tuple[float, float] | None = None
        # Hover highlight (ring around hovered draggable point)
        self._hover_highlight_item: QGraphicsEllipseItem | None = None
        self._hover_point_id: str | None = None

        # Semi-transparent mask over stale inset during recompute
        self._inset_mask_item: QGraphicsRectItem | None = None
        # Upper Y limit for draggable area (inset/chart boundary); None = no limit
        self._drag_y_limit: float | None = None

        # Pre-rendered QPixmap of clean base (no overlay) for instant swap on drag
        self._clean_base_pixmap: QPixmap | None = None

        # Loading overlay (Matrix rain, parented to viewport)
        self._loading_overlay = MatrixRainWidget(self.viewport())
        self._loading_overlay.hide()
        self._fade_out_in_progress = False  # guard against stop_loading during fade
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
                logger.warning(
                    'set_image: unexpected mode %s, converting to RGB', pil_image.mode
                )
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

            # Release backing memory — QPixmap owns a deep copy of pixel data
            del qimage
            self._qimage_bytes = None
            self._original_image = None

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
            self._image_item.setFlag(
                QGraphicsPixmapItem.GraphicsItemFlag.ItemClipsChildrenToShape
            )
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
                width,
                height,
                t_fade_out - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
                time.monotonic() - t0,
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
        self._drag_colors = {}
        self._drag_anchors = {}
        self._dragging_point_id = None
        self._drag_marker_items = []
        self._drag_line_item = None
        self._drag_other_point = None
        self._inset_mask_item = None
        self._drag_y_limit = None
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

        # Parent lines to _image_item so they are clipped to the map boundary
        parent = self._image_item

        self._azimuth_line_item = QGraphicsLineItem(cx, cy, ex, ey, parent)
        self._azimuth_line_item.setPen(pen)
        self._azimuth_line_item.setZValue(20)

        # Label with geographic azimuth angle (not compensated) — placed at
        # ~15% of the line from center so it stays visible.
        label_frac = min(0.30, 400.0 / length_px) if length_px > 0 else 0.30
        label_x = cx + length_px * label_frac * math.sin(az_rad)
        label_y = cy - length_px * label_frac * math.cos(az_rad)

        label_text = f'{azimuth_deg:.0f}°'
        self._azimuth_label_item = QGraphicsTextItem(label_text, parent)
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
                item = QGraphicsLineItem(cx, cy, sx, sy, parent)
                item.setPen(sector_pen)
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
    # Draggable points (link profile A/B, control point CP, etc.)
    # ------------------------------------------------------------------

    def set_draggable_points(
        self,
        points: dict[str, tuple[float, float]],
        *,
        colors: dict[str, tuple[int, int, int]] | None = None,
        anchors: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """
        Replace all draggable points with optional per-point config.

        Args:
            points: id → (x, y) scene coordinates.
            colors: id → (r, g, b) crosshair colour during drag.
            anchors: id → (x, y) endpoint for rubber band line.

        """
        self._draggable_points = dict(points)
        self._drag_colors = {k: QColor(*v) for k, v in colors.items()} if colors else {}
        self._drag_anchors = dict(anchors) if anchors else {}
        if points:
            logger.info(
                'DRAG-DEBUG set_draggable_points: %s',
                {k: (f'{v[0]:.1f}', f'{v[1]:.1f}') for k, v in points.items()},
            )
        else:
            logger.info('DRAG-DEBUG set_draggable_points: {} (cleared)')

    def merge_draggable_points(
        self,
        points: dict[str, tuple[float, float]],
        *,
        colors: dict[str, tuple[int, int, int]] | None = None,
        anchors: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """
        Add/update draggable points without removing existing ones.

        Args:
            points: id → (x, y) scene coordinates to add/update.
            colors: id → (r, g, b) crosshair colour during drag.
            anchors: id → (x, y) endpoint for rubber band line.

        """
        self._draggable_points.update(points)
        if colors:
            self._drag_colors.update({k: QColor(*v) for k, v in colors.items()})
        if anchors:
            self._drag_anchors.update(anchors)
        if points:
            logger.info(
                'DRAG-DEBUG merge_draggable_points: added %s, total: %s',
                list(points.keys()),
                list(self._draggable_points.keys()),
            )

    def remove_draggable_point(self, point_id: str) -> None:
        """Remove a single draggable point by id (no-op if absent)."""
        removed = self._draggable_points.pop(point_id, None)
        self._drag_colors.pop(point_id, None)
        self._drag_anchors.pop(point_id, None)
        if removed is not None:
            logger.info(
                'DRAG-DEBUG remove_draggable_point: %s, remaining: %s',
                point_id,
                list(self._draggable_points.keys()),
            )

    def _clamp_to_scene(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Clamp scene coordinates to the draggable area bounds.

        Respects both the scene rect and the optional inset boundary
        (``_drag_y_limit``) so that points cannot be dragged into the
        profile chart area.
        """
        rect = self._scene.sceneRect()
        y_bottom = (
            self._drag_y_limit if self._drag_y_limit is not None else rect.bottom()
        )
        cx = max(rect.left(), min(sx, rect.right()))
        cy = max(rect.top(), min(sy, y_bottom))
        return cx, cy

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
            dist_sq = dx * dx + dy * dy
            if dist_sq <= self._drag_hit_radius**2:
                logger.info(
                    'DRAG-DEBUG hit_test HIT %s: dist=%.1f px '
                    '(view=%d,%d  point_view=%d,%d  scene=%.1f,%.1f)',
                    pid,
                    dist_sq**0.5,
                    view_pos.x(),
                    view_pos.y(),
                    view_pt.x(),
                    view_pt.y(),
                    sx,
                    sy,
                )
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
        """
        Pre-render and cache QPixmap of the clean base map for instant swap on drag.

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
        """Prepare visual state for dragging: swap pixmap or clear markers."""
        if self._clean_base_pixmap is not None and self._image_item is not None:
            # Link profile: swap to clean base (removes overlay/inset)
            self._image_item.setPixmap(self._clean_base_pixmap)
        else:
            # Generic point (e.g. CP): just clear scene markers
            self.clear_control_point_markers()

        # Anchor for rubber band line — from per-point config
        anchor = self._drag_anchors.get(point_id)
        self._drag_other_point = anchor

    def _update_drag_feedback(self, sx: float, sy: float) -> None:
        """Draw crosshair at (sx, sy) and a dashed line to the anchor point."""
        self.clear_drag_feedback()

        # Crosshair colour from per-point config; default red
        color = self._drag_colors.get(self._dragging_point_id or '', QColor(255, 0, 0))

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

        # Rubber band dashed line to anchor (yellow for A/B)
        if self._drag_other_point is not None:
            ox, oy = self._drag_other_point
            # A/B link profile → yellow; other points → crosshair colour
            pid = self._dragging_point_id or ''
            line_color = QColor(255, 255, 0) if pid in ('A', 'B') else color
            line_pen = QPen(line_color)
            line_pen.setWidth(2)
            line_pen.setStyle(Qt.PenStyle.DashLine)
            line_pen.setCosmetic(True)
            self._drag_line_item = self._scene.addLine(sx, sy, ox, oy, line_pen)
            self._drag_line_item.setZValue(29)

    def set_drag_y_limit(self, y_limit: float | None) -> None:
        """Set the maximum Y coordinate for dragging (inset boundary)."""
        self._drag_y_limit = y_limit

    def show_inset_mask(self, y_top: float) -> None:
        """Show a semi-transparent grey rectangle over the inset area (below y_top)."""
        self._hide_inset_mask()
        self._drag_y_limit = y_top
        if self._image_item is None:
            return
        img_rect = self._image_item.pixmap().rect()
        if y_top >= img_rect.height():
            return
        rect = QRectF(0, y_top, img_rect.width(), img_rect.height() - y_top)
        brush = QBrush(QColor(128, 128, 128, 140))
        self._inset_mask_item = self._scene.addRect(
            rect, QPen(Qt.PenStyle.NoPen), brush
        )
        self._inset_mask_item.setZValue(20)

    def _hide_inset_mask(self) -> None:
        """Remove the inset mask from the scene."""
        if self._inset_mask_item is not None:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._inset_mask_item)
            self._inset_mask_item = None
            self._drag_y_limit = None

    def clear_drag_feedback(self) -> None:
        """Remove drag feedback items (crosshair + rubber band) from the scene."""
        for item in self._drag_marker_items:
            with contextlib.suppress(Exception):
                self._scene.removeItem(item)
        self._drag_marker_items.clear()

        if self._drag_line_item is not None:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._drag_line_item)
            self._drag_line_item = None

    # ------------------------------------------------------------------
    # Hover highlight for draggable points
    # ------------------------------------------------------------------

    def _show_hover_highlight(self, point_id: str) -> None:
        """Draw a pulsing ring around the hovered draggable point."""
        if point_id == self._hover_point_id:
            return  # already showing
        self._hide_hover_highlight()
        pos = self._draggable_points.get(point_id)
        if pos is None:
            return
        sx, sy = pos
        color = self._drag_colors.get(point_id, QColor(255, 0, 0))

        # Ring radius in screen pixels, converted to scene coords
        radius_px = 16
        scale = self.transform().m11()
        r = radius_px / scale if scale > 0 else radius_px

        ring_color = QColor(color)
        ring_color.setAlpha(160)
        pen = QPen(ring_color)
        pen.setWidthF(2.5 / scale if scale > 0 else 2.5)

        fill = QColor(color)
        fill.setAlpha(40)

        ellipse = self._scene.addEllipse(
            sx - r, sy - r, 2 * r, 2 * r, pen, QBrush(fill)
        )
        ellipse.setZValue(25)
        self._hover_highlight_item = ellipse
        self._hover_point_id = point_id

    def _hide_hover_highlight(self) -> None:
        """Remove the hover highlight ring from the scene."""
        if self._hover_highlight_item is not None:
            with contextlib.suppress(Exception):
                self._scene.removeItem(self._hover_highlight_item)
            self._hover_highlight_item = None
            self._hover_point_id = None

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
        if self._fade_out_in_progress:
            return  # _fade_out_loading_sync handles cleanup
        self._loading_overlay.stop()
        self._loading_overlay.hide()

    def _fade_out_loading_sync(self) -> None:
        """
        Fade-out the Matrix rain and block until fully black.

        Runs a local event loop so the animation keeps playing while we wait.
        Uses _fade_out_in_progress guard to prevent stop_loading() from
        killing the animation during the nested event loop.
        """
        self._fade_out_in_progress = True
        try:
            loop = QEventLoop()
            self._loading_overlay.faded_out.connect(loop.quit)
            self._loading_overlay.fade_out()
            loop.exec()
        finally:
            self._fade_out_in_progress = False
        self._loading_overlay.stop()
        self._loading_overlay.hide()

    def _start_fade_in(self, duration_ms: int = LOADING_FADE_IN_MS) -> None:
        """Add a black rect on top of the scene that dissolves to reveal the map."""
        self._remove_fade_rect()
        scene_rect = self._scene.sceneRect()
        self._fade_rect = self._scene.addRect(
            scene_rect,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(0, 0, 0)),
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
                logger.info('DRAG-DEBUG LMB press → start drag %s', hit)
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
                scene_pos = self.mapToScene(event.position().toPoint())
                sx, sy = self._clamp_to_scene(scene_pos.x(), scene_pos.y())
                logger.info(
                    'DRAG-DEBUG LMB release → finish drag %s at scene(%.1f, %.1f)',
                    self._dragging_point_id,
                    sx,
                    sy,
                )
                self.point_drag_finished.emit(self._dragging_point_id, sx, sy)
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
            self._hide_hover_highlight()
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            scene_pos = self.mapToScene(event.position().toPoint())
            sx, sy = self._clamp_to_scene(scene_pos.x(), scene_pos.y())
            self._update_drag_feedback(sx, sy)
            self.mouse_moved_on_map.emit(sx, sy)
            event.accept()
            return

        super().mouseMoveEvent(event)

        if self._image_item is None:
            self.mouse_moved_on_map.emit(None, None)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if self._image_item.contains(scene_pos):
            self.mouse_moved_on_map.emit(scene_pos.x(), scene_pos.y())
            # Show open hand cursor + highlight ring over a draggable point
            hit = self._hit_test_draggable(event)
            if hit is not None:
                self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                self._show_hover_highlight(hit)
            else:
                self.viewport().unsetCursor()
                self._hide_hover_highlight()
        else:
            self.mouse_moved_on_map.emit(None, None)
            self.viewport().unsetCursor()
            self._hide_hover_highlight()

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

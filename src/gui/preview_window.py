"""Optimized preview window using QGraphicsView for better performance."""

from __future__ import annotations

import contextlib
import logging
import math
import time

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QEventLoop, QRectF, Qt, QThread, QTimer, Signal
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
    PREVIEW_MAX_DIM,
    PREVIEW_MIN_LINE_LENGTH_FOR_LABEL,
    PREVIEW_ROTATION_ANGLE,
    PREVIEW_UPRIGHT_TEXT_ANGLE_LIMIT,
)

logger = logging.getLogger(__name__)


class _SetImageWorker(QThread):
    """
    Background QThread that turns a PIL image into QImage(s) ready for
    QPixmap.fromImage on the main thread.

    Two-pass pipeline:
      Stage 1 (fast)  — cv2.resize INTER_AREA → PREVIEW_MAX_DIM along
                        long side. Emit with is_final=False so main can
                        display a small QPixmap immediately and fade out
                        the loading overlay (~30 ms QPixmap on 4k²).
      Stage 2 (refinement) — full-resolution buffer. Emit with
                        is_final=True so main can setPixmap() on the
                        same image_item, swapping the small preview for
                        the crisp full-res view without any loading
                        overlay or visible gap.

    Buffers (small, full) are kept alive on the instance until the
    receiving slot has run QPixmap.fromImage (deep-copies into Qt's
    pixmap cache). After the final swap the caller releases them.

    Each invocation carries a sequence number `seq` so the main thread
    can ignore late stage-2 emits from a worker whose build was
    superseded by a newer set_image() call.
    """

    finished_qimage = Signal(QImage, bool, int)  # (qimage, is_final, seq)

    def __init__(
        self, pil_image: Image.Image, max_dim: int | None,
        seq: int, parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._pil = pil_image
        self._max_dim = max_dim or 0
        self._seq = seq
        self._small_buffer: np.ndarray | None = None
        self._full_buffer: np.ndarray | None = None

    def run(self) -> None:
        arr = np.asarray(self._pil)
        h, w = arr.shape[:2]
        long_side = max(h, w)
        downsample_needed = (
            self._max_dim > 0 and long_side > self._max_dim
        )

        # Stage 1: small preview (or pass-through if image is already
        # under PREVIEW_MAX_DIM).
        if downsample_needed:
            scale = self._max_dim / long_side
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            t0 = time.monotonic()
            small = cv2.resize(
                arr, (new_w, new_h), interpolation=cv2.INTER_AREA,
            )
            logger.info(
                'SetImageWorker stage1 resize %d×%d → %d×%d in %.3fs',
                w, h, new_w, new_h, time.monotonic() - t0,
            )
            self._small_buffer = small
            sh, sw = small.shape[:2]
            qimg_small = QImage(
                small.data, sw, sh, small.strides[0],
                QImage.Format.Format_RGB888,
            )
            self.finished_qimage.emit(qimg_small, False, self._seq)
        else:
            # No downsample needed — there will be only one emit at
            # full size below. Don't emit a stage-1 here.
            pass

        # Stage 2 cancellation point: main thread may have called
        # requestInterruption() after a user input (right-click, drag,
        # …) to skip the expensive full-res copy + setPixmap. Bail out
        # before the np.ascontiguousarray.copy() — saves ~300 ms of
        # memcpy AND prevents the setPixmap-on-16k blocking that would
        # otherwise lag the user's status messages.
        if self.isInterruptionRequested():
            return

        # Stage 2: full resolution. Always emit (in pass-through case
        # this IS the first and only emit).
        t1 = time.monotonic()
        full = np.ascontiguousarray(arr).copy()
        self._full_buffer = full
        qimg_full = QImage(
            full.data, w, h, full.strides[0],
            QImage.Format.Format_RGB888,
        )
        logger.info(
            'SetImageWorker stage2 full %d×%d ready in %.3fs',
            w, h, time.monotonic() - t1,
        )
        self.finished_qimage.emit(qimg_full, True, self._seq)


class OptimizedImageView(QGraphicsView):
    """High-performance image view using QGraphicsView for zoom and pan operations."""

    mouse_moved_on_map = Signal(object, object)  # (x, y) or (None, None)
    map_right_clicked = Signal(float, float)  # (x, y)
    shift_wheel_rotated = Signal(float)  # delta_degrees (positive = CW)
    shift_key_released = Signal()  # Shift released after rotation
    point_drag_started = Signal(
        str
    )  # emitted when user grabs a draggable point (point_id)
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

        # Two-pass set_image: each call increments this seq. The worker
        # captures it on construction; late stage-2 emits from a superseded
        # worker (newer set_image started before old stage-2 arrived) are
        # ignored by comparing against the current seq.
        self._set_image_seq = 0
        self._active_set_image_worker: _SetImageWorker | None = None
        # When the user triggers a new build (right-click on NSU, drag,
        # etc.) we may have a pending Stage-2 setPixmap from the previous
        # build sitting in the event queue — running it would block main
        # thread for ~300-500 ms on 16k² images, lagging the user's input.
        # cancel_pending_full_swap() sets this flag; _on_full_qimage_ready
        # checks it and skips the heavy QPixmap.fromImage + setPixmap.
        self._stage2_swap_cancelled = False

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

        # NSU target point markers (QGraphicsItems for instant visual feedback)
        self._nsu_marker_items: list[QGraphicsLineItem] = []
        # Hiding patches (persists until set_image scene.clear)
        self._nsu_hide_items: list = []

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

        Loading-overlay timing: we DON'T fade out the matrix-rain loading
        overlay before building the new QPixmap. PIL.tobytes + QImage +
        QPixmap.fromImage block the main thread for ~1.5 s on 16k²
        images, and on some setups (Linux X11/Wayland compositing under
        a dark Qt theme) the GUI doesn't actually paint scene changes
        until the main thread returns to the event loop. If we hid the
        overlay first, Qt would queue the hide+repaint but never deliver
        it until set_image returned, so the user kept seeing the
        last-painted frame of the overlay (often nearly-transparent
        matrix-rain over a dark theme background → "black flash").
        Instead, leave the overlay shown — even frozen — across the
        whole pixmap-build window, then fade it out only AFTER the new
        image is on screen.
        """
        t0 = time.monotonic()
        was_loading = self._loading_overlay.isVisible()
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

            # Don't clear the scene yet — keep the OLD pixmap visible while
            # we build the new one (PIL.tobytes + QImage + QPixmap.fromImage
            # cost 0.3–0.5 s on 16k² images). Clearing first leaves the
            # viewport showing Qt's default black background, which is what
            # users saw as a brief black flash on the final preview step.
            #
            # Free the cheap overlay items now (labels, crosses, lines —
            # they don't pin large pixmap memory), then keep building the
            # new pixmap. The old image is removed AFTER the new one is
            # added, so the scene is never empty.
            self._fade_in_timer.stop()
            old_image_item = self._image_item
            self._image_item = None
            self._original_image = None
            self._qimage_bytes = None

            # Only the fade-in rect needs proactive removal here — it's an
            # internal display-only artefact, not user-meaningful overlay.
            # Other overlay items (cp_cross, cp_line, NSU markers, link
            # line, drag markers, sector lines, etc.) MUST be left alone:
            # they're managed by callers in view.py and represent things
            # the user just interacted with (e.g. an NSU cross they just
            # right-clicked into existence and that should morph straight
            # into the triangle on the new map). Clearing them here made
            # them blink out during the build window and only return as
            # triangles after the recompute finished — the bug user
            # reported. Callers do their own cleanup via
            # clear_control_point_markers / clear_nsu_markers /
            # clear_drag_feedback / set_draggable_points({}) when they
            # actually need to drop stale overlay state (e.g. map-type
            # change).
            with contextlib.suppress(Exception):
                if self._fade_rect is not None:
                    self._scene.removeItem(self._fade_rect)
            self._fade_rect = None

            # Drop the stale-preview inset mask if it was painted before
            # this build. _dim_stale_preview() adds a semi-transparent
            # grey QGraphicsRectItem поверх inset-зоны при нажатии
            # «Скачать», чтобы пользователь видел что текущий inset уже
            # неактуален во время build. Без явного hide здесь mask
            # оставался в scene и накладывался на свежий composite —
            # inset выглядел «слегка затемнённым и далее всё время
            # остаётся таким» (Vladimir, LINK_PROFILE second build).
            self._hide_inset_mask()

            # Off-main two-pass pipeline:
            #
            #  Stage 1 (synchronous wait via QEventLoop) — worker builds
            #  a downsampled preview at PREVIEW_MAX_DIM. We block here
            #  (Qt event loop active → loading anim keeps playing) until
            #  Stage 1 arrives. QPixmap.fromImage on the 4k² result is
            #  ~30 ms, so the user sees a usable map almost immediately.
            #
            #  Stage 2 (asynchronous swap) — same worker continues in
            #  the background and produces the full-resolution QImage.
            #  The receiving slot _on_full_qimage_ready calls
            #  self._image_item.setPixmap(full), which atomically
            #  swaps the displayed pixmap for the crisp version. No
            #  loading overlay, no flash; the visual is a smooth
            #  refinement.
            #
            # The worker carries a sequence number; late Stage-2 emits
            # from a superseded build (newer set_image already running)
            # are ignored.
            self._set_image_seq += 1
            seq = self._set_image_seq
            self._stage2_swap_cancelled = False
            t1 = time.monotonic()
            qimage_holder: list[QImage] = []
            loop = QEventLoop()
            worker = _SetImageWorker(
                pil_image, PREVIEW_MAX_DIM, seq=seq, parent=self,
            )

            def _on_qimage(qimg: QImage, is_final: bool, emit_seq: int) -> None:
                # First emit always feeds the synchronous wait below —
                # whether it's Stage 1 (small) or, for already-small
                # source images, the single is_final emit. Any later
                # emit (the Stage 2 swap) routes through the async
                # slot.
                if not qimage_holder:
                    qimage_holder.append(qimg)
                    loop.quit()
                else:
                    self._on_full_qimage_ready(qimg, emit_seq)

            worker.finished_qimage.connect(_on_qimage)
            self._active_set_image_worker = worker
            worker.start()
            # Watchdog timeout: если _SetImageWorker не emit'нет
            # finished_qimage за 5 секунд (worker дёшев, реальное время
            # на 8404² ~50 ms), главный event loop возвращается без
            # qimage'а — и мы graceful'но возвращаемся вместо вечного
            # ожидания. Vladimir поймал такое зависание в режиме НСУ:
            # после нескольких добавлений точек handler упёрся в
            # loop.exec(), индикатор статус-бара застрял, UI был
            # blocked 3 минуты пока не закрыл окно.
            QTimer.singleShot(5000, loop.quit)
            loop.exec()
            if not qimage_holder:
                # Worker не emit'нул сигнал в окно 5s (или приложение
                # закрывается — main event loop остановился раньше).
                # Логируем и тихо выходим: caller'ы не critical-path,
                # GUI всё ещё функционален.
                logger.warning(
                    'set_image: watchdog timeout / nested loop exit '
                    '(_SetImageWorker did not emit finished_qimage)'
                )
                return
            qimage = qimage_holder[0]
            t3 = time.monotonic()
            qpixmap = QPixmap.fromImage(qimage)
            t4 = time.monotonic()

            # Release Stage-1 small buffer — QPixmap.fromImage deep-
            # copies pixel data into Qt's pixmap cache, so the numpy
            # buffer that backed qimage is no longer needed.
            del qimage
            worker._small_buffer = None  # noqa: SLF001
            self._qimage_bytes = None
            self._original_image = None
            t2 = t3  # legacy log slot — tobytes is gone

            # Add new pixmap first, THEN remove old. Z-order: addPixmap
            # appends to scene → new item drawn on top of old one. Between
            # addPixmap and removeItem the scene is never empty, so there
            # is no black flash. Peak memory during this overlap is ~1×
            # image size higher than `scene.clear()` first; for 16k² that's
            # +864 MB transient on the GUI process.
            self._image_item = self._scene.addPixmap(qpixmap)
            self._image_item.setFlag(
                QGraphicsPixmapItem.GraphicsItemFlag.ItemClipsChildrenToShape
            )
            if old_image_item is not None:
                with contextlib.suppress(Exception):
                    self._scene.removeItem(old_image_item)
            t5 = time.monotonic()

            # CRITICAL for click-to-coord correctness:
            # The Stage-1 pixmap may be downsampled (e.g. 4 k from a 16 k
            # source), but scene coordinates MUST match the full-resolution
            # source pixel space — callers convert scene-clicks to
            # image-pixels via the metadata's full size, not via
            # pixmap.width(). Scale the item up so its bounding box in
            # scene units equals the FULL source size (= `width` ×
            # `height` here). Stage 2 will setPixmap(full) + setScale(1.0)
            # so the bounding box stays the same: click coordinates are
            # stable across the swap. Source resolution is preserved in
            # `width, height` from pil_image.size at function entry.
            pixmap_w = qpixmap.width()
            stage1_scale = (
                width / pixmap_w if pixmap_w > 0 and pixmap_w != width
                else 1.0
            )
            transform = QTransform()
            if PREVIEW_ROTATION_ANGLE != 0:
                transform.rotate(PREVIEW_ROTATION_ANGLE)
            if stage1_scale != 1.0:
                transform.scale(stage1_scale, stage1_scale)
            self._image_item.setTransform(transform)

            # Scene rect is the displayed extent of image_item in scene
            # coordinates — i.e. full source size in both stages.
            if PREVIEW_ROTATION_ANGLE != 0:
                rotated_rect = transform.mapRect(qpixmap.rect())
                self._scene.setSceneRect(rotated_rect)
            else:
                self._scene.setSceneRect(QRectF(0, 0, width, height))

            # Fit or restore transform
            if preserve_transform and current_transform is not None:
                self.setTransform(current_transform)
                if current_center is not None:
                    self.centerOn(current_center)
            else:
                self.fit_to_window()

            # NOW that the new pixmap is on screen, fade out the loading
            # overlay. The new image is already rendered behind it (Qt
            # will composite in the next paint cycle once we yield),
            # so the fade reveals the finished map directly — no black
            # gap between overlay hide and map show.
            t_fade_out_start = time.monotonic()
            if was_loading:
                self._fade_out_loading_sync()
            else:
                self.stop_loading()
            t_fade_out = time.monotonic() - t_fade_out_start

            # Smooth fade-in when replacing the loading screen
            if was_loading:
                self._start_fade_in()

            logger.info(
                'set_image %dx%d: tobytes=%.3fs  '
                'QImage=%.3fs  QPixmap=%.3fs  scene=%.3fs  '
                'fade_out=%.3fs  TOTAL=%.3fs',
                width,
                height,
                t2 - t1,
                t3 - t2,
                t4 - t3,
                t5 - t4,
                t_fade_out,
                time.monotonic() - t0,
            )

        finally:
            self._updating_image = False

    def cancel_pending_full_swap(self) -> None:
        """
        Skip any pending Stage-2 full-res setPixmap from a previous
        build. Call this from user-input handlers that are about to
        trigger a new build (right-click, drag start, etc.) — the
        Stage-2 swap costs ~300-500 ms of blocking QPixmap.fromImage +
        setPixmap on a 16k² image and would lag the user's response.

        Both layers cooperate:
          - worker.requestInterruption() so the Stage-2 copy in the
            worker thread bails out before doing the 800 MB memcpy
            (if it hasn't started yet);
          - self._stage2_swap_cancelled so if the worker already
            emitted finished_qimage(is_final=True) and the slot is
            sitting in the event queue, the slot returns immediately
            instead of running setPixmap.
        """
        if self._active_set_image_worker is not None:
            with contextlib.suppress(Exception):
                self._active_set_image_worker.requestInterruption()
        self._stage2_swap_cancelled = True

    def _on_full_qimage_ready(self, qimg: QImage, emit_seq: int) -> None:
        """
        Stage-2 slot for _SetImageWorker. Called via QueuedConnection on
        the main thread.

        Critically, this slot DOES NOT run the heavy QPixmap.fromImage +
        setPixmap synchronously. Reason: Qt's event queue is FIFO, and
        the user's right-click MouseRelease event is queued AFTER the
        worker's finished_qimage emit — so by the time the click handler
        gets to call cancel_pending_full_swap(), the slot is already
        mid-setPixmap and the cancel arrives too late. The user feels
        300-500 ms of lag between releasing the mouse and seeing the
        status messages / recompute spinner.

        Defer the heavy work to a QTimer.singleShot(0, ...): that posts
        the do-swap callback to the END of the event queue. Click event
        is now ahead of it, runs first, sets _stage2_swap_cancelled, and
        the deferred swap bails out without doing any heavy work.

        On an idle scene (no queued user input), the deferred swap fires
        within ~one paint cycle — visually indistinguishable from the
        previous sync behaviour.
        """
        # Stale emit: a newer set_image() superseded this build.
        if emit_seq != self._set_image_seq:
            return
        # Already cancelled — release worker buffers without scheduling.
        if self._stage2_swap_cancelled:
            self._release_set_image_worker()
            return
        # Defer the actual swap so that any queued user input (mouse
        # release, key press) processes first and can cancel us.
        QTimer.singleShot(
            0, lambda: self._do_stage2_swap(qimg, emit_seq),
        )

    def _do_stage2_swap(self, qimg: QImage, emit_seq: int) -> None:
        """
        Deferred body of _on_full_qimage_ready. Runs when the event
        queue is otherwise empty (or when its singleShot turn comes up).
        Bails out if the user cancelled in the interim.

        Invariant: the image item's displayed extent in scene units is
        the FULL source size in both stages — Stage 1 sets `setScale(
        full_w/small_w)`; Stage 2 sets `setScale(1.0)`. The bounding box
        stays identical, so scene → image-pixel conversions (right-
        click → SK-42 coord, draggable point positions, etc.) give the
        same answer in both stages.
        """
        if emit_seq != self._set_image_seq:
            self._release_set_image_worker()
            return
        if self._stage2_swap_cancelled:
            logger.info('set_image stage-2 swap cancelled by user input')
            self._release_set_image_worker()
            return
        if self._image_item is None:
            self._release_set_image_worker()
            return
        try:
            old_pixmap_w = self._image_item.pixmap().width()
            t0 = time.monotonic()
            full_pixmap = QPixmap.fromImage(qimg)
            t1 = time.monotonic()
            new_pixmap_w = full_pixmap.width()
            self._image_item.setPixmap(full_pixmap)
            # Stage 2 has the full-res pixmap; revert to identity scale
            # (the bounding box in scene coords is already `width` since
            # Stage 1 scaled the small pixmap up to match).
            t = QTransform()
            if PREVIEW_ROTATION_ANGLE != 0:
                t.rotate(PREVIEW_ROTATION_ANGLE)
            self._image_item.setTransform(t)
            logger.info(
                'set_image stage-2 swap: pixmap %d → %d  '
                'QPixmap=%.3fs  setPixmap=%.3fs',
                old_pixmap_w, new_pixmap_w,
                t1 - t0, time.monotonic() - t1,
            )
        finally:
            self._release_set_image_worker()

    def _release_set_image_worker(self) -> None:
        """Drop the worker reference + free its numpy buffers."""
        worker = self._active_set_image_worker
        if worker is not None:
            worker._small_buffer = None  # noqa: SLF001
            worker._full_buffer = None  # noqa: SLF001
            worker.deleteLater()
            self._active_set_image_worker = None

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

    def _sample_pixmap_luminance(self, x: float, y: float) -> float:
        """Sample luminance from the displayed pixmap at (x, y). Returns 0..255."""
        if self._image_item is None:
            return 128.0
        pixmap = self._image_item.pixmap()
        px_int = round(x)
        py_int = round(y)
        if (
            px_int < 0
            or py_int < 0
            or px_int >= pixmap.width()
            or py_int >= pixmap.height()
        ):
            return 128.0
        img = pixmap.toImage()
        c = img.pixelColor(px_int, py_int)
        return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()

    def set_nsu_markers(
        self, points: list[tuple[float, float, tuple[int, int, int]]]
    ) -> None:
        """
        Draw colored cross markers for NSU target points (instant).

        Args:
            points: list of (x, y, (r, g, b)) in scene coordinates.

        """
        self.clear_nsu_markers()
        if self._image_item is None or self._meters_per_px <= 0:
            return
        ppm = 1.0 / self._meters_per_px
        size_px = max(8, round(CONTROL_POINT_SIZE_M * ppm * 0.7))
        half = size_px / 2.0
        for x, y, color in points:
            # Адаптивная обводка: чёрная на светлом, белая на тёмном
            lum = self._sample_pixmap_luminance(x, y)
            lum_threshold = 128
            outline_color = (
                QColor(0, 0, 0) if lum > lum_threshold else QColor(255, 255, 255)
            )
            outline_pen = QPen(outline_color)
            outline_pen.setWidth(5)
            outline_pen.setCosmetic(True)
            # Толстая обводка крестика
            ol1 = self._scene.addLine(
                x - half, y - half, x + half, y + half, outline_pen
            )
            ol2 = self._scene.addLine(
                x - half, y + half, x + half, y - half, outline_pen
            )
            # Цветная заливка (тоньше, поверх)
            pen = QPen(QColor(*color))
            pen.setWidth(3)
            pen.setCosmetic(True)
            line1 = self._scene.addLine(x - half, y - half, x + half, y + half, pen)
            line2 = self._scene.addLine(x - half, y + half, x + half, y - half, pen)
            self._nsu_marker_items.extend([ol1, ol2, line1, line2])

    def clear_nsu_markers(self) -> None:
        """Remove NSU target point markers from the scene."""
        for item in self._nsu_marker_items:
            with contextlib.suppress(Exception):
                self._scene.removeItem(item)
        self._nsu_marker_items.clear()

    def hide_marker_at(self, x: float, y: float) -> None:
        """
        Cover a marker at (x, y) with a filled circle matching background.

        Instant — uses QGraphicsEllipseItem, no image processing.
        Items are stored in _nsu_marker_items and cleaned with clear_nsu_markers
        or scene.clear().
        """
        if self._image_item is None:
            logger.info('hide_marker_at: no image_item')
            return
        ppm = 1.0 / self._meters_per_px if self._meters_per_px > 0 else 1.0
        # Radius must cover triangle corners (half*√2) + outline margin
        radius = max(12, round(CONTROL_POINT_SIZE_M * ppm * 0.9))
        logger.info(
            'hide_marker_at: x=%.1f y=%.1f ppm=%.4f radius=%d mpp=%.2f',
            x,
            y,
            ppm,
            radius,
            self._meters_per_px,
        )
        # Sample background from several points around the marker, average
        offsets = [
            (0, radius + 4),  # below
            (0, -(radius + 4)),  # above
            (radius + 4, 0),  # right
            (-(radius + 4), 0),  # left
        ]
        r_sum, g_sum, b_sum, n = 0, 0, 0, 0
        for dx, dy in offsets:
            c = self._sample_pixmap_luminance_color(x + dx, y + dy)
            r_sum += c.red()
            g_sum += c.green()
            b_sum += c.blue()
            n += 1
        bg_color = QColor(r_sum // n, g_sum // n, b_sum // n)
        logger.info(
            'hide_marker_at: bg_color=(%d,%d,%d), radius=%d',
            bg_color.red(),
            bg_color.green(),
            bg_color.blue(),
            radius,
        )
        ellipse = self._scene.addEllipse(
            x - radius,
            y - radius,
            radius * 2,
            radius * 2,
            QPen(Qt.PenStyle.NoPen),
            QBrush(bg_color),
        )
        ellipse.setZValue(10)  # ensure above pixmap (z=0)
        self._nsu_hide_items.append(ellipse)
        self.viewport().update()

    def _sample_pixmap_luminance_color(self, x: float, y: float) -> QColor:
        """Sample the actual pixel color from the displayed pixmap."""
        if self._image_item is None:
            return QColor(200, 200, 200)
        pixmap = self._image_item.pixmap()
        ix, iy = int(x), int(y)
        # Clamp to valid range
        ix = max(0, min(ix, pixmap.width() - 1))
        iy = max(0, min(iy, pixmap.height() - 1))
        # Copy a tiny 3x3 area and sample the center
        tiny = pixmap.copy(max(0, ix - 1), max(0, iy - 1), 3, 3)
        img = tiny.toImage()
        return img.pixelColor(min(1, img.width() - 1), min(1, img.height() - 1))

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

    def get_draggable_point(self, point_id: str) -> tuple[float, float] | None:
        """Get the (x, y) scene position of a draggable point, or None."""
        return self._draggable_points.get(point_id)

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

        При MemoryError (тесный GUI VMS под RLIMIT_AS) функция тихо
        логирует ошибку и оставляет существующий `_clean_base_pixmap`
        как есть — drag покажет stale dim inset, но preview не
        упадёт. Это лучше чем рушить весь _show_preview, который
        зовёт нас.

        Args:
            pil_image: Clean base (map only, no overlay).
            full_image: Full composite (map + inset). Inset area is blended with grey.

        """
        try:
            self._set_clean_base_pixmap_impl(pil_image, full_image)
        except MemoryError:
            logger.exception(
                'set_clean_base_pixmap: not enough memory — keeping stale _clean_base_pixmap',
            )

    def _set_clean_base_pixmap_impl(
        self,
        pil_image: Image.Image,
        full_image: Image.Image | None = None,
    ) -> None:
        """Implementation; wrapped by set_clean_base_pixmap for MemoryError safety."""
        # Streamed-chunk path. Pre-my-fix старая PIL-логика делала
        # tobytes на extended 8404×10505 RGB → peak ~530 MB transient
        # → MemoryError на тесном GUI RLIMIT_AS. Первый мой numpy-fix
        # (5a5d7ce) тоже peak'ил ~530 MB т.к. np.asarray(full_image)
        # внутри зовёт PIL.tobytes — рандомно валился, см. mil_mapper
        # .log 23:36:33.
        #
        # Сейчас собираем canvas построчно через PIL.crop стрипами по
        # ~1024 строк: tobytes-peak per-стрипа ~25 MB (= STRIPE_H × W ×
        # 3 × 2 transient в b"".join). Net peak: canvas (265 MB, kept) +
        # stripe (25 MB, transient) = ~290 MB вместо 530 MB.
        #
        # Если даже это не пройдёт по памяти — try/except MemoryError
        # outside: dim-pixmap fall back на stale (drag покажет старый
        # dim inset, но preview не упадёт).
        STRIPE_H = 1024
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        w, h = pil_image.size
        if full_image is not None and full_image.height > h:
            full_h = full_image.height
            canvas = np.empty((full_h, w, 3), dtype=np.uint8)
            # Top: clean map as-is, streamed.
            for y0 in range(0, h, STRIPE_H):
                y1 = min(y0 + STRIPE_H, h)
                stripe = pil_image.crop((0, y0, w, y1))
                canvas[y0:y1] = np.asarray(stripe)
                stripe.close()
            # Bottom: inset blended with mid-grey at alpha=0.55, streamed.
            # blend formula matches PIL.Image.blend(inset, grey=128, 0.55):
            #   out = inset*0.45 + 128*0.55 ≈ (inset*115 + 17_664) >> 8
            need_convert = full_image.mode != 'RGB'
            for y0 in range(h, full_h, STRIPE_H):
                y1 = min(y0 + STRIPE_H, full_h)
                stripe = full_image.crop((0, y0, w, y1))
                if need_convert:
                    rgb_stripe = stripe.convert('RGB')
                    stripe.close()
                    stripe = rgb_stripe
                arr = np.asarray(stripe).astype(np.uint16)
                canvas[y0:y1] = ((arr * 115 + 17_664) >> 8).astype(np.uint8)
                stripe.close()
            h = full_h
            buffer = canvas
        else:
            buffer = np.asarray(pil_image)
        # QImage takes the numpy buffer via Buffer Protocol (zero-copy);
        # QPixmap.fromImage deep-copies into Qt's pixmap cache, so the
        # numpy buffer can be released as soon as we return. bytes() copy
        # was wasteful here — caused full +265 MB transient.
        buffer = np.ascontiguousarray(buffer)
        qimg = QImage(
            buffer.data,
            w, h, w * 3,
            QImage.Format.Format_RGB888,
        )
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
        # Lower the overlay below the viewport so even any stale frame Qt
        # already buffered won't render on top of the scene.
        self._loading_overlay.lower()
        self._loading_overlay.hide()
        # Pump Qt events so the widget-state change from hide() actually
        # propagates to the rendered viewport before the caller's
        # tobytes/QPixmap block (~1.5 s on 16k² images) steals the main
        # thread. A bare viewport().repaint() isn't enough — the loading
        # overlay is a child of the viewport, and hide() only takes
        # visible effect after Qt processes the widget-state event.
        # processEvents flushes both the hide and the resulting paint.
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        # Belt-and-braces: now that the overlay is logically out of the
        # way, force an immediate scene paint so the old image is on the
        # screen before tobytes() takes the thread.
        self.viewport().repaint()

    def _start_fade_in(self, duration_ms: int = LOADING_FADE_IN_MS) -> None:
        """Add a black rect on top of the scene that dissolves to reveal the map."""
        self._remove_fade_rect()
        if duration_ms <= 0:
            # No fade — map is shown immediately, no overlay added at all.
            self.fade_in_finished.emit()
            return
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
                self.point_drag_started.emit(hit)
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

"""Matrix digital rain effect widget — programmatic QPainter implementation."""

from __future__ import annotations

import math
from random import Random

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPaintEvent,
    QResizeEvent,
)
from PySide6.QtWidgets import QWidget

from shared.constants import LOADING_FADE_OUT_MS

# Dedicated RNG instance for visual effects (not for cryptographic purposes)
_rng = Random()

# Latin + Cyrillic + digits
_MATRIX_CHARS = (
    [chr(c) for c in range(0x0041, 0x005B)]  # A-Z
    + [chr(c) for c in range(0x0061, 0x007B)]  # a-z
    + [chr(c) for c in range(0x0410, 0x0430)]  # А-Я
    + [chr(c) for c in range(0x0430, 0x0450)]  # а-я
    + [chr(c) for c in range(0x0030, 0x003A)]  # 0-9
    + list('.,;:!?@#$%&*+-=<>()[]{}/\\|~^')
)

# Timing
_FPS = 25
_INTERVAL_MS = 1000 // _FPS

# Visual tuning
_FONT_SIZE = 28
_COLUMN_SPACING = 1.2  # columns are font_size * this apart
_TRAIL_LENGTH = 18  # chars of fading green trail
_SPAWN_PROBABILITY = 0.04  # chance per column per tick to spawn a new drop
_CHAR_FLICKER_PROB = 0.08  # chance to randomise a trail character each tick

# Y-axis spin: character "rotates" around its vertical axis
_SPIN_PROBABILITY = 0.036  # chance per char per tick to start spinning
_SPIN_SPEED = 0.18  # radians per tick (full flip ≈ 35 ticks ≈ 1.4 s)

# Thresholds
_INITIAL_DROP_PROB = 0.5  # probability to seed initial drops
_EDGE_ON_THRESHOLD = 0.05  # cos(phase) below which char is invisible


class MatrixRainWidget(QWidget):
    """Full-viewport widget that renders the Matrix digital rain effect."""

    faded_out = Signal()  # emitted when fade-out reaches full black

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._bg_color = QColor(0, 0, 0)

        self._font = QFont('Consolas', _FONT_SIZE)
        self._font.setStyleHint(QFont.StyleHint.Monospace)
        self._fm = QFontMetricsF(self._font)

        self._col_w = int(_FONT_SIZE * _COLUMN_SPACING)
        self._row_h = int(_FONT_SIZE * 1.35)
        self._half_col_w = self._col_w / 2.0

        # Drop structure: [row, speed, chars, alive, spins]
        # where spins is list of float|None per char (None=still, float=phase)
        self._drops: list[list] = []
        self._num_cols = 0
        self._num_rows = 0

        # Fade-out state
        self._fade_alpha = 1.0  # master opacity multiplier (1.0 = full, 0.0 = black)
        self._fading_out = False
        self._fade_step = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._fade_alpha = 1.0
        self._fading_out = False
        self._init_grid()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._drops.clear()
        self._fading_out = False

    def fade_out(self, duration_ms: int = LOADING_FADE_OUT_MS) -> None:
        """Begin fading to black. Emits faded_out when fully dark."""
        self._fading_out = True
        ticks = max(1, duration_ms // _INTERVAL_MS)
        self._fade_step = self._fade_alpha / ticks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_grid(self) -> None:
        w, h = self.width(), self.height()
        if w < 1 or h < 1:
            return
        self._num_cols = max(1, w // self._col_w)
        self._num_rows = max(1, h // self._row_h) + _TRAIL_LENGTH
        self._drops = [[] for _ in range(self._num_cols)]
        # Seed some initial drops so it doesn't start blank
        for col_drops in self._drops:
            if _rng.random() < _INITIAL_DROP_PROB:
                col_drops.append(self._make_drop(scatter=True))

    def _make_drop(self, *, scatter: bool = False) -> list:
        """Create: [row, speed, chars, alive, spins]."""
        start_row = -_rng.randint(0, self._num_rows) if scatter else 0.0
        speed = _rng.uniform(0.2, 0.6)
        chars = [_rng.choice(_MATRIX_CHARS) for _ in range(_TRAIL_LENGTH)]
        spins: list[float | None] = [None] * _TRAIL_LENGTH
        return [float(start_row), speed, chars, True, spins]

    def _tick(self) -> None:
        # Fade-out progression
        if self._fading_out:
            self._fade_alpha = max(0.0, self._fade_alpha - self._fade_step)
            if self._fade_alpha <= 0:
                self._fading_out = False
                self._timer.stop()
                self.update()
                self.faded_out.emit()
                return

        for col_drops in self._drops:
            # Spawn new drops (stop spawning during fade-out)
            if not self._fading_out and _rng.random() < _SPAWN_PROBABILITY:
                col_drops.append(self._make_drop())

            alive = []
            for drop in col_drops:
                drop[0] += drop[1]  # row += speed
                chars = drop[2]
                spins = drop[4]
                for i in range(_TRAIL_LENGTH):
                    # Flicker
                    if _rng.random() < _CHAR_FLICKER_PROB:
                        chars[i] = _rng.choice(_MATRIX_CHARS)
                    # Spin logic
                    if spins[i] is None:
                        if _rng.random() < _SPIN_PROBABILITY:
                            spins[i] = 0.0
                    else:
                        spins[i] += _SPIN_SPEED
                        if spins[i] >= math.tau:  # full revolution done
                            spins[i] = None
                # Mark dead if fully off-screen
                if drop[0] - _TRAIL_LENGTH > self._num_rows:
                    continue
                alive.append(drop)

            col_drops[:] = alive

        self.update()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color)
        painter.setFont(self._font)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        fa = self._fade_alpha
        if fa <= 0:
            painter.end()
            return

        w = self.width()
        x_offset = max(0, (w - self._num_cols * self._col_w) // 2)
        half_cw = self._half_col_w
        fm = self._fm

        for col_idx, col_drops in enumerate(self._drops):
            cx = x_offset + col_idx * self._col_w + half_cw
            for drop in col_drops:
                head_row = drop[0]
                chars = drop[2]
                spins = drop[4]
                for i in range(_TRAIL_LENGTH):
                    row = int(head_row) - i
                    if row < 0 or row >= self._num_rows:
                        continue
                    y = row * self._row_h

                    # Color with fade-out multiplier
                    if i == 0:
                        color = QColor(180, 255, 180, int(255 * fa))
                    elif i == 1:
                        color = QColor(0, 230, 0, int(255 * fa))
                    else:
                        t = i / _TRAIL_LENGTH
                        g = int(200 * (1.0 - t * 0.85))
                        a = int(255 * (1.0 - t * 0.9) * fa)
                        color = QColor(0, g, 0, a)

                    ch = chars[i]
                    spin_phase = spins[i]

                    if spin_phase is not None:
                        # Y-axis rotation: horizontal scale = cos(phase)
                        sx = math.cos(spin_phase)
                        if abs(sx) < _EDGE_ON_THRESHOLD:
                            continue  # edge-on — invisible
                        char_w = fm.horizontalAdvance(ch)
                        painter.save()
                        painter.translate(cx, y)
                        painter.scale(sx, 1.0)
                        painter.setPen(color)
                        painter.drawText(QPointF(-char_w / 2.0, 0.0), ch)
                        painter.restore()
                    else:
                        char_w = fm.horizontalAdvance(ch)
                        painter.setPen(color)
                        painter.drawText(QPointF(cx - char_w / 2.0, y), ch)

        painter.end()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._timer.isActive():
            self._init_grid()

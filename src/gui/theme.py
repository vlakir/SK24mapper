"""
Dark theme for SK42 (PyCharm Darcula style).

Provides a complete QSS-based dark theme inspired by JetBrains Darcula.
All styling is centralized here — individual widgets should NOT use inline
setStyleSheet() unless they need a truly unique override (e.g. translucent overlay).
"""

from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtGui import QColor, QPalette

from shared.constants import DWMWA_USE_IMMERSIVE_DARK_MODE

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DarculaPalette:
    """Frozen colour tokens for the Darcula-inspired dark theme."""

    bg_darkest: str = '#1e1f22'  # inputs, status bar
    bg_base: str = '#2b2d30'  # main background
    bg_surface: str = '#313335'  # panels, toolbars, groupboxes
    bg_input: str = '#1e1f22'  # input fields (same as darkest)

    accent_primary: str = '#375fac'  # focused/OK button blue
    accent_hover: str = '#4270c4'  # blue hover
    accent_pressed: str = '#2d508e'  # blue pressed
    accent_secondary: str = '#bcbec4'  # group headers (same as text)

    text_primary: str = '#bcbec4'  # main text
    text_secondary: str = '#6f737a'  # descriptions, hints
    text_accent: str = '#6897bb'  # coordinates, constants
    text_disabled: str = '#5a5d63'  # disabled text

    border_default: str = '#43454a'  # subtle borders
    border_focus: str = '#4882d4'  # focused input border

    selection_bg: str = '#2d5f9a'  # selection/highlight
    selection_text: str = '#bcbec4'  # text on selection

    blue: str = '#4882d4'  # primary blue (sliders, checkboxes, progress)
    blue_hover: str = '#5c9be0'  # blue hover (slider handles)

    maptype_bg: str = 'rgba(54,88,128,20)'  # subtle blue tint
    maptype_border: str = 'rgba(54,88,128,50)'  # blue border tint


PALETTE = DarculaPalette()

_ICONS_DIR = Path(__file__).parent / 'icons'


# ---------------------------------------------------------------------------
# QSS builder
# ---------------------------------------------------------------------------


def _icon_path(name: str) -> str:
    """Return forward-slash absolute path to an icon (for QSS url())."""
    return str(_ICONS_DIR / name).replace('\\', '/')


def build_stylesheet(p: DarculaPalette) -> str:
    """Return a complete QSS string covering all application widgets."""
    # Icon paths for checkbox / radio indicators
    chk_on = _icon_path('checkbox_checked.svg')
    chk_on_dis = _icon_path('checkbox_checked_disabled.svg')
    rad_on = _icon_path('radio_checked.svg')
    rad_on_dis = _icon_path('radio_checked_disabled.svg')
    chevron_down = _icon_path('chevron_down.svg')

    return f"""
/* ===== Base ===== */
QMainWindow, QDialog, QWidget {{
    background-color: {p.bg_base};
    color: {p.text_primary};
    font-size: 13px;
}}

/* ===== GroupBox ===== */
QGroupBox {{
    background-color: transparent;
    border: none;
    border-top: 1px solid {p.border_default};
    margin-top: 16px;
    padding: 12px 4px 4px 4px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 6px;
    color: {p.text_primary};
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: #4c5052;
    color: {p.text_primary};
    border: 1px solid {p.border_default};
    border-radius: 4px;
    padding: 4px 14px;
}}
QPushButton:hover {{
    background-color: #5c5f63;
    border-color: #5c5f63;
}}
QPushButton:pressed {{
    background-color: #3c3f41;
}}
QPushButton:disabled {{
    background-color: #393b40;
    color: {p.text_disabled};
    border-color: #393b40;
}}

/* Accent buttons (Создать / Сохранить карту) — IntelliJ focused button */
QPushButton#accentButton {{
    background-color: {p.accent_primary};
    color: #dfe1e5;
    border: 1px solid {p.accent_hover};
    border-radius: 4px;
    font-weight: bold;
    font-size: 14px;
    padding: 5px 18px;
    margin: 0 30px;
}}
QPushButton#accentButton:hover {{
    background-color: {p.accent_hover};
    color: #ffffff;
}}
QPushButton#accentButton:pressed {{
    background-color: {p.accent_pressed};
}}
QPushButton#accentButton:disabled {{
    background-color: #393b40;
    color: {p.text_disabled};
    border-color: #393b40;
}}

/* Small accent buttons (Сохранить / Сохранить как профиль) */
QPushButton#accentButtonSmall {{
    background-color: {p.accent_primary};
    color: #dfe1e5;
    border: 1px solid {p.accent_hover};
    border-radius: 4px;
    font-weight: bold;
    font-size: 14px;
    padding: 4px 14px;
}}
QPushButton#accentButtonSmall:hover {{
    background-color: {p.accent_hover};
    color: #ffffff;
}}
QPushButton#accentButtonSmall:pressed {{
    background-color: {p.accent_pressed};
}}
QPushButton#accentButtonSmall:disabled {{
    background-color: #393b40;
    color: {p.text_disabled};
    border-color: #393b40;
}}

/* ===== Frames ===== */
QFrame {{
    background-color: {p.bg_base};
    border: none;
}}
QFrame#maptypeFrame {{
    background-color: {p.maptype_bg};
    border: 1px solid {p.maptype_border};
    border-radius: 3px;
}}

/* ===== Labels ===== */
QLabel {{
    background-color: transparent;
    color: {p.text_primary};
    border: none;
}}
QLabel:disabled {{
    color: {p.text_disabled};
}}
QLabel#coordsLabel {{
    font-family: "Consolas", "Courier New", monospace;
    font-weight: bold;
    color: {p.text_accent};
    font-size: 16px;
}}
QLabel#infoLabel {{
    color: {p.text_secondary};
    font-size: 12px;
}}

/* ===== ComboBox ===== */
QComboBox {{
    background-color: {p.bg_input};
    color: {p.text_primary};
    border: 1px solid {p.border_default};
    border-radius: 4px;
    padding: 3px 8px;
}}
QComboBox:hover {{
    border-color: {p.border_focus};
}}
QComboBox:disabled {{
    background-color: {p.bg_darkest};
    color: {p.text_disabled};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: url('{chevron_down}');
    width: 10px;
    height: 10px;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {p.bg_surface};
    color: {p.text_primary};
    border: 1px solid {p.border_default};
    selection-background-color: {p.selection_bg};
    selection-color: {p.selection_text};
}}

/* ===== Sliders ===== */
QSlider::groove:horizontal {{
    height: 4px;
    background: {p.border_default};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {p.blue};
    border: none;
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{
    background: {p.blue_hover};
}}
QSlider::groove:vertical {{
    width: 4px;
    background: {p.border_default};
    border-radius: 2px;
}}
QSlider::handle:vertical {{
    background: {p.blue};
    border: none;
    height: 12px;
    margin: 0 -4px;
    border-radius: 6px;
}}
QSlider::handle:vertical:hover {{
    background: {p.blue_hover};
}}

/* ===== QDial ===== */
QDial {{
    background-color: {p.bg_surface};
}}

/* ===== Checkboxes & Radio ===== */
QCheckBox, QRadioButton {{
    color: {p.text_primary};
    spacing: 6px;
    background-color: transparent;
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: {p.text_disabled};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {p.border_default};
    background-color: {p.bg_input};
}}
QCheckBox::indicator {{
    border-radius: 3px;
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:checked {{
    image: url('{chk_on}');
    background-color: transparent;
    border: none;
}}
QRadioButton::indicator:checked {{
    image: url('{rad_on}');
    background-color: transparent;
    border: none;
}}
QCheckBox::indicator:checked:disabled {{
    image: url('{chk_on_dis}');
    background-color: transparent;
    border: none;
}}
QRadioButton::indicator:checked:disabled {{
    image: url('{rad_on_dis}');
    background-color: transparent;
    border: none;
}}
QCheckBox::indicator:unchecked:disabled, QRadioButton::indicator:unchecked:disabled {{
    background-color: {p.bg_darkest};
    border-color: #393b40;
}}

/* ===== Scroll areas ===== */
QScrollArea {{
    background-color: {p.bg_base};
    border: none;
}}
QScrollBar:vertical {{
    background: {p.bg_darkest};
    width: 10px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: #5a5a5a;
    min-height: 30px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: #6e6e6e;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}
QScrollBar:horizontal {{
    background: {p.bg_darkest};
    height: 10px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: #5a5a5a;
    min-width: 30px;
    border-radius: 5px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #6e6e6e;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* ===== StatusBar ===== */
QStatusBar {{
    background-color: {p.bg_base};
    color: {p.text_secondary};
    border-top: 1px solid {p.border_default};
}}
QStatusBar::item {{
    border: none;
}}
QStatusBar QLabel {{
    color: {p.text_secondary};
    padding: 0 4px;
}}
QStatusBar QLabel#progressLabel {{
    color: {p.text_primary};
}}

/* ===== ProgressBar (QProgressDialog fallback only) ===== */
QProgressBar {{
    background-color: transparent;
    border: none;
    max-height: 3px;
    min-height: 3px;
}}
QProgressBar::chunk {{
    background-color: {p.blue};
}}

/* ===== MenuBar & Menu ===== */
QMenuBar {{
    background-color: {p.bg_base};
    color: {p.text_primary};
    border-bottom: 1px solid {p.border_default};
}}
QMenuBar:disabled {{
    color: {p.text_disabled};
}}
QMenuBar::item:selected {{
    background-color: {p.selection_bg};
    color: {p.text_primary};
}}
QMenu {{
    background-color: {p.bg_base};
    color: {p.text_primary};
    border: 1px solid {p.border_default};
    padding: 4px 0;
}}
QMenu::item {{
    padding: 4px 24px;
}}
QMenu::item:selected {{
    background-color: {p.selection_bg};
    color: {p.text_primary};
}}
QMenu::item:disabled {{
    color: {p.text_disabled};
}}
QMenu::separator {{
    height: 1px;
    background: {p.border_default};
    margin: 4px 8px;
}}

/* ===== LineEdit / SpinBoxes ===== */
QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: {p.bg_input};
    color: {p.text_primary};
    border: 1px solid {p.border_default};
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: {p.selection_bg};
    selection-color: {p.selection_text};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {p.border_focus};
}}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
    background-color: {p.bg_darkest};
    color: {p.text_disabled};
}}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    width: 0;
    height: 0;
    border: none;
}}

/* ===== Splitter ===== */
QSplitter::handle {{
    background-color: {p.border_default};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QSplitter::handle:vertical {{
    height: 3px;
}}
QSplitter::handle:hover {{
    background-color: {p.blue};
}}

/* ===== ToolTip ===== */
QToolTip {{
    background-color: #4b4b4b;
    color: #bfbfbf;
    border: 1px solid #5c5c5c;
    padding: 4px 8px;
    font-size: 12px;
}}

/* ===== QGraphicsView ===== */
QGraphicsView {{
    background-color: #000000;
    border: 1px solid {p.border_default};
}}

/* ===== DialogButtonBox ===== */
QDialogButtonBox QPushButton {{
    min-width: 80px;
}}

/* ===== Cancel button (PyCharm-style: × in a pale circle) ===== */
QPushButton#cancelButton {{
    background-color: rgba(255, 255, 255, 12);
    border: none;
    border-radius: 8px;
    color: {p.text_secondary};
    font-size: 11px;
    padding: 0;
    margin: 0 2px;
    min-width: 16px;
    max-width: 16px;
    min-height: 16px;
    max-height: 16px;
}}
QPushButton#cancelButton:hover {{
    color: #ffffff;
    background-color: rgba(255, 255, 255, 25);
}}
QPushButton#cancelButton:pressed {{
    color: #cccccc;
    background-color: rgba(255, 255, 255, 35);
}}

/* ===== ProgressDialog ===== */
QProgressDialog {{
    background-color: {p.bg_base};
}}
"""


# ---------------------------------------------------------------------------
# QPalette builder (for QDial, QRangeSlider, native-painted widgets)
# ---------------------------------------------------------------------------


def _build_palette(p: DarculaPalette) -> QPalette:
    """Build a QPalette matching the Darcula theme."""
    pal = QPalette()

    pal.setColor(QPalette.ColorRole.Window, QColor(p.bg_base))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p.text_primary))
    pal.setColor(QPalette.ColorRole.Base, QColor(p.bg_input))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p.bg_surface))
    pal.setColor(QPalette.ColorRole.Text, QColor(p.text_primary))
    pal.setColor(QPalette.ColorRole.Button, QColor(p.bg_surface))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p.text_primary))
    pal.setColor(QPalette.ColorRole.BrightText, QColor(p.text_accent))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p.selection_bg))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(p.selection_text))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p.bg_surface))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p.text_accent))
    pal.setColor(QPalette.ColorRole.Link, QColor(p.accent_hover))
    pal.setColor(QPalette.ColorRole.LinkVisited, QColor(p.accent_pressed))
    pal.setColor(QPalette.ColorRole.Mid, QColor(p.border_default))
    pal.setColor(QPalette.ColorRole.Dark, QColor(p.bg_darkest))
    pal.setColor(QPalette.ColorRole.Shadow, QColor(p.bg_darkest))
    pal.setColor(QPalette.ColorRole.Light, QColor(p.bg_surface))
    pal.setColor(QPalette.ColorRole.Midlight, QColor(p.bg_surface))

    # Disabled group
    pal.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(p.text_disabled),
    )
    pal.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.Text,
        QColor(p.text_disabled),
    )
    pal.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(p.text_disabled),
    )

    return pal


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def apply_dark_title_bar(hwnd: int) -> None:
    """Tell Windows DWM to use a dark title bar for the given window handle."""
    if sys.platform != 'win32':
        return
    try:
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
    except Exception:
        logger.debug('DwmSetWindowAttribute failed — older Windows?')


def apply_theme(app: QApplication) -> None:
    """Apply the Darcula dark theme to *app*."""
    p = PALETTE
    app.setPalette(_build_palette(p))
    app.setStyleSheet(build_stylesheet(p))

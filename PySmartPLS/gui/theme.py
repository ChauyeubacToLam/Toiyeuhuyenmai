"""Central design system for PySmartPLS.

A single source of truth for colours, elevation and the application stylesheet.
The visual language is "Modern Premium": generous spacing, soft hairline
borders, 8-12px radii, a royal-blue accent and quiet, refined chrome that still
reads as a serious research tool (SmartPLS parity, more polished).

Public API
----------
- ``PALETTES``           : dict[str, dict]  design tokens per theme
- ``palette(theme)``     : resolve a palette (falls back to the default)
- ``build_stylesheet(theme)`` : full Qt stylesheet for the app + dialogs
- ``apply_shadow(widget, ...)`` : soft drop-shadow (elevation) helper
- ``node_colors(theme)`` : canvas node/edge colours for the active theme
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget

def _qss_asset_dir() -> Path:
    override = os.environ.get("PYSMARTPLS_QSS_CACHE")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "PySmartPLS"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "PySmartPLS"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pysmartpls"
    target = base / "qss"
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        return Path(tempfile.gettempdir()) / "pysmartpls-qss"


ASSETS_DIR = _qss_asset_dir()


DEFAULT_THEME = "classic"  # Premium Light (royal blue) — the app default.

FONT_FAMILY = '"Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif'
MONO_FAMILY = '"Cascadia Code", "Consolas", "SF Mono", monospace'


# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
PALETTES: dict[str, dict] = {
    # Premium Light — the default. Cool neutral chrome so white cards lift.
    "classic": {
        "bg": "#EEF1F6", "surface": "#FFFFFF", "surface_alt": "#F3F6FB",
        "surface_sunken": "#F7F9FC", "elevated": "#FFFFFF",
        "border": "#E3E8F0", "border_strong": "#CFD8E3", "divider": "#EDF1F6",
        "text": "#1B2433", "subtext": "#5B6776", "muted": "#8A96A6", "on_accent": "#FFFFFF",
        "accent": "#1D6FE0", "accent_hover": "#3B82F6", "accent_pressed": "#1657B0",
        "accent_soft": "#E6F0FD", "accent_softer": "#F2F7FE",
        "sel_bg": "#DBEAFD", "sel_text": "#0F2A50",
        "success": "#0E9F6E", "success_soft": "#E4F6EF",
        "warning": "#D98A00", "warning_soft": "#FBF1DC",
        "danger": "#E11D48", "danger_soft": "#FCE7EC",
        "canvas": "#FBFCFE", "grid": "#E9EEF5",
        "scrollbar": "#C7D0DD", "scrollbar_hover": "#A7B4C6",
        "shadow": "#16224033",
        "node_reflective_a": "#5AA0F7", "node_reflective_b": "#1D6FE0",
        "node_formative_a": "#FFD267", "node_formative_b": "#F2A93B",
        "node_incomplete_a": "#FF8A8A", "node_incomplete_b": "#EF4444",
        "node_indicator_a": "#FFE9A6", "node_indicator_b": "#FFD15C",
        "node_indicator_border": "#E0AE3D", "node_text": "#10243F",
        "node_border": "#1B5BBE", "edge": "#7A8699", "edge_strong": "#3B82F6",
    },
    # Airy alternate light — whiter chrome.
    "light": {
        "bg": "#F4F6FA", "surface": "#FFFFFF", "surface_alt": "#F1F4F9",
        "surface_sunken": "#F6F8FC", "elevated": "#FFFFFF",
        "border": "#E5E9F0", "border_strong": "#D2D9E3", "divider": "#EEF1F6",
        "text": "#111827", "subtext": "#5A6473", "muted": "#909AA8", "on_accent": "#FFFFFF",
        "accent": "#2563EB", "accent_hover": "#3B82F6", "accent_pressed": "#1D4FB8",
        "accent_soft": "#E8F0FE", "accent_softer": "#F3F7FE",
        "sel_bg": "#DCE9FD", "sel_text": "#0B2A55",
        "success": "#0E9F6E", "success_soft": "#E4F6EF",
        "warning": "#D98A00", "warning_soft": "#FBF1DC",
        "danger": "#E11D48", "danger_soft": "#FCE7EC",
        "canvas": "#FCFDFE", "grid": "#EBEFF5",
        "scrollbar": "#CBD4E0", "scrollbar_hover": "#AAB6C7",
        "shadow": "#1A2A4A2E",
        "node_reflective_a": "#5AA0F7", "node_reflective_b": "#2563EB",
        "node_formative_a": "#FFD267", "node_formative_b": "#F2A93B",
        "node_incomplete_a": "#FF8A8A", "node_incomplete_b": "#EF4444",
        "node_indicator_a": "#FFE9A6", "node_indicator_b": "#FFD15C",
        "node_indicator_border": "#E0AE3D", "node_text": "#10243F",
        "node_border": "#1D4FB8", "edge": "#7A8699", "edge_strong": "#3B82F6",
    },
    # Premium Dark.
    "dark": {
        "bg": "#15171C", "surface": "#1E2128", "surface_alt": "#262A33",
        "surface_sunken": "#191C22", "elevated": "#242831",
        "border": "#2E333D", "border_strong": "#3B414D", "divider": "#23272F",
        "text": "#E7EAEF", "subtext": "#9BA5B3", "muted": "#6C7482", "on_accent": "#FFFFFF",
        "accent": "#4F8FF0", "accent_hover": "#6AA3F5", "accent_pressed": "#3C78D6",
        "accent_soft": "#1E3050", "accent_softer": "#1A2433",
        "sel_bg": "#2B445F", "sel_text": "#EAF2FF",
        "success": "#2BB98A", "success_soft": "#15302A",
        "warning": "#E0A23A", "warning_soft": "#332815",
        "danger": "#F26D7D", "danger_soft": "#3A1F25",
        "canvas": "#1A1D23", "grid": "#262A32",
        "scrollbar": "#3A414D", "scrollbar_hover": "#4C5562",
        "shadow": "#00000066",
        "node_reflective_a": "#5C9CF5", "node_reflective_b": "#2C6FD4",
        "node_formative_a": "#F4C572", "node_formative_b": "#D69A33",
        "node_incomplete_a": "#F47C7C", "node_incomplete_b": "#D63B3B",
        "node_indicator_a": "#5A5230", "node_indicator_b": "#7A6A2E",
        "node_indicator_border": "#A98F3B", "node_text": "#F3F2E8",
        "node_border": "#5C9CF5", "edge": "#737E8C", "edge_strong": "#6AA3F5",
    },
    # Accessible (Okabe–Ito derived) high-contrast palette.
    "colorblind": {
        "bg": "#EDEFF2", "surface": "#FFFFFF", "surface_alt": "#F0F2F5",
        "surface_sunken": "#F6F7F9", "elevated": "#FFFFFF",
        "border": "#C9CDD3", "border_strong": "#A8AEB6", "divider": "#E6E8EC",
        "text": "#101418", "subtext": "#41474F", "muted": "#6B7178", "on_accent": "#FFFFFF",
        "accent": "#0072B2", "accent_hover": "#2189C9", "accent_pressed": "#005A8E",
        "accent_soft": "#DBEDF7", "accent_softer": "#EEF6FB",
        "sel_bg": "#CFE6F5", "sel_text": "#06324B",
        "success": "#009E73", "success_soft": "#DCF1EA",
        "warning": "#E69F00", "warning_soft": "#FBEFD6",
        "danger": "#D55E00", "danger_soft": "#FBE6D7",
        "canvas": "#FCFCFD", "grid": "#E2E4E8",
        "scrollbar": "#B6BBC2", "scrollbar_hover": "#969CA4",
        "shadow": "#10141833",
        "node_reflective_a": "#3FA0DE", "node_reflective_b": "#0072B2",
        "node_formative_a": "#F2C24A", "node_formative_b": "#E69F00",
        "node_incomplete_a": "#E8784A", "node_incomplete_b": "#D55E00",
        "node_indicator_a": "#FBE6B0", "node_indicator_b": "#F0CC6A",
        "node_indicator_border": "#C79A3A", "node_text": "#101418",
        "node_border": "#005A8E", "edge": "#6B7178", "edge_strong": "#0072B2",
    },
    # Tóc Mây — premium rose signature.
    "pink": {
        "bg": "#FBECF3", "surface": "#FFFFFF", "surface_alt": "#FCF1F6",
        "surface_sunken": "#FEF6FA", "elevated": "#FFFFFF",
        "border": "#F4D7E5", "border_strong": "#EBBBD2", "divider": "#F7E2EC",
        "text": "#3A1226", "subtext": "#8A5570", "muted": "#B189A0", "on_accent": "#FFFFFF",
        "accent": "#DB2777", "accent_hover": "#EC4899", "accent_pressed": "#BE185D",
        "accent_soft": "#FBE3EF", "accent_softer": "#FDF1F7",
        "sel_bg": "#FBD5E6", "sel_text": "#5A1538",
        "success": "#0E9F6E", "success_soft": "#E4F6EF",
        "warning": "#D98A00", "warning_soft": "#FBF1DC",
        "danger": "#E11D48", "danger_soft": "#FCE7EC",
        "canvas": "#FFFAFD", "grid": "#F6DEEB",
        "scrollbar": "#EEC3D8", "scrollbar_hover": "#E29CBE",
        "shadow": "#7A164A2E",
        "node_reflective_a": "#F06FA6", "node_reflective_b": "#DB2777",
        "node_formative_a": "#FFD267", "node_formative_b": "#F2A93B",
        "node_incomplete_a": "#FF8A8A", "node_incomplete_b": "#EF4444",
        "node_indicator_a": "#FCE3EF", "node_indicator_b": "#F7C4DC",
        "node_indicator_border": "#E29CBE", "node_text": "#3A1226",
        "node_border": "#BE185D", "edge": "#B98AA3", "edge_strong": "#EC4899",
    },
}


# Second accent (violet) — the "Phi tuyến tính" (Nonlinear ML) identity. Added to
# every palette so theme-switching never breaks; mirrors the violet used by the
# nonlinear icons (gui/icons.py VIOLET) and engine charts (core/nonlinear_engine).
_ACCENT2 = {
    #            base       hover      pressed    soft       softer
    "classic": ("#7C5CFC", "#8E72FD", "#6A47E8", "#ECE7FF", "#F6F3FF"),
    "light": ("#7C5CFC", "#8E72FD", "#6A47E8", "#ECE7FF", "#F6F3FF"),
    "dark": ("#9B85F5", "#AD9BF7", "#876FE6", "#2A2347", "#211C33"),
    "colorblind": ("#5D3FD3", "#6E52DC", "#4E33B8", "#E6E0FA", "#F2EFFB"),
    "pink": ("#9333EA", "#A24EEE", "#7E27CC", "#F3E8FD", "#FAF4FE"),
}
for _name, _pal in PALETTES.items():
    _a2, _a2_hover, _a2_pressed, _a2_soft, _a2_softer = _ACCENT2.get(_name, _ACCENT2["classic"])
    _pal.setdefault("accent2", _a2)
    _pal.setdefault("accent2_hover", _a2_hover)
    _pal.setdefault("accent2_pressed", _a2_pressed)
    _pal.setdefault("accent2_soft", _a2_soft)
    _pal.setdefault("accent2_softer", _a2_softer)


def palette(theme: str) -> dict:
    return PALETTES.get(theme, PALETTES[DEFAULT_THEME])


def node_colors(theme: str) -> dict:
    """Convenience accessor used by the canvas."""
    return palette(theme)


def _write_check(path: Path, color: str) -> None:
    s = 36
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 4.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    path_ = QPainterPath()
    path_.moveTo(8, 18)
    path_.lineTo(15, 25)
    path_.lineTo(28, 11)
    p.drawPath(path_)
    p.end()
    pm.save(str(path))


def _write_dot(path: Path, color: str) -> None:
    s = 18
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(QPointF(s / 2, s / 2), 4.0, 4.0)
    p.end()
    pm.save(str(path))


def _write_chevron(path: Path, color: str) -> None:
    s = 28
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 2.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.drawPolyline([QPointF(7, 11), QPointF(14, 18), QPointF(21, 11)])
    p.end()
    pm.save(str(path))


def ensure_qss_assets(theme: str) -> dict[str, str]:
    """Generate small QSS sub-control bitmaps (checkmark, chevron) on demand.

    Returns absolute, forward-slashed paths suitable for ``url(...)`` in QSS.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    c = palette(theme)
    check = ASSETS_DIR / "qss_check.png"
    if not check.exists():
        _write_check(check, "#FFFFFF")
    chevron = ASSETS_DIR / f"qss_chevron_{theme}.png"
    if not chevron.exists():
        _write_chevron(chevron, c["subtext"])
    chevron_accent = ASSETS_DIR / f"qss_chevron_accent_{theme}.png"
    if not chevron_accent.exists():
        _write_chevron(chevron_accent, c["accent"])
    dot = ASSETS_DIR / f"qss_dot_{theme}.png"
    if not dot.exists():
        _write_dot(dot, c["accent"])
    return {
        "check": str(check).replace("\\", "/"),
        "chevron": str(chevron).replace("\\", "/"),
        "chevron_accent": str(chevron_accent).replace("\\", "/"),
        "dot": str(dot).replace("\\", "/"),
    }


def apply_shadow(widget: QWidget, *, blur: int = 28, y: int = 8, x: int = 0,
                 color: str = "#16224033") -> QGraphicsDropShadowEffect:
    """Attach a soft, premium drop shadow to a widget for elevation."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(x, y)
    effect.setColor(QColor(color))
    widget.setGraphicsEffect(effect)
    return effect


# --------------------------------------------------------------------------- #
# Stylesheet
# --------------------------------------------------------------------------- #
def build_stylesheet(theme: str = DEFAULT_THEME) -> str:
    c = palette(theme)
    a = ensure_qss_assets(theme)
    return f"""
    /* ---------- base ---------- */
    * {{ outline: 0; }}
    QWidget {{
        background: transparent; color: {c['text']};
        font-family: {FONT_FAMILY}; font-size: 9.5pt;
    }}
    QMainWindow, QDialog {{ background: {c['bg']}; }}
    QToolTip {{
        background: {c['text']}; color: {c['surface']};
        border: 0; border-radius: 7px; padding: 6px 10px; font-size: 9pt;
    }}

    /* ---------- menu bar & menus ---------- */
    QMenuBar {{ background: {c['surface']}; color: {c['text']};
        border-bottom: 1px solid {c['border']}; padding: 3px 6px; spacing: 2px; }}
    QMenuBar::item {{ background: transparent; padding: 6px 12px; border-radius: 7px; }}
    QMenuBar::item:selected {{ background: {c['accent_soft']}; color: {c['accent']}; }}
    QMenuBar::item:pressed {{ background: {c['accent_soft']}; color: {c['accent']}; }}
    QMenu {{ background: {c['elevated']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 12px; padding: 7px; }}
    QMenu::item {{ padding: 8px 30px 8px 30px; border-radius: 8px; min-width: 230px; }}
    QMenu::item:selected {{ background: {c['accent_soft']}; color: {c['accent']}; }}
    QMenu::item:disabled {{ color: {c['muted']}; }}
    QMenu::separator {{ height: 1px; background: {c['divider']}; margin: 6px 10px; }}
    QMenu::icon {{ padding-left: 8px; }}
    QMenu::indicator {{ width: 18px; height: 18px; left: 8px; }}

    /* ---------- main toolbar ---------- */
    QToolBar#MainToolbar {{ background: {c['surface']};
        border-bottom: 1px solid {c['border']}; spacing: 2px; padding: 6px 8px; }}
    QToolBar#MainToolbar::separator {{ background: {c['divider']}; width: 1px; margin: 10px 5px; }}
    QToolBar#MainToolbar QToolButton {{ background: transparent; color: {c['subtext']};
        border: 1px solid transparent; border-radius: 9px;
        min-width: 50px; min-height: 46px; padding: 4px 4px; font-size: 8.2pt; }}
    QToolBar#MainToolbar QToolButton:hover {{ background: {c['surface_alt']};
        color: {c['text']}; border-color: {c['border']}; }}
    QToolBar#MainToolbar QToolButton:pressed {{ background: {c['accent_soft']}; }}
    QToolBar#MainToolbar QToolButton:checked {{ background: {c['accent_soft']};
        color: {c['accent']}; border-color: {c['accent']}; }}
    QToolBar#MainToolbar QToolButton:disabled {{ color: {c['muted']}; }}

    /* ---------- splitters ---------- */
    QSplitter#MainSplitter, QSplitter#LeftSidebar {{ background: {c['bg']}; }}
    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:horizontal {{ width: 10px; }}
    QSplitter::handle:vertical {{ height: 10px; }}

    /* ---------- side panels (cards) ---------- */
    QFrame#SidePanel {{ background: {c['surface']};
        border: 1px solid {c['border']}; border-radius: 14px; }}
    QFrame#PanelHeader {{ background: transparent;
        border: 0; border-bottom: 1px solid {c['divider']};
        border-top-left-radius: 14px; border-top-right-radius: 14px; }}
    QLabel#PanelTitleText {{ color: {c['text']}; font-size: 10pt; font-weight: 700;
        letter-spacing: 0.2px; }}
    QLabel#BestCorrelation {{ color: {c['success']}; background: {c['success_soft']};
        border-top: 1px solid {c['border']};
        border-bottom-left-radius: 14px; border-bottom-right-radius: 14px;
        padding: 7px; font-weight: 600; }}
    QToolButton#PanelButton {{ background: transparent; border: 1px solid transparent;
        border-radius: 8px; padding: 3px; }}
    QToolButton#PanelButton:hover {{ background: {c['accent_soft']}; border-color: {c['accent']}; }}
    QToolButton#PanelButton:pressed {{ background: {c['sel_bg']}; }}

    /* ---------- trees, lists, tables ---------- */
    QTreeWidget#ProjectTree, QListWidget#IndicatorList, QTableWidget#IndicatorList {{
        background: {c['surface']}; color: {c['text']};
        border: 0; border-radius: 0 0 14px 14px; padding: 6px; }}
    QTableWidget#IndicatorList {{ gridline-color: transparent; }}
    QTreeWidget::item, QListWidget::item {{
        min-height: 30px; border-radius: 8px; padding: 2px 6px; }}
    QTreeWidget::item:hover, QListWidget::item:hover,
    QTableWidget#IndicatorList::item:hover {{ background: {c['surface_alt']}; }}
    QTreeWidget::item:selected, QListWidget::item:selected,
    QTableWidget#IndicatorList::item:selected {{ background: {c['accent_soft']}; color: {c['accent']}; }}
    QTableWidget#IndicatorList::item {{ padding: 5px 8px; }}

    /* ---------- workspace tabs ---------- */
    QFrame#WorkspaceContainer {{ background: {c['bg']}; border: 0; }}
    QTabWidget::pane {{ background: {c['surface']};
        border: 1px solid {c['border']}; border-radius: 14px; top: -1px; }}
    QTabWidget#WorkspaceTabs::pane {{ border-radius: 14px; }}
    QTabBar {{ qproperty-drawBase: 0; }}
    QTabBar::tab {{ background: transparent; color: {c['subtext']};
        border: 0; padding: 9px 18px; margin-right: 4px;
        border-top-left-radius: 10px; border-top-right-radius: 10px; font-weight: 600; }}
    QTabBar::tab:hover {{ color: {c['text']}; background: {c['surface_alt']}; }}
    QTabBar::tab:selected {{ color: {c['accent']}; background: {c['surface']};
        border: 1px solid {c['border']}; border-bottom: 2px solid {c['accent']}; }}

    /* report tabs (closable) */
    QTabWidget#ReportTabs::pane {{ background: {c['surface']};
        border: 1px solid {c['border']}; border-radius: 0 14px 14px 14px; }}
    QTabWidget#ReportTabs QTabBar::tab {{ background: {c['surface_alt']}; color: {c['subtext']};
        border: 1px solid {c['border']}; border-bottom: 0;
        border-top-left-radius: 10px; border-top-right-radius: 10px;
        padding: 8px 16px; margin-right: 3px; }}
    QTabWidget#ReportTabs QTabBar::tab:selected {{ background: {c['surface']};
        color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}
    QTabWidget#ReportTabs QTabBar::tab:hover {{ background: {c['accent_soft']}; }}
    QTabBar::close-button {{ subcontrol-position: right; }}

    QWidget#BlankWorkspace, QWidget#DataPage, QWidget#ResultsPage {{ background: {c['surface']}; }}

    /* ---------- welcome / empty workspace ---------- */
    QWidget#WelcomeRoot {{ background: {c['surface']}; }}
    QLabel#WelcomeBadge {{ background: {c['accent_soft']}; border-radius: 20px; }}
    QLabel#WelcomeKicker {{ color: {c['accent']}; font-size: 9pt; font-weight: 800;
        letter-spacing: 1.5px; }}
    QLabel#WelcomeTitle {{ color: {c['text']}; font-size: 27pt; font-weight: 800;
        letter-spacing: -0.6px; }}
    QLabel#WelcomeSubtitle {{ color: {c['subtext']}; font-size: 11.5pt; }}
    QFrame#WelcomeCard {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 16px; }}
    QFrame#WelcomeCard:hover {{ border: 1px solid {c['accent']}; background: {c['accent_softer']}; }}
    QLabel#WelcomeCardIcon {{ background: {c['accent_soft']}; border-radius: 13px; }}
    QLabel#WelcomeCardTitle {{ color: {c['text']}; font-size: 12.5pt; font-weight: 800; }}
    QLabel#WelcomeCardBody {{ color: {c['subtext']}; font-size: 9.6pt; }}
    QLabel#WelcomeFootnote {{ color: {c['muted']}; font-size: 9pt; }}
    QLabel#DataMetaValue {{ color: {c['accent']}; font-size: 11pt; font-weight: 700; }}
    QLabel#EmptyResult {{ color: {c['muted']}; font-size: 12pt; }}

    /* ---------- data screening view ---------- */
    QScrollArea#ScreenScroll {{ background: {c['surface']}; border: 0; }}
    QWidget#ScreenContent {{ background: {c['surface']}; }}
    QLabel#ScreenEmpty {{ color: {c['muted']}; font-size: 11pt; padding: 40px; }}
    QFrame#ScreenCard {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 14px; }}
    QLabel#ScreenSectionTitle {{ color: {c['text']}; font-size: 11.5pt; font-weight: 800;
        letter-spacing: 0.2px; }}
    QLabel#ScreenSectionNote {{ color: {c['subtext']}; font-size: 9pt; }}
    QFrame#ScreenChip {{ background: {c['surface_alt']}; border: 1px solid {c['border']};
        border-radius: 12px; min-width: 96px; }}
    QFrame#ScreenChip[tone="warning"] {{ background: {c['warning_soft']}; border-color: {c['warning']}; }}
    QFrame#ScreenChip[tone="danger"] {{ background: {c['danger_soft']}; border-color: {c['danger']}; }}
    QLabel#ScreenChipValue {{ color: {c['accent']}; font-size: 17pt; font-weight: 800; }}
    QFrame#ScreenChip[tone="warning"] QLabel#ScreenChipValue {{ color: {c['warning']}; }}
    QFrame#ScreenChip[tone="danger"] QLabel#ScreenChipValue {{ color: {c['danger']}; }}
    QLabel#ScreenChipTitle {{ color: {c['subtext']}; font-size: 8.4pt; font-weight: 600; }}
    QTableView#ScreenTable {{ background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        gridline-color: {c['divider']}; selection-background-color: {c['sel_bg']};
        selection-color: {c['sel_text']}; }}
    QTableView#ScreenTable::item {{ padding: 5px 8px; }}
    QTextBrowser#ScreenInterpret, QTextBrowser#ScreenStrobe {{ background: {c['surface_alt']};
        color: {c['text']}; border: 1px solid {c['border']}; border-radius: 10px;
        padding: 10px 12px; font-size: 9.8pt; }}
    QTextBrowser#ScreenStrobe {{ background: {c['accent_softer']}; }}

    /* ---------- inputs ---------- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border_strong']}; border-radius: 9px;
        padding: 7px 10px; selection-background-color: {c['sel_bg']};
        selection-color: {c['sel_text']}; }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {c['muted']}; }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QTextEdit:focus, QPlainTextEdit:focus {{
        border: 2px solid {c['accent']}; padding: 6px 9px; background: {c['accent_softer']}; }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background: {c['surface_alt']}; color: {c['muted']}; }}
    QComboBox::drop-down {{ border: 0; width: 28px; }}
    QComboBox::down-arrow {{ image: url({a['chevron']}); width: 14px; height: 14px; }}
    QComboBox::down-arrow:on {{ image: url({a['chevron_accent']}); }}
    QComboBox QAbstractItemView {{ background: {c['elevated']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px; padding: 5px;
        selection-background-color: {c['accent_soft']}; selection-color: {c['accent']};
        outline: 0; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background: transparent; border: 0; width: 20px; }}

    /* ---------- check & radio ---------- */
    QCheckBox, QRadioButton {{ color: {c['text']}; spacing: 8px; padding: 2px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 18px; height: 18px; }}
    QCheckBox::indicator {{ border: 1.6px solid {c['border_strong']};
        border-radius: 6px; background: {c['surface']}; }}
    QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
    QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']};
        image: url({a['check']}); }}
    QCheckBox::indicator:disabled {{ background: {c['surface_alt']}; border-color: {c['border']}; }}
    QRadioButton::indicator {{ border: 1.6px solid {c['border_strong']};
        border-radius: 9px; background: {c['surface']}; }}
    QRadioButton::indicator:hover {{ border-color: {c['accent']}; }}
    QRadioButton::indicator:checked {{ border: 1.6px solid {c['accent']};
        background: {c['surface']}; image: url({a['dot']}); }}
    QRadioButton::indicator:disabled {{ border-color: {c['border']}; background: {c['surface_alt']}; }}

    /* ---------- group boxes ---------- */
    QGroupBox {{ background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 12px;
        margin-top: 16px; padding: 14px 12px 12px 12px; font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
        left: 14px; top: 2px; padding: 0 6px; color: {c['subtext']};
        background: {c['surface']}; }}

    /* ---------- buttons ---------- */
    QPushButton {{ background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border_strong']}; border-radius: 9px;
        padding: 8px 16px; font-weight: 600; }}
    QPushButton:hover {{ background: {c['surface_alt']}; border-color: {c['muted']}; }}
    QPushButton:pressed {{ background: {c['sel_bg']}; }}
    QPushButton:disabled {{ background: {c['surface_alt']}; color: {c['muted']};
        border-color: {c['border']}; }}
    QPushButton:default {{ border-color: {c['accent']}; }}
    QPushButton#PrimaryButton {{ background: {c['accent']}; color: {c['on_accent']};
        border: 1px solid {c['accent']}; padding: 9px 20px; }}
    QPushButton#PrimaryButton:hover {{ background: {c['accent_hover']}; border-color: {c['accent_hover']}; }}
    QPushButton#PrimaryButton:pressed {{ background: {c['accent_pressed']}; border-color: {c['accent_pressed']}; }}
    QPushButton#PrimaryButton:disabled {{ background: {c['muted']}; border-color: {c['muted']}; color: {c['surface']}; }}
    QPushButton#GhostButton {{ background: transparent; border: 1px solid transparent;
        color: {c['accent']}; }}
    QPushButton#GhostButton:hover {{ background: {c['accent_soft']}; }}
    QPushButton#DangerButton {{ background: {c['danger']}; color: #FFFFFF; border-color: {c['danger']}; }}
    QPushButton#DangerButton:hover {{ background: {c['danger']}; }}

    /* ---------- headers ---------- */
    QHeaderView {{ background: transparent; }}
    QHeaderView::section {{ background: {c['surface_alt']}; color: {c['subtext']};
        border: 0; border-right: 1px solid {c['border']}; border-bottom: 1px solid {c['border']};
        padding: 8px 10px; font-weight: 700; font-size: 8.6pt; }}
    QHeaderView::section:first {{ border-top-left-radius: 0; }}
    QTableCornerButton::section {{ background: {c['surface_alt']}; border: 0;
        border-bottom: 1px solid {c['border']}; }}

    /* ---------- generic tables, lists, trees ---------- */
    QTableWidget, QTableView {{ background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 12px;
        gridline-color: {c['divider']}; alternate-background-color: {c['surface_alt']};
        selection-background-color: {c['sel_bg']}; selection-color: {c['sel_text']}; }}
    QTableWidget::item, QTableView::item {{ padding: 5px 8px; }}
    QListWidget, QTreeWidget {{ background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 12px;
        selection-background-color: {c['accent_soft']}; selection-color: {c['accent']}; }}

    /* ---------- progress ---------- */
    QProgressBar {{ background: {c['surface_alt']}; color: {c['text']};
        border: 0; border-radius: 9px; height: 16px; text-align: center; font-weight: 700;
        font-size: 8.2pt; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 9px; margin: 0; }}

    /* ---------- scrollbars (premium, thin) ---------- */
    QScrollBar:vertical {{ background: transparent; width: 13px; margin: 3px; }}
    QScrollBar::handle:vertical {{ background: {c['scrollbar']}; border-radius: 5px; min-height: 36px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['scrollbar_hover']}; }}
    QScrollBar:horizontal {{ background: transparent; height: 13px; margin: 3px; }}
    QScrollBar::handle:horizontal {{ background: {c['scrollbar']}; border-radius: 5px; min-width: 36px; }}
    QScrollBar::handle:horizontal:hover {{ background: {c['scrollbar_hover']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; background: transparent; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    /* ---------- status bar ---------- */
    QStatusBar {{ background: {c['surface']}; color: {c['subtext']};
        border-top: 1px solid {c['border']}; }}

    /* ---------- model styling sidebar ---------- */
    QFrame#ModelSidebar {{ background: {c['surface']};
        border: 1px solid {c['border']}; border-radius: 14px; }}
    QFrame#ModelSidebar QToolButton {{ background: transparent; color: {c['text']};
        border: 1px solid transparent; border-radius: 8px; padding: 4px; }}
    QFrame#ModelSidebar QToolButton:hover {{ background: {c['surface_alt']}; border-color: {c['border']}; }}
    QFrame#ModelSidebar QToolButton:checked {{ background: {c['accent_soft']};
        color: {c['accent']}; border-color: {c['accent']}; }}
    QFrame#ModelSidebar QPushButton {{ background: {c['surface_alt']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 8px; padding: 6px 8px; }}
    QFrame#ModelSidebar QPushButton:hover {{ background: {c['accent_soft']}; border-color: {c['accent']}; }}
    QLabel#SidebarHeading {{ font-weight: 700; font-size: 8.6pt; color: {c['subtext']};
        letter-spacing: 0.6px; padding-top: 4px; }}
    QLabel#SidebarSwatchLabel {{ color: {c['subtext']}; font-size: 8.6pt; font-weight: 700;
        letter-spacing: 0.6px; }}

    /* ---------- result views ---------- */
    QLabel#ResultTitle {{ color: {c['text']}; font-size: 15pt; font-weight: 800;
        letter-spacing: -0.2px; }}
    QLabel#ResultSubtitle {{ color: {c['subtext']}; font-size: 9.5pt; }}
    QLabel#CopyLabel {{ color: {c['subtext']}; font-size: 9pt; }}
    QPushButton#CopyButton {{ background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border_strong']}; border-radius: 8px; padding: 6px 14px; font-weight: 600; }}
    QPushButton#CopyButton:hover {{ background: {c['accent_soft']}; border-color: {c['accent']}; color: {c['accent']}; }}
    QFrame#MatrixTabHolder {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-bottom: 2px solid {c['accent']};
        border-top-left-radius: 10px; border-top-right-radius: 10px; }}
    QLabel#MatrixTabText {{ color: {c['accent']}; font-weight: 700; }}
    QTableWidget#ResultTable {{ background: {c['surface']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 12px;
        gridline-color: {c['divider']}; alternate-background-color: {c['surface_alt']}; }}
    QTableWidget#ResultTable::item {{ padding: 6px 10px; }}
    QTableWidget#ResultTable::item:selected {{ background: {c['sel_bg']}; color: {c['sel_text']}; }}
    QFrame#ResultNav {{ background: {c['surface_alt']}; border: 1px solid {c['border']};
        border-radius: 12px; }}
    QLabel#NavHeader {{ color: {c['subtext']}; font-weight: 800; font-size: 8.4pt;
        letter-spacing: 0.6px; padding-bottom: 3px; }}
    QLabel#NavLink {{ font-size: 9.2pt; padding: 3px 6px; border-radius: 7px; color: {c['text']}; }}
    QLabel#NavLink:hover {{ background: {c['accent_soft']}; color: {c['accent']}; }}
    QFrame#BootSubtabBar {{ background: transparent; }}
    QPushButton#BootSubtab {{ background: {c['surface_alt']}; color: {c['subtext']};
        border: 1px solid {c['border']}; border-bottom: 2px solid transparent;
        border-top-left-radius: 9px; border-top-right-radius: 9px;
        padding: 6px 14px; font-weight: 600; }}
    QPushButton#BootSubtab:hover {{ color: {c['accent']}; background: {c['accent_soft']}; }}
    QPushButton#BootSubtab:checked {{ background: {c['surface']}; color: {c['accent']};
        border-color: {c['border']}; border-bottom: 2px solid {c['accent']}; }}
    QScrollArea#HistScroll {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 12px; }}

    /* ---------- premium dialog shell ---------- */
    QDialog#PremiumDialog {{ background: transparent; }}
    QWidget#DialogBody {{ background: {c['surface']}; }}
    QFrame#DialogCard {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 18px; }}
    QFrame#DialogHeader {{ background: {c['surface_alt']};
        border-top-left-radius: 18px; border-top-right-radius: 18px;
        border-bottom: 1px solid {c['border']}; }}
    QLabel#DialogTitle {{ color: {c['text']}; font-size: 14.5pt; font-weight: 800;
        letter-spacing: -0.2px; }}
    QLabel#DialogSubtitle {{ color: {c['subtext']}; font-size: 9.5pt; }}
    QLabel#DialogIcon {{ background: {c['accent_soft']}; border-radius: 12px; }}
    QToolButton#DialogClose {{ background: transparent; border: 0; border-radius: 9px;
        color: {c['subtext']}; font-size: 14pt; padding: 2px; }}
    QToolButton#DialogClose:hover {{ background: {c['danger_soft']}; color: {c['danger']}; }}
    QFrame#DialogFooter {{ background: {c['surface_alt']};
        border-top: 1px solid {c['border']};
        border-bottom-left-radius: 18px; border-bottom-right-radius: 18px; }}
    QLabel#FieldLabel {{ color: {c['subtext']}; font-weight: 600; font-size: 9.2pt; }}
    QLabel#HintLabel {{ color: {c['muted']}; font-size: 8.6pt; }}
    QFrame#HLine {{ background: {c['divider']}; max-height: 1px; min-height: 1px; border: 0; }}

    /* ---------- native message / input dialogs ---------- */
    QMessageBox, QInputDialog {{ background: {c['surface']}; }}
    QMessageBox QLabel, QInputDialog QLabel {{ color: {c['text']}; font-size: 9.8pt; }}
    QMessageBox QPushButton, QInputDialog QPushButton {{ min-width: 94px; }}
    QFileDialog {{ background: {c['surface']}; }}

    /* ===================================================================== */
    /* Toolbar split — two captioned halves (PLS-SEM | Phi tuyến tính)        */
    /* ===================================================================== */
    QLabel#ToolbarCaption {{ color: {c['muted']}; font-size: 7.4pt; font-weight: 800;
        letter-spacing: 1.1px; padding: 0 6px 0 5px; background: transparent; }}
    QLabel#ToolbarCaptionML {{ color: {c['accent2']}; font-size: 7.4pt; font-weight: 800;
        letter-spacing: 1.1px; padding: 0 6px 0 7px; background: transparent; }}
    QFrame#ToolbarSeam {{ background: {c['border_strong']}; min-width: 2px; max-width: 2px;
        margin: 9px 8px; border-radius: 1px; }}
    QToolBar#MainToolbar QToolButton#PrimaryToolAction {{ background: {c['accent_softer']};
        border: 1px solid {c['accent_soft']}; color: {c['accent']}; font-weight: 700; }}
    QToolBar#MainToolbar QToolButton#PrimaryToolAction:hover {{ background: {c['accent']};
        color: {c['on_accent']}; border-color: {c['accent']}; }}
    QToolBar#MainToolbar QToolButton#PrimaryToolActionML {{ background: {c['accent2_softer']};
        border: 1px solid {c['accent2_soft']}; color: {c['accent2']}; font-weight: 700; }}
    QToolBar#MainToolbar QToolButton#PrimaryToolActionML:hover {{ background: {c['accent2']};
        color: #FFFFFF; border-color: {c['accent2']}; }}
    QToolBar#MainToolbar QToolButton[mlSide="true"]:hover {{ background: {c['accent2_softer']};
        color: {c['accent2']}; border-color: {c['accent2_soft']}; }}
    QToolBar#MainToolbar QToolButton[mlSide="true"]:checked {{ background: {c['accent2_soft']};
        color: {c['accent2']}; border-color: {c['accent2']}; }}
    QToolBar#MainToolbar QToolButton::menu-indicator {{ image: none; width: 0; }}

    /* ===================================================================== */
    /* Nonlinear ML workspace                                                */
    /* ===================================================================== */
    QWidget#NonlinearWorkspace {{ background: {c['bg']}; }}
    QFrame#NLNavRail {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 16px; }}
    QLabel#NLBrandTitle {{ color: {c['text']}; font-size: 12.5pt; font-weight: 800;
        letter-spacing: -0.2px; }}
    QLabel#NLBrandSub {{ color: {c['muted']}; font-size: 8pt; font-weight: 600;
        letter-spacing: 0.2px; }}
    QLabel#NLBrandChip {{ background: {c['accent2_soft']}; border-radius: 13px; }}
    QToolButton#NLNavItem {{ background: transparent; color: {c['subtext']}; border: 0;
        border-radius: 10px; padding: 9px 12px; text-align: left; font-size: 9.6pt;
        font-weight: 600; }}
    QToolButton#NLNavItem:hover {{ background: {c['surface_alt']}; color: {c['text']}; }}
    QToolButton#NLNavItem:checked {{ background: {c['accent2_soft']}; color: {c['accent2']};
        font-weight: 700; }}
    QToolButton#NLNavItem:disabled {{ color: {c['muted']}; }}
    QToolButton#NLNavSection {{ color: {c['muted']}; font-size: 7.6pt; font-weight: 800;
        letter-spacing: 1.2px; background: transparent; border: 0; padding: 2px 12px; }}

    QFrame#NLStageHeader {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 16px; }}
    QLabel#NLKicker {{ color: {c['accent2']}; font-size: 8pt; font-weight: 800;
        letter-spacing: 1.4px; }}
    QLabel#NLStageTitle {{ color: {c['text']}; font-size: 16.5pt; font-weight: 800;
        letter-spacing: -0.4px; }}
    QLabel#NLStageSub {{ color: {c['subtext']}; font-size: 9.6pt; }}

    QFrame#NLCard {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 14px; }}
    QFrame#NLConfigPanel {{ background: transparent; border: 0; }}
    QLabel#NLCardTitle {{ color: {c['text']}; font-size: 10.5pt; font-weight: 700;
        letter-spacing: -0.1px; }}
    QFrame#NLSectionTick {{ background: {c['accent2']}; border-radius: 2px; }}
    QLabel#NLSectionDesc {{ color: {c['subtext']}; font-size: 8.8pt; }}
    QLabel#NLFieldLabel {{ color: {c['subtext']}; font-size: 9.1pt; font-weight: 600; }}
    QLabel#NLFieldHint {{ color: {c['muted']}; font-size: 8.4pt; }}
    QFrame#NLRunFooter {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 14px; }}

    /* Stat cards */
    QFrame#StatCard {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 14px; }}
    QFrame#StatCard[tone="good"] {{ border: 1px solid {c['success']}; background: {c['success_soft']}; }}
    QFrame#StatCard[tone="warn"] {{ border: 1px solid {c['warning']}; background: {c['warning_soft']}; }}
    QFrame#StatCard[tone="bad"] {{ border: 1px solid {c['danger']}; background: {c['danger_soft']}; }}
    QFrame#StatCard[tone="accent"] {{ border: 1px solid {c['accent2']}; background: {c['accent2_softer']}; }}
    QLabel#StatCaption {{ color: {c['muted']}; font-size: 8.3pt; font-weight: 700; letter-spacing: 0.3px; }}
    QLabel#StatValue {{ color: {c['text']}; font-size: 16.5pt; font-weight: 800; letter-spacing: -0.4px; }}
    QLabel#StatFoot {{ color: {c['subtext']}; font-size: 8.4pt; }}

    /* Metric chips */
    QLabel#MetricChip {{ background: {c['accent2_soft']}; color: {c['accent2']};
        border-radius: 10px; padding: 3px 11px; font-size: 8.4pt; font-weight: 700; }}
    QLabel#MetricChip[tone="good"] {{ background: {c['success_soft']}; color: {c['success']}; }}
    QLabel#MetricChip[tone="warn"] {{ background: {c['warning_soft']}; color: {c['warning']}; }}
    QLabel#MetricChip[tone="bad"] {{ background: {c['danger_soft']}; color: {c['danger']}; }}
    QLabel#MetricChip[tone="muted"] {{ background: {c['surface_alt']}; color: {c['muted']}; }}
    QLabel#MetricChip[tone="info"] {{ background: {c['accent_soft']}; color: {c['accent']}; }}

    /* Equation hero card */
    QFrame#EquationCard {{ background: {c['accent2_softer']}; border: 1px solid {c['accent2_soft']};
        border-left: 3px solid {c['accent2']}; border-radius: 14px; }}
    QLabel#EquationText {{ font-family: {MONO_FAMILY}; font-size: 13pt; color: {c['text']}; }}

    /* Chart card (holds a rendered PNG) */
    QFrame#ChartCard {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 14px; }}
    QLabel#ChartTitle {{ color: {c['text']}; font-size: 10pt; font-weight: 700; }}
    QLabel#ChartImage {{ background: #FFFFFF; border-radius: 10px; }}

    /* Empty / dependency states */
    QLabel#NLEmptyTitle {{ color: {c['subtext']}; font-size: 12.5pt; font-weight: 700; }}
    QLabel#NLEmptyBody {{ color: {c['muted']}; font-size: 9.6pt; }}
    QLabel#NLEmptyChip {{ background: {c['accent2_soft']}; border-radius: 22px; }}
    QFrame#DepCard {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 14px; }}
    QFrame#DepCard[missing="true"] {{ border: 1px dashed {c['border_strong']}; background: {c['surface_sunken']}; }}
    QLabel#DepName {{ color: {c['text']}; font-size: 10.5pt; font-weight: 700; }}
    QLabel#DepPurpose {{ color: {c['subtext']}; font-size: 8.6pt; }}

    /* Nonlinear primary CTA — violet identity, used in the run footer */
    QPushButton#NLPrimaryButton {{ background: {c['accent2']}; color: #FFFFFF;
        border: 1px solid {c['accent2']}; border-radius: 11px;
        padding: 11px 22px; font-weight: 700; font-size: 10pt; }}
    QPushButton#NLPrimaryButton:hover {{ background: {c['accent2_hover']};
        border-color: {c['accent2_hover']}; }}
    QPushButton#NLPrimaryButton:pressed {{ background: {c['accent2_pressed']};
        border-color: {c['accent2_pressed']}; }}
    QPushButton#NLPrimaryButton:disabled {{ background: {c['muted']};
        border-color: {c['muted']}; color: {c['surface']}; }}

    /* Secondary / neutral action button */
    QPushButton#NLSecondaryButton {{ background: {c['surface_alt']}; color: {c['text']};
        border: 1px solid {c['border']}; border-radius: 10px;
        padding: 9px 16px; font-weight: 600; text-align: left; }}
    QPushButton#NLSecondaryButton:hover {{ background: {c['accent2_soft']};
        border-color: {c['accent2_soft']}; color: {c['accent2']}; }}
    QPushButton#NLSecondaryButton:pressed {{ background: {c['accent2_softer']}; }}
    QPushButton#NLSecondaryButton[accent="true"] {{ background: {c['accent2_softer']};
        border: 1px solid {c['accent2_soft']}; color: {c['accent2']}; font-weight: 700; }}
    QPushButton#NLSecondaryButton[accent="true"]:hover {{ background: {c['accent2_soft']}; }}

    /* Small inline text actions (Chọn tất cả / Bỏ chọn) */
    QToolButton#NLLinkButton {{ background: transparent; border: 0; color: {c['accent2']};
        font-size: 8.7pt; font-weight: 700; padding: 3px 8px; border-radius: 7px; }}
    QToolButton#NLLinkButton:hover {{ background: {c['accent2_soft']}; }}

    /* Fits / inline hint text (was a full-width pill) */
    QLabel#NLInlineHint {{ color: {c['subtext']}; font-size: 8.7pt; font-weight: 600; }}

    /* Stage intro / explainer panel — fills the results pane before a run */
    QFrame#NLIntroCard {{ background: {c['surface']}; border: 1px solid {c['border']};
        border-radius: 18px; }}
    QLabel#NLIntroChip {{ background: {c['accent2_soft']}; border-radius: 17px; }}
    QLabel#NLIntroTitle {{ color: {c['text']}; font-size: 15pt; font-weight: 800;
        letter-spacing: -0.3px; }}
    QLabel#NLIntroDesc {{ color: {c['subtext']}; font-size: 10pt; }}
    QLabel#NLIntroOutLabel {{ color: {c['accent2']}; font-size: 7.8pt; font-weight: 800;
        letter-spacing: 1.2px; }}
    QLabel#NLIntroOut {{ color: {c['text']}; font-size: 9.7pt; }}
    QFrame#NLIntroDot {{ background: {c['accent2']}; border-radius: 3px; }}
    QLabel#NLIntroCta {{ color: {c['muted']}; font-size: 9pt; }}
    QFrame#NLIntroCtaBar {{ background: {c['accent2_softer']};
        border: 1px solid {c['accent2_soft']}; border-radius: 11px; }}
    """

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import shutil
import re
import sys
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QInputDialog,
    QMenu,
    QToolButton,
    QAbstractItemView,
    QButtonGroup,
)

from core.data_manager import coerce_numeric_frame, export_cleaned_data, normalize_column_name, profile_dataset, read_dataset
from core.data_screening import screen_dataset
from core.pls_engine import ModelValidationError, PLSEngine
from core.project_store import export_project_zip, import_project_zip, load_project, new_project_state, normalize_project_state, save_project
from gui.canvas import ConnectionLine, IndicatorNode, LatentNode, ModelCanvasView
from gui.data_screening_view import DataScreeningView
from gui.data_table_model import clear_table, make_fast_table, set_dataframe
from gui.dialogs import DataImportDialog, GroupDialog, NewProjectDialog, PLSSetupDialog, PredictDialog, PremiumDialog
from gui.icons import icon
from gui.nonlinear_view import NonlinearWorkspace
from gui.results_view import BootstrapResultsWidget, PLSResultsWidget, make_report_widget
from gui import theme as ui_theme


class IndicatorListWidget(QListWidget):
    def mimeTypes(self):
        return super().mimeTypes() + ["text/plain"]

    def mimeData(self, items):
        mime_data = super().mimeData(items)
        if items:
            texts = [item.text() for item in items]
            mime_data.setText(",".join(texts))
        return mime_data


class IndicatorTableWidget(QTableWidget):
    def mimeTypes(self):
        return ["text/plain"]

    def mimeData(self, items):
        mime_data = super().mimeData(items)
        rows = sorted({item.row() for item in items})
        names = [self.item(row, 1).text() for row in rows if self.item(row, 1)]
        mime_data.setText(",".join(names))
        return mime_data


class WorkspaceTransitionOverlay(QWidget):
    finished = Signal()

    def __init__(self, old_frame: QPixmap, new_frame: QPixmap, direction: int, parent=None, duration_ms: int = 220):
        super().__init__(parent)
        self.old_frame = old_frame
        self.new_frame = new_frame
        self.direction = 1 if direction >= 0 else -1
        self.duration_ms = max(120, duration_ms)
        self.started_at = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)

    def start(self) -> None:
        self.started_at = time.perf_counter()
        self.show()
        self.raise_()
        self.timer.start()

    def _tick(self) -> None:
        progress = min(1.0, (time.perf_counter() - self.started_at) * 1000.0 / self.duration_ms)
        self.update()
        if progress >= 1.0:
            self.timer.stop()
            self.finished.emit()

    def paintEvent(self, event) -> None:
        progress = min(1.0, (time.perf_counter() - self.started_at) * 1000.0 / self.duration_ms)
        eased = 1.0 - (1.0 - progress) ** 3
        width = max(1, self.width())
        height = max(1, self.height())
        travel = min(36, max(16, int(width * 0.025)))
        old_x = int(-self.direction * travel * eased)
        new_x = int(self.direction * travel * (1.0 - eased))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.setOpacity(max(0.0, 1.0 - eased))
        painter.drawPixmap(old_x, 0, self.old_frame)
        painter.setOpacity(min(1.0, eased))
        painter.drawPixmap(new_x, 0, self.new_frame)


class HeroBanner(QFrame):
    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap(str(image_path))
        self.setObjectName("HeroBanner")
        self.setMinimumHeight(280)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor("#fff7fb"))

        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(rect.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (rect.width() - scaled.width()) // 2
            y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        veil = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
        veil.setColorAt(0.0, QColor(255, 247, 251, 238))
        veil.setColorAt(0.45, QColor(255, 247, 251, 190))
        veil.setColorAt(0.78, QColor(255, 247, 251, 35))
        painter.fillRect(rect, veil)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "Đang tính..."
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_data_view_payload(
    frame: pd.DataFrame,
    used_columns: list[str] | None = None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    used = used_columns or []
    profile, warnings = profile_dataset(frame, used)
    warnings = list(extra_warnings or []) + warnings
    profile_columns = list(profile.columns)
    profile = profile.set_index(profile_columns[0])
    display = pd.DataFrame(index=profile.index)
    display["No."] = range(1, len(profile) + 1)
    display["Missing"] = profile[profile_columns[2]]
    display["Mean"] = profile[profile_columns[4]]
    display["Median"] = profile[profile_columns[5]]
    display["Min"] = profile[profile_columns[6]]
    display["Max"] = profile[profile_columns[7]]
    display["Standard Deviation"] = profile[profile_columns[8]]
    display["Excess Kurtosis"] = profile[profile_columns[10]]
    display["Skewness"] = profile[profile_columns[9]]

    numeric, _ = coerce_numeric_frame(frame)
    max_corr_columns = 200
    if numeric.shape[1] > max_corr_columns:
        warnings.append(
            f"Correlation preview is limited to the first {max_corr_columns} variables to keep the UI responsive."
        )
    correlations = numeric.iloc[:, :max_corr_columns].corr()
    best_text = ""
    if correlations.shape[0] > 1:
        mask = np.triu(np.ones(correlations.shape, dtype=bool), k=1)
        values = correlations.where(mask).abs().stack()
        if not values.empty:
            left, right = values.idxmax()
            best_text = f"<b>Best correlation</b><br>{left} â†’ {right}: {correlations.loc[left, right]:.3f}"

    return {
        "profile": display,
        "correlations": correlations,
        "preview": frame.head(500).copy(),
        "screening": screen_dataset(frame, used),
        "warnings": warnings,
        "best_text": best_text,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing": int(frame.isna().sum().sum()),
    }


class CalculationProgressDialog(PremiumDialog):
    cancel_requested = Signal()

    def __init__(
        self,
        title: str,
        total: int,
        parent=None,
        running_label: str = "Đang bootstrapping...",
        total_label: str = "Tổng subsamples",
    ):
        super().__init__(title, "Đang chạy mô phỏng — vui lòng đợi trong giây lát.",
                         icon_name="bootstrap", parent=parent, width=560)
        self._total = max(1, int(total))
        self._started_at = time.monotonic()
        self._cancelled = False
        self._done = False
        self._running_label = running_label

        self.progress = QProgressBar()
        self.progress.setRange(0, self._total)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m mẫu  ·  %p%")
        self.progress.setFixedHeight(20)
        self.add_widget(self.progress)

        self.table = QTableWidget(7, 2)
        self.table.setObjectName("ResultTable")
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.setColumnWidth(0, 190)
        self.table.setColumnWidth(1, 300)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.add_widget(self.table)

        rows = [
            ("Trạng thái", "Đang chuẩn bị..."),
            (total_label, f"{self._total:,}"),
            ("Đã xử lý", "0"),
            ("Mẫu hợp lệ", "0"),
            ("Thời gian đã chạy", "00:00"),
            ("Ước lượng còn lại", "Đang tính..."),
            ("Tốc độ", "Đang tính..."),
        ]
        for row, (name, value) in enumerate(rows):
            key = QTableWidgetItem(name)
            key.setFont(QFont(key.font().family(), key.font().pointSize(), QFont.Weight.DemiBold))
            self.table.setItem(row, 0, key)
            self.table.setItem(row, 1, QTableWidgetItem(value))
        self.table.resizeRowsToContents()
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table_h = sum(self.table.rowHeight(r) for r in range(self.table.rowCount())) + 6
        self.table.setMinimumHeight(table_h)

        self.cancel_button = self.add_button("Hủy", "secondary", on_click=self._cancel)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(lambda: self.update_progress(self.progress.value(), self._total, self._valid_samples()))
        self._timer.start()

    def reject(self) -> None:
        # Route the header ✕ / Esc through the cancel guard instead of closing mid-run.
        self._cancel()

    def _valid_samples(self) -> int:
        item = self.table.item(3, 1)
        try:
            return int((item.text() if item else "0").replace(",", ""))
        except ValueError:
            return 0

    def _set_value(self, row: int, value: str) -> None:
        item = self.table.item(row, 1)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, 1, item)
        item.setText(value)

    def update_progress(self, completed: int, total: int, valid_samples: int) -> None:
        total = max(1, int(total))
        completed = max(0, min(int(completed), total))
        elapsed = max(0.0, time.monotonic() - self._started_at)
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (total - completed) / rate if rate > 0 and completed > 0 else None

        if self.progress.maximum() != total:
            self.progress.setMaximum(total)
        self.progress.setValue(completed)
        self._set_value(0, "Đang hủy..." if self._cancelled else self._running_label)
        self._set_value(1, f"{total:,}")
        self._set_value(2, f"{completed:,}")
        self._set_value(3, f"{int(valid_samples):,}")
        self._set_value(4, _format_duration(elapsed))
        self._set_value(5, _format_duration(remaining))
        self._set_value(6, f"{rate:.1f} mẫu/giây" if rate > 0 else "Đang tính...")

    def mark_finished(self) -> None:
        self._done = True
        self._timer.stop()
        self._set_value(0, "Hoàn tất")
        self.cancel_button.setText("Đóng")
        self.cancel_button.setEnabled(True)

    def _cancel(self) -> None:
        if self._done:
            self.accept()
            return
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self._set_value(0, "Đang hủy...")
        self.cancel_requested.emit()

    def closeEvent(self, event) -> None:
        if self._done:
            super().closeEvent(event)
            return
        if self.cancel_button.isEnabled():
            self._cancel()
            event.ignore()
            return
        super().closeEvent(event)


class CalculationWorker(QObject):
    progress = Signal(int, int, int)
    finished = Signal(dict)
    failed = Signal(str, str)

    def __init__(
        self,
        data_frame: pd.DataFrame,
        measurement: dict[str, list[str]],
        structural: list[tuple[str, str]],
        modes: dict[str, str],
        effects: list[dict[str, str]],
        settings: dict[str, Any],
    ):
        super().__init__()
        self.data_frame = data_frame.copy()
        self.measurement = measurement
        self.structural = structural
        self.modes = modes
        self.effects = effects
        self.settings = dict(settings)
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        try:
            settings = dict(self.settings)
            settings["progress_callback"] = self._progress
            engine = PLSEngine(self.data_frame)
            engine.set_model(self.measurement, self.structural, self.modes, self.effects)
            if settings.get("algorithm") == "Sum scores / OLS":
                results = engine.calculate_sum_scores(settings)
            else:
                results = engine.calculate(settings)
            self.finished.emit(results)
        except ModelValidationError as exc:
            self.failed.emit("validation", str(exc))
        except Exception as exc:
            self.failed.emit("error", str(exc))

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True

    def _progress(self, completed: int, total: int, valid_samples: int) -> bool:
        self.progress.emit(int(completed), int(total), int(valid_samples))
        return not self._cancelled


class DataLoadWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str, used_columns: list[str], data_name: str | None, existing_id: str):
        super().__init__()
        self.path = path
        self.used_columns = list(used_columns)
        self.data_name = data_name
        self.existing_id = existing_id

    @Slot()
    def run(self) -> None:
        try:
            loaded = read_dataset(self.path)
            payload = _build_data_view_payload(loaded.frame, self.used_columns, loaded.warnings)
            self.finished.emit(
                {
                    "loaded": loaded,
                    "payload": payload,
                    "data_name": self.data_name,
                    "existing_id": self.existing_id,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class AnalysisTaskWorker(QObject):
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, engine: PLSEngine, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None):
        super().__init__()
        self.engine = engine
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs or {}

    @Slot()
    def run(self) -> None:
        try:
            method = getattr(self.engine, self.method_name)
            self.finished.emit(method(*self.args, **self.kwargs))
        except ModelValidationError as exc:
            self.failed.emit("validation", str(exc))
        except Exception as exc:
            self.failed.emit("error", str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project_path = ""
        self.project_state: dict[str, Any] = new_project_state("Untitled Project")
        self.data_frame: pd.DataFrame | None = None
        self.data_path = ""
        self.current_results: dict[str, Any] | None = None
        self.workspace_dir = Path(__file__).resolve().parents[1] / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.current_model_id = ""
        self.current_data_id = ""
        self.actions: dict[str, QAction] = {}
        self.ui_language = "en"
        self.current_theme = ui_theme.DEFAULT_THEME
        self.bootstrap_widget: BootstrapResultsWidget | None = None
        self._results_tab_label = ""
        self._calculation_running = False
        self._data_load_running = False
        self._data_view_dirty = False
        self._data_payload: dict[str, Any] | None = None
        self._rendered_subtabs: set[int] = set()
        self._project_tree_limit = 300
        self._project_panel_user_visible = True
        self._indicator_panel_user_visible = True
        self._workspace_transition_overlay: WorkspaceTransitionOverlay | None = None

        self.setWindowTitle("PySmartPLS")
        self.resize(1440, 900)
        self.setMinimumSize(980, 620)
        self._init_ui()
        self._tree_refresh_timer = QTimer(self)
        self._tree_refresh_timer.setSingleShot(True)
        self._tree_refresh_timer.setInterval(250)
        self._tree_refresh_timer.timeout.connect(self.update_project_tree)
        self._apply_styles()
        self._set_project_tree_loading()
        QTimer.singleShot(400, self.update_project_tree)

    def _init_ui(self) -> None:
        self.canvas_view = ModelCanvasView()
        self.results_widget = PLSResultsWidget()

        self._build_menus()
        self._build_toolbar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("MainSplitter")
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        self.left_sidebar = self._build_left_panel()
        splitter.addWidget(self.left_sidebar)
        splitter.addWidget(self._build_center_panel())
        self.model_sidebar = self._build_right_panel()
        splitter.addWidget(self.model_sidebar)
        self.model_sidebar.setVisible(False)
        splitter.setSizes([195, 1245, 0])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self.statusBar().showMessage("Sẵn sàng")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setVisible(False)
        self._action_state_timer = QTimer(self)
        self._action_state_timer.setSingleShot(True)
        self._action_state_timer.setInterval(80)
        self._action_state_timer.timeout.connect(self._update_action_state)
        self.canvas_view.scene.selectionChanged.connect(self._action_state_timer.start)
        self.canvas_view.scene.selectionChanged.connect(self._update_style_controls_enabled)
        self.canvas_view.workspace_swipe_requested.connect(self._switch_workspace_by_gesture)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("File")
        self._menu_action(file_menu, "new_project", "Create New Project", self.new_project, "new-project")
        self._menu_action(file_menu, "new_model", "Create New Path Model", self.create_new_path_model, "new-model")
        file_menu.addSeparator()
        self._menu_action(file_menu, "save", "Save", self.save_project, "save", QKeySequence.Save)
        self._menu_action(file_menu, "save_as", "Save As...", self.save_project_as, "save-as")
        self._menu_action(file_menu, "duplicate", "Duplicate", self.duplicate_project, "duplicate", "Ctrl+D")
        file_menu.addSeparator()
        self._menu_action(file_menu, "workspace", "Switch Workspace", self.switch_workspace, "workspace")
        self._menu_action(file_menu, "archive", "Archive Project", self.archive_project, "archive")
        self._menu_action(file_menu, "restore", "Restore Project from Archive", self.restore_project, "restore")
        self._menu_action(file_menu, "active_data", "Select Active Data File", self.import_data, "data")
        file_menu.addSeparator()
        self._menu_action(file_menu, "import_backup", "Import Project from Backup File", self.import_project_backup, "import")
        self._menu_action(file_menu, "import_folder", "Import Projects from a Folder", self.import_projects_folder, "import")
        self._menu_action(file_menu, "import_data", "Import Data File", self.import_data, "import")
        self._menu_action(file_menu, "sample", "Import Sample Projects", self.load_sample_project, "import")
        file_menu.addSeparator()
        self._menu_action(file_menu, "export_project", "Export Project", self.export_project, "export")
        self._menu_action(file_menu, "export_r", "Export Model for semPLS Package in R", self.export_sempls, "export")
        self._menu_action(file_menu, "export_image", "Export as Image to File", self.export_diagram, "image")
        self._menu_action(file_menu, "copy_image", "Export as Image to Clipboard", self.copy_diagram_to_clipboard, "clipboard")
        file_menu.addSeparator()
        self._menu_action(file_menu, "print", "Print", self.print_workspace, "print", QKeySequence.Print)
        self._menu_action(file_menu, "exit", "Exit", self.close, "exit")

        edit = bar.addMenu("Edit")
        self._menu_action(edit, "copy", "Copy", self.copy_selected, "copy", QKeySequence.Copy)
        self._menu_action(edit, "paste", "Paste", self.paste_selected, "paste", QKeySequence.Paste)
        self._menu_action(edit, "select_all", "Select All", self.select_all_model, "pointer", QKeySequence.SelectAll)
        self._menu_action(edit, "delete", "Delete", self.canvas_view.delete_selected, "delete", QKeySequence.Delete)
        self._menu_action(edit, "rename", "Rename", self.rename_selected, "rename", "F2")
        edit.addSeparator()
        self._menu_action(edit, "undo", "Undo", self.canvas_view.undo, "undo", QKeySequence.Undo)
        self._menu_action(edit, "redo", "Redo", self.canvas_view.redo, "redo", QKeySequence.Redo)
        edit.addSeparator()
        self._menu_action(edit, "pointer", "Pointer", lambda: self._activate_mode("select"), "pointer")
        self._menu_action(edit, "latent", "Add Latent Variable ...", self.add_latent_variable, "latent")
        self._menu_action(edit, "latents", "Add Latent Variable(s) ...", self.add_latent_variables, "latent")
        self._menu_action(edit, "connection", "Add Connection ...", lambda: self._activate_mode("connect"), "connection")
        self._menu_action(edit, "note", "Add Note ...", self.add_note, "note")
        self._menu_action(edit, "moderating", "Add Moderating Effect ...", self.add_moderating_effect, "moderating")
        self._menu_action(edit, "quadratic", "Add Quadratic Effect ...", self.add_quadratic_effect, "quadratic")
        edit.addSeparator()
        self._menu_action(edit, "switch_mode", "Switch Between Formative/Reflective", self.switch_measurement_mode, "connection")
        for key, title in (("automatic", "Automatic"), ("mode_a", "Mode A"), ("mode_b", "Mode B"), ("sumscores", "Sumscores"), ("predefined", "Predefined")):
            self._menu_action(edit, f"weight_{key}", f"Set Indicator Weighting to '{title}'", lambda checked=False, value=key: self.set_indicator_weighting(value), "connection")
        edit.addSeparator()
        align_shortcuts = {"top": "Ctrl+I", "left": "Ctrl+J", "bottom": "Ctrl+K", "right": "Ctrl+L"}
        for side in ("right", "bottom", "left", "top"):
            self._menu_action(edit, f"align_indicators_{side}", f"Align Indicators {side.title()}", lambda checked=False, value=side: self.canvas_view.align_indicators(value), f"align-{side}", align_shortcuts[side])
        for side in ("right", "bottom", "left", "top"):
            self._menu_action(edit, f"align_selected_{side}", f"Align Selected Element {side.title()}", lambda checked=False, value=side: self.canvas_view.align_selected(value), f"align-{side}")
        self._menu_action(edit, "match_height", "Match Height", lambda: self.canvas_view.match_selected("height"), "match-height")
        self._menu_action(edit, "match_width", "Match Width", lambda: self.canvas_view.match_selected("width"), "match-width")
        edit.addSeparator()
        self._menu_action(edit, "preferences", "Preferences", self.show_preferences, "preferences")

        view = bar.addMenu("View")
        self._menu_action(view, "show_explorer", "Project Explorer", self.toggle_project_panel, "folder", checkable=True, checked=True)
        self._menu_action(view, "show_indicators", "Indicators", self.toggle_indicator_panel, "indicator", checkable=True, checked=True)
        view.addSeparator()
        self._menu_action(view, "zoom_in", "Zoom In", self.canvas_view.zoom_in, "zoom-in", QKeySequence.ZoomIn)
        self._menu_action(view, "zoom_out", "Zoom Out", self.canvas_view.zoom_out, "zoom-out", QKeySequence.ZoomOut)
        self._menu_action(view, "fit", "Fit to Window", self.canvas_view.fit_model, "fit", "Ctrl+0")

        themes = bar.addMenu("Themes")
        self._themes_menu = themes
        theme_group = QActionGroup(self)
        for key, title in (("classic", "SmartPLS Classic"), ("light", "Light"), ("dark", "Dark"), ("colorblind", "Color-blind"), ("pink", "Tóc Mây Pink")):
            action = self._menu_action(themes, f"theme_{key}", title, lambda checked=False, value=key: self.set_theme(value), "theme", checkable=True, checked=key == "classic")
            theme_group.addAction(action)

        calculate = bar.addMenu("Calculate")
        self._menu_action(calculate, "pls", "PLS Algorithm", self.run_pls_algorithm, "calculate")
        self._menu_action(calculate, "bootstrap", "Bootstrapping", self.run_bootstrapping, "bootstrap")
        implemented = {
            "ipma": self.run_ipma, "predict": self.run_predict, "blindfolding": self.run_blindfolding,
            "mga": self.run_mga, "permutation": self.run_permutation,
        }
        for key, title, image_name in (
            ("blindfolding", "Blindfolding", "blindfolding"), ("cta", "Confirmatory Tetrad Analyses (CTA)", "analysis"),
            ("ipma", "Importance-Performance Map Analysis (IPMA)", "analysis"), ("predict", "PLS Predict", "analysis"),
            ("fimix", "Finite Mixture (FIMIX) Segmentation", "analysis"), ("pos", "Prediction-Oriented Segmentation (POS)", "analysis"),
            ("mga", "Multi-Group Analysis (MGA)", "analysis"), ("permutation", "Permutation", "analysis")):
            if key in implemented:
                self._menu_action(calculate, key, title, implemented[key], image_name)
            else:
                self._menu_action(calculate, key, title, lambda checked=False, value=title: self.show_planned_analysis(value), image_name)
        consistent = calculate.addMenu(icon("plus", 16), "Consistent PLS Algorithms")
        self._menu_action(consistent, "cpls", "Consistent PLS", lambda: self.show_planned_analysis("Consistent PLS"), "analysis")
        calculate.addSeparator()
        self._menu_action(calculate, "nonlinear_ml", "Phân tích phi tuyến tính (XGBoost · SHAP · PySR)",
                          lambda: self.open_nonlinear("workspace"), "nonlinear")

        info = bar.addMenu("Info")
        self._menu_action(info, "guide", "Quick Start Guide", self.show_quick_guide, "help")
        self._menu_action(info, "about", "About PySmartPLS", self.show_about, "help")

        language = bar.addMenu("Language")
        lang_group = QActionGroup(self)
        en = self._menu_action(language, "lang_en", "English", lambda: self.set_language("en"), "language", checkable=True, checked=True)
        vi = self._menu_action(language, "lang_vi", "Tiếng Việt", lambda: self.set_language("vi"), "language", checkable=True)
        lang_group.addAction(en); lang_group.addAction(vi)

        file_menu.aboutToShow.connect(self._update_action_state)
        edit.aboutToShow.connect(self._update_action_state)
        calculate.aboutToShow.connect(self._update_action_state)

    def _build_toolbar(self) -> None:
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setObjectName("MainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setFixedHeight(62)
        self.toolbar.setContentsMargins(18, 0, 0, 0)
        self.addToolBar(self.toolbar)
        self.tool_actions: dict[str, QAction] = {}
        specs = {
            "save": ("Save", "save-toolbar", self.save_project, False),
            "new_project": ("New Project", "new-project", self.new_project, False),
            "new_model": ("New Path Model", "path-model", self.create_new_path_model, False),
            "undo": ("Undo", "undo", self.canvas_view.undo, False),
            "redo": ("Redo", "redo", self.canvas_view.redo, False),
            "zoom_out": ("Zoom Out", "zoom-out", self.canvas_view.zoom_out, False),
            "zoom_in": ("Zoom In", "zoom-in", self.canvas_view.zoom_in, False),
            "select": ("Select", "select-tool", lambda: self._activate_mode("select"), True),
            "latent": ("Latent Variable", "latent-tool", lambda: self._activate_mode("latent"), True),
            "connect": ("Connect", "connect-tool", lambda: self._activate_mode("connect"), True),
            "quadratic": ("Quadratic Effect", "quadratic-tool", self.add_quadratic_effect, False),
            "moderating": ("Moderating Effect", "moderating-tool", self.add_moderating_effect, False),
            "comment": ("Comment", "comment-tool", lambda: self._activate_mode("comment"), True),
            "calculate": ("PLS Algorithm", "calculate-tool", self.run_pls_algorithm, False),
            "bootstrap": ("Bootstrapping", "bootstrap", self.run_bootstrapping, False),
            "add_group": ("Add Data Group", "add-data-group", lambda: self.show_planned_analysis("Add Data Group"), False),
            "generate_groups": ("Generate Data Groups", "generate-data-groups", lambda: self.show_planned_analysis("Generate Data Groups"), False),
            "clear_groups": ("Clear Data Groups", "clear-data-groups", lambda: self.show_planned_analysis("Clear Data Groups"), False),
            # --- Phi tuyến tính (Nonlinear ML) half ---
            "nl_workspace": ("Phi tuyến tính", "nonlinear", lambda: self.open_nonlinear("workspace"), False),
            "nl_load": ("Nạp dữ liệu", "nl-data", lambda: self.open_nonlinear("load"), False),
            "nl_train": ("Huấn luyện", "xgboost", lambda: self.open_nonlinear("train"), False),
            "nl_shap": ("SHAP", "shap", lambda: self.open_nonlinear("shap"), False),
            "nl_symbolic": ("Hồi quy biểu thức", "symbolic", lambda: self.open_nonlinear("symbolic"), False),
            "nl_sensitivity": ("Độ nhạy Sobol", "sensitivity", lambda: self.open_nonlinear("sensitivity"), False),
            "nl_optimize": ("Tối ưu hóa", "optimize", lambda: self.open_nonlinear("optimize"), False),
            "nl_report": ("Báo cáo ML", "report", lambda: self.open_nonlinear("report"), False),
            "nl_back_model": ("Mô hình PLS", "path-model", self.go_to_canvas, False),
        }
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        for key, (title, image_name, callback, checkable) in specs.items():
            action = QAction(icon(image_name, 30), title, self)
            action.setCheckable(checkable)
            action.triggered.connect(callback)
            if checkable:
                mode_group.addAction(action)
            self.tool_actions[key] = action
        self.tool_actions["select"].setChecked(True)

        # Results-context actions (independent of the exclusive canvas mode group).
        hide_zero = QAction(icon("hide-zero", 30), "Hide Zero Values", self)
        hide_zero.setCheckable(True)
        hide_zero.setChecked(self.results_widget.hide_zeros)
        hide_zero.toggled.connect(self._report_set_hide_zeros)
        self.tool_actions["hide_zero"] = hide_zero
        for key, title, image_name, callback in (
            ("decimals_up", "Increase Decimals", "decimals-up", self._report_increase_decimals),
            ("decimals_down", "Decrease Decimals", "decimals-down", self._report_decrease_decimals),
            ("export_excel", "Export to Excel", "export-excel", self._report_export_excel),
            ("export_web", "Export to Web", "export-web", self._report_export_html),
            ("export_r_results", "Export to R", "export-r", self._report_export_r),
        ):
            action = QAction(icon(image_name, 30), title, self)
            action.triggered.connect(callback)
            self.tool_actions[key] = action

        self._set_toolbar_context("home")

    def _set_toolbar_context(self, context: str) -> None:
        if not hasattr(self, "toolbar"):
            return
        self.toolbar.clear()
        if context == "model":
            self._build_split_toolbar()
            return
        if context == "nonlinear":
            self._build_nonlinear_toolbar()
            return
        if context == "data":
            keys = ["save", "new_project", "new_model", "add_group", "generate_groups", "clear_groups"]
        elif context == "results":
            keys = ["save", "new_project", "new_model", "hide_zero", "decimals_up", "decimals_down", "export_excel", "export_web", "export_r_results"]
        else:
            keys = ["save", "new_project", "new_model"]
        for key in keys:
            self.toolbar.addAction(self.tool_actions[key])

    # ---- two-half toolbar (PLS-SEM | Phi tuyến tính) ---------------------- #
    def _tb_caption(self, text: str, ml: bool = False) -> None:
        label = QLabel(text.upper())
        label.setObjectName("ToolbarCaptionML" if ml else "ToolbarCaption")
        label.setAlignment(Qt.AlignVCenter)
        self.toolbar.addWidget(label)

    def _tb_seam(self) -> None:
        seam = QFrame()
        seam.setObjectName("ToolbarSeam")
        seam.setFrameShape(QFrame.NoFrame)
        self.toolbar.addWidget(seam)

    def _tb_action(self, key: str, *, obj_name: str = "", ml: bool = False) -> None:
        action = self.tool_actions[key]
        self.toolbar.addAction(action)
        button = self.toolbar.widgetForAction(action)
        if button is None:
            return
        if obj_name:
            button.setObjectName(obj_name)
        if ml:
            button.setProperty("mlSide", True)
        if obj_name or ml:
            button.style().unpolish(button)
            button.style().polish(button)

    def _tb_popup(self, text: str, icon_name: str, action_keys: list[str], ml: bool = False) -> None:
        button = QToolButton()
        button.setText(text)
        button.setIcon(icon(icon_name, 30))
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setPopupMode(QToolButton.InstantPopup)
        button.setCursor(Qt.PointingHandCursor)
        menu = QMenu(button)
        for key in action_keys:
            menu.addAction(self.tool_actions[key])
        button.setMenu(menu)
        if ml:
            button.setProperty("mlSide", True)
            button.style().unpolish(button)
            button.style().polish(button)
        self.toolbar.addWidget(button)

    def _build_split_toolbar(self) -> None:
        # LEFT zone — PLS-SEM (tidied via overflow popups to kill clutter)
        self._tb_caption("PLS-SEM")
        self.toolbar.addAction(self.tool_actions["save"])
        self.toolbar.addAction(self.tool_actions["new_model"])
        self.toolbar.addSeparator()
        self._tb_popup("Lịch sử", "undo", ["undo", "redo"])
        self._tb_popup("Thu phóng", "zoom-in", ["zoom_out", "zoom_in"])
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.tool_actions["select"])
        self.toolbar.addAction(self.tool_actions["latent"])
        self.toolbar.addAction(self.tool_actions["connect"])
        self._tb_popup("Hiệu ứng", "effects", ["quadratic", "moderating", "comment"])
        self.toolbar.addSeparator()
        self._tb_action("calculate", obj_name="PrimaryToolAction")
        self.toolbar.addAction(self.tool_actions["bootstrap"])
        # SEAM — the one strong divider between the two worlds
        self._tb_seam()
        # RIGHT zone — Phi tuyến tính
        self._tb_caption("Phi tuyến tính", ml=True)
        self._tb_action("nl_workspace", obj_name="PrimaryToolActionML", ml=True)
        self.toolbar.addSeparator()
        self._tb_action("nl_load", ml=True)
        self._tb_action("nl_train", ml=True)
        self._tb_popup("Phân tích", "sensitivity",
                       ["nl_shap", "nl_symbolic", "nl_sensitivity", "nl_optimize", "nl_report"], ml=True)

    def _build_nonlinear_toolbar(self) -> None:
        # When the nonlinear tab is active: file cluster + the full ML pipeline + back.
        self.toolbar.addAction(self.tool_actions["save"])
        self.toolbar.addAction(self.tool_actions["new_model"])
        self._tb_seam()
        self._tb_caption("Phi tuyến tính", ml=True)
        self._tb_action("nl_load", ml=True)
        self._tb_action("nl_train", ml=True)
        self._tb_action("nl_shap", ml=True)
        self._tb_action("nl_symbolic", ml=True)
        self._tb_action("nl_sensitivity", ml=True)
        self._tb_action("nl_optimize", ml=True)
        self.toolbar.addSeparator()
        self._tb_action("nl_report", ml=True)
        self.toolbar.addSeparator()
        self._tb_action("nl_back_model", obj_name="PrimaryToolAction")

    def _add_toolbar_action(self, toolbar: QToolBar, title: str, callback) -> QAction:
        action = QAction(title, self)
        action.triggered.connect(callback)
        toolbar.addAction(action)
        return action

    def _menu_action(self, menu: QMenu, key: str, title: str, callback, icon_name: str = "", shortcut=None, checkable: bool = False, checked: bool = False) -> QAction:
        action = QAction(icon(icon_name, 18) if icon_name else QIcon(), title, self)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        menu.addAction(action)
        self.actions[key] = action
        return action

    def _mode_action(self, title: str, mode: str, group: QActionGroup) -> QAction:
        action = QAction(title, self)
        action.setCheckable(True)
        action.triggered.connect(lambda: self.canvas_view.set_mode(mode))
        group.addAction(action)
        return action

    def _build_left_panel(self) -> QWidget:
        panel = QSplitter(Qt.Vertical)
        panel.setObjectName("LeftSidebar")
        panel.setChildrenCollapsible(False)

        self.project_panel = QFrame()
        self.project_panel.setObjectName("SidePanel")
        layout = QVBoxLayout(self.project_panel)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._panel_header("Project Explorer", "folder", [
            ("plus", self.new_project, "Create project"), ("minus", self.remove_project_entry, "Remove from explorer"), ("star", self.load_sample_project, "Sample project")
        ]))
        self.project_panel_title_text = self._last_panel_title_text
        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderHidden(True)
        self.project_tree.setObjectName("ProjectTree")
        self.project_tree.itemDoubleClicked.connect(self.open_tree_item)
        self.project_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.project_tree.customContextMenuRequested.connect(self.show_project_context_menu)
        self.project_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        layout.addWidget(self.project_tree, 1)
        panel.addWidget(self.project_panel)

        self.indicator_panel = QFrame()
        self.indicator_panel.setObjectName("SidePanel")
        indicator_layout = QVBoxLayout(self.indicator_panel)
        indicator_layout.setSpacing(0)
        indicator_layout.setContentsMargins(0, 0, 0, 0)
        indicator_layout.addWidget(self._panel_header("Indicators", "indicators", [
            ("filter", lambda: self.apply_indicator_filter("all"), "Show all"),
            ("filter-yellow", lambda: self.apply_indicator_filter("unused"), "Unused indicators"),
            ("filter-blue", lambda: self.apply_indicator_filter("used"), "Used indicators"),
        ]))
        self.indicator_panel_title_text = self._last_panel_title_text
        self.indicator_filter = QLineEdit()
        self.indicator_filter.setPlaceholderText("Search indicators...")
        self.indicator_filter.setClearButtonEnabled(True)
        self.indicator_filter.textChanged.connect(self.filter_indicators)
        self.indicator_filter.setVisible(False)
        indicator_layout.addWidget(self.indicator_filter)

        self.indicator_list = IndicatorTableWidget()
        self.indicator_list.setObjectName("IndicatorList")
        self.indicator_list.setColumnCount(2)
        self.indicator_list.setHorizontalHeaderLabels(["No.", "Indicator"])
        indicator_header = self.indicator_list.horizontalHeader()
        indicator_header.setMinimumSectionSize(42)
        indicator_header.setSectionResizeMode(0, QHeaderView.Fixed)
        indicator_header.setSectionResizeMode(1, QHeaderView.Interactive)
        self.indicator_list.setColumnWidth(0, 74 if sys.platform == "darwin" else 58)
        self.indicator_list.setColumnWidth(1, 150)
        self.indicator_list.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.indicator_list.verticalHeader().setDefaultSectionSize(32 if sys.platform == "darwin" else 28)
        self.indicator_list.verticalHeader().hide()
        self.indicator_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.indicator_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.indicator_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.indicator_list.setAlternatingRowColors(True)
        self.indicator_list.setDragEnabled(True)
        indicator_layout.addWidget(self.indicator_list, 1)
        self.best_correlation_label = QLabel()
        self.best_correlation_label.setObjectName("BestCorrelation")
        self.best_correlation_label.setAlignment(Qt.AlignCenter)
        self.best_correlation_label.setVisible(False)
        indicator_layout.addWidget(self.best_correlation_label)
        panel.addWidget(self.indicator_panel)
        panel.setSizes([430, 430])
        return panel

    def _panel_header(self, title: str, icon_name: str, buttons: list[tuple[str, Any, str]]) -> QWidget:
        header = QFrame()
        header.setObjectName("PanelHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(7, 5, 7, 5)
        row.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("PanelTitle")
        title_label.setPixmap(icon(icon_name, 24).pixmap(24, 24))
        row.addWidget(title_label)
        text = QLabel(title)
        text.setObjectName("PanelTitleText")
        self._last_panel_title_text = text
        row.addWidget(text)
        row.addStretch()
        for image_name, callback, tooltip in buttons:
            button = QToolButton()
            button.setObjectName("PanelButton")
            button.setIcon(icon(image_name, 22))
            button.setIconSize(QSize(18, 18))
            button.setFixedSize(22, 22)
            button.setToolTip(tooltip)
            button.clicked.connect(callback)
            row.addWidget(button)
        return header

    def _build_center_panel(self) -> QWidget:
        container = QFrame()
        container.setObjectName("WorkspaceContainer")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(7, 7, 2, 2)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("WorkspaceTabs")
        self.tabs.addTab(self._welcome_tab(), "Workspace")
        self.tabs.addTab(self._data_tab(), "Data")
        self.tabs.addTab(self.canvas_view, "Path Model")
        self.tabs.addTab(self._results_page(), "Results")
        self._nonlinear_placeholder = QWidget()
        self._nonlinear_placeholder.setObjectName("NonlinearWorkspace")
        self.tabs.addTab(self._nonlinear_placeholder, "Phi tuyến tính")
        self.tabs.tabBar().setVisible(False)
        outer.addWidget(self.tabs)

        # Kept off-screen because the validation/property methods use these widgets.
        self.properties_box = QTextEdit()
        self.checker_box = QTextEdit()
        return container

    def _ensure_nonlinear_workspace(self) -> NonlinearWorkspace:
        nonlinear = getattr(self, "nonlinear", None)
        if nonlinear is not None:
            return nonlinear
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.statusBar().showMessage("Đang mở không gian Phi tuyến tính...")
            QApplication.processEvents()
            nonlinear = NonlinearWorkspace(self)
            self.nonlinear = nonlinear
            self.tabs.removeTab(4)
            self.tabs.insertTab(4, nonlinear, "Phi tuyến tính")
            return nonlinear
        finally:
            QApplication.restoreOverrideCursor()

    def _results_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("ResultsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.results_empty = QLabel(
            "Chưa có kết quả. Hãy nhập dữ liệu, vẽ mô hình rồi chạy PLS Algorithm hoặc Bootstrapping."
        )
        self.results_empty.setObjectName("EmptyResult")
        self.results_empty.setAlignment(Qt.AlignCenter)
        self.results_empty.setWordWrap(True)
        layout.addWidget(self.results_empty, 1)
        self.results_tabs = QTabWidget()
        self.results_tabs.setObjectName("ReportTabs")
        self.results_tabs.setDocumentMode(True)
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.setMovable(True)
        self.results_tabs.tabCloseRequested.connect(self._close_report_tab)
        self.results_tabs.currentChanged.connect(lambda *_: self._sync_results_toolbar())
        layout.addWidget(self.results_tabs, 1)
        self._update_results_placeholder()
        return page

    def _update_results_placeholder(self) -> None:
        if not hasattr(self, "results_tabs"):
            return
        has_reports = self.results_tabs.count() > 0
        self.results_empty.setVisible(not has_reports)
        self.results_tabs.setVisible(has_reports)

    def _ensure_report(self, widget: QWidget, title: str) -> int:
        index = self.results_tabs.indexOf(widget)
        if index == -1:
            index = self.results_tabs.addTab(widget, title)
        else:
            self.results_tabs.setTabText(index, title)
        self.results_tabs.setCurrentIndex(index)
        self._update_results_placeholder()
        return index

    def _close_report_tab(self, index: int) -> None:
        widget = self.results_tabs.widget(index)
        self.results_tabs.removeTab(index)
        if widget is self.bootstrap_widget:
            self.bootstrap_widget = None
        elif widget is self.results_widget:
            self.results_widget.set_empty_state()
        self._update_results_placeholder()

    def _reset_reports(self) -> None:
        if hasattr(self, "results_tabs"):
            while self.results_tabs.count():
                self.results_tabs.removeTab(0)
        self.bootstrap_widget = None
        if hasattr(self, "results_widget"):
            self.results_widget.set_empty_state()
        self._update_results_placeholder()

    def _current_report(self):
        if hasattr(self, "results_tabs") and self.results_tabs.count():
            widget = self.results_tabs.currentWidget()
            if widget is not None:
                return widget
        return self.results_widget

    def _report_increase_decimals(self) -> None:
        self._current_report().increase_decimals()

    def _report_decrease_decimals(self) -> None:
        self._current_report().decrease_decimals()

    def _report_set_hide_zeros(self, value: bool) -> None:
        widget = self._current_report()
        if hasattr(widget, "set_hide_zeros"):
            widget.set_hide_zeros(value)

    def _report_export_excel(self) -> None:
        self._current_report().export_excel()

    def _report_export_html(self) -> None:
        self._current_report().export_html()

    def _report_export_r(self) -> None:
        self._current_report().export_r()

    def _sync_results_toolbar(self) -> None:
        widget = self._current_report()
        if "hide_zero" in getattr(self, "tool_actions", {}) and hasattr(widget, "hide_zeros"):
            action = self.tool_actions["hide_zero"]
            action.blockSignals(True)
            action.setChecked(widget.hide_zeros)
            action.blockSignals(False)

    def _welcome_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("WelcomeRoot")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        return page

    def _welcome_card(self, icon_name: str, title: str, body: str, action: str,
                      callback, primary: bool = False) -> QFrame:
        card = QFrame()
        card.setObjectName("WelcomeCard")
        card.setMinimumWidth(236)
        card.setMaximumWidth(300)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 22, 20, 20)
        layout.setSpacing(12)
        chip = QLabel()
        chip.setObjectName("WelcomeCardIcon")
        chip.setFixedSize(50, 50)
        chip.setAlignment(Qt.AlignCenter)
        chip.setPixmap(icon(icon_name, 26).pixmap(26, 26))
        layout.addWidget(chip)
        heading = QLabel(title)
        heading.setObjectName("WelcomeCardTitle")
        layout.addWidget(heading)
        text = QLabel(body)
        text.setObjectName("WelcomeCardBody")
        text.setWordWrap(True)
        layout.addWidget(text, 1)
        button = QPushButton(action)
        if primary:
            button.setObjectName("PrimaryButton")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return card

    def _home_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)

        hero = HeroBanner(self._asset_path("rapunzel_banner.png"))
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(28, 26, 28, 26)
        title = QLabel("PLS Studio Tóc Mây")
        title.setObjectName("HeroTitle")
        title.setWordWrap(True)
        subtitle = QLabel(
            "Không gian phân tích PLS-SEM tiếng Việt với giao diện hồng baby, "
            "mô hình trực quan và kết quả dễ đọc."
        )
        subtitle.setObjectName("HeroSubtitle")
        subtitle.setWordWrap(True)
        hero_buttons = QHBoxLayout()
        import_button = QPushButton("Nhập dữ liệu")
        import_button.clicked.connect(self.import_data)
        model_button = QPushButton("Vẽ mô hình")
        model_button.clicked.connect(self.go_to_canvas)
        calc_button = QPushButton("Chạy PLS")
        calc_button.clicked.connect(self.run_pls_algorithm)
        hero_buttons.addWidget(import_button)
        hero_buttons.addWidget(model_button)
        hero_buttons.addWidget(calc_button)
        hero_buttons.addStretch()
        hero_layout.addStretch()
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addLayout(hero_buttons)
        layout.addWidget(hero)

        cards = QHBoxLayout()
        cards.addWidget(self._home_card("1. Dữ liệu", "Nhập CSV/TXT/XLSX/XLS/SAV, xem trước dữ liệu, kiểm tra thiếu dữ liệu và kiểu thang đo.", self.import_data))
        cards.addWidget(self._home_card("2. Mô hình", "Tạo biến tiềm ẩn, kéo biến quan sát, vẽ đường dẫn cấu trúc và kiểm tra mô hình.", self.go_to_canvas))
        cards.addWidget(self._home_card("3. Kết quả", "Chạy PLS, bootstrap, xem độ tin cậy/giá trị, HTMT, R², hệ số đường dẫn và xuất báo cáo.", self.run_pls_algorithm))
        layout.addLayout(cards)
        layout.addStretch()
        return page

    def _home_card(self, title: str, body: str, callback) -> QFrame:
        card = QFrame()
        card.setObjectName("HomeCard")
        layout = QVBoxLayout(card)
        heading = QLabel(f"<h3>{title}</h3>")
        text = QLabel(body)
        text.setWordWrap(True)
        button = QPushButton(title.split(". ", 1)[-1])
        button.clicked.connect(callback)
        layout.addWidget(heading)
        layout.addWidget(text, 1)
        layout.addWidget(button)
        return card

    def _data_tab(self) -> QWidget:
        page = QWidget()
        page.setObjectName("DataPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 6)
        metadata = QGridLayout()
        metadata.setHorizontalSpacing(14)
        self.data_meta_labels: dict[str, QLabel] = {}
        fields = [
            ("Delimiter", "Delimiter:", "Automatic"), ("Encoding", "Encoding:", "UTF-8"),
            ("Quote", "Value Quote Character:", "Automatic"), ("Rows", "Sample size:", "0"),
            ("Format", "Number Format:", "Automatic"), ("Columns", "Indicators:", "0"),
            ("MissingMarker", "Missing Value Marker:", "None"), ("Missing", "Missing Values:", "0"),
        ]
        for index, (key, caption, value) in enumerate(fields):
            row, group = index // 2, index % 2
            column = group * 3
            caption_label = QLabel(caption)
            value_label = QLabel(value)
            value_label.setObjectName("DataMetaValue")
            self.data_meta_labels[key] = value_label
            metadata.addWidget(caption_label, row, column)
            metadata.addWidget(value_label, row, column + 1)
        metadata.setColumnMinimumWidth(0, 145)
        metadata.setColumnMinimumWidth(1, 210)
        metadata.setColumnMinimumWidth(2, 20)
        metadata.setColumnMinimumWidth(3, 120)
        metadata.setColumnMinimumWidth(4, 155)
        metadata.setColumnStretch(5, 1)
        reanalyze = QPushButton("Re-Analyze")
        reanalyze.clicked.connect(lambda: self._load_dataset(self.data_path, existing_id=self.current_data_id) if self.data_path else None)
        external = QPushButton("Open External")
        external.clicked.connect(self.open_data_external)
        metadata.addWidget(reanalyze, 0, 6)
        metadata.addWidget(external, 0, 7)
        layout.addLayout(metadata)
        self.data_warning_box = QTextEdit()
        self.data_warning_box.setReadOnly(True)
        self.data_warning_box.setMaximumHeight(54)
        self.data_warning_box.setVisible(False)
        layout.addWidget(self.data_warning_box)

        self.data_tabs = QTabWidget()
        self.profile_table = make_fast_table("ProfileTable")
        self.correlation_table = make_fast_table("CorrelationTable")
        self.preview_table = make_fast_table("PreviewTable")
        self.screening_view = DataScreeningView()
        self.data_tabs.addTab(self.profile_table, "Chỉ báo (Indicators)")
        self.data_tabs.addTab(self.correlation_table, "Tương quan")
        self.data_tabs.addTab(self.preview_table, "Dữ liệu thô")
        self.data_tabs.addTab(self.screening_view, "Sàng lọc dữ liệu")
        self.data_tabs.currentChanged.connect(self._render_data_subtab)
        layout.addWidget(self.data_tabs, 1)
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_data_table)
        layout.addWidget(copy_button, alignment=Qt.AlignRight)
        return page

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ModelSidebar")
        panel.setMinimumWidth(180)
        panel.setMaximumWidth(205)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        # Editable element controls collected here so they can be greyed out
        # when nothing is selected on the canvas (they act on the selection).
        self._style_widgets: list[QWidget] = []

        top = QHBoxLayout()
        for title, image_name, callback, tip in (
            ("Lưới", "grid", self.canvas_view.toggle_grid, "Hiện/ẩn lưới canvas"),
            ("Bắt dính", "snap", self.canvas_view.toggle_snap, "Bắt phần tử dính vào lưới"),
        ):
            button = QToolButton()
            button.setText(title); button.setIcon(icon(image_name, 30)); button.setIconSize(QSize(30, 30))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon); button.setCheckable(True)
            button.setToolTip(tip); button.setCursor(Qt.PointingHandCursor); button.toggled.connect(callback)
            top.addWidget(button)
        layout.addLayout(top)
        themes_button = QPushButton("Giao diện khác")
        themes_button.setCursor(Qt.PointingHandCursor)
        themes_button.setToolTip("Chọn giao diện (theme)")
        themes_button.clicked.connect(
            lambda: self._themes_menu.exec(themes_button.mapToGlobal(themes_button.rect().bottomLeft()))
            if getattr(self, "_themes_menu", None) is not None else None
        )
        layout.addWidget(themes_button)

        from PySide6.QtWidgets import QGraphicsOpacityEffect
        style_container = QWidget()
        style_layout = QVBoxLayout(style_container)
        style_layout.setContentsMargins(0, 0, 0, 0)
        style_layout.setSpacing(7)
        self._style_panel = style_container
        self._style_opacity = QGraphicsOpacityEffect(style_container)
        self._style_opacity.setOpacity(1.0)
        style_container.setGraphicsEffect(self._style_opacity)

        swatch_heading = QLabel("Màu phần tử"); swatch_heading.setObjectName("SidebarHeading"); swatch_heading.setAlignment(Qt.AlignCenter)
        style_layout.addWidget(swatch_heading)
        palette = QGridLayout(); palette.setSpacing(4)
        colors = [
            "#7a2b00", "#a94700", "#d96500", "#ff7900", "#ffb27a", "#ffe0c8",
            "#85000b", "#b00017", "#e00027", "#ff1744", "#ff82a2", "#ffd2e0",
            "#6c087c", "#9b0aa6", "#c20acb", "#ef33ff", "#f496ff", "#ffd5ff",
            "#110080", "#2112b5", "#3630ef", "#575cff", "#9aa8ff", "#d9e0ff",
            "#006c1c", "#159d22", "#24d82b", "#21ef43", "#8cf59e", "#d8ffe0",
            "#7d8e00", "#a9c300", "#d3f300", "#efff27", "#f5ff8a", "#fbffd5",
            "#000000", "#666666", "#999999", "#c7c7c7", "#e7e7e7", "#ffffff",
        ]
        for index, color in enumerate(colors):
            swatch = QPushButton()
            swatch.setFixedSize(27, 16)
            swatch.setCursor(Qt.PointingHandCursor)
            swatch.setToolTip(color)
            swatch.setStyleSheet(
                f"QPushButton{{background:{color}; border:1px solid rgba(0,0,0,0.18); border-radius:5px;}}"
                f"QPushButton:hover{{border:2px solid #1D6FE0;}}"
            )
            swatch.clicked.connect(lambda checked=False, value=color: self.canvas_view.set_selected_color(value))
            self._style_widgets.append(swatch)
            palette.addWidget(swatch, index // 6, index % 6)
        style_layout.addLayout(palette)

        label = QLabel("Cỡ chữ"); label.setObjectName("SidebarHeading"); label.setAlignment(Qt.AlignCenter); style_layout.addWidget(label)
        font_row = QHBoxLayout()
        for text, delta, tip in (("−1", -1, "Giảm cỡ chữ"), ("0", 0, "Đặt lại cỡ chữ"), ("+1", 1, "Tăng cỡ chữ")):
            button = QPushButton(text); button.setToolTip(tip)
            button.clicked.connect(lambda checked=False, value=delta: self.canvas_view.adjust_selected_font(delta=value))
            self._style_widgets.append(button); font_row.addWidget(button)
        style_layout.addLayout(font_row)
        style_row = QHBoxLayout()
        bold = QPushButton("Đậm"); bold.setCheckable(True); bold.setStyleSheet("font-weight:bold;"); bold.setToolTip("Chữ đậm"); bold.toggled.connect(lambda value: self.canvas_view.adjust_selected_font(bold=value))
        normal = QPushButton("Aa"); normal.setToolTip("Bỏ đậm & nghiêng"); normal.clicked.connect(lambda: (self.canvas_view.adjust_selected_font(bold=False), self.canvas_view.adjust_selected_font(italic=False)))
        italic = QPushButton("Nghiêng"); italic.setCheckable(True); italic.setStyleSheet("font-style:italic;"); italic.setToolTip("Chữ nghiêng"); italic.toggled.connect(lambda value: self.canvas_view.adjust_selected_font(italic=value))
        self._style_widgets.extend([bold, normal, italic])
        style_row.addWidget(bold); style_row.addWidget(normal); style_row.addWidget(italic); style_layout.addLayout(style_row)
        heading = QLabel("Độ dày viền"); heading.setObjectName("SidebarHeading"); heading.setAlignment(Qt.AlignCenter); style_layout.addWidget(heading)
        border_row = QHBoxLayout()
        for text, delta, tip in (("−1", -1, "Giảm độ dày viền"), ("0", 0, "Đặt lại viền"), ("+1", 1, "Tăng độ dày viền")):
            button = QPushButton(text); button.setToolTip(tip)
            button.clicked.connect(lambda checked=False, value=delta: self.canvas_view.adjust_selected_border(value))
            self._style_widgets.append(button); border_row.addWidget(button)
        style_layout.addLayout(border_row)
        heading = QLabel("Căn chỉnh"); heading.setObjectName("SidebarHeading"); heading.setAlignment(Qt.AlignCenter); style_layout.addWidget(heading)
        side_vn = {"top": "trên", "left": "trái", "bottom": "dưới", "right": "phải"}
        align_grid = QGridLayout()
        sides = [("top", 0, 0), ("left", 0, 1), ("bottom", 0, 2), ("right", 0, 3)]
        for side, row, column in sides:
            button = QToolButton(); button.setIcon(icon(f"align-{side}", 25)); button.setCursor(Qt.PointingHandCursor)
            button.setToolTip(f"Căn chỉ báo về {side_vn[side]}")
            button.clicked.connect(lambda checked=False, value=side: self.canvas_view.align_indicators(value))
            self._style_widgets.append(button); align_grid.addWidget(button, row, column)
            button2 = QToolButton(); button2.setIcon(icon(f"align-{side}", 25)); button2.setCursor(Qt.PointingHandCursor)
            button2.setToolTip(f"Căn phần tử đã chọn về {side_vn[side]}")
            button2.clicked.connect(lambda checked=False, value=side: self.canvas_view.align_selected(value))
            self._style_widgets.append(button2); align_grid.addWidget(button2, 1, column)
        style_layout.addLayout(align_grid)
        layout.addWidget(style_container)
        layout.addStretch()

        self.properties_box = QTextEdit(); self.properties_box.setVisible(False)
        self.checker_box = QTextEdit(); self.checker_box.setVisible(False)
        self._update_style_controls_enabled()
        return panel

    def _update_style_controls_enabled(self) -> None:
        """Grey out + dim the element-styling controls when nothing is selected."""
        if not hasattr(self, "_style_panel"):
            return
        has_selection = bool(self.canvas_view.scene.selectedItems())
        self._style_panel.setEnabled(has_selection)
        if hasattr(self, "_style_opacity"):
            self._style_opacity.setOpacity(1.0 if has_selection else 0.4)

    def _switch_workspace_by_gesture(self, direction: int) -> None:
        if direction == 0 or self._workspace_transition_overlay is not None:
            return
        pages = [1, 2, 3]
        current = self.tabs.currentIndex()
        if current not in pages:
            return
        next_index = pages[(pages.index(current) + (1 if direction > 0 else -1)) % len(pages)]
        self._show_workspace_tab(next_index)

    def _show_workspace_tab(self, index: int) -> None:
        self.tabs.setTabVisible(0, index == 0)
        self.tabs.tabBar().setVisible(index != 0)
        current = self.tabs.currentIndex()
        if current == index:
            self._on_tab_changed(index)
            return
        if index in {2, 4} or current in {2, 4}:
            self._animate_workspace_switch(index)
            return
        self._set_workspace_index(index)

    def _set_workspace_index(self, index: int) -> None:
        blocked = self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(index)
        self.tabs.blockSignals(blocked)
        self._on_tab_changed(index)

    def _animate_workspace_switch(self, index: int) -> None:
        if self._workspace_transition_overlay is not None:
            self._workspace_transition_overlay.deleteLater()
            self._workspace_transition_overlay = None

        target_widget = self.tabs.widget(index)
        if target_widget is None or self.width() <= 0 or self.height() <= 0:
            self._set_workspace_index(index)
            return

        old_frame = self.grab()
        direction = 1 if index > self.tabs.currentIndex() else -1
        self._set_workspace_index(index)
        new_frame = self.grab()

        overlay = WorkspaceTransitionOverlay(old_frame, new_frame, direction, self)
        overlay.setGeometry(self.rect())
        self._workspace_transition_overlay = overlay

        def cleanup() -> None:
            if self._workspace_transition_overlay is overlay:
                self._workspace_transition_overlay = None
            overlay.deleteLater()

        overlay.finished.connect(cleanup)
        overlay.start()

    def _on_tab_changed(self, index: int) -> None:
        if not hasattr(self, "model_sidebar"):
            return
        if index == 1 and self._data_view_dirty:
            self.update_data_views()
        is_model = index == 2
        is_nonlinear = index == 4
        self.model_sidebar.setVisible(is_model)
        self._sync_left_sidebar_visibility(force_hide=is_nonlinear)
        context = ("model" if is_model else "data" if index == 1
                   else "results" if index == 3 else "nonlinear" if index == 4 else "home")
        self._set_toolbar_context(context)
        if index == 3:
            self._sync_results_toolbar()

    def _sync_left_sidebar_visibility(self, *, force_hide: bool = False) -> None:
        if not hasattr(self, "project_panel"):
            return
        project_visible = bool(self._project_panel_user_visible and not force_hide)
        indicator_visible = bool(self._indicator_panel_user_visible and not force_hide)
        self.project_panel.setVisible(project_visible)
        self.indicator_panel.setVisible(indicator_visible)
        if hasattr(self, "left_sidebar"):
            self.left_sidebar.setVisible(project_visible or indicator_visible)
        for key, visible in (("show_explorer", project_visible), ("show_indicators", indicator_visible)):
            action = self.actions.get(key)
            if action is None:
                continue
            blocked = action.blockSignals(True)
            action.setChecked(visible)
            action.setEnabled(not force_hide)
            action.blockSignals(blocked)

    @staticmethod
    def _safe_filename(name: str) -> str:
        value = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
        return value or "Project"

    def _unique_project_path(self, name: str) -> Path:
        base = self.workspace_dir / f"{self._safe_filename(name)}.plsproj"
        if not base.exists():
            return base
        index = 2
        while True:
            candidate = self.workspace_dir / f"{self._safe_filename(name)}_{index}.plsproj"
            if not candidate.exists():
                return candidate
            index += 1

    def _active_model_entry(self) -> dict[str, Any] | None:
        model_id = self.current_model_id or self.project_state.get("active_model_id", "")
        return next((item for item in self.project_state.get("models", []) if item.get("id") == model_id), None)

    def _active_data_entry(self) -> dict[str, Any] | None:
        data_id = self.current_data_id or self.project_state.get("active_data_id", "")
        return next((item for item in self.project_state.get("data_files", []) if item.get("id") == data_id), None)

    def _sync_active_model(self) -> None:
        model = self._active_model_entry()
        if not model:
            return
        model["model"] = self.canvas_view.model_state()
        model["data_file_id"] = self.current_data_id or model.get("data_file_id", "")
        self.project_state["active_model_id"] = model["id"]
        self.project_state["model"] = model["model"]
        self.project_state["model_name"] = model["name"]

    def _save_current_silently(self) -> None:
        if not self.project_path:
            return
        self._sync_active_model()
        self.project_state["active_data_id"] = self.current_data_id
        active_data = self._active_data_entry()
        self.project_state["data_path"] = active_data.get("path", "") if active_data else ""
        save_project(self.project_path, self.project_state)

    def _selected_project_path(self) -> str:
        item = self.project_tree.currentItem() if hasattr(self, "project_tree") else None
        while item:
            path = item.data(0, Qt.UserRole + 1)
            if path:
                return str(path)
            item = item.parent()
        return self.project_path

    def _activate_project(self, path: str, open_default: bool = False) -> None:
        self._save_current_silently()
        state = normalize_project_state(load_project(path))
        project_dir = Path(path).parent
        for entry in state.get("data_files", []):
            raw_path = str(entry.get("path", ""))
            data_path = Path(raw_path) if raw_path else None
            if data_path is not None and not data_path.is_absolute():
                entry["path"] = str((project_dir / data_path).resolve())
        self.project_state = state
        self.project_path = path
        self.workspace_dir = project_dir if project_dir.name != "Archive" else project_dir.parent
        self.current_model_id = state.get("active_model_id", "")
        self.current_data_id = state.get("active_data_id", "")
        model = self._active_model_entry()
        if model:
            self.canvas_view.load_model_state(model.get("model", {}))
            self.current_model_id = model["id"]
            self.current_data_id = model.get("data_file_id", "") or self.current_data_id
            self._load_active_data(show_tab=False)
            if open_default:
                self._show_workspace_tab(2)
        else:
            self.canvas_view.load_model_state({"nodes": [], "connections": []})
            self.data_frame = None
            self.data_path = ""
            self.current_data_id = ""
            self._populate_indicators([])
            if open_default:
                self._show_workspace_tab(0)
        self.update_project_tree()

    def _activate_model(self, project_path: str, model_id: str) -> None:
        if project_path != self.project_path:
            self._save_current_silently()
            state = normalize_project_state(load_project(project_path))
            project_dir = Path(project_path).parent
            for entry in state.get("data_files", []):
                raw_path = str(entry.get("path", ""))
                data_path = Path(raw_path) if raw_path else None
                if data_path is not None and not data_path.is_absolute():
                    entry["path"] = str((project_dir / data_path).resolve())
            self.project_state = state
            self.project_path = project_path
            self.workspace_dir = project_dir if project_dir.name != "Archive" else project_dir.parent
        else:
            self._sync_active_model()
        model = next((item for item in self.project_state.get("models", []) if item.get("id") == model_id), None)
        if not model:
            return
        self.current_model_id = model_id
        self.project_state["active_model_id"] = model_id
        self.current_data_id = model.get("data_file_id", "")
        self.project_state["active_data_id"] = self.current_data_id
        self.canvas_view.load_model_state(model.get("model", {}))
        self.tabs.setTabText(2, f"{model.get('name', 'Path Model')}.splsm")
        self._show_workspace_tab(2)
        QTimer.singleShot(25, lambda path=project_path, mid=model_id: self._finish_model_activation(path, mid))

    def _finish_model_activation(self, project_path: str, model_id: str) -> None:
        if project_path != self.project_path or model_id != self.current_model_id:
            return
        self._load_active_data(show_tab=False)
        self._save_current_silently()
        self.update_project_tree()

    def _activate_data_file(self, project_path: str, data_id: str, show_tab: bool = True) -> None:
        if project_path != self.project_path:
            self._activate_project(project_path)
        self.current_data_id = data_id
        self.project_state["active_data_id"] = data_id
        model = self._active_model_entry()
        if model:
            model["data_file_id"] = data_id
        entry = self._active_data_entry()
        if show_tab and entry and entry.get("path") and Path(entry["path"]).exists():
            # Read + profile + screening run on a worker thread so clicking a
            # data file in the tree no longer freezes the UI; the worker's
            # callback updates the views, saves and refreshes the tree.
            self._load_dataset(entry["path"], data_name=entry.get("name"), existing_id=data_id)
            return
        self._load_active_data(show_tab=show_tab)
        self._save_current_silently()
        self.update_project_tree()

    def _load_active_data(self, show_tab: bool = False) -> None:
        entry = self._active_data_entry()
        if not entry or not entry.get("path") or not Path(entry["path"]).exists():
            self.data_frame = None
            self.data_path = ""
            self._populate_indicators([])
            self.update_data_views()
            return
        loaded = read_dataset(entry["path"])
        self.data_frame = loaded.frame
        self.data_path = loaded.path
        self._populate_indicators(self.data_frame.columns)
        if show_tab:
            self.update_data_views(extra_warnings=loaded.warnings)
        else:
            self._data_view_dirty = True
        self.tabs.setTabText(1, f"{entry.get('name', Path(loaded.path).stem)}{Path(loaded.path).suffix}")
        if show_tab:
            self._show_workspace_tab(1)

    def _activate_mode(self, mode: str) -> None:
        self._show_workspace_tab(2)
        self.canvas_view.set_mode(mode)

    def create_new_path_model(self) -> None:
        selected_path = self._selected_project_path()
        if selected_path and selected_path != self.project_path:
            self._activate_project(selected_path)
        if not self.project_path:
            self.new_project()
        if not self.project_path:
            return
        default_name = self.project_state.get("name", "Path Model")
        name, ok = QInputDialog.getText(self, "Create New Path Model", "Name:", text=default_name)
        if not ok:
            return
        self._sync_active_model()
        model_id = str(uuid.uuid4())
        model_name = name.strip() or "Path Model"
        model = {"id": model_id, "name": model_name, "data_file_id": "", "model": {"nodes": [], "connections": []}}
        self.project_state.setdefault("models", []).append(model)
        self.current_model_id = model_id
        self.current_data_id = ""
        self.project_state["active_model_id"] = model_id
        self.project_state["active_data_id"] = ""
        self.canvas_view.load_model_state(model["model"])
        self.data_frame = None
        self.data_path = ""
        self._populate_indicators([])
        self.tabs.setTabText(2, f"{model_name}.splsm")
        self._show_workspace_tab(2)
        self._save_current_silently()
        self.update_project_tree()
        self.statusBar().showMessage(f"Created path model: {model_name}")

    def duplicate_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Duplicate Project", str(self.workspace_dir / "Project copy.plsproj"), "PLS Project (*.plsproj)")
        if not path:
            return
        if not path.lower().endswith(".plsproj"):
            path += ".plsproj"
        self._sync_project_state()
        copy_state = dict(self.project_state)
        copy_state["name"] = f"{copy_state.get('name', 'Project')} Copy"
        save_project(path, copy_state)
        self.workspace_dir = Path(path).parent
        self.statusBar().showMessage(f"Duplicated project: {path}")
        self.update_project_tree()

    def switch_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Switch Workspace", str(self.workspace_dir))
        if path:
            self.workspace_dir = Path(path)
            self.update_project_tree()
            self.statusBar().showMessage(f"Workspace: {path}")

    def archive_project(self) -> None:
        if not self.project_path:
            QMessageBox.information(self, "Archive Project", "Save the project before archiving it.")
            self.save_project_as()
            if not self.project_path:
                return
        archive_dir = self.workspace_dir / "Archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        self._sync_project_state()
        target = archive_dir / f"{Path(self.project_path).stem}.zip"
        export_project_zip(str(target), self.project_state, self.data_path)
        self.update_project_tree()
        self.statusBar().showMessage(f"Archived project: {target.name}")

    def restore_project(self) -> None:
        archive_dir = self.workspace_dir / "Archive"
        path, _ = QFileDialog.getOpenFileName(self, "Restore Project from Archive", str(archive_dir), "Project Backup (*.zip)")
        if not path:
            return
        target = QFileDialog.getExistingDirectory(self, "Restore Into", str(self.workspace_dir))
        if not target:
            return
        try:
            project_json = import_project_zip(path, target)
            self._activate_project(project_json, open_default=True)
        except Exception as exc:
            QMessageBox.critical(self, "Restore Failed", str(exc))

    def import_projects_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Import Projects from a Folder", str(self.workspace_dir))
        if path:
            self.workspace_dir = Path(path)
            self.update_project_tree()
            count = len(self._project_files_for_tree())
            suffix = f" (showing first {self._project_tree_limit})" if count >= self._project_tree_limit else ""
            self.statusBar().showMessage(f"Found {count} project file(s) in {path}{suffix}")

    def _project_files_for_tree(self) -> list[Path]:
        files: list[Path] = []
        try:
            with os.scandir(self.workspace_dir) as entries:
                for entry in entries:
                    if len(files) >= self._project_tree_limit:
                        break
                    if not entry.is_file():
                        continue
                    suffix = Path(entry.name).suffix.lower()
                    if suffix in {".plsproj", ".json"}:
                        files.append(Path(entry.path))
        except OSError:
            return []
        return sorted(files, key=lambda path: path.name.lower())

    def export_sempls(self) -> None:
        measurement, structural, _ = self.canvas_view.extract_model()
        if not measurement:
            QMessageBox.warning(self, "Export Model", "Create a path model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Model for semPLS", "model_sempls.R", "R Script (*.R)")
        if not path:
            return
        # Produce a directly editable and valid data representation without relying on semPLS internals.
        lines = ["# Generated by PySmartPLS", "library(semPLS)", "", "measurement_model <- data.frame(", "  construct=c(" + ", ".join(repr(c) for c, values in measurement.items() for _ in values) + "),", "  indicator=c(" + ", ".join(repr(v) for values in measurement.values() for v in values) + ")", ")", "", "structural_model <- data.frame(", "  source=c(" + ", ".join(repr(a) for a, _ in structural) + "),", "  target=c(" + ", ".join(repr(b) for _, b in structural) + ")", ")"]
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        self.statusBar().showMessage(f"Exported R model: {path}")

    def copy_diagram_to_clipboard(self) -> None:
        image = self.canvas_view.render_image()
        if image.isNull():
            QMessageBox.information(self, "Copy Diagram", "The model is empty.")
            return
        QApplication.clipboard().setImage(image)
        self.statusBar().showMessage("Model image copied to clipboard")

    def print_workspace(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Print to PDF", "PySmartPLS-model.pdf", "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            from PySide6.QtGui import QPageSize, QPdfWriter
            writer = QPdfWriter(path)
            writer.setPageSize(QPageSize(QPageSize.A4))
            painter = QPainter(writer)
            self.canvas_view.scene.render(painter)
            painter.end()
            self.statusBar().showMessage(f"Printed model to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Print Failed", str(exc))

    def copy_selected(self) -> None:
        self.canvas_view.copy_selected()

    def paste_selected(self) -> None:
        self._show_workspace_tab(2)
        self.canvas_view.paste_selected()

    def select_all_model(self) -> None:
        self._show_workspace_tab(2)
        self.canvas_view.select_all_items()

    def rename_selected(self) -> None:
        if not self.canvas_view.rename_selected():
            QMessageBox.information(self, "Rename", "Select a model element first.")

    def add_latent_variable(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Latent Variable", "Name:", text="Latent Variable")
        if ok and name.strip():
            self._show_workspace_tab(2)
            self.canvas_view.add_latent_at_center(name.strip())

    def add_latent_variables(self) -> None:
        value, ok = QInputDialog.getMultiLineText(self, "Add Latent Variables", "One name per line:", "Construct 1\nConstruct 2")
        if ok:
            names = [item.strip() for item in value.splitlines() if item.strip()]
            self._show_workspace_tab(2)
            self.canvas_view.add_latents(names)

    def add_note(self) -> None:
        value, ok = QInputDialog.getMultiLineText(self, "Add Note", "Text:", "Hypothesis note")
        if ok and value.strip():
            self._show_workspace_tab(2)
            self.canvas_view.add_note_at_center(value.strip())

    def add_moderating_effect(self) -> None:
        self._show_workspace_tab(2)
        self.canvas_view.add_effect("moderating")

    def add_quadratic_effect(self) -> None:
        self._show_workspace_tab(2)
        self.canvas_view.add_effect("quadratic")

    def switch_measurement_mode(self) -> None:
        if not self.canvas_view.switch_selected_modes():
            QMessageBox.information(self, "Measurement Mode", "Select at least one latent variable.")

    def set_indicator_weighting(self, weighting: str) -> None:
        if not self.canvas_view.set_selected_weighting(weighting):
            QMessageBox.information(self, "Indicator Weighting", "Select at least one latent variable.")

    def show_preferences(self) -> None:
        QMessageBox.information(self, "Preferences", "Preferences are applied immediately through View, Themes and Language menus.")

    def show_planned_analysis(self, title: str) -> None:
        QMessageBox.information(self, title, f"{title} is visible for SmartPLS-compatible workflow planning. The current engine does not calculate this method yet.")

    def show_quick_guide(self) -> None:
        QMessageBox.information(
            self,
            "Quick Start Guide",
            "1. Create a project\n"
            "2. Import a data file\n"
            "3. Create a path model\n"
            "4. Drag indicators to the model\n"
            "5. Connect constructs\n"
            "6. Run PLS Algorithm\n\n"
            "Trackpad on Path Model:\n"
            "- Two-finger scroll pans the canvas\n"
            "- Pinch zooms under the pointer\n"
            "- Smart zoom / double-click empty canvas toggles fit and 100%\n"
            "- Horizontal swipe switches Data, Path Model and Results\n"
            "- Vertical swipe up fits the model; down resets zoom",
        )

    def show_about(self) -> None:
        QMessageBox.about(self, "About PySmartPLS", "PySmartPLS\nVisual PLS-SEM workbench built with Python and PySide6.")

    def set_language(self, language: str) -> None:
        self.ui_language = language
        menu_titles = ["Tệp", "Chỉnh sửa", "Hiển thị", "Giao diện", "Tính toán", "Thông tin", "Ngôn ngữ"] if language == "vi" else ["File", "Edit", "View", "Themes", "Calculate", "Info", "Language"]
        for action, title in zip(self.menuBar().actions(), menu_titles):
            action.setText(title)
        project_title = "Trình quản lý dự án" if language == "vi" else "Project Explorer"
        indicator_title = "Biến quan sát" if language == "vi" else "Indicators"
        if hasattr(self, "project_panel_title_text"):
            self.project_panel_title_text.setText(project_title)
        if hasattr(self, "indicator_panel_title_text"):
            self.indicator_panel_title_text.setText(indicator_title)
        tab_titles = ["Không gian làm việc", "Dữ liệu", "Mô hình đường dẫn", "Kết quả"] if language == "vi" else ["Workspace", "Data", "Path Model", "Results"]
        for index, title in enumerate(tab_titles):
            self.tabs.setTabText(index, title)
        # Keep document tab names that reflect loaded data / model / run.
        if self.data_path:
            self.tabs.setTabText(1, f"{Path(self.data_path).stem}{Path(self.data_path).suffix}")
        model_entry = self._active_model_entry()
        if model_entry:
            self.tabs.setTabText(2, f"{model_entry.get('name', 'Path Model')}.splsm")
        if getattr(self, "_results_tab_label", ""):
            self.tabs.setTabText(3, self._results_tab_label)
        toolbar_titles = ["Lưu", "Dự án mới", "Mô hình mới"] if language == "vi" else ["Save", "New Project", "New Path Model"]
        toolbar = self.findChild(QToolBar, "MainToolbar")
        if toolbar:
            for action, title in zip(toolbar.actions(), toolbar_titles):
                action.setText(title)
        results_titles = {
            "hide_zero": ("Hide Zero Values", "Ẩn giá trị 0"),
            "decimals_up": ("Increase Decimals", "Tăng số thập phân"),
            "decimals_down": ("Decrease Decimals", "Giảm số thập phân"),
            "export_excel": ("Export to Excel", "Xuất ra Excel"),
            "export_web": ("Export to Web", "Xuất ra Web"),
            "export_r_results": ("Export to R", "Xuất ra R"),
        }
        index = 1 if language == "vi" else 0
        for key, titles in results_titles.items():
            if key in self.tool_actions:
                self.tool_actions[key].setText(titles[index])
        if hasattr(self, "results_widget"):
            self.results_widget.retranslate(language)
        if hasattr(self, "results_tabs"):
            for index in range(self.results_tabs.count()):
                widget = self.results_tabs.widget(index)
                if isinstance(widget, BootstrapResultsWidget):
                    widget.retranslate(language)
        elif self.bootstrap_widget is not None:
            self.bootstrap_widget.retranslate(language)
        if hasattr(self, "results_empty"):
            self.results_empty.setText(
                "Chưa có kết quả. Hãy nhập dữ liệu, vẽ mô hình rồi chạy PLS Algorithm hoặc Bootstrapping."
                if language == "vi"
                else "No results yet. Import data, draw the model, then run the PLS Algorithm or Bootstrapping."
            )
        message = "Đã chuyển điều hướng sang tiếng Việt." if language == "vi" else "Navigation language changed to English."
        self.statusBar().showMessage(message)

    def set_theme(self, theme: str) -> None:
        self.setProperty("theme", theme)
        self._apply_styles(theme)

    def toggle_project_panel(self, checked: bool) -> None:
        self._project_panel_user_visible = checked
        self._sync_left_sidebar_visibility(force_hide=self.tabs.currentIndex() == 4)

    def toggle_indicator_panel(self, checked: bool) -> None:
        self._indicator_panel_user_visible = checked
        self._sync_left_sidebar_visibility(force_hide=self.tabs.currentIndex() == 4)

    def apply_indicator_filter(self, mode: str) -> None:
        used = set(self.canvas_view.used_indicators())
        for row in range(self.indicator_list.rowCount()):
            item = self.indicator_list.item(row, 1)
            is_used = bool(item and item.text() in used)
            self.indicator_list.setRowHidden(row, (mode == "used" and not is_used) or (mode == "unused" and is_used))
        self.statusBar().showMessage({"all": "Showing all indicators", "used": "Showing used indicators", "unused": "Showing unused indicators"}[mode])

    def remove_project_entry(self) -> None:
        item = self.project_tree.currentItem()
        if item:
            self._delete_tree_item(item)

    def open_tree_item(self, item: QTreeWidgetItem, column: int) -> None:
        item_type = item.data(0, Qt.UserRole + 2)
        path = item.data(0, Qt.UserRole + 1)
        item_id = item.data(0, Qt.UserRole + 3)
        if item_type == "project" and path:
            is_expanded = item.isExpanded()
            self._activate_project(str(path))
            for i in range(self.project_tree.topLevelItemCount()):
                top_item = self.project_tree.topLevelItem(i)
                if top_item.data(0, Qt.UserRole + 1) == path:
                    top_item.setExpanded(not is_expanded)
                    break
        elif item_type == "import-data" and path:
            self._activate_project(str(path))
            self.import_data()
        elif item_type == "model" and path and item_id:
            self._activate_model(str(path), str(item_id))
        elif item_type == "data" and path and item_id:
            self._activate_data_file(str(path), str(item_id), show_tab=True)

    def show_project_context_menu(self, position) -> None:
        item = self.project_tree.itemAt(position)
        if not item:
            return
        self.project_tree.setCurrentItem(item)
        item_type = item.data(0, Qt.UserRole + 2)
        menu = QMenu(self)
        if item_type in {"project", "model", "data"}:
            menu.addAction(icon("rename", 17), "Rename", lambda: self._rename_tree_item(item)).setShortcut("F2")
            menu.addAction(icon("copy", 17), "Copy", lambda: QApplication.clipboard().setText(item.text(0))).setShortcut("Ctrl+C")
            paste = menu.addAction(icon("paste", 17), "Paste"); paste.setEnabled(False); paste.setShortcut("Ctrl+V")
            menu.addAction(icon("duplicate", 17), "Duplicate", lambda: self._duplicate_tree_item(item)).setShortcut("Ctrl+D")
            menu.addSeparator()
            menu.addAction(icon("delete", 17), "Delete", lambda: self._delete_tree_item(item)).setShortcut("Delete")
        if item_type == "project":
            menu.addAction(icon("archive", 17), "Archive Project", lambda: self._archive_tree_project(item))
        if item_type in {"project", "model", "import-data"}:
            menu.addSeparator()
            menu.addAction(icon("new-project", 17), "Create New Project", self.new_project)
            menu.addAction(icon("path-model", 17), "Create New Path Model", self.create_new_path_model)
            menu.addSeparator()
            menu.addAction(icon("import", 17), "Import Project from Backup File", self.import_project_backup)
            menu.addAction(icon("import", 17), "Import Projects from a Folder", self.import_projects_folder)
            menu.addAction(icon("import", 17), "Import Data File", self.import_data)
            menu.addAction(icon("import", 17), "Import Sample Projects", self.load_sample_project)
            menu.addSeparator()
            menu.addAction(icon("export", 17), "Export Project", self.export_project)
            menu.addAction(icon("export", 17), "Export Model for SemPLS Package in R", self.export_sempls)
        if item_type == "data":
            menu.addSeparator()
            menu.addAction(icon("data-green", 17), "Select as Active Data File", lambda: self.open_tree_item(item, 0))
            menu.addAction(icon("export", 17), "Open External", lambda: self._open_external_tree_data(item))
        menu.exec(self.project_tree.viewport().mapToGlobal(position))

    def _rename_tree_item(self, item: QTreeWidgetItem) -> None:
        item_type = item.data(0, Qt.UserRole + 2)
        path = str(item.data(0, Qt.UserRole + 1) or "")
        item_id = str(item.data(0, Qt.UserRole + 3) or "")
        if path and path != self.project_path:
            self._activate_project(path)
        current_name = self.project_state.get("name", "Project") if item_type == "project" else item.text(0).split(" [", 1)[0]
        name, ok = QInputDialog.getText(self, "Rename", "Name:", text=current_name)
        if not ok or not name.strip():
            return
        if item_type == "project":
            self.project_state["name"] = name.strip()
        elif item_type == "model":
            model = next((entry for entry in self.project_state.get("models", []) if entry.get("id") == item_id), None)
            if model: model["name"] = name.strip()
        elif item_type == "data":
            data = next((entry for entry in self.project_state.get("data_files", []) if entry.get("id") == item_id), None)
            if data: data["name"] = name.strip()
        self._save_current_silently(); self.update_project_tree()

    def _open_external_tree_data(self, item: QTreeWidgetItem) -> None:
        path = str(item.data(0, Qt.UserRole + 1) or "")
        data_id = str(item.data(0, Qt.UserRole + 3) or "")
        self._activate_data_file(path, data_id, show_tab=False)
        self.open_data_external()

    def _duplicate_tree_item(self, item: QTreeWidgetItem) -> None:
        item_type = item.data(0, Qt.UserRole + 2)
        path = str(item.data(0, Qt.UserRole + 1) or "")
        item_id = str(item.data(0, Qt.UserRole + 3) or "")
        if path and path != self.project_path:
            self._activate_project(path)
        if item_type == "project":
            self.duplicate_project()
        elif item_type == "model":
            self._sync_active_model()
            source = next((entry for entry in self.project_state.get("models", []) if entry.get("id") == item_id), None)
            if source:
                import copy
                duplicate = copy.deepcopy(source); duplicate["id"] = str(uuid.uuid4()); duplicate["name"] += " Copy"
                self.project_state["models"].append(duplicate); self._save_current_silently(); self.update_project_tree()
        elif item_type == "data":
            source = next((entry for entry in self.project_state.get("data_files", []) if entry.get("id") == item_id), None)
            if source:
                duplicate = dict(source); duplicate["id"] = str(uuid.uuid4()); duplicate["name"] += " Copy"
                self.project_state["data_files"].append(duplicate); self._save_current_silently(); self.update_project_tree()

    def _delete_tree_item(self, item: QTreeWidgetItem) -> None:
        item_type = item.data(0, Qt.UserRole + 2)
        path = str(item.data(0, Qt.UserRole + 1) or "")
        item_id = str(item.data(0, Qt.UserRole + 3) or "")
        if QMessageBox.question(self, "Delete", f"Delete '{item.text(0)}'?") != QMessageBox.Yes:
            return
        if item_type == "project" and path:
            if path == self.project_path:
                self.project_path = ""; self.project_state = new_project_state("Untitled Project"); self.current_model_id = ""; self.current_data_id = ""
            Path(path).unlink(missing_ok=True)
        else:
            if path and path != self.project_path: self._activate_project(path)
            if item_type == "model":
                self.project_state["models"] = [entry for entry in self.project_state.get("models", []) if entry.get("id") != item_id]
                if self.current_model_id == item_id:
                    self.current_model_id = ""; self.canvas_view.load_model_state({"nodes": [], "connections": []})
            if item_type == "data":
                self.project_state["data_files"] = [entry for entry in self.project_state.get("data_files", []) if entry.get("id") != item_id]
                for model in self.project_state.get("models", []):
                    if model.get("data_file_id") == item_id: model["data_file_id"] = ""
                if self.current_data_id == item_id:
                    self.current_data_id = ""; self.data_frame = None; self.data_path = ""; self._populate_indicators([]); self.update_data_views()
            self._save_current_silently()
        self.update_project_tree()

    def _archive_tree_project(self, item: QTreeWidgetItem) -> None:
        path = str(item.data(0, Qt.UserRole + 1) or "")
        if path and path != self.project_path:
            self._activate_project(path)
        self.archive_project()

    def _update_action_state(self) -> None:
        if not self.actions:
            return
        selected = self.canvas_view.scene.selectedItems()
        has_selection = bool(selected)
        has_nodes = self.canvas_view.has_model_nodes()
        has_model = self.canvas_view.has_latent_constructs()
        for key in ("copy", "delete", "rename", "switch_mode", "weight_automatic", "weight_mode_a", "weight_mode_b", "weight_sumscores", "weight_predefined"):
            if key in self.actions:
                self.actions[key].setEnabled(has_selection)
        for key in ("export_r", "export_image", "copy_image", "print"):
            if key in self.actions:
                self.actions[key].setEnabled(has_nodes)
        for key in ("pls", "bootstrap", "ipma", "predict", "blindfolding", "mga", "permutation"):
            if key in self.actions:
                self.actions[key].setEnabled(has_model and self.data_frame is not None)
        if hasattr(self, "tool_actions"):
            for key in ("calculate", "bootstrap"):
                if key in self.tool_actions:
                    self.tool_actions[key].setEnabled(has_model and self.data_frame is not None)
        if getattr(self, "_calculation_running", False):
            self._set_calculation_actions_enabled(False)
        for key in ("cta", "fimix", "pos", "cpls"):
            if key in self.actions:
                self.actions[key].setEnabled(False)

    def new_project(self) -> None:
        dialog = NewProjectDialog(self)
        if not dialog.exec():
            return
        self._save_current_silently()
        name = dialog.project_name()
        self.project_state = new_project_state(name)
        self.project_path = str(self._unique_project_path(name))
        self.current_model_id = ""
        self.current_data_id = ""
        self.data_frame = None
        self.data_path = ""
        self.current_results = None
        self._populate_indicators([])
        self.canvas_view.load_model_state({"nodes": [], "connections": []})
        self._reset_reports()
        self.update_data_views()
        save_project(self.project_path, self.project_state)
        self.update_project_tree()
        self.update_properties_panel()
        self._show_workspace_tab(0)
        self.statusBar().showMessage("Đã tạo dự án mới")

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Mở dự án", "", "Dự án PLS (*.plsproj *.json)")
        if not path:
            return
        self._open_project_path(path)

    def _open_project_path(self, path: str) -> None:
        try:
            self._activate_project(path, open_default=True)
            self.statusBar().showMessage(f"Đã mở dự án: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Không mở được dự án", str(exc))

    def import_project_backup(self) -> None:
        zip_path, _ = QFileDialog.getOpenFileName(self, "Nhập bản sao dự án", "", "Bản sao dự án (*.zip)")
        if not zip_path:
            return
        target_dir = QFileDialog.getExistingDirectory(self, "Chọn thư mục giải nén")
        if not target_dir:
            return
        try:
            project_json = import_project_zip(zip_path, target_dir)
            self._activate_project(project_json, open_default=True)
            self.statusBar().showMessage(f"Đã nhập bản sao dự án: {zip_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Nhập dự án thất bại", str(exc))

    def save_project(self) -> None:
        if not self.project_path:
            self.save_project_as()
            return
        try:
            self._save_current_silently()
            self.update_project_tree()
            self.statusBar().showMessage(f"Đã lưu dự án: {self.project_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Không lưu được dự án", str(exc))

    def save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Lưu dự án", "", "Dự án PLS (*.plsproj)")
        if not path:
            return
        if not path.lower().endswith(".plsproj"):
            path += ".plsproj"
        self.project_path = path
        self.workspace_dir = Path(path).parent
        self._save_current_silently()
        self.update_project_tree()

    def export_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Xuất bản sao dự án", "", "Bản sao dự án (*.zip)")
        if not path:
            return
        self._sync_project_state()
        try:
            export_project_zip(path, self.project_state, self.data_path)
            self.statusBar().showMessage(f"Đã xuất dự án: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Xuất dự án thất bại", str(exc))

    def import_data(self) -> None:
        selected_item = self.project_tree.currentItem() if hasattr(self, "project_tree") else None
        selected_type = selected_item.data(0, Qt.UserRole + 2) if selected_item else ""
        selected_model_id = str(selected_item.data(0, Qt.UserRole + 3) or "") if selected_item else ""
        selected_path = self._selected_project_path()
        if selected_path and selected_path != self.project_path:
            self._activate_project(selected_path)
        if selected_type == "model" and selected_model_id:
            self._activate_model(selected_path or self.project_path, selected_model_id)
        if not self.project_path:
            self.new_project()
        if not self.project_path:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Nhập dữ liệu",
            "",
            "Tệp dữ liệu (*.csv *.txt *.xlsx *.xls *.sav);;Tất cả tệp (*)",
        )
        if not path:
            return
        dialog = DataImportDialog(path, self)
        if dialog.exec():
            self._load_dataset(path, data_name=dialog.data_name())

    def _load_dataset(self, path: str, data_name: str | None = None, existing_id: str = "") -> None:
        if self._data_load_running:
            QMessageBox.information(self, "Import Data", "A data file is already loading. Please wait for it to finish.")
            return
        self._data_load_running = True
        used = self.canvas_view.used_indicators() if hasattr(self, "canvas_view") else []
        self.statusBar().showMessage(f"Loading data: {Path(path).name}...")
        self._data_thread = QThread(self)
        self._data_worker = DataLoadWorker(path, used, data_name, existing_id)
        self._data_worker.moveToThread(self._data_thread)
        self._data_thread.started.connect(self._data_worker.run)
        self._data_worker.finished.connect(self._on_dataset_loaded)
        self._data_worker.failed.connect(self._on_dataset_load_failed)
        self._data_worker.finished.connect(self._data_thread.quit)
        self._data_worker.failed.connect(self._data_thread.quit)
        self._data_thread.finished.connect(self._data_worker.deleteLater)
        self._data_thread.finished.connect(self._data_thread.deleteLater)
        self._data_thread.start()

    @Slot(object)
    def _on_dataset_loaded(self, result: dict[str, Any]) -> None:
        try:
            loaded = result["loaded"]
            data_name = result.get("data_name")
            existing_id = result.get("existing_id", "")
            self.data_frame = loaded.frame
            self.data_path = loaded.path
            entry = next((item for item in self.project_state.get("data_files", []) if item.get("id") == existing_id), None)
            if entry:
                entry.update({"name": data_name or entry.get("name", Path(loaded.path).stem), "path": loaded.path, "rows": len(loaded.frame), "columns": len(loaded.frame.columns)})
            else:
                entry = {"id": str(uuid.uuid4()), "name": data_name or Path(loaded.path).stem, "path": loaded.path, "rows": len(loaded.frame), "columns": len(loaded.frame.columns)}
                self.project_state.setdefault("data_files", []).append(entry)
            self.current_data_id = entry["id"]
            self.project_state["active_data_id"] = entry["id"]
            model = self._active_model_entry()
            if model:
                model["data_file_id"] = entry["id"]
            self._populate_indicators(self.data_frame.columns)
            self.update_data_views(payload=result.get("payload"))
            self.tabs.setTabText(1, f"{entry['name']}{Path(loaded.path).suffix}")
            self._save_current_silently()
            self.update_project_tree()
            self._show_workspace_tab(1)
            self.statusBar().showMessage(f"Imported data: {Path(loaded.path).name} ({self.data_frame.shape[0]} rows)")
        except Exception as exc:
            QMessageBox.critical(self, "Import Data Failed", str(exc))
        finally:
            self._finish_data_load()

    @Slot(str)
    def _on_dataset_load_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Import Data Failed", message)
        self.statusBar().showMessage("Import data failed")
        self._finish_data_load()

    def _finish_data_load(self) -> None:
        self._data_load_running = False
        self._data_thread = None
        self._data_worker = None

    def _populate_indicators(self, columns) -> None:
        names = [normalize_column_name(column) for column in columns]
        self.indicator_list.setRowCount(len(names))
        for row, name in enumerate(names):
            number = QTableWidgetItem(str(row + 1))
            number.setTextAlignment(Qt.AlignCenter)
            indicator = QTableWidgetItem(name)
            self.indicator_list.setItem(row, 0, number)
            self.indicator_list.setItem(row, 1, indicator)

    def filter_indicators(self, text: str) -> None:
        text = text.strip().lower()
        for row in range(self.indicator_list.rowCount()):
            item = self.indicator_list.item(row, 1)
            self.indicator_list.setRowHidden(row, bool(item and text not in item.text().lower()))

    def indicator_filter_clear(self) -> None:
        self.indicator_filter.clear()

    def update_data_views(self, extra_warnings: list[str] | None = None, payload: dict[str, Any] | None = None) -> None:
        if self.data_frame is None:
            clear_table(self.profile_table)
            clear_table(self.correlation_table)
            clear_table(self.preview_table)
            self.screening_view.clear()
            self._data_payload = None
            self._rendered_subtabs.clear()
            self.data_warning_box.setPlainText("Chưa có dữ liệu.")
            self.best_correlation_label.setVisible(False)
            for key, value in (("Rows", "0"), ("Columns", "0"), ("Missing", "0")):
                if key in self.data_meta_labels:
                    self.data_meta_labels[key].setText(value)
            self._data_view_dirty = False
            return

        # Profiling, correlations and screening are computed in a worker thread
        # (see DataLoadWorker); when that has not happened yet (e.g. data loaded
        # silently for a calculation) build the payload here on demand.
        if payload is None:
            used = self.canvas_view.used_indicators() if hasattr(self, "canvas_view") else []
            payload = _build_data_view_payload(self.data_frame, used, extra_warnings)

        self._data_payload = payload
        self._rendered_subtabs.clear()

        best_text = payload.get("best_text", "")
        self.best_correlation_label.setText(best_text)
        self.best_correlation_label.setVisible(bool(best_text))
        self.data_meta_labels["Delimiter"].setText("Automatic")
        self.data_meta_labels["Encoding"].setText("UTF-8")
        self.data_meta_labels["Quote"].setText("Automatic")
        self.data_meta_labels["Rows"].setText(str(payload.get("rows", len(self.data_frame))))
        self.data_meta_labels["Format"].setText("Automatic")
        self.data_meta_labels["Columns"].setText(str(payload.get("columns", len(self.data_frame.columns))))
        self.data_meta_labels["MissingMarker"].setText("None")
        self.data_meta_labels["Missing"].setText(str(payload.get("missing", int(self.data_frame.isna().sum().sum()))))
        warnings = payload.get("warnings", [])
        self.data_warning_box.setPlainText("\n".join(f"- {warning}" for warning in warnings) if warnings else "Không có cảnh báo dữ liệu.")

        # Only render the visible sub-tab now; the rest render lazily on click.
        self._render_data_subtab(self.data_tabs.currentIndex())
        self._data_view_dirty = False

    def _render_data_subtab(self, index: int) -> None:
        payload = self._data_payload
        if payload is None or index in self._rendered_subtabs:
            return
        if index == 0:
            set_dataframe(self.profile_table, payload["profile"], show_index=True, decimals=4)
        elif index == 1:
            # Correlation matrix is read in variable order; skip the sort proxy
            # so a large matrix renders instantly.
            set_dataframe(self.correlation_table, payload["correlations"], show_index=True,
                          decimals=3, sortable=False)
        elif index == 2:
            # Raw file view mirrors the source rows; sorting it is unnecessary
            # and a sort proxy is the slowest part of a wide preview.
            set_dataframe(self.preview_table, payload["preview"], show_index=False,
                          decimals=4, sortable=False)
        elif index == 3:
            self.screening_view.set_result(payload.get("screening"))
        else:
            return
        self._rendered_subtabs.add(index)

    def copy_data_table(self) -> None:
        if self.data_frame is None:
            return
        QApplication.clipboard().setText(self.data_frame.to_csv(index=False, sep="\t"))

    def open_data_external(self) -> None:
        if self.data_path and Path(self.data_path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.data_path))

    def export_cleaned_dataset(self) -> None:
        if self.data_frame is None:
            QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy nhập dữ liệu trước.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Xuất dữ liệu đã làm sạch", "", "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        try:
            numeric, warnings = coerce_numeric_frame(self.data_frame)
            export_cleaned_data(numeric, path)
            if warnings:
                QMessageBox.information(self, "Đã xuất kèm cảnh báo", "\n".join(warnings[:20]))
            self.statusBar().showMessage(f"Đã xuất dữ liệu sạch: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Xuất dữ liệu thất bại", str(exc))

    def go_to_canvas(self) -> None:
        self._show_workspace_tab(2)

    def open_nonlinear(self, step: str = "workspace") -> None:
        """Open the Phi tuyến tính (Nonlinear ML) workspace at the given stage."""
        if self.data_frame is None:
            self._load_active_data(show_tab=False)
        name = ""
        entry = self._active_data_entry()
        if entry:
            name = entry.get("name", "")
        nonlinear = self._ensure_nonlinear_workspace()
        nonlinear.set_app_data(self.data_frame, name)
        self._show_workspace_tab(4)
        nonlinear.go_to_stage(step)

    def run_pls_algorithm(self) -> None:
        self.run_calculate(bootstrap=False)

    def run_bootstrapping(self) -> None:
        self.run_calculate(bootstrap=True)

    def run_calculate(self, bootstrap: bool = False) -> None:
        if getattr(self, "_calculation_running", False):
            QMessageBox.information(self, "Đang tính toán", "Một lần tính toán đang chạy. Hãy chờ hoàn tất hoặc hủy trước khi chạy tiếp.")
            return
        if self.data_frame is None:
            self._load_active_data(show_tab=False)
        if self.data_frame is None:
            QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy nhập dữ liệu trước khi chạy thuật toán.")
            return

        self.data_frame = self.data_frame.rename(columns=normalize_column_name)
        measurement, structural, modes = self.canvas_view.extract_model()
        effects = self.canvas_view.extract_effects()
        engine = PLSEngine(self.data_frame)
        engine.set_model(measurement, structural, modes, effects)
        errors, warnings = engine.validate_model(self.data_frame)
        self.show_checker_messages(errors, warnings)
        if errors:
            QMessageBox.warning(self, "Kiểm tra mô hình", "\n".join(errors))
            return
        if warnings:
            QMessageBox.information(self, "Cảnh báo mô hình", "\n".join(warnings[:20]))

        setup = PLSSetupDialog(self)
        if bootstrap:
            setup.set_bootstrap_mode(True)
        if not setup.exec():
            return
        settings = setup.get_settings()
        # The PLS Algorithm command never bootstraps; the Bootstrapping command always does.
        if bootstrap:
            settings["bootstrap_enabled"] = settings.get("algorithm") == "PLS-SEM"
        else:
            settings["bootstrap_enabled"] = False
        self._start_background_calculation(
            self.data_frame.copy(),
            measurement,
            structural,
            modes,
            effects,
            settings,
            bootstrap=bootstrap,
        )

    def _start_background_calculation(
        self,
        data_frame: pd.DataFrame,
        measurement: dict[str, list[str]],
        structural: list[tuple[str, str]],
        modes: dict[str, str],
        effects: list[dict[str, str]],
        settings: dict[str, Any],
        bootstrap: bool = False,
    ) -> None:
        self._calculation_running = True
        self._background_bootstrap = bool(bootstrap)
        if bootstrap:
            total = int(settings.get("bootstrap_subsamples", 0) or 0)
            title = "Đang chạy Bootstrapping"
            running_label = "Đang bootstrapping..."
            total_label = "Tổng subsamples"
        else:
            total = 1
            title = "Đang chạy PLS Algorithm"
            running_label = "Đang chạy thuật toán PLS..."
            total_label = "Tác vụ"
        self._progress_dialog = CalculationProgressDialog(
            title,
            total,
            self,
            running_label=running_label,
            total_label=total_label,
        )
        if not bootstrap:
            self._progress_dialog.cancel_button.setEnabled(False)
            self._progress_dialog.cancel_button.setText("Đang chạy")
        self._progress_dialog.show()

        self._calc_thread = QThread(self)
        self._calc_worker = CalculationWorker(data_frame, measurement, structural, modes, effects, settings)
        self._calc_worker.moveToThread(self._calc_thread)
        self._calc_thread.started.connect(self._calc_worker.run)
        self._calc_worker.progress.connect(self._progress_dialog.update_progress)
        self._calc_worker.finished.connect(self._on_background_calculation_finished)
        self._calc_worker.failed.connect(self._on_background_calculation_failed)
        self._progress_dialog.cancel_requested.connect(self._calc_worker.cancel)
        self._calc_worker.finished.connect(self._calc_thread.quit)
        self._calc_worker.failed.connect(self._calc_thread.quit)
        self._calc_thread.finished.connect(self._calc_worker.deleteLater)
        self._calc_thread.finished.connect(self._calc_thread.deleteLater)

        self._set_calculation_actions_enabled(False)
        self.statusBar().showMessage("Đang chạy bootstrapping trong nền..." if bootstrap else "Đang chạy PLS Algorithm trong nền...")
        self._calc_thread.start()

    @Slot(dict)
    def _on_background_calculation_finished(self, results: dict) -> None:
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.mark_finished()
        bootstrap = bool(getattr(self, "_background_bootstrap", False))
        try:
            self._handle_calculation_results(results, bootstrap=bootstrap)
        except ModelValidationError as exc:
            QMessageBox.warning(self, "Kiểm tra mô hình", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Tính toán thất bại", str(exc))
        finally:
            self._finish_background_calculation()

    @Slot(str, str)
    def _on_background_calculation_failed(self, kind: str, message: str) -> None:
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.mark_finished()
        if kind == "validation":
            QMessageBox.warning(self, "Kiểm tra mô hình", message)
        else:
            QMessageBox.critical(self, "Tính toán thất bại", message)
        if getattr(self, "_background_bootstrap", False):
            self.statusBar().showMessage("Bootstrapping đã dừng.")
        else:
            self.statusBar().showMessage("PLS Algorithm đã dừng.")
        self._finish_background_calculation()

    def _finish_background_calculation(self) -> None:
        self._calculation_running = False
        self._background_bootstrap = False
        self._set_calculation_actions_enabled(True)
        self._update_action_state()
        self._calc_thread = None
        self._calc_worker = None

    def _set_calculation_actions_enabled(self, enabled: bool) -> None:
        for key in ("pls", "bootstrap", "ipma", "predict", "blindfolding", "mga", "permutation"):
            if key in self.actions:
                self.actions[key].setEnabled(enabled)
        for key in ("calculate", "bootstrap"):
            if hasattr(self, "tool_actions") and key in self.tool_actions:
                self.tool_actions[key].setEnabled(enabled)

    def _handle_calculation_results(self, results: dict[str, Any], bootstrap: bool) -> None:
        self.current_results = results
        self.canvas_view.show_results(results)
        self._pls_run_count = getattr(self, "_pls_run_count", 0) + 1
        bootstrap_frame = results.get("bootstrap")
        is_bootstrap = bootstrap and isinstance(bootstrap_frame, pd.DataFrame) and not bootstrap_frame.empty
        if bootstrap and not is_bootstrap:
            raise ModelValidationError("Bootstrapping không tạo được bảng kết quả. Hãy kiểm tra số mẫu lặp và mô hình PLS-SEM.")
        if is_bootstrap:
            self.bootstrap_widget = BootstrapResultsWidget()
            self.bootstrap_widget.retranslate(self.ui_language)
            self.bootstrap_widget.load_results(results)
            label = f"Bootstrapping (Run No. {self._pls_run_count})"
            self._ensure_report(self.bootstrap_widget, label)
        else:
            self.results_widget.load_results(results)
            label = f"PLS Algorithm (Run No. {self._pls_run_count})"
            self._ensure_report(self.results_widget, label)
        self._results_tab_label = label
        self._show_workspace_tab(3)
        self._sync_results_toolbar()
        self._record_result_history(results)
        self._save_current_silently()
        self.update_project_tree()
        done = "Đã chạy xong Bootstrapping" if is_bootstrap else "Đã chạy xong PLS và gắn hệ số lên mô hình"
        self.statusBar().showMessage(done)

    def _prepare_analysis_engine(self, title: str):
        """Validate the model + data and return a configured PLSEngine, or None."""
        if self.data_frame is None:
            self._load_active_data(show_tab=False)
        if self.data_frame is None:
            QMessageBox.warning(self, title, "Hãy nhập dữ liệu trước.")
            return None
        self.data_frame = self.data_frame.rename(columns=normalize_column_name)
        measurement, structural, modes = self.canvas_view.extract_model()
        effects = self.canvas_view.extract_effects()
        engine = PLSEngine(self.data_frame)
        engine.set_model(measurement, structural, modes, effects)
        errors, _warnings = engine.validate_model(self.data_frame)
        if errors:
            QMessageBox.warning(self, "Kiểm tra mô hình", "\n".join(errors))
            return None
        return engine

    def _open_analysis_report(self, widget, label: str) -> None:
        self._pls_run_count = getattr(self, "_pls_run_count", 0) + 1
        self._ensure_report(widget, f"{label} (Run No. {self._pls_run_count})")
        self._show_workspace_tab(3)
        self._sync_results_toolbar()

    def _start_analysis_task(
        self,
        title: str,
        engine: PLSEngine,
        method_name: str,
        args: tuple[Any, ...],
        on_finished,
    ) -> None:
        if getattr(self, "_calculation_running", False):
            QMessageBox.information(self, title, "Một tác vụ đang chạy. Hãy chờ tác vụ hiện tại hoàn tất.")
            return
        self._calculation_running = True
        self._progress_dialog = CalculationProgressDialog(
            f"Đang chạy {title}",
            1,
            self,
            running_label=f"Đang chạy {title}...",
            total_label="Tác vụ",
        )
        self._progress_dialog.cancel_button.setEnabled(False)
        self._progress_dialog.cancel_button.setText("Đang chạy")
        self._progress_dialog.show()

        self._analysis_thread = QThread(self)
        self._analysis_worker = AnalysisTaskWorker(engine, method_name, args)
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(lambda result: self._on_analysis_task_finished(title, result, on_finished))
        self._analysis_worker.failed.connect(lambda kind, message: self._on_analysis_task_failed(title, kind, message))
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.failed.connect(self._analysis_thread.quit)
        self._analysis_thread.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
        self._set_calculation_actions_enabled(False)
        self.statusBar().showMessage(f"Đang chạy {title} trong nền...")
        self._analysis_thread.start()

    @Slot(object)
    def _on_analysis_task_finished(self, title: str, result: Any, on_finished) -> None:
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.mark_finished()
        try:
            on_finished(result)
        except Exception as exc:
            QMessageBox.critical(self, f"{title} thất bại", str(exc))
        finally:
            self._finish_analysis_task()

    @Slot(str, str)
    def _on_analysis_task_failed(self, title: str, kind: str, message: str) -> None:
        if hasattr(self, "_progress_dialog") and self._progress_dialog:
            self._progress_dialog.mark_finished()
        if kind == "validation":
            QMessageBox.warning(self, title, message)
        else:
            QMessageBox.critical(self, f"{title} thất bại", message)
        self.statusBar().showMessage(f"{title} đã dừng.")
        self._finish_analysis_task()

    def _finish_analysis_task(self) -> None:
        self._calculation_running = False
        self._set_calculation_actions_enabled(True)
        self._update_action_state()
        self._analysis_thread = None
        self._analysis_worker = None

    def run_ipma(self) -> None:
        engine = self._prepare_analysis_engine("IPMA")
        if engine is None:
            return
        _measurement, structural, _modes = self.canvas_view.extract_model()
        targets = sorted({target for _source, target in structural})
        if not targets:
            QMessageBox.information(self, "IPMA", "Cần ít nhất một biến nội sinh (có đường dẫn vào) để chạy IPMA.")
            return
        target, ok = QInputDialog.getItem(self, "IPMA", "Chọn biến mục tiêu:", targets, 0, False)
        if not ok:
            return
        def finish(result: dict[str, Any]) -> None:
            sections = [
                {"key": "ipma", "title": f"Importance-Performance Map — {target}", "category": "final", "frame": result["ipma"]},
                {"key": "performance", "title": "Construct Performance (Index 0-100)", "category": "quality", "frame": result["ipma_performance"]},
            ]
            self._open_analysis_report(make_report_widget(sections), f"IPMA: {target}")
            self.statusBar().showMessage(f"Đã chạy IPMA cho biến mục tiêu {target}")

        self._start_analysis_task("IPMA", engine, "calculate_ipma", (target, {"weighting_scheme": "path"}), finish)

    def run_predict(self) -> None:
        engine = self._prepare_analysis_engine("PLS Predict")
        if engine is None:
            return
        dialog = PredictDialog(self, "PLS Predict")
        if not dialog.exec():
            return
        settings = dialog.get_settings()
        def finish(result: dict[str, Any]) -> None:
            candidates = [
                ("mv_pls", "MV Prediction Summary (PLS)", "final", result.get("mv_prediction"), "predict"),
                ("mv_lm", "MV Prediction Summary (LM)", "final", result.get("mv_lm"), "predict"),
                ("mv_cmp", "MV PLS vs LM (RMSE/MAE)", "final", result.get("mv_compare"), "predict"),
                ("lv_pls", "LV Prediction Summary (PLS)", "quality", result.get("lv_prediction"), "predict"),
                ("mv_err", "PLS MV Prediction Error (Descriptives)", "base", result.get("mv_error_desc"), None),
                ("mv_prd", "PLS MV Predictions (Descriptives)", "base", result.get("mv_pred_desc"), None),
                ("lv_err", "PLS LV Prediction Error (Descriptives)", "base", result.get("lv_error_desc"), None),
                ("lv_prd", "PLS LV Predictions (Descriptives)", "base", result.get("lv_pred_desc"), None),
            ]
            sections = [
                {"key": key, "title": title, "category": cat, "frame": frame, "color": color}
                for key, title, cat, frame, color in candidates
                if frame is not None and not frame.empty
            ]
            self._open_analysis_report(make_report_widget(sections), "PLS Predict")
            note = "" if result.get("has_lm") else " — không có biến ngoại sinh nên bỏ qua benchmark LM"
            self.statusBar().showMessage(
                f"Đã chạy PLS Predict ({result['folds']} folds × {result['repetitions']} lần lặp){note}"
            )

        self._start_analysis_task("PLS Predict", engine, "calculate_predict", (settings,), finish)

    def run_blindfolding(self) -> None:
        engine = self._prepare_analysis_engine("Blindfolding")
        if engine is None:
            return
        dialog = PredictDialog(self, "Blindfolding (Q²)")
        if not dialog.exec():
            return
        settings = dialog.get_settings()
        def finish(result: dict[str, Any]) -> None:
            q2 = result["lv_prediction"][["Q²predict"]].rename(columns={"Q²predict": "Q² (=1-SSE/SSO)"})
            sections = [
                {"key": "q2", "title": "Construct Crossvalidated Redundancy (Q²)", "category": "quality", "frame": q2, "color": "predict"},
            ]
            self._open_analysis_report(make_report_widget(sections), "Blindfolding Q²")
            self.statusBar().showMessage("Đã tính Q² predictive relevance")

        self._start_analysis_task("Blindfolding", engine, "calculate_predict", (settings,), finish)

    def _group_columns(self) -> dict[str, list[str]]:
        """Columns usable as a grouping variable: 2..20 distinct values."""
        result: dict[str, list[str]] = {}
        if self.data_frame is None:
            return result
        for column in self.data_frame.columns:
            values = self.data_frame[column].dropna().astype(str)
            unique = sorted(values.unique().tolist())
            if 2 <= len(unique) <= 20:
                result[column] = unique
        return result

    def run_mga(self) -> None:
        engine = self._prepare_analysis_engine("MGA")
        if engine is None:
            return
        columns = self._group_columns()
        if not columns:
            QMessageBox.information(self, "MGA", "Không tìm thấy biến phân loại phù hợp (cần biến có 2–20 giá trị) để chia nhóm.")
            return
        dialog = GroupDialog(columns, self, "Multi-Group Analysis (MGA)")
        if not dialog.exec():
            return
        spec = dialog.get_spec()
        if spec["value_a"] == spec["value_b"]:
            QMessageBox.information(self, "MGA", "Hãy chọn hai nhóm khác nhau.")
            return
        def finish(result: dict[str, Any]) -> None:
            title = f"MGA — {spec['column']}: {result['value_a']} (n={result['n_a']}) vs {result['value_b']} (n={result['n_b']})"
            sections = [{"key": "mga", "title": title, "category": "final", "frame": result["mga"], "color": "mga"}]
            self._open_analysis_report(make_report_widget(sections), f"MGA: {spec['column']}")
            self.statusBar().showMessage("Đã chạy xong MGA")

        self._start_analysis_task("MGA", engine, "calculate_mga", (spec["column"], spec["value_a"], spec["value_b"], spec), finish)

    def run_permutation(self) -> None:
        engine = self._prepare_analysis_engine("Permutation")
        if engine is None:
            return
        columns = self._group_columns()
        if not columns:
            QMessageBox.information(self, "Permutation", "Không tìm thấy biến phân loại phù hợp để chia nhóm.")
            return
        dialog = GroupDialog(columns, self, "Permutation (Multi-Group)")
        if not dialog.exec():
            return
        spec = dialog.get_spec()
        if spec["value_a"] == spec["value_b"]:
            QMessageBox.information(self, "Permutation", "Hãy chọn hai nhóm khác nhau.")
            return
        def finish(result: dict[str, Any]) -> None:
            title = f"Permutation — {spec['column']}: {result['value_a']} vs {result['value_b']}"
            sections = [{"key": "perm", "title": title, "category": "final", "frame": result["permutation"], "color": "mga"}]
            self._open_analysis_report(make_report_widget(sections), f"Permutation: {spec['column']}")
            self.statusBar().showMessage(f"Đã chạy Permutation ({result['permutations']} hoán vị)")

        self._start_analysis_task(
            "Permutation",
            engine,
            "calculate_permutation",
            (spec["column"], spec["value_a"], spec["value_b"], spec),
            finish,
        )

    def validate_current_model(self) -> None:
        if self.data_frame is not None:
            self.data_frame = self.data_frame.rename(columns=normalize_column_name)
        measurement, structural, modes = self.canvas_view.extract_model()
        engine = PLSEngine(self.data_frame) if self.data_frame is not None else PLSEngine()
        engine.set_model(measurement, structural, modes)
        errors, warnings = engine.validate_model(self.data_frame)
        self.show_checker_messages(errors, warnings)
        if errors:
            QMessageBox.warning(self, "Kiểm tra mô hình", "\n".join(errors))
        else:
            QMessageBox.information(self, "Kiểm tra mô hình", "Mô hình có thể chạy." if not warnings else "Mô hình có thể chạy nhưng có cảnh báo.")

    def show_checker_messages(self, errors: list[str], warnings: list[str]) -> None:
        lines = []
        if errors:
            lines.append("Lỗi")
            lines.extend(f"- {item}" for item in errors)
        if warnings:
            if lines:
                lines.append("")
            lines.append("Cảnh báo")
            lines.extend(f"- {item}" for item in warnings)
        if not lines:
            lines.append("Không phát hiện lỗi mô hình.")
        self.checker_box.setPlainText("\n".join(lines))

    def export_diagram(self) -> None:
        self.canvas_view.export_image()

    def load_sample_project(self) -> None:
        rng = np.random.default_rng(2026)
        n = 260
        service_quality = rng.normal(size=n)
        satisfaction = 0.62 * service_quality + rng.normal(scale=0.75, size=n)
        loyalty = 0.55 * satisfaction + 0.25 * service_quality + rng.normal(scale=0.75, size=n)

        def indicators(latent, prefix):
            block = {}
            for index in range(1, 4):
                raw = latent + rng.normal(scale=0.45, size=n)
                scaled = np.clip(np.round(3 + raw), 1, 5)
                block[f"{prefix}{index}"] = scaled
            return block

        data = {}
        data.update(indicators(service_quality, "SQ"))
        data.update(indicators(satisfaction, "SAT"))
        data.update(indicators(loyalty, "LOY"))
        self._save_current_silently()
        self.data_frame = pd.DataFrame(data)
        self.project_state = new_project_state("Dự án mẫu PLS-SEM")
        self.project_path = str(self._unique_project_path("Du_an_mau_PLS_SEM"))
        data_path = self.workspace_dir / f"{Path(self.project_path).stem}_data.csv"
        self.data_frame.to_csv(data_path, index=False)
        self.data_path = str(data_path)
        data_id = str(uuid.uuid4())
        model_id = str(uuid.uuid4())
        self.current_data_id = data_id
        self.current_model_id = model_id
        self._populate_indicators(self.data_frame.columns)

        model = {
            "nodes": [
                {"id": "sq", "type": "latent", "name": "Chất lượng dịch vụ", "x": 130, "y": 210, "mode": "reflective"},
                {"id": "sat", "type": "latent", "name": "Sự hài lòng", "x": 440, "y": 210, "mode": "reflective"},
                {"id": "loy", "type": "latent", "name": "Lòng trung thành", "x": 750, "y": 210, "mode": "reflective"},
                *[
                    {"id": f"sq{i}", "type": "indicator", "name": f"SQ{i}", "x": 130, "y": 40 + i * 48}
                    for i in range(1, 4)
                ],
                *[
                    {"id": f"sat{i}", "type": "indicator", "name": f"SAT{i}", "x": 440, "y": 40 + i * 48}
                    for i in range(1, 4)
                ],
                *[
                    {"id": f"loy{i}", "type": "indicator", "name": f"LOY{i}", "x": 750, "y": 40 + i * 48}
                    for i in range(1, 4)
                ],
            ],
            "connections": [
                {"source": "sq", "target": "sat"},
                {"source": "sat", "target": "loy"},
                {"source": "sq", "target": "loy"},
                *[{"source": "sq", "target": f"sq{i}"} for i in range(1, 4)],
                *[{"source": "sat", "target": f"sat{i}"} for i in range(1, 4)],
                *[{"source": "loy", "target": f"loy{i}"} for i in range(1, 4)],
            ],
        }
        self.project_state["data_files"] = [{"id": data_id, "name": "Sample Data", "path": self.data_path, "rows": len(self.data_frame), "columns": len(self.data_frame.columns)}]
        self.project_state["models"] = [{"id": model_id, "name": "Sample Path Model", "data_file_id": data_id, "model": model}]
        self.project_state["active_model_id"] = model_id
        self.project_state["active_data_id"] = data_id
        self.canvas_view.load_model_state(model)
        self.tabs.setTabText(1, "Sample Data.csv")
        self.tabs.setTabText(2, "Sample Path Model.splsm")
        self.update_data_views()
        self._save_current_silently()
        self.update_project_tree()
        self._show_workspace_tab(2)
        self.statusBar().showMessage("Đã mở dự án mẫu")

    def update_project_tree(self) -> None:
        if not hasattr(self, "project_tree"):
            return
        self.project_tree.setUpdatesEnabled(False)
        self.project_tree.clear()
        try:
            self._sync_active_model()
            current = Path(self.project_path).resolve() if self.project_path else None
            project_files = self._project_files_for_tree()
            current_known = {path.resolve() for path in project_files}
            if current and current not in current_known:
                project_files.insert(0, current)
            current_item = None
            for project_file in project_files:
                try:
                    is_current = bool(current and project_file.resolve() == current)
                    state = self.project_state if is_current else normalize_project_state(load_project(str(project_file)))
                except Exception:
                    continue
                root = QTreeWidgetItem([state.get("name", project_file.stem)])
                root.setData(0, Qt.UserRole + 1, str(project_file))
                root.setData(0, Qt.UserRole + 2, "project")
                root.setIcon(0, icon("project-ok" if self._project_complete(state, check_data_files=False) else "project-error", 19))
                self.project_tree.addTopLevelItem(root)
                if is_current:
                    current_item = root

                data_files = state.get("data_files", [])
                if not data_files:
                    placeholder = QTreeWidgetItem(["Double-click to import"])
                    placeholder.setIcon(0, icon("info", 17))
                    placeholder.setData(0, Qt.UserRole + 1, str(project_file))
                    placeholder.setData(0, Qt.UserRole + 2, "import-data")
                    font = placeholder.font(0); font.setItalic(True); placeholder.setFont(0, font)
                    placeholder.setForeground(0, QBrush(QColor("#666666")))
                    root.addChild(placeholder)

                for model in state.get("models", []):
                    complete = self._model_complete(model.get("model", {})) and bool(model.get("data_file_id"))
                    child = QTreeWidgetItem([model.get("name", "Path Model")])
                    child.setIcon(0, icon("path-ok" if complete else "path-error", 18))
                    child.setData(0, Qt.UserRole + 1, str(project_file))
                    child.setData(0, Qt.UserRole + 2, "model")
                    child.setData(0, Qt.UserRole + 3, model.get("id", ""))
                    root.addChild(child)

                for data in data_files:
                    rows = data.get("rows", "")
                    suffix = f" [{rows} records]" if rows != "" else ""
                    child = QTreeWidgetItem([f"{data.get('name', 'Data')}{suffix}"])
                    data_exists = bool(data.get("path"))
                    child.setIcon(0, icon("data-green" if data_exists else "project-error", 18))
                    child.setData(0, Qt.UserRole + 1, str(project_file))
                    child.setData(0, Qt.UserRole + 2, "data")
                    child.setData(0, Qt.UserRole + 3, data.get("id", ""))
                    if is_current and data.get("id") == self.current_data_id:
                        child.setForeground(0, QBrush(QColor("#00a818")))
                        font = child.font(0); font.setBold(True); child.setFont(0, font)
                    root.addChild(child)

            archive_item = QTreeWidgetItem(["Archive"])
            archive_item.setIcon(0, icon("archive", 18))
            archive_item.setData(0, Qt.UserRole + 2, "archive")
            archive_dir = self.workspace_dir / "Archive"
            if archive_dir.exists():
                for backup in sorted(archive_dir.glob("*.zip"), key=lambda p: p.name.lower()):
                    child = QTreeWidgetItem([backup.stem])
                    child.setIcon(0, icon("archive", 16))
                    archive_item.addChild(child)
            self.project_tree.addTopLevelItem(archive_item)
            if current_item:
                current_item.setExpanded(True)
                self.project_tree.setCurrentItem(current_item)
            self._update_action_state()
        finally:
            self.project_tree.setUpdatesEnabled(True)

    def _set_project_tree_loading(self) -> None:
        if not hasattr(self, "project_tree"):
            return
        self.project_tree.clear()
        item = QTreeWidgetItem(["Đang tải Project Explorer..."])
        item.setIcon(0, icon("info", 17))
        font = item.font(0); font.setItalic(True); item.setFont(0, font)
        item.setForeground(0, QBrush(QColor("#666666")))
        self.project_tree.addTopLevelItem(item)

    def _model_complete(self, model: dict[str, Any]) -> bool:
        nodes = {item.get("id"): item for item in model.get("nodes", [])}
        # Interaction / quadratic terms are not normal constructs and need no indicators.
        latent_ids = {
            node_id for node_id, item in nodes.items()
            if item.get("type") == "latent" and not item.get("effect_type")
        }
        if len(latent_ids) < 2:
            return False
        indicator_links = {node_id: 0 for node_id in latent_ids}
        structural_links = {node_id: 0 for node_id in latent_ids}
        for connection in model.get("connections", []):
            source, target = connection.get("source"), connection.get("target")
            source_item = nodes.get(source, {})
            target_item = nodes.get(target, {})
            if source_item.get("effect_type") or target_item.get("effect_type"):
                continue
            source_type = source_item.get("type")
            target_type = target_item.get("type")
            if source_type == "latent" and target_type == "indicator" and source in indicator_links:
                indicator_links[source] += 1
            if source_type == "latent" and target_type == "latent" and source in structural_links and target in structural_links:
                structural_links[source] += 1; structural_links[target] += 1
        return all(indicator_links.values()) and all(structural_links.values())

    def _project_complete(self, state: dict[str, Any], *, check_data_files: bool = True) -> bool:
        data_ids = {
            item.get("id")
            for item in state.get("data_files", [])
            if item.get("path") and (not check_data_files or Path(str(item.get("path"))).exists())
        }
        models = state.get("models", [])
        return bool(models) and all(self._model_complete(item.get("model", {})) and item.get("data_file_id") in data_ids for item in models)

    def update_properties_panel(self) -> None:
        if not hasattr(self, "canvas_view"):
            return
        selected = self.canvas_view.scene.selectedItems()
        if not selected:
            measurement, structural, modes = self.canvas_view.extract_model()
            lines = [
                f"Dự án: {self.project_state.get('name', '')}",
                f"Dữ liệu: {self.data_frame.shape[0]} dòng x {self.data_frame.shape[1]} biến" if self.data_frame is not None else "Dữ liệu: chưa nhập",
                f"Biến tiềm ẩn: {len(measurement)}",
                f"Đường dẫn cấu trúc: {len(structural)}",
                "",
                "Chế độ đo lường:",
            ]
            lines.extend(f"- {construct}: {self._mode_label(modes.get(construct, 'reflective'))}" for construct in measurement)
            self.properties_box.setPlainText("\n".join(lines))
            return

        item = selected[0]
        if isinstance(item, LatentNode):
            indicators = [
                line.target_node.name
                for line in item.connections
                if isinstance(line.target_node, IndicatorNode)
            ]
            outgoing = [
                line.target_node.name
                for line in item.connections
                if line.source_node == item and isinstance(line.target_node, LatentNode)
            ]
            incoming = [
                line.source_node.name
                for line in item.connections
                if line.target_node == item and isinstance(line.source_node, LatentNode)
            ]
            lines = [
                "Biến tiềm ẩn",
                f"Tên: {item.name}",
                f"Chế độ đo lường: {self._mode_label(item.measurement_mode)}",
                f"Biến quan sát: {', '.join(indicators) if indicators else 'chưa có'}",
                f"Đường dẫn vào: {', '.join(incoming) if incoming else 'chưa có'}",
                f"Đường dẫn ra: {', '.join(outgoing) if outgoing else 'chưa có'}",
                "",
                "Gợi ý: nhấp đúp để đổi tên; chuột phải để đổi phản ánh/hình thành.",
            ]
        elif isinstance(item, IndicatorNode):
            attached = [
                line.source_node.name
                for line in item.connections
                if isinstance(line.source_node, LatentNode)
            ]
            lines = [
                "Biến quan sát",
                f"Tên: {item.name}",
                f"Được gán cho: {', '.join(attached) if attached else 'chưa gán'}",
            ]
        elif isinstance(item, ConnectionLine) and item.source_node and item.target_node:
            lines = [
                "Đường dẫn",
                f"Nguồn: {item.source_node.name}",
                f"Đích: {item.target_node.name if item.target_node else ''}",
            ]
        else:
            lines = ["Đang chọn"]
        self.properties_box.setPlainText("\n".join(lines))

    def _record_result_history(self, results: dict[str, Any]) -> None:
        self.project_state.setdefault("results_history", []).append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "algorithm": results.get("algorithm", ""),
                "observations": results.get("n_observations", ""),
                "converged": results.get("converged", ""),
            }
        )

    def _sync_project_state(self) -> None:
        self._sync_active_model()
        self.project_state["active_model_id"] = self.current_model_id
        self.project_state["active_data_id"] = self.current_data_id
        active_data = self._active_data_entry()
        self.project_state["data_path"] = active_data.get("path", "") if active_data else ""
        self.project_state["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def _asset_path(self, name: str) -> Path:
        roots = [
            Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1])),
            Path(__file__).resolve().parents[1],
            Path.cwd() / "PySmartPLS",
            Path.cwd(),
        ]
        for root in roots:
            candidate = root / "assets" / name
            if candidate.exists():
                return candidate
        return roots[0] / "assets" / name

    def _mode_label(self, mode: str) -> str:
        return "Hình thành / Mode B" if mode == "formative" else "Phản ánh / Mode A"

    def closeEvent(self, event) -> None:
        # If a nonlinear ML job is running, refuse to close (its worker is one long
        # blocking call that can't be interrupted) so Qt never destroys a live QThread.
        nonlinear = getattr(self, "nonlinear", None)
        if nonlinear is not None and nonlinear.has_running_job():
            QMessageBox.information(
                self, "Đang chạy tác vụ",
                "Một tác vụ phân tích phi tuyến đang chạy. Hãy đợi hoàn tất trước khi đóng.")
            event.ignore()
            return
        try:
            self._save_current_silently()
        except OSError:
            pass
        if nonlinear is not None:
            nonlinear.shutdown()
        super().closeEvent(event)

    def _apply_styles(self, theme: str = ui_theme.DEFAULT_THEME) -> None:
        self.current_theme = theme
        p = ui_theme.palette(theme)
        # Update the canvas backdrop + node palette to match the theme.
        if hasattr(self, "canvas_view"):
            self.canvas_view.set_theme_colors(p["canvas"], p["grid"])
            if hasattr(self.canvas_view, "set_palette"):
                self.canvas_view.set_palette(theme)
        style = ui_theme.build_stylesheet(theme)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(style)
        self.setStyleSheet(style)

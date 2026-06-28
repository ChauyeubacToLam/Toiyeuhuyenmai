from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QRectF, QSortFilterProxyModel, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.icons import icon

# Text colours used by SmartPLS for reflective loadings (green = good, red = below threshold).
LOAD_GOOD = QColor("#1b7f3b")
LOAD_BAD = QColor("#c0182c")
LOADING_THRESHOLD = 0.7

# ----------------------------------------------------------------------------
# Bilingual strings
# ----------------------------------------------------------------------------
TR: dict[str, dict[str, str]] = {
    "copy_to_clipboard": {"en": "Copy to Clipboard:", "vi": "Sao chép vào clipboard:"},
    "excel_format": {"en": "Excel Format", "vi": "Định dạng Excel"},
    "r_format": {"en": "R Format", "vi": "Định dạng R"},
    "matrix": {"en": "Matrix", "vi": "Ma trận"},
    "empty": {
        "en": "No results yet. Import data, draw the model, then run the PLS Algorithm.",
        "vi": "Chưa có kết quả. Hãy nhập dữ liệu, vẽ mô hình rồi chạy thuật toán PLS.",
    },
    "not_available": {
        "en": "This result is not available for the current run.",
        "vi": "Kết quả này chưa khả dụng cho lần chạy hiện tại.",
    },
    "cat_final": {"en": "Final Results", "vi": "Kết quả cuối cùng"},
    "cat_quality": {"en": "Quality Criteria", "vi": "Tiêu chí chất lượng"},
    "cat_interim": {"en": "Interim Results", "vi": "Kết quả trung gian"},
    "cat_base": {"en": "Base Data", "vi": "Dữ liệu gốc"},
}

# Category ordering for the bottom navigation panel.
CATEGORIES = ["final", "quality", "interim", "base"]

# Each result: key, category, English label, Vietnamese label.
RESULT_SPECS: list[tuple[str, str, str, str]] = [
    ("path_coefficients", "final", "Path Coefficients", "Hệ số đường dẫn"),
    ("indirect_effects", "final", "Indirect Effects", "Tác động gián tiếp"),
    ("total_effects", "final", "Total Effects", "Tổng tác động"),
    ("outer_loadings", "final", "Outer Loadings", "Hệ số tải ngoài"),
    ("outer_weights", "final", "Outer Weights", "Hệ số trọng số ngoài"),
    ("latent_variable", "final", "Latent Variable", "Biến tiềm ẩn"),
    ("residuals", "final", "Residuals", "Phần dư"),
    ("r_square", "quality", "R Square", "R bình phương"),
    ("f_square", "quality", "f Square", "f bình phương"),
    ("reliability", "quality", "Construct Reliability and Validity", "Độ tin cậy và giá trị"),
    ("discriminant", "quality", "Discriminant Validity", "Giá trị phân biệt"),
    ("vif", "quality", "Collinearity Statistics (VIF)", "Thống kê cộng tuyến (VIF)"),
    ("model_fit", "quality", "Model_Fit", "Độ phù hợp mô hình"),
    ("model_selection", "quality", "Model Selection Criteria", "Tiêu chí chọn mô hình"),
    ("stop_criterion", "interim", "Stop Criterion Changes", "Thay đổi tiêu chí dừng"),
    ("inner_model", "interim", "Inner Model", "Mô hình bên trong"),
    ("outer_model", "interim", "Outer Model", "Mô hình bên ngoài"),
    ("indicator_data_original", "interim", "Indicator Data (Original)", "Dữ liệu chỉ báo (Gốc)"),
    ("indicator_data_standardized", "interim", "Indicator Data (Standardized)", "Dữ liệu chỉ báo (Chuẩn hóa)"),
    ("indicator_correlations", "interim", "Indicator Data (Correlations)", "Dữ liệu chỉ báo (Tương quan)"),
    ("setting", "base", "Setting", "Thiết lập"),
]

SPEC_BY_KEY = {key: (cat, en, vi) for key, cat, en, vi in RESULT_SPECS}


def _t(key: str, lang: str) -> str:
    entry = TR.get(key, {})
    return entry.get(lang, entry.get("en", key))


def _result_name(key: str, lang: str) -> str:
    cat, en, vi = SPEC_BY_KEY[key]
    return vi if lang == "vi" else en


# ----------------------------------------------------------------------------
# Builders: turn the raw engine result dict into display frames + colour mode.
# ----------------------------------------------------------------------------
class ResultView:
    def __init__(self, frame: pd.DataFrame | None, color: str | None = None, show_index: bool = True):
        self.frame = frame
        self.color = color
        self.show_index = show_index


class SortableTableWidgetItem(QTableWidgetItem):
    def __init__(self, text: str, sort_value: Any = None):
        super().__init__(text)
        self.sort_value = sort_value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        left = self.sort_value
        right = getattr(other, "sort_value", None)
        if left is not None and right is not None:
            try:
                return float(left) < float(right)
            except (TypeError, ValueError):
                return str(left).casefold() < str(right).casefold()
        return self.text().casefold() < other.text().casefold()


def _sort_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        return number if np.isfinite(number) else None
    return str(value)


class ResultTableModel(QAbstractTableModel):
    def __init__(self, owner: "PLSResultsWidget", view: ResultView):
        super().__init__()
        self.owner = owner
        self.view = view
        self.frame = view.frame if view.frame is not None else pd.DataFrame()
        self.columns = [str(column) for column in self.frame.columns]
        self.index_labels = [_stringify(index) for index in self.frame.index]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        col_label = self.columns[index.column()]
        if role == Qt.DisplayRole:
            return self.owner._format(value)
        if role == Qt.TextAlignmentRole:
            if col_label == "Result":
                return int(Qt.AlignVCenter | Qt.AlignLeft)
            return int(Qt.AlignCenter)
        if role == Qt.ForegroundRole:
            return self.owner._foreground(self.view.color, value, col_label)
        if role == Qt.UserRole:
            return _sort_value(value)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.columns[section] if 0 <= section < len(self.columns) else ""
        if self.view.show_index and 0 <= section < len(self.index_labels):
            return self.index_labels[section]
        return ""


class ResultSortProxyModel(QSortFilterProxyModel):
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        left_value = left.data(Qt.UserRole)
        right_value = right.data(Qt.UserRole)
        if left_value is None and right_value is None:
            return str(left.data(Qt.DisplayRole)).casefold() < str(right.data(Qt.DisplayRole)).casefold()
        if left_value is None:
            return False
        if right_value is None:
            return True
        try:
            return float(left_value) < float(right_value)
        except (TypeError, ValueError):
            return str(left_value).casefold() < str(right_value).casefold()


def _lvs(results: dict[str, Any]) -> list[str]:
    mm = results.get("measurement_model") or {}
    return list(mm.keys())


def _loadings_matrix(results: dict[str, Any]) -> pd.DataFrame | None:
    mm = results.get("measurement_model") or {}
    ol = results.get("outer_loadings")
    if not isinstance(ol, pd.DataFrame) or not mm:
        return None
    lvs = list(mm)
    inds = [ind for lv in lvs for ind in mm[lv]]
    mat = pd.DataFrame(np.nan, index=inds, columns=lvs)
    for lv in lvs:
        for ind in mm[lv]:
            if ind in ol.index:
                mat.loc[ind, lv] = float(ol.loc[ind, "Primary loading"])
    return mat


def _weights_matrix(results: dict[str, Any]) -> pd.DataFrame | None:
    mm = results.get("measurement_model") or {}
    ow = results.get("outer_weights")
    if not isinstance(ow, pd.DataFrame) or not mm:
        return None
    lvs = list(mm)
    inds = [ind for lv in lvs for ind in mm[lv]]
    mat = pd.DataFrame(np.nan, index=inds, columns=lvs)
    for lv in lvs:
        for ind in mm[lv]:
            if ind in ow.index:
                mat.loc[ind, lv] = float(ow.loc[ind, lv])
    return mat


def _f_square_matrix(results: dict[str, Any]) -> pd.DataFrame | None:
    fs = results.get("f_square")
    lvs = _lvs(results)
    if not isinstance(fs, pd.DataFrame) or fs.empty or not lvs:
        return None
    mat = pd.DataFrame(np.nan, index=lvs, columns=lvs)
    for label, row in fs.iterrows():
        if "->" in str(label):
            source, target = [part.strip() for part in str(label).split("->")]
            if source in mat.index and target in mat.columns:
                mat.loc[source, target] = float(row.get("f2", np.nan))
    return mat


def _vif_matrix(results: dict[str, Any]) -> pd.DataFrame | None:
    vif = results.get("inner_vif")
    lvs = _lvs(results)
    if not isinstance(vif, pd.DataFrame) or vif.empty or not lvs:
        return None
    mat = pd.DataFrame(np.nan, index=lvs, columns=lvs)
    for index, row in vif.iterrows():
        if isinstance(index, tuple) and len(index) == 2:
            target, predictor = index
            if predictor in mat.index and target in mat.columns:
                mat.loc[predictor, target] = float(row.get("VIF", np.nan))
    return mat


def _r_square_table(results: dict[str, Any]) -> pd.DataFrame | None:
    r2 = results.get("r_square")
    adj = results.get("adjusted_r_square")
    if r2 is None or len(r2) == 0:
        return None
    frame = pd.DataFrame({"R Square": r2})
    if adj is not None:
        frame["R Square Adjusted"] = adj
    return frame


def _reliability_table(results: dict[str, Any]) -> pd.DataFrame | None:
    rel = results.get("reliability")
    if not isinstance(rel, pd.DataFrame) or rel.empty:
        return None
    rename = {
        "Cronbach alpha": "Cronbach's Alpha",
        "rho_A approx.": "rho_A",
        "Composite reliability": "Composite Reliability",
        "AVE": "Average Variance Extracted (AVE)",
    }
    columns = [column for column in rename if column in rel.columns]
    return rel[columns].rename(columns=rename)


def _model_fit_table(results: dict[str, Any]) -> pd.DataFrame | None:
    mf = results.get("model_fit")
    if not isinstance(mf, pd.DataFrame) or mf.empty:
        return None
    return mf


def _stop_criterion_table(results: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Value": [results.get("iterations", ""), "Yes" if results.get("converged") else "No"]},
        index=["Number of Iterations", "Converged"],
    )


def _setting_table(results: dict[str, Any]) -> pd.DataFrame:
    rows = {
        "Algorithm": results.get("algorithm", ""),
        "Weighting Scheme": results.get("weighting_scheme", ""),
        "Maximum Iterations": results.get("max_iterations", ""),
        "Stop Criterion (10^-x)": results.get("stop_criterion", ""),
        "Sample Size": results.get("n_observations", ""),
        "Indicators": results.get("n_indicators", ""),
    }
    return pd.DataFrame({"Value": list(rows.values())}, index=list(rows.keys()))


def build_views(results: dict[str, Any]) -> dict[str, ResultView]:
    """Return display frames keyed by result key. Missing results map to an empty ResultView."""
    views: dict[str, ResultView] = {}

    def matrix(key: str) -> pd.DataFrame | None:
        value = results.get(key)
        return value if isinstance(value, pd.DataFrame) and not value.empty else None

    views["path_coefficients"] = ResultView(matrix("path_coefficients"))
    views["indirect_effects"] = ResultView(matrix("indirect_effects"))
    views["total_effects"] = ResultView(matrix("total_effects"))
    views["outer_loadings"] = ResultView(_loadings_matrix(results), color="loading")
    views["outer_weights"] = ResultView(_weights_matrix(results))
    views["latent_variable"] = ResultView(matrix("scores"))
    views["residuals"] = ResultView(None)
    views["r_square"] = ResultView(_r_square_table(results))
    views["f_square"] = ResultView(_f_square_matrix(results))
    views["reliability"] = ResultView(_reliability_table(results))
    views["discriminant"] = ResultView(matrix("fornell_larcker"))
    views["vif"] = ResultView(_vif_matrix(results))
    views["model_fit"] = ResultView(_model_fit_table(results))
    views["model_selection"] = ResultView(None)
    views["stop_criterion"] = ResultView(_stop_criterion_table(results))
    views["inner_model"] = ResultView(matrix("path_coefficients"))
    views["outer_model"] = ResultView(_loadings_matrix(results), color="loading")
    views["indicator_data_original"] = ResultView(matrix("indicator_data_original"))
    views["indicator_data_standardized"] = ResultView(matrix("indicator_data_standardized"))
    views["indicator_correlations"] = ResultView(matrix("indicator_correlations"))
    views["setting"] = ResultView(_setting_table(results))
    return views


# ----------------------------------------------------------------------------
# Bootstrapping report — full SmartPLS-style report (categories, sub-tabs, histograms).
# ----------------------------------------------------------------------------
# Each entry: key, category, kind ('stats' | 'hist' | 'plain'), English, Vietnamese.
BOOT_ENTRIES: list[tuple[str, str, str, str, str]] = [
    ("path_coefficients", "final", "stats", "Path Coefficients", "Hệ số đường dẫn"),
    ("total_indirect", "final", "stats", "Total Indirect Effects", "Tổng tác động gián tiếp"),
    ("specific", "final", "stats", "Specific Indirect Effects", "Tác động gián tiếp cụ thể"),
    ("total", "final", "stats", "Total Effects", "Tổng tác động"),
    ("outer_loadings", "final", "stats", "Outer Loadings", "Hệ số tải ngoài"),
    ("outer_weights", "final", "stats", "Outer Weights", "Trọng số ngoài"),
    ("r2", "quality", "stats", "R Square", "R bình phương"),
    ("radj", "quality", "stats", "R Square Adjusted", "R bình phương hiệu chỉnh"),
    ("f2", "quality", "stats", "f Square", "f bình phương"),
    ("ave", "quality", "stats", "Average Variance Extracted (AVE)", "Phương sai trích (AVE)"),
    ("cr", "quality", "stats", "Composite Reliability", "Độ tin cậy tổng hợp"),
    ("rho_a", "quality", "stats", "rho_A", "rho_A"),
    ("cronbach", "quality", "stats", "Cronbach's Alpha", "Cronbach's Alpha"),
    ("htmt", "quality", "stats", "Heterotrait-Monotrait Ratio (HTMT)", "Tỷ số HTMT"),
    ("lvcorr", "quality", "stats", "Latent Variable Correlations", "Tương quan biến tiềm ẩn"),
    ("model_fit", "modelfit", "stats", "Model Fit (SRMR, d_ULS, d_G)", "Độ phù hợp mô hình"),
    ("hist_path", "hist", "hist", "Path Coefficients Histogram", "Biểu đồ hệ số đường dẫn"),
    ("hist_specific", "hist", "hist", "Specific Indirect Effects Histogram", "Biểu đồ tác động gián tiếp"),
    ("hist_total_indirect", "hist", "hist", "Total Indirect Effects Histogram", "Biểu đồ tổng gián tiếp"),
    ("hist_total", "hist", "hist", "Total Effects Histogram", "Biểu đồ tổng tác động"),
    ("setting", "base", "plain", "Setting", "Thiết lập"),
    ("inner_model", "base", "plain", "Inner Model", "Mô hình bên trong"),
    ("outer_model", "base", "plain", "Outer Model", "Mô hình bên ngoài"),
    ("indicator_data_original", "base", "plain", "Indicator Data (Original)", "Dữ liệu chỉ báo (Gốc)"),
    ("indicator_data_standardized", "base", "plain", "Indicator Data (Standardized)", "Dữ liệu chỉ báo (Chuẩn hóa)"),
]

BOOT_CAT_ORDER = ["final", "quality", "modelfit", "hist", "base"]
BOOT_CAT_LABELS = {
    "final": ("Final Results", "Kết quả cuối cùng"),
    "quality": ("Quality Criteria", "Tiêu chí chất lượng"),
    "modelfit": ("Model Fit", "Độ phù hợp mô hình"),
    "hist": ("Histograms", "Biểu đồ tần suất"),
    "base": ("Base Data", "Dữ liệu gốc"),
}

# Map an engine bootstrap-metric prefix to its stats entry key.
_PREFIX_TO_ENTRY = {
    "path": "path_coefficients",
    "total_indirect": "total_indirect",
    "specific": "specific",
    "total": "total",
    "loading": "outer_loadings",
    "weight": "outer_weights",
    "r2": "r2",
    "radj": "radj",
    "f2": "f2",
    "AVE": "ave",
    "Composite reliability": "cr",
    "rho_A": "rho_a",
    "Cronbach alpha": "cronbach",
    "htmt": "htmt",
    "lvcorr": "lvcorr",
    "model_fit": "model_fit",
}
# Which metric prefix feeds each histogram group.
_HIST_PREFIX = {
    "hist_path": "path",
    "hist_specific": "specific",
    "hist_total_indirect": "total_indirect",
    "hist_total": "total",
}

# The four SmartPLS sub-tabs shown for every matrix-style result.
BOOT_SUBTABS = [
    ("stats", "Mean, STDEV, T-Values, P-Values"),
    ("ci", "Confidence Intervals"),
    ("cibc", "Confidence Intervals Bias Corrected"),
    ("samples", "Samples"),
]


class BootstrapEntry:
    """A single nav entry's rendered payload for the bootstrap report."""

    def __init__(self, kind: str, en: str, vi: str):
        self.kind = kind
        self.en = en
        self.vi = vi
        self.frames: dict[str, ResultView] = {}     # stats: keyed by sub-tab id
        self.plots: list[dict[str, Any]] = []        # hist: one dict per metric
        self.frame: pd.DataFrame | None = None       # plain: a single table
        self.color: str | None = None

    def available(self) -> bool:
        if self.kind == "stats":
            stats = self.frames.get("stats")
            return stats is not None and stats.frame is not None and not stats.frame.empty
        if self.kind == "hist":
            return bool(self.plots)
        return self.frame is not None and not self.frame.empty


def _samples_frame(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per subsample, one column per metric in the group (NaN-padded)."""
    if not columns:
        return pd.DataFrame()
    max_len = max((array.size for array in columns.values()), default=0)
    if max_len == 0:
        return pd.DataFrame()
    data: dict[str, np.ndarray] = {}
    for label, array in columns.items():
        if array.size < max_len:
            array = np.concatenate([array, np.full(max_len - array.size, np.nan)])
        data[label] = array
    frame = pd.DataFrame(data)
    frame.index = range(1, max_len + 1)
    frame.index.name = "Sample"
    return frame


def _boot_setting_table(results: dict[str, Any]) -> pd.DataFrame:
    meta = results.get("bootstrap_meta") or {}
    confidence = float(meta.get("confidence", results.get("bootstrap_confidence", 0.95)))
    rows = {
        "Algorithm": "Bootstrapping",
        "PLS Weighting Scheme": results.get("weighting_scheme", ""),
        "Subsamples": meta.get("subsamples", results.get("bootstrap_subsamples", "")),
        "Valid Subsamples": meta.get("valid", ""),
        "Confidence Interval Method": str(meta.get("ci_method", "percentile")).capitalize(),
        "Test Type": meta.get("test_type", "two-tailed"),
        "Significance Level": f"{(1 - confidence):.2f}",
        "Confidence Level": f"{confidence:.0%}",
        "Random Seed": meta.get("seed", ""),
        "Sample Size": results.get("n_observations", ""),
        "Indicators": results.get("n_indicators", ""),
    }
    return pd.DataFrame({"Value": list(rows.values())}, index=list(rows.keys()))


def build_bootstrap_report(results: dict[str, Any]) -> dict[str, BootstrapEntry]:
    """Turn the engine's flat bootstrap output into per-entry rendered payloads."""
    by_key = {key: (cat, kind, en, vi) for key, cat, kind, en, vi in BOOT_ENTRIES}
    entries: dict[str, BootstrapEntry] = {
        key: BootstrapEntry(kind, en, vi) for key, (_cat, kind, en, vi) in by_key.items()
    }

    boot = results.get("bootstrap")
    samples: dict[str, np.ndarray] = results.get("bootstrap_samples") or {}
    meta = results.get("bootstrap_meta") or {}
    confidence = float(meta.get("confidence", results.get("bootstrap_confidence", 0.95)))
    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = 100 - lower_pct
    lo_label = f"{lower_pct:.1f}%"
    hi_label = f"{upper_pct:.1f}%"

    # Base-data (plain) tables come straight from the PLS result.
    entries["setting"].frame = _boot_setting_table(results)
    entries["inner_model"].frame = results.get("path_coefficients")
    outer = _loadings_matrix(results)
    entries["outer_model"].frame = outer
    entries["outer_model"].color = "loading"
    entries["indicator_data_original"].frame = results.get("indicator_data_original")
    entries["indicator_data_standardized"].frame = results.get("indicator_data_standardized")

    if not isinstance(boot, pd.DataFrame) or boot.empty:
        return entries

    buckets: dict[str, list[tuple[str, str, pd.Series]]] = {}
    for metric, row in boot.iterrows():
        prefix, _, label = str(metric).partition(":")
        entry_key = _PREFIX_TO_ENTRY.get(prefix)
        if entry_key is None:
            continue
        buckets.setdefault(entry_key, []).append((str(metric), label, row))

    stats_cols = ["No.", "Result", "Original Sample (O)", "Sample Mean (M)",
                  "Standard Deviation (STDEV)", "T Statistics (|O/STDEV|)", "P Values"]
    ci_cols = ["No.", "Result", "Original Sample (O)", "Sample Mean (M)", "Bias", lo_label, hi_label]

    for entry_key, items in buckets.items():
        stats_rows, ci_rows, cibc_rows = [], [], []
        sample_columns: dict[str, np.ndarray] = {}
        for number, (metric, label, row) in enumerate(items, start=1):
            stats_rows.append({
                "No.": str(number), "Result": label,
                "Original Sample (O)": float(row.get("Original", np.nan)),
                "Sample Mean (M)": float(row.get("Mean", np.nan)),
                "Standard Deviation (STDEV)": float(row.get("STDEV", np.nan)),
                "T Statistics (|O/STDEV|)": float(row.get("T statistic", np.nan)),
                "P Values": float(row.get("P value", np.nan)),
            })
            ci_rows.append({
                "No.": str(number), "Result": label,
                "Original Sample (O)": float(row.get("Original", np.nan)),
                "Sample Mean (M)": float(row.get("Mean", np.nan)),
                "Bias": float(row.get("Bias", np.nan)),
                lo_label: float(row.get("CI lower", np.nan)),
                hi_label: float(row.get("CI upper", np.nan)),
            })
            cibc_rows.append({
                "No.": str(number), "Result": label,
                "Original Sample (O)": float(row.get("Original", np.nan)),
                "Sample Mean (M)": float(row.get("Mean", np.nan)),
                "Bias": float(row.get("Bias", np.nan)),
                lo_label: float(row.get("CI lower BC", np.nan)),
                hi_label: float(row.get("CI upper BC", np.nan)),
            })
            sample_columns[label] = np.asarray(samples.get(metric, []), dtype=float)

        entry = entries[entry_key]
        entry.color = "bootstrap"
        entry.frames = {
            "stats": ResultView(pd.DataFrame(stats_rows, columns=stats_cols), color="bootstrap", show_index=False),
            "ci": ResultView(pd.DataFrame(ci_rows, columns=ci_cols), color="bootstrap", show_index=False),
            "cibc": ResultView(pd.DataFrame(cibc_rows, columns=ci_cols), color="bootstrap", show_index=False),
            "samples": ResultView(_samples_frame(sample_columns), color=None, show_index=True),
        }

    # Histogram groups.
    for hist_key, prefix in _HIST_PREFIX.items():
        plots: list[dict[str, Any]] = []
        for metric, row in boot.iterrows():
            metric_prefix, _, label = str(metric).partition(":")
            if metric_prefix != prefix:
                continue
            plots.append({
                "label": label,
                "samples": np.asarray(samples.get(metric, []), dtype=float),
                "original": float(row.get("Original", np.nan)),
                "mean": float(row.get("Mean", np.nan)),
            })
        entries[hist_key].plots = plots

    return entries


class HistogramPlot(QWidget):
    """Lightweight QPainter histogram of one bootstrap distribution with an O marker."""

    def __init__(self, label: str, samples: np.ndarray, original: float, mean: float, parent=None):
        super().__init__(parent)
        self.label = label
        values = np.asarray(samples, dtype=float)
        self.samples = values[np.isfinite(values)]
        self.original = original
        self.mean = mean
        self.setMinimumSize(248, 168)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._counts = np.array([])
        self._edges = np.array([])
        if self.samples.size >= 2:
            low, high = float(self.samples.min()), float(self.samples.max())
            if high <= low:
                high = low + 1e-6
            self._counts, self._edges = np.histogram(self.samples, bins=24, range=(low, high))

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QColor("#ffffff"))

        painter.setPen(QColor("#0f172a"))
        title_font = QFont()
        title_font.setPointSize(8)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(QRectF(2, 3, width - 4, 16), int(Qt.AlignHCenter | Qt.AlignVCenter), self.label)

        margin_left, margin_right, margin_top, margin_bottom = 10, 10, 24, 22
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        if self._counts.size == 0 or plot_w <= 0 or plot_h <= 0:
            painter.end()
            return

        count_max = float(self._counts.max()) or 1.0
        low = float(self._edges[0])
        high = float(self._edges[-1])
        span = (high - low) or 1e-6
        bins = self._counts.size
        bar_w = plot_w / bins

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#5b8def")))
        for i, count in enumerate(self._counts):
            bar_h = (count / count_max) * plot_h
            x = margin_left + i * bar_w
            y = margin_top + plot_h - bar_h
            painter.drawRect(QRectF(x, y, max(bar_w - 1.0, 1.0), bar_h))

        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.drawLine(QPointF(margin_left, margin_top + plot_h), QPointF(margin_left + plot_w, margin_top + plot_h))

        if self.original is not None and np.isfinite(self.original) and low <= self.original <= high:
            x_pos = margin_left + (self.original - low) / span * plot_w
            painter.setPen(QPen(QColor("#c0182c"), 1.6))
            painter.drawLine(QPointF(x_pos, margin_top), QPointF(x_pos, margin_top + plot_h))

        painter.setPen(QColor("#475569"))
        axis_font = QFont()
        axis_font.setPointSize(7)
        painter.setFont(axis_font)
        painter.drawText(QRectF(margin_left, margin_top + plot_h + 2, plot_w / 2, 16), int(Qt.AlignLeft), f"{low:.3f}")
        painter.drawText(QRectF(margin_left + plot_w / 2, margin_top + plot_h + 2, plot_w / 2, 16), int(Qt.AlignRight), f"{high:.3f}")
        painter.end()


# ----------------------------------------------------------------------------
# Main results widget — SmartPLS style.
# ----------------------------------------------------------------------------
class PLSResultsWidget(QWidget):
    def __init__(
        self,
        parent=None,
        specs: list[tuple[str, str, str, str]] | None = None,
        builder=None,
        default_key: str = "outer_loadings",
        hide_zeros_default: bool = True,
    ):
        super().__init__(parent)
        self.results: dict[str, Any] | None = None
        self.views: dict[str, ResultView] = {}
        self.lang = "en"
        self.decimals = 3
        self.hide_zeros = hide_zeros_default
        self.specs = specs or RESULT_SPECS
        self.spec_by_key = {key: (cat, en, vi) for key, cat, en, vi in self.specs}
        self.categories = list(dict.fromkeys(cat for _k, cat, _e, _v in self.specs))
        self.builder = builder or build_views
        self.default_key = default_key
        self.current_key = default_key
        self._link_labels: dict[str, QLabel] = {}
        self._category_headers: dict[str, QLabel] = {}
        self._init_ui()

    def _name(self, key: str) -> str:
        cat, en, vi = self.spec_by_key[key]
        return vi if self.lang == "vi" else en

    # -- UI construction ----------------------------------------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.title_label = QLabel("Outer Loadings")
        self.title_label.setObjectName("ResultTitle")
        header.addWidget(self.title_label)
        header.addStretch()
        self.copy_label = QLabel(_t("copy_to_clipboard", self.lang))
        self.copy_label.setObjectName("CopyLabel")
        self.copy_excel_button = QPushButton(_t("excel_format", self.lang))
        self.copy_excel_button.setObjectName("CopyButton")
        self.copy_excel_button.clicked.connect(self.copy_excel)
        self.copy_r_button = QPushButton(_t("r_format", self.lang))
        self.copy_r_button.setObjectName("CopyButton")
        self.copy_r_button.clicked.connect(self.copy_r)
        header.addWidget(self.copy_label)
        header.addWidget(self.copy_excel_button)
        header.addWidget(self.copy_r_button)
        root.addLayout(header)

        tab_row = QHBoxLayout()
        self.matrix_tab = QLabel()
        self.matrix_tab.setObjectName("MatrixTab")
        self.matrix_tab.setText(f'<img/>  {_t("matrix", self.lang)}')
        self.matrix_tab.setPixmap(icon("matrix", 18).pixmap(16, 16))
        self.matrix_tab_text = QLabel(_t("matrix", self.lang))
        self.matrix_tab_text.setObjectName("MatrixTabText")
        tab_holder = QFrame()
        tab_holder.setObjectName("MatrixTabHolder")
        holder_layout = QHBoxLayout(tab_holder)
        holder_layout.setContentsMargins(10, 4, 14, 4)
        holder_layout.setSpacing(6)
        holder_layout.addWidget(self.matrix_tab)
        holder_layout.addWidget(self.matrix_tab_text)
        tab_row.addWidget(tab_holder)
        tab_row.addStretch()
        root.addLayout(tab_row)

        self.table = QTableView()
        self.table.setObjectName("ResultTable")
        self._source_model: ResultTableModel | None = None
        self._proxy_model: ResultSortProxyModel | None = None
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setWordWrap(False)
        self.table.setShowGrid(True)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setSectionsClickable(True)
        root.addWidget(self.table, 1)

        self.empty_label = QLabel(_t("empty", self.lang))
        self.empty_label.setObjectName("EmptyResult")
        self.empty_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.empty_label, 1)

        self.nav_panel = self._build_nav_panel()
        root.addWidget(self.nav_panel)

        self._show_empty(True)

    def _build_nav_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("ResultNav")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(26)
        for category in self.categories:
            column = QVBoxLayout()
            column.setSpacing(3)
            header = QLabel(_t(f"cat_{category}", self.lang))
            header.setObjectName("NavHeader")
            self._category_headers[category] = header
            column.addWidget(header)
            for key, cat, _en, _vi in self.specs:
                if cat != category:
                    continue
                link = QLabel()
                link.setObjectName("NavLink")
                link.setTextFormat(Qt.RichText)
                link.linkActivated.connect(self.show_result)
                self._link_labels[key] = link
                column.addWidget(link)
            column.addStretch()
            layout.addLayout(column)
        layout.addStretch()
        return panel

    # -- state --------------------------------------------------------------
    def set_empty_state(self) -> None:
        self.results = None
        self.views = {}
        self._refresh_links()
        self._show_empty(True)

    def _show_empty(self, empty: bool) -> None:
        self.empty_label.setVisible(empty)
        self.table.setVisible(not empty)
        self.title_label.setVisible(not empty)
        self.copy_label.setVisible(not empty)
        self.copy_excel_button.setVisible(not empty)
        self.copy_r_button.setVisible(not empty)
        self.nav_panel.setVisible(not empty)
        if hasattr(self, "matrix_tab"):
            self.matrix_tab.parentWidget().setVisible(not empty)

    def load_results(self, results: dict[str, Any]) -> None:
        self.results = results
        self.views = self.builder(results)
        if self.current_key not in self.views or self.views[self.current_key].frame is None:
            self.current_key = next(
                (key for key, _c, _e, _v in self.specs if self.views.get(key) and self.views[key].frame is not None),
                self.default_key,
            )
        self._refresh_links()
        self._show_empty(False)
        self.show_result(self.current_key)

    def show_result(self, key: str) -> None:
        if key not in self.spec_by_key:
            return
        self.current_key = key
        self.title_label.setText(self._name(key))
        self._refresh_links()
        view = self.views.get(key)
        if view is None or view.frame is None:
            frame = pd.DataFrame({"": [_t("not_available", self.lang)]})
            self._render(ResultView(frame, show_index=False))
            return
        self._render(view)

    # -- rendering ----------------------------------------------------------
    def _render(self, view: ResultView) -> None:
        frame = view.frame
        rows, cols = frame.shape
        column_labels = [str(column) for column in frame.columns]
        self._source_model = ResultTableModel(self, view)
        self._proxy_model = ResultSortProxyModel(self)
        self._proxy_model.setSourceModel(self._source_model)
        self.table.setModel(self._proxy_model)
        self.table.verticalHeader().setVisible(view.show_index)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(72)
        if "Result" in column_labels:
            for column, label in enumerate(column_labels):
                if label == "No.":
                    header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
                elif label == "Result":
                    header.setSectionResizeMode(column, QHeaderView.Stretch)
                else:
                    header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        elif cols <= 8:
            for column in range(cols):
                header.setSectionResizeMode(column, QHeaderView.Stretch)
        else:
            for column in range(cols):
                header.setSectionResizeMode(column, QHeaderView.Interactive)
            if rows <= 1000:
                self.table.resizeColumnsToContents()
            for column in range(cols):
                self.table.setColumnWidth(column, min(max(self.table.columnWidth(column), 86), 220))
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(24)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            number = float(value)
            if np.isnan(number):
                return ""
            if self.hide_zeros and number == 0:
                return ""
            return f"{number:.{self.decimals}f}"
        if isinstance(value, float) and pd.isna(value):
            return ""
        text = str(value)
        return "" if text in {"nan", "NaN", "None"} else text

    def _color(self, item: QTableWidgetItem, mode: str | None, value: Any, col_label: str = "") -> None:
        color = self._foreground(mode, value, col_label)
        if color is not None:
            item.setForeground(color)

    def _foreground(self, mode: str | None, value: Any, col_label: str = "") -> QColor | None:
        if mode not in {"loading", "bootstrap", "predict", "mga"}:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(number):
            return None
        if mode == "loading":
            return LOAD_GOOD if abs(number) >= LOADING_THRESHOLD else LOAD_BAD
        if mode == "predict":
            if col_label.startswith("Q²"):
                return LOAD_GOOD if number >= 0 else LOAD_BAD
            elif col_label.startswith("RMSE (PLS"):
                return LOAD_GOOD if number < 0 else LOAD_BAD
            return None
        if mode == "mga":
            # significant group difference: two-tailed p<0.05, or PLS-MGA one-tailed p<0.05 / p>0.95
            if col_label.startswith("p ("):
                significant = number < 0.05 or number > 0.95
                return LOAD_GOOD if significant else None
            return None
        # bootstrap: highlight significance on the T-statistic and P-value columns
        if col_label.startswith("P Value"):
            return LOAD_GOOD if number < 0.05 else LOAD_BAD
        elif col_label.startswith("T Statistic"):
            return LOAD_GOOD if abs(number) >= 1.96 else LOAD_BAD
        return None

    # -- navigation links ---------------------------------------------------
    def _refresh_links(self) -> None:
        for key, label in self._link_labels.items():
            name = self._name(key)
            view = self.views.get(key)
            available = bool(view and view.frame is not None)
            active = key == self.current_key and self.results is not None
            if active:
                color = "#0b3d91"
                style = f'color:{color}; font-weight:bold; text-decoration:none;'
            elif available:
                color = "#1a5fb4"
                style = f'color:{color}; text-decoration:underline;'
            else:
                color = "#9aa3ad"
                style = f'color:{color}; text-decoration:none;'
            label.setText(f'<a href="{key}" style="{style}">{name}</a>')

    # -- toolbar hooks ------------------------------------------------------
    def increase_decimals(self) -> None:
        self.decimals = min(self.decimals + 1, 8)
        if self.results:
            self.show_result(self.current_key)

    def decrease_decimals(self) -> None:
        self.decimals = max(self.decimals - 1, 0)
        if self.results:
            self.show_result(self.current_key)

    def set_hide_zeros(self, hide: bool) -> None:
        self.hide_zeros = bool(hide)
        if self.results:
            self.show_result(self.current_key)

    def toggle_hide_zeros(self) -> None:
        self.set_hide_zeros(not self.hide_zeros)

    # -- clipboard / export -------------------------------------------------
    def _current_frame(self) -> pd.DataFrame | None:
        view = self.views.get(self.current_key)
        return view.frame if view else None

    def copy_excel(self) -> None:
        frame = self._current_frame()
        if frame is None:
            return
        text = self._frame_to_text(frame, sep="\t")
        QApplication.clipboard().setText(text)

    def copy_r(self) -> None:
        frame = self._current_frame()
        if frame is None:
            return
        cat, en, vi = self.spec_by_key[self.current_key]
        QApplication.clipboard().setText(_frame_to_r(frame, en))

    def _frame_to_text(self, frame: pd.DataFrame, sep: str = "\t") -> str:
        lines = [sep.join([""] + [str(c) for c in frame.columns])]
        for index, row in frame.iterrows():
            cells = [self._format(value) for value in row]
            lines.append(sep.join([_stringify(index)] + cells))
        return "\n".join(lines)

    def _available_frames(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for key, _cat, en, _vi in self.specs:
            view = self.views.get(key)
            if view and view.frame is not None:
                frames[en] = view.frame
        return frames

    def export_excel(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", "", "Excel Workbook (*.xlsx)")
        if path:
            export_tables_to_excel(self._available_frames(), path)

    def export_html(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export to Web (HTML)", "", "HTML Report (*.html)")
        if path:
            export_tables_to_html(self._available_frames(), path, self.results)

    def export_r(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export to R", "", "R Script (*.R)")
        if path:
            export_tables_to_r(self._available_frames(), path)

    # -- language -----------------------------------------------------------
    def retranslate(self, lang: str) -> None:
        self.lang = lang
        self.copy_label.setText(_t("copy_to_clipboard", lang))
        self.copy_excel_button.setText(_t("excel_format", lang))
        self.copy_r_button.setText(_t("r_format", lang))
        self.matrix_tab_text.setText(_t("matrix", lang))
        self.empty_label.setText(_t("empty", lang))
        for category, header in self._category_headers.items():
            header.setText(_t(f"cat_{category}", lang))
        self._refresh_links()
        if self.results:
            self.show_result(self.current_key)


class BootstrapResultsWidget(QWidget):
    """Full SmartPLS-style Bootstrapping report: categorized nav, 4 sub-tabs, histograms."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: dict[str, Any] | None = None
        self.entries: dict[str, BootstrapEntry] = {}
        self.lang = "en"
        self.decimals = 3
        self.hide_zeros = False
        self.current_key = "path_coefficients"
        self.current_subtab = "stats"
        self.spec_by_key = {key: (cat, kind, en, vi) for key, cat, kind, en, vi in BOOT_ENTRIES}
        self._link_labels: dict[str, QLabel] = {}
        self._category_headers: dict[str, QLabel] = {}
        self._source_model: ResultTableModel | None = None
        self._proxy_model: ResultSortProxyModel | None = None
        self._init_ui()

    def _name(self, key: str) -> str:
        _cat, _kind, en, vi = self.spec_by_key[key]
        return vi if self.lang == "vi" else en

    # -- UI construction ----------------------------------------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        self.title_label = QLabel("Path Coefficients")
        self.title_label.setObjectName("ResultTitle")
        header.addWidget(self.title_label)
        header.addStretch()
        self.copy_label = QLabel(_t("copy_to_clipboard", self.lang))
        self.copy_label.setObjectName("CopyLabel")
        self.copy_excel_button = QPushButton(_t("excel_format", self.lang))
        self.copy_excel_button.setObjectName("CopyButton")
        self.copy_excel_button.clicked.connect(self.copy_excel)
        self.copy_r_button = QPushButton(_t("r_format", self.lang))
        self.copy_r_button.setObjectName("CopyButton")
        self.copy_r_button.clicked.connect(self.copy_r)
        header.addWidget(self.copy_label)
        header.addWidget(self.copy_excel_button)
        header.addWidget(self.copy_r_button)
        root.addLayout(header)

        # Sub-tab bar (Mean/STDEV/T/P · Confidence Intervals · ... · Samples).
        self.subtab_bar = QFrame()
        self.subtab_bar.setObjectName("BootSubtabBar")
        subtab_row = QHBoxLayout(self.subtab_bar)
        subtab_row.setContentsMargins(0, 2, 0, 2)
        subtab_row.setSpacing(6)
        self.subtab_group = QButtonGroup(self)
        self.subtab_group.setExclusive(True)
        self.subtab_buttons: dict[str, QPushButton] = {}
        for sub_id, label in BOOT_SUBTABS:
            button = QPushButton(label)
            button.setObjectName("BootSubtab")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, sub=sub_id: self._on_subtab(sub))
            self.subtab_group.addButton(button)
            self.subtab_buttons[sub_id] = button
            subtab_row.addWidget(button)
        self.subtab_buttons["stats"].setChecked(True)
        subtab_row.addStretch()
        root.addWidget(self.subtab_bar)

        # Content stack: matrix table (0) and histogram grid (1).
        self.content_stack = QStackedWidget()
        self.table = QTableView()
        self.table.setObjectName("ResultTable")
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setWordWrap(False)
        self.table.setShowGrid(True)
        self.table.horizontalHeader().setHighlightSections(False)
        self.table.verticalHeader().setHighlightSections(False)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.content_stack.addWidget(self.table)

        self.hist_scroll = QScrollArea()
        self.hist_scroll.setObjectName("HistScroll")
        self.hist_scroll.setWidgetResizable(True)
        self.hist_scroll.setFrameShape(QFrame.NoFrame)
        self.hist_container = QWidget()
        self.hist_grid = QGridLayout(self.hist_container)
        self.hist_grid.setContentsMargins(4, 4, 4, 4)
        self.hist_grid.setSpacing(10)
        self.hist_scroll.setWidget(self.hist_container)
        self.content_stack.addWidget(self.hist_scroll)
        root.addWidget(self.content_stack, 1)

        self.empty_label = QLabel(_t("empty", self.lang))
        self.empty_label.setObjectName("EmptyResult")
        self.empty_label.setAlignment(Qt.AlignCenter)
        root.addWidget(self.empty_label, 1)

        self.nav_panel = self._build_nav_panel()
        root.addWidget(self.nav_panel)
        self._show_empty(True)

    def _build_nav_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("ResultNav")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(24)
        for category in BOOT_CAT_ORDER:
            column = QVBoxLayout()
            column.setSpacing(3)
            header = QLabel(BOOT_CAT_LABELS[category][0])
            header.setObjectName("NavHeader")
            self._category_headers[category] = header
            column.addWidget(header)
            for key, cat, _kind, _en, _vi in BOOT_ENTRIES:
                if cat != category:
                    continue
                link = QLabel()
                link.setObjectName("NavLink")
                link.setTextFormat(Qt.RichText)
                link.linkActivated.connect(self.show_entry)
                self._link_labels[key] = link
                column.addWidget(link)
            column.addStretch()
            layout.addLayout(column)
        layout.addStretch()
        return panel

    # -- state --------------------------------------------------------------
    def set_empty_state(self) -> None:
        self.results = None
        self.entries = {}
        self._refresh_links()
        self._show_empty(True)

    def _show_empty(self, empty: bool) -> None:
        self.empty_label.setVisible(empty)
        self.content_stack.setVisible(not empty)
        self.subtab_bar.setVisible(not empty)
        self.title_label.setVisible(not empty)
        self.copy_label.setVisible(not empty)
        self.copy_excel_button.setVisible(not empty)
        self.copy_r_button.setVisible(not empty)
        self.nav_panel.setVisible(not empty)

    def load_results(self, results: dict[str, Any]) -> None:
        self.results = results
        self.entries = build_bootstrap_report(results)
        if self.current_key not in self.entries or not self.entries[self.current_key].available():
            self.current_key = next(
                (key for key, _c, _k, _e, _v in BOOT_ENTRIES if self.entries.get(key) and self.entries[key].available()),
                "path_coefficients",
            )
        self._show_empty(False)
        self.show_entry(self.current_key)

    def show_entry(self, key: str) -> None:
        if key not in self.spec_by_key:
            return
        self.current_key = key
        entry = self.entries.get(key)
        self.title_label.setText(self._name(key))
        is_stats = bool(entry and entry.kind == "stats" and entry.available())
        self.subtab_bar.setVisible(is_stats)

        if entry is None or not entry.available():
            self.content_stack.setCurrentIndex(0)
            self._render_view(ResultView(pd.DataFrame({"": [_t("not_available", self.lang)]}), show_index=False))
            self._refresh_links()
            return
        if entry.kind == "stats":
            self._render_stats(entry)
        elif entry.kind == "hist":
            self._render_hist(entry)
        else:
            self.content_stack.setCurrentIndex(0)
            self._render_view(ResultView(entry.frame, color=entry.color, show_index=True))
        self._refresh_links()

    def _on_subtab(self, sub_id: str) -> None:
        self.current_subtab = sub_id
        entry = self.entries.get(self.current_key)
        if entry and entry.kind == "stats" and entry.available():
            self._render_stats(entry)

    def _render_stats(self, entry: BootstrapEntry) -> None:
        self.content_stack.setCurrentIndex(0)
        view = entry.frames.get(self.current_subtab) or entry.frames.get("stats")
        if view is not None:
            self._render_view(view)

    def _render_hist(self, entry: BootstrapEntry) -> None:
        self.content_stack.setCurrentIndex(1)
        while self.hist_grid.count():
            item = self.hist_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        columns = 3
        row = col = 0
        for plot_spec in entry.plots:
            samples = np.asarray(plot_spec["samples"], dtype=float)
            if not np.isfinite(samples).any():
                continue
            plot = HistogramPlot(plot_spec["label"], samples, plot_spec["original"], plot_spec["mean"])
            self.hist_grid.addWidget(plot, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1
        self.hist_grid.setRowStretch(row + 1, 1)

    # -- rendering (shared table logic) -------------------------------------
    def _render_view(self, view: ResultView) -> None:
        frame = view.frame if view.frame is not None else pd.DataFrame()
        rows, cols = frame.shape
        column_labels = [str(column) for column in frame.columns]
        self._source_model = ResultTableModel(self, view)
        self._proxy_model = ResultSortProxyModel(self)
        self._proxy_model.setSourceModel(self._source_model)
        self.table.setModel(self._proxy_model)
        self.table.verticalHeader().setVisible(view.show_index)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(72)
        if "Result" in column_labels:
            for column, label in enumerate(column_labels):
                if label == "Result":
                    header.setSectionResizeMode(column, QHeaderView.Stretch)
                else:
                    header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        elif cols and cols <= 8:
            for column in range(cols):
                header.setSectionResizeMode(column, QHeaderView.Stretch)
        else:
            for column in range(cols):
                header.setSectionResizeMode(column, QHeaderView.Interactive)
            if 0 < rows <= 1000:
                self.table.resizeColumnsToContents()
            for column in range(cols):
                self.table.setColumnWidth(column, min(max(self.table.columnWidth(column), 86), 220))
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(24)

    def _format(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            number = float(value)
            if np.isnan(number):
                return ""
            if self.hide_zeros and number == 0:
                return ""
            return f"{number:.{self.decimals}f}"
        text = str(value)
        return "" if text in {"nan", "NaN", "None"} else text

    def _foreground(self, mode: str | None, value: Any, col_label: str = "") -> QColor | None:
        if mode not in {"loading", "bootstrap"}:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(number):
            return None
        if mode == "loading":
            return LOAD_GOOD if abs(number) >= LOADING_THRESHOLD else LOAD_BAD
        if col_label.startswith("P Value"):
            return LOAD_GOOD if number < 0.05 else LOAD_BAD
        if col_label.startswith("T Statistic"):
            return LOAD_GOOD if abs(number) >= 1.96 else LOAD_BAD
        return None

    # -- navigation links ---------------------------------------------------
    def _refresh_links(self) -> None:
        for key, label in self._link_labels.items():
            name = self._name(key)
            entry = self.entries.get(key)
            available = bool(entry and entry.available())
            active = key == self.current_key and self.results is not None
            if active:
                style = "color:#0b3d91; font-weight:bold; text-decoration:none;"
            elif available:
                style = "color:#1a5fb4; text-decoration:underline;"
            else:
                style = "color:#9aa3ad; text-decoration:none;"
            label.setText(f'<a href="{key}" style="{style}">{name}</a>')

    # -- toolbar hooks ------------------------------------------------------
    def increase_decimals(self) -> None:
        self.decimals = min(self.decimals + 1, 8)
        if self.results:
            self.show_entry(self.current_key)

    def decrease_decimals(self) -> None:
        self.decimals = max(self.decimals - 1, 0)
        if self.results:
            self.show_entry(self.current_key)

    def set_hide_zeros(self, hide: bool) -> None:
        self.hide_zeros = bool(hide)
        if self.results:
            self.show_entry(self.current_key)

    def toggle_hide_zeros(self) -> None:
        self.set_hide_zeros(not self.hide_zeros)

    # -- clipboard / export -------------------------------------------------
    def _current_frame(self) -> pd.DataFrame | None:
        entry = self.entries.get(self.current_key)
        if not entry:
            return None
        if entry.kind == "stats":
            view = entry.frames.get(self.current_subtab)
            return view.frame if view else None
        if entry.kind == "plain":
            return entry.frame
        return None

    def copy_excel(self) -> None:
        frame = self._current_frame()
        if frame is None:
            return
        QApplication.clipboard().setText(self._frame_to_text(frame, sep="\t"))

    def copy_r(self) -> None:
        frame = self._current_frame()
        if frame is None:
            return
        QApplication.clipboard().setText(_frame_to_r(frame, self._name(self.current_key)))

    def _frame_to_text(self, frame: pd.DataFrame, sep: str = "\t") -> str:
        lines = [sep.join([""] + [str(c) for c in frame.columns])]
        for index, row in frame.iterrows():
            cells = [self._format(value) for value in row]
            lines.append(sep.join([_stringify(index)] + cells))
        return "\n".join(lines)

    def _available_frames(self) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        for key, _cat, kind, en, _vi in BOOT_ENTRIES:
            entry = self.entries.get(key)
            if not entry:
                continue
            if kind == "stats" and entry.available():
                for sub_id, suffix in (("stats", ""), ("ci", " (CI)"), ("cibc", " (CI BC)")):
                    view = entry.frames.get(sub_id)
                    if view is not None and view.frame is not None and not view.frame.empty:
                        frames[f"{en}{suffix}"] = view.frame
            elif kind == "plain" and entry.frame is not None and not entry.frame.empty:
                frames[en] = entry.frame
        return frames

    def export_excel(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export to Excel", "", "Excel Workbook (*.xlsx)")
        if path:
            export_tables_to_excel(self._available_frames(), path)

    def export_html(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export to Web (HTML)", "", "HTML Report (*.html)")
        if path:
            export_tables_to_html(self._available_frames(), path, self.results)

    def export_r(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export to R", "", "R Script (*.R)")
        if path:
            export_tables_to_r(self._available_frames(), path)

    # -- language -----------------------------------------------------------
    def retranslate(self, lang: str) -> None:
        self.lang = lang
        self.copy_label.setText(_t("copy_to_clipboard", lang))
        self.copy_excel_button.setText(_t("excel_format", lang))
        self.copy_r_button.setText(_t("r_format", lang))
        self.empty_label.setText(_t("empty", lang))
        for category, header in self._category_headers.items():
            header.setText(BOOT_CAT_LABELS[category][1 if lang == "vi" else 0])
        self._refresh_links()
        if self.results:
            self.show_entry(self.current_key)


def make_report_widget(sections: list[dict[str, Any]], parent=None) -> "PLSResultsWidget":
    """Build a self-contained report widget from explicit sections.

    Each section: {key, title, category('final'/'quality'/'interim'/'base'), frame, color?}.
    Used for analyses (IPMA, Blindfolding, PLSpredict, ...) that have their own tables.
    """
    specs = [(s["key"], s.get("category", "final"), s["title"], s["title"]) for s in sections]
    views = {s["key"]: ResultView(s.get("frame"), color=s.get("color")) for s in sections}
    default = next((s["key"] for s in sections if s.get("frame") is not None), specs[0][0] if specs else "none")
    widget = PLSResultsWidget(
        parent,
        specs=specs,
        builder=lambda _results: dict(views),
        default_key=default,
        hide_zeros_default=False,
    )
    widget.load_results({"algorithm": ""})
    return widget


class PLSResultsDialog(QDialog):
    def __init__(self, results: dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle("PLS-SEM Results")
        self.resize(1040, 700)
        layout = QVBoxLayout(self)
        widget = PLSResultsWidget(self)
        widget.load_results(results)
        layout.addWidget(widget)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignRight)


# ----------------------------------------------------------------------------
# Shared helpers (also used by the Data tab).
# ----------------------------------------------------------------------------
def fill_table(table: QTableWidget, frame: pd.DataFrame) -> None:
    display = frame.copy()
    index_labels = [_stringify(item) for item in display.index]
    column_labels = [str(column) for column in display.columns]

    previous_sort = table.isSortingEnabled()
    table.setSortingEnabled(False)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    table.setRowCount(display.shape[0])
    table.setColumnCount(display.shape[1])
    table.setHorizontalHeaderLabels(column_labels)
    table.setVerticalHeaderLabels(index_labels)

    for row in range(display.shape[0]):
        for column in range(display.shape[1]):
            value = display.iloc[row, column]
            item = SortableTableWidgetItem(_format_plain(value), _sort_value(value))
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, column, item)

    header = table.horizontalHeader()
    header.setSectionsClickable(True)
    header.setStretchLastSection(False)
    if display.shape[1] <= 8:
        for column in range(display.shape[1]):
            header.setSectionResizeMode(column, QHeaderView.Stretch)
    else:
        for column in range(display.shape[1]):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        table.resizeColumnsToContents()
        for column in range(display.shape[1]):
            table.setColumnWidth(column, min(max(table.columnWidth(column), 84), 220))
    table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.setSortingEnabled(True)
    if previous_sort:
        table.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())


def _format_plain(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return str(value)


def _stringify(value: Any) -> str:
    if isinstance(value, tuple):
        return " / ".join(str(part) for part in value)
    return str(value)


def _frame_to_r(frame: pd.DataFrame, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "result"
    csv_text = frame.to_csv(sep=",", lineterminator="\n")
    escaped = csv_text.replace("\\", "\\\\").replace('"', '\\"')
    return f'{safe} <- read.csv(text="{escaped}", row.names=1, check.names=FALSE)\n'


def export_tables_to_excel(tables: dict[str, pd.DataFrame], path: str) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for title, frame in tables.items():
            frame.to_excel(writer, sheet_name=_safe_sheet_name(title))


def export_tables_to_html(tables: dict[str, pd.DataFrame], path: str, results: dict[str, Any]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>PLS-SEM Report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#0f172a}"
        "table{border-collapse:collapse;width:100%;margin-bottom:28px}"
        "th,td{border:1px solid #cbd5e1;padding:6px 8px;text-align:right}"
        "th{background:#e2e8f0}td:first-child,th:first-child{text-align:left}"
        "h1,h2{color:#0f172a}</style></head><body>",
        "<h1>PLS-SEM Report</h1>",
        f"<p><b>Algorithm:</b> {results.get('algorithm', '')}</p>",
    ]
    for title, frame in tables.items():
        parts.append(f"<h2>{title}</h2>")
        parts.append(frame.to_html(float_format=lambda value: f"{value:.4f}", border=0, na_rep=""))
    parts.append("</body></html>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def export_tables_to_r(tables: dict[str, pd.DataFrame], path: str) -> None:
    parts = ["# PLS-SEM results exported from PySmartPLS", ""]
    for title, frame in tables.items():
        parts.append(_frame_to_r(frame, title))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def _safe_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\[\]:*?/\\]", " ", name).strip()
    return cleaned[:31] or "Sheet"

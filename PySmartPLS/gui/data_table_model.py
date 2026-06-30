"""Fast, model-backed tables for the data view.

The data view used to populate ``QTableWidget`` cell-by-cell (one
``QTableWidgetItem`` per cell). For a dataset with hundreds of indicators that
means tens of thousands of widget objects created on the GUI thread on every
click — which is what made selecting a data file or a data tab feel laggy.

``DataFrameTableModel`` exposes a pandas ``DataFrame`` through Qt's
model/view architecture instead: ``QTableView`` only asks the model for the
cells it actually paints, so rendering cost no longer scales with the size of
the frame. This mirrors the pattern already used for the results tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableView

from gui.results_view import ResultSortProxyModel, _sort_value, _stringify


def _format_cell(value: Any, decimals: int) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return ""
        return f"{number:.{decimals}f}"
    return str(value)


class DataFrameTableModel(QAbstractTableModel):
    """Read-only Qt table model wrapping a pandas DataFrame."""

    def __init__(self, frame: pd.DataFrame | None, *, show_index: bool = True,
                 decimals: int = 4, parent=None) -> None:
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()
        self._decimals = decimals
        self._show_index = show_index
        self._columns = [str(column) for column in self._frame.columns]
        self._index = [_stringify(label) for label in self._frame.index]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            return _format_cell(value, self._decimals)
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignCenter)
        if role == Qt.UserRole:
            return _sort_value(value)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._columns[section] if 0 <= section < len(self._columns) else ""
        if self._show_index and 0 <= section < len(self._index):
            return self._index[section]
        return ""


def make_fast_table(object_name: str = "") -> QTableView:
    """Create a QTableView pre-configured to match the app's table look."""

    view = QTableView()
    if object_name:
        view.setObjectName(object_name)
    view.setAlternatingRowColors(True)
    view.setEditTriggers(QAbstractItemView.NoEditTriggers)
    view.setSelectionBehavior(QAbstractItemView.SelectRows)
    view.setWordWrap(False)
    view.setShowGrid(True)
    view.setCornerButtonEnabled(False)
    view.horizontalHeader().setHighlightSections(False)
    view.verticalHeader().setHighlightSections(False)
    view.horizontalHeader().setSectionsClickable(True)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    view.verticalHeader().setSectionResizeMode(QHeaderView.Interactive)
    view.verticalHeader().setDefaultSectionSize(28)
    return view


def set_dataframe(view: QTableView, frame: pd.DataFrame | None, *,
                  show_index: bool = True, decimals: int = 4,
                  sortable: bool = True, stretch_threshold: int = 8,
                  default_width: int = 100) -> DataFrameTableModel:
    """Bind ``frame`` to ``view`` using the fast model. Returns the model.

    Column sizing mirrors the previous behaviour: stretch when the table is
    narrow, otherwise use a fixed width so wide previews never trigger an
    expensive content scan over every row.
    """

    # Detach the previous model from the view BEFORE disposing it, so we never
    # reparent/schedule-delete a proxy that is still installed (matches clear_table).
    if view.model() is not None:
        view.setModel(None)
    _dispose_models(view)
    model = DataFrameTableModel(frame, show_index=show_index, decimals=decimals)
    if sortable:
        proxy = ResultSortProxyModel(view)
        proxy.setSourceModel(model)
        view.setModel(proxy)
        view._df_proxy = proxy  # keep a strong reference alongside the view
    else:
        view.setModel(model)
        view._df_proxy = None
    view._df_model = model  # prevent garbage collection of the source model
    view.setSortingEnabled(sortable)
    view.verticalHeader().setVisible(show_index)

    header = view.horizontalHeader()
    header.setStretchLastSection(False)
    header.setMinimumSectionSize(56)
    columns = 0 if frame is None else len(frame.columns)
    header.setDefaultSectionSize(default_width)
    for column in range(columns):
        header.setSectionResizeMode(column, QHeaderView.Interactive)
        view.setColumnWidth(column, default_width)
    return model


def _dispose_models(view: QTableView) -> None:
    """Drop the proxy/model from a previous bind so they do not accumulate."""

    proxy = getattr(view, "_df_proxy", None)
    if proxy is not None:
        proxy.setParent(None)
        proxy.deleteLater()
    view._df_proxy = None
    view._df_model = None


def clear_table(view: QTableView) -> None:
    """Detach any model so the view shows nothing (and frees memory)."""

    view.setModel(None)
    _dispose_models(view)

"""Data-screening view — renders the Buổi 3 academic screening report.

Shows missing-value analysis, outlier detection, normality checks, an academic
interpretation and a STROBE-ready report paragraph, all computed by
``core.data_screening.screen_dataset`` (off the GUI thread).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui.data_table_model import make_fast_table, set_dataframe

_ROW_HEIGHT = 24
_HEADER_HEIGHT = 28


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        else:
            child = item.layout()
            if child is not None:
                _clear_layout(child)


def _fit_table_height(view, n_rows: int, cap: int = 340) -> None:
    height = _HEADER_HEIGHT + max(n_rows, 1) * _ROW_HEIGHT + 8
    view.setMinimumHeight(min(height, cap))
    view.setMaximumHeight(16777215)


class DataScreeningView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: dict[str, Any] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("ScreenScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._scroll)

        self._content = QWidget()
        self._content.setObjectName("ScreenContent")
        self._body = QVBoxLayout(self._content)
        self._body.setContentsMargins(14, 12, 14, 16)
        self._body.setSpacing(14)
        self._scroll.setWidget(self._content)

        self._empty = QLabel("Nhập dữ liệu để xem kết quả sàng lọc (missing, outliers, normality).")
        self._empty.setObjectName("ScreenEmpty")
        self._empty.setAlignment(Qt.AlignCenter)
        self._body.addWidget(self._empty)
        self._body.addStretch(1)

    # -- public API ---------------------------------------------------------
    def clear(self) -> None:
        self._result = None
        _clear_layout(self._body)
        self._empty = QLabel("Nhập dữ liệu để xem kết quả sàng lọc (missing, outliers, normality).")
        self._empty.setObjectName("ScreenEmpty")
        self._empty.setAlignment(Qt.AlignCenter)
        self._body.addWidget(self._empty)
        self._body.addStretch(1)

    def set_result(self, result: dict[str, Any] | None) -> None:
        self._result = result
        _clear_layout(self._body)
        if not result:
            self.clear()
            return

        self._body.addWidget(self._build_summary(result.get("summary", {})))

        if not result.get("summary", {}).get("scipy", True):
            warn = QLabel("Lưu ý: không tìm thấy SciPy nên bỏ qua kiểm định Shapiro–Wilk / Kolmogorov–Smirnov.")
            warn.setObjectName("ScreenSectionNote")
            warn.setWordWrap(True)
            self._body.addWidget(warn)

        self._body.addWidget(self._table_section(
            "1 · Giá trị thiếu (Missing Values)",
            "Thiếu ≤5% → thay bằng trung bình; >5% → Multiple Imputation hoặc loại biến (Hair et al., 2021).",
            result.get("missing"), decimals=2,
        ))
        self._body.addWidget(self._table_section(
            "2 · Phân phối & Chuẩn (Normality)",
            "Chấp nhận cho PLS-SEM khi |Skewness| ≤ 2 và |Kurtosis| ≤ 7. SW/K-S p > 0.05 ⇒ phân phối chuẩn.",
            result.get("normality"), decimals=3,
        ))
        self._body.addWidget(self._table_section(
            "3 · Ngoại lai (Outliers)",
            "Z > ±3.29, hoặc nằm ngoài 1.5×IQR (nhẹ) / 3.0×IQR (cực trị), hoặc ngoài thang đo hợp lệ.",
            result.get("outliers"), decimals=2,
        ))
        self._body.addWidget(self._table_section(
            "4 · Giá trị cực trị (Extreme Values)",
            "5 giá trị nhỏ nhất và lớn nhất kèm số thứ tự quan sát (case) để truy vết lỗi nhập liệu.",
            result.get("extreme_values"), decimals=2, show_index=False,
        ))

        interp = result.get("interpretation_html")
        if interp:
            self._body.addWidget(self._text_section("5 · Diễn giải học thuật", interp, "ScreenInterpret"))
        strobe = result.get("strobe_html")
        if strobe:
            self._body.addWidget(self._text_section(
                "6 · Báo cáo theo chuẩn STROBE", strobe, "ScreenStrobe", copyable=True))

        self._body.addStretch(1)

    # -- builders -----------------------------------------------------------
    def _build_summary(self, summary: dict[str, Any]) -> QWidget:
        card = QFrame()
        card.setObjectName("ScreenCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)
        chips = [
            ("Quan sát (N)", summary.get("rows", 0), "accent"),
            ("Biến đo lường", summary.get("variables", 0), "accent"),
            ("Ô dữ liệu thiếu", summary.get("missing_cells", 0), "warning"),
            ("Biến không chuẩn", summary.get("non_normal", 0), "warning"),
            ("Biến có ngoại lai", summary.get("outlier_vars", 0), "danger"),
        ]
        for title, value, tone in chips:
            layout.addWidget(self._chip(title, str(value), tone))
        layout.addStretch(1)
        return card

    def _chip(self, title: str, value: str, tone: str) -> QWidget:
        chip = QFrame()
        chip.setObjectName("ScreenChip")
        chip.setProperty("tone", tone)
        box = QVBoxLayout(chip)
        box.setContentsMargins(14, 8, 14, 8)
        box.setSpacing(2)
        value_label = QLabel(value)
        value_label.setObjectName("ScreenChipValue")
        title_label = QLabel(title)
        title_label.setObjectName("ScreenChipTitle")
        box.addWidget(value_label)
        box.addWidget(title_label)
        return chip

    def _section_shell(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("ScreenCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        header = QLabel(title)
        header.setObjectName("ScreenSectionTitle")
        layout.addWidget(header)
        return card, layout

    def _table_section(self, title: str, note: str, frame, *, decimals: int = 3,
                        show_index: bool = False) -> QWidget:
        card, layout = self._section_shell(title)
        note_label = QLabel(note)
        note_label.setObjectName("ScreenSectionNote")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)

        if frame is None or getattr(frame, "empty", True):
            empty = QLabel("Không có dữ liệu phù hợp.")
            empty.setObjectName("ScreenSectionNote")
            layout.addWidget(empty)
            return card

        view = make_fast_table("ScreenTable")
        set_dataframe(view, frame, show_index=show_index, decimals=decimals, sortable=True)
        _fit_table_height(view, len(frame.index))
        layout.addWidget(view)
        return card

    def _text_section(self, title: str, html: str, object_name: str,
                      copyable: bool = False) -> QWidget:
        card, layout = self._section_shell(title)
        browser = QTextBrowser()
        browser.setObjectName(object_name)
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        browser.document().setDocumentMargin(2)
        browser.setMinimumHeight(120 if copyable else 150)
        layout.addWidget(browser)
        if copyable:
            copy_button = QPushButton("Sao chép báo cáo")
            copy_button.clicked.connect(lambda: QApplication.clipboard().setText(browser.toPlainText()))
            layout.addWidget(copy_button, alignment=Qt.AlignRight)
        return card

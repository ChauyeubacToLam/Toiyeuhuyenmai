from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.icons import icon
from gui.theme import apply_shadow


class PremiumDialog(QDialog):
    """Frameless, rounded "card" dialog with a header, body and footer.

    Subclasses fill ``self.content`` (a QVBoxLayout) — usually via ``add_row``
    for labelled fields and ``add_note`` for hints — and register footer
    buttons via ``add_button``. The shell provides a soft drop shadow, an
    accent icon chip, a close button and header dragging so every popup in the
    app shares one polished, high-end look.
    """

    def __init__(self, title: str, subtitle: str = "", icon_name: str = "",
                 parent=None, width: int = 460):
        super().__init__(parent)
        self.setObjectName("PremiumDialog")
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setModal(True)
        self._drag_pos: QPoint | None = None
        self._form: QFormLayout | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 24, 26, 30)

        self.card = QFrame()
        self.card.setObjectName("DialogCard")
        apply_shadow(self.card, blur=46, y=14, color="#15223f4d")
        outer.addWidget(self.card)

        card = QVBoxLayout(self.card)
        card.setContentsMargins(0, 0, 0, 0)
        card.setSpacing(0)

        # ----- header -----
        header = QFrame()
        header.setObjectName("DialogHeader")
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(22, 18, 14, 18)
        hrow.setSpacing(14)
        if icon_name:
            chip = QLabel()
            chip.setObjectName("DialogIcon")
            chip.setFixedSize(46, 46)
            chip.setAlignment(Qt.AlignCenter)
            chip.setPixmap(icon(icon_name, 26).pixmap(26, 26))
            hrow.addWidget(chip, 0, Qt.AlignVCenter)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        title_label = QLabel(title)
        title_label.setObjectName("DialogTitle")
        titles.addWidget(title_label)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("DialogSubtitle")
            sub.setWordWrap(True)
            titles.addWidget(sub)
        hrow.addLayout(titles, 1)
        close = QToolButton()
        close.setObjectName("DialogClose")
        close.setIcon(icon("x", 16))
        close.setIconSize(QSize(16, 16))
        close.setCursor(Qt.PointingHandCursor)
        close.setFixedSize(30, 30)
        close.clicked.connect(self.reject)
        hrow.addWidget(close, 0, Qt.AlignTop)
        card.addWidget(header)
        self._header = header

        # ----- body -----
        self.body = QWidget()
        self.body.setObjectName("DialogBody")
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(24, 20, 24, 22)
        self.content.setSpacing(14)
        card.addWidget(self.body, 1)

        # ----- footer -----
        self.footer = QFrame()
        self.footer.setObjectName("DialogFooter")
        self._footer_row = QHBoxLayout(self.footer)
        self._footer_row.setContentsMargins(22, 14, 22, 16)
        self._footer_row.setSpacing(10)
        self._footer_row.addStretch()
        card.addWidget(self.footer)

        self.setMinimumWidth(width)

    # ----- content helpers -----
    def _ensure_form(self) -> QFormLayout:
        if self._form is None:
            self._form = QFormLayout()
            self._form.setHorizontalSpacing(18)
            self._form.setVerticalSpacing(13)
            self._form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self._form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            self.content.addLayout(self._form)
        return self._form

    def add_row(self, label: str, widget: QWidget) -> QWidget:
        lbl = QLabel(label)
        lbl.setObjectName("FieldLabel")
        self._ensure_form().addRow(lbl, widget)
        return widget

    def add_widget(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.content.addWidget(widget, stretch)
        return widget

    def add_note(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("HintLabel")
        lbl.setWordWrap(True)
        self.content.addWidget(lbl)
        return lbl

    def add_button(self, text: str, role: str = "secondary", default: bool = False,
                   on_click=None) -> QPushButton:
        btn = QPushButton(text)
        if role == "primary":
            btn.setObjectName("PrimaryButton")
        elif role == "ghost":
            btn.setObjectName("GhostButton")
        elif role == "danger":
            btn.setObjectName("DangerButton")
        btn.setMinimumWidth(116)
        btn.setCursor(Qt.PointingHandCursor)
        if default:
            btn.setDefault(True)
            btn.setAutoDefault(True)
        if on_click is not None:
            btn.clicked.connect(on_click)
        self._footer_row.addWidget(btn)
        return btn

    # ----- window behaviour -----
    def showEvent(self, event) -> None:
        super().showEvent(event)
        parent = self.parentWidget()
        if parent is not None and parent.isVisible():
            geo = parent.frameGeometry()
            self.move(geo.center() - self.rect().center())

    def _header_rect(self) -> QRect:
        top_left = self._header.mapTo(self, self._header.rect().topLeft())
        return QRect(top_left, self._header.size())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._header_rect().contains(event.position().toPoint()):
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)


class NewProjectDialog(PremiumDialog):
    def __init__(self, parent=None):
        super().__init__(
            "Tạo dự án mới",
            "Đặt tên cho dự án PLS-SEM mới của bạn.",
            icon_name="new-project", parent=parent, width=460,
        )
        self.name_input = QLineEdit("Untitled Project")
        self.name_input.selectAll()
        self.add_row("Tên dự án", self.name_input)
        self.add_button("Hủy", "secondary", on_click=self.reject)
        self.add_button("Tạo dự án", "primary", default=True, on_click=self.accept)

    def project_name(self) -> str:
        return self.name_input.text().strip() or "Untitled Project"


class DataImportDialog(PremiumDialog):
    def __init__(self, file_path: str, parent=None):
        self.file_path = file_path
        super().__init__(
            "Nhập dữ liệu",
            f"Đang nhập tệp “{Path(file_path).name}”.",
            icon_name="data", parent=parent, width=560,
        )
        self.name_input = QLineEdit(Path(file_path).stem)
        self.name_input.selectAll()
        self.add_row("Tên bộ dữ liệu", self.name_input)
        self.add_note("Tên này hiển thị trong Project Explorer và dùng cho các báo cáo.")
        self.add_button("Hủy", "secondary", on_click=self.reject)
        self.add_button("OK", "primary", default=True, on_click=self.accept)

    def data_name(self) -> str:
        return self.name_input.text().strip() or Path(self.file_path).stem


class GroupDialog(PremiumDialog):
    """Pick a grouping variable and two groups to compare (MGA / Permutation)."""

    def __init__(self, columns_values: dict, parent=None, title: str = "Multi-Group Analysis"):
        self.columns_values = columns_values
        super().__init__(
            title,
            "Chọn biến phân nhóm (biến phân loại) và hai nhóm cần so sánh.",
            icon_name="analysis", parent=parent, width=480,
        )
        self.column = QComboBox()
        self.column.addItems(list(columns_values.keys()))
        self.column.currentTextChanged.connect(self._reload_values)
        self.add_row("Biến phân nhóm", self.column)
        self.value_a = QComboBox()
        self.value_b = QComboBox()
        self.add_row("Nhóm A", self.value_a)
        self.add_row("Nhóm B", self.value_b)
        self.subsamples = QSpinBox()
        self.subsamples.setRange(50, 10000)
        self.subsamples.setValue(300)
        self.subsamples.setSingleStep(50)
        self.add_row("Số mẫu (bootstrap/permutation)", self.subsamples)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(12345)
        self.add_row("Seed", self.seed)
        self.add_button("Hủy", "secondary", on_click=self.reject)
        self.add_button("Bắt đầu", "primary", default=True, on_click=self.accept)
        if self.column.count():
            self._reload_values(self.column.currentText())

    def _reload_values(self, column: str) -> None:
        values = [str(v) for v in self.columns_values.get(column, [])]
        self.value_a.clear()
        self.value_b.clear()
        self.value_a.addItems(values)
        self.value_b.addItems(values)
        if len(values) >= 2:
            self.value_a.setCurrentIndex(0)
            self.value_b.setCurrentIndex(1)

    def get_spec(self) -> dict:
        return {
            "column": self.column.currentText(),
            "value_a": self.value_a.currentText(),
            "value_b": self.value_b.currentText(),
            "bootstrap_subsamples": self.subsamples.value(),
            "permutations": self.subsamples.value(),
            "random_seed": self.seed.value(),
            "weighting_scheme": "path",
        }


class PredictDialog(PremiumDialog):
    """Settings for PLSpredict / cross-validated Q² (folds, repetitions, seed)."""

    def __init__(self, parent=None, title: str = "PLS Predict"):
        super().__init__(
            title,
            "Kiểm định chéo k-fold ngoài mẫu (out-of-sample).",
            icon_name="analysis", parent=parent, width=440,
        )
        self.folds = QSpinBox()
        self.folds.setRange(2, 50)
        self.folds.setValue(10)
        self.add_row("Số fold (k)", self.folds)
        self.reps = QSpinBox()
        self.reps.setRange(1, 100)
        self.reps.setValue(10)
        self.add_row("Số lần lặp lại", self.reps)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(12345)
        self.add_row("Seed", self.seed)
        self.add_button("Hủy", "secondary", on_click=self.reject)
        self.add_button("Bắt đầu", "primary", default=True, on_click=self.accept)

    def get_settings(self) -> dict:
        return {
            "folds": self.folds.value(),
            "repetitions": self.reps.value(),
            "random_seed": self.seed.value(),
            "weighting_scheme": "path",
        }


class EffectDialog(PremiumDialog):
    """Choose constructs for a moderating (interaction) or quadratic effect, SmartPLS-style."""

    def __init__(self, kind: str, constructs: list[tuple[str, str]], preselected: list[str] | None = None, parent=None):
        self.kind = kind
        self._ids = [cid for cid, _name in constructs]
        names = [name for _cid, name in constructs]
        preselected = preselected or []
        is_mod = kind == "moderating"
        super().__init__(
            "Thêm hiệu ứng điều tiết" if is_mod else "Thêm hiệu ứng bậc hai",
            ("Tạo hiệu ứng điều tiết X×M lên biến phụ thuộc (two-stage)."
             if is_mod else "Tạo hiệu ứng bậc hai X² lên biến phụ thuộc (two-stage)."),
            icon_name="moderating" if is_mod else "quadratic", parent=parent, width=460,
        )

        def combo(default_index: int) -> QComboBox:
            box = QComboBox()
            box.addItems(names)
            if 0 <= default_index < len(names):
                box.setCurrentIndex(default_index)
            return box

        sel_idx = [self._ids.index(cid) for cid in preselected if cid in self._ids]
        if is_mod:
            p = sel_idx[0] if len(sel_idx) > 0 else 0
            m = sel_idx[1] if len(sel_idx) > 1 else (1 if len(names) > 1 else 0)
            y = sel_idx[2] if len(sel_idx) > 2 else (len(names) - 1)
            self.predictor = combo(p)
            self.moderator = combo(m)
            self.outcome = combo(y)
            self.add_row("Biến độc lập (Predictor)", self.predictor)
            self.add_row("Biến điều tiết (Moderator)", self.moderator)
            self.add_row("Biến phụ thuộc (Outcome)", self.outcome)
        else:
            s = sel_idx[0] if len(sel_idx) > 0 else 0
            y = sel_idx[1] if len(sel_idx) > 1 else (len(names) - 1)
            self.source = combo(s)
            self.outcome = combo(y)
            self.add_row("Biến nguồn (Source)", self.source)
            self.add_row("Biến phụ thuộc (Outcome)", self.outcome)

        self.method = QComboBox()
        self.method.addItem("Two-Stage (khuyến nghị)", "two_stage")
        self.method.addItem("Product Indicator", "product")
        self.method.addItem("Orthogonalizing", "ortho")
        self.add_row("Phương pháp", self.method)
        self.add_note("Hiện engine tính theo two-stage cho mọi phương pháp; "
                      "product / orthogonalizing sẽ bổ sung sau.")
        self.add_button("Hủy", "secondary", on_click=self.reject)
        self.add_button("Tạo hiệu ứng", "primary", default=True, on_click=self.accept)

    def get_spec(self) -> dict:
        method = self.method.currentData()
        if self.kind == "moderating":
            return {
                "predictor": self._ids[self.predictor.currentIndex()],
                "moderator": self._ids[self.moderator.currentIndex()],
                "outcome": self._ids[self.outcome.currentIndex()],
                "method": method,
            }
        return {
            "source": self._ids[self.source.currentIndex()],
            "outcome": self._ids[self.outcome.currentIndex()],
            "method": method,
        }


class PLSSetupDialog(PremiumDialog):
    def __init__(self, parent=None):
        super().__init__(
            "Thuật toán PLS",
            "Thiết lập lần chạy PLS-SEM: sơ đồ trọng số path/factor/centroid, "
            "tiêu chí hội tụ, bootstrap percentile và xử lý dữ liệu.",
            icon_name="calculate", parent=parent, width=760,
        )
        self.resize(760, 560)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._algorithm_tab(), "Thuật toán")
        self.tabs.addTab(self._bootstrap_tab(), "Bootstrap")
        self.tabs.addTab(self._planned_tab("Blindfolding / Q²", "Stone-Geisser Q² sẽ được bổ sung ở mô-đun phân tích tiếp theo."), "Blindfolding")
        self.tabs.addTab(self._planned_tab("PLSpredict", "Đã hỗ trợ đầy đủ. Mở từ menu Tính toán → PLS Predict: kiểm định chéo k-fold ngoài mẫu với RMSE, MAE, MAPE, Q²predict ở cấp chỉ báo (MV) và biến tiềm ẩn (LV), so sánh benchmark LM và thống kê mô tả sai số/dự báo."), "PLSpredict")
        self.tabs.addTab(self._planned_tab("MGA / MICOM", "So sánh nhóm sẽ được bật sau khi có trình chọn biến nhóm."), "MGA/MICOM")
        self.tabs.addTab(self._advanced_tab(), "Nâng cao")
        self.add_widget(self.tabs, 1)

        self.add_button("Đóng", "secondary", on_click=self.reject)
        self.start_button = self.add_button("Bắt đầu tính", "primary", default=True, on_click=self.accept)

    def _algorithm_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(14)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["PLS-SEM", "Sum scores / OLS"])
        layout.addRow(self._lbl("Thuật toán"), self.algorithm_combo)

        self.scheme_group = QButtonGroup(self)
        scheme_box = QWidget()
        scheme_layout = QHBoxLayout(scheme_box)
        scheme_layout.setContentsMargins(0, 0, 0, 0)
        scheme_layout.setSpacing(18)
        self.radio_path = QRadioButton("Path")
        self.radio_factor = QRadioButton("Factor")
        self.radio_centroid = QRadioButton("Centroid")
        self.radio_path.setChecked(True)
        for radio in (self.radio_path, self.radio_factor, self.radio_centroid):
            self.scheme_group.addButton(radio)
            scheme_layout.addWidget(radio)
        scheme_layout.addStretch()
        layout.addRow(self._lbl("Sơ đồ trọng số"), scheme_box)

        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 10000)
        self.spin_iter.setValue(300)
        layout.addRow(self._lbl("Số vòng lặp tối đa"), self.spin_iter)

        self.spin_stop = QSpinBox()
        self.spin_stop.setRange(1, 15)
        self.spin_stop.setValue(7)
        layout.addRow(self._lbl("Tiêu chí dừng (10^-X)"), self.spin_stop)
        return page

    def _bootstrap_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(14)

        self.bootstrap_enabled = QCheckBox("Chạy bootstrap sau thuật toán PLS")
        self.bootstrap_enabled.setChecked(False)
        layout.addRow(self._lbl("Bật"), self.bootstrap_enabled)

        self.bootstrap_subsamples = QSpinBox()
        self.bootstrap_subsamples.setRange(50, 10000)
        self.bootstrap_subsamples.setValue(2000)
        self.bootstrap_subsamples.setSingleStep(500)
        layout.addRow(self._lbl("Số mẫu con"), self.bootstrap_subsamples)

        presets = QWidget()
        presets_row = QHBoxLayout(presets)
        presets_row.setContentsMargins(0, 0, 0, 0)
        presets_row.setSpacing(6)
        for label, value in [("500 (nhanh)", 500), ("1.000", 1000), ("2.000", 2000), ("5.000 (~3 phút)", 5000)]:
            chip = QPushButton(label)
            chip.setCursor(Qt.PointingHandCursor)
            chip.clicked.connect(lambda _checked=False, v=value: self.bootstrap_subsamples.setValue(v))
            presets_row.addWidget(chip)
        presets_row.addStretch()
        layout.addRow(self._lbl("Nhanh chọn"), presets)

        self.amount_combo = QComboBox()
        self.amount_combo.addItem("Đầy đủ (Complete)", "complete")
        self.amount_combo.addItem("Quan trọng nhất (nhanh hơn)", "important")
        layout.addRow(self._lbl("Lượng kết quả"), self.amount_combo)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 2147483647)
        self.seed_input.setValue(12345)
        layout.addRow(self._lbl("Seed ngẫu nhiên"), self.seed_input)

        self.confidence_level = QDoubleSpinBox()
        self.confidence_level.setRange(0.50, 0.999)
        self.confidence_level.setDecimals(3)
        self.confidence_level.setSingleStep(0.01)
        self.confidence_level.setValue(0.95)
        layout.addRow(self._lbl("Mức tin cậy"), self.confidence_level)

        self.ci_method = QComboBox()
        self.ci_method.addItem("Percentile", "percentile")
        self.ci_method.addItem("Bias-Corrected (BCa)", "bca")
        layout.addRow(self._lbl("Phương pháp khoảng tin cậy"), self.ci_method)

        self.test_type = QComboBox()
        self.test_type.addItem("Hai phía (two-tailed)", "two-tailed")
        self.test_type.addItem("Một phía (one-tailed)", "one-tailed")
        layout.addRow(self._lbl("Kiểu kiểm định"), self.test_type)

        note = QLabel("Mọi kết quả (hệ số đường dẫn, tác động gián tiếp cụ thể, HTMT, "
                      "model fit, biểu đồ…) đều có 4 tab con: Mean/STDEV/T/P, Confidence "
                      "Intervals, Bias Corrected và Samples. Khuyến nghị 2.000 mẫu con để "
                      "ra kết quả dưới 3 phút.")
        note.setObjectName("HintLabel")
        note.setWordWrap(True)
        layout.addRow(note)
        return page

    def _advanced_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(14)

        self.missing_strategy = QComboBox()
        self.missing_strategy.addItem("Loại bỏ dòng thiếu", "casewise")
        self.missing_strategy.addItem("Thay bằng trung bình", "mean")
        self.missing_strategy.addItem("Pairwise (quy về casewise)", "pairwise")
        layout.addRow(self._lbl("Xử lý giá trị thiếu"), self.missing_strategy)

        self.standardized = QCheckBox("Trả kết quả chuẩn hóa")
        self.standardized.setChecked(True)
        self.standardized.setEnabled(False)
        layout.addRow(self._lbl("Đầu ra"), self.standardized)

        self.parallel = QCheckBox("Bootstrap song song")
        self.parallel.setChecked(True)
        layout.addRow(self._lbl("Hiệu năng"), self.parallel)
        return page

    def _planned_tab(self, title: str, body: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 20, 18, 18)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 12pt; font-weight: 800;")
        label = QLabel(body)
        label.setObjectName("HintLabel")
        label.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(label)
        layout.addStretch()
        return page

    @staticmethod
    def _lbl(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    def set_bootstrap_mode(self, enabled: bool = True) -> None:
        """Focus the dialog on bootstrapping when launched from the Bootstrapping command."""
        self.bootstrap_enabled.setChecked(enabled)
        if enabled:
            self.setWindowTitle("Bootstrapping")
            self.algorithm_combo.setCurrentText("PLS-SEM")
            self.algorithm_combo.setEnabled(False)
            self.bootstrap_enabled.setEnabled(False)
            self.tabs.setCurrentIndex(1)
            self.start_button.setText("Bắt đầu Bootstrapping")

    def get_settings(self) -> dict:
        scheme = "path"
        if self.radio_centroid.isChecked():
            scheme = "centroid"
        elif self.radio_factor.isChecked():
            scheme = "factor"
        return {
            "algorithm": self.algorithm_combo.currentText(),
            "weighting_scheme": scheme,
            "max_iterations": self.spin_iter.value(),
            "stop_criterion": 10 ** -self.spin_stop.value(),
            "missing_strategy": self.missing_strategy.currentData(),
            "bootstrap_enabled": self.bootstrap_enabled.isChecked()
            and self.algorithm_combo.currentText() == "PLS-SEM",
            "bootstrap_subsamples": self.bootstrap_subsamples.value(),
            "random_seed": self.seed_input.value(),
            "confidence_level": self.confidence_level.value(),
            "test_type": self.test_type.currentData(),
            "ci_method": self.ci_method.currentData(),
            "bootstrap_amount": self.amount_combo.currentData(),
            "bootstrap_parallel": self.parallel.isChecked(),
        }

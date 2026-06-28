"""The "Phi tuyến tính" (Nonlinear ML) workspace.

A premium, dashboard-style center tab that drives the full nonlinear pipeline
(XGBoost → SHAP → Symbolic Regression → Sobol sensitivity → Optimization) on
top of :class:`core.nonlinear_engine.NonlinearEngine`. Heavy work runs on a
``QThread`` worker; charts arrive as pre-rendered PNG bytes and are shown in
elegant chart cards, so nothing blocks or paints across threads.

Layout: a persistent left nav rail of pipeline stages + a stacked stage area.
Each stage is a split view — a scrollable column of config section cards with a
*pinned* violet run button as its footer (left), and a results panel (right)
that shows a rich "what this step does / what you'll get" explainer card until
the stage has been run, then swaps to the results. All styling is driven by
object-names defined in ``gui/theme.py``.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QLocale, QObject, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QClipboard, QColor, QFont, QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui.dialogs import PremiumDialog
from gui.icons import icon
from gui.theme import MONO_FAMILY, apply_shadow

STAGES = [
    ("data", "Dữ liệu", "nl-data", "Nạp & chọn biến mục tiêu / đầu vào"),
    ("xgboost", "XGBoost", "xgboost", "Huấn luyện + dò siêu tham số (GridSearchCV)"),
    ("shap", "SHAP", "shap", "Diễn giải mức độ ảnh hưởng của biến"),
    ("symbolic", "Hồi quy biểu thức", "symbolic", "Tìm công thức toán học (PySR)"),
    ("sensitivity", "Độ nhạy Sobol", "sensitivity", "Chỉ số S1 / ST & hướng tác động"),
    ("optimize", "Tối ưu hóa", "optimize", "Tìm cực trị của mô hình"),
    ("report", "Báo cáo", "report", "Tổng hợp toàn bộ kết quả"),
    ("deps", "Thư viện", "deps", "Trạng thái phụ thuộc ML"),
]

DEP_PURPOSE = {
    "sklearn": ("scikit-learn", "Chia dữ liệu, GridSearchCV, chỉ số đánh giá", "pip install scikit-learn"),
    "xgboost": ("XGBoost", "Mô hình gradient boosting hồi quy", "pip install xgboost"),
    "shap": ("SHAP", "Diễn giải mức ảnh hưởng của từng biến", "pip install shap"),
    "SALib": ("SALib", "Phân tích độ nhạy Sobol", "pip install SALib"),
    "sympy": ("SymPy", "Biểu diễn công thức toán học", "pip install sympy"),
    "matplotlib": ("Matplotlib", "Vẽ toàn bộ biểu đồ", "pip install matplotlib"),
    "statsmodels": ("statsmodels", "Đường LOWESS trên biểu đồ SHAP", "pip install statsmodels"),
    "pysr": ("PySR", "Hồi quy biểu thức (cần Julia)", "pip install pysr"),
}


# --------------------------------------------------------------------------- #
# Small premium primitives
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def metric_chip(text: str, tone: str = "default") -> QLabel:
    chip = QLabel(text)
    chip.setObjectName("MetricChip")
    if tone != "default":
        chip.setProperty("tone", tone)
    chip.setAlignment(Qt.AlignCenter)
    return chip


def _r2_tone(value: float) -> str:
    if value >= 0.75:
        return "good"
    if value >= 0.5:
        return "warn"
    return "bad"


def stat_card(caption: str, value: str, tone: str = "default", foot: str = "") -> QFrame:
    card = QFrame()
    card.setObjectName("StatCard")
    if tone != "default":
        card.setProperty("tone", tone)
    apply_shadow(card, blur=18, y=4, color="#1A2A4A1F")
    box = QVBoxLayout(card)
    box.setContentsMargins(16, 13, 16, 13)
    box.setSpacing(3)
    cap = QLabel(caption)
    cap.setObjectName("StatCaption")
    val = QLabel(value)
    val.setObjectName("StatValue")
    box.addWidget(cap)
    box.addWidget(val)
    if foot:
        ft = QLabel(foot)
        ft.setObjectName("StatFoot")
        ft.setWordWrap(True)
        box.addWidget(ft)
    return card


def _section_card(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    """A premium section card: accent tick + title + optional one-line subtitle."""
    card = QFrame()
    card.setObjectName("NLCard")
    apply_shadow(card, blur=20, y=5, color="#1A2A4A14")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(20, 17, 20, 18)
    outer.setSpacing(14)

    head = QHBoxLayout()
    head.setContentsMargins(0, 0, 0, 0)
    head.setSpacing(11)
    tick = QFrame()
    tick.setObjectName("NLSectionTick")
    tick.setFixedWidth(4)
    tick.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
    head.addWidget(tick)
    titles = QVBoxLayout()
    titles.setContentsMargins(0, 0, 0, 0)
    titles.setSpacing(2)
    tlbl = QLabel(title)
    tlbl.setObjectName("NLCardTitle")
    titles.addWidget(tlbl)
    if subtitle:
        slbl = QLabel(subtitle)
        slbl.setObjectName("NLSectionDesc")
        slbl.setWordWrap(True)
        titles.addWidget(slbl)
    head.addLayout(titles, 1)
    outer.addLayout(head)
    return card, outer


def _field_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("NLFieldLabel")
    return lbl


def _field_input(widget: QWidget) -> QWidget:
    """Give every form control one consistent height + an expanding width."""
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
    widget.setMinimumHeight(34)
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return widget


def _form_grid() -> QGridLayout:
    """One shared 2-column grid so every label/field lines up perfectly."""
    grid = QGridLayout()
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(11)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setColumnMinimumWidth(0, 128)
    grid.setColumnStretch(0, 0)
    grid.setColumnStretch(1, 1)
    return grid


def _grid_row(grid: QGridLayout, row: int, label: str, widget: QWidget) -> None:
    lbl = _field_label(label)
    grid.addWidget(lbl, row, 0, Qt.AlignLeft | Qt.AlignVCenter)
    grid.addWidget(_field_input(widget), row, 1)


def stage_intro(icon_name: str, title: str, desc: str,
                outputs: list[str], cta: str) -> QWidget:
    """The right-pane explainer shown before a stage has produced results.

    Replaces the old tiny floating icon: a real card that fills the pane and
    tells the user exactly what the step does and what they'll get back.
    """
    page = QWidget()
    wrap = QVBoxLayout(page)
    wrap.setContentsMargins(8, 8, 8, 8)
    wrap.setSpacing(0)
    wrap.addStretch(1)

    card = QFrame()
    card.setObjectName("NLIntroCard")
    # Fixed width so word-wrapped labels compute their height correctly
    # (heightForWidth misfires when a wrapping QLabel is centered via an align flag).
    card.setFixedWidth(516)
    apply_shadow(card, blur=28, y=9, color="#3A2A6A1A")
    box = QVBoxLayout(card)
    box.setContentsMargins(34, 32, 34, 30)
    box.setSpacing(15)

    chip = QLabel()
    chip.setObjectName("NLIntroChip")
    chip.setFixedSize(60, 60)
    chip.setAlignment(Qt.AlignCenter)
    chip.setPixmap(icon(icon_name, 30).pixmap(30, 30))
    box.addWidget(chip, 0, Qt.AlignLeft)

    tlbl = QLabel(title)
    tlbl.setObjectName("NLIntroTitle")
    tlbl.setWordWrap(True)
    box.addWidget(tlbl)

    dlbl = QLabel(desc)
    dlbl.setObjectName("NLIntroDesc")
    dlbl.setWordWrap(True)
    box.addWidget(dlbl)

    olbl = QLabel("BẠN SẼ NHẬN ĐƯỢC")
    olbl.setObjectName("NLIntroOutLabel")
    box.addSpacing(2)
    box.addWidget(olbl)
    for text in outputs:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(11)
        dot = QFrame()
        dot.setObjectName("NLIntroDot")
        dot.setFixedSize(6, 6)
        col = QVBoxLayout()
        col.setContentsMargins(0, 6, 0, 0)
        col.addWidget(dot)
        col.addStretch()
        dot_host = QWidget()
        dot_host.setLayout(col)
        dot_host.setFixedWidth(6)
        row.addWidget(dot_host, 0, Qt.AlignTop)
        ol = QLabel(text)
        ol.setObjectName("NLIntroOut")
        ol.setWordWrap(True)
        row.addWidget(ol, 1)
        row_host = QWidget()
        row_host.setLayout(row)
        box.addWidget(row_host)

    if cta:
        box.addSpacing(4)
        bar = QFrame()
        bar.setObjectName("NLIntroCtaBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 11, 14, 11)
        bl.setSpacing(10)
        play = QLabel()
        play.setPixmap(icon("run-nl", 18).pixmap(18, 18))
        bl.addWidget(play, 0, Qt.AlignVCenter)
        ctal = QLabel(cta)
        ctal.setObjectName("NLIntroCta")
        ctal.setWordWrap(True)
        bl.addWidget(ctal, 1)
        box.addWidget(bar)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch(1)
    row.addWidget(card)
    row.addStretch(1)
    wrap.addLayout(row)
    wrap.addStretch(1)
    return page


STAGE_INTRO: dict[str, tuple] = {
    "data": (
        "nl-data", "Nạp dữ liệu cho phân tích phi tuyến",
        "Chọn tệp CSV/Excel hoặc dùng dữ liệu đang mở trong app, rồi chỉ định "
        "biến mục tiêu (Y) và các biến đầu vào (X).",
        ["Tóm tắt số dòng, số biến và giá trị thiếu",
         "Bảng xem trước 200 dòng đầu của dữ liệu",
         "Khởi tạo engine dùng chung cho toàn bộ quy trình"],
        "Cấu hình bên trái rồi nhấn “Nạp và xác nhận dữ liệu”.",
    ),
    "xgboost": (
        "xgboost", "Huấn luyện mô hình XGBoost",
        "Gradient boosting học quan hệ phi tuyến giữa các biến đầu vào và mục tiêu, "
        "có thể tự dò siêu tham số bằng GridSearchCV.",
        ["R² cho tập Train · CV · Test",
         "Bảng RMSE và MAE chi tiết",
         "Bộ siêu tham số tối ưu",
         "Biểu đồ Dự đoán vs Thực tế và Độ quan trọng (Gain)"],
        "Chỉnh lưới tham số rồi nhấn “Huấn luyện mô hình”.",
    ),
    "shap": (
        "shap", "Diễn giải mô hình bằng SHAP",
        "TreeExplainer tính đóng góp của từng biến cho mỗi dự đoán — cho biết biến nào "
        "quan trọng và tác động theo hướng nào.",
        ["Biểu đồ Beeswarm tổng quan",
         "% tầm quan trọng của từng biến",
         "Scatter SHAP kèm đường LOWESS",
         "Bảng xếp hạng mức đóng góp"],
        "Nhấn “Tính SHAP” (cần đã huấn luyện XGBoost trước).",
    ),
    "symbolic": (
        "symbolic", "Tìm công thức toán học (PySR)",
        "Hồi quy biểu thức tiến hóa hàng triệu phương trình để tìm công thức tường minh "
        "xấp xỉ mô hình.",
        ["Công thức tối ưu dạng tường minh",
         "Hệ số xác định R² của công thức",
         "Mặt trận Pareto: độ phức tạp – sai số",
         "Bảng các công thức ứng viên"],
        "Đặt tham số rồi nhấn “Tìm công thức”.",
    ),
    "sensitivity": (
        "sensitivity", "Phân tích độ nhạy Sobol",
        "Lượng hóa mức ảnh hưởng của từng biến và tương tác giữa các biến lên đầu ra "
        "qua chỉ số Sobol bậc một (S1) và tổng (ST).",
        ["Chỉ số S1 và ST cho từng biến",
         "Đường hội tụ theo cỡ mẫu N",
         "Hướng tác động (tăng/giảm) của mỗi biến",
         "Bảng mức tương tác ST − S1"],
        "Chọn mô hình và cỡ mẫu rồi nhấn “Chạy Sobol”.",
    ),
    "optimize": (
        "optimize", "Tối ưu hóa mô hình",
        "Differential Evolution tìm tổ hợp biến đầu vào cho giá trị mục tiêu lớn nhất "
        "và nhỏ nhất trên miền dữ liệu thực tế.",
        ["Giá trị lớn nhất và nhỏ nhất của mục tiêu",
         "Cấu hình biến tại điểm tối ưu",
         "Biểu đồ profile các biến"],
        "Chọn hàm mục tiêu rồi nhấn “Tìm tối ưu”.",
    ),
    "report": (
        "report", "Báo cáo tổng hợp",
        "Tổng hợp toàn bộ kết quả của các bước đã chạy thành một báo cáo gọn gàng, "
        "kèm tùy chọn xuất tất cả biểu đồ.",
        ["Bảng tiến trình các bước đã hoàn thành",
         "Các chỉ số R² nổi bật",
         "Công thức tối ưu (nếu có)",
         "Xuất toàn bộ biểu đồ ra PNG"],
        "Nhấn “Cập nhật báo cáo” để tổng hợp.",
    ),
}


def chart_card(title: str, png: bytes, max_width: int = 620) -> QFrame:
    card = QFrame()
    card.setObjectName("ChartCard")
    apply_shadow(card, blur=20, y=5, color="#1A2A4A1A")
    box = QVBoxLayout(card)
    box.setContentsMargins(16, 14, 16, 16)
    box.setSpacing(10)
    head = QHBoxLayout()
    tlbl = QLabel(title)
    tlbl.setObjectName("ChartTitle")
    head.addWidget(tlbl)
    head.addStretch()
    save = QToolButton()
    save.setObjectName("PanelButton")
    save.setIcon(icon("image", 18))
    save.setIconSize(QSize(16, 16))
    save.setToolTip("Lưu ảnh PNG")
    save.setCursor(Qt.PointingHandCursor)

    def _save() -> None:
        path, _ = QFileDialog.getSaveFileName(card, "Lưu biểu đồ", f"{title}.png", "PNG (*.png)")
        if path:
            Path(path).write_bytes(png)

    save.clicked.connect(_save)
    head.addWidget(save)
    box.addLayout(head)

    image = QLabel()
    image.setObjectName("ChartImage")
    image.setAlignment(Qt.AlignCenter)
    pixmap = QPixmap()
    pixmap.loadFromData(png, "PNG")
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    image.setPixmap(pixmap)
    box.addWidget(image)
    return card


def empty_state(icon_name: str, title: str, body: str) -> QWidget:
    page = QWidget()
    box = QVBoxLayout(page)
    box.setAlignment(Qt.AlignCenter)
    box.setSpacing(10)
    chip = QLabel()
    chip.setObjectName("NLEmptyChip")
    chip.setFixedSize(72, 72)
    chip.setAlignment(Qt.AlignCenter)
    chip.setPixmap(icon(icon_name, 34).pixmap(34, 34))
    box.addWidget(chip, 0, Qt.AlignHCenter)
    tlbl = QLabel(title)
    tlbl.setObjectName("NLEmptyTitle")
    tlbl.setAlignment(Qt.AlignCenter)
    box.addWidget(tlbl)
    blbl = QLabel(body)
    blbl.setObjectName("NLEmptyBody")
    blbl.setAlignment(Qt.AlignCenter)
    blbl.setWordWrap(True)
    box.addWidget(blbl)
    return page


def result_table(headers: list[str], rows: list[list[str]]) -> QTableWidget:
    table = QTableWidget(len(rows), len(headers))
    table.setObjectName("ResultTable")
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().hide()
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionMode(QTableWidget.NoSelection)
    table.setShowGrid(False)
    two_col = len(headers) == 2
    if two_col:
        # Key/value: key hugs the left, value sits right next to it (not far right).
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
    else:
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, len(headers)):
            table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
    for r, row in enumerate(rows):
        for col, value in enumerate(row):
            item = QTableWidgetItem(str(value))
            if col > 0:
                align = Qt.AlignLeft if two_col else Qt.AlignRight
                item.setTextAlignment(align | Qt.AlignVCenter)
            table.setItem(r, col, item)
    table.resizeRowsToContents()
    height = table.horizontalHeader().height() + sum(table.rowHeight(r) for r in range(len(rows))) + 8
    table.setFixedHeight(min(height, 360))
    return table


# --------------------------------------------------------------------------- #
# Threaded worker + progress dialog
# --------------------------------------------------------------------------- #
class NLWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[Callable[[str], None]], Any]) -> None:
        super().__init__()
        self._fn = fn

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(self.progress.emit)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()[-1400:]}")


class NLProgressDialog(PremiumDialog):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, "Đang xử lý — vui lòng đợi trong giây lát.",
                         icon_name="run-nl", parent=parent, width=520)
        self._start = time.monotonic()
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # busy/indeterminate
        self.bar.setFixedHeight(16)
        self.add_widget(self.bar)
        self.status = QLabel("Đang chuẩn bị...")
        self.status.setWordWrap(True)
        self.add_widget(self.status)
        self.elapsed = QLabel("Thời gian: 00:00")
        self.elapsed.setObjectName("HintLabel")
        self.add_widget(self.elapsed)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        secs = int(time.monotonic() - self._start)
        self.elapsed.setText(f"Thời gian: {secs // 60:02d}:{secs % 60:02d}")

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def reject(self) -> None:  # block the ✕ / Esc while a job runs
        pass

    def finish(self) -> None:
        self._timer.stop()
        self.accept()


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #
class NonlinearWorkspace(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("NonlinearWorkspace")
        self.engine = None
        self._app_frame = None
        self._app_name = ""
        self._raw_frame = None
        self._raw_name = ""
        self._results: dict[str, Any] = {}
        self._job_running = False
        self._nav_buttons: dict[str, QToolButton] = {}
        self._result_stacks: dict[str, QStackedWidget] = {}
        self._result_bodies: dict[str, QVBoxLayout] = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)
        root.addWidget(self._build_nav())

        right = QVBoxLayout()
        right.setSpacing(12)
        right.addWidget(self._build_header())
        self.stack = QStackedWidget()
        right.addWidget(self.stack, 1)
        root.addLayout(right, 1)

        self._stage_index: dict[str, int] = {}
        for key, label, icon_name, _hint in STAGES:
            page = self._build_stage(key, label, icon_name)
            self._stage_index[key] = self.stack.addWidget(page)

        self._select_stage("data")
        self._refresh_gating()

    # ---- nav rail --------------------------------------------------------- #
    def _build_nav(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("NLNavRail")
        rail.setFixedWidth(246)
        box = QVBoxLayout(rail)
        box.setContentsMargins(14, 16, 14, 16)
        box.setSpacing(3)

        brand = QHBoxLayout()
        brand.setSpacing(11)
        chip = QLabel()
        chip.setObjectName("NLBrandChip")
        chip.setFixedSize(40, 40)
        chip.setAlignment(Qt.AlignCenter)
        chip.setPixmap(icon("nonlinear", 24).pixmap(24, 24))
        brand.addWidget(chip, 0, Qt.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        t = QLabel("Phi tuyến tính")
        t.setObjectName("NLBrandTitle")
        s = QLabel("XGBoost · SHAP · PySR · Sobol")
        s.setObjectName("NLBrandSub")
        s.setWordWrap(True)
        titles.addWidget(t)
        titles.addWidget(s)
        brand.addLayout(titles, 1)
        box.addLayout(brand)
        box.addSpacing(14)

        section = QLabel("QUY TRÌNH")
        section.setObjectName("NLNavSection")
        box.addWidget(section)
        box.addSpacing(2)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for key, label, icon_name, _hint in STAGES:
            if key == "deps":
                box.addStretch(1)
                line = QFrame()
                line.setObjectName("HLine")
                line.setFixedHeight(1)
                box.addWidget(line)
                box.addSpacing(8)
                tools = QLabel("CÔNG CỤ")
                tools.setObjectName("NLNavSection")
                box.addWidget(tools)
                box.addSpacing(2)
            btn = QToolButton()
            btn.setObjectName("NLNavItem")
            btn.setText(label)
            btn.setIcon(icon(icon_name, 20))
            btn.setIconSize(QSize(19, 19))
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _=False, k=key: self._select_stage(k))
            self._nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            box.addWidget(btn)
        return rail

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("NLStageHeader")
        apply_shadow(header, blur=18, y=4, color="#1A2A4A18")
        row = QHBoxLayout(header)
        row.setContentsMargins(22, 16, 20, 16)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.header_kicker = QLabel("BƯỚC 1 / 7")
        self.header_kicker.setObjectName("NLKicker")
        self.header_title = QLabel("Dữ liệu")
        self.header_title.setObjectName("NLStageTitle")
        self.header_sub = QLabel("Nạp dữ liệu và chọn biến.")
        self.header_sub.setObjectName("NLStageSub")
        titles.addWidget(self.header_kicker)
        titles.addWidget(self.header_title)
        titles.addWidget(self.header_sub)
        row.addLayout(titles, 1)
        self.header_chip = metric_chip("Chưa chạy", "muted")
        row.addWidget(self.header_chip, 0, Qt.AlignTop)
        return header

    # ---- generic stage shell --------------------------------------------- #
    def _build_stage(self, key: str, label: str, icon_name: str) -> QWidget:
        if key == "deps":
            return self._build_deps_stage()
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("NLSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(14)

        # ---- left: config sections (scroll) + a pinned run footer ----------- #
        panel = QFrame()
        panel.setObjectName("NLConfigPanel")
        panel_box = QVBoxLayout(panel)
        panel_box.setContentsMargins(0, 0, 0, 0)
        panel_box.setSpacing(12)

        config = QScrollArea()
        config.setWidgetResizable(True)
        config.setFrameShape(QFrame.NoFrame)
        config.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        config_host = QWidget()
        config_layout = QVBoxLayout(config_host)
        config_layout.setContentsMargins(2, 2, 10, 2)
        config_layout.setSpacing(12)
        builder = getattr(self, f"_config_{key}")
        primary = builder(config_layout)
        config_layout.addStretch(1)
        config.setWidget(config_host)
        panel_box.addWidget(config, 1)

        if primary is not None:
            footer = QFrame()
            footer.setObjectName("NLRunFooter")
            apply_shadow(footer, blur=22, y=-3, color="#1A2A4A14")
            fbox = QVBoxLayout(footer)
            fbox.setContentsMargins(14, 13, 14, 13)
            fbox.addWidget(primary)
            panel_box.addWidget(footer)
        splitter.addWidget(panel)

        # ---- right: explainer (before run) → results (after run) ----------- #
        results_stack = QStackedWidget()
        results_stack.addWidget(stage_intro(*STAGE_INTRO[key]))
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setFrameShape(QFrame.NoFrame)
        results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        results_host = QWidget()
        results_body = QVBoxLayout(results_host)
        results_body.setContentsMargins(2, 2, 10, 2)
        results_body.setSpacing(14)
        results_body.addStretch(1)
        results_scroll.setWidget(results_host)
        results_stack.addWidget(results_scroll)
        self._result_stacks[key] = results_stack
        self._result_bodies[key] = results_body
        splitter.addWidget(results_stack)

        panel.setMinimumWidth(386)
        panel.setMaximumWidth(470)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 980])
        return splitter

    # ---- button factories ------------------------------------------------ #
    def _primary_button(self, label: str, callback, attr: str,
                        icon_name: str = "run-nl") -> QPushButton:
        button = QPushButton(label)
        button.setObjectName("NLPrimaryButton")
        button.setIcon(icon(icon_name, 18))
        button.setIconSize(QSize(18, 18))
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(46)
        button.clicked.connect(callback)
        setattr(self, attr, button)
        return button

    def _secondary_button(self, label: str, callback, icon_name: str | None = None,
                         accent: bool = False) -> QPushButton:
        button = QPushButton(label)
        button.setObjectName("NLSecondaryButton")
        if accent:
            button.setProperty("accent", "true")
        if icon_name:
            button.setIcon(icon(icon_name, 18))
            button.setIconSize(QSize(17, 17))
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(40)
        button.clicked.connect(callback)
        return button

    def _link_button(self, label: str, callback) -> QToolButton:
        button = QToolButton()
        button.setObjectName("NLLinkButton")
        button.setText(label)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _clear_results(self, key: str) -> None:
        body = self._result_bodies[key]
        while body.count() > 1:  # keep the trailing stretch
            item = body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _add_result(self, key: str, widget: QWidget) -> None:
        body = self._result_bodies[key]
        body.insertWidget(body.count() - 1, widget)

    def _show_results(self, key: str) -> None:
        self._result_stacks[key].setCurrentIndex(1)

    def _stat_row(self, cards: list[QFrame]) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        for card in cards:
            row.addWidget(card, 1)
        return host

    # ================================================================== #
    # Stage: DATA
    # ================================================================== #
    def _config_data(self, layout: QVBoxLayout) -> QPushButton:
        card, box = _section_card("Nguồn dữ liệu",
                                  "Chọn tệp hoặc dùng dữ liệu đang mở trong app")
        pick = self._secondary_button("Chọn tệp CSV / Excel…", self._pick_csv,
                                      icon_name="nl-data", accent=True)
        box.addWidget(pick)
        self.use_app_btn = self._secondary_button("Dùng dữ liệu đang mở trong app",
                                                  self._use_app_data, icon_name="data")
        box.addWidget(self.use_app_btn)
        self.data_path_label = QLabel("Chưa chọn dữ liệu.")
        self.data_path_label.setObjectName("NLFieldHint")
        self.data_path_label.setWordWrap(True)
        box.addWidget(self.data_path_label)

        grid = _form_grid()
        self.sep_combo = QComboBox()
        self.sep_combo.addItems(["; (chấm phẩy)", ", (phẩy)", "Tab", "| (gạch đứng)"])
        self.dec_combo = QComboBox()
        self.dec_combo.addItems([", (phẩy)", ". (chấm)"])
        _grid_row(grid, 0, "Dấu phân cách", self.sep_combo)
        _grid_row(grid, 1, "Dấu thập phân", self.dec_combo)
        box.addLayout(grid)
        layout.addWidget(card)

        vcard, vbox = _section_card("Chọn biến",
                                    "Biến mục tiêu (Y) và các biến đầu vào (X)")
        vbox.addWidget(_field_label("Biến mục tiêu (Y)"))
        self.target_combo = QComboBox()
        _field_input(self.target_combo)
        vbox.addWidget(self.target_combo)
        head = QHBoxLayout()
        head.setContentsMargins(0, 4, 0, 0)
        head.addWidget(_field_label("Biến đầu vào (X)"))
        head.addStretch()
        head.addWidget(self._link_button("Chọn tất cả", lambda: self._set_all_features(True)))
        head.addWidget(self._link_button("Bỏ chọn", lambda: self._set_all_features(False)))
        vbox.addLayout(head)
        self.feature_list = QListWidget()
        self.feature_list.setMinimumHeight(150)
        vbox.addWidget(self.feature_list)
        self.target_combo.currentIndexChanged.connect(self._sync_feature_list)
        layout.addWidget(vcard)

        self.data_run = self._primary_button("Nạp và xác nhận dữ liệu",
                                            self._confirm_data, "data_run")
        self.data_run.setEnabled(False)
        return self.data_run

    def _pick_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn tệp dữ liệu", "", "Dữ liệu (*.csv *.txt *.xlsx *.xls);;Tất cả (*.*)")
        if not path:
            return
        try:
            frame = self._read_with_options(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi đọc dữ liệu", str(exc))
            return
        self._raw_frame = frame
        self._raw_name = Path(path).stem
        self.data_path_label.setText(f"{path}\n{len(frame):,} dòng × {len(frame.columns)} cột")
        self._populate_columns(frame)

    def _read_with_options(self, path: str):
        import pandas as pd

        suffix = Path(path).suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        sep = {0: ";", 1: ",", 2: "\t", 3: "|"}[self.sep_combo.currentIndex()]
        dec = {0: ",", 1: "."}[self.dec_combo.currentIndex()]
        frame = pd.read_csv(path, sep=sep, decimal=dec, encoding="utf-8-sig", engine="python")
        frame.columns = [str(c).strip() for c in frame.columns]
        return frame

    def _use_app_data(self) -> None:
        if self._app_frame is None or self._app_frame.empty:
            QMessageBox.information(self, "Chưa có dữ liệu",
                                    "Hãy nhập dữ liệu trong tab Data của app trước, hoặc chọn tệp CSV.")
            return
        self._raw_frame = self._app_frame.copy()
        self._raw_name = self._app_name or "Dữ liệu app"
        self.data_path_label.setText(
            f"Dữ liệu đang mở: {self._raw_name}\n"
            f"{len(self._raw_frame):,} dòng × {len(self._raw_frame.columns)} cột")
        self._populate_columns(self._raw_frame)

    def _populate_columns(self, frame) -> None:
        columns = [str(c) for c in frame.columns]
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems(columns)
        if columns:
            self.target_combo.setCurrentIndex(len(columns) - 1)
        self.target_combo.blockSignals(False)
        self._sync_feature_list()
        self.data_run.setEnabled(True)

    def _sync_feature_list(self) -> None:
        if self._raw_frame is None:
            return
        target = self.target_combo.currentText()
        self.feature_list.clear()
        for col in [str(c) for c in self._raw_frame.columns]:
            if col == target:
                continue
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.feature_list.addItem(item)

    def _set_all_features(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(state)

    def _confirm_data(self) -> None:
        if self._raw_frame is None:
            return
        target = self.target_combo.currentText()
        features = [self.feature_list.item(i).text()
                    for i in range(self.feature_list.count())
                    if self.feature_list.item(i).checkState() == Qt.Checked]
        if not features:
            QMessageBox.warning(self, "Thiếu biến", "Hãy chọn ít nhất một biến đầu vào (X).")
            return
        try:
            from core.nonlinear_engine import NonlinearEngine

            self.engine = NonlinearEngine(self._raw_frame, features, target)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Lỗi dữ liệu", str(exc))
            return
        if self.engine.n_rows < 20:
            QMessageBox.warning(self, "Dữ liệu quá ít",
                                f"Chỉ còn {self.engine.n_rows} dòng sau khi loại bỏ giá trị thiếu.")
        constant = [c for c in self.engine.features
                    if float(self.engine.X[c].min()) == float(self.engine.X[c].max())]
        if constant:
            QMessageBox.warning(self, "Biến không đổi",
                                "Các biến sau không thay đổi giá trị (sẽ ít/không có ý nghĩa "
                                f"trong phân tích): {', '.join(constant)}.")
        # invalidate downstream
        self._results.clear()
        self._render_data_results()
        self.header_chip.setProperty("tone", "good")
        self.header_chip.setText(f"Sẵn sàng · {self.engine.n_rows} dòng")
        self._repolish(self.header_chip)
        self._refresh_gating()

    def _render_data_results(self) -> None:
        key = "data"
        self._clear_results(key)
        eng = self.engine
        miss = 0
        cards = [
            stat_card("Số dòng", f"{eng.n_rows:,}", "accent"),
            stat_card("Biến X", str(len(eng.features))),
            stat_card("Biến mục tiêu", eng.target),
            stat_card("Giá trị thiếu", f"{miss}", "good", foot="đã loại bỏ"),
        ]
        self._add_result(key, self._stat_row(cards))

        card, box = _section_card("Xem trước dữ liệu", "200 dòng đầu")
        preview = eng.frame.head(200)
        headers = [str(c) for c in preview.columns]
        rows = [[_fmt(preview.iloc[r, c], 3) for c in range(len(headers))]
                for r in range(len(preview))]
        table = result_table(headers, rows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setFixedHeight(360)
        box.addWidget(table)
        self._add_result(key, card)
        self._show_results(key)

    # ================================================================== #
    # Stage: XGBOOST
    # ================================================================== #
    def _config_xgboost(self, layout: QVBoxLayout) -> QPushButton:
        card, box = _section_card("Lưới siêu tham số",
                                  "Mỗi ô là danh sách giá trị, phân tách bằng dấu phẩy")
        self.use_grid = QCheckBox("Dò siêu tham số bằng GridSearchCV")
        self.use_grid.setChecked(True)
        box.addWidget(self.use_grid)
        grid = _form_grid()
        self.grid_edits: dict[str, QLineEdit] = {}
        defaults = {
            "max_depth": "3,4,5", "learning_rate": "0.01,0.05,0.1",
            "n_estimators": "200,400", "subsample": "0.7,0.8,1.0",
            "colsample_bytree": "0.7,0.8,1.0", "reg_lambda": "1,5,10",
        }
        for r, (name, value) in enumerate(defaults.items()):
            edit = QLineEdit(value)
            self.grid_edits[name] = edit
            _grid_row(grid, r, name, edit)
        box.addLayout(grid)
        self.fits_chip = QLabel("")
        self.fits_chip.setObjectName("NLInlineHint")
        self.fits_chip.setWordWrap(True)
        box.addWidget(self.fits_chip)
        layout.addWidget(card)

        card2, box2 = _section_card("Đánh giá và tái lập",
                                    "Phân chia dữ liệu và kiểm định chéo")
        opts = _form_grid()
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1, 0.4)
        self.test_size.setSingleStep(0.05)
        self.test_size.setValue(0.20)
        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(3, 10)
        self.cv_folds.setValue(5)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        _grid_row(opts, 0, "Tỉ lệ tập test", self.test_size)
        _grid_row(opts, 1, "Số fold CV", self.cv_folds)
        _grid_row(opts, 2, "Random state", self.seed_spin)
        box2.addLayout(opts)
        layout.addWidget(card2)

        for edit in self.grid_edits.values():
            edit.textChanged.connect(self._update_fits)
        self.cv_folds.valueChanged.connect(self._update_fits)
        self.use_grid.toggled.connect(self._update_fits)
        self._update_fits()
        return self._primary_button("Huấn luyện mô hình", self._run_xgboost, "xgb_run")

    def _update_fits(self) -> None:
        if not self.use_grid.isChecked():
            self.fits_chip.setText(f"Chế độ nhanh · {self.cv_folds.value()} fold")
            return
        try:
            total = 1
            for edit in self.grid_edits.values():
                total *= max(1, len([x for x in edit.text().split(",") if x.strip()]))
            self.fits_chip.setText(f"≈ {total:,} tổ hợp × {self.cv_folds.value()} fold = {total * self.cv_folds.value():,} lần fit")
        except Exception:  # noqa: BLE001
            self.fits_chip.setText("")

    def _parse_grid(self) -> dict[str, list]:
        out: dict[str, list] = {}
        for name, edit in self.grid_edits.items():
            values = []
            for token in edit.text().split(","):
                token = token.strip()
                if not token:
                    continue
                values.append(int(token) if name in {"max_depth", "n_estimators"} else float(token))
            out[name] = values or [QUICK_DEFAULTS[name]]
        return out

    def _run_xgboost(self) -> None:
        if not self._require_engine():
            return
        use_grid = self.use_grid.isChecked()
        grid = self._parse_grid()
        cv = self.cv_folds.value()
        ts = self.test_size.value()
        seed = self.seed_spin.value()

        def job(emit):
            return self.engine.train_xgboost(
                use_grid=use_grid, param_grid=grid, cv_folds=cv,
                test_size=ts, seed=seed, progress=emit)

        self._run_job("Huấn luyện XGBoost", "xgboost", job, self._render_xgboost)

    def _render_xgboost(self, result: dict) -> None:
        key = "xgboost"
        self._results[key] = result
        self._clear_results(key)
        m = result["metrics"]
        cards = [
            stat_card("R² Train", _fmt(m["train_r2"], 3), _r2_tone(m["train_r2"])),
            stat_card("R² CV (5-fold)", _fmt(m["cv_r2"], 3), _r2_tone(m["cv_r2"])),
            stat_card("R² Test", _fmt(m["test_r2"], 3), _r2_tone(m["test_r2"]), foot="dùng để báo cáo"),
        ]
        self._add_result(key, self._stat_row(cards))

        metric_card, mbox = _section_card("Chỉ số chi tiết", "R² · RMSE · MAE")
        rows = [
            ["Train", _fmt(m["train_r2"], 4), "—", "—"],
            ["CV (5-fold)", _fmt(m["cv_r2"], 4), _fmt(m["cv_rmse"], 4), _fmt(m["cv_mae"], 4)],
            ["Test", _fmt(m["test_r2"], 4), _fmt(m["test_rmse"], 4), _fmt(m["test_mae"], 4)],
        ]
        mbox.addWidget(result_table(["Tập", "R²", "RMSE", "MAE"], rows))
        self._add_result(key, metric_card)

        param_card, pbox = _section_card("Tham số tối ưu", f"{result['n_fits']:,} lần fit")
        prows = [[str(k), str(v)] for k, v in result["best_params"].items()]
        pbox.addWidget(result_table(["Tham số", "Giá trị"], prows))
        self._add_result(key, param_card)

        for title, fig_key in (("Dự đoán vs Thực tế", "pred_vs_actual"),
                               ("Độ quan trọng (Gain)", "importance_gain")):
            if fig_key in result["figures"]:
                self._add_result(key, chart_card(title, result["figures"][fig_key]))
        self._show_results(key)
        self._refresh_gating()

    # ================================================================== #
    # Stage: SHAP
    # ================================================================== #
    def _config_shap(self, layout: QVBoxLayout) -> QPushButton:
        card, box = _section_card("Diễn giải SHAP",
                                  "TreeExplainer trên mô hình XGBoost đã huấn luyện")
        info = QLabel("Tính đóng góp của từng biến cho mỗi dự đoán: biểu đồ beeswarm, "
                      "% tầm quan trọng và scatter kèm đường LOWESS.")
        info.setObjectName("NLFieldHint")
        info.setWordWrap(True)
        box.addWidget(info)
        layout.addWidget(card)
        return self._primary_button("Tính SHAP", self._run_shap, "shap_run")

    def _run_shap(self) -> None:
        if not self._require_model():
            return

        def job(emit):
            return self.engine.compute_shap(progress=emit)

        self._run_job("Tính SHAP", "shap", job, self._render_shap)

    def _render_shap(self, result: dict) -> None:
        key = "shap"
        self._results[key] = result
        self._clear_results(key)
        importance = sorted(result["importance"], key=lambda kv: kv[1], reverse=True)
        chips_host = QWidget()
        chips = QHBoxLayout(chips_host)
        chips.setContentsMargins(0, 0, 0, 0)
        chips.setSpacing(8)
        chips.addWidget(QLabel("Quan trọng nhất:"))
        for name, pct in importance[:3]:
            chips.addWidget(metric_chip(f"{name} · {pct:.1f}%", "info"))
        chips.addStretch()
        self._add_result(key, chips_host)

        for title, fig_key in (("Beeswarm", "beeswarm"),
                               ("% Tầm quan trọng", "importance_bar"),
                               ("SHAP scatter + LOWESS", "scatter_grid")):
            if fig_key in result["figures"]:
                self._add_result(key, chart_card(title, result["figures"][fig_key]))

        table_card, tbox = _section_card("Bảng tầm quan trọng", "% đóng góp")
        rows = [[name, f"{pct:.2f}%"] for name, pct in importance]
        tbox.addWidget(result_table(["Biến", "Đóng góp"], rows))
        self._add_result(key, table_card)
        self._show_results(key)

    # ================================================================== #
    # Stage: SYMBOLIC
    # ================================================================== #
    def _config_symbolic(self, layout: QVBoxLayout) -> QPushButton:
        card, box = _section_card("Hồi quy biểu thức (PySR)",
                                  "Tham số tiến hóa công thức")
        grid = _form_grid()
        self.sr_niter = QSpinBox()
        self.sr_niter.setRange(20, 2000)
        self.sr_niter.setValue(80)
        self.sr_maxsize = QSpinBox()
        self.sr_maxsize.setRange(10, 60)
        self.sr_maxsize.setValue(30)
        self.sr_pops = QSpinBox()
        self.sr_pops.setRange(5, 40)
        self.sr_pops.setValue(15)
        self.sr_pars = QDoubleSpinBox()
        self.sr_pars.setRange(0.0, 1.0)
        self.sr_pars.setSingleStep(0.001)
        self.sr_pars.setDecimals(4)
        self.sr_pars.setValue(0.0)
        _grid_row(grid, 0, "Số thế hệ", self.sr_niter)
        _grid_row(grid, 1, "Độ phức tạp tối đa", self.sr_maxsize)
        _grid_row(grid, 2, "Số quần thể", self.sr_pops)
        _grid_row(grid, 3, "Parsimony", self.sr_pars)
        box.addLayout(grid)
        self.sr_train_only = QCheckBox("Học trên tập train (khuyến nghị)")
        self.sr_train_only.setChecked(True)
        box.addWidget(self.sr_train_only)
        note = QLabel("PySR dùng Julia để tiến hóa hàng triệu phương trình — có thể mất vài phút.")
        note.setObjectName("NLFieldHint")
        note.setWordWrap(True)
        box.addWidget(note)
        layout.addWidget(card)
        return self._primary_button("Tìm công thức", self._run_symbolic, "sr_run")

    def _run_symbolic(self) -> None:
        if not self._require_model():
            return
        niter = self.sr_niter.value()
        maxsize = self.sr_maxsize.value()
        pops = self.sr_pops.value()
        pars = self.sr_pars.value()
        train_only = self.sr_train_only.isChecked()
        seed = getattr(self, "seed_spin", None)
        seed = seed.value() if seed else 42

        def job(emit):
            return self.engine.run_symbolic_regression(
                niterations=niter, maxsize=maxsize, populations=pops,
                parsimony=pars, seed=seed, use_train=train_only, progress=emit)

        self._run_job("Hồi quy biểu thức (PySR)", "symbolic", job, self._render_symbolic)

    def _render_symbolic(self, result: dict) -> None:
        key = "symbolic"
        self._results[key] = result
        self._clear_results(key)

        hero = QFrame()
        hero.setObjectName("EquationCard")
        apply_shadow(hero, blur=22, y=6, color="#3A2A6A22")
        hbox = QVBoxLayout(hero)
        hbox.setContentsMargins(18, 16, 18, 16)
        hbox.setSpacing(10)
        head = QHBoxLayout()
        klbl = QLabel("CÔNG THỨC TỐI ƯU")
        klbl.setObjectName("NLCardKicker")
        head.addWidget(klbl)
        head.addStretch()
        copy = QPushButton("Sao chép")
        copy.setObjectName("CopyButton")
        copy.setCursor(Qt.PointingHandCursor)
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(result["equation"]))
        head.addWidget(copy)
        hbox.addLayout(head)
        eq = QLabel(f"{self.engine.target} ≈ {result['equation']}")
        eq.setObjectName("EquationText")
        eq.setWordWrap(True)
        eq.setTextInteractionFlags(Qt.TextSelectableByMouse)
        hbox.addWidget(eq)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        chips.addWidget(metric_chip(f"R² = {result['r2']:.4f}", _r2_tone(result["r2"])))
        if result["table"]:
            best = min(result["table"], key=lambda r: r["loss"])
            chips.addWidget(metric_chip(f"Độ phức tạp {best['complexity']}", "info"))
            chips.addWidget(metric_chip(f"Loss {best['loss']:.4g}", "muted"))
        chips.addStretch()
        hbox.addLayout(chips)
        self._add_result(key, hero)

        if "pareto" in result["figures"]:
            self._add_result(key, chart_card("Mặt trận Pareto", result["figures"]["pareto"]))

        if result["table"]:
            tcard, tbox = _section_card("Bảng tiến hóa", "Pareto front")
            rows = [[str(r["complexity"]), f"{r['loss']:.5g}",
                     (f"{r['score']:.4g}" if r["score"] == r["score"] else "—"), r["equation"]]
                    for r in result["table"]]
            table = result_table(["Độ phức tạp", "Loss", "Score", "Biểu thức"], rows)
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
            tbox.addWidget(table)
            self._add_result(key, tcard)
        self._show_results(key)
        self._refresh_gating()

    # ================================================================== #
    # Stage: SENSITIVITY
    # ================================================================== #
    def _config_sensitivity(self, layout: QVBoxLayout) -> QPushButton:
        card, box = _section_card("Phân tích độ nhạy Sobol",
                                  "Chỉ số S1 / ST và hướng tác động")
        grid = _form_grid()
        self.sob_predictor = QComboBox()
        self.sob_predictor.addItems(["Mô hình XGBoost", "Công thức PySR"])
        self.sob_pow = QComboBox()
        for k in (7, 9, 11, 13):
            self.sob_pow.addItem(f"2^{k}", k)
        self.sob_pow.setCurrentIndex(2)
        _grid_row(grid, 0, "Mô hình dự đoán", self.sob_predictor)
        _grid_row(grid, 1, "Cỡ mẫu N tối đa", self.sob_pow)
        box.addLayout(grid)
        note = QLabel("Tính chỉ số bậc một (S1) và tổng (ST), khảo sát hội tụ N=2^5..2^k, "
                      "kèm phân tích hướng tác động qua đạo hàm bậc 1 và 2.")
        note.setObjectName("NLFieldHint")
        note.setWordWrap(True)
        box.addWidget(note)
        layout.addWidget(card)
        return self._primary_button("Chạy Sobol", self._run_sensitivity, "sob_run")

    def _run_sensitivity(self) -> None:
        if not self._require_model():
            return
        predictor = "formula" if self.sob_predictor.currentIndex() == 1 else "model"
        if predictor == "formula" and self.engine.formula_callable is None:
            QMessageBox.warning(self, "Chưa có công thức",
                                "Hãy chạy Hồi quy biểu thức (PySR) trước, hoặc chọn Mô hình XGBoost.")
            return
        max_pow = self.sob_pow.currentData()

        def job(emit):
            return self.engine.sobol_sensitivity(predictor=predictor, max_pow=max_pow, progress=emit)

        self._run_job("Phân tích Sobol", "sensitivity", job, self._render_sensitivity)

    def _render_sensitivity(self, result: dict) -> None:
        key = "sensitivity"
        self._results[key] = result
        self._clear_results(key)
        table = result["table"]
        top = max(table, key=lambda r: r["ST"]) if table else {"feature": "—", "ST": 0}
        total_inter = sum(r["ST"] - r["S1"] for r in table)
        cards = [
            stat_card("Nhạy nhất (ST)", top["feature"], "accent", foot=f"ST = {top['ST']:.3f}"),
            stat_card("Tổng tương tác", _fmt(total_inter, 3), foot="ΣST − ΣS1"),
        ]
        self._add_result(key, self._stat_row(cards))

        for title, fig_key in (("Hội tụ S1 / ST", "convergence"),
                               ("Hướng tác động", "directional")):
            if fig_key in result["figures"]:
                self._add_result(key, chart_card(title, result["figures"][fig_key]))

        idx_card, ibox = _section_card("Chỉ số Sobol", "Tại N tối đa")
        rows = [[r["feature"], _fmt(r["S1"], 4), _fmt(r["ST"], 4), _fmt(r["ST"] - r["S1"], 4)]
                for r in table]
        ibox.addWidget(result_table(["Biến", "S1", "ST", "Tương tác"], rows))
        self._add_result(key, idx_card)

        dir_card, dbox = _section_card("Hướng tác động", "Theo đạo hàm")
        drows = [[r["feature"], r["dominant"]] for r in result["directional_table"]]
        dbox.addWidget(result_table(["Biến", "Xu hướng chủ đạo"], drows))
        self._add_result(key, dir_card)
        self._show_results(key)

    # ================================================================== #
    # Stage: OPTIMIZE
    # ================================================================== #
    def _config_optimize(self, layout: QVBoxLayout) -> QPushButton:
        card, box = _section_card("Tối ưu hóa mô hình",
                                  "Tìm cực đại / cực tiểu của mục tiêu")
        grid = _form_grid()
        self.opt_predictor = QComboBox()
        self.opt_predictor.addItems(["Công thức PySR", "Mô hình XGBoost"])
        _grid_row(grid, 0, "Hàm mục tiêu", self.opt_predictor)
        box.addLayout(grid)
        self.opt_integer = QCheckBox("Ràng buộc số nguyên (theo miền dữ liệu)")
        self.opt_integer.setChecked(True)
        box.addWidget(self.opt_integer)
        note = QLabel("Tìm giá trị lớn nhất và nhỏ nhất bằng Differential Evolution (SciPy) "
                      "trên miền giá trị thực tế của từng biến.")
        note.setObjectName("NLFieldHint")
        note.setWordWrap(True)
        box.addWidget(note)
        layout.addWidget(card)
        return self._primary_button("Tìm tối ưu", self._run_optimize, "opt_run")

    def _run_optimize(self) -> None:
        if not self._require_model():
            return
        predictor = "formula" if self.opt_predictor.currentIndex() == 0 else "model"
        if predictor == "formula" and self.engine.formula_callable is None:
            QMessageBox.warning(self, "Chưa có công thức",
                                "Hãy chạy Hồi quy biểu thức (PySR) trước, hoặc chọn Mô hình XGBoost.")
            return
        integer = self.opt_integer.isChecked()
        seed = getattr(self, "seed_spin", None)
        seed = seed.value() if seed else 42

        def job(emit):
            return self.engine.optimize(predictor=predictor, integer=integer, seed=seed, progress=emit)

        self._run_job("Tối ưu hóa", "optimize", job, self._render_optimize)

    def _render_optimize(self, result: dict) -> None:
        key = "optimize"
        self._results[key] = result
        self._clear_results(key)
        cards = [
            stat_card("Giá trị lớn nhất", _fmt(result["max_value"], 4), "accent"),
            stat_card("Giá trị nhỏ nhất", _fmt(result["min_value"], 4)),
        ]
        self._add_result(key, self._stat_row(cards))

        cfg_card, cbox = _section_card("Cấu hình tối ưu", result["method"])
        feats = list(result["max_vars"].keys())
        rows = [[f, str(result["max_vars"][f]), str(result["min_vars"][f])] for f in feats]
        cbox.addWidget(result_table(["Biến", "Tại MAX", "Tại MIN"], rows))
        self._add_result(key, cfg_card)

        if "profile" in result["figures"]:
            self._add_result(key, chart_card("Cấu hình biến tại điểm tối ưu", result["figures"]["profile"]))
        self._show_results(key)

    # ================================================================== #
    # Stage: REPORT
    # ================================================================== #
    def _config_report(self, layout: QVBoxLayout) -> QPushButton:
        card, box = _section_card("Báo cáo tổng hợp",
                                  "Tổng hợp kết quả của tất cả các bước đã chạy")
        info = QLabel("Mỗi bước hoàn thành sẽ tự xuất hiện trong báo cáo bên phải. "
                      "Bạn cũng có thể xuất toàn bộ biểu đồ ra PNG.")
        info.setObjectName("NLFieldHint")
        info.setWordWrap(True)
        box.addWidget(info)
        save_all = self._secondary_button("Lưu tất cả biểu đồ (PNG)",
                                          self._save_all_charts, icon_name="image")
        box.addWidget(save_all)
        layout.addWidget(card)
        return self._primary_button("Cập nhật báo cáo", self._render_report,
                                   "report_refresh", icon_name="report")

    def _render_report(self) -> None:
        key = "report"
        self._clear_results(key)
        done = [k for k in ("data", "xgboost", "shap", "symbolic", "sensitivity", "optimize")
                if k in self._results or (k == "data" and self.engine is not None)]
        banner = QFrame()
        banner.setObjectName("NLCard")
        bbox = QVBoxLayout(banner)
        bbox.setContentsMargins(18, 16, 18, 16)
        title = QLabel("Tiến trình phân tích phi tuyến")
        title.setObjectName("NLCardTitle")
        bbox.addWidget(title)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        labels = {"data": "Dữ liệu", "xgboost": "XGBoost", "shap": "SHAP",
                  "symbolic": "Công thức", "sensitivity": "Độ nhạy", "optimize": "Tối ưu"}
        for k, lbl in labels.items():
            tone = "good" if (k in self._results or (k == "data" and self.engine)) else "muted"
            chips.addWidget(metric_chip(lbl, tone))
        chips.addStretch()
        bbox.addLayout(chips)
        self._add_result(key, banner)

        headline = []
        if "xgboost" in self._results:
            headline.append(stat_card("R² Test", _fmt(self._results["xgboost"]["metrics"]["test_r2"], 3),
                                      _r2_tone(self._results["xgboost"]["metrics"]["test_r2"])))
        if "symbolic" in self._results:
            headline.append(stat_card("R² công thức", _fmt(self._results["symbolic"]["r2"], 3), "accent"))
        if "optimize" in self._results:
            headline.append(stat_card("Giá trị tối ưu", _fmt(self._results["optimize"]["max_value"], 3)))
        if headline:
            self._add_result(key, self._stat_row(headline))

        if "symbolic" in self._results:
            eq_card, ebox = _section_card("Công thức tối ưu", "PySR")
            eq = QLabel(f"{self.engine.target} ≈ {self._results['symbolic']['equation']}")
            eq.setObjectName("EquationText")
            eq.setWordWrap(True)
            ebox.addWidget(eq)
            self._add_result(key, eq_card)
        if len(done) <= 1:
            self._add_result(key, empty_state("report", "Chưa có gì để báo cáo",
                                              "Hãy chạy ít nhất bước XGBoost."))
        self._show_results(key)

    def _save_all_charts(self) -> None:
        figs = []
        for stage, res in self._results.items():
            for name, png in (res.get("figures", {}) or {}).items():
                figs.append((f"{stage}_{name}", png))
        if not figs:
            QMessageBox.information(self, "Chưa có biểu đồ", "Hãy chạy các bước phân tích trước.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu biểu đồ")
        if not folder:
            return
        for name, png in figs:
            Path(folder, f"{name}.png").write_bytes(png)
        QMessageBox.information(self, "Đã lưu", f"Đã lưu {len(figs)} biểu đồ vào:\n{folder}")

    # ================================================================== #
    # Stage: DEPS
    # ================================================================== #
    def _build_deps_stage(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        host = QWidget()
        self._deps_body = QVBoxLayout(host)
        self._deps_body.setContentsMargins(2, 2, 8, 2)
        self._deps_body.setSpacing(12)
        self._deps_body.addStretch(1)
        scroll.setWidget(host)
        QTimer.singleShot(0, self._render_deps)
        return scroll

    def _render_deps(self) -> None:
        from core.nonlinear_engine import dependency_report

        while self._deps_body.count() > 1:
            item = self._deps_body.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        report = dependency_report()
        ok = sum(1 for v in report.values() if v["available"])
        total = len(report)
        head = stat_card("Thư viện đã sẵn sàng", f"{ok}/{total}",
                         "good" if ok == total else "warn",
                         foot="Engine sẽ tự dùng site-packages global nếu cần.")
        self._deps_body.insertWidget(0, head)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for i, (name, info) in enumerate(report.items()):
            display, purpose, install = DEP_PURPOSE.get(name, (name, "", f"pip install {name}"))
            card = QFrame()
            card.setObjectName("DepCard")
            if not info["available"]:
                card.setProperty("missing", "true")
            cbox = QVBoxLayout(card)
            cbox.setContentsMargins(16, 14, 16, 14)
            cbox.setSpacing(6)
            top = QHBoxLayout()
            nm = QLabel(display)
            nm.setObjectName("DepName")
            top.addWidget(nm)
            top.addStretch()
            if info["available"]:
                top.addWidget(metric_chip(f"v{info['version']}", "muted"))
            else:
                top.addWidget(metric_chip("Chưa cài", "bad"))
            cbox.addLayout(top)
            pl = QLabel(purpose)
            pl.setObjectName("DepPurpose")
            pl.setWordWrap(True)
            cbox.addWidget(pl)
            if not info["available"]:
                copy = QPushButton(f"Sao chép: {install}")
                copy.setObjectName("GhostButton")
                copy.setCursor(Qt.PointingHandCursor)
                copy.clicked.connect(lambda _=False, cmd=install: QGuiApplication.clipboard().setText(cmd))
                cbox.addWidget(copy)
            grid.addWidget(card, i // 2, i % 2)
        self._deps_body.insertWidget(1, grid_host)

    # ---- guards ----------------------------------------------------------- #
    def _require_engine(self) -> bool:
        if self.engine is None:
            QMessageBox.information(self, "Chưa có dữ liệu", "Hãy hoàn tất bước Dữ liệu trước.")
            self._select_stage("data")
            return False
        return True

    def _require_model(self) -> bool:
        if not self._require_engine():
            return False
        if self.engine.model is None:
            QMessageBox.information(self, "Chưa có mô hình", "Hãy huấn luyện XGBoost trước.")
            self._select_stage("xgboost")
            return False
        return True

    # Libraries each stage genuinely needs (statsmodels is optional → not gated).
    STAGE_LIBS = {
        "xgboost": ["sklearn", "xgboost", "matplotlib"],
        "shap": ["shap", "matplotlib"],
        "symbolic": ["pysr", "sympy"],
        "sensitivity": ["SALib", "matplotlib"],
        "optimize": ["matplotlib"],
    }

    # ---- threaded job runner --------------------------------------------- #
    def _run_job(self, title: str, stage: str, job, on_done) -> None:
        if self._job_running:
            QMessageBox.information(self, "Đang chạy", "Một tác vụ đang chạy. Hãy đợi hoàn tất.")
            return
        from core.nonlinear_engine import dependency_report

        report = dependency_report()
        missing = [lib for lib in self.STAGE_LIBS.get(stage, [])
                   if not report.get(lib, {}).get("available", False)]
        if missing:
            QMessageBox.warning(self, "Thiếu thư viện",
                                f"Bước này cần thư viện chưa cài: {', '.join(missing)}.\n"
                                "Mở mục 'Thư viện' để xem lệnh cài đặt.")
            self._select_stage("deps")
            return
        self._job_running = True
        self._set_running_chip(True)
        dialog = NLProgressDialog(title, self.window())
        dialog.show()

        thread = QThread(self)
        worker = NLWorker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(dialog.set_status)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        # Connect to bound methods of self (a QObject living in the GUI thread) so
        # the slots run via QueuedConnection on the GUI thread — never the worker.
        worker.finished.connect(self._on_job_finished)
        worker.failed.connect(self._on_job_failed)
        self._job_dialog = dialog
        self._job_title = title
        self._job_stage = stage
        self._job_on_done = on_done
        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    @Slot(object)
    def _on_job_finished(self, result) -> None:
        if getattr(self, "_job_dialog", None) is not None:
            self._job_dialog.finish()
        self._job_running = False
        try:
            self._job_on_done(result)
            self._set_done_chip(self._job_stage)
        except Exception as exc:  # noqa: BLE001
            self._set_running_chip(False)
            QMessageBox.critical(self, f"{self._job_title} — lỗi hiển thị",
                                 f"{exc}\n\n{traceback.format_exc()[-1200:]}")
        finally:
            self._job_dialog = None
            self._job_on_done = None

    @Slot(str)
    def _on_job_failed(self, message: str) -> None:
        if getattr(self, "_job_dialog", None) is not None:
            self._job_dialog.finish()
        self._job_running = False
        self._set_running_chip(False)
        QMessageBox.critical(self, f"{self._job_title} thất bại", message)
        self._job_dialog = None
        self._job_on_done = None

    def _set_running_chip(self, running: bool) -> None:
        if running:
            self.header_chip.setText("Đang chạy…")
            self.header_chip.setProperty("tone", "info")
        else:
            self.header_chip.setText("Sẵn sàng")
            self.header_chip.setProperty("tone", "muted")
        self._repolish(self.header_chip)

    def _set_done_chip(self, stage: str) -> None:
        self.header_chip.setText("Đã hoàn tất")
        self.header_chip.setProperty("tone", "good")
        self._repolish(self.header_chip)

    # ---- navigation / gating --------------------------------------------- #
    def _select_stage(self, key: str) -> None:
        if key in self._nav_buttons:
            self._nav_buttons[key].setChecked(True)
        if key in self._stage_index:
            self.stack.setCurrentIndex(self._stage_index[key])
        for i, (k, label, _icon, hint) in enumerate(STAGES):
            if k == key:
                if k == "deps":
                    self.header_kicker.setText("PHỤ THUỘC")
                else:
                    self.header_kicker.setText(f"BƯỚC {i + 1} / 7")
                self.header_title.setText(label)
                self.header_sub.setText(hint)
                break
        if key in self._results or (key == "data" and self.engine is not None):
            self._set_done_chip(key)
        elif key == "deps":
            self.header_chip.setText("Thông tin")
            self.header_chip.setProperty("tone", "info")
            self._repolish(self.header_chip)
        else:
            self.header_chip.setText("Chưa chạy")
            self.header_chip.setProperty("tone", "muted")
            self._repolish(self.header_chip)

    def _refresh_gating(self) -> None:
        has_engine = self.engine is not None
        has_model = has_engine and self.engine.model is not None
        gates = {
            "xgboost": has_engine, "shap": has_model, "symbolic": has_model,
            "sensitivity": has_model, "optimize": has_model,
        }
        for key, enabled in gates.items():
            if key in self._nav_buttons:
                self._nav_buttons[key].setEnabled(enabled)

    def _repolish(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    # ---- public API for main_window -------------------------------------- #
    def shutdown(self) -> None:
        """Quit + wait the worker thread so the app can close without aborting.

        A running job is one long blocking call that can't be interrupted mid-fit,
        so this waits (bounded) for it to finish to avoid Qt's fatal
        'QThread: Destroyed while thread is still running'.
        """
        thread = getattr(self, "_active_thread", None)
        try:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(8000)
        except RuntimeError:
            pass  # already deleted

    def has_running_job(self) -> bool:
        return bool(self._job_running)

    def set_app_data(self, frame, name: str = "") -> None:
        self._app_frame = frame
        self._app_name = name
        if hasattr(self, "use_app_btn"):
            self.use_app_btn.setEnabled(frame is not None and not getattr(frame, "empty", True))

    def go_to_stage(self, key: str) -> None:
        mapping = {"workspace": "data", "open": "data", "load": "data", "train": "xgboost",
                   "shap": "shap", "symbolic": "symbolic", "sensitivity": "sensitivity",
                   "optimize": "optimize", "report": "report", "deps": "deps", "data": "data"}
        target = mapping.get(key, "data")
        if target not in ("data", "deps", "report") and not self._nav_buttons.get(target, QToolButton()).isEnabled():
            target = "data" if self.engine is None else "xgboost"
        self._select_stage(target)


# Defaults used when a grid field is left empty (mirrors engine.QUICK_PARAMS).
QUICK_DEFAULTS = {
    "max_depth": 3, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0,
}

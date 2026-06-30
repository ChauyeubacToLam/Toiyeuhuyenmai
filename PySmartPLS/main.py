import sys
import ctypes
import subprocess
import unicodedata
from pathlib import Path

from PySide6.QtCore import Property, QEasingCurve, QObject, QPoint, QProcess, QPropertyAnimation, QSequentialAnimationGroup, QSize, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from gui.main_window import MainWindow


SOURCE_APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_APP_DIR.parent
APP_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_APP_DIR)).resolve()
UPDATE_REMOTE = "origin"
UPDATE_BRANCH = "main"


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


RESOURCE_ROOTS = _unique_paths([
    APP_DIR,
    APP_DIR.parent,
    SOURCE_APP_DIR,
    SOURCE_APP_DIR.parent,
    Path.cwd(),
])


def resource_path(name: str) -> Path:
    """Resolve source/bundled assets on Windows, macOS, and frozen builds."""
    for root in RESOURCE_ROOTS:
        candidate = root / name
        if candidate.exists():
            return candidate
    wanted = {unicodedata.normalize("NFC", name), unicodedata.normalize("NFD", name)}
    for root in RESOURCE_ROOTS:
        try:
            for child in root.iterdir():
                child_names = {unicodedata.normalize("NFC", child.name), unicodedata.normalize("NFD", child.name)}
                if child_names & wanted:
                    return child
        except OSError:
            continue
    return RESOURCE_ROOTS[0] / name


def find_git_root() -> Path:
    starts = _unique_paths([SOURCE_APP_DIR, SOURCE_APP_DIR.parent, APP_DIR, APP_DIR.parent, Path.cwd()])
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                return candidate
    return PROJECT_ROOT


def splash_window_flags():
    flags = Qt.FramelessWindowHint
    if sys.platform != "darwin":
        flags |= Qt.WindowStaysOnTopHint
    return flags


def app_font_family() -> str:
    return "Helvetica Neue" if sys.platform == "darwin" else "Segoe UI"


GIT_ROOT = find_git_root()
LOGO_PATH = resource_path("logo.jpg")
INTRO_IMAGE_PATH = resource_path("Ảnh.jpg")
INTRO_MESSAGE = "Ứng dụng này được phát triển dành riêng cho Huyền Mai - tình yêu và thế giới của tôi!"


def set_windows_app_id() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("huyenmai.pysmartpls")
    except Exception:
        pass


class ScaledImageLabel(QWidget):
    def __init__(self, image_path: Path, parent=None, aspect_mode=Qt.KeepAspectRatio):
        super().__init__(parent)
        self._pixmap = QPixmap(str(image_path))
        self._scaled_pixmap = QPixmap()
        self._scaled_key: tuple[int, int, float] | None = None
        self._aspect_mode = aspect_mode
        self._opacity = 1.0
        self.setStyleSheet("background: transparent;")

    def opacity(self) -> float:
        return self._opacity

    def setOpacity(self, value: float) -> None:
        self._opacity = max(0.0, min(1.0, value))
        self.update()

    opacity = Property(float, opacity, setOpacity)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setOpacity(self._opacity)

        if self._pixmap.isNull():
            painter.setPen(QColor("#b84f7a"))
            painter.setFont(QFont(app_font_family(), 16, QFont.Weight.DemiBold))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, "Không tìm thấy ảnh")
            return

        dpr = self.devicePixelRatioF()
        physical_size = QSize(max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
        scaled_key = (physical_size.width(), physical_size.height(), dpr)
        if self._scaled_key != scaled_key:
            self._scaled_pixmap = self._pixmap.scaled(physical_size, self._aspect_mode, Qt.SmoothTransformation)
            self._scaled_pixmap.setDevicePixelRatio(dpr)
            self._scaled_key = scaled_key

        draw_width = self._scaled_pixmap.width() / dpr
        draw_height = self._scaled_pixmap.height() / dpr
        x = round((self.width() - draw_width) / 2)
        y = round((self.height() - draw_height) / 2)
        painter.drawPixmap(x, y, self._scaled_pixmap)


class IntroWindow(QWidget):
    finished = Signal()

    def __init__(self, logo_path: Path, image_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowFlags(splash_window_flags())
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setMinimumSize(900, 540)

        self.logo = ScaledImageLabel(logo_path, self)
        self.logo.setOpacity(0)

        self.photo = ScaledImageLabel(image_path, self, Qt.KeepAspectRatioByExpanding)
        self.photo.setOpacity(0)

        self.message = QLabel("", self)
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.message.setStyleSheet("color: #7a244e; background: transparent;")
        self.message_opacity = QGraphicsOpacityEffect(self.message)
        self.message_opacity.setOpacity(0)
        self.message.setGraphicsEffect(self.message_opacity)

        self._finished = False
        self._message_fade: QPropertyAnimation | None = None
        self._intro_group: QSequentialAnimationGroup | None = None
        self._finish_group: QSequentialAnimationGroup | None = None
        self._photo_start_pos = QPoint()
        self._photo_end_pos = QPoint()
        self._text_index = 0
        self._typing_timer = QTimer(self)
        self._typing_timer.setInterval(34)
        self._typing_timer.timeout.connect(self._type_next_character)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.fillRect(self.rect(), QColor("#fff7fb"))

        width = self.width()
        height = self.height()
        painter.setBrush(QColor(255, 255, 255, 130))
        painter.drawEllipse(-int(width * 0.08), -int(height * 0.14), int(width * 0.42), int(height * 0.36))
        painter.setBrush(QColor(255, 198, 219, 82))
        painter.drawEllipse(int(width * 0.70), int(height * 0.62), int(width * 0.34), int(height * 0.32))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._layout_intro()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._intro_group or self._intro_group.state() != QSequentialAnimationGroup.State.Running:
            self._layout_intro()

    def mousePressEvent(self, event) -> None:
        self._finish()

    def keyPressEvent(self, event) -> None:
        self._finish()

    def start(self) -> None:
        self.setWindowOpacity(1)
        self._layout_intro()
        self.logo.setOpacity(0)
        self.photo.setOpacity(0)
        self.message_opacity.setOpacity(0)
        self.message.setText("")

        self._intro_group = QSequentialAnimationGroup(self)
        self._intro_group.addAnimation(self._opacity_animation(self.logo, 0, 1, 900))
        self._intro_group.addPause(550)
        self._intro_group.addAnimation(self._opacity_animation(self.logo, 1, 0, 720))
        self._intro_group.addPause(160)
        self._intro_group.addAnimation(self._opacity_animation(self.photo, 0, 1, 950))
        self._intro_group.addPause(420)
        self._intro_group.addAnimation(self._move_animation(self.photo, self._photo_start_pos, self._photo_end_pos, 1050))
        self._intro_group.finished.connect(self._start_message)

        self._finish_group = QSequentialAnimationGroup(self)
        self._finish_group.addPause(1050)
        self._finish_group.addAnimation(self._window_opacity_animation(1, 0, 850))
        self._finish_group.finished.connect(self._finish)
        self._intro_group.start()

    def _layout_intro(self) -> None:
        width = max(self.width(), 900)
        height = max(self.height(), 540)
        margin = max(40, int(width * 0.05))
        gap = max(48, int(width * 0.045))
        top_margin = max(42, int(height * 0.08))
        available_height = height - top_margin * 2

        photo_size = height
        if width - (margin + gap + photo_size) < 420:
            photo_size = max(int(height * 0.84), width - (margin + gap + 420))

        photo_y = (height - photo_size) // 2
        photo_start_x = (width - photo_size) // 2
        photo_end_x = 0
        self.photo.setGeometry(photo_start_x, photo_y, photo_size, photo_size)
        self._photo_start_pos = QPoint(photo_start_x, photo_y)
        self._photo_end_pos = QPoint(photo_end_x, photo_y)

        text_x = photo_end_x + photo_size + gap
        text_width = max(360, width - text_x - margin)
        self.message.setGeometry(text_x, top_margin, text_width, available_height)
        self.message.setFont(QFont("Segoe UI", max(24, min(42, int(height * 0.047))), QFont.Weight.DemiBold))

        logo_size = min(int(width * 0.24), int(height * 0.34), 380)
        self.logo.setGeometry((width - logo_size) // 2, (height - logo_size) // 2, logo_size, logo_size)

    def _opacity_animation(self, target, start: float, end: float, duration: int) -> QPropertyAnimation:
        animation = QPropertyAnimation(target, b"opacity", self)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        return animation

    def _move_animation(self, widget: QWidget, start: QPoint, end: QPoint, duration: int) -> QPropertyAnimation:
        animation = QPropertyAnimation(widget, b"pos", self)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        return animation

    def _window_opacity_animation(self, start: float, end: float, duration: int) -> QPropertyAnimation:
        animation = QPropertyAnimation(self, b"windowOpacity", self)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.InOutCubic)
        return animation

    def _start_message(self) -> None:
        self._text_index = 0
        self.message.setText("")
        self._message_fade = self._opacity_animation(self.message_opacity, 0, 1, 600)
        self._message_fade.start()
        self._typing_timer.start()

    def _type_next_character(self) -> None:
        if self._text_index >= len(INTRO_MESSAGE):
            self._typing_timer.stop()
            self._finish_group.start()
            return

        self._text_index += 1
        self.message.setText(INTRO_MESSAGE[:self._text_index])

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._typing_timer.isActive():
            self._typing_timer.stop()
        if self._intro_group and self._intro_group.state() == QSequentialAnimationGroup.State.Running:
            self._intro_group.stop()
        if self._finish_group and self._finish_group.state() == QSequentialAnimationGroup.State.Running:
            self._finish_group.stop()
        self.hide()
        self.finished.emit()


class GitUpdateWorker(QObject):
    finished = Signal(dict)

    def __init__(self, action: str):
        super().__init__()
        self.action = action

    @Slot()
    def run(self) -> None:
        try:
            if self.action == "check":
                self.finished.emit(self._check())
            else:
                self.finished.emit(self._update())
        except Exception as exc:
            self.finished.emit({"status": "error", "message": str(exc)})

    def _run_git(self, args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
        startupinfo = None
        creationflags = 0
        if sys.platform.startswith("win"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW
        return subprocess.run(
            ["git", "-C", str(GIT_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )

    def _git_text(self, args: list[str], timeout: int = 30) -> str:
        result = self._run_git(args, timeout)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Git command failed.").strip())
        return result.stdout.strip()

    def _check(self) -> dict:
        if not (GIT_ROOT / ".git").exists():
            return {"status": "unavailable", "message": "Không tìm thấy thư mục .git để kiểm tra cập nhật."}
        remote_url = self._git_text(["remote", "get-url", UPDATE_REMOTE])
        self._git_text(["fetch", "--quiet", UPDATE_REMOTE, UPDATE_BRANCH], timeout=90)
        local = self._git_text(["rev-parse", "HEAD"])
        remote = self._git_text(["rev-parse", f"{UPDATE_REMOTE}/{UPDATE_BRANCH}"])
        return {
            "status": "current" if local == remote else "update_available",
            "local": local,
            "remote": remote,
            "remote_url": remote_url,
        }

    def _update(self) -> dict:
        dirty = self._git_text(["status", "--porcelain"])
        if dirty:
            return {
                "status": "update_failed",
                "message": "Có thay đổi local chưa commit, nên không thể tự cập nhật an toàn.",
            }
        result = self._run_git(["pull", "--ff-only", UPDATE_REMOTE, UPDATE_BRANCH], timeout=120)
        if result.returncode != 0:
            return {"status": "update_failed", "message": (result.stderr or result.stdout).strip()}
        return {"status": "updated", "message": result.stdout.strip()}


class UpdateCheckWindow(QWidget):
    finished = Signal()
    restart_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(splash_window_flags())
        self.setMinimumSize(900, 540)
        self._thread: QThread | None = None
        self._worker: GitUpdateWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(1)

        self.card = QFrame()
        self.card.setStyleSheet(
            "QFrame { background: white; border: 1px solid #f4d7e5; border-radius: 22px; }"
            "QLabel { color: #3a1226; background: transparent; }"
            "QPushButton { background: #db2777; color: white; border: 0; border-radius: 10px; "
            "padding: 10px 18px; font-weight: 700; }"
            "QPushButton#Secondary { background: #fcf1f6; color: #8a5570; border: 1px solid #f4d7e5; }"
        )
        card_box = QVBoxLayout(self.card)
        card_box.setContentsMargins(38, 34, 38, 34)
        card_box.setSpacing(14)

        self.title = QLabel("Đang kiểm tra bản cập nhật mới nhất")
        self.title.setFont(QFont(app_font_family(), 22, QFont.Weight.DemiBold))
        self.title.setAlignment(Qt.AlignCenter)
        card_box.addWidget(self.title)

        self.message = QLabel("Đang so sánh phiên bản local với repo Toiyeuhuyenmai trên GitHub...")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setFont(QFont(app_font_family(), 11))
        card_box.addWidget(self.message)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedHeight(16)
        card_box.addWidget(self.progress)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignCenter)
        self.detail.setStyleSheet("color: #8a5570;")
        card_box.addWidget(self.detail)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.update_btn = QPushButton("Cập nhật và khởi động lại")
        self.update_btn.clicked.connect(self.start_update)
        self.update_btn.hide()
        buttons.addWidget(self.update_btn)
        self.continue_btn = QPushButton("Tiếp tục")
        self.continue_btn.setObjectName("Secondary")
        self.continue_btn.clicked.connect(self._finish)
        self.continue_btn.hide()
        buttons.addWidget(self.continue_btn)
        buttons.addStretch(1)
        card_box.addLayout(buttons)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addWidget(self.card)
        wrap.addStretch(1)
        root.addLayout(wrap)
        root.addStretch(1)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#fff7fb"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 198, 219, 82))
        painter.drawEllipse(int(self.width() * 0.72), int(self.height() * 0.62), int(self.width() * 0.32), int(self.height() * 0.32))
        painter.setBrush(QColor(255, 255, 255, 130))
        painter.drawEllipse(-int(self.width() * 0.08), -int(self.height() * 0.12), int(self.width() * 0.38), int(self.height() * 0.34))

    def start(self) -> None:
        self._run_worker("check")

    def _run_worker(self, action: str) -> None:
        self.progress.setRange(0, 0)
        self.update_btn.hide()
        self.continue_btn.hide()
        thread = QThread(self)
        worker = GitUpdateWorker(action)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_result)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(dict)
    def _handle_result(self, result: dict) -> None:
        status = result.get("status", "error")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if status == "current":
            self.title.setText("Phiên bản đã mới nhất")
            self.message.setText("Bản local đang trùng với commit mới nhất trên repo Toiyeuhuyenmai.")
            self.detail.setText(f"Commit: {result.get('local', '')[:7]}")
            QTimer.singleShot(900, self._finish)
        elif status == "update_available":
            self.title.setText("Cần cập nhật phiên bản mới nhất")
            self.message.setText("Repo Toiyeuhuyenmai đang có commit mới hơn bản đang chạy.")
            self.detail.setText(
                f"Local: {result.get('local', '')[:7]}  |  Mới nhất: {result.get('remote', '')[:7]}"
            )
            self.update_btn.show()
        elif status == "updated":
            self.title.setText("Đã cập nhật xong")
            self.message.setText("Ứng dụng sẽ khởi động lại để chạy phiên bản mới nhất.")
            self.detail.setText(result.get("message", ""))
            QTimer.singleShot(900, self.restart_requested.emit)
        elif status == "unavailable":
            self.title.setText("Bỏ qua kiểm tra cập nhật")
            self.message.setText(result.get("message", "Không tìm thấy thông tin Git của bản cài đặt."))
            self.detail.setText("Ứng dụng sẽ tiếp tục chạy bình thường.")
            QTimer.singleShot(1200, self._finish)
        elif status == "update_failed":
            self.title.setText("Không thể tự cập nhật")
            self.message.setText(result.get("message", "Git pull thất bại."))
            self.detail.setText("Bạn có thể commit/stash thay đổi local rồi mở lại app để cập nhật.")
            self.continue_btn.show()
        else:
            self.title.setText("Không kiểm tra được cập nhật")
            self.message.setText(result.get("message", "Không thể kết nối hoặc không chạy được Git."))
            self.detail.setText("App sẽ cho phép tiếp tục để không chặn công việc.")
            self.continue_btn.show()

    def start_update(self) -> None:
        self.title.setText("Đang cập nhật phiên bản mới nhất")
        self.message.setText("Đang kéo code mới từ repo Toiyeuhuyenmai...")
        self.detail.setText("")
        self._run_worker("update")

    def _finish(self) -> None:
        self.hide()
        self.finished.emit()


def main():
    set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName("PySmartPLS")
    app.setApplicationDisplayName("PySmartPLS")
    app.setOrganizationName("HuyenMai")
    app_icon = QIcon(str(LOGO_PATH))
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    
    app.setStyle("Fusion")
    if sys.platform == "darwin":
        app.setFont(QFont(app_font_family(), 11))

    windows = {}

    def show_main_window() -> None:
        window = MainWindow()
        if not app_icon.isNull():
            window.setWindowIcon(app_icon)
        windows["main"] = window
        window.showMaximized()

    def restart_application() -> None:
        if sys.platform == "darwin" and getattr(sys, "frozen", False):
            executable = Path(sys.executable).resolve()
            app_bundle = next((parent for parent in executable.parents if parent.suffix == ".app"), None)
            if app_bundle is not None:
                args = ["-n", str(app_bundle)]
                if len(sys.argv) > 1:
                    args.extend(["--args", *sys.argv[1:]])
                QProcess.startDetached("open", args)
                app.quit()
                return
        restart_args = list(sys.argv[1:] if getattr(sys, "frozen", False) else sys.argv)
        QProcess.startDetached(sys.executable, restart_args)
        app.quit()

    def show_update_check() -> None:
        checker = UpdateCheckWindow()
        if not app_icon.isNull():
            checker.setWindowIcon(app_icon)
        windows["update"] = checker
        checker.finished.connect(show_main_window)
        checker.restart_requested.connect(restart_application)
        checker.showFullScreen()
        QTimer.singleShot(80, checker.start)

    intro = IntroWindow(LOGO_PATH, INTRO_IMAGE_PATH)
    intro.finished.connect(show_update_check)
    intro.showFullScreen()
    QTimer.singleShot(80, intro.start)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

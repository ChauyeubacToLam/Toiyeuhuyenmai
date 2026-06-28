import sys
import ctypes
from pathlib import Path

from PySide6.QtCore import Property, QEasingCurve, QPoint, QPropertyAnimation, QSequentialAnimationGroup, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsOpacityEffect, QLabel, QWidget

from gui.main_window import MainWindow


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
LOGO_PATH = PROJECT_ROOT / "logo.jpg"
INTRO_IMAGE_PATH = PROJECT_ROOT / "Ảnh.jpg"
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
            painter.setFont(QFont("Segoe UI", 16, QFont.Weight.DemiBold))
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
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
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


def main():
    set_windows_app_id()
    app = QApplication(sys.argv)
    app_icon = QIcon(str(LOGO_PATH))
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    
    app.setStyle("Fusion")

    windows = {}

    def show_main_window() -> None:
        window = MainWindow()
        if not app_icon.isNull():
            window.setWindowIcon(app_icon)
        windows["main"] = window
        window.showMaximized()

    intro = IntroWindow(LOGO_PATH, INTRO_IMAGE_PATH)
    intro.finished.connect(show_main_window)
    intro.showFullScreen()
    QTimer.singleShot(80, intro.start)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF


BLUE = QColor("#2684ff")
BLUE_DARK = QColor("#2368c4")
GREY = QColor("#8b8f94")
GREY_DARK = QColor("#666a6f")
GREEN = QColor("#45b649")
YELLOW = QColor("#ffd21f")
ORANGE = QColor("#ff9d22")


def icon(name: str, size: int = 24) -> QIcon:
    """Small, dependency-free icons matching the classic SmartPLS visual language."""
    ratio = 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(ratio, ratio)
    _paint_icon(painter, name, size)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return QIcon(pixmap)


def _pen(color=GREY_DARK, width: float = 1.2) -> QPen:
    return QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)


def _paint_icon(p: QPainter, name: str, s: int) -> None:
    k = s / 24.0
    p.scale(k, k)
    if name in {"folder", "archive", "workspace"}:
        _folder(p, archive=name == "archive")
    elif name in {"project-ok", "project-error"}:
        _project_status(p, error=name.endswith("error"))
    elif name in {"path-ok", "path-error", "path-model"}:
        _path_status(p, error=name.endswith("error"))
    elif name == "data-green":
        _data_stack(p)
    elif name == "info":
        _info(p)
    elif name in {"project", "new-project", "data"}:
        _document(p, plus=name == "new-project", table=name == "data")
    elif name in {"model", "new-model", "connection", "moderating", "quadratic"}:
        _model(p, plus=name == "new-model", variant=name)
    elif name in {"save", "save-as", "duplicate"}:
        _save(p, overlay=name)
    elif name == "save-toolbar":
        _tray_arrow(p, down=True)
    elif name in {"import", "restore"}:
        _tray_arrow(p, down=True)
    elif name == "export":
        _tray_arrow(p, down=False)
    elif name in {"plus", "minus"}:
        _square_symbol(p, name == "plus")
    elif name == "star":
        _star(p)
    elif name.startswith("filter"):
        _filter(p, name)
    elif name in {"calculate", "bootstrap", "blindfolding", "analysis"}:
        _calculate(p, name)
    elif name in {"copy", "paste"}:
        _copy(p, paste=name == "paste")
    elif name in {"delete", "close"}:
        _delete(p)
    elif name == "rename":
        _rename(p)
    elif name in {"undo", "redo"}:
        _undo(p, reverse=name == "redo")
    elif name == "pointer":
        _pointer(p)
    elif name == "indicators":
        _indicators(p)
    elif name in {"latent", "indicator"}:
        _node(p, latent=name == "latent")
    elif name == "note":
        _note(p)
    elif name.startswith("align") or name.startswith("match"):
        _align(p, name)
    elif name == "preferences":
        _gear(p)
    elif name == "print":
        _print(p)
    elif name == "image":
        _image(p)
    elif name == "clipboard":
        _clipboard(p)
    elif name == "help":
        _help(p)
    elif name == "language":
        _language(p)
    elif name == "theme":
        _theme(p)
    elif name == "zoom-in":
        _magnifier(p, True)
    elif name == "zoom-out":
        _magnifier(p, False)
    elif name == "fit":
        _fit(p)
    elif name == "check":
        _check(p)
    elif name in {"select-tool", "latent-tool", "connect-tool", "quadratic-tool", "moderating-tool", "comment-tool", "calculate-tool", "grid", "snap", "add-data-group", "generate-data-groups", "clear-data-groups"}:
        _tool_icon(p, name)
    elif name in {"matrix", "table"}:
        _matrix(p)
    elif name == "hide-zero":
        _hide_zero(p)
    elif name in {"decimals-up", "decimals-down"}:
        _decimals(p, up=name == "decimals-up")
    elif name == "export-excel":
        _export_excel(p)
    elif name == "export-web":
        _export_web(p)
    elif name == "export-r":
        _export_r(p)
    elif name == "exit":
        _exit(p)
    elif name == "x":
        _x_mark(p)
    else:
        _square_symbol(p, True)


def _x_mark(p: QPainter) -> None:
    p.setPen(QPen(QColor("#8a93a3"), 2.1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawLine(QPointF(7, 7), QPointF(17, 17))
    p.drawLine(QPointF(17, 7), QPointF(7, 17))


def _folder(p: QPainter, archive: bool = False) -> None:
    p.setPen(_pen(BLUE_DARK, 1))
    p.setBrush(BLUE)
    p.drawRoundedRect(QRectF(2, 7, 20, 14), 1.2, 1.2)
    p.drawPolygon(QPolygonF([QPointF(3, 7), QPointF(3, 4), QPointF(10, 4), QPointF(12, 7)]))
    if archive:
        p.setBrush(QColor("#ffffff"))
        p.drawRect(QRectF(8, 11, 8, 7))
        p.setBrush(GREY)
        p.drawRect(QRectF(7, 10, 10, 3))


def _document(p: QPainter, plus: bool = False, table: bool = False) -> None:
    p.setPen(_pen(QColor("#6f8eb5"), 1))
    p.setBrush(QColor("#ffffff"))
    p.drawRect(QRectF(3, 3, 16, 17))
    p.setBrush(QColor("#d9e8fb"))
    p.drawRect(QRectF(4, 4, 14, 4))
    if table:
        p.setPen(_pen(QColor("#8aa4c5"), .8))
        for y in (11, 14, 17):
            p.drawLine(QPointF(6, y), QPointF(16, y))
        p.drawLine(QPointF(10, 9), QPointF(10, 19))
    if plus:
        p.setPen(_pen(QColor("#ffffff"), 1.5))
        p.setBrush(GREEN)
        p.drawEllipse(QRectF(14, 14, 8, 8))
        p.drawLine(QPointF(18, 16), QPointF(18, 20))
        p.drawLine(QPointF(16, 18), QPointF(20, 18))


def _project_status(p: QPainter, error: bool) -> None:
    p.setPen(_pen(QColor("#6994c8"), 1))
    p.setBrush(QColor("#ffffff"))
    p.drawRect(QRectF(3, 5, 18, 15))
    p.setBrush(QColor("#dbe9fb"))
    p.drawRect(QRectF(4, 6, 16, 3))
    if error:
        p.setPen(_pen(QColor("#ffffff"), 1.5))
        p.setBrush(QColor("#ff7b28"))
        p.drawRect(QRectF(14, 12, 7, 8))
        p.drawLine(QPointF(17.5, 13.5), QPointF(17.5, 17))
        p.drawPoint(QPointF(17.5, 19))


def _path_status(p: QPainter, error: bool) -> None:
    p.setPen(_pen(QColor("#2c6fbd"), 1))
    p.drawLine(QPointF(12, 7), QPointF(7, 17))
    p.drawLine(QPointF(12, 7), QPointF(17, 17))
    p.setBrush(QColor("#388cff"))
    p.drawEllipse(QRectF(8, 2, 8, 8))
    p.setBrush(QColor("#ffc400"))
    p.drawRect(QRectF(4, 16, 6, 5)); p.drawRect(QRectF(14, 16, 6, 5))
    if error:
        p.setBrush(QColor("#f23b2d")); p.setPen(_pen(QColor("#b62419"), .8)); p.drawEllipse(QRectF(17, 2, 6, 6))


def _data_stack(p: QPainter) -> None:
    p.setPen(_pen(QColor("#006600"), 1))
    p.setBrush(QColor("#00d414"))
    p.drawRect(QRectF(4, 5, 16, 14))
    p.drawEllipse(QRectF(4, 2, 16, 7))
    p.drawEllipse(QRectF(4, 15, 16, 7))
    p.drawLine(QPointF(4, 9), QPointF(20, 9)); p.drawLine(QPointF(4, 14), QPointF(20, 14))


def _info(p: QPainter) -> None:
    p.setPen(_pen(QColor("#0a5b9f"), 1))
    p.setBrush(QColor("#187ac5")); p.drawEllipse(QRectF(3, 3, 18, 18))
    p.setPen(_pen(QColor("#ffffff"), 2)); p.drawText(QRectF(3, 2, 18, 19), Qt.AlignCenter, "i")


def _model(p: QPainter, plus: bool = False, variant: str = "model") -> None:
    p.setPen(_pen(BLUE_DARK, 1.25))
    p.drawLine(QPointF(7, 12), QPointF(17, 6))
    p.drawLine(QPointF(7, 12), QPointF(17, 18))
    if variant == "moderating":
        p.drawLine(QPointF(12, 4), QPointF(12, 12))
    elif variant == "quadratic":
        p.drawArc(QRectF(5, 4, 14, 14), 180 * 16, -180 * 16)
    p.setBrush(BLUE)
    for x, y in ((5, 12), (18, 5), (18, 19)):
        p.drawEllipse(QPointF(x, y), 3, 3)
    if plus:
        p.setPen(_pen(QColor("#ffffff"), 1.3))
        p.setBrush(YELLOW)
        p.drawEllipse(QRectF(1, 1, 8, 8))
        p.setPen(_pen(QColor("#6f6500"), 1.2))
        p.drawLine(QPointF(5, 3), QPointF(5, 7))
        p.drawLine(QPointF(3, 5), QPointF(7, 5))


def _save(p: QPainter, overlay: str) -> None:
    p.setPen(_pen(GREY_DARK, 1))
    p.setBrush(QColor("#aeb2b6"))
    p.drawRect(QRectF(4, 3, 16, 18))
    p.setBrush(QColor("#edf0f2"))
    p.drawRect(QRectF(7, 4, 9, 6))
    p.drawRect(QRectF(7, 14, 10, 7))
    if overlay == "save-as":
        p.setPen(_pen(ORANGE, 2))
        p.drawLine(QPointF(14, 18), QPointF(21, 11))
    elif overlay == "duplicate":
        p.setPen(_pen(BLUE_DARK, 1))
        p.setBrush(QColor("#ffffff"))
        p.drawRect(QRectF(13, 12, 8, 9))


def _tray_arrow(p: QPainter, down: bool) -> None:
    p.setPen(_pen(GREY_DARK, 1.2))
    p.setBrush(Qt.NoBrush)
    p.drawRect(QRectF(3, 14, 18, 7))
    p.setBrush(GREY)
    if down:
        p.drawLine(QPointF(12, 2), QPointF(12, 15))
        p.drawPolygon(QPolygonF([QPointF(7, 10), QPointF(17, 10), QPointF(12, 16)]))
    else:
        p.drawLine(QPointF(12, 18), QPointF(12, 4))
        p.drawPolygon(QPolygonF([QPointF(7, 8), QPointF(17, 8), QPointF(12, 2)]))


def _square_symbol(p: QPainter, plus: bool) -> None:
    p.setPen(_pen(QColor("#71777b"), 1))
    p.setBrush(QColor("#9da2a6"))
    p.drawRoundedRect(QRectF(3, 3, 18, 18), 2, 2)
    p.setPen(_pen(QColor("#ffffff"), 2))
    p.drawLine(QPointF(7, 12), QPointF(17, 12))
    if plus:
        p.drawLine(QPointF(12, 7), QPointF(12, 17))


def _star(p: QPainter) -> None:
    points = []
    for i in range(10):
        import math
        angle = -math.pi / 2 + i * math.pi / 5
        radius = 10 if i % 2 == 0 else 4.5
        points.append(QPointF(12 + math.cos(angle) * radius, 12 + math.sin(angle) * radius))
    p.setPen(_pen(QColor("#d99d00"), 1))
    p.setBrush(YELLOW)
    p.drawPolygon(QPolygonF(points))


def _filter(p: QPainter, name: str) -> None:
    p.setPen(_pen(GREY_DARK, 1))
    p.setBrush(QColor("#a6aaad"))
    p.drawPolygon(QPolygonF([QPointF(3, 4), QPointF(21, 4), QPointF(15, 11), QPointF(15, 19), QPointF(9, 19), QPointF(9, 11)]))
    color = GREY_DARK
    if name == "filter-yellow": color = QColor("#d8d800")
    if name == "filter-blue": color = QColor("#117cb0")
    p.setBrush(color)
    p.drawRect(QRectF(9, 17, 6, 4))


def _calculate(p: QPainter, name: str) -> None:
    p.setPen(_pen(QColor("#b8babc"), 1))
    p.setBrush(QColor("#d1d3d4"))
    if name == "bootstrap":
        p.drawEllipse(QRectF(2, 6, 9, 11)); p.drawEllipse(QRectF(13, 6, 9, 11))
    elif name == "blindfolding":
        p.drawEllipse(QRectF(3, 7, 18, 10)); p.drawLine(QPointF(5, 5), QPointF(19, 19))
    else:
        p.drawEllipse(QRectF(4, 4, 16, 16))
        for a, b in ((12, 1), (12, 23), (1, 12), (23, 12), (4, 4), (20, 20), (20, 4), (4, 20)):
            p.drawLine(QPointF(12, 12), QPointF(a, b))
        p.setBrush(QColor("#f2f2f2")); p.drawEllipse(QRectF(8, 8, 8, 8))


def _copy(p: QPainter, paste: bool = False) -> None:
    p.setPen(_pen(GREY, 1))
    p.setBrush(QColor("#f5f5f5"))
    p.drawRect(QRectF(7, 6, 13, 15))
    p.drawRect(QRectF(3, 2, 13, 15))
    if paste:
        p.setBrush(QColor("#b9bdc0")); p.drawRoundedRect(QRectF(6, 1, 8, 4), 1, 1)


def _delete(p: QPainter) -> None:
    p.setPen(_pen(QColor("#ffffff"), 1.8))
    p.setBrush(QColor("#969a9e"))
    p.drawEllipse(QRectF(3, 3, 18, 18))
    p.drawLine(QPointF(8, 8), QPointF(16, 16)); p.drawLine(QPointF(16, 8), QPointF(8, 16))


def _rename(p: QPainter) -> None:
    p.setPen(_pen(GREY, 1))
    p.setBrush(QColor("#ffffff")); p.drawRect(QRectF(2, 7, 20, 10))
    p.setPen(_pen(GREY_DARK, 1.5)); p.drawLine(QPointF(8, 9), QPointF(8, 15))


def _undo(p: QPainter, reverse: bool) -> None:
    p.save()
    if reverse:
        p.translate(24, 0); p.scale(-1, 1)
    p.setPen(_pen(GREY, 2)); p.setBrush(Qt.NoBrush)
    p.drawArc(QRectF(5, 5, 16, 14), 30 * 16, 230 * 16)
    p.setBrush(GREY); p.drawPolygon(QPolygonF([QPointF(3, 9), QPointF(10, 5), QPointF(9, 13)]))
    p.restore()


def _pointer(p: QPainter) -> None:
    p.setPen(_pen(GREY, 1)); p.setBrush(QColor("#d7d9da"))
    p.drawPolygon(QPolygonF([QPointF(5, 2), QPointF(18, 14), QPointF(12, 15), QPointF(9, 21)]))


def _node(p: QPainter, latent: bool) -> None:
    p.setPen(_pen(GREY, 1)); p.setBrush(QColor("#d4d6d7"))
    if latent: p.drawEllipse(QRectF(3, 5, 18, 14))
    else: p.drawRect(QRectF(3, 7, 18, 11))
    p.setBrush(QColor("#ffffff")); p.drawEllipse(QRectF(17, 2, 5, 5))


def _indicators(p: QPainter) -> None:
    p.setPen(_pen(QColor("#e1a900"), 1))
    p.setBrush(QColor("#ffc400"))
    p.drawEllipse(QRectF(2, 2, 20, 20))
    p.setPen(_pen(QColor("#ffffff"), 1.7))
    p.drawLine(QPointF(8, 8), QPointF(16, 16))
    p.drawLine(QPointF(7, 14), QPointF(14, 7))
    p.drawEllipse(QRectF(7, 7, 4, 4))
    p.drawEllipse(QRectF(13, 13, 4, 4))


def _note(p: QPainter) -> None:
    p.setPen(_pen(GREY, 1)); p.setBrush(QColor("#eceeef")); p.drawRect(QRectF(3, 3, 18, 18))
    for y in (8, 12, 16): p.drawLine(QPointF(6, y), QPointF(18, y))


def _align(p: QPainter, name: str) -> None:
    p.setPen(_pen(QColor("#5b6776"), 1.2)); p.setBrush(QColor("#9aa6b5"))
    if "bottom" in name:
        p.drawRect(QRectF(4, 5, 6, 13)); p.drawRect(QRectF(14, 9, 6, 9)); p.drawLine(QPointF(2, 20), QPointF(22, 20))
    elif "top" in name:
        p.drawRect(QRectF(4, 6, 6, 13)); p.drawRect(QRectF(14, 6, 6, 9)); p.drawLine(QPointF(2, 4), QPointF(22, 4))
    elif "left" in name:
        p.drawRect(QRectF(5, 4, 13, 6)); p.drawRect(QRectF(5, 14, 9, 6)); p.drawLine(QPointF(3, 2), QPointF(3, 22))
    elif "right" in name:
        p.drawRect(QRectF(6, 4, 13, 6)); p.drawRect(QRectF(10, 14, 9, 6)); p.drawLine(QPointF(21, 2), QPointF(21, 22))
    else:
        p.drawRect(QRectF(4, 5, 16, 5)); p.drawRect(QRectF(7, 14, 10, 5))


def _gear(p: QPainter) -> None:
    p.setPen(_pen(QColor("#6f7882"), 1)); p.setBrush(QColor("#9ca3aa")); p.drawEllipse(QRectF(3, 3, 18, 18))
    p.setBrush(QColor("#ffffff")); p.drawEllipse(QRectF(9, 9, 6, 6))


def _print(p: QPainter) -> None:
    p.setPen(_pen(GREY, 1)); p.setBrush(QColor("#d7d9da")); p.drawRect(QRectF(3, 8, 18, 10))
    p.setBrush(QColor("#ffffff")); p.drawRect(QRectF(6, 2, 12, 8)); p.drawRect(QRectF(6, 14, 12, 8))


def _image(p: QPainter) -> None:
    p.setPen(_pen(QColor("#6685a5"), 1)); p.setBrush(QColor("#ffffff")); p.drawRect(QRectF(2, 3, 20, 18))
    p.setBrush(QColor("#f0c84d")); p.drawEllipse(QRectF(15, 6, 4, 4))
    p.setBrush(QColor("#67a66f")); p.drawPolygon(QPolygonF([QPointF(4, 18), QPointF(10, 10), QPointF(14, 15), QPointF(17, 12), QPointF(21, 18)]))


def _clipboard(p: QPainter) -> None:
    p.setPen(_pen(GREY, 1)); p.setBrush(QColor("#e4e6e7")); p.drawRoundedRect(QRectF(4, 3, 16, 19), 2, 2)
    p.setBrush(QColor("#a8acb0")); p.drawRoundedRect(QRectF(8, 1, 8, 5), 1, 1)


def _help(p: QPainter) -> None:
    p.setPen(_pen(BLUE_DARK, 1)); p.setBrush(BLUE); p.drawEllipse(QRectF(3, 3, 18, 18))
    p.setPen(_pen(QColor("#ffffff"), 2)); p.drawText(QRectF(3, 2, 18, 19), Qt.AlignCenter, "?")


def _language(p: QPainter) -> None:
    p.setPen(_pen(BLUE_DARK, 1)); p.setBrush(QColor("#dcecff")); p.drawEllipse(QRectF(2, 2, 20, 20))
    p.drawEllipse(QRectF(7, 2, 10, 20)); p.drawLine(QPointF(2, 12), QPointF(22, 12))


def _theme(p: QPainter) -> None:
    p.setPen(_pen(GREY_DARK, 1)); p.setBrush(QColor("#d6d8da")); p.drawEllipse(QRectF(2, 2, 20, 20))
    for color, rect in ((BLUE, QRectF(4, 5, 5, 5)), (GREEN, QRectF(10, 3, 5, 5)), (ORANGE, QRectF(15, 7, 5, 5))):
        p.setBrush(color); p.drawEllipse(rect)


def _magnifier(p: QPainter, plus: bool) -> None:
    p.setPen(_pen(GREY_DARK, 2)); p.setBrush(Qt.NoBrush); p.drawEllipse(QRectF(3, 3, 13, 13)); p.drawLine(QPointF(14, 14), QPointF(21, 21))
    p.drawLine(QPointF(6, 9.5), QPointF(13, 9.5))
    if plus: p.drawLine(QPointF(9.5, 6), QPointF(9.5, 13))


def _fit(p: QPainter) -> None:
    p.setPen(_pen(GREY_DARK, 1.5));
    for a, b, c in ((3, 9, 3), (21, 15, 21)):
        p.drawLine(QPointF(a, b), QPointF(a, c)); p.drawLine(QPointF(a, c), QPointF(9 if a == 3 else 15, c))
    p.drawRect(QRectF(7, 7, 10, 10))


def _check(p: QPainter) -> None:
    p.setPen(_pen(GREEN, 2.4)); p.setBrush(Qt.NoBrush); p.drawEllipse(QRectF(2, 2, 20, 20)); p.drawLine(QPointF(6, 12), QPointF(10, 16)); p.drawLine(QPointF(10, 16), QPointF(18, 7))


def _tool_icon(p: QPainter, name: str) -> None:
    if name == "select-tool":
        _pointer(p); return
    if name == "latent-tool":
        p.setPen(_pen(QColor("#2d6590"), 1)); p.setBrush(QColor("#24a9d8")); p.drawEllipse(QRectF(3, 6, 17, 13))
        p.setBrush(GREEN); p.drawRect(QRectF(16, 2, 7, 7)); p.setPen(_pen(QColor("#ffffff"), 1.2)); p.drawLine(QPointF(19.5, 3.5), QPointF(19.5, 7.5)); p.drawLine(QPointF(17.5, 5.5), QPointF(21.5, 5.5)); return
    if name == "connect-tool":
        p.setPen(_pen(QColor("#1e1e1e"), 2)); p.drawLine(QPointF(3, 12), QPointF(20, 12)); p.setBrush(QColor("#1e1e1e")); p.drawPolygon(QPolygonF([QPointF(20, 8), QPointF(24, 12), QPointF(20, 16)]))
        p.setBrush(GREEN); p.setPen(_pen(QColor("#208a24"), .8)); p.drawRect(QRectF(16, 1, 7, 7)); return
    if name in {"quadratic-tool", "moderating-tool"}:
        p.setPen(_pen(QColor("#333333"), 1)); p.setBrush(QColor("#333333")); p.drawEllipse(QRectF(8, 2, 8, 9)); p.drawRect(QRectF(6, 11, 12, 9))
        p.setBrush(QColor("#f6a623")); p.drawRect(QRectF(17, 14, 5, 5)); return
    if name == "comment-tool":
        p.setPen(Qt.NoPen); p.setBrush(QColor("#ffd500")); p.drawRoundedRect(QRectF(2, 3, 20, 15), 5, 5); p.drawPolygon(QPolygonF([QPointF(15, 17), QPointF(19, 22), QPointF(18, 16)])); return
    if name == "calculate-tool":
        p.setPen(Qt.NoPen); p.setBrush(QColor("#0878d8"));
        for i in range(12):
            import math
            angle = i * math.pi / 6
            p.drawEllipse(QPointF(12 + math.cos(angle) * 8, 12 + math.sin(angle) * 8), 2.4, 2.4)
        p.setBrush(QColor("#ffffff")); p.drawEllipse(QRectF(8, 8, 8, 8)); return
    if name == "grid":
        p.setPen(_pen(QColor("#999999"), .8));
        for value in (4, 10, 16, 22): p.drawLine(QPointF(value, 3), QPointF(value, 21)); p.drawLine(QPointF(3, value), QPointF(21, value))
        return
    if name == "snap":
        p.setPen(_pen(QColor("#505050"), 2)); p.drawArc(QRectF(4, 4, 16, 16), -90 * 16, 250 * 16); p.drawLine(QPointF(4, 5), QPointF(9, 5)); p.drawLine(QPointF(4, 5), QPointF(4, 10)); return
    if name in {"add-data-group", "generate-data-groups", "clear-data-groups"}:
        p.setPen(_pen(QColor("#999999"), 1)); p.setBrush(QColor("#c4c4c4")); p.drawEllipse(QRectF(7, 2, 8, 8)); p.drawRect(QRectF(4, 10, 14, 10))
        p.setBrush(QColor("#a0a0a0")); p.drawEllipse(QRectF(1, 7, 7, 7)); p.drawEllipse(QRectF(16, 7, 7, 7))
        p.setBrush(QColor("#aaaaaa"));
        if name == "clear-data-groups": p.drawLine(QPointF(16, 3), QPointF(23, 10)); p.drawLine(QPointF(23, 3), QPointF(16, 10))
        else: p.drawRect(QRectF(17, 1, 6, 6))


def _exit(p: QPainter) -> None:
    p.setPen(_pen(GREY_DARK, 1.4)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(3, 3, 11, 18)); p.drawLine(QPointF(9, 12), QPointF(22, 12)); p.drawLine(QPointF(18, 8), QPointF(22, 12)); p.drawLine(QPointF(18, 16), QPointF(22, 12))


def _grid_table(p: QPainter, header: QColor) -> None:
    p.setPen(_pen(QColor("#8aa0bd"), 1))
    p.setBrush(QColor("#ffffff"))
    p.drawRect(QRectF(3, 4, 18, 16))
    p.setBrush(header)
    p.drawRect(QRectF(3, 4, 18, 4))
    p.setPen(_pen(QColor("#b6c4d6"), 0.8))
    for x in (9, 15):
        p.drawLine(QPointF(x, 4), QPointF(x, 20))
    for y in (12, 16):
        p.drawLine(QPointF(3, y), QPointF(21, y))


def _matrix(p: QPainter) -> None:
    _grid_table(p, QColor("#cfe0f5"))


def _hide_zero(p: QPainter) -> None:
    p.setPen(_pen(QColor("#2f6fc0"), 1.6))
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(QRectF(6, 4, 12, 16))
    p.setPen(_pen(QColor("#d22f2f"), 1.8))
    p.drawLine(QPointF(5, 21), QPointF(19, 3))


def _decimals(p: QPainter, up: bool) -> None:
    from PySide6.QtGui import QFont
    p.setPen(_pen(QColor("#23476e"), 1))
    font = QFont("Segoe UI", 8)
    font.setBold(True)
    p.setFont(font)
    p.drawText(QRectF(0, 6, 17, 14), Qt.AlignVCenter | Qt.AlignLeft, "0.0")
    p.setPen(_pen(GREEN if up else QColor("#d22f2f"), 1.8))
    if up:
        p.drawLine(QPointF(16, 16), QPointF(20, 9))
        p.drawLine(QPointF(20, 9), QPointF(23, 16))
    else:
        p.drawLine(QPointF(16, 9), QPointF(20, 16))
        p.drawLine(QPointF(20, 16), QPointF(23, 9))


def _rounded_glyph(p: QPainter, fill: QColor, letter: str, letter_color: QColor = QColor("#ffffff")) -> None:
    from PySide6.QtGui import QFont
    p.setPen(Qt.NoPen)
    p.setBrush(fill)
    p.drawEllipse(QRectF(2, 2, 20, 20))
    p.setPen(_pen(letter_color, 1))
    font = QFont("Segoe UI", 9)
    font.setBold(True)
    p.setFont(font)
    p.drawText(QRectF(2, 2, 20, 20), Qt.AlignCenter, letter)


def _export_excel(p: QPainter) -> None:
    _rounded_glyph(p, QColor("#1d7244"), "X")


def _export_web(p: QPainter) -> None:
    from PySide6.QtGui import QFont
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#e8732a"))
    p.drawEllipse(QRectF(2, 2, 20, 20))
    p.setPen(_pen(QColor("#ffffff"), 1.6))
    p.drawLine(QPointF(10, 8), QPointF(7, 12)); p.drawLine(QPointF(7, 12), QPointF(10, 16))
    p.drawLine(QPointF(14, 8), QPointF(17, 12)); p.drawLine(QPointF(17, 12), QPointF(14, 16))


def _export_r(p: QPainter) -> None:
    _rounded_glyph(p, QColor("#2569b6"), "R")

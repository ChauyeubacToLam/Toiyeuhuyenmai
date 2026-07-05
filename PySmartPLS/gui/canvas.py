from __future__ import annotations

import math
import uuid
from typing import Any

from PySide6.QtCore import QEvent, QSignalBlocker, QPointF, QRectF, Qt, QLineF, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QFontMetrics, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QTextOption, QTransform
from PySide6.QtWidgets import (
    QFileDialog,
    QGraphicsDropShadowEffect,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QInputDialog,
    QMenu,
    QMessageBox,
    QSwipeGesture,
)
from gui.icons import icon
from gui import theme as ui_theme


# Active canvas colour palette (node/edge colours). Updated by the view when the
# application theme changes; kept module-level so freshly-created nodes can style
# themselves before they are attached to a scene.
NODE_PALETTE: dict = dict(ui_theme.palette(ui_theme.DEFAULT_THEME))
INDICATOR_WIDTH = 112.0
INDICATOR_HEIGHT = 36.0
INDICATOR_GAP = 14.0
CONSTRUCT_INDICATOR_GAP = 40.0


def _round_rect_path(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    return path


def _node_shadow(blur: float = 16.0, dy: float = 3.0, alpha: int = 55) -> QGraphicsDropShadowEffect:
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(blur)
    effect.setOffset(0, dy)
    effect.setColor(QColor(20, 30, 60, alpha))
    return effect


class ConnectionLine(QGraphicsLineItem):
    def __init__(self, source_node: "BaseNode", target_node: "BaseNode | None" = None):
        super().__init__()
        self.source_node = source_node
        self.target_node = target_node
        self.result_badge: ResultBadge | None = None
        self.arrow_size = 9
        self.setPen(QPen(QColor(NODE_PALETTE["edge"]), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self.setZValue(-10)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.update_position()

    def update_position(self) -> None:
        if not (self.source_node and self.target_node):
            return
        try:
            start = self._node_rect(self.source_node).center()
            end = self._node_rect(self.target_node).center()
        except RuntimeError:
            return
        self.setLine(QLineF(start, end))
        self.update_result_badge_position()

    def update_result_badge_position(self) -> None:
        if not self.result_badge:
            return
        try:
            if self.result_badge.scene() is None:
                self.result_badge = None
                return
        except RuntimeError:
            self.result_badge = None
            return
        line = self.line()
        p1, p2 = line.p1(), line.p2()
        dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length  # unit normal to the edge
        kind = getattr(self.result_badge, "kind", "loading")
        if kind == "beta":
            # Structural β floats just above the connector midpoint.
            t = 0.5
            mag = self.result_badge.rect.height() / 2 + 6.0
            if ny > 0:
                nx, ny = -nx, -ny  # push to the upper side of the line
        else:
            # Loadings sit toward the indicator end (so siblings spread out) and
            # are nudged perpendicular to clear the indicator chip: a big sideways
            # push for steep edges (stacked indicators), a small one for flat edges.
            t = 0.72
            verticality = abs(dy) / length
            mag = 12.0 + 58.0 * verticality
            if abs(nx) >= abs(ny):
                if nx < 0:
                    nx, ny = -nx, -ny  # steep edge -> push right of the chip
            elif ny > 0:
                nx, ny = -nx, -ny      # flat edge -> push above the chip
        bx, by = p1.x() + dx * t + nx * mag, p1.y() + dy * t + ny * mag
        self.result_badge.set_center(bx, by)

    def _arrow_at_source(self) -> bool:
        """Formative constructs reverse the indicator arrow so it points INTO the
        construct. Indicator connections are stored as (latent=source,
        indicator=target), so a formative construct needs the head at the source
        end; everything else keeps the head at the target end."""
        if isinstance(self.source_node, LatentNode) and isinstance(self.target_node, IndicatorNode):
            return self.source_node.measurement_mode == "formative"
        if isinstance(self.source_node, IndicatorNode) and isinstance(self.target_node, LatentNode):
            return self.target_node.measurement_mode != "formative"
        return False

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if not self.target_node or self.line().length() == 0:
            return
        line = self.line()
        if self._arrow_at_source():
            head_node, seg = self.source_node, QLineF(line.p2(), line.p1())
        else:
            head_node, seg = self.target_node, QLineF(line.p1(), line.p2())
        angle = math.atan2(-(seg.dy()), seg.dx())
        end = self._edge_point(seg, head_node)
        p1 = end + QPointF(
            math.sin(angle - math.pi / 3) * self.arrow_size,
            math.cos(angle - math.pi / 3) * self.arrow_size,
        )
        p2 = end + QPointF(
            math.sin(angle - math.pi + math.pi / 3) * self.arrow_size,
            math.cos(angle - math.pi + math.pi / 3) * self.arrow_size,
        )
        painter.setPen(self.pen())
        painter.setBrush(QBrush(self.pen().color()))
        painter.drawPolygon(QPolygonF([end, p1, p2]))

    def refresh_theme(self) -> None:
        pen = self.pen()
        pen.setColor(QColor(NODE_PALETTE["edge"]))
        self.setPen(pen)
        self.update()

    def _edge_point(self, line: QLineF, head_node: "BaseNode | None" = None) -> QPointF:
        target_rect = self._node_rect(head_node or self.target_node).adjusted(-2, -2, 2, 2)
        center = target_rect.center()
        dx = line.p1().x() - center.x()
        dy = line.p1().y() - center.y()
        if dx == 0 and dy == 0:
            return line.p2()
        scale_x = abs((target_rect.width() / 2) / dx) if dx else float("inf")
        scale_y = abs((target_rect.height() / 2) / dy) if dy else float("inf")
        scale = min(scale_x, scale_y)
        return QPointF(center.x() + dx * scale, center.y() + dy * scale)

    def _node_rect(self, node: "BaseNode") -> QRectF:
        connection_rect = getattr(node, "connection_rect", None)
        if callable(connection_rect):
            return connection_rect()
        return node.sceneBoundingRect()

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source_node.node_id,
            "target": self.target_node.node_id if self.target_node else "",
        }


class ResultBadge(QGraphicsItem):
    """A floating result pill on the canvas.

    ``kind`` selects the visual treatment and colours (read live from
    ``NODE_PALETTE`` so badges follow the active theme):
    - ``"loading"`` — outer loading; ``good`` tints it success/danger by threshold
    - ``"beta"``    — structural path coefficient (accent)
    - ``"rsquare"`` — endogenous R², drawn as a dark pill that reads on the node
    """

    def __init__(self, text: str, x: float, y: float, kind: str = "loading",
                 good: bool | None = None):
        super().__init__()
        self.text = text
        self.kind = kind
        self.good = good
        self.font = QFont("Segoe UI", 8, QFont.Bold)
        metrics = QFontMetrics(self.font)
        self.rect = QRectF(0, 0, metrics.horizontalAdvance(text) + 16, metrics.height() + 8)
        self.set_center(x, y)
        self.setZValue(40)

    def _colors(self) -> tuple[QColor, QColor, QColor]:
        p = NODE_PALETTE
        if self.kind == "rsquare":
            return QColor(18, 26, 44, 220), QColor("#FFFFFF"), QColor(p.get("accent", "#1D6FE0"))
        if self.kind == "beta":
            return (QColor(p.get("surface", "#FFFFFF")), QColor(p.get("text", "#10243F")),
                    QColor(p.get("accent", "#1D6FE0")))
        # loading — colour by threshold so the model echoes the report
        if self.good is True:
            edge = QColor(p.get("success", "#0E9F6E"))
        elif self.good is False:
            edge = QColor(p.get("danger", "#E11D48"))
        else:
            edge = QColor(p.get("accent", "#1D6FE0"))
        return QColor(p.get("surface", "#FFFFFF")), edge, edge

    def set_center(self, x: float, y: float) -> None:
        self.setPos(x - self.rect.width() / 2, y - self.rect.height() / 2)

    def boundingRect(self) -> QRectF:
        return self.rect

    def refresh_theme(self) -> None:
        self.update()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        fill, text_color, border = self._colors()
        painter.setPen(QPen(border, 1.2))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(self.rect, 7, 7)
        painter.setPen(text_color)
        painter.setFont(self.font)
        painter.drawText(self.rect, Qt.AlignCenter, self.text)


class BaseNode(QGraphicsItem):
    node_type = "base"

    def __init__(self, name: str, x: float, y: float, node_id: str | None = None):
        super().__init__()
        self.node_id = node_id or str(uuid.uuid4())
        self.connections: list[ConnectionLine] = []
        self._suppress_child_move = False
        self._last_pos = QPointF(x, y)
        self.setPos(x, y)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.text = QGraphicsTextItem(name, self)
        self.text.setZValue(20)

    def connection_rect(self) -> QRectF:
        return self.sceneBoundingRect()

    def linked_indicators(self) -> list["IndicatorNode"]:
        """Indicator nodes attached to this node through any connection."""
        result: list[IndicatorNode] = []
        for line in self.connections:
            source = getattr(line, "source_node", None)
            target = getattr(line, "target_node", None)
            if not (source and target):
                continue
            other = target if source is self else source
            if isinstance(other, IndicatorNode) and other not in result:
                result.append(other)
        return result

    def linked_latents(self) -> list["LatentNode"]:
        """Latent constructs this node is attached to through any connection."""
        result: list[LatentNode] = []
        for line in self.connections:
            source = getattr(line, "source_node", None)
            target = getattr(line, "target_node", None)
            if not (source and target):
                continue
            other = target if source is self else source
            if isinstance(other, LatentNode) and other not in result:
                result.append(other)
        return result

    @property
    def name(self) -> str:
        return self.text.toPlainText().strip()

    def set_name(self, name: str) -> None:
        self.text.setPlainText(name.strip() or self.name)
        self.update_text_pos()

    def add_connection(self, line: ConnectionLine) -> None:
        if line not in self.connections:
            self.connections.append(line)

    def remove_connection(self, line: ConnectionLine) -> None:
        if line in self.connections:
            self.connections.remove(line)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() and getattr(self.scene(), "snap_enabled", False):
            grid = getattr(self.scene(), "grid_size", 20)
            return QPointF(round(value.x() / grid) * grid, round(value.y() / grid) * grid)
        if change == QGraphicsItem.ItemPositionHasChanged:
            scene = self.scene()
            if scene and getattr(scene, "_deleting_items", False):
                return super().itemChange(change, value)
            last = getattr(self, "_last_pos", None)
            self._last_pos = QPointF(value)
            # A construct owns its indicators: moving it carries the indicators along
            # (unless they are part of the same drag selection, which Qt already moves).
            if (
                last is not None
                and self.node_type == "latent"
                and not getattr(self, "_suppress_child_move", False)
            ):
                delta = value - last
                if not delta.isNull():
                    self._suppress_child_move = True
                    try:
                        for indicator in self.linked_indicators():
                            if not indicator.isSelected():
                                indicator.moveBy(delta.x(), delta.y())
                    finally:
                        self._suppress_child_move = False
            for line in list(self.connections):
                line.update_position()
            if scene and hasattr(scene, "update_result_overlays"):
                scene.update_result_overlays()
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.on_selected_changed(bool(value))
        return super().itemChange(change, value)

    def on_selected_changed(self, selected: bool) -> None:
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        self.rename()
        super().mouseDoubleClickEvent(event)

    def rename(self) -> None:
        new_name, ok = QInputDialog.getText(None, "Đổi tên", "Tên:", text=self.name)
        if ok and new_name.strip():
            if self.scene() and self.scene().views() and hasattr(self.scene().views()[0], "_remember"):
                self.scene().views()[0]._remember()
            self.set_name(new_name)

    def contextMenuEvent(self, event) -> None:
        if self.scene() and not self.isSelected():
            self.scene().clearSelection()
            self.setSelected(True)
        menu = QMenu()
        rename_action = menu.addAction(icon("rename", 17), "Rename")
        rename_action.setShortcut("F2")
        delete_action = menu.addAction(icon("delete", 17), "Delete")
        delete_action.setShortcut("Delete")
        selected = menu.exec(event.screenPos())
        if selected == rename_action:
            self.rename()
        elif selected == delete_action and self.scene():
            self.delete_from_context_menu()

    def delete_from_context_menu(self) -> None:
        scene = self.scene()
        if not scene:
            return
        items = list(scene.selectedItems()) or [self]
        QTimer.singleShot(0, lambda scene=scene, items=items: scene.delete_items(items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "name": self.name,
            "x": float(self.pos().x()),
            "y": float(self.pos().y()),
        }


class LatentNode(BaseNode):
    node_type = "latent"

    def __init__(self, name: str, x: float, y: float, mode: str = "reflective", node_id: str | None = None):
        super().__init__(name, x - 52, y - 52, node_id=node_id)
        self.measurement_mode = mode
        self.indicator_weighting = "automatic"
        self.status_complete = False
        self.custom_color: str | None = None
        self.effect_type: str = ""  # "", "interaction", or "quadratic"
        self.effect_refs: dict[str, str] = {}
        self.ellipse = QGraphicsEllipseItem(0, 0, 104, 104, self)
        self.ellipse.setZValue(0)
        self.ellipse.setGraphicsEffect(_node_shadow())
        self.text.setDefaultTextColor(QColor(NODE_PALETTE["text"]))
        self.text.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        option = QTextOption(Qt.AlignHCenter)
        self.text.document().setDefaultTextOption(option)
        self.apply_style()
        self.update_text_pos()

    def _set_gradient(self, top: str, bottom: str, border: str) -> None:
        grad = QLinearGradient(0, 0, 0, 104)
        grad.setColorAt(0.0, QColor(top))
        grad.setColorAt(1.0, QColor(bottom))
        self.ellipse.setBrush(QBrush(grad))
        self.ellipse.setPen(QPen(QColor(border), 1.6))

    def apply_style(self) -> None:
        p = NODE_PALETTE
        if self.custom_color:
            base = QColor(self.custom_color)
            self._set_gradient(base.lighter(116).name(), base.name(), base.darker(135).name())
        elif not self.status_complete:
            self._set_gradient(p["node_incomplete_a"], p["node_incomplete_b"],
                               QColor(p["node_incomplete_b"]).darker(115).name())
        elif self.measurement_mode == "reflective":
            self._set_gradient(p["node_reflective_a"], p["node_reflective_b"], p["node_border"])
        else:
            self._set_gradient(p["node_formative_a"], p["node_formative_b"], p["node_indicator_border"])

    def refresh_theme(self) -> None:
        self.apply_style()
        self.text.setDefaultTextColor(QColor(NODE_PALETTE["text"]))

    def set_measurement_mode(self, mode: str) -> None:
        self.measurement_mode = mode
        self.apply_style()
        # The arrow on each indicator connection flips direction with the mode,
        # so repaint them (formative -> arrow points into the construct).
        for line in self.connections:
            line.update()

    def update_text_pos(self) -> None:
        self.text.setTextWidth(132)
        rect = self.text.boundingRect()
        self.text.setPos(52 - 66, 106)

    def boundingRect(self) -> QRectF:
        text_rect = QRectF(self.text.pos(), self.text.boundingRect().size())
        return self.ellipse.boundingRect().adjusted(-6, -6, 6, 6).united(text_rect)

    def connection_rect(self) -> QRectF:
        return self.mapRectToScene(self.ellipse.rect())

    def paint(self, painter, option, widget=None) -> None:
        if not self.isSelected():
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(NODE_PALETTE["accent"]), 2.4, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(self.ellipse.rect().adjusted(-5, -5, 5, 5))

    def contextMenuEvent(self, event) -> None:
        if not self.isSelected():
            self.scene().clearSelection(); self.setSelected(True)
        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        menu = QMenu()
        delete_action = menu.addAction(icon("delete", 17), "Delete")
        delete_action.setShortcut("Delete")
        rename_action = menu.addAction(icon("rename", 17), "Rename")
        rename_action.setShortcut("F2")
        menu.addSeparator()
        moderating_action = menu.addAction(icon("moderating", 17), "Add Moderating Effect ...")
        quadratic_action = menu.addAction(icon("quadratic", 17), "Add Quadratic Effect ...")
        menu.addSeparator()
        switch_action = menu.addAction(icon("connection", 17), "Switch Between Formative/Reflective")
        show_action = menu.addAction(icon("plus", 17), "Show Indicators of Selected Constructs")
        hide_action = menu.addAction(icon("minus", 17), "Hide Indicators of Selected Constructs")
        menu.addSeparator()
        weighting_actions = {}
        for value, title in (("automatic", "Automatic"), ("mode_a", "Mode A"), ("mode_b", "Mode B"), ("sumscores", "Sumscores"), ("predefined", "Predefined")):
            weighting_actions[menu.addAction(icon("connection", 17), f"Set Indicator Weighting to '{title}'")] = value
        menu.addSeparator()
        indicator_align = {}
        for side in ("top", "left", "bottom", "right"):
            indicator_align[menu.addAction(icon(f"align-{side}", 17), f"Align Indicators {side.title()}")] = side
        selected_align = {}
        for side in ("top", "left", "bottom", "right"):
            selected_align[menu.addAction(icon(f"align-{side}", 17), f"Align Selected Element {side.title()}")] = side
        menu.addSeparator()
        match_width = menu.addAction(icon("match-width", 17), "Match Width")
        match_height = menu.addAction(icon("match-height", 17), "Match Height")
        menu.addSeparator()
        export_file = menu.addAction(icon("image", 17), "Export as Image to File")
        export_clipboard = menu.addAction(icon("clipboard", 17), "Export as Image to Clipboard")
        selected = menu.exec(event.screenPos())
        if selected == delete_action and self.scene():
            self.delete_from_context_menu()
        elif selected == rename_action:
            self.rename()
        elif view and selected == moderating_action:
            view.add_effect("moderating")
        elif view and selected == quadratic_action:
            view.add_effect("quadratic")
        elif view and selected == switch_action:
            view.switch_selected_modes()
        elif view and selected == show_action:
            view.set_selected_indicators_visible(True)
        elif view and selected == hide_action:
            view.set_selected_indicators_visible(False)
        elif view and selected in weighting_actions:
            view.set_selected_weighting(weighting_actions[selected])
        elif view and selected in indicator_align:
            view.align_indicators(indicator_align[selected])
        elif view and selected in selected_align:
            view.align_selected(selected_align[selected])
        elif view and selected == match_width:
            view.match_selected("width")
        elif view and selected == match_height:
            view.match_selected("height")
        elif view and selected == export_file:
            view.export_image()
        elif view and selected == export_clipboard:
            image = view.render_image()
            if not image.isNull():
                from PySide6.QtWidgets import QApplication
                QApplication.clipboard().setImage(image)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["mode"] = self.measurement_mode
        data["weighting"] = self.indicator_weighting
        data["color"] = self.custom_color or ""
        data["effect_type"] = self.effect_type
        data["effect_refs"] = dict(self.effect_refs)
        return data


class IndicatorNode(BaseNode):
    node_type = "indicator"

    def __init__(self, name: str, x: float, y: float, node_id: str | None = None):
        super().__init__(name, x - INDICATOR_WIDTH / 2, y - INDICATOR_HEIGHT / 2, node_id=node_id)
        self._rect = QRectF(0, 0, INDICATOR_WIDTH, INDICATOR_HEIGHT)
        self.rect_item = QGraphicsPathItem(_round_rect_path(self._rect, 7), self)
        self.rect_item.setZValue(0)
        self.rect_item.setGraphicsEffect(_node_shadow(blur=12, dy=2, alpha=45))
        self.text.setFont(QFont("Segoe UI", 9, QFont.Weight.Medium))
        option = QTextOption(Qt.AlignCenter)
        self.text.document().setDefaultTextOption(option)
        self.apply_style()
        self.update_text_pos()

    def apply_style(self) -> None:
        p = NODE_PALETTE
        grad = QLinearGradient(0, 0, 0, INDICATOR_HEIGHT)
        grad.setColorAt(0.0, QColor(p["node_indicator_a"]))
        grad.setColorAt(1.0, QColor(p["node_indicator_b"]))
        self.rect_item.setBrush(QBrush(grad))
        self.rect_item.setPen(QPen(QColor(p["node_indicator_border"]), 1.1))
        self.text.setDefaultTextColor(QColor(p["node_text"]))

    def refresh_theme(self) -> None:
        self.apply_style()

    def update_text_pos(self) -> None:
        self.text.setTextWidth(INDICATOR_WIDTH - 8)
        rect = self.text.boundingRect()
        self.text.setPos(4, INDICATOR_HEIGHT / 2 - rect.height() / 2)

    def boundingRect(self) -> QRectF:
        return self._rect.adjusted(-6, -6, 6, 6)

    def connection_rect(self) -> QRectF:
        return self.mapRectToScene(self._rect)

    def paint(self, painter, option, widget=None) -> None:
        if not self.isSelected():
            return
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(NODE_PALETTE["accent"]), 2.0, Qt.SolidLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self._rect.adjusted(-3, -3, 3, 3), 8, 8)


class CommentNode(BaseNode):
    node_type = "comment"

    def __init__(self, name: str, x: float, y: float, node_id: str | None = None):
        super().__init__(name, x, y, node_id=node_id)
        self.rect_item = QGraphicsRectItem(0, 0, 180, 72, self)
        self.rect_item.setZValue(0)
        self.rect_item.setBrush(QBrush(QColor("#fff7c7")))
        self.rect_item.setPen(QPen(QColor("#c5a83b"), 1.1, Qt.DashLine))
        self.text.setDefaultTextColor(QColor("#4d4528"))
        self.text.setFont(QFont("Segoe UI", 9))
        self.update_text_pos()

    def update_text_pos(self) -> None:
        self.text.setTextWidth(164)
        self.text.setPos(8, 8)

    def boundingRect(self) -> QRectF:
        return self.rect_item.boundingRect()

    def paint(self, painter, option, widget=None) -> None:
        return


class ModelCanvasScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(0, 0, 2600, 1800)
        self.mode = "select"
        self.temp_line: ConnectionLine | None = None
        self.result_badges: list[ResultBadge] = []
        self.node_count = 1
        self.grid_visible = False
        self.snap_enabled = False
        self.grid_size = 20
        self.background_color = QColor("#ffffff")
        self.grid_color = QColor("#e2e2e2")
        self._press_node: BaseNode | None = None
        self._drag_remembered = False
        self._deleting_items = False
        self._retired_items: list[QGraphicsItem] = []

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def _remember_view(self) -> None:
        if self.views() and hasattr(self.views()[0], "_remember"):
            self.views()[0]._remember()

    def node_at(self, position: QPointF) -> BaseNode | None:
        item = self.itemAt(position, QTransform())
        while item is not None:
            if isinstance(item, BaseNode):
                return item
            item = item.parentItem()
        return None

    def mousePressEvent(self, event) -> None:
        if self.mode == "latent" and event.button() == Qt.LeftButton:
            self._remember_view()
            pos = event.scenePos()
            self.addItem(LatentNode(f"Biến_{self.node_count}", pos.x(), pos.y()))
            self.node_count += 1
            return

        if self.mode == "comment" and event.button() == Qt.LeftButton:
            self._remember_view()
            pos = event.scenePos()
            self.addItem(CommentNode("Ghi chú giả thuyết", pos.x(), pos.y()))
            return

        if self.mode == "connect" and event.button() == Qt.LeftButton:
            node = self.node_at(event.scenePos())
            if isinstance(node, (LatentNode, IndicatorNode)):
                self._remember_view()
                self.temp_line = ConnectionLine(node)
                self.temp_line.setLine(QLineF(node.connection_rect().center(), event.scenePos()))
                self.addItem(self.temp_line)
            return

        if self.mode == "select" and event.button() == Qt.LeftButton:
            self._press_node = self.node_at(event.scenePos())
            self._drag_remembered = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.mode == "connect" and self.temp_line:
            self.temp_line.setLine(QLineF(self.temp_line.source_node.sceneBoundingRect().center(), event.scenePos()))
            return
        if (
            self.mode == "select"
            and self._press_node is not None
            and not self._drag_remembered
            and event.buttons() & Qt.LeftButton
            and QLineF(event.buttonDownScenePos(Qt.LeftButton), event.scenePos()).length() > 3
        ):
            self._remember_view()
            self._drag_remembered = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.mode == "connect" and self.temp_line:
            target = self.node_at(event.scenePos())
            source = self.temp_line.source_node
            normalized = self._normalize_connection(source, target)
            if normalized:
                source, target = normalized
                self.temp_line.source_node = source
                self.temp_line.target_node = target
                self.temp_line.update_position()
                source.add_connection(self.temp_line)
                target.add_connection(self.temp_line)
                if isinstance(source, LatentNode):
                    source.apply_style()
                self.refresh_node_status()
            else:
                self._retire_item(self.temp_line)
            self.temp_line = None
            return
        super().mouseReleaseEvent(event)
        self._press_node = None
        self._drag_remembered = False

    def _normalize_connection(self, source: BaseNode, target: BaseNode | None) -> tuple[BaseNode, BaseNode] | None:
        if not target or target == source:
            return None
        if isinstance(source, LatentNode) and isinstance(target, (LatentNode, IndicatorNode)):
            return source, target
        if isinstance(source, IndicatorNode) and isinstance(target, LatentNode):
            return target, source
        return None

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_items(list(self.selectedItems()), remember=True)
            return
        super().keyPressEvent(event)

    def delete_items(self, items: list[QGraphicsItem], remember: bool = True) -> None:
        items = self._live_model_items(items)
        if not items:
            return
        if remember:
            self._remember_view()
        targets = [
            item for item in self._expand_with_owned_indicators(items)
            if self._is_live_model_item(item)
        ]
        targets = self._dedupe_items(targets)
        nodes = [item for item in targets if isinstance(item, BaseNode)]
        lines = {item for item in targets if isinstance(item, ConnectionLine)}
        for node in nodes:
            lines.update(line for line in list(node.connections) if self._is_live_model_item(line))

        blocker = QSignalBlocker(self)
        previous_deleting = self._deleting_items
        self._deleting_items = True
        try:
            self.clearSelection()
            self.clear_result_overlays()
            for line in list(lines):
                self._remove_line(line, refresh=False)
            for node in nodes:
                node.connections.clear()
                self._retire_item(node)
        finally:
            self._deleting_items = previous_deleting
            del blocker
        self.refresh_node_status()
        self.update()

    def _retire_item(self, item: QGraphicsItem | None) -> None:
        if item is None:
            return
        try:
            item.setSelected(False)
            item.setVisible(False)
            item.setEnabled(False)
            if item.scene() is self:
                self.removeItem(item)
            self._retired_items.append(item)
        except RuntimeError:
            return

    def _retire_scene_items(self) -> None:
        for item in list(self.items()):
            try:
                if item.parentItem() is None:
                    self._retire_item(item)
            except RuntimeError:
                continue

    def _is_live_model_item(self, item: QGraphicsItem) -> bool:
        try:
            return isinstance(item, (BaseNode, ConnectionLine)) and item.scene() is self
        except RuntimeError:
            return False

    def _live_model_items(self, items: list[QGraphicsItem]) -> list[QGraphicsItem]:
        return self._dedupe_items([item for item in list(items) if self._is_live_model_item(item)])

    @staticmethod
    def _dedupe_items(items: list[QGraphicsItem]) -> list[QGraphicsItem]:
        result: list[QGraphicsItem] = []
        seen: set[int] = set()
        for item in items:
            marker = id(item)
            if marker in seen:
                continue
            result.append(item)
            seen.add(marker)
        return result

    def _expand_with_owned_indicators(self, items: list[QGraphicsItem]) -> list[QGraphicsItem]:
        """Deleting a construct must also delete the indicators it owns (SmartPLS behavior).

        An indicator is removed only when every construct it belongs to is being
        deleted, so shared indicators survive while a sibling construct remains.
        """
        result = list(items)
        seen = {id(item) for item in result}
        deleted_latents = {item for item in items if isinstance(item, LatentNode)}
        for latent in deleted_latents:
            for indicator in latent.linked_indicators():
                owners = indicator.linked_latents()
                if owners and all(owner in deleted_latents for owner in owners):
                    if id(indicator) not in seen:
                        result.append(indicator)
                        seen.add(id(indicator))
        return result

    def _remove_line(self, line: ConnectionLine, refresh: bool = True) -> None:
        source = getattr(line, "source_node", None)
        target = getattr(line, "target_node", None)
        if source:
            source.remove_connection(line)
            if isinstance(source, LatentNode):
                source.apply_style()
        if target:
            target.remove_connection(line)
        line.result_badge = None
        if self._is_live_model_item(line):
            self._retire_item(line)
        line.source_node = None
        line.target_node = None
        if refresh:
            self.refresh_node_status()

    def refresh_node_status(self) -> None:
        latents = [item for item in self.items() if isinstance(item, LatentNode)]
        normal = [node for node in latents if not getattr(node, "effect_type", "")]
        for node in latents:
            if getattr(node, "effect_type", ""):
                node.status_complete = True
                node.apply_style()
                continue
            live_connections = [
                line for line in node.connections
                if getattr(line, "source_node", None) and getattr(line, "target_node", None)
            ]
            indicators = [
                line for line in live_connections
                if isinstance(line.target_node, IndicatorNode) or isinstance(line.source_node, IndicatorNode)
            ]
            structural = [
                line for line in live_connections
                if isinstance(line.source_node, LatentNode) and isinstance(line.target_node, LatentNode)
            ]
            node.status_complete = bool(indicators) and len(normal) > 1 and bool(structural)
            node.apply_style()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        painter.fillRect(rect, self.background_color)
        if not self.grid_visible:
            return
        left = int(math.floor(rect.left() / self.grid_size) * self.grid_size)
        top = int(math.floor(rect.top() / self.grid_size) * self.grid_size)
        lines = []
        for x in range(left, int(rect.right()) + self.grid_size, self.grid_size):
            lines.append(QLineF(x, rect.top(), x, rect.bottom()))
        for y in range(top, int(rect.bottom()) + self.grid_size, self.grid_size):
            lines.append(QLineF(rect.left(), y, rect.right(), y))
        painter.setPen(QPen(self.grid_color, .7))
        painter.drawLines(lines)

    def serialize(self) -> dict[str, Any]:
        nodes = [item.to_dict() for item in self.items() if isinstance(item, BaseNode)]
        connections = [
            item.to_dict() for item in self.items()
            if isinstance(item, ConnectionLine) and item.source_node and item.target_node
        ]
        return {"nodes": nodes, "connections": connections}

    def load_model(self, model: dict[str, Any]) -> None:
        views = self.views()
        for view in views:
            view.setUpdatesEnabled(False)
        previous_signal_state = self.blockSignals(True)
        try:
            self._retire_scene_items()
            self.result_badges = []
            node_map: dict[str, BaseNode] = {}
            for node_data in model.get("nodes", []):
                node_type = node_data.get("type")
                if node_type == "latent":
                    node = LatentNode(
                        node_data.get("name", "LV"),
                        float(node_data.get("x", 0)) + 52,
                        float(node_data.get("y", 0)) + 52,
                        mode=node_data.get("mode", "reflective"),
                        node_id=node_data.get("id"),
                    )
                    node.indicator_weighting = node_data.get("weighting", "automatic")
                    node.custom_color = node_data.get("color") or None
                    node.effect_type = node_data.get("effect_type", "") or ""
                    node.effect_refs = dict(node_data.get("effect_refs", {}) or {})
                elif node_type == "indicator":
                    node = IndicatorNode(
                        node_data.get("name", "Indicator"),
                        float(node_data.get("x", 0)) + 56,
                        float(node_data.get("y", 0)) + 18,
                        node_id=node_data.get("id"),
                    )
                elif node_type == "comment":
                    node = CommentNode(
                        node_data.get("name", "Note"),
                        float(node_data.get("x", 0)),
                        float(node_data.get("y", 0)),
                        node_id=node_data.get("id"),
                    )
                else:
                    continue
                node_map[node.node_id] = node
                self.addItem(node)

            for line_data in model.get("connections", []):
                source = node_map.get(line_data.get("source"))
                target = node_map.get(line_data.get("target"))
                if source and target:
                    line = ConnectionLine(source, target)
                    source.add_connection(line)
                    target.add_connection(line)
                    self.addItem(line)
            self.node_count = 1 + len([node for node in node_map.values() if isinstance(node, LatentNode)])
            self.refresh_node_status()
        finally:
            self.blockSignals(previous_signal_state)
            for view in views:
                view.setUpdatesEnabled(True)
                view.viewport().update()
            self.update()

    def clear_result_overlays(self) -> None:
        for item in self.items():
            if isinstance(item, ConnectionLine):
                item.result_badge = None
            elif isinstance(item, LatentNode) and hasattr(item, "r_square_badge"):
                item.r_square_badge = None
        for badge in list(self.result_badges):
            if badge.scene() is self:
                self._retire_item(badge)
        self.result_badges = []

    def update_result_overlays(self) -> None:
        for item in self.items():
            if isinstance(item, ConnectionLine):
                item.update_result_badge_position()
            elif isinstance(item, LatentNode):
                badge = getattr(item, "r_square_badge", None)
                if badge:
                    center = item.connection_rect().center()
                    badge.set_center(center.x(), center.y() + 24)

    def show_results(self, results: dict[str, Any]) -> None:
        self.clear_result_overlays()
        paths = results.get("path_coefficients")
        loadings = results.get("outer_loadings")
        r_square = results.get("r_square")

        for item in self.items():
            if not isinstance(item, ConnectionLine) or not (item.source_node and item.target_node):
                continue
            source = item.source_node
            target = item.target_node
            line = item.line()
            mid = QPointF((line.x1() + line.x2()) / 2, (line.y1() + line.y2()) / 2)
            text = ""
            kind = "loading"
            good: bool | None = None
            if isinstance(source, LatentNode) and isinstance(target, LatentNode) and paths is not None:
                try:
                    text = f"β={float(paths.loc[source.name, target.name]):.3f}"
                    kind = "beta"
                except Exception:
                    text = ""
            elif isinstance(source, LatentNode) and isinstance(target, IndicatorNode) and loadings is not None:
                try:
                    value = float(loadings.loc[target.name, source.name])
                    text = f"{value:.3f}"
                    good = abs(value) >= 0.708
                except Exception:
                    text = ""
            elif isinstance(source, IndicatorNode) and isinstance(target, LatentNode) and loadings is not None:
                try:
                    value = float(loadings.loc[source.name, target.name])
                    text = f"{value:.3f}"
                    good = abs(value) >= 0.708
                except Exception:
                    text = ""
            if text:
                badge = ResultBadge(text, mid.x(), mid.y(), kind=kind, good=good)
                item.result_badge = badge
                item.update_result_badge_position()
                self.result_badges.append(badge)
                self.addItem(badge)

        if r_square is not None:
            for item in self.items():
                if isinstance(item, LatentNode):
                    try:
                        value = float(r_square.loc[item.name])
                    except Exception:
                        continue
                    center = item.connection_rect().center()
                    badge = ResultBadge(f"R²={value:.3f}", center.x(), center.y() + 24, kind="rsquare")
                    item.r_square_badge = badge
                    self.result_badges.append(badge)
                    self.addItem(badge)
        self.update_result_overlays()

    @staticmethod
    def _is_effect_node(item: QGraphicsItem) -> bool:
        return isinstance(item, LatentNode) and bool(getattr(item, "effect_type", ""))

    def extract_model(self) -> tuple[dict[str, list[str]], list[tuple[str, str]], dict[str, str]]:
        measurement: dict[str, list[str]] = {
            item.name: []
            for item in self.items()
            if isinstance(item, LatentNode) and not self._is_effect_node(item)
        }
        structural: list[tuple[str, str]] = []
        modes: dict[str, str] = {
            item.name: item.measurement_mode
            for item in self.items()
            if isinstance(item, LatentNode) and not self._is_effect_node(item)
        }
        indicator_positions: dict[str, tuple[float, float]] = {}

        for item in self.items():
            if not isinstance(item, ConnectionLine) or not (item.source_node and item.target_node):
                continue
            source = item.source_node
            target = item.target_node
            # Interaction / quadratic terms are handled separately via extract_effects().
            if self._is_effect_node(source) or self._is_effect_node(target):
                continue
            if isinstance(source, LatentNode) and isinstance(target, IndicatorNode):
                measurement.setdefault(source.name, []).append(target.name)
                rect = target.sceneBoundingRect()
                indicator_positions[target.name] = (rect.top(), rect.left())
            elif isinstance(source, IndicatorNode) and isinstance(target, LatentNode):
                measurement.setdefault(target.name, []).append(source.name)
                rect = source.sceneBoundingRect()
                indicator_positions[source.name] = (rect.top(), rect.left())
            elif isinstance(source, LatentNode) and isinstance(target, LatentNode):
                structural.append((source.name, target.name))

        measurement = {
            key: sorted(list(dict.fromkeys(value)), key=lambda name: indicator_positions.get(name, (0.0, 0.0)))
            for key, value in measurement.items()
        }
        structural = list(dict.fromkeys(structural))
        return measurement, structural, modes

    def has_model_nodes(self) -> bool:
        return any(isinstance(item, (BaseNode, ConnectionLine)) for item in self.items())

    def has_latent_constructs(self) -> bool:
        return any(isinstance(item, LatentNode) and not self._is_effect_node(item) for item in self.items())

    def used_indicators(self) -> list[str]:
        measurement, _, _ = self.extract_model()
        return list(dict.fromkeys(indicator for indicators in measurement.values() for indicator in indicators))

    def extract_effects(self) -> list[dict[str, str]]:
        """Interaction / quadratic terms, with construct references resolved to names."""
        id_to_name = {
            item.node_id: item.name
            for item in self.items()
            if isinstance(item, LatentNode)
        }
        effects: list[dict[str, str]] = []
        for item in self.items():
            if not self._is_effect_node(item):
                continue
            refs = item.effect_refs or {}
            if item.effect_type == "interaction":
                predictor = id_to_name.get(refs.get("predictor", ""))
                moderator = id_to_name.get(refs.get("moderator", ""))
                outcome = id_to_name.get(refs.get("outcome", ""))
                if predictor and moderator and outcome:
                    effects.append({
                        "type": "interaction", "name": item.name,
                        "predictor": predictor, "moderator": moderator,
                        "outcome": outcome, "method": refs.get("method", "two_stage"),
                    })
            elif item.effect_type == "quadratic":
                source = id_to_name.get(refs.get("source", ""))
                outcome = id_to_name.get(refs.get("outcome", ""))
                if source and outcome:
                    effects.append({
                        "type": "quadratic", "name": item.name,
                        "source": source, "outcome": outcome,
                        "method": refs.get("method", "two_stage"),
                    })
        return effects


class ModelCanvasView(QGraphicsView):
    workspace_swipe_requested = Signal(int)

    def __init__(self):
        super().__init__()
        self.scene = ModelCanvasScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setAcceptDrops(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#ffffff")))
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.SwipeGesture)
        self.viewport().grabGesture(Qt.PinchGesture)
        self.viewport().grabGesture(Qt.SwipeGesture)
        self.viewport().installEventFilter(self)
        self._zoom_factor = 1.0
        self._min_zoom = 0.2
        self._max_zoom = 5.0
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self._clipboard_model: dict[str, Any] = {"nodes": [], "connections": []}

    def set_mode(self, mode: str) -> None:
        self.scene.set_mode(mode)
        self.setDragMode(QGraphicsView.RubberBandDrag if mode == "select" else QGraphicsView.NoDrag)

    def zoom_in(self) -> None:
        self._zoom_by(1.15)

    def zoom_out(self) -> None:
        self._zoom_by(1 / 1.15)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom_factor = 1.0

    def fit_model(self) -> None:
        rect = self.scene.itemsBoundingRect()
        if rect.isEmpty():
            self.reset_zoom()
        else:
            self.fitInView(rect.adjusted(-60, -60, 60, 60), Qt.KeepAspectRatio)
            self._sync_zoom_factor()

    def _sync_zoom_factor(self) -> None:
        self._zoom_factor = max(self._min_zoom, min(self._max_zoom, float(self.transform().m11() or 1.0)))

    def _zoom_by(self, factor: float) -> None:
        if factor <= 0:
            return
        target = max(self._min_zoom, min(self._max_zoom, self._zoom_factor * factor))
        applied = target / self._zoom_factor if self._zoom_factor else target
        if abs(applied - 1.0) < 0.001:
            return
        self.scale(applied, applied)
        self._zoom_factor = target

    def smart_zoom(self) -> None:
        if not self.scene.has_model_nodes():
            self.reset_zoom()
            return
        if abs(self._zoom_factor - 1.0) > 0.08:
            self.reset_zoom()
        else:
            self.fit_model()

    def _pan_view_by(self, dx: float, dy: float) -> bool:
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return False
        horizontal = self.horizontalScrollBar()
        vertical = self.verticalScrollBar()
        horizontal.setValue(horizontal.value() - round(dx))
        vertical.setValue(vertical.value() - round(dy))
        return True

    def wheelEvent(self, event) -> None:
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        modifiers = event.modifiers()
        wants_zoom = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))

        if wants_zoom:
            delta = angle_delta.y() or pixel_delta.y()
            if delta == 0:
                event.ignore()
                return
            factor = 1.15 ** (delta / 120.0) if angle_delta.y() else 1.0015 ** delta
            self._zoom_by(factor)
            event.accept()
            return

        if not pixel_delta.isNull():
            dx, dy = float(pixel_delta.x()), float(pixel_delta.y())
        else:
            dx, dy = float(angle_delta.x()) * 0.5, float(angle_delta.y()) * 0.5
        if modifiers & Qt.ShiftModifier and abs(dx) < 0.01:
            dx, dy = dy, 0.0
        if self._pan_view_by(dx, dy):
            event.accept()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.LeftButton and item is None:
            self.smart_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.viewport() and self._handle_trackpad_event(event):
            return True
        return super().eventFilter(watched, event)

    def event(self, event) -> bool:
        if self._handle_trackpad_event(event):
            return True
        return super().event(event)

    def _handle_trackpad_event(self, event) -> bool:
        if event.type() == QEvent.Gesture:
            gesture = event.gesture(Qt.PinchGesture)
            if gesture is not None:
                scale = float(gesture.scaleFactor())
                last_scale = float(gesture.lastScaleFactor() or 1.0)
                factor = scale / last_scale if last_scale > 0 else scale
                if factor > 0:
                    self._zoom_by(factor)
                    event.accept()
                    return True
            swipe = event.gesture(Qt.SwipeGesture)
            if swipe is not None:
                horizontal = getattr(swipe, "horizontalDirection", lambda: None)()
                vertical = getattr(swipe, "verticalDirection", lambda: None)()
                if horizontal == QSwipeGesture.SwipeDirection.Left:
                    self.workspace_swipe_requested.emit(1)
                    event.accept()
                    return True
                if horizontal == QSwipeGesture.SwipeDirection.Right:
                    self.workspace_swipe_requested.emit(-1)
                    event.accept()
                    return True
                if vertical == QSwipeGesture.SwipeDirection.Up:
                    self.fit_model()
                    event.accept()
                    return True
                if vertical == QSwipeGesture.SwipeDirection.Down:
                    self.reset_zoom()
                    event.accept()
                    return True
        if event.type() == QEvent.NativeGesture:
            gesture_type = getattr(event, "gestureType", lambda: None)()
            if gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
                value = float(getattr(event, "value", lambda: 0.0)())
                factor = 1.0 + value
                if factor > 0:
                    self._zoom_by(factor)
                    event.accept()
                    return True
            if gesture_type == Qt.NativeGestureType.PanNativeGesture:
                delta = self._native_delta(event)
                if self._pan_view_by(delta.x(), delta.y()):
                    event.accept()
                    return True
            if gesture_type == Qt.NativeGestureType.SmartZoomNativeGesture:
                self.smart_zoom()
                event.accept()
                return True
            if gesture_type == Qt.NativeGestureType.SwipeNativeGesture:
                if self._handle_native_swipe(event):
                    event.accept()
                    return True
        return False

    def _native_delta(self, event) -> QPointF:
        delta_getter = getattr(event, "delta", None)
        if callable(delta_getter):
            delta = delta_getter()
            if delta is not None:
                return QPointF(float(delta.x()), float(delta.y()))
        value_getter = getattr(event, "value", None)
        if callable(value_getter):
            return QPointF(float(value_getter()), 0.0)
        return QPointF()

    def _handle_native_swipe(self, event) -> bool:
        delta = self._native_delta(event)
        dx, dy = delta.x(), delta.y()
        if abs(dx) >= abs(dy) and abs(dx) > 0.01:
            self.workspace_swipe_requested.emit(1 if dx < 0 else -1)
            return True
        if abs(dy) > 0.01:
            if dy < 0:
                self.fit_model()
            else:
                self.reset_zoom()
            return True
        return False

    def delete_selected(self) -> None:
        selected = list(self.scene.selectedItems())
        if not selected:
            return
        self._remember()
        self.scene.delete_items(selected, remember=False)

    def _remember(self) -> None:
        self._undo_stack.append(self.model_state())
        self._undo_stack = self._undo_stack[-50:]
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append(self.model_state())
        self.scene.load_model(self._undo_stack.pop())

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append(self.model_state())
        self.scene.load_model(self._redo_stack.pop())

    def select_all_items(self) -> None:
        for item in self.scene.items():
            if isinstance(item, (BaseNode, ConnectionLine)):
                item.setSelected(True)

    def rename_selected(self) -> bool:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, BaseNode)]
        if not nodes:
            return False
        nodes[0].rename()
        return True

    def copy_selected(self) -> None:
        selected = {item.node_id: item for item in self.scene.selectedItems() if isinstance(item, BaseNode)}
        if not selected:
            return
        nodes = [item.to_dict() for item in selected.values()]
        connections = [
            item.to_dict() for item in self.scene.items()
            if isinstance(item, ConnectionLine) and item.source_node and item.target_node
            and item.source_node.node_id in selected and item.target_node.node_id in selected
        ]
        self._clipboard_model = {"nodes": nodes, "connections": connections}

    def paste_selected(self) -> None:
        if not self._clipboard_model.get("nodes"):
            return
        self._remember()
        id_map: dict[str, BaseNode] = {}
        for data in self._clipboard_model["nodes"]:
            old_id = data["id"]
            x, y = float(data.get("x", 0)) + 28, float(data.get("y", 0)) + 28
            if data.get("type") == "latent":
                node = LatentNode(data.get("name", "LV"), x + 52, y + 52, data.get("mode", "reflective"))
                node.indicator_weighting = data.get("weighting", "automatic")
            elif data.get("type") == "indicator":
                node = IndicatorNode(data.get("name", "Indicator"), x + 56, y + 18)
            else:
                node = CommentNode(data.get("name", "Note"), x, y)
            self.scene.addItem(node)
            id_map[old_id] = node
        for data in self._clipboard_model.get("connections", []):
            source, target = id_map.get(data.get("source")), id_map.get(data.get("target"))
            if source and target:
                line = ConnectionLine(source, target)
                source.add_connection(line); target.add_connection(line); self.scene.addItem(line)
        self.scene.refresh_node_status()
        self.scene.clearSelection()
        for node in id_map.values():
            node.setSelected(True)
        self.copy_selected()

    def _view_center(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def add_latent_at_center(self, name: str) -> LatentNode:
        self._remember()
        center = self._view_center()
        node = LatentNode(name, center.x(), center.y())
        self.scene.addItem(node)
        self.scene.clearSelection(); node.setSelected(True)
        return node

    def add_latents(self, names: list[str]) -> None:
        if not names:
            return
        self._remember()
        center = self._view_center()
        self.scene.clearSelection()
        for index, name in enumerate(names):
            node = LatentNode(name, center.x() + (index % 3) * 190, center.y() + (index // 3) * 130)
            self.scene.addItem(node); node.setSelected(True)

    def add_note_at_center(self, text: str) -> None:
        self._remember()
        center = self._view_center()
        note = CommentNode(text, center.x(), center.y())
        self.scene.addItem(note); note.setSelected(True)

    def _connect(self, source: BaseNode, target: BaseNode) -> None:
        if any(
            getattr(line, "source_node", None) is source and getattr(line, "target_node", None) is target
            for line in source.connections
        ):
            return
        line = ConnectionLine(source, target)
        source.add_connection(line); target.add_connection(line); self.scene.addItem(line)
        self.scene.refresh_node_status()

    def _construct_nodes(self) -> list["LatentNode"]:
        nodes = [
            item for item in self.scene.items()
            if isinstance(item, LatentNode) and not getattr(item, "effect_type", "")
        ]
        nodes.sort(key=lambda node: node.name.lower())
        return nodes

    def add_effect(self, effect: str) -> bool:
        from gui.dialogs import EffectDialog

        constructs = self._construct_nodes()
        if effect == "moderating" and len(constructs) < 3:
            QMessageBox.information(
                self, "Moderating Effect",
                "Cần ít nhất 3 biến tiềm ẩn (biến độc lập, biến điều tiết, biến phụ thuộc) trước khi thêm hiệu ứng điều tiết.",
            )
            return False
        if effect == "quadratic" and len(constructs) < 2:
            QMessageBox.information(
                self, "Quadratic Effect",
                "Cần ít nhất 2 biến tiềm ẩn (biến nguồn và biến phụ thuộc) trước khi thêm hiệu ứng bậc hai.",
            )
            return False

        selected = [node.node_id for node in self.scene.selectedItems() if isinstance(node, LatentNode)]
        dialog = EffectDialog(effect, [(node.node_id, node.name) for node in constructs], selected, self)
        if not dialog.exec():
            return False
        spec = dialog.get_spec()
        if effect == "moderating":
            self.create_interaction_effect(spec["predictor"], spec["moderator"], spec["outcome"], spec["method"])
        else:
            self.create_quadratic_effect(spec["source"], spec["outcome"], spec["method"])
        return True

    def _latent_by_id(self, node_id: str) -> "LatentNode | None":
        return next(
            (item for item in self.scene.items() if isinstance(item, LatentNode) and item.node_id == node_id),
            None,
        )

    def create_interaction_effect(self, predictor_id: str, moderator_id: str, outcome_id: str, method: str = "two_stage") -> bool:
        predictor = self._latent_by_id(predictor_id)
        moderator = self._latent_by_id(moderator_id)
        outcome = self._latent_by_id(outcome_id)
        if not (predictor and moderator and outcome) or len({predictor_id, moderator_id, outcome_id}) < 3:
            QMessageBox.information(self, "Moderating Effect", "Hãy chọn ba biến tiềm ẩn khác nhau.")
            return False
        self._remember()
        center_p = predictor.connection_rect().center()
        center_m = moderator.connection_rect().center()
        x = (center_p.x() + center_m.x()) / 2
        y = max(center_p.y(), center_m.y()) + 160
        node = LatentNode(f"{predictor.name} x {moderator.name}", x, y, mode="formative")
        node.effect_type = "interaction"
        node.effect_refs = {"predictor": predictor_id, "moderator": moderator_id, "outcome": outcome_id, "method": method}
        node.custom_color = "#f6a623"
        node.apply_style()
        self.scene.addItem(node)
        self._connect(node, outcome)
        self._connect(moderator, outcome)
        self._connect(predictor, outcome)
        self.scene.refresh_node_status()
        self.scene.clearSelection(); node.setSelected(True)
        return True

    def create_quadratic_effect(self, source_id: str, outcome_id: str, method: str = "two_stage") -> bool:
        source = self._latent_by_id(source_id)
        outcome = self._latent_by_id(outcome_id)
        if not (source and outcome) or source_id == outcome_id:
            QMessageBox.information(self, "Quadratic Effect", "Hãy chọn biến nguồn và biến phụ thuộc khác nhau.")
            return False
        self._remember()
        center = source.connection_rect().center()
        node = LatentNode(f"{source.name}²", center.x() + 40, center.y() + 170, mode="formative")
        node.effect_type = "quadratic"
        node.effect_refs = {"source": source_id, "outcome": outcome_id, "method": method}
        node.custom_color = "#9b59b6"
        node.apply_style()
        self.scene.addItem(node)
        self._connect(node, outcome)
        self._connect(source, outcome)
        self.scene.refresh_node_status()
        self.scene.clearSelection(); node.setSelected(True)
        return True

    def switch_selected_modes(self) -> bool:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, LatentNode)]
        if not nodes:
            return False
        self._remember()
        for node in nodes:
            node.set_measurement_mode("formative" if node.measurement_mode == "reflective" else "reflective")
        return True

    def set_selected_weighting(self, weighting: str) -> bool:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, LatentNode)]
        if not nodes:
            return False
        self._remember()
        for node in nodes:
            node.indicator_weighting = weighting
            if weighting == "mode_a": node.set_measurement_mode("reflective")
            if weighting == "mode_b": node.set_measurement_mode("formative")
        return True

    def set_selected_indicators_visible(self, visible: bool) -> None:
        constructs = [item for item in self.scene.selectedItems() if isinstance(item, LatentNode)]
        if not constructs:
            return
        self._remember()
        for construct in constructs:
            for line in construct.connections:
                source = getattr(line, "source_node", None)
                target = getattr(line, "target_node", None)
                if not (source and target):
                    continue
                indicator = target if source is construct else source
                if isinstance(indicator, IndicatorNode):
                    line.setVisible(visible)
                    indicator.setVisible(visible)
            # SmartPLS bolds the construct label while its indicators are collapsed.
            font = construct.text.font()
            font.setBold(not visible)
            construct.text.setFont(font)
            construct.update_text_pos()

    def align_indicators(self, side: str) -> None:
        constructs = [item for item in self.scene.selectedItems() if isinstance(item, LatentNode)]
        if not constructs:
            constructs = [item for item in self.scene.items() if isinstance(item, LatentNode)]
        if not constructs:
            return
        self._remember()
        for construct in constructs:
            self._place_indicators(construct, self._indicator_nodes(construct), side)

    def _indicator_nodes(self, construct: LatentNode) -> list[IndicatorNode]:
        indicators: list[IndicatorNode] = []
        for line in construct.connections:
            source = getattr(line, "source_node", None)
            target = getattr(line, "target_node", None)
            if not (source and target):
                continue
            if source is construct and isinstance(target, IndicatorNode):
                indicator = line.target_node
            elif target is construct and isinstance(source, IndicatorNode):
                indicator = line.source_node
            else:
                continue
            if indicator not in indicators:
                indicators.append(indicator)
        return indicators

    def _indicator_side_for_drop(self, position: QPointF, construct: LatentNode) -> str:
        body = construct.connection_rect()
        center = body.center()
        dx = position.x() - center.x()
        dy = position.y() - center.y()
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        if dy > body.height() * 0.25:
            return "bottom"
        if dy < -body.height() * 0.25:
            return "top"
        return "left"

    def _place_indicators(self, construct: LatentNode, indicators: list[IndicatorNode], side: str) -> None:
        if not indicators:
            return
        side = side if side in {"top", "left", "bottom", "right"} else "left"
        body = construct.connection_rect()
        center = body.center()
        horizontal_step = INDICATOR_WIDTH + INDICATOR_GAP
        vertical_step = INDICATOR_HEIGHT + INDICATOR_GAP
        for index, node in enumerate(indicators):
            if side in {"top", "bottom"}:
                offset = (index - (len(indicators) - 1) / 2) * horizontal_step
                x = center.x() + offset - INDICATOR_WIDTH / 2
                y = body.top() - CONSTRUCT_INDICATOR_GAP - INDICATOR_HEIGHT if side == "top" else body.bottom() + CONSTRUCT_INDICATOR_GAP
            else:
                offset = (index - (len(indicators) - 1) / 2) * vertical_step
                x = body.right() + CONSTRUCT_INDICATOR_GAP if side == "right" else body.left() - CONSTRUCT_INDICATOR_GAP - INDICATOR_WIDTH
                y = center.y() + offset - INDICATOR_HEIGHT / 2
            node.setPos(x, y)

    def align_selected(self, side: str) -> None:
        items = [item for item in self.scene.selectedItems() if isinstance(item, BaseNode)]
        if len(items) < 2:
            return
        self._remember()
        rects = [item.sceneBoundingRect() for item in items]
        target = {"left": min(r.left() for r in rects), "right": max(r.right() for r in rects), "top": min(r.top() for r in rects), "bottom": max(r.bottom() for r in rects)}[side]
        for item in items:
            rect = item.sceneBoundingRect()
            if side == "left": item.moveBy(target - rect.left(), 0)
            elif side == "right": item.moveBy(target - rect.right(), 0)
            elif side == "top": item.moveBy(0, target - rect.top())
            else: item.moveBy(0, target - rect.bottom())

    def match_selected(self, dimension: str) -> None:
        items = [item for item in self.scene.selectedItems() if isinstance(item, (IndicatorNode, CommentNode))]
        if len(items) < 2:
            return
        self._remember()
        source = items[0].boundingRect()
        for item in items[1:]:
            rect_item = getattr(item, "rect_item", None)
            if rect_item:
                rect = rect_item.rect()
                if dimension == "width": rect.setWidth(source.width())
                else: rect.setHeight(source.height())
                rect_item.setRect(rect)
                item.update_text_pos()

    def render_image(self) -> QImage:
        rect = self.scene.itemsBoundingRect().adjusted(-40, -40, 40, 40)
        if rect.isEmpty():
            return QImage()
        image = QImage(max(1, int(rect.width())), max(1, int(rect.height())), QImage.Format_ARGB32)
        image.fill(QColor("#ffffff"))
        painter = QPainter(image)
        self.scene.render(painter, QRectF(image.rect()), rect)
        painter.end()
        return image

    def export_image(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Xuất sơ đồ", "", "Ảnh PNG (*.png)")
        if not path:
            return
        image = self.render_image()
        if image.isNull():
            return
        image.save(path)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasText():
            super().dropEvent(event)
            return

        texts = [text.strip() for text in event.mimeData().text().split(",") if text.strip()]
        if not texts:
            return
        pos = self.mapToScene(event.position().toPoint())
        target = self.scene.node_at(pos)

        self.add_indicator_group(texts, pos, target if isinstance(target, LatentNode) else None)
        event.acceptProposedAction()

    def add_indicator_group(
        self,
        indicators: list[str],
        position: QPointF,
        target: LatentNode | None = None,
    ) -> LatentNode:
        clean_indicators = [item.strip() for item in indicators if item.strip()]
        if not clean_indicators:
            raise ValueError("Không có biến quan sát để thêm vào mô hình.")

        self._remember()
        if target is None:
            construct_name = self._suggest_construct_name(clean_indicators)
            target = LatentNode(construct_name, position.x(), position.y())
            self.scene.addItem(target)
            self.scene.node_count += 1

        side = self._indicator_side_for_drop(position, target)
        for text in clean_indicators:
            node = IndicatorNode(text, position.x(), position.y())
            self.scene.addItem(node)
            line = ConnectionLine(target, node)
            target.add_connection(line)
            node.add_connection(line)
            self.scene.addItem(line)
        self._place_indicators(target, self._indicator_nodes(target), side)
        target.apply_style()
        self.scene.refresh_node_status()
        self.scene.clearSelection()
        target.setSelected(True)
        return target

    def _suggest_construct_name(self, indicators: list[str]) -> str:
        prefixes: list[str] = []
        for indicator in indicators:
            prefix = indicator.rstrip("0123456789_-. ")
            prefixes.append(prefix or indicator)
        candidate = prefixes[0]
        if candidate and all(prefix == candidate for prefix in prefixes):
            return candidate
        return f"Biến tiềm ẩn {self.scene.node_count}"

    def model_state(self) -> dict[str, Any]:
        return self.scene.serialize()

    def load_model_state(self, model: dict[str, Any]) -> None:
        self.scene.load_model(model)
        self._undo_stack.clear()
        self._redo_stack.clear()
        if self.scene.items():
            self.fit_model()

    def show_results(self, results: dict[str, Any]) -> None:
        self.scene.show_results(results)

    def extract_model(self) -> tuple[dict[str, list[str]], list[tuple[str, str]], dict[str, str]]:
        return self.scene.extract_model()

    def has_model_nodes(self) -> bool:
        return self.scene.has_model_nodes()

    def has_latent_constructs(self) -> bool:
        return self.scene.has_latent_constructs()

    def extract_effects(self) -> list[dict[str, str]]:
        return self.scene.extract_effects()

    def used_indicators(self) -> list[str]:
        return self.scene.used_indicators()

    def set_theme_colors(self, canvas: str, grid: str) -> None:
        self.scene.background_color = QColor(canvas)
        self.scene.grid_color = QColor(grid)
        self.setBackgroundBrush(QBrush(QColor(canvas)))
        self.scene.update()
        self.viewport().update()

    def set_palette(self, theme: str) -> None:
        """Recolour every node/edge to match the active application theme."""
        NODE_PALETTE.clear()
        NODE_PALETTE.update(ui_theme.palette(theme))
        for item in self.scene.items():
            refresh = getattr(item, "refresh_theme", None)
            if callable(refresh):
                refresh()
        self.scene.update()
        self.viewport().update()

    def toggle_grid(self, enabled: bool) -> None:
        self.scene.grid_visible = enabled
        self.scene.update()

    def toggle_snap(self, enabled: bool) -> None:
        self.scene.snap_enabled = enabled

    def set_selected_color(self, color: str) -> None:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, LatentNode)]
        if not nodes:
            return
        self._remember()
        for node in nodes:
            node.custom_color = color
            node.apply_style()

    def adjust_selected_font(self, delta: int = 0, bold: bool | None = None, italic: bool | None = None) -> None:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, BaseNode)]
        if not nodes:
            return
        self._remember()
        for node in nodes:
            font = node.text.font()
            if delta:
                font.setPointSize(max(6, font.pointSize() + delta))
            if bold is not None:
                font.setBold(bold)
            if italic is not None:
                font.setItalic(italic)
            node.text.setFont(font)
            node.update_text_pos()

    def adjust_selected_border(self, delta: float) -> None:
        nodes = [item for item in self.scene.selectedItems() if isinstance(item, BaseNode)]
        if not nodes:
            return
        self._remember()
        for node in nodes:
            shape = getattr(node, "ellipse", None) or getattr(node, "rect_item", None)
            if shape:
                pen = shape.pen()
                pen.setWidthF(max(.5, pen.widthF() + delta))
                shape.setPen(pen)

    def show_model_hint(self) -> None:
        QMessageBox.information(
            self,
            "Mô hình",
            "Dùng Biến tiềm ẩn để thêm construct, kéo biến quan sát từ cột trái, "
            "sau đó dùng Vẽ đường dẫn để tạo quan hệ cấu trúc.",
        )

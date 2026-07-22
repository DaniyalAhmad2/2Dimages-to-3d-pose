"""Camera panel: image + draggable joint overlay + bones.

QGraphicsView/Scene with a QGraphicsPixmapItem background and one draggable
JointItem per joint. Dragging emits (cam, joint, scene-pos); bones are
QGraphicsLineItems refreshed when either endpoint moves. RAG colouring per joint.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget,
)

from pose3d.core.skeleton import BONES, NUM_JOINTS, rag_status

RAG_COLORS = {
    "green": QColor(80, 220, 120),
    "amber": QColor(240, 190, 70),
    "red": QColor(235, 90, 90),
    "corrected": QColor(170, 120, 240),
}


class _JointSignals(QObject):
    moved = Signal(int, QPointF)     # joint id, new scene pos
    picked = Signal(int)             # joint id selected


class JointItem(QGraphicsEllipseItem):
    R = 6.0

    def __init__(self, joint_id: int):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self.joint_id = joint_id
        self.signals = _JointSignals()
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(10)
        self.setPen(QPen(QColor(20, 20, 20), 1))
        self.set_status("green")

    def set_status(self, status: str):
        self.setBrush(QBrush(RAG_COLORS.get(status, RAG_COLORS["red"])))

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            self.signals.moved.emit(self.joint_id, value)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self.signals.picked.emit(self.joint_id)
        super().mousePressEvent(event)


class CameraView(QGraphicsView):
    """One camera's editable overlay."""
    jointDragged = Signal(str, int, QPointF)   # cam, joint, pos
    jointPicked = Signal(str, int)

    def __init__(self, cam: str):
        super().__init__()
        self.cam = cam
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self._pixmap_item = None
        self._joints: list[JointItem] = []
        self._bones: list[QGraphicsLineItem] = []
        self._show_joints = True
        self._show_bones = True
        self._build_items()

    def _build_items(self):
        for a, b in BONES:
            line = QGraphicsLineItem()
            line.setPen(QPen(QColor(220, 220, 220, 180), 2))
            line.setZValue(5)
            self._scene.addItem(line)
            self._bones.append(line)
        for j in range(NUM_JOINTS):
            item = JointItem(j)
            item.signals.moved.connect(self._on_moved)
            item.signals.picked.connect(
                lambda jid: self.jointPicked.emit(self.cam, jid))
            self._scene.addItem(item)
            self._joints.append(item)

    def set_image(self, path: str):
        pm = QPixmap(path)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pm)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pm)
        if not pm.isNull():
            self._scene.setSceneRect(QRectF(pm.rect()))
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_pose(self, xy: np.ndarray, scores: np.ndarray,
                 corrected: np.ndarray | None = None):
        """Place joints from (NUM_JOINTS,2) pixel coords + scores."""
        for j, item in enumerate(self._joints):
            p = xy[j]
            if np.isnan(p).any():
                item.setVisible(False)
                continue
            item.setVisible(self._show_joints)
            # suppress the move signal while we set position programmatically
            # (QGraphicsItem is not a QObject; the Signal lives on item.signals)
            item.signals.blockSignals(True)
            item.setPos(float(p[0]), float(p[1]))
            item.signals.blockSignals(False)
            if corrected is not None and corrected[j]:
                item.set_status("corrected")
            else:
                s = scores[j]
                item.set_status("red" if np.isnan(s) else rag_status(float(s)))
        self._refresh_bones()

    def _on_moved(self, joint_id: int, pos: QPointF):
        self._refresh_bones()
        self.jointDragged.emit(self.cam, joint_id, pos)

    def _refresh_bones(self):
        for (a, b), line in zip(BONES, self._bones):
            ja, jb = self._joints[int(a)], self._joints[int(b)]
            if not (ja.isVisible() and jb.isVisible()) or not self._show_bones:
                line.setVisible(False)
                continue
            line.setVisible(True)
            pa, pb = ja.pos(), jb.pos()
            line.setLine(pa.x(), pa.y(), pb.x(), pb.y())

    def set_show_joints(self, on: bool):
        self._show_joints = on
        for it in self._joints:
            it.setVisible(on and not np.isnan(it.pos().x()))
        self._refresh_bones()

    def set_show_bones(self, on: bool):
        self._show_bones = on
        self._refresh_bones()

    def resizeEvent(self, event):
        if self._pixmap_item is not None and not self._zoomed:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

    # --- toolbar actions ---
    _zoomed = False

    def zoom(self, factor: float):
        self._zoomed = True
        self.scale(factor, factor)

    def fit(self):
        self._zoomed = False
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_pan_mode(self, on: bool):
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if on
                         else QGraphicsView.DragMode.NoDrag)

    def wheelEvent(self, event):
        self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)


class CameraPanel(QWidget):
    """Camera view with a header (title + filename) and a vertical toolbar."""

    def __init__(self, cam: str, title: str):
        super().__init__()
        self.view = CameraView(cam)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        tlab = QLabel(title); tlab.setObjectName("panelTitle")
        self.filename = QLabel(""); self.filename.setObjectName("fileLabel")
        header.addWidget(tlab); header.addStretch(1); header.addWidget(self.filename)
        outer.addLayout(header)

        body = QHBoxLayout(); body.setSpacing(4)
        tools = QVBoxLayout(); tools.setSpacing(4)
        specs = [("⭱", "Select", lambda: self.view.set_pan_mode(False)),
                 ("✋", "Pan", lambda: self.view.set_pan_mode(True)),
                 ("＋", "Zoom in", lambda: self.view.zoom(1.25)),
                 ("－", "Zoom out", lambda: self.view.zoom(1 / 1.25)),
                 ("⤢", "Fit", self.view.fit)]
        for glyph, tip, fn in specs:
            b = QToolButton(); b.setText(glyph); b.setToolTip(tip)
            b.setObjectName("camTool"); b.clicked.connect(fn)
            tools.addWidget(b)
        tools.addStretch(1)
        body.addLayout(tools)
        body.addWidget(self.view, 1)
        outer.addLayout(body, 1)

    def set_filename(self, name: str):
        self.filename.setText(name)

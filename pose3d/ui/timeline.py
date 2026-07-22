"""Filmstrip timeline: horizontal thumbnails with RAG status dots.

QListView in IconMode (single non-wrapping row) + a delegate that paints the
thumbnail and a status dot. currentChanged drives frame selection.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QListView, QStyledItemDelegate, QStyle,
)

_STATUS_ROLE = Qt.ItemDataRole.UserRole + 1

STATUS_COLORS = {
    "green": QColor(80, 220, 120),
    "amber": QColor(240, 190, 70),
    "red": QColor(235, 90, 90),
    "corrected": QColor(170, 120, 240),
}


class _ThumbDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        # selection border
        if option.state & QStyle.StateFlag.State_Selected:
            painter.save()
            painter.setPen(QColor(120, 130, 255))
            r = option.rect.adjusted(1, 1, -2, -2)
            painter.drawRect(r)
            painter.restore()
        # status dot (top-right)
        status = index.data(_STATUS_ROLE) or "green"
        painter.save()
        painter.setBrush(STATUS_COLORS.get(status, STATUS_COLORS["red"]))
        painter.setPen(Qt.PenStyle.NoPen)
        r = option.rect
        painter.drawEllipse(r.right() - 16, r.top() + 6, 10, 10)
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(104, 92)


class Timeline(QListView):
    frameSelected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setIconSize(QSize(84, 64))
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(120)
        self._model = QStandardItemModel(self)
        self.setModel(self._model)
        self.setItemDelegate(_ThumbDelegate(self))
        self.selectionModel().currentChanged.connect(
            lambda cur, _prev: self.frameSelected.emit(cur.row()))

    def populate(self, frames, load_thumb):
        self._model.clear()
        for f in frames:
            item = QStandardItem(f.frame_id)
            path = f.images.get("left") or next(iter(f.images.values()), None)
            if path:
                pm = QPixmap(path)
                if not pm.isNull():
                    item.setData(pm.scaled(84, 64, Qt.AspectRatioMode.KeepAspectRatio),
                                 Qt.ItemDataRole.DecorationRole)
            item.setData("green", _STATUS_ROLE)
            item.setEditable(False)
            self._model.appendRow(item)

    def set_status(self, idx: int, status: str):
        it = self._model.item(idx)
        if it:
            it.setData(status, _STATUS_ROLE)

    def select(self, idx: int):
        self.setCurrentIndex(self._model.index(idx, 0))

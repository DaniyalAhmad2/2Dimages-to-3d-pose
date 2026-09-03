"""Filmstrip timeline: horizontal thumbnails with RAG status dots.

QListView in IconMode (single non-wrapping row) + a delegate that paints the
thumbnail and a status dot. currentChanged drives frame selection.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QListView, QSizePolicy,
    QStyledItemDelegate, QStyle, QWidget,
)

_STATUS_ROLE = Qt.ItemDataRole.UserRole + 1

#: Preferred height of the filmstrip. A minimum, never a maximum.
TIMELINE_HEIGHT = 120

STATUS_COLORS = {
    "green": QColor(80, 220, 120),
    "amber": QColor(240, 190, 70),
    "red": QColor(235, 90, 90),
    "corrected": QColor(170, 120, 240),
}


class TimelineHeader(QWidget):
    """'TIMELINE (N FRAMES)' + status legend + a Show filter dropdown."""

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("TIMELINE"); self.title.setObjectName("sectionHeader")
        lay.addWidget(self.title); lay.addStretch(1)
        for txt, key in (("High", "green"), ("Low", "amber"),
                         ("Missing", "red"), ("Corrected", "corrected")):
            dot = QLabel("●"); dot.setStyleSheet(
                f"color: {STATUS_COLORS[key].name()};")
            lab = QLabel(txt); lab.setObjectName("legendLabel")
            lay.addWidget(dot); lay.addWidget(lab)
        lay.addSpacing(10)
        lay.addWidget(QLabel("Show"))
        combo = QComboBox(); combo.addItems(["All Frames", "Needs Review", "Corrected"])
        lay.addWidget(combo)

    def set_count(self, n: int):
        self.title.setText(f"TIMELINE ({n} FRAMES)")


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

    def sizeHint(self):
        return QSize(super().sizeHint().width(), TIMELINE_HEIGHT)

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
        # A MINIMUM, not a fixed height: setFixedHeight pins the maximum
        # too, so the filmstrip could never be given another pixel. sizeHint()
        # keeps 120 as the preferred height, so the dashboard is unchanged.
        self.setMinimumHeight(TIMELINE_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Minimum)
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

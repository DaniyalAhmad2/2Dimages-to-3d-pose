"""Filmstrip timeline: horizontal thumbnails with RAG status dots.

QListView in IconMode (single non-wrapping row) + a delegate that paints the
thumbnail and a status dot. currentChanged drives frame selection.
"""
from __future__ import annotations

import numpy as np
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

#: The Show filter: (label, the frame status it keeps); "" keeps every frame.
#: The statuses are the ones the dots are painted with and the legend names,
#: so the control can never come to mean something the strip does not show.
SHOW_FILTERS = (("All Frames", ""), ("Low", "amber"), ("Missing", "red"),
                ("Corrected", "corrected"))


def _has_correction(frame) -> bool:
    """Has the user hand-corrected any keypoint in this frame, in any view?

    `Frame.has_corrections()` and nothing else. The question has two halves —
    the body joints keep `corrected`, the face points keep `head_corrected` —
    and this used to read the first alone, so a frame whose only hand work was
    a nose placed by hand reported itself uncorrected: green dot, and absent
    from Show = Corrected, the one view whose whole job is to find the frames
    the user has worked on. The frame answers it, so a third kind of hand edit
    reaches the dot without this module hearing about it.
    """
    return bool(frame.has_corrections())


class TimelineHeader(QWidget):
    """'TIMELINE (N FRAMES)' + status legend + a Show filter dropdown."""

    #: the status the strip should keep, "" for all of them
    filterChanged = Signal(str)

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
        # the option carries the STATUS it keeps, so the labels are free to
        # read like the legend without the filtering depending on the words
        self.show_combo = QComboBox()
        for label, status in SHOW_FILTERS:
            self.show_combo.addItem(label, status)
        self.show_combo.currentIndexChanged.connect(
            lambda _i: self.filterChanged.emit(self.show_combo.currentData()))
        lay.addWidget(self.show_combo)

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
        # the frame's accuracy band, and whether the user has corrected it,
        # kept apart: the caller bands frames by residual and knows nothing
        # about corrections, so it must not be able to erase one
        self._band: dict[int, str] = {}
        self._corrected: set[int] = set()
        self._filter = ""
        # True while the MODEL is moving the highlight — see `select`
        self._syncing = False
        self.selectionModel().currentChanged.connect(self._on_current_changed)

    def _on_current_changed(self, cur, _prev):
        if self._syncing:
            return
        self.frameSelected.emit(cur.row())

    def populate(self, frames, load_thumb):
        self._model.clear()
        self._band.clear()
        self._corrected.clear()
        for i, f in enumerate(frames):
            item = QStandardItem(f.frame_id)
            path = f.images.get("left") or next(iter(f.images.values()), None)
            if path:
                pm = QPixmap(path)
                if not pm.isNull():
                    item.setData(pm.scaled(84, 64, Qt.AspectRatioMode.KeepAspectRatio),
                                 Qt.ItemDataRole.DecorationRole)
            item.setEditable(False)
            self._model.appendRow(item)
            # a reopened project brings its corrections with it, and the
            # Corrected filter would otherwise be empty in every one of them
            if _has_correction(f):
                self._corrected.add(i)
            self._paint(i)
        self._apply_filter()

    def status(self, idx: int) -> str:
        """What this frame's dot says: corrected outranks the band."""
        if idx in self._corrected:
            return "corrected"
        return self._band.get(idx, "green")

    def set_status(self, idx: int, status: str):
        self._band[idx] = status
        self._paint(idx)
        self._apply_filter(idx)

    def refresh_corrected(self, idx: int, frame):
        """Re-read this frame's corrections: a drag has just made one, or an
        undo has just taken the last one back."""
        if _has_correction(frame):
            self._corrected.add(idx)
        else:
            self._corrected.discard(idx)
        self._paint(idx)
        self._apply_filter(idx)

    def set_filter(self, status: str):
        """Show only the frames whose dot is `status` ("" for all of them).

        Rows are HIDDEN, never removed or reordered, so a frame's row number
        stays its frame index: `select` and `frameSelected` mean the same
        thing filtered or not, and the current frame is left where it is —
        picking a filter is a way of looking at the take, not a way of moving
        through it.
        """
        self._filter = status or ""
        self._apply_filter()

    def _paint(self, idx: int):
        it = self._model.item(idx)
        if it:
            it.setData(self.status(idx), _STATUS_ROLE)

    def _apply_filter(self, idx: int | None = None):
        rows = range(self._model.rowCount()) if idx is None else (idx,)
        for i in rows:
            self.setRowHidden(i, bool(self._filter)
                              and self.status(i) != self._filter)

    def select(self, idx: int):
        """Highlight frame `idx` because the MODEL moved there.

        Silent: the caller is answering `frameChanged`, so re-emitting
        `frameSelected` would run back through `set_frame` -> `frameChanged`
        -> `select` for a frame the model is already on, once per step.
        """
        if idx == self.currentIndex().row():
            return
        self._syncing = True
        try:
            self.setCurrentIndex(self._model.index(idx, 0))
        finally:
            self._syncing = False

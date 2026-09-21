"""The filmstrip: a highlight that follows the frame, and a Show filter that
does something.

The "Show" dropdown in the timeline header came from the client's own
dashboard layout and was wired to nothing — a local variable in the
constructor with no receiver, so picking an option did nothing at all. With
his 26-frame take banding 2 green / 22 amber / 2 red, triaging by status is
exactly what it appears to offer.

The highlight had the matching gap in the other direction: `select()` was only
ever called when a project loaded, so a frame change that did not come from a
thumbnail click (keyboard stepping, any programmatic `set_frame`) left the
strip pointing at the wrong frame.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pose3d.core.skeleton import NUM_JOINTS                          # noqa: E402
from pose3d.ui.timeline import (                                     # noqa: E402
    SHOW_FILTERS, Timeline, TimelineHeader)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _Frame:
    """The three things `populate` reads off a frame.

    `has_corrections` among them, and not the raw `corrected` array: the
    frame answers "did a human place any point here" itself, because the
    answer has two halves (body joints and face points keep separate flags)
    and a caller that reads one of them under-reports the other.
    """

    def __init__(self, i: int, corrected: bool = False):
        self.frame_id = f"{i:04d}"
        self.images = {}
        flags = np.zeros(NUM_JOINTS, bool)
        flags[0] = corrected
        self.corrected = {"left": flags, "right": np.zeros(NUM_JOINTS, bool)}

    def has_corrections(self, cam: str | None = None) -> bool:
        cams = self.corrected if cam is None else (cam,)
        return any(bool(np.any(self.corrected[c])) for c in cams)


def _timeline(qapp, n=14, corrected=()):
    tl = Timeline()
    tl.populate([_Frame(i, i in corrected) for i in range(n)], None)
    return tl


# --- the highlight follows the model -------------------------------------

def test_select_highlights_without_re_emitting_frame_selected(qapp):
    """`select` answers frameChanged. Emitting frameSelected from it would
    run straight back into set_frame -> frameChanged -> select, once per
    frame step, for a frame the model has already moved to."""
    tl = _timeline(qapp)
    heard = []
    tl.frameSelected.connect(heard.append)

    tl.select(12)

    assert tl.currentIndex().row() == 12
    assert heard == []


def test_selecting_the_frame_already_current_changes_nothing(qapp):
    tl = _timeline(qapp)
    tl.select(5)
    heard = []
    tl.frameSelected.connect(heard.append)

    tl.select(5)

    assert heard == []
    assert tl.currentIndex().row() == 5


def test_the_user_picking_a_thumbnail_still_reports_it(qapp):
    """The suppression is one-directional: a click must still drive the
    model, which is the only reason the filmstrip is clickable."""
    tl = _timeline(qapp)
    heard = []
    tl.frameSelected.connect(heard.append)

    tl.setCurrentIndex(tl.model().index(7, 0))      # what a click does

    assert heard == [7]


# --- the Show filter ------------------------------------------------------

def test_the_filter_offers_the_statuses_the_legend_paints(qapp):
    header = TimelineHeader()
    labels = [header.show_combo.itemText(i)
              for i in range(header.show_combo.count())]
    assert labels == [label for label, _ in SHOW_FILTERS]
    assert header.show_combo.currentData() == "", "'All Frames' is the default"

    heard = []
    header.filterChanged.connect(heard.append)
    header.show_combo.setCurrentIndex(2)
    assert heard == ["red"], "the header must report the STATUS, not the label"


def test_missing_hides_every_frame_that_is_not_missing(qapp):
    tl = _timeline(qapp, n=6)
    for i, status in enumerate(
            ["green", "amber", "red", "amber", "green", "red"]):
        tl.set_status(i, status)

    tl.set_filter("red")

    assert [i for i in range(6) if not tl.isRowHidden(i)] == [2, 5]


def test_a_filter_leaves_the_frame_indices_alone(qapp):
    """Rows are hidden, never removed or reordered: `select(idx)` and
    `frameSelected` both still mean 'frame idx' while a filter is on."""
    tl = _timeline(qapp, n=6)
    for i, status in enumerate(
            ["green", "amber", "red", "amber", "green", "red"]):
        tl.set_status(i, status)
    heard = []
    tl.frameSelected.connect(heard.append)

    tl.set_filter("red")
    tl.select(5)
    assert tl.currentIndex().row() == 5
    assert tl.model().rowCount() == 6

    tl.setCurrentIndex(tl.model().index(2, 0))
    assert heard == [2]


def test_all_frames_brings_the_hidden_ones_back(qapp):
    tl = _timeline(qapp, n=6)
    for i, status in enumerate(
            ["green", "amber", "red", "amber", "green", "red"]):
        tl.set_status(i, status)

    tl.set_filter("red")
    tl.set_filter("")

    assert not any(tl.isRowHidden(i) for i in range(6))


def test_a_status_set_later_re_applies_the_filter(qapp):
    """Re-banding happens after a recalculation; the strip must not keep
    showing a frame that no longer matches what the user asked for."""
    tl = _timeline(qapp, n=4)
    for i in range(4):
        tl.set_status(i, "amber")
    tl.set_filter("red")
    assert all(tl.isRowHidden(i) for i in range(4))

    tl.set_status(2, "red")
    assert not tl.isRowHidden(2)


# --- corrected is a fact about the frame, not a band ----------------------

def test_a_corrected_frame_keeps_its_dot_through_a_re_band(qapp):
    """`set_status` bands frames by residual and knows nothing about
    corrections, so it may not erase one. The legend and the filter both
    promise 'Corrected'."""
    tl = _timeline(qapp, n=4)
    tl.set_status(2, "amber")
    tl.refresh_corrected(2, _Frame(2, corrected=True))
    assert tl.status(2) == "corrected"

    tl.set_status(2, "green")                # a recalculation re-bands it
    assert tl.status(2) == "corrected"

    tl.set_filter("corrected")
    assert [i for i in range(4) if not tl.isRowHidden(i)] == [2]


def test_a_reopened_project_remembers_which_frames_were_corrected(qapp):
    """The corrections are on the frames; `populate` must read them, or the
    Corrected filter would be empty in every reopened project."""
    tl = _timeline(qapp, n=4, corrected=(1, 3))

    assert tl.status(1) == "corrected" and tl.status(3) == "corrected"
    tl.set_filter("corrected")
    assert [i for i in range(4) if not tl.isRowHidden(i)] == [1, 3]


def test_undoing_the_last_correction_clears_the_dot(qapp):
    """It is re-read from the frame, not latched: `undo` puts the keypoint
    back, and a frame with nothing corrected must not claim it is."""
    tl = _timeline(qapp, n=4)
    tl.set_status(2, "amber")
    tl.refresh_corrected(2, _Frame(2, corrected=True))

    tl.refresh_corrected(2, _Frame(2, corrected=False))

    assert tl.status(2) == "amber"

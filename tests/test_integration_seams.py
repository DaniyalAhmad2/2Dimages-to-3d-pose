"""The seams five parallel tasks left between each other, closed and pinned.

Each task was built, reviewed and merged on its own branch, so every fact that
crosses two of them — a signal one task emits and another must answer, a rule
two tasks each stated, a value one returns and another must show — had nobody
to hold it. These tests are that holder: they assert the JOIN, end to end
where the join is only observable end to end, so a future edit to either side
cannot quietly take it apart again.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pose3d.core.project import CAM_LEFT, CAM_RIGHT                # noqa: E402
from pose3d.core.skeleton import NUM_JOINTS, Joint                 # noqa: E402
from tests.test_ui_smoke import _project_with_rig                  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------
# Seam 1 — the 3D joint colouring T4 built had no hook to reach it
# --------------------------------------------------------------------------

def _window_with_three_kinds_of_joint(tmp_path):
    """A window whose current frame holds a corrected, a missing and an
    ordinary joint — the three cases the client's complaint names."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, _gt = _project_with_rig()
    f = data.frames[0]
    # missing in BOTH views: no 3D was ever reconstructed for it
    missing = int(Joint.LEFT_WRIST)
    for cam in (CAM_LEFT, CAM_RIGHT):
        f.kp2d[cam][missing] = np.nan
        f.scores[cam][missing] = np.nan
    f.pose3d[missing] = np.nan
    f.fitted3d[missing] = np.nan

    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    win = MainWindow(model)
    model.set_frame(0)
    # a hand correction, made the way the user makes one
    corrected = int(Joint.RIGHT_KNEE)
    xy = f.kp2d[CAM_LEFT][corrected]
    model.set_joint_2d(CAM_LEFT, corrected, float(xy[0]) + 3.0, float(xy[1]))
    xy = f.kp2d[CAM_RIGHT][corrected]
    model.set_joint_2d(CAM_RIGHT, corrected, float(xy[0]) + 3.0, float(xy[1]))
    return win, model, {"missing": missing, "corrected": corrected}


def test_the_3d_view_is_banded_by_the_same_rule_as_the_camera_panels(
        qapp, tmp_path):
    """The whole point of T4's banding, and it reached nothing.

    `View3D.set_joint_status` was written, tested and merged, and no line in
    `MainWindow` ever called it — so on the running app every 3D joint stayed
    one cyan, which is the client's 2026-07-26 complaint verbatim. Asserted
    against the CAMERA PANEL's own answer for the same joint, not against a
    copy of the rule.
    """
    win, model, j = _window_with_three_kinds_of_joint(tmp_path)

    statuses = win.view3d._joint_statuses(model.frame().filled)
    assert statuses is not None, "the 3D view was never told the frame's status"

    for joint in range(NUM_JOINTS):
        left = win.cam_left.view._joint_status(joint)[0]
        right = win.cam_right.view._joint_status(joint)[0]
        if left == right:          # the merge has nothing to choose between
            assert statuses[joint] == left, (
                f"joint {joint}: 3D says {statuses[joint]!r}, "
                f"the camera panels say {left!r}")

    assert statuses[j["corrected"]] == "corrected"
    assert statuses[j["missing"]] == "unmeasured"


def test_the_3d_joints_are_actually_drawn_in_those_colours(qapp, tmp_path):
    """…and the colours reach the scatter, not just the status cache.

    The pose and the banding arrive on two different signals and nothing
    orders them, so a hook that ran before the pose would leave the drawing
    uncoloured.
    """
    from pose3d.ui.camera_view import RAG_COLORS
    from pose3d.ui.view3d import _status_rgba

    win, model, j = _window_with_three_kinds_of_joint(tmp_path)
    colors = np.asarray(win.view3d._scatter.color, float)
    assert colors.ndim == 2, "the 3D joints were drawn in one flat colour"

    # the overlay draws the joints its own mask keeps, in that order
    _pts, mask, _filled = win.view3d._last_draw
    drawn = np.flatnonzero(mask)
    row = int(np.flatnonzero(drawn == j["corrected"])[0])
    assert np.allclose(colors[row], _status_rgba("corrected"))
    assert RAG_COLORS["corrected"].name() != RAG_COLORS["green"].name()


# --------------------------------------------------------------------------
# Seam 2 — T2 and T4 each wrote the joint-status rule
# --------------------------------------------------------------------------

def test_the_camera_view_asks_the_shared_joint_status_rule(qapp, monkeypatch):
    """Two copies of the rule is the defect, not two copies that agree today.

    `camera_view._joint_status` (T2) and `panels.joint_status` (T4) stated the
    same five-clause order independently. They agree on every input as merged,
    which is exactly why nothing would catch the next edit to one of them —
    and "the two panels disagree about which joints are flagged" is the
    client's own complaint. So this pins the CALL: the camera view must get
    its band from the shared function, not from a copy that matches it.
    """
    from pose3d.ui import camera_view as cv

    monkeypatch.setattr(cv, "joint_status", lambda *a, **k: "corrected")
    panel = cv.CameraPanel("LEFT VIEW", CAM_LEFT)
    panel.view.set_pose(np.zeros((NUM_JOINTS, 2)), np.ones(NUM_JOINTS))
    panel.set_accuracy(np.full(NUM_JOINTS, 0.001), None, None)

    assert panel.view._joint_status(0)[0] == "corrected"


# --------------------------------------------------------------------------
# Seam 3 — T1 gave face edits their own flag; T2's dot and filter never asked
# --------------------------------------------------------------------------

def test_a_face_only_edit_turns_the_frames_dot_purple(qapp, tmp_path):
    """A nose placed by hand IS a correction to that frame.

    T1 split the flags in two — `Frame.corrected` for the body joints,
    `Frame.head_corrected` for the face points — and gave the question one
    answer, `Frame.has_corrections()`. T2's timeline had already shipped
    reading `Frame.corrected` alone, so a frame whose only hand work was on
    the face reported itself uncorrected: its dot stayed green and it was
    missing from Show = Corrected, which is the one view that exists to find
    the frames the user has worked on.
    """
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, _gt = _project_with_rig()
    for f in data.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.head2d[cam][:] = 100.0
            f.head_scores[cam][:] = 1.0
    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    win = MainWindow(model)
    model.set_frame(1)
    assert win.timeline.status(1) == "green"

    nose = NUM_JOINTS + 0
    model.set_joint_2d(CAM_LEFT, nose, 140.0, 160.0)

    assert data.frames[1].has_corrections()
    assert not data.frames[1].corrected[CAM_LEFT].any(), "a FACE edit only"
    assert win.timeline.status(1) == "corrected"

    win.timeline_header.show_combo.setCurrentIndex(
        win.timeline_header.show_combo.findData("corrected"))
    assert not win.timeline.isRowHidden(1)
    assert win.timeline.isRowHidden(0) and win.timeline.isRowHidden(2)


def test_a_reopened_project_shows_its_face_corrections_too(qapp):
    """`populate` asks the same question — a project carries its corrections
    back from disk, and the filter was empty in every reopened one."""
    from pose3d.ui.timeline import Timeline

    data, _rig, _gt = _project_with_rig()
    data.frames[2].set_head_kp(CAM_RIGHT, 0, 10.0, 20.0, corrected=True)

    strip = Timeline()
    strip.populate(data.frames, load_thumb=None)

    assert strip.status(2) == "corrected"
    assert strip.status(0) == "green"


@pytest.mark.parametrize("state", ["ok", "rejected", "not_measured"])
@pytest.mark.parametrize("err", [0.0005, 0.007, 0.05, float("nan")])
@pytest.mark.parametrize("filled,corrected", [(False, False), (True, False),
                                              (False, True), (True, True)])
def test_both_panels_band_a_joint_the_same_way(state, err, filled, corrected):
    """…and the answer itself is unchanged for every combination."""
    from pose3d.ui.panels import joint_status
    from pose3d.ui.model import (
        STATE_NOT_MEASURED, STATE_OK, STATE_REJECTED)

    states = {"ok": STATE_OK, "rejected": STATE_REJECTED,
              "not_measured": STATE_NOT_MEASURED}
    got = joint_status(states[state], err, filled, corrected)
    if corrected:
        assert got == "corrected"
    elif filled:
        assert got == "filled"
    elif state == "rejected":
        assert got == "rejected"
    elif state == "not_measured" or not np.isfinite(err):
        assert got == "unmeasured"
    else:
        assert got in ("green", "amber", "red")

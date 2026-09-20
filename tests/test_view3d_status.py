"""The 3D preview bands its joints exactly as the 2D overlays band theirs.

The client's oldest open complaint, 2026-07-26: "you also cant see which
joints are flagged as red on either the images on the left or the generated
one on the right. This is something that was highlighted in the reference
image." The 2D half was delivered; the 3D preview went on drawing all fifteen
joints in one cyan, and a joint the cross-view gate refused — purple in the 2D
views — was simply absent from it with no marking at all.

These tests compare the colour the 3D view gives a joint against the colour
the CAMERA PANEL gives the same joint for the same frame, so "the same rule"
is asserted against the other panel rather than against a copy of the rule.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pose3d.core.skeleton import NUM_JOINTS, Joint            # noqa: E402
from pose3d.ui.camera_view import (                           # noqa: E402
    HOLLOW_STATES, RAG_COLORS, CameraPanel)
from pose3d.ui.model import (                                 # noqa: E402
    STATE_NOT_MEASURED, STATE_OK, STATE_REJECTED)
from pose3d.ui.view3d import View3D                           # noqa: E402


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _view():
    """A View3D with no GL behind it, recording what it draws.

    `__new__`, like tests/test_view_orientation.py's helper: the banding is
    arithmetic over the model's own numbers and must be assertable without a
    GL context. Only the scatter is stubbed — the colours are what this file
    is about, so `_draw_skeleton` itself is the real one.
    """
    v = View3D.__new__(View3D)
    v._R = np.eye(3)
    v._vaxis, v._vsign = 2, 1.0
    v._character = None
    v._take = None
    v._place = None
    v._framed = True
    v._show_body = True
    v._show_capture = False
    v._char_error = ""
    v._char_error_source = ""
    v._status = None
    v._last_draw = None
    v.drawn = {}

    class _Item:
        def __init__(self, name):
            self.name, self.data = name, {}

        def setData(self, **kw):
            self.data.update(kw)
            v.drawn[self.name] = self.data

        def setVisible(self, on):
            pass

    v._scatter, v._lines = _Item("char"), _Item("char_lines")
    v._cap_scatter, v._cap_lines = _Item("capture"), _Item("capture_lines")
    v._body = _Item("body")
    v._body.opts = {}
    v._set_body = lambda verts, faces: None
    v._skin = lambda vpose, vhead=None: (None, None, None, 0.0)
    return v


def _rgba(color):
    return np.array([color.redF(), color.greenF(), color.blueF(), 1.0])


def _drawn_joints(view):
    """Which joint each drawn point is — the overlay draws only valid ones."""
    return np.flatnonzero(view._last_draw[1])


def _colors(view):
    """{joint: rgba} for the joints the character overlay drew."""
    data = view.drawn["char"]
    return {int(j): np.asarray(c, float)
            for j, c in zip(_drawn_joints(view), data["color"])}


def _pose():
    """A standing pose in view space, one joint per row, nothing NaN."""
    p = np.zeros((NUM_JOINTS, 3), float)
    p[:, 2] = np.linspace(1.7, 0.0, NUM_JOINTS)
    return p


# The frame under test: one excellent joint, one poor one, one the pipeline
# never triangulated, one interpolated across a dropout, one placed by hand
# and one the cross-view gate refused.
GOOD, BAD = int(Joint.NECK), int(Joint.LEFT_WRIST)
MISSING, FILLED = int(Joint.RIGHT_ELBOW), int(Joint.LEFT_KNEE)
CORRECTED, REJECTED = int(Joint.RIGHT_HIP), int(Joint.LEFT_ANKLE)


def _frame():
    """(states, errors, corrected, filled) as main_window holds them."""
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT
    states = [STATE_OK] * NUM_JOINTS
    states[MISSING] = STATE_NOT_MEASURED
    states[REJECTED] = STATE_REJECTED
    errs = np.full(NUM_JOINTS, 0.001)          # 0.1 % of height: High
    errs[BAD] = 0.025                          # ...and one that is not
    errs[MISSING] = np.nan
    errs[REJECTED] = np.nan
    corrected = np.zeros(NUM_JOINTS, bool); corrected[CORRECTED] = True
    filled = np.zeros(NUM_JOINTS, bool); filled[FILLED] = True
    errors = {c: {"measured": errs.copy(), "delivered": errs.copy()}
              for c in (CAM_LEFT, CAM_RIGHT)}
    return ({c: list(states) for c in (CAM_LEFT, CAM_RIGHT)}, errors,
            {c: corrected.copy() for c in (CAM_LEFT, CAM_RIGHT)}, filled)


def _panel_colors(qapp):
    """What the 2D camera panel paints the same frame's joints."""
    from PySide6.QtCore import Qt
    from pose3d.core.project import CAM_LEFT
    states, errors, corrected, filled = _frame()
    p = CameraPanel("left", "LEFT CAMERA")
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    p.view.set_pose(xy, np.full(NUM_JOINTS, 0.9),
                    corrected=corrected[CAM_LEFT], filled=filled)
    p.set_accuracy(errors[CAM_LEFT]["measured"],
                   errors[CAM_LEFT]["delivered"], states[CAM_LEFT])
    out = {}
    for j, item in enumerate(p.view._joints):
        # a hollow joint carries its colour in the pen, a measured one in the
        # brush — the 2D overlay's own way of saying "this is not a
        # measurement", which the 3D view says with size and opacity
        hollow = item.brush().style() == Qt.BrushStyle.NoBrush
        out[j] = item.pen().color() if hollow else item.brush().color()
    return out


def test_every_3d_joint_takes_the_2d_overlay_s_colour(qapp):
    states, errors, corrected, filled = _frame()
    view = _view()
    view.set_joint_status(states, errors, corrected)
    view.set_pose(_pose(), filled=filled)

    got, want = _colors(view), _panel_colors(qapp)
    for j in range(NUM_JOINTS):
        assert np.allclose(got[j][:3], _rgba(want[j])[:3], atol=1e-6), (
            f"joint {j}: 3D {got[j]} vs 2D {want[j].name()}")


def test_the_named_states_are_the_colours_the_client_was_promised(qapp):
    """The colours themselves, so the cross-check above cannot pass by both
    panels being wrong together."""
    states, errors, corrected, filled = _frame()
    view = _view()
    view.set_joint_status(states, errors, corrected)
    view.set_pose(_pose(), filled=filled)
    got = _colors(view)

    for j, key in ((GOOD, "green"), (BAD, "red"), (MISSING, "unmeasured"),
                   (FILLED, "filled"), (CORRECTED, "corrected"),
                   (REJECTED, "rejected")):
        assert np.allclose(got[j][:3], _rgba(RAG_COLORS[key])[:3], atol=1e-6), j


def test_a_joint_with_no_measurement_is_not_drawn_as_one(qapp):
    """Filled, unmeasured and rejected are drawn as the 2D views draw them —
    as something other than a solid measured dot. The 2D overlay has a hollow
    ring for it; a GL scatter has size and opacity, so they are used."""
    states, errors, corrected, filled = _frame()
    view = _view()
    view.set_joint_status(states, errors, corrected)
    view.set_pose(_pose(), filled=filled)
    data = view.drawn["char"]
    idx = {int(j): k for k, j in enumerate(_drawn_joints(view))}
    size, color = np.asarray(data["size"]), np.asarray(data["color"])

    for j in (MISSING, FILLED, REJECTED):
        assert size[idx[j]] < size[idx[GOOD]], j
        assert color[idx[j]][3] < color[idx[GOOD]][3], j
    # ...and a hand-placed joint IS a position the user chose, so it stays a
    # solid dot, exactly as the 2D view keeps it one
    assert "corrected" not in HOLLOW_STATES
    assert color[idx[CORRECTED]][3] == color[idx[GOOD]][3]


def test_a_rejected_joint_is_marked_rather_than_missing(qapp):
    """A joint the gate refused has NO 3D of its own, so the capture skeleton
    cannot show it. The character's own joint is there — the rig poses the
    limb anyway — and that is what carries the purple, which is the whole
    point: the client could not tell a refused joint from one that simply is
    not drawn."""
    states, errors, corrected, filled = _frame()
    pose = _pose()
    pose[REJECTED] = np.nan                    # no triangulation for it
    cj = _pose()                               # ...but the rig posed the leg
    view = _view()
    view._skin = lambda vpose, vhead=None: (np.zeros((1, 3)), None, cj, 0.0)
    view.set_joint_status(states, errors, corrected)
    view.set_pose(pose, filled=filled)

    got = _colors(view)
    assert REJECTED in got
    assert np.allclose(got[REJECTED][:3], _rgba(RAG_COLORS["rejected"])[:3])
    # and the MEASURED overlay still refuses to draw a point it does not have
    measured = ~np.isnan(pose).any(1) & ~filled
    assert not measured[REJECTED]
    assert len(view.drawn["capture"]["pos"]) == int(measured.sum())


def test_without_statuses_the_view_draws_as_it_always_did(qapp):
    """`set_joint_status` is additive: a View3D nobody hands statuses to keeps
    the single-colour skeleton, so a caller that has not been wired up yet
    draws exactly what it drew before."""
    view = _view()
    view.set_pose(_pose())
    colors = np.asarray(view.drawn["char"]["color"], float)
    assert np.allclose(colors, np.asarray(View3D.JOINT_COLOR, float))


def test_statuses_arriving_after_the_pose_still_colour_it(qapp):
    """The model emits the pose and the accuracy on two signals, and nothing
    guarantees which reaches the view first."""
    states, errors, corrected, filled = _frame()
    view = _view()
    view.set_pose(_pose(), filled=filled)
    view.set_joint_status(states, errors, corrected)
    got = _colors(view)
    assert np.allclose(got[BAD][:3], _rgba(RAG_COLORS["red"])[:3])

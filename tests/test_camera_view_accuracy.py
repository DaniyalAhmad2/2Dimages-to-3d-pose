"""Accuracy lives on the keypoints: colour bands + hover tooltips.

This replaced the SELECTED JOINT panel — the joint's name and quality appear
where the user is already looking, on the dot they are about to drag.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pose3d.core.skeleton import JOINT_NAMES, NUM_JOINTS  # noqa: E402
from pose3d.ui.camera_view import RAG_COLORS, CameraPanel  # noqa: E402


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _panel(qapp):
    p = CameraPanel("left", "LEFT CAMERA")
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    scores = np.full(NUM_JOINTS, 0.9)
    p.view.set_pose(xy, scores)
    return p


def test_dots_band_by_accuracy_and_tooltips_name_the_joint(qapp):
    p = _panel(qapp)
    errs = np.full(NUM_JOINTS, 0.5)      # ~92%: High everywhere...
    errs[3] = 30.0                       # ...except one terrible joint
    p.set_accuracy(errs)

    good, bad = p.view._joints[2], p.view._joints[3]
    assert good.brush().color() == RAG_COLORS["green"]
    assert bad.brush().color() == RAG_COLORS["red"]

    # hover text: the joint's name and its accuracy figure
    assert JOINT_NAMES[3] in bad.toolTip()
    assert "%" in bad.toolTip()
    assert "accuracy" in bad.toolTip().lower()


def test_without_accuracy_confidence_still_colours_the_dots(qapp):
    """Before calibration/triangulation there is no reprojection error; the
    detector's confidence keeps the RAG colouring meaningful."""
    p = _panel(qapp)
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    scores = np.full(NUM_JOINTS, 0.9)
    scores[5] = 0.1                      # a joint the detector barely saw
    p.view.set_pose(xy, scores)
    p.set_accuracy(None)

    assert p.view._joints[0].brush().color() == RAG_COLORS["green"]
    assert p.view._joints[5].brush().color() == RAG_COLORS["red"]
    assert "confidence" in p.view._joints[5].toolTip().lower()
    assert JOINT_NAMES[5] in p.view._joints[5].toolTip()


def test_corrected_joints_stay_purple_and_say_so(qapp):
    p = _panel(qapp)
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    corrected = np.zeros(NUM_JOINTS, bool)
    corrected[4] = True
    p.view.set_pose(xy, np.full(NUM_JOINTS, 0.9), corrected)
    p.set_accuracy(np.full(NUM_JOINTS, 1.0))

    assert p.view._joints[4].brush().color() == RAG_COLORS["corrected"]
    assert "corrected" in p.view._joints[4].toolTip().lower()


def test_accuracy_then_pose_order_does_not_matter(qapp):
    """The two updates arrive from different signals; whichever lands last
    must not erase the other's contribution."""
    p = _panel(qapp)
    errs = np.full(NUM_JOINTS, 30.0)
    p.set_accuracy(errs)                 # accuracy first...
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    p.view.set_pose(xy, np.full(NUM_JOINTS, 0.9))   # ...then the pose
    assert p.view._joints[0].brush().color() == RAG_COLORS["red"]

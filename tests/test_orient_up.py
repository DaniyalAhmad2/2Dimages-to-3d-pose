"""Which way is up — and why the calibration cannot tell us.

The world frame's axes come from whichever ArUco tag the calibration happened
to pick, and tags taped at different rotations define different "ups" (6, 92
and 89 degrees apart on the client's rig). So resolve_up levels on the
subject's own body line instead, and the UI says that is what it did.
"""
import numpy as np

from pose3d.core.skeleton import Joint
from pose3d.geometry.orient import (
    _frame_up, de_tilt_matrix, resolve_up, sequence_up,
)
from tests.synth import sample_skeleton_3d


def _rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, float); k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def _torso_tilt(p):
    t = p[int(Joint.NECK)] - p[int(Joint.PELVIS)]
    return float(np.degrees(np.arccos(np.clip(t[2] / np.linalg.norm(t), -1, 1))))


def test_markers_taped_at_different_rotations_are_not_a_vertical_reference():
    """Why resolve_up does NOT trust the calibration frame's axes.

    ArUco tags encode their own orientation, and on the client's rig the wall
    tags are taped at different rotations — three markers in one image put
    "up" 6, 92 and 89 degrees from the camera's. resolve_calibration picks the
    lowest-id common tag, so the world frame's vertical is whatever rotation
    that one happens to have. A version of resolve_up that snapped to the
    nearest world axis inherited that and leaned the figure ~26 degrees.
    """
    base = sample_skeleton_3d()
    # a world frame rotated 90 deg about the view axis, as a sideways tag gives
    R = _rot([0, 1, 0], 90.0)
    seq = np.stack([base @ R.T] * 3)
    up, source = resolve_up(seq)
    assert source == "estimated"
    # levelling still stands the figure up rather than following the bogus axis
    assert _torso_tilt(seq[0] @ de_tilt_matrix(up).T) < 6.0


def test_levelling_stands_the_subject_up_whatever_the_world_frame():
    """The world frame is arbitrary, so orientation must not depend on it."""
    base = sample_skeleton_3d()
    R = _rot([1, 1, 0], 45.0)           # up lands ~45 deg from every axis
    seq = np.stack([base @ R.T] * 3)
    up, source = resolve_up(seq)
    assert source == "estimated"
    out = seq[0] @ de_tilt_matrix(up).T
    assert _torso_tilt(out) < 6.0       # normalised, as before


def test_no_usable_frames_returns_none():
    up, source = resolve_up(np.full((2, 15, 3), np.nan))
    assert up is None and source == "estimated"


def test_frame_up_ignores_the_nose():
    """HEAD is the nose — forward of the body axis. It tipped the estimated
    up ~10 deg forward on a real take; the NECK must be the top reference."""
    p = sample_skeleton_3d()
    u1 = _frame_up(p, ~np.isnan(p).any(1))
    q = p.copy()
    q[int(Joint.HEAD)] = q[int(Joint.HEAD)] + np.array([0.0, 0.5, 0.0])
    u2 = _frame_up(q, ~np.isnan(q).any(1))
    assert np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1))) < 0.1

    # ...but HEAD still serves when the neck is missing
    q[int(Joint.NECK)] = np.nan
    assert _frame_up(q, ~np.isnan(q).any(1)) is not None

"""Which way is up — and whose up is it.

The world frame comes from the calibration markers; when they are taped
square its AXES are gravity-true even though none is nominally +Z. resolve_up
snaps to the nearest axis so the subject's genuine lean — including lean held
through a whole take, which body-line levelling silently erased — survives
into the 3D view and the export.
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


def test_up_snaps_to_a_world_axis_when_one_is_close():
    """An upright subject in a world whose up is -Y (a wall-mounted board):
    resolve_up must return the exact axis, not the subject's own body line."""
    base = sample_skeleton_3d()
    R = _rot([1, 0, 0], -90.0)          # world up becomes -Y
    seq = np.stack([base @ R.T for _ in range(3)])
    up, source = resolve_up(seq)
    assert source == "axis"
    assert np.allclose(np.abs(up), [0, 1, 0])
    assert np.linalg.norm(up) == 1.0


def test_sustained_lean_survives_axis_snapping():
    """The reported defect: the subject leans through the WHOLE take, and the
    old body-line levelling defined that lean as vertical and erased it."""
    base = sample_skeleton_3d()
    piv = base[int(Joint.PELVIS)].copy()
    leaned = (base - piv) @ _rot([1, 0, 0], 15.0).T + piv
    seq = np.stack([leaned] * 3)        # 15 deg lean, every frame

    up, source = resolve_up(seq)
    assert source == "axis"             # 15 < 35: the axis wins
    out = seq[0] @ de_tilt_matrix(up).T
    assert _torso_tilt(out) > 10.0, "sustained lean was flattened"

    # the old behaviour, for contrast: body-line levelling erases it
    old = seq[0] @ de_tilt_matrix(sequence_up(seq)).T
    assert _torso_tilt(old) < 6.0


def test_single_frame_project_keeps_its_lean():
    base = sample_skeleton_3d()
    piv = base[int(Joint.PELVIS)].copy()
    leaned = (base - piv) @ _rot([0, 1, 0], 20.0).T + piv
    up, source = resolve_up(leaned[None])
    assert source == "axis"
    out = leaned @ de_tilt_matrix(up).T
    assert _torso_tilt(out) > 14.0


def test_arbitrary_world_frame_falls_back_to_estimation():
    """Up far from every axis (a diagonal frame) cannot be trusted as
    gravity; the old self-levelling remains the fallback."""
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

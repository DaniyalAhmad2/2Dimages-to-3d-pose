"""Sequence de-tilt: a consistent world-frame tilt is removed while the
subject's genuine per-frame lean is preserved."""
import numpy as np

from pose3d.core.skeleton import Joint
from pose3d.geometry.orient import _frame_up, de_tilt_matrix, sequence_up
from tests.synth import sample_skeleton_3d


def _rotX(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _updown_tilt(p):
    """Angle (deg) of the head->feet axis away from vertical."""
    u = _frame_up(p, ~np.isnan(p).any(1))
    return float(np.degrees(np.arccos(np.clip(u[2], -1, 1))))


def test_de_tilt_removes_constant_tilt_keeps_lean():
    base = sample_skeleton_3d()
    piv = base[int(Joint.PELVIS)].copy()

    # each frame: genuine per-frame lean (-4/0/+4°) then a constant 12° world
    # tilt, both about the pelvis
    def build(lean):
        p = (base - piv) @ _rotX(lean).T + piv        # genuine lean
        return (p - piv) @ _rotX(12).T + piv          # constant world tilt
    seq = np.stack([build(g) for g in (-4, 0, 4)])

    R = de_tilt_matrix(sequence_up(seq))
    tilts = [_updown_tilt(p @ R.T) for p in seq]
    # the no-lean frame becomes vertical (world tilt removed)...
    assert tilts[1] < 1.0
    # ...while the leaning frames keep their ~4° lean (not flattened)
    assert tilts[0] > 2.5 and tilts[2] > 2.5


def test_upright_sequence_stays_upright():
    p = sample_skeleton_3d()
    R = de_tilt_matrix(sequence_up(p[None]))
    assert _updown_tilt(p @ R.T) < 1.0

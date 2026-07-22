"""Phase 4 verification: bone-length fit + temporal smoothing."""
import numpy as np

from pose3d.core.skeleton import BONES, NUM_JOINTS
from pose3d.geometry.bonefit import (
    fit_bone_lengths, measure_bone_lengths, smooth_temporal,
)
from tests.synth import sample_skeleton_3d


def _bone_len(pose, a, b):
    return np.linalg.norm(pose[int(b)] - pose[int(a)])


def test_measure_bone_lengths():
    gt = sample_skeleton_3d()
    lengths = measure_bone_lengths(gt[None])
    for a, b in BONES:
        assert abs(lengths[(int(a), int(b))] - _bone_len(gt, a, b)) < 1e-9


def test_fit_enforces_bone_lengths():
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    rng = np.random.default_rng(1)
    noisy = gt + rng.normal(0, 0.03, gt.shape)   # 3 cm jitter
    fitted = fit_bone_lengths(noisy, target, bone_weight=20.0)
    for a, b in BONES:
        L = _bone_len(fitted, a, b)
        assert abs(L - target[(int(a), int(b))]) < 0.01, (a, b, L)


def test_fit_fills_occluded_joint():
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    raw = gt.copy()
    raw[6] = np.nan   # LEFT_WRIST occluded in 3D
    fitted = fit_bone_lengths(raw, target, fill_missing=True)
    assert not np.isnan(fitted[6]).any()   # placed, not dropped
    assert fitted.shape == (NUM_JOINTS, 3)


def test_fit_no_fill_leaves_missing_joint_nan():
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    raw = gt.copy()
    raw[6] = np.nan   # dropped observation
    fitted = fit_bone_lengths(raw, target, fill_missing=False)
    assert np.isnan(fitted[6]).all()       # not invented
    assert not np.isnan(fitted[5]).any()   # observed joints still fitted


def test_smoothing_reduces_variance():
    gt = sample_skeleton_3d()
    rng = np.random.default_rng(2)
    seq = np.stack([gt + rng.normal(0, 0.02, gt.shape) for _ in range(20)])
    sm = smooth_temporal(seq, alpha=0.4)
    raw_var = np.var(np.diff(seq, axis=0))
    sm_var = np.var(np.diff(sm, axis=0))
    assert sm_var < raw_var


def test_smoothing_nan_safe():
    gt = sample_skeleton_3d()
    seq = np.stack([gt, gt.copy(), gt.copy()])
    seq[1, 4] = np.nan
    sm = smooth_temporal(seq)
    assert sm.shape == seq.shape

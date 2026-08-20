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


# --- sparse frames --------------------------------------------------------
# Regression guards for "Import failed — Method 'lm' doesn't work when the
# number of residuals is less than the number of variables". The solver has
# NUM_JOINTS*3 = 45 variables and 3 residuals per OBSERVED joint plus one per
# bone, so Levenberg-Marquardt needs ~11 of the 15 joints. Frames sparser than
# that are routine (the detector is unreliable on the grey mannequin, and
# cross-view validation drops inconsistent observations) and used to abort the
# entire import.

def _keep_only(pose, keep):
    out = np.full_like(pose, np.nan)
    for j in keep:
        out[int(j)] = pose[int(j)]
    return out


def test_a_very_sparse_frame_still_fits():
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    keep = [0, 1, 4, 8, 11]                       # 5 joints: 29 residuals < 45
    raw = _keep_only(gt, keep)

    fitted = fit_bone_lengths(raw, target, fill_missing=False)

    for j in keep:                                # observed joints honoured
        assert np.linalg.norm(fitted[j] - gt[j]) < 0.05, j
    missing = [j for j in range(NUM_JOINTS) if j not in keep]
    assert np.isnan(fitted[missing]).all(), "un-observed joints were invented"


def test_either_side_of_the_lm_threshold_fits():
    """10 observed joints takes the trf path, 11 keeps the old lm path."""
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    for n in (10, 11):
        fitted = fit_bone_lengths(_keep_only(gt, range(n)), target,
                                  fill_missing=False)
        assert not np.isnan(fitted[:n]).any(), n


def test_an_empty_frame_invents_nothing():
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    empty = np.full((NUM_JOINTS, 3), np.nan)
    # even asked to fill, there is no observation to anchor a skeleton to
    assert np.isnan(fit_bone_lengths(empty, target, fill_missing=True)).all()
    assert np.isnan(fit_bone_lengths(empty, target, fill_missing=False)).all()


def test_one_bad_frame_does_not_lose_the_take():
    """fit_project must not let a single frame abort a whole import."""
    from pose3d.core.project import Frame, ProjectData
    from pose3d.pipeline import fit_project

    gt = sample_skeleton_3d()
    project = ProjectData(name="sparse")
    for i, pose in enumerate((gt, _keep_only(gt, [0, 1]), gt)):
        f = Frame(frame_id=f"{i:04d}")
        f.pose3d = pose.copy()
        project.frames.append(f)

    fit_project(project, smooth=False)

    assert all(f.fitted3d is not None for f in project.frames)
    for i in (0, 2):                              # the good frames still fit
        assert not np.isnan(project.frames[i].fitted3d).any()

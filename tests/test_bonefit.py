"""Phase 4 verification: the bone-length fit.

Temporal smoothing has its own suite (tests/test_smoothing.py): the only
guard it ever had here was that it reduced variance, which a lag does too,
and that is how a 21.8 %-of-height defect stayed green.
"""
import numpy as np

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint
from pose3d.geometry.bonefit import (
    fit_bone_lengths, measure_bone_lengths, solvable_joints,
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


def test_sparse_frame_uses_lm():
    """A frame with 5 joints missing must fit on the fast solver.

    Levenberg-Marquardt refuses an under-determined problem, and with all 15
    joints as variables a 10-joint frame is one. Freezing the joints no
    observation can pin removes their variables (and their bone residuals, so
    a frozen joint cannot drag an observed one), which puts the frame back on
    lm: 746.78 ms -> ~6 ms measured on the client's take.
    """
    import time

    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    raw = gt.copy()
    missing = [int(j) for j in (6, 7, 11, 13, 14)]      # wrists, knee, ankles
    raw[missing] = np.nan

    observed = ~np.isnan(raw).any(1)
    free = solvable_joints(observed)
    assert not free[missing].any(), "a dangling joint was left as a variable"
    n_res = 3 * int(observed.sum()) + sum(
        1 for a, b in BONES if free[int(a)] and free[int(b)])
    assert n_res >= 3 * int(free.sum()), "lm would still refuse this frame"

    t0 = time.perf_counter()
    fitted = fit_bone_lengths(raw, target, fill_missing=False)
    elapsed = (time.perf_counter() - t0) * 1000

    assert elapsed < 100, f"{elapsed:.1f} ms"           # ~6 ms measured
    for j in np.flatnonzero(observed):
        assert np.linalg.norm(fitted[j] - gt[j]) < 0.05, j
    assert np.isnan(fitted[missing]).all()


def test_a_joint_between_two_observed_ones_is_still_solved():
    """Freezing must not throw away a joint the skeleton genuinely pins:
    an unobserved elbow with a shoulder AND a wrist either side of it."""
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    raw = gt.copy()
    raw[int(Joint.LEFT_ELBOW)] = np.nan

    assert solvable_joints(~np.isnan(raw).any(1))[int(Joint.LEFT_ELBOW)]

    fitted = fit_bone_lengths(raw, target, fill_missing=True)
    assert np.linalg.norm(fitted[int(Joint.LEFT_ELBOW)]
                          - gt[int(Joint.LEFT_ELBOW)]) < 0.05


def test_a_frozen_joint_cannot_drag_the_observed_ones():
    """The hazard the old trf path lived with: an unobserved joint parked at
    the centroid pulls its observed neighbour through the shared bone term."""
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    full = fit_bone_lengths(gt, target, fill_missing=False)

    raw = gt.copy()
    raw[int(Joint.LEFT_ANKLE)] = np.nan            # dangling: frozen
    partial = fit_bone_lengths(raw, target, fill_missing=False)

    moved = np.linalg.norm(partial[int(Joint.LEFT_KNEE)]
                           - full[int(Joint.LEFT_KNEE)])
    assert moved < 1e-3, f"knee moved {1000 * moved:.2f} mm"

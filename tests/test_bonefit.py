"""Phase 4 verification: the bone-length fit.

Temporal smoothing has its own suite (tests/test_smoothing.py): the only
guard it ever had here was that it reduced variance, which a lag does too,
and that is how a 21.8 %-of-height defect stayed green.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d import pipeline
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


# --- fallback lengths are proportions of THIS subject (F18) ----------------

FIXTURE = Path(__file__).parent / "fixtures" / "client_take"


def test_the_fallback_table_is_a_proportion_not_a_body():
    """Ratios of the measured PELVIS-NECK spine, and the historical metres
    when there is nothing to scale to."""
    from pose3d.geometry.bonefit import fallback_bone_lengths

    plain = fallback_bone_lengths()
    assert plain[(int(Joint.PELVIS), int(Joint.NECK))] == 0.55   # as it was
    assert plain[(int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE))] == 0.43
    assert fallback_bone_lengths(0.55) == plain                  # the anchor

    # a 12 cm mannequin with a 41 mm spine gets a 41 mm spine, not 550 mm
    small = fallback_bone_lengths(0.0411)
    assert small[(int(Joint.PELVIS), int(Joint.NECK))] == pytest.approx(0.0411)
    for k, v in small.items():
        assert v == pytest.approx(plain[k] * 0.0411 / 0.55)
    # and a reference that is not one is ignored rather than exploded
    for bad in (None, 0.0, -1.0, float("nan")):
        assert fallback_bone_lengths(bad) == plain


def test_the_reference_survives_the_spine_itself_falling_back():
    """No spine, no problem: the largest measured bone divided by its own
    share of the spine.

    On the client take that lands 15.7 % above the directly measured spine,
    and it cannot do much better: the reference is recovered through an ADULT
    proportion table, and this subject is a stylised mannequin whose thigh is
    0.90 of its spine where the table says 0.78. Every measured bone implies a
    spine of 43.9-48.9 mm against a true 41.1 mm, so the error is the
    subject's proportions, not the choice of bone. 16 % beats the 1240 % that
    an unscaled table was out by.
    """
    from pose3d.core.io_project import load_project
    from pose3d.geometry.bonefit import reference_from_measured
    from pose3d.quality import load_rig

    project = load_project(FIXTURE)
    pipeline.triangulate_project(project, load_rig(FIXTURE / "calibration"))
    measured = measure_bone_lengths(
        np.stack([f.pose3d for f in project.frames]))
    direct = reference_from_measured(measured)
    assert direct == pytest.approx(
        measured[(int(Joint.PELVIS), int(Joint.NECK))])

    without_spine = dict(measured)
    without_spine[(int(Joint.PELVIS), int(Joint.NECK))] = 0.0
    derived = reference_from_measured(without_spine)
    assert abs(derived - direct) / direct < 0.20      # today 15.7 %
    # nothing measured at all: there is no subject to scale to
    assert reference_from_measured({k: 0.0 for k in measured}) is None


def test_fallback_lengths_scale_to_the_subject():
    """Black out one hip in one view for a WHOLE take and the joints that were
    still observed must barely move.

    The fallback residuals sit in the least-squares whether or not the joint
    they belong to is being solved for, so an unmeasurable bone drags every
    observed joint around it. Measured on the committed fixture (a 119 mm
    subject), blacking out the right hip in the left view for all 26 frames:
    the other joints moved 18.95 mm median / 190.73 mm max — 16 % and 160 % of
    body height — because the solver was pulling a 12 cm mannequin onto a
    1.75 m skeleton. Scaled to the subject's own 41 mm spine the same blackout
    costs 0.04 mm median, 1.74 mm p99, 6.99 mm max.
    """
    from pose3d.core.io_project import load_project
    from pose3d.core.project import CAM_LEFT
    from pose3d.geometry.bonefit import fallback_bone_lengths
    from pose3d.quality import load_rig

    rig = load_rig(FIXTURE / "calibration")

    def fit(blackout=None, scaled=True):
        project = load_project(FIXTURE)
        if blackout is not None:
            for f in project.frames:
                f.kp2d[CAM_LEFT][int(blackout)] = np.nan
                f.scores[CAM_LEFT][int(blackout)] = 0.0
        pipeline.triangulate_project(project, rig)
        targets, fell = pipeline.bone_length_targets(project)
        if not scaled:                     # the pre-6.3 absolute table
            plain = fallback_bone_lengths()
            measured = measure_bone_lengths(
                np.stack([f.pose3d for f in project.frames]))
            targets = {k: (v if v > 1e-6 else plain[k])
                       for k, v in measured.items()}
        pipeline.fit_project(project, bone_lengths=targets)
        return np.stack([f.fitted3d for f in project.frames]), fell

    clean, fell_clean = fit()
    assert fell_clean == []                # nothing falls back on a good take

    damaged, fell = fit(Joint.RIGHT_HIP)
    assert "PELVIS-RIGHT_HIP" in fell and "RIGHT_HIP-RIGHT_KNEE" in fell

    keep = [j for j in range(NUM_JOINTS) if j != int(Joint.RIGHT_HIP)]
    moved = np.linalg.norm(damaged[:, keep] - clean[:, keep], axis=2)
    moved = moved[np.isfinite(moved)]
    assert np.median(moved) < 0.0005, np.median(moved)          # today 0.04 mm
    assert np.percentile(moved, 99) < 0.0025, moved             # today 1.74 mm
    assert moved.max() < 0.008, moved.max()                     # today 6.99 mm

    # ...and the same blackout against the unscaled table, so this test cannot
    # pass by measuring nothing
    old, _ = fit(Joint.RIGHT_HIP, scaled=False)
    old_moved = np.linalg.norm(old[:, keep] - clean[:, keep], axis=2)
    old_moved = old_moved[np.isfinite(old_moved)]
    assert old_moved.max() > 0.10, old_moved.max()              # today 190.7 mm


def test_the_report_names_the_bones_that_fell_back():
    """"2 bones fell back" does not say whether it was the two collarbones or
    both thighs."""
    from pose3d.core.io_project import load_project
    from pose3d.core.project import CAM_LEFT
    from pose3d.quality import load_rig

    project = load_project(FIXTURE)
    for f in project.frames:
        f.kp2d[CAM_LEFT][int(Joint.RIGHT_HIP)] = np.nan
    pipeline.triangulate_project(project, load_rig(FIXTURE / "calibration"))
    report = pipeline.fit_project(project)

    assert report.fallback_bones == ["PELVIS-RIGHT_HIP", "RIGHT_HIP-RIGHT_KNEE"]
    note = report.note()
    assert "PELVIS-RIGHT_HIP" in note and "2 bone(s)" in note


def test_the_library_default_does_not_invent_joints():
    """`fill_missing` defaults to False: the safe behaviour is the one a
    caller gets by not thinking about it. `pipeline.fit_frame` — the app's
    only fit — always passed False; every other caller had to remember."""
    gt = sample_skeleton_3d()
    target = measure_bone_lengths(gt[None])
    raw = gt.copy()
    raw[6] = np.nan
    assert np.isnan(fit_bone_lengths(raw, target)[6]).all()

"""Temporal smoothing: no lag, off by default, and it invents nothing.

The shipped build ran a CAUSAL exponential moving average over every take by
default. On the client's 26 stop-motion frames that dragged the delivered pose
4.92 mm behind the keypoints (median; 25.65 mm max = 21.8 % of body height),
and the displacement pointed BACKWARD along the subject's motion in 95.4 % of
371 samples — median cos(displacement, motion) = -0.976. The filter had
exactly one test, that it reduced variance, which a lag does.
"""
import numpy as np

from pose3d.core.project import Frame, ProjectData
from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.geometry.bonefit import measure_bone_lengths, smooth_temporal
from pose3d.pipeline import fit_project
from tests.synth import rot_about, sample_skeleton_3d


def _moving_sequence(n=24, rot_per_frame=5.0, step=0.03):
    """A subject turning steadily while translating along the turn axis.

    Constant speed on purpose. Any low-pass filter also pulls a CURVED path
    toward its chord, and on a path whose speed varies that pull has a
    component along the direction of travel — which would show up in the lag
    metric below as a lag that is not one. Rotating about z while translating
    along z gives a constant-speed helix, whose curvature is purely radial, so
    what the metric sees is phase and nothing else.
    """
    gt = sample_skeleton_3d()
    centre = gt.mean(0)
    poses = []
    for t in range(n):
        R = rot_about((0, 0, 1), rot_per_frame * t)
        poses.append((gt - centre) @ R.T + centre
                     + np.array([0.0, 0.0, step * t]))
    return np.stack(poses)


def test_no_backward_lag():
    """The smoothed pose must not sit behind the subject's motion.

    Measured on the client's take with the causal EMA: median cos = -0.976,
    negative in 95.4 % of 371 samples — the correction points almost exactly
    opposite the direction of travel. On this synthetic sequence the same
    causal filter scores -0.9998 and the zero-phase one -0.020; -0.1 is the
    gate the audit set.
    """
    raw = _moving_sequence()
    sm = smooth_temporal(raw, alpha=0.6)

    motion = np.linalg.norm(np.diff(raw, axis=0), axis=2)
    floor = 0.02 * float(np.median(motion))   # ignore displacements at noise level

    cos = []
    for t in range(1, len(raw)):
        for j in range(NUM_JOINTS):
            disp = sm[t, j] - raw[t, j]
            step = raw[t, j] - raw[t - 1, j]
            nd, nm = np.linalg.norm(disp), np.linalg.norm(step)
            if nd < floor or nm < 1e-9:
                continue
            cos.append(float(np.dot(disp, step) / (nd * nm)))

    assert cos, "no usable samples"
    assert float(np.median(cos)) >= -0.1, f"median cos {np.median(cos)}"


def test_smoothing_is_off_by_default():
    """fit_project must leave each frame equal to its own fit.

    The default was already `smooth=False` before this change and it made no
    difference, because every caller passed smooth=True explicitly. This test
    exercises the default the callers now use.
    """
    gt = sample_skeleton_3d()
    project = ProjectData(name="two")
    for i, pose in enumerate((gt, gt + np.array([0.05, 0.0, 0.0]))):
        f = Frame(frame_id=f"{i:04d}")
        f.pose3d = pose.copy()
        project.frames.append(f)

    fit_project(project)

    from pose3d.pipeline import bone_length_targets, fit_frame
    targets, _ = bone_length_targets(project)
    own = fit_frame(project.frames[1].pose3d, targets)
    assert np.allclose(project.frames[1].fitted3d, own, atol=1e-9)
    # ... and frame 1 is NOT dragged toward frame 0
    moved = np.linalg.norm(project.frames[1].fitted3d - project.frames[0].fitted3d,
                           axis=1)
    assert moved.min() > 0.04, moved.min()


def test_nan_stays_nan():
    """A joint the fit refused to invent must not reappear.

    The deleted line `blend = np.where(np.isnan(cur), prev, blend)` carried the
    previous frame's value forward across a dropout — 2 joints on the client's
    take came back as exact copies of the frame before, 20.2 mm and 38.5 mm
    (17 % / 33 % of height) from where the joint actually next appeared.
    """
    gt = sample_skeleton_3d()
    seq = np.stack([gt, gt.copy(), gt.copy()])
    seq[1, int(Joint.LEFT_WRIST)] = np.nan

    sm = smooth_temporal(seq)

    assert np.isnan(sm[1, int(Joint.LEFT_WRIST)]).all()
    assert not np.isnan(sm[0]).any() and not np.isnan(sm[2]).any()


def test_smoothing_still_reduces_jitter():
    """Kept as a capability (client question 7 is unanswered), so it must
    still do the one thing it is for."""
    gt = sample_skeleton_3d()
    rng = np.random.default_rng(2)
    seq = np.stack([gt + rng.normal(0, 0.02, gt.shape) for _ in range(20)])
    sm = smooth_temporal(seq, alpha=0.4)
    assert np.var(np.diff(sm, axis=0)) < np.var(np.diff(seq, axis=0))


def test_alpha_one_is_a_true_no_op():
    """alpha=1.0 used not to be an off switch: the NaN branch was
    alpha-independent and fabricated joints regardless."""
    seq = _moving_sequence(6)
    seq[2, int(Joint.RIGHT_ANKLE)] = np.nan
    sm = smooth_temporal(seq, alpha=1.0)
    assert np.isnan(sm[2, int(Joint.RIGHT_ANKLE)]).all()
    finite = ~np.isnan(seq)
    assert np.allclose(sm[finite], seq[finite])


def test_bone_lengths_survive_smoothing_better_than_the_causal_filter():
    """The causal EMA re-broke the bones the fit had just enforced (L forearm
    19 % short on frame 0013). A zero-phase pass on a rigid sequence must not.
    """
    seq = _moving_sequence()            # rigid subject: every bone is constant
    sm = smooth_temporal(seq, alpha=0.6)
    before = measure_bone_lengths(seq)
    after = measure_bone_lengths(sm)
    for key, L in before.items():
        assert abs(after[key] - L) < 0.02 * L, key

"""Explicit, flagged gap fill — and the same flags reaching view and export.

The shipped build filled dropouts by accident: the causal smoother's NaN
branch carried the previous frame's value forward, so two joints on the
client's take (0012 RIGHT_KNEE, 0021 LEFT_ANKLE) came back as exact copies of
the frame before — 20.2 mm and 38.5 mm from where the joint actually next
appeared — and the reconstructed-joint count reported 15/15 for both frames.
"""
import numpy as np

from pose3d.core.project import Frame, ProjectData
from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.pipeline import fill_gaps, fit_project
from tests.synth import sample_skeleton_3d

WRIST = int(Joint.LEFT_WRIST)


def _sequence(n=6):
    """A project whose frame t has the subject shifted t * 0.1 m along x."""
    gt = sample_skeleton_3d()
    project = ProjectData(name="gaps")
    for t in range(n):
        f = Frame(frame_id=f"{t:04d}")
        f.pose3d = gt + np.array([0.1 * t, 0.0, 0.0])
        project.frames.append(f)
    return project


def test_one_frame_gap_is_filled_to_the_midpoint_and_flagged():
    project = _sequence()
    before, after = project.frames[1].pose3d[WRIST].copy(), \
        project.frames[3].pose3d[WRIST].copy()
    project.frames[2].pose3d[WRIST] = np.nan

    poses, n = fill_gaps(project)

    assert n == 1
    assert np.allclose(poses[2, WRIST], 0.5 * (before + after))
    assert project.frames[2].filled[WRIST]
    # nothing else is touched
    assert not project.frames[2].filled[int(Joint.HEAD)]
    assert not project.frames[1].filled.any()


def test_the_fill_never_writes_into_the_measurement():
    """`pose3d` is what the cameras gave us. An interpolated joint reaches the
    user through `fitted3d` (flagged) and nowhere else, so every "measured"
    number in pose3d.quality is computed on observations only."""
    project = _sequence()
    project.frames[2].pose3d[WRIST] = np.nan

    poses, n = fill_gaps(project)
    assert n == 1
    assert np.isnan(project.frames[2].pose3d[WRIST]).all()
    assert not np.isnan(poses[2, WRIST]).any()

    # and it stays NaN through the whole fit
    fit_project(project)
    assert np.isnan(project.frames[2].pose3d[WRIST]).all()
    assert not np.isnan(project.frames[2].fitted3d[WRIST]).any()
    assert project.frames[2].filled[WRIST]


def test_bone_targets_are_measured_without_the_invented_joints():
    """The fit's own targets are medians of MEASURED bone lengths: a filled
    joint is a midpoint of two frames, not an observation of a bone."""
    from pose3d.pipeline import bone_length_targets

    project = _sequence()
    clean, _ = bone_length_targets(project)
    project.frames[2].pose3d[WRIST] = np.nan
    fit_project(project)
    after, _ = bone_length_targets(project)

    assert after == clean         # unchanged: the fill never entered pose3d


def test_a_two_frame_gap_stays_a_hole():
    project = _sequence()
    for t in (2, 3):
        project.frames[t].pose3d[WRIST] = np.nan

    poses, n = fill_gaps(project, max_gap=1)
    assert n == 0

    for t in (2, 3):
        assert np.isnan(poses[t, WRIST]).all()
        assert not project.frames[t].filled[WRIST]


def test_a_gap_at_the_end_of_the_take_stays_a_hole():
    """Nothing on one side means nothing to interpolate between."""
    project = _sequence()
    project.frames[0].pose3d[WRIST] = np.nan
    project.frames[-1].pose3d[WRIST] = np.nan

    poses, n = fill_gaps(project)
    assert n == 0

    assert np.isnan(poses[0, WRIST]).all()
    assert np.isnan(poses[-1, WRIST]).all()


def test_fill_is_re_runnable():
    """Re-running must not lock a stale fill in: the flags always describe the
    2D as it is now."""
    project = _sequence()
    project.frames[2].pose3d[WRIST] = np.nan
    poses, n = fill_gaps(project)
    assert n == 1
    filled_value = poses[2, WRIST].copy()
    assert fill_gaps(project)[1] == 1                 # idempotent

    # the joint next door goes missing too: now it is a wider hole
    project.frames[3].pose3d[WRIST] = np.nan
    poses, n = fill_gaps(project)
    assert n == 0
    assert np.isnan(poses[2, WRIST]).all()
    assert not project.frames[2].filled[WRIST]
    assert not np.isnan(filled_value).any()           # it really had been filled


def test_a_recovered_joint_is_a_measurement_again():
    """A re-triangulation that finds the joint the fill had invented must give
    the observation back and clear the flag — the flags can never be stale,
    because they are derived from the raw 3D every time."""
    project = _sequence()
    truth = project.frames[2].pose3d[WRIST].copy()
    project.frames[2].pose3d[WRIST] = np.nan
    fit_project(project)
    assert project.frames[2].filled[WRIST]

    # the joint comes back (a hand correction, a re-detect, a re-triangulation)
    project.frames[2].pose3d[WRIST] = truth
    fit_project(project)

    assert not project.frames[2].filled[WRIST]
    assert np.allclose(project.frames[2].pose3d[WRIST], truth)


def test_the_fit_treats_a_filled_joint_as_observed():
    project = _sequence()
    project.frames[2].pose3d[WRIST] = np.nan

    report = fit_project(project)

    assert report.gaps_filled == 1
    assert "interpolated" in report.note()
    assert not np.isnan(project.frames[2].fitted3d[WRIST]).any()
    assert not np.isnan(project.frames[2].fitted3d).any()


def test_a_longer_gap_is_absent_from_the_fitted_pose():
    project = _sequence()
    for t in (2, 3):
        project.frames[t].pose3d[WRIST] = np.nan

    report = fit_project(project)

    assert report.gaps_filled == 0
    for t in (2, 3):
        assert np.isnan(project.frames[t].fitted3d[WRIST]).all()


# --- the same array reaches the 3D view and the export --------------------

class _FakeItem:
    """Stands in for a pyqtgraph GL item: records what it was asked to draw."""

    def __init__(self):
        self.kwargs = {}

    def setData(self, **kwargs):
        self.kwargs.update(kwargs)


def test_the_view_draws_a_filled_joint_in_the_filled_colour():
    from pose3d.ui.view3d import View3D

    pts = sample_skeleton_3d()
    valid = np.ones(NUM_JOINTS, bool)
    filled = np.zeros(NUM_JOINTS, bool)
    filled[WRIST] = True
    scatter, lines = _FakeItem(), _FakeItem()

    View3D._draw_skeleton(scatter, lines, pts, valid, View3D.JOINT_COLOR, filled)

    colors = scatter.kwargs["color"]
    assert np.allclose(colors[WRIST], View3D.FILLED_COLOR)
    assert np.allclose(colors[int(Joint.HEAD)], View3D.JOINT_COLOR)


def test_view_and_export_agree_on_which_joints_are_posed():
    """The export reads the SAME flags the 3D view draws from, so a one-frame
    fill is posed in both and a longer hole is absent from both."""
    from pose3d.export.blender_export import _character_bone_frames

    project = _sequence(n=7)
    project.frames[2].pose3d[WRIST] = np.nan          # 1-frame gap  -> filled
    project.frames[4].pose3d[WRIST] = np.nan          # 2-frame gap  -> hole
    project.frames[5].pose3d[WRIST] = np.nan
    fit_project(project)

    poses = np.stack([f.fitted3d for f in project.frames])
    filled = np.stack([f.filled for f in project.frames])

    # what the 3D view considers present
    view_mask = ~np.isnan(poses).any(2)
    assert view_mask[2, WRIST] and filled[2, WRIST]
    assert not view_mask[4, WRIST] and not filled[4, WRIST]

    frames, _ = _character_bone_frames(poses, 0, None, filled=filled)
    if frames is None:
        import pytest
        pytest.skip("bundled character asset unavailable")
    # the export posed the frame whose joint was filled, and the one whose
    # joint is genuinely missing is posed from the rest of the skeleton
    assert frames[2] is not None and frames[4] is not None


def test_the_measured_overlay_omits_a_filled_joint():
    """The reference overlay is "what the cameras measured", so a joint that
    was interpolated must not appear in it at all."""
    from pose3d.ui.view3d import View3D

    pts = sample_skeleton_3d()
    valid = np.ones(NUM_JOINTS, bool)
    filled = np.zeros(NUM_JOINTS, bool)
    filled[WRIST] = True
    scatter, lines = _FakeItem(), _FakeItem()

    measured = valid & ~filled
    View3D._draw_skeleton(scatter, lines, pts, measured, View3D.CAPTURE_COLOR)

    assert len(scatter.kwargs["pos"]) == NUM_JOINTS - 1
    assert not any(np.allclose(p, pts[WRIST]) for p in scatter.kwargs["pos"])


# --- the shipped gate for the fill itself ---------------------------------
#
# The gate the plan first wrote ("the jump at a filled joint stays under 2 %
# of body height") is not reachable by ANY interpolation on this take: the
# subject's own median inter-frame motion is 7.9 % of height, so 2 % is below
# the take's noise floor. The controller restated it as: a filled joint is
# FLAGGED, and its error is BELOW the hold-previous error of the behaviour it
# replaces (the old smoother's NaN branch carried the previous frame forward).
# That is what this measures, on the client's own take, where the truth is
# known: every joint-frame with both neighbours present is filled both ways
# and compared with the observation that is actually there.

def _fill_errors_pct_of_height(poses, height):
    """(midpoint error, hold-previous error) in % of body height, over every
    joint-frame whose neighbours and whose own observation are all present."""
    mid, hold = [], []
    for t in range(1, len(poses) - 1):
        for j in range(NUM_JOINTS):
            a, truth, b = poses[t - 1, j], poses[t, j], poses[t + 1, j]
            if np.isnan([a, truth, b]).any():
                continue
            mid.append(np.linalg.norm(0.5 * (a + b) - truth))
            hold.append(np.linalg.norm(a - truth))
    return (100.0 * np.array(mid) / height, 100.0 * np.array(hold) / height)


def test_a_filled_joint_is_flagged_and_beats_holding_the_previous_frame():
    from pathlib import Path

    from pose3d.core.io_project import load_project
    from pose3d.quality import subject_height

    project = load_project(Path(__file__).parent / "fixtures" / "client_take")
    raw = np.stack([np.asarray(f.pose3d, float) for f in project.frames])
    height = subject_height(
        np.stack([np.asarray(f.fitted3d, float) for f in project.frames]))

    poses, n = fill_gaps(project)

    # every value the fill wrote is flagged, and every flag has a value
    assert n == 2                                    # 0012 R_KNEE, 0021 L_ANKLE
    for t, (f, before) in enumerate(zip(project.frames, raw)):
        written = np.isnan(before).any(1) & ~np.isnan(poses[t]).any(1)
        assert np.array_equal(written, np.asarray(f.filled, bool))
        assert np.isfinite(poses[t][f.filled]).all()
        # and the measurement is untouched
        assert np.array_equal(np.isnan(f.pose3d), np.isnan(before))

    # measured on the take as it was DETECTED (before the fill), so every
    # sample has an observation to be right or wrong about
    mid, hold = _fill_errors_pct_of_height(raw, height)
    assert mid.size == 354
    # today: midpoint 4.40 median / 13.29 p90, hold-previous 8.04 / 23.19
    assert np.median(mid) < np.median(hold)
    assert np.percentile(mid, 90) < np.percentile(hold, 90)
    assert np.median(mid) <= 5.1, f"{np.median(mid):.2f} % of height"

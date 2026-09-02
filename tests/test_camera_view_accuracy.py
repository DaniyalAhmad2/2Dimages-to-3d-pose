"""Accuracy lives on the keypoints: colour bands + hover tooltips.

This replaced the SELECTED JOINT panel — the joint's name and quality appear
where the user is already looking, on the dot they are about to drag.

Every residual here is NORMALISED by that camera's figure height
(`pose3d.quality.figure_height_px`), because a pixel is not a unit anyone can
compare: the client's subject stands 863 px tall in the left image and 446 px
in the right, so the same error read 1.9x worse on the right, and the old
`100*exp(-err_px/6)` map needed under 1 px on a 3072x4080 frame to call
anything green — which is how a perfectly ordinary take came out 70.6 % red
with a timeline of 21 red frames out of 26 and no green one at all.
"""
import copy
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pose3d import pipeline                                          # noqa: E402
from pose3d.core.io_project import load_project                      # noqa: E402
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS         # noqa: E402
from pose3d.core.skeleton import JOINT_NAMES, NUM_JOINTS             # noqa: E402
from pose3d.quality import load_rig                                  # noqa: E402
from pose3d.ui.camera_view import RAG_COLORS, CameraPanel            # noqa: E402
from pose3d.ui.model import (                                        # noqa: E402
    STATE_NOT_MEASURED, STATE_REJECTED, ProjectModel, frame_stat,
    worst_per_joint)
from pose3d.ui.panels import (                                       # noqa: E402
    ACC_AMBER_FRAC, ACC_GREEN_FRAC, acc_band, accuracy_pct)

FIXTURE = Path(__file__).parent / "fixtures" / "client_take"


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


# --- the mapping itself ----------------------------------------------------

def test_the_bands_are_fractions_of_figure_height():
    """The anchors, stated as the thing they are anchored to."""
    assert accuracy_pct(0.0) == 100.0
    assert accuracy_pct(ACC_GREEN_FRAC) == pytest.approx(85.0)
    assert accuracy_pct(ACC_AMBER_FRAC) == pytest.approx(70.0)
    assert accuracy_pct(0.030) == pytest.approx(0.0)
    # and the px form: 5 px on an 863 px figure is the same number as 0.00579
    assert accuracy_pct(5.0, 863.0) == pytest.approx(accuracy_pct(5.0 / 863.0))
    assert np.isnan(accuracy_pct(np.nan))


def test_a_joint_with_no_3d_is_not_banded_green_by_its_confidence(qapp):
    """The detector-confidence fallback is gone.

    It fired 0 times in 390 joint-frames of the client's take where an
    accuracy existed, and where one did NOT it painted a joint with no
    reconstruction at all in the same green as a good one — 90 % confident
    that a point the pipeline never triangulated was fine.
    """
    p = _panel(qapp)
    p.view.set_pose(
        np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5,
        np.full(NUM_JOINTS, 0.9))
    p.set_accuracy(np.full(NUM_JOINTS, np.nan))

    from PySide6.QtCore import Qt
    dot = p.view._joints[5]
    assert dot.brush().style() == Qt.BrushStyle.NoBrush
    assert dot.pen().color() == RAG_COLORS["unmeasured"]
    assert "not measured" in dot.toolTip()
    assert JOINT_NAMES[5] in dot.toolTip()
    # the confidence is still REPORTED, it just no longer decides a colour
    assert "confidence" in dot.toolTip().lower()


def test_the_three_no_number_states_are_drawn_apart(qapp):
    """"not measured", "interpolated" and "rejected by the cross-view check"
    are three different facts and used to be one colour between them."""
    from PySide6.QtCore import Qt
    p = _panel(qapp)
    filled = np.zeros(NUM_JOINTS, bool); filled[3] = True
    p.view.set_pose(
        np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5,
        np.full(NUM_JOINTS, 0.9), filled=filled)
    states = ["ok"] * NUM_JOINTS
    states[6] = STATE_NOT_MEASURED
    states[7] = STATE_REJECTED
    errs = np.full(NUM_JOINTS, 0.001)          # everything else is excellent
    p.set_accuracy(errs, states=states)

    hollow = {j: p.view._joints[j] for j in (3, 6, 7)}
    for j, item in hollow.items():
        assert item.brush().style() == Qt.BrushStyle.NoBrush, j
    assert hollow[3].pen().color() == RAG_COLORS["filled"]
    assert hollow[6].pen().color() == RAG_COLORS["unmeasured"]
    assert hollow[7].pen().color() == RAG_COLORS["rejected"]
    assert "interpolated" in hollow[3].toolTip()
    assert "not measured" in hollow[6].toolTip()
    assert "cross-view check" in hollow[7].toolTip()
    # a measured joint still gets a solid dot
    assert p.view._joints[4].brush().style() != Qt.BrushStyle.NoBrush
    assert p.view._joints[4].brush().color() == RAG_COLORS["green"]


def test_dots_band_by_accuracy_and_tooltips_name_the_joint(qapp):
    p = _panel(qapp)
    errs = np.full(NUM_JOINTS, 0.001)    # 0.1 % of figure height: High
    errs[3] = 0.025                      # ...except one terrible joint
    p.set_accuracy(errs)

    good, bad = p.view._joints[2], p.view._joints[3]
    assert good.brush().color() == RAG_COLORS["green"]
    assert bad.brush().color() == RAG_COLORS["red"]

    # hover text: the joint's name, its accuracy figure, and the RAW
    # normalised number the band came from — the banding is a judgement call
    # and the client has to be able to see through it
    assert JOINT_NAMES[3] in bad.toolTip()
    assert "accuracy" in bad.toolTip().lower()
    assert "2.50%" in bad.toolTip()
    assert "figure" in bad.toolTip()


def test_the_tooltip_reports_the_delivered_pose_too(qapp):
    """The measured/delivered gap is the permanent regression detector, and
    the joint that pays it is where it should be readable."""
    p = _panel(qapp)
    p.set_accuracy(np.full(NUM_JOINTS, 0.002),
                   delivered=np.full(NUM_JOINTS, 0.009))
    assert "pose shown" in p.view._joints[1].toolTip()
    assert "0.90%" in p.view._joints[1].toolTip()


def test_corrected_joints_stay_purple_and_say_so(qapp):
    p = _panel(qapp)
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    corrected = np.zeros(NUM_JOINTS, bool)
    corrected[4] = True
    p.view.set_pose(xy, np.full(NUM_JOINTS, 0.9), corrected)
    p.set_accuracy(np.full(NUM_JOINTS, 0.002))

    assert p.view._joints[4].brush().color() == RAG_COLORS["corrected"]
    assert "corrected" in p.view._joints[4].toolTip().lower()


def test_accuracy_then_pose_order_does_not_matter(qapp):
    """The two updates arrive from different signals; whichever lands last
    must not erase the other's contribution."""
    p = _panel(qapp)
    p.set_accuracy(np.full(NUM_JOINTS, 0.025))       # accuracy first...
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    p.view.set_pose(xy, np.full(NUM_JOINTS, 0.9))    # ...then the pose
    assert p.view._joints[0].brush().color() == RAG_COLORS["red"]


# --- on the client's own take ----------------------------------------------

def _delivered_model():
    """The client's take, reconstructed by the path the app itself runs."""
    p = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    pipeline.triangulate_project(p, rig)
    pipeline.fit_project(p)
    return ProjectModel(p, rig)


def _band_counts(model, stage="measured"):
    counts = {"green": 0, "amber": 0, "red": 0}
    for i in range(len(model.project.frames)):
        errs = model._accuracy(i)
        for cam in CAMERAS:
            for v in errs[cam][stage]:
                if np.isfinite(v):
                    counts[acc_band(accuracy_pct(v))] += 1
    return counts


def _timeline_bands(model):
    counts = {"green": 0, "amber": 0, "red": 0}
    for i in range(len(model.project.frames)):
        frac = frame_stat(
            worst_per_joint(model._accuracy(i), "measured"))
        counts["red" if not np.isfinite(frac) else
               "green" if frac < ACC_GREEN_FRAC else
               "amber" if frac < ACC_AMBER_FRAC else "red"] += 1
    return counts


def test_a_good_take_is_not_majority_red():
    """The client's own take, banded. It is not a perfect take and it should
    not read green — but under the old map every dot and every frame of it was
    red, which told the user precisely nothing.

    Measured on the fixture today: 30.7 % of joint-views red, where the old
    `100*exp(-err_px/6)` map made it 70.6 %; and 24 of 26 frames non-red (2
    green, 22 amber), where the old literal 5 px / 12 px timeline cuts left
    21 of 26 red and not one green.
    """
    model = _delivered_model()
    counts = _band_counts(model, "measured")
    total = sum(counts.values())
    assert total > 700, "the fixture is not the 26-frame client take"
    red = counts["red"] / total
    assert red <= 0.35, f"{100 * red:.1f} % of joint-views are red"  # today 30.7

    tl = _timeline_bands(model)
    assert tl["green"] + tl["amber"] >= 20, tl                       # today 24


def test_the_delivered_pose_is_scored_separately_from_the_measurement():
    """`_accuracy` returns both stages, per camera, never averaged. Their
    ratio is the regression detector every later phase is judged on — the
    shipped build put the delivered pose 4.2x/6.2x further from the keypoints
    than the measurement and no readout in the app could say so."""
    model = _delivered_model()
    errs = model._accuracy(0)
    assert set(errs) == set(CAMERAS)
    for cam in CAMERAS:
        assert set(errs[cam]) == {"measured", "delivered"}
        for stage in ("measured", "delivered"):
            assert errs[cam][stage].shape == (NUM_JOINTS,)
    # and the two cameras are genuinely separate numbers
    assert not np.allclose(errs[CAM_LEFT]["measured"],
                           errs[CAM_RIGHT]["measured"], equal_nan=True)


def test_bands_are_resolution_invariant():
    """Scale K and every 2D point by the same factor — a similarity that
    leaves the 3D pose and the physical error identical — and nothing about
    the banding may move.

    In raw pixels it moved a great deal: 0.265x (the client's right camera
    against the left) took the red fraction from 52.3 % to 84.0 % and the
    green frame count from 0 to 21, purely because the subject was smaller in
    the image.
    """
    s = 0.265
    base = _delivered_model()
    small = copy.deepcopy(base.project)
    rig = copy.deepcopy(base.rig)
    for cam in CAMERAS:
        intr = rig.intr[cam]
        K = np.asarray(intr.K, float).copy()
        K[:2, :] *= s
        intr.K = K
        w, h = intr.image_size
        intr.image_size = (int(round(w * s)), int(round(h * s)))
    for f in small.frames:
        for cam in CAMERAS:
            f.kp2d[cam] = f.kp2d[cam] * s
    shrunk = ProjectModel(small, rig)

    # the 3D is untouched by construction
    assert np.allclose(np.stack([f.fitted3d for f in small.frames]),
                       np.stack([f.fitted3d for f in base.project.frames]),
                       equal_nan=True)
    for i in range(len(small.frames)):
        a, b = base._accuracy(i), shrunk._accuracy(i)
        for cam in CAMERAS:
            for stage in ("measured", "delivered"):
                assert np.allclose(a[cam][stage], b[cam][stage],
                                   rtol=1e-6, atol=1e-9, equal_nan=True), (
                    f"frame {i} {cam} {stage} moved when only the image "
                    f"resolution changed")
    assert _band_counts(shrunk) == _band_counts(base)
    assert _timeline_bands(shrunk) == _timeline_bands(base)


def test_the_readout_sees_a_lagged_pose():
    """The defect the client complained about, injected: deliver each frame
    the PREVIOUS frame's pose.

    Under the shipped readout this was bit-identical — it scored `pose3d`
    only, so whatever happened to the delivered pose afterwards was invisible.
    """
    model = _delivered_model()
    good = [model._accuracy(i) for i in range(len(model.project.frames))]

    lagged = copy.deepcopy(model.project)
    for f, prev in zip(lagged.frames[1:], model.project.frames[:-1]):
        f.fitted3d = np.asarray(prev.fitted3d, float).copy()
    lag_model = ProjectModel(lagged, model.rig)
    bad = [lag_model._accuracy(i) for i in range(len(lagged.frames))]

    def median(rows, stage):
        v = np.concatenate([r[cam][stage] for r in rows for cam in CAMERAS])
        return float(np.nanmedian(v))

    # the measurement is untouched — only what was DELIVERED changed
    assert median(bad, "measured") == pytest.approx(median(good, "measured"))
    ratio = median(bad, "delivered") / median(good, "delivered")
    assert ratio > 2.0, f"a one-frame lag only cost {ratio:.2f}x"   # today 6.2x


def test_a_rejected_joint_names_the_number_it_was_rejected_by(qapp):
    """A purple dot alone is a new kind of silence.

    "The two views disagree" does not tell the user whether to drag the point
    or recalibrate the rig; "by 84 px, gate 29 px" does. The numbers ride on
    the state itself (`ui.model.RejectedState`), which still IS the string
    `STATE_REJECTED`, so nothing between the model and the dot had to grow a
    parameter to carry them.
    """
    from pose3d.ui.model import RejectedState

    p = _panel(qapp)
    states = ["ok"] * NUM_JOINTS
    states[7] = RejectedState(84.2, 29.4)
    p.set_accuracy(np.full(NUM_JOINTS, 0.001), states=states)

    tip = p.view._joints[7].toolTip()
    assert "84 px" in tip and "gate 29 px" in tip
    assert "cross-view" in tip
    assert p.view._joints[7].pen().color() == RAG_COLORS["rejected"]
    # the state is still the state: a plain string still renders, just without
    # the numbers
    assert states[7] == STATE_REJECTED
    states[7] = STATE_REJECTED
    p.set_accuracy(np.full(NUM_JOINTS, 0.001), states=states)
    assert "px" not in p.view._joints[7].toolTip().split("confidence")[0]


def test_the_gate_the_dot_quotes_is_the_gate_the_gate_used(qapp):
    """One number, or the tooltip is fiction: the model's cached gate must be
    the threshold `validate_cross_view` was run with."""
    project = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    m = ProjectModel(project, rig)
    assert m.epipolar_gate() == pytest.approx(
        pipeline.epipolar_threshold(rig, project))


def _hallucinated_ankle(project):
    """The gate's own example: the LEFT view's ankle detected on the knee.

    Written straight into `kp2d` with a low score, which is what the detector
    does — not through `set_joint_2d`, which would flag it `corrected` and
    make the gate protect it.
    """
    from pose3d.core.skeleton import Joint
    j = int(Joint.LEFT_ANKLE)
    f = project.frames[0]
    f.kp2d[CAM_LEFT][j] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][j] = 0.3
    return j


def test_the_live_re_solve_agrees_with_the_batch_path():
    """A drag must not smuggle a rejected observation into the 3D view.

    The gate masks the LEFT ankle; dragging the RIGHT view of the same joint
    used to triangulate `f.kp2d` directly — the still-rejected left point
    included — so the joint came back with a 3D position that reached the
    view and the export while the tooltip went on saying "it was not
    triangulated", and the next recompute deleted it again. Live and batch
    now run the same verdict (`pipeline.revalidate_joint` -> `gated_kp2d`)
    over the same 2D, which the project's constraints require of every manual
    correction.
    """
    project = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    j = _hallucinated_ankle(project)
    m = ProjectModel(project, rig)
    m.recompute_all()
    f = m.frame()
    assert f.rejected[CAM_LEFT][j], "the hallucination must be rejected first"
    assert np.isnan(f.pose3d[j]).all()

    # drag the OTHER view of the same joint by a couple of px
    x, y = f.kp2d[CAM_RIGHT][j]
    m.set_joint_2d(CAM_RIGHT, j, float(x) + 2.0, float(y) + 2.0)
    live = np.array(f.pose3d, copy=True)
    states = m.joint_states(0)

    m.recompute_all()                     # the batch path over the same 2D
    assert np.array_equal(live, f.pose3d, equal_nan=True)
    assert np.isnan(live[j]).all(), "a rejected pair triangulated anyway"
    # ...and what the dot says matches what the arithmetic did
    assert states[CAM_LEFT][j] == STATE_REJECTED


def test_a_drag_does_not_split_the_gate_in_two():
    """The tooltip's gate and the sidebar's gate are the same number.

    `set_joint_2d` drops the take-wide quality (the sidebar recomputes it) and
    deliberately keeps the cached gate, on the argument that one dragged point
    cannot move a median over every pair in the take. This is that argument,
    measured — if it ever stops holding, the two readouts start disagreeing
    about the threshold a purple dot was judged against.
    """
    project = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    m = ProjectModel(project, rig)
    m.recompute_all()
    before = m.epipolar_gate()

    f = m.frame()
    j = int(np.flatnonzero(np.isfinite(f.kp2d[CAM_RIGHT]).all(1))[0])
    x, y = f.kp2d[CAM_RIGHT][j]
    m.set_joint_2d(CAM_RIGHT, j, float(x) + 3.0, float(y) + 3.0)

    assert m.epipolar_gate() == before             # cached, deliberately
    sidebar = m.quality().epipolar["threshold_px"]
    assert sidebar == pytest.approx(before, abs=0.5), (sidebar, before)


def test_an_unflagged_per_image_failure_still_reads_as_rejected():
    """`joint_states` runs the WHOLE gate, not the Sampson half of it.

    Between a hand edit and the next recompute the mask is stale for whatever
    the edit touched, so the state is inferred from the 2D — and the inference
    used to test only the Sampson distance. Sampson is below BOTH per-image
    distances by construction, so once the gate is at its ceiling (1.4 % of
    the smaller image's diagonal, 35.84 px here) the per-image half is the
    only one that can fire: this pair sits 36.00 px off its epipolar line in
    the right image, 0.16 px over that image's own allowance, at a Sampson
    distance of 31.4 px the ceiling waves through. It used to read as OK on a
    joint the next recompute refuses.

    The take is the fixture with every right-view point pushed 8 px off, which
    is what puts the gate on its ceiling (median 4.91 -> 5.97+ px).
    """
    from pose3d.geometry.triangulate import fundamental_matrix
    from pose3d.pipeline import per_image_allowances, point_line_distances

    project = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    for f in project.frames:
        f.kp2d[CAM_RIGHT][:, 1] += 8.0
    m = ProjectModel(project, rig)
    m.recompute_all()
    f = m.frame()
    thr, allow = m.epipolar_gate(), per_image_allowances(rig)
    assert thr == pytest.approx(allow[CAM_RIGHT]), "the gate must be on its cap"
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])

    j = int(np.flatnonzero(np.isfinite(f.pose3d).all(1))[0])
    base = f.kp2d[CAM_RIGHT][j].copy()
    for step in np.arange(1.0, 400.0, 1.0):
        f.kp2d[CAM_RIGHT][j] = base + (0.0, step)
        e = pipeline.epipolar_distance(
            f.kp2d[CAM_LEFT][j], f.kp2d[CAM_RIGHT][j],
            rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
            rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
        _, d_r = point_line_distances(f.kp2d[CAM_LEFT][j],
                                      f.kp2d[CAM_RIGHT][j], F)
        if d_r > allow[CAM_RIGHT]:
            break
    else:                                            # pragma: no cover
        pytest.fail("could not push the point past its own image's allowance")
    assert e <= thr, (e, thr)          # the Sampson half does NOT fire: 31.4 px

    assert not f.rejected[CAM_LEFT][j] and not f.rejected[CAM_RIGHT][j]
    assert m.joint_states(0)[CAM_RIGHT][j] == STATE_REJECTED


def test_a_joint_corrected_in_both_views_is_not_painted_rejected():
    """The dot must not call a joint rejected that the gate deliberately kept.

    `_judge`'s last rule is that a pair whose BOTH views were placed by hand
    is kept however far apart they look — the user has overruled the gate, and
    the joint is triangulated, fitted and exported. `joint_states` inferred
    its verdict from `cross_view_verdict` alone, which is only the first half
    of that rule, so exactly those joints came back purple with a tooltip
    saying they "were not triangulated" — permanently, on the points the user
    had already fixed by hand, while their 3D was in the export. The
    inference now asks the question the mask answers
    (`pipeline.cross_view_rejection`: which view WOULD be flagged), so the two
    cannot disagree about a joint they are both looking at.
    """
    from pose3d.core.skeleton import Joint

    project = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    m = ProjectModel(project, rig)
    m.recompute_all()
    f = m.frame()

    # hand-place BOTH views of the ankle, the left one onto the knee — a
    # disagreement far past any gate, made of two corrected points.
    j = int(Joint.LEFT_ANKLE)
    knee = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    m.set_joint_2d(CAM_LEFT, j, float(knee[0]), float(knee[1]))
    rx, ry = f.kp2d[CAM_RIGHT][j]
    m.set_joint_2d(CAM_RIGHT, j, float(rx), float(ry))
    assert f.corrected[CAM_LEFT][j] and f.corrected[CAM_RIGHT][j]

    m.recompute_all()
    assert not f.rejected[CAM_LEFT][j] and not f.rejected[CAM_RIGHT][j], (
        "the gate keeps a pair corrected in both views")
    assert np.isfinite(f.pose3d[j]).all(), "...and triangulates it"

    states = m.joint_states(0)
    for c in CAMERAS:
        assert states[c][j] != STATE_REJECTED, (
            "a joint that IS triangulated cannot be painted 'not triangulated'")
        assert states[c][j] == "ok"


def _headline_color(tip: str) -> str:
    """The colour of the tooltip's FIRST line — the one under the joint name."""
    import re
    m = re.search(r"<span style='color:(#[0-9a-fA-F]{6});'>", tip)
    assert m, tip
    return m.group(1)


def test_the_tooltip_headline_is_coloured_by_what_the_joint_ENDS_UP_being(qapp):
    """The dot and its tooltip must agree about what the joint is.

    The headline was coloured with the accuracy band before the state was
    overridden to "filled" or "corrected", so an interpolated joint drew an
    amber dot over a tooltip whose first line was still green.
    """
    p = _panel(qapp)
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    filled = np.zeros(NUM_JOINTS, bool); filled[3] = True
    corrected = np.zeros(NUM_JOINTS, bool); corrected[4] = True
    p.view.set_pose(xy, np.full(NUM_JOINTS, 0.9), corrected, filled=filled)
    p.set_accuracy(np.full(NUM_JOINTS, 0.001))       # everything else: green

    assert _headline_color(p.view._joints[3].toolTip()) \
        == RAG_COLORS["filled"].name()
    assert _headline_color(p.view._joints[4].toolTip()) \
        == RAG_COLORS["corrected"].name()
    # ...and an ordinary joint still gets its accuracy band
    assert _headline_color(p.view._joints[5].toolTip()) \
        == RAG_COLORS["green"].name()
    # every line the tooltip used to carry is still there
    assert "interpolated" in p.view._joints[3].toolTip()
    assert "corrected by hand" in p.view._joints[4].toolTip()
    assert "confidence" in p.view._joints[5].toolTip().lower()

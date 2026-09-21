"""The readouts that see what reprojection cannot.

Reprojection error is weak by construction: a freshly triangulated point
always reprojects near its own two observations, whatever the rig did to get
it there. These three rows are in the sidebar because between them they cover
that blind spot, and because they cover each other's:

  * bone-length spread is ~7x more responsive than the gauge to a wrong camera
    pose, and almost blind to a focal length shared by both cameras;
  * epipolar disagreement moves on exactly that shared focal;
  * L/R symmetry sees a keypoint the detector has confused, which neither of
    the other two can distinguish from a bad camera.

Numbers are `docs/audit-2026-09/wf_baseline.md`; the metric definitions are
`pose3d.quality`, ported verbatim from the audit harness.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from pose3d import pipeline
from pose3d.core.io_project import load_project
from pose3d.core.project import CAMERAS
from pose3d.core.skeleton import JOINT_NAMES, Joint
from pose3d.quality import (
    SYMMETRY_FLAG_PCT, load_rig, symmetry_notes, take_quality,
)

FIXTURE = Path(__file__).parent / "fixtures" / "client_take"


def _rotation(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _measure(mutate=None):
    """Reconstruct the client take with an optionally damaged rig, and measure.

    Cross-view validation is turned OFF for the damaged rigs on purpose: it
    would silently throw away the very observations the damage disagrees
    about, and the question here is what the READOUT says, not what the gate
    salvages.
    """
    project = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    validate = mutate is None
    if mutate is not None:
        mutate(rig)
    pipeline.triangulate_project(project, rig, validate=validate)
    pipeline.fit_project(project)
    return take_quality(project, rig)


@pytest.fixture(scope="module")
def baseline():
    return _measure()


def test_the_baseline_is_the_client_take(baseline):
    """What the two damage tests are measured against."""
    assert baseline.n_frames == 26
    # re-centred on the take as the app now reconstructs it: Halpe-26
    # detection (Phase 5b) through the data-driven cross-view gate (Phase 6.2),
    # which refuses two of its pairs. Bands are +-11 % and +-13 % of those.
    assert baseline.bone_cv["median_cv_pct"] == pytest.approx(5.23, abs=0.6)
    assert baseline.epipolar["median_px"] == pytest.approx(4.67, abs=0.6)


def test_bone_cv_responds_to_a_wrong_rig(baseline):
    """Rotate the right camera by 15 deg — 19.13 mm of pose error, 16.2 % of
    the subject's height — and the bone-length spread must say so.

    This is the acceptance test for the whole phase: that error costs the
    reprojection gauge 2.9 points and never changes its band, so if nothing on
    screen moved for it, nothing on screen describes the reconstruction.
    Today: 5.23 % -> 6.66 %, a 27 % rise.

    It read 5.44 % -> 8.18 % (a 50 % rise) while the fixture was detected with
    COCO-17. Phase 5b re-detected it with Halpe-26: the INTACT take is
    unchanged to two figures, while the DAMAGED one is markedly steadier
    (8.18 % -> 6.66 %) because Halpe's keypoints survive the 15 deg error
    better. So the signal shrank, and lowering the bar to match it (1.30 ->
    1.12) is what made this a weaker acceptance test than it was: a 12 %
    response to a 15 deg error would now pass.

    So the strength is taken back from the DAMAGE instead of the bar. The
    response is monotone in the error — 12 deg 1.16, 15 deg 1.27, 18 deg 1.40,
    20 deg 1.51, 25 deg 1.80 — and a gauge whose response halved would still
    clear 1.12 at 15 deg while failing 1.31 at 20 deg. Both are asserted, each
    with the suite's usual 10-15 % margin under today's measurement.
    """
    def turned(deg):
        return _measure(
            lambda rig: setattr(rig.ext["right"], "R",
                                _rotation([0, 1, 0], deg) @ rig.ext["right"].R))

    before = baseline.bone_cv["median_cv_pct"]
    for deg, bar, today in ((15.0, 1.12, 1.27), (20.0, 1.31, 1.51)):
        after = turned(deg).bone_cv["median_cv_pct"]
        rise = after / before
        assert rise >= bar, (                   # today 1.27 / 1.51
            f"bone-length spread only moved {100 * (rise - 1):.0f} % for a "
            f"{deg:.0f} deg camera error ({before:.2f} % -> {after:.2f} %); "
            f"it moved {100 * (today - 1):.0f} % when this was written")


def test_epipolar_responds_to_a_wrong_focal(baseline):
    """Scale BOTH focal lengths by 0.7 and the bone spread barely notices —
    a similarity-like error keeps the skeleton self-consistent while making it
    the wrong size and shape — but the two views stop agreeing about where a
    keypoint has to lie, which is what the epipolar row reports.

    Today: epipolar 4.67 px -> 5.48 px (+17 %), bone spread 5.23 % -> 5.44 %
    (+4 %). Neither row alone would have caught both this and the 15 deg
    rotation above. (It read +27 % / +3 % under COCO-17; the ratio shrank
    with the re-detection, and the `epi >= 1.15` bar below is now only 2 %
    under the measurement instead of the usual 10-15 % — recorded, not
    lowered, because lowering it is the one direction that cannot be right.)
    """
    def shrink(rig):
        for cam in CAMERAS:
            K = np.asarray(rig.intr[cam].K, float).copy()
            K[0, 0] *= 0.7
            K[1, 1] *= 0.7
            rig.intr[cam].K = K

    damaged = _measure(shrink)
    epi = damaged.epipolar["median_px"] / baseline.epipolar["median_px"]
    cv = (damaged.bone_cv["median_cv_pct"]
          / baseline.bone_cv["median_cv_pct"])
    assert epi >= 1.15, f"epipolar only moved {100 * (epi - 1):.0f} %"  # 1.27
    assert cv <= 1.15, (                                               # 1.03
        f"bone spread moved {100 * (cv - 1):.0f} % — this test is only "
        f"meaningful while it is the row that CANNOT see a shared focal error")
    # and it is reported per image, against that image's own diagonal, because
    # the two images are 2:1 apart in size
    for cam in CAMERAS:
        per = damaged.epipolar["per_image"][cam]
        assert np.isfinite(per["median_px"])
        assert 0.0 < per["median_pct_diag"] < 5.0


# --- the symmetry wording (F34) --------------------------------------------

def test_symmetry_flags_the_client_takes_forearm_and_names_a_keypoint(baseline):
    """The take's forearms measure 6.8 % apart on a subject moulded
    symmetrically, so it must be flagged — and the sentence must send the user
    to a KEYPOINT, not to the calibration.

    On the ground-truth Panoptic reference RTMPose alone produces 6.5 % thigh
    asymmetry against 1.4 % in the ground truth, so a left/right difference is
    evidence about a keypoint the detector confused and is NOT evidence about
    the rig: a rig error moves both sides together.
    """
    notes = symmetry_notes(baseline)
    assert notes, "the take's 6.8 % forearm asymmetry was not flagged"
    text = " ".join(notes)
    assert "forearm" in text
    assert JOINT_NAMES[int(Joint.LEFT_WRIST)] in text
    assert JOINT_NAMES[int(Joint.RIGHT_WRIST)] in text
    assert "check that keypoint in both images" in text.lower()
    assert "px" in text, "the endpoint residuals are not shown"
    for forbidden in ("calibration", "calibrat"):
        assert forbidden not in text.lower(), (
            "the symmetry note must never point at the calibration")


def test_symmetry_flags_at_five_percent_not_three(baseline):
    """A Monte-Carlo over this take's own keypoint noise puts the p99 of a
    perfectly symmetric subject at 4.24 %, so a 3 % flag would fire on a rigid
    mannequin most of the time and mean nothing when it mattered."""
    assert SYMMETRY_FLAG_PCT == 5.0
    quiet = copy.copy(baseline)
    quiet.symmetry = {"thigh": {"asym_pct": 4.2}}
    assert symmetry_notes(quiet) == []
    loud = copy.copy(baseline)
    loud.symmetry = {"thigh": {"asym_pct": 5.6}}
    assert len(symmetry_notes(loud)) == 1


# --- the toes are extremities: no take-wide number counts them -------------

def test_a_never_detected_toe_is_not_a_gap_in_the_reconstruction():
    """`gap_stats` is a take-wide readout, so it runs over CORE_INDEX.

    Every project detected before the toes existed has two NaN toes in every
    frame. Counting them would report 2 + 2*n_frames holes and an `undetected`
    of 4*n_frames on a take the client has already accepted — a number that
    moves the day a foot comes into frame, which is exactly what the core-set
    rule exists to prevent.
    """
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
    from pose3d.core.skeleton import CORE_INDEX, EXTREMITY_JOINTS
    from pose3d.quality import gap_stats

    p = ProjectData(name="pre-toes", keypoint_model="halpe26")
    for fid in ("0001", "0002"):
        f = Frame(frame_id=fid)
        f.pose3d[CORE_INDEX] = 1.0               # every core joint reconstructed
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.kp2d[cam][CORE_INDEX] = 1.0        # and detected in both views
        p.frames.append(f)                       # the toes stay NaN throughout

    g = gap_stats(p)
    assert g["n_missing"] == 0 and g["missing"] == []
    assert g["missing_pct"] == 0.0
    assert g["undetected"] == 0

    # and a CORE joint that is genuinely absent is still counted
    p.frames[0].pose3d[int(Joint.LEFT_WRIST)] = np.nan
    assert gap_stats(p)["n_missing"] == 1
    # the denominator is the core set, not every joint
    assert gap_stats(p)["missing_pct"] == 100.0 / (2 * len(CORE_INDEX))
    assert len(EXTREMITY_JOINTS) == 2


def test_the_figure_height_denominator_does_not_move_when_a_foot_appears():
    """`figure_height_px` is the denominator of every px percentage the client
    reads, so a foot coming into frame must not change it."""
    from pose3d.core.skeleton import CORE_INDEX, NUM_JOINTS
    from pose3d.quality import figure_height_px

    kp = np.full((3, NUM_JOINTS, 2), np.nan)
    kp[:, CORE_INDEX, 0] = 10.0
    kp[:, CORE_INDEX, 1] = np.linspace(100.0, 400.0, len(CORE_INDEX))
    without = figure_height_px(kp)
    assert without == pytest.approx(300.0)

    kp[:, int(Joint.LEFT_TOE)] = (10.0, 900.0)   # a toe well below the bbox
    kp[:, int(Joint.RIGHT_TOE)] = (10.0, 900.0)
    assert figure_height_px(kp) == pytest.approx(without)


def test_the_subject_height_is_head_to_ankle_whether_or_not_the_toes_are_seen():
    """`subject_height` is THE denominator of every "% of body height" the
    client reads, and the Set-scale math on top of it. A take that happens to
    have its feet in frame must not measure a different subject."""
    from pose3d.core.skeleton import NUM_JOINTS
    from pose3d.quality import subject_height
    from tests.synth import sample_skeleton_3d

    with_toes = np.stack([sample_skeleton_3d() for _ in range(3)])
    assert np.isfinite(with_toes[:, int(Joint.LEFT_TOE)]).all(), "fixture"
    assert (with_toes[0, int(Joint.LEFT_TOE), 2]
            < with_toes[0, int(Joint.LEFT_ANKLE), 2]), "the toes are lower"

    cropped = with_toes.copy()
    cropped[:, [int(Joint.LEFT_TOE), int(Joint.RIGHT_TOE)]] = np.nan

    assert subject_height(with_toes) == pytest.approx(subject_height(cropped))
    assert with_toes.shape[1] == NUM_JOINTS

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
    assert baseline.bone_cv["median_cv_pct"] == pytest.approx(5.4, abs=0.6)
    assert baseline.epipolar["median_px"] == pytest.approx(4.9, abs=0.6)


def test_bone_cv_responds_to_a_wrong_rig(baseline):
    """Rotate the right camera by 15 deg — 19.13 mm of pose error, 16.2 % of
    the subject's height — and the bone-length spread must say so.

    This is the acceptance test for the whole phase: that error costs the
    reprojection gauge 2.9 points and never changes its band, so if nothing on
    screen moved for it, nothing on screen describes the reconstruction.
    Today: 5.44 % -> 8.18 %, a 50 % rise.
    """
    damaged = _measure(
        lambda rig: setattr(rig.ext["right"], "R",
                            _rotation([0, 1, 0], 15.0) @ rig.ext["right"].R))
    before = baseline.bone_cv["median_cv_pct"]
    after = damaged.bone_cv["median_cv_pct"]
    rise = after / before
    assert rise >= 1.30, (                      # today 1.51
        f"bone-length spread only moved {100 * (rise - 1):.0f} % for a 15 deg "
        f"camera error ({before:.2f} % -> {after:.2f} %)")


def test_epipolar_responds_to_a_wrong_focal(baseline):
    """Scale BOTH focal lengths by 0.7 and the bone spread barely notices —
    a similarity-like error keeps the skeleton self-consistent while making it
    the wrong size and shape — but the two views stop agreeing about where a
    keypoint has to lie, which is what the epipolar row reports.

    Today: epipolar 4.91 px -> 6.23 px (+27 %), bone spread 5.44 % -> 5.58 %
    (+3 %). Neither row alone would have caught both this and the 15 deg
    rotation above.
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
    """The take's forearms measure 6.7 % apart on a subject moulded
    symmetrically, so it must be flagged — and the sentence must send the user
    to a KEYPOINT, not to the calibration.

    On the ground-truth Panoptic reference RTMPose alone produces 6.5 % thigh
    asymmetry against 1.4 % in the ground truth, so a left/right difference is
    evidence about a keypoint the detector confused and is NOT evidence about
    the rig: a rig error moves both sides together.
    """
    notes = symmetry_notes(baseline)
    assert notes, "the take's 6.7 % forearm asymmetry was not flagged"
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

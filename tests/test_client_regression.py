"""The regression net: the client's own take, measured, with thresholds.

Nothing in the suite used to run the reconstruct path with an accuracy
threshold on it, so every defect the September 2026 audit found was invisible
to CI (F32, F45). These assertions are that net. They run on the committed
fixture (`tests/fixtures/client_take`, built by `regen_client_take.py` from
`workspace/pose3d_projects/Imported_Session`) — no images, no detector, ~1 s.

Every threshold sits 10-15 % above the value measured on 2026-09-01, with that
value in a comment, so a threshold can never be met by having been set at the
measurement. The two `xfail(strict=True)` tests at the bottom are the two
things that are wrong today and that Phase 1 fixes: they must report XFAIL
until then, and XPASS (a failure) the moment they start passing, which is the
signal to promote them.

The numbers are `docs/audit-2026-09/wf_baseline.md`; the metric definitions are
`pose3d.quality`.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from pose3d import pipeline
from pose3d.core.io_project import load_project
from pose3d.core.project import CAMERAS
from pose3d.quality import SYMMETRY_PAIRS, load_rig, take_quality
from tests.gates import needs_character

FIXTURE = Path(__file__).parent / "fixtures" / "client_take"


@pytest.fixture(scope="module")
def project():
    return load_project(FIXTURE)


@pytest.fixture(scope="module")
def rig():
    return load_rig(FIXTURE / "calibration")


@pytest.fixture(scope="module")
def quality(project, rig):
    """Geometry only — no character, so this runs without the rig asset."""
    return take_quality(project, rig)


@pytest.fixture(scope="module")
def quality_with_character(project, rig):
    from pose3d.geometry.character import Character
    return take_quality(project, rig, Character())


def test_the_fixture_is_the_client_take(quality):
    """The denominator every other threshold is expressed in. If the fixture
    is truncated or re-detected, this is what says so."""
    assert quality.n_frames == 26
    assert quality.subject_height_m == pytest.approx(0.1178, abs=0.0012)  # today 0.11778
    assert quality.figure_h_px["left"] == pytest.approx(776, rel=0.02)    # today 776
    assert quality.figure_h_px["right"] == pytest.approx(407, rel=0.02)   # today 407


def test_raw_bone_lengths_are_consistent_across_frames(quality):
    """The mannequin is rigid, so every bone length is constant by
    construction: all of this spread is pipeline error."""
    assert quality.bone_cv["median_cv_pct"] <= 6.0    # today 5.3
    assert quality.bone_cv["max_cv_pct"] <= 10.5      # today 9.5


def test_left_and_right_limbs_measure_the_same(quality):
    """Same moulded limb on both sides of one mannequin; a systematic L/R
    difference is reconstruction error, not anatomy."""
    for label, *_ in SYMMETRY_PAIRS:
        asym = quality.symmetry[label]["asym_pct"]
        # today: thigh 4.6, shin 0.7, upper arm 0.5, forearm 6.7
        assert asym <= 7.5, f"{label} L/R asymmetry {asym:.1f} %"


def test_the_two_views_agree_about_the_body_keypoints(quality):
    """Independent of the tags: the body keypoints play no part in the
    calibration, so this is the rig's own consistency."""
    assert quality.epipolar["median_px"] <= 6.0   # today 4.9
    assert quality.epipolar["p90_px"] <= 14.0     # today 12.5


@needs_character()
def test_the_character_follows_the_captured_pose(quality_with_character):
    """How faithfully the skinning reproduces its input — it cannot see error
    already in the input, which is why it is only one of these assertions."""
    median = quality_with_character.retarget_pct_height["median_pct_height"]
    assert median <= 2.0, f"{median:.2f} % of body height"   # today 1.5


@pytest.mark.xfail(strict=True, reason=(
    "F01: fit_project ends with a causal EMA over frames that are discrete "
    "hand-posed shots, so the pose that ships lags the keypoints by 4.2x "
    "(left) / 6.2x (right) more than the measurement does. Phase 1 removes "
    "the smoother from the default path and promotes this to a live gate."))
def test_the_delivered_pose_reprojects_like_the_measurement(quality):
    """The pose the app shows and exports must be no further from the 2D
    keypoints than the raw triangulation is."""
    for cam in CAMERAS:
        ratio = quality.reproj_ratio(cam)
        # today 4.24 left / 6.17 right
        assert ratio <= 1.5, f"{cam}: delivered reprojects {ratio:.2f}x worse"


@pytest.mark.xfail(strict=True, reason=(
    "F01/F14: the same smoother. The stored fitted3d is the bone fit's answer "
    "dragged backwards in time by up to 21.8 % of body height, so the shipped "
    "pose is not the pose the fit produced. Phase 1 removes it."))
def test_the_delivered_pose_is_the_pose_the_fit_produced(project, quality):
    """Re-run the fit with no smoothing: the stored pose should be it."""
    refit = copy.deepcopy(project)
    pipeline.fit_project(refit, smooth=False)
    stored = np.stack([f.fitted3d for f in project.frames])
    fresh = np.stack([f.fitted3d for f in refit.frames])
    worst = float(np.nanmax(np.linalg.norm(stored - fresh, axis=2)))
    pct = 100.0 * worst / quality.subject_height_m
    # today 21.8 % of body height (25.7 mm on a 0.118 m subject)
    assert pct <= 3.0, f"delivered pose is {pct:.1f} % of height off the fit"

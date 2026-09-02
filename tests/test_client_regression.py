"""The regression net: the client's own take, RECONSTRUCTED, with thresholds.

Nothing in the suite used to run the reconstruct path with an accuracy
threshold on it, so every defect the September 2026 audit found was invisible
to CI (F32, F45). These assertions are that net. They run on the committed
fixture (`tests/fixtures/client_take`, built by `regen_client_take.py` from
`workspace/pose3d_projects/Imported_Session`) — no images, no detector, ~2 s.

Every number here is measured by RUNNING the pipeline: the fixture supplies the
2D keypoints and the calibration, and `triangulate_project` + `fit_project`
produce the `pose3d` and `fitted3d` that are then measured. That is the whole
point — asserting on the arrays stored in the fixture would measure a build
that shipped in August and could not notice a sign error in the DLT, a broken
cross-view gate or a regressed bone fit, because no change to this repo can
move a frozen array.

The fixture's STORED `fitted3d` is kept for exactly one job: it is the pose the
client was actually sent, and the last test pins that it still is, so the two
gates that were `xfail(strict=True)` while Phase 1 was written stay meaningful.

Every threshold sits 10-15 % above the value measured on 2026-09-02, with that
value in a comment, so a threshold can never be met by having been set at the
measurement.

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
    """The fixture as committed — 2D keypoints, calibration, and the arrays the
    OLD build stored. Only the last test reads its `fitted3d`."""
    return load_project(FIXTURE)


@pytest.fixture(scope="module")
def rig():
    return load_rig(FIXTURE / "calibration")


@pytest.fixture(scope="module")
def delivered(project, rig):
    """The take AS THIS BUILD RECONSTRUCTS IT, by the app's own path.

    `ProjectModel.recompute_all` is exactly these two calls, so `pose3d` here
    is what the DLT produces from the fixture's 2D today and `fitted3d` is what
    opening the client's project now shows and exports. Everything below is
    measured on this, so a regression anywhere in triangulate / cross-view /
    bone-fit moves a number in this file.
    """
    p = copy.deepcopy(project)
    pipeline.triangulate_project(p, rig)
    pipeline.fit_project(p)                  # shipped defaults: no smoothing
    return p


@pytest.fixture(scope="module")
def quality(delivered, rig):
    """Geometry only — no character, so this runs without the rig asset."""
    return take_quality(delivered, rig)


@pytest.fixture(scope="module")
def quality_with_character(delivered, rig):
    from pose3d.geometry.character import Character
    return take_quality(delivered, rig, Character())


def test_the_fixture_is_the_client_take(quality):
    """The denominator every other threshold is expressed in. If the fixture
    is truncated or re-detected, this is what says so."""
    assert quality.n_frames == 26
    # today 0.11940 on the reconstructed pose. The audit's 0.1178 is the same
    # measurement on the fixture's STORED pose: the smoother it was made with
    # shrank the figure, which is the defect, not a change of subject.
    assert quality.subject_height_m == pytest.approx(0.1194, abs=0.0014)
    assert quality.figure_h_px["left"] == pytest.approx(776, rel=0.02)    # today 776
    assert quality.figure_h_px["right"] == pytest.approx(407, rel=0.02)   # today 407


def test_the_reconstruction_is_measured_not_replayed(project, delivered):
    """What makes every threshold in this file a gate on the CODE.

    The reconstructed `pose3d` must be the fixture's 2D put through today's
    triangulation — not the array the fixture happens to carry — so a sign or
    DLT regression shows up here first, as a pose that no longer matches what
    the same calibration produced on 2026-09-01.
    """
    stored = np.stack([f.pose3d for f in project.frames])
    fresh = np.stack([f.pose3d for f in delivered.frames])
    assert np.array_equal(np.isnan(stored), np.isnan(fresh)), \
        "a different set of joints reconstructs than on 2026-09-01"
    worst = float(np.nanmax(np.abs(stored - fresh)))
    assert worst <= 3e-7, f"triangulation moved by {worst:.2e} m"  # today 2.4e-7


def test_an_interpolated_joint_is_not_counted_as_a_measurement(quality,
                                                               delivered):
    """The gap fill writes into the fit's INPUT, never into `pose3d`, so the
    bone-length CV, the symmetry and the "measured" reprojection above are all
    computed on observations only. Two joint-frames on this take are filled
    (0012 RIGHT_KNEE, 0021 LEFT_ANKLE) and they are counted, separately."""
    assert quality.gaps["filled"] == 2
    assert quality.gaps["n_missing"] == 2      # still holes in the measurement
    for f in delivered.frames:
        if f.filled.any():
            assert np.isnan(f.pose3d[f.filled]).all()
            assert np.isfinite(f.fitted3d[f.filled]).all()


def test_raw_bone_lengths_are_consistent_across_frames(quality):
    """The mannequin is rigid, so every bone length is constant by
    construction: all of this spread is pipeline error."""
    assert quality.bone_cv["median_cv_pct"] <= 6.0    # today 5.27
    assert quality.bone_cv["max_cv_pct"] <= 10.5      # today 9.52


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
    # today 1.65 on the reconstructed pose (1.62 on the fixture's stored one)
    assert median <= 1.9, f"{median:.2f} % of body height"


def test_the_delivered_pose_reprojects_like_the_measurement(quality):
    """F01. The pose the app shows and exports must be no further from the 2D
    keypoints than the raw triangulation is.

    This was `xfail(strict=True)` until Phase 1: the causal EMA over frames
    that are discrete hand-posed shots put the delivered pose 4.24x (left) /
    6.17x (right) further from the keypoints than the measurement — 21.65 /
    17.18 px. It is a live gate now."""
    for cam in CAMERAS:
        ratio = quality.reproj_ratio(cam)
        # today 1.44 left / 1.44 right (7.37 / 4.02 px); was 4.24 / 6.17
        assert ratio <= 1.5, f"{cam}: delivered reprojects {ratio:.2f}x worse"


def test_the_delivered_pose_is_the_pose_the_fit_produced(project, rig,
                                                         delivered, quality):
    """F01/F14. Nothing may sit between the bone fit and the pose the app
    delivers: re-run the fit with smoothing explicitly off and the delivered
    pose IS it.

    Also `xfail(strict=True)` until Phase 1, when the stored `fitted3d` was
    the fit's answer dragged backwards in time by up to 21.8 % of body height.
    The gate is live and it fails again the day smoothing is re-enabled by
    default at any call site on this path."""
    refit = copy.deepcopy(project)
    pipeline.triangulate_project(refit, rig)
    pipeline.fit_project(refit, smooth=False)
    shown = np.stack([f.fitted3d for f in delivered.frames])
    fresh = np.stack([f.fitted3d for f in refit.frames])
    worst = float(np.nanmax(np.linalg.norm(shown - fresh, axis=2)))
    pct = 100.0 * worst / quality.subject_height_m
    # today 0.0 % of body height (bit-identical); 21.8 % before Phase 1
    assert pct <= 3.0, f"delivered pose is {pct:.1f} % of height off the fit"


def test_the_fixture_still_carries_the_pose_the_client_was_sent(project,
                                                                delivered,
                                                                quality):
    """The two gates above are only worth something while the fixture holds
    the OLD build's `fitted3d`. If it is ever regenerated from a fixed build
    this fails, and says to re-baseline the two comments above rather than
    quietly measuring a fixed pose against a fixed pose."""
    stored = np.stack([f.fitted3d for f in project.frames])
    shown = np.stack([f.fitted3d for f in delivered.frames])
    worst = float(np.nanmax(np.linalg.norm(stored - shown, axis=2)))
    pct = 100.0 * worst / quality.subject_height_m
    # today 21.5 % of height: the lag the client complained about
    assert pct >= 10.0, (
        f"the fixture's stored pose is only {pct:.1f} % of height from what "
        f"this build delivers — was it regenerated on a fixed build?")

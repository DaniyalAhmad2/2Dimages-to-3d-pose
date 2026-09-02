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

The pose the client was actually SENT is `fixtures/client_take/as_delivered.json`
— the August build's own `fitted3d`, copied out of the source project and never
recomputed. The last test measures against it, so the lag defect stays pinned
even now that the fixture itself has been regenerated on a fixed build, and the
two gates that were `xfail(strict=True)` while Phase 1 was written stay
meaningful.

Every threshold sits 10-15 % above the value measured on 2026-09-03, with that
value in a comment, so a threshold can never be met by having been set at the
measurement. That date is Phase 5b: the app now detects with Halpe-26 (a skull
HEAD instead of the nose — `detect.rtmpose.USE_HALPE26`), the fixture was
re-detected with it, and the subject the DLT reconstructs is 9 % taller than
the COCO-17 one, so every "% of body height" number moved with it.

The numbers are `docs/audit-2026-09/wf_baseline.md`; the metric definitions are
`pose3d.quality`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pose3d import pipeline
from pose3d.core.io_project import load_project
from pose3d.core.project import CAMERAS
from pose3d.core.skeleton import Joint
from pose3d.quality import SYMMETRY_PAIRS, load_rig, take_quality
from tests.gates import needs_character

FIXTURE = Path(__file__).parent / "fixtures" / "client_take"

#: The observations today's cross-view gate refuses on this take, as
#: (frame_id, joint name). Both are detection errors and the evidence for
#: calling them that is in test_cross_view.py::
#: test_the_client_take_is_gated_on_its_own_distribution. They are named — not
#: counted — because "the gate rejected two things" is only reassuring while
#: they are the same two.
GATED_OUT = {("0013", "LEFT_ELBOW"), ("0021", "LEFT_ANKLE")}


@pytest.fixture(scope="module")
def project():
    """The fixture as committed — the 2D keypoints and the calibration every
    test below reconstructs from, plus the 3D the build that regenerated it
    recorded. Only `test_the_reconstruction_is_measured_not_replayed` reads
    that 3D, and only to check this build still produces it."""
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

    # the take's OWN head convention. The fixture is a "skull" project, and
    # measuring it under the default "nose" one would apply the 45 deg nose
    # correction to a point that needs none — a retarget number for a pose the
    # app never shows. Nothing outside the UI may rely on the process default.
    return take_quality(delivered, rig,
                        Character(head_source=delivered.head_source))


def test_the_fixture_is_the_client_take(quality, delivered):
    """The denominator every other threshold is expressed in. If the fixture
    is truncated or re-detected, this is what says so."""
    assert quality.n_frames == 26
    # and WHICH detector it was re-detected with: the "% of height" thresholds
    # below are only comparable within one keypoint layout, and the retarget
    # ones are only correct if the character is told the same thing
    assert delivered.keypoint_model == "halpe26"
    assert delivered.head_source == "skull"
    # today 0.13019 on the reconstructed pose (0.11940 under COCO-17, whose
    # HEAD is the nose: the skull vertex sits ~9 % of a body height higher, and
    # the height is a z-extent). The audit's 0.1178 was the same measurement on
    # the pose the client was sent, which the smoother had shrunk.
    assert quality.subject_height_m == pytest.approx(0.1302, abs=0.0015)
    assert quality.figure_h_px["left"] == pytest.approx(863, rel=0.02)    # today 863
    assert quality.figure_h_px["right"] == pytest.approx(446, rel=0.02)   # today 446


def test_the_reconstruction_is_measured_not_replayed(project, delivered):
    """What makes every threshold in this file a gate on the CODE.

    The reconstructed `pose3d` must be the fixture's 2D put through today's
    triangulation — not the array the fixture happens to carry — so a sign or
    DLT regression shows up here first, as a pose that no longer matches what
    the same calibration produced on 2026-09-03.

    The residual is not zero and must not be asserted to be: the fixture's 2D
    is stored at 3 decimal places of a pixel (regen_client_take.KP_DECIMALS)
    while its 3D was computed from the full-precision detections.

    Nor is the set of reconstructed joints identical to the fixture's: the
    fixture's `pose3d` was written before the data-driven cross-view gate
    (Phase 6.2) existed, and that gate refuses two of this take's 390 pairs —
    0013 LEFT_ELBOW and 0021 LEFT_ANKLE, both detection errors
    (test_cross_view.py::test_the_client_take_is_gated_on_its_own_distribution
    names the evidence). So today's reconstruction has exactly those two
    joints fewer, and every other joint to 2.0e-7 m. A joint appearing where
    the fixture has none, or a third one going missing, still fails here.
    """
    stored = np.stack([f.pose3d for f in project.frames])
    fresh = np.stack([f.pose3d for f in delivered.frames])
    lost = np.isnan(fresh).any(2) & ~np.isnan(stored).any(2)
    assert not (np.isnan(stored).any(2) & ~np.isnan(fresh).any(2)).any(), \
        "a joint reconstructs that did not on 2026-09-03"
    assert {(project.frames[t].frame_id, Joint(int(j)).name)
            for t, j in np.argwhere(lost)} == GATED_OUT, \
        "a different set of joints is gated out than on 2026-09-03"
    common = ~np.isnan(stored) & ~np.isnan(fresh)
    worst = float(np.max(np.abs(stored - fresh)[common]))
    assert worst <= 3e-7, f"triangulation moved by {worst:.2e} m"  # today 2.0e-7


def test_an_interpolated_joint_is_not_counted_as_a_measurement(quality,
                                                               delivered):
    """The gap fill writes into the fit's INPUT, never into `pose3d`, so the
    bone-length CV, the symmetry and the "measured" reprojection above are all
    computed on observations only.

    The Halpe-26 detection of this take leaves no holes of its own, but the
    cross-view gate refuses two of its pairs (0013 LEFT_ELBOW, 0021
    LEFT_ANKLE), and a gated observation is a hole in the measurement like any
    other: 2 missing, 2 filled, and those two joints must be NaN in `pose3d`
    and finite in `fitted3d` — an interpolation the export uses and no metric
    counts. Under COCO-17 the holes were the detector's own (0012 RIGHT_KNEE,
    0021 LEFT_ANKLE) and the count was the same 2.
    """
    assert quality.gaps["n_missing"] == 2
    assert quality.gaps["filled"] == 2
    assert set(map(tuple, quality.gaps["missing"])) == GATED_OUT
    for f in delivered.frames:
        if f.filled.any():
            assert np.isnan(f.pose3d[f.filled]).all()
            assert np.isfinite(f.fitted3d[f.filled]).all()


def test_raw_bone_lengths_are_consistent_across_frames(quality):
    """The mannequin is rigid, so every bone length is constant by
    construction: all of this spread is pipeline error."""
    assert quality.bone_cv["median_cv_pct"] <= 6.0    # today 5.30
    assert quality.bone_cv["max_cv_pct"] <= 8.9       # today 7.86 (was 9.52)


def test_left_and_right_limbs_measure_the_same(quality):
    """Same moulded limb on both sides of one mannequin; a systematic L/R
    difference is reconstruction error, not anatomy."""
    for label, *_ in SYMMETRY_PAIRS:
        asym = quality.symmetry[label]["asym_pct"]
        # today: thigh 4.6, shin 2.2, upper arm 0.1, forearm 6.9
        assert asym <= 7.8, f"{label} L/R asymmetry {asym:.1f} %"


def test_the_two_views_agree_about_the_body_keypoints(quality):
    """Independent of the tags: the body keypoints play no part in the
    calibration, so this is the rig's own consistency."""
    assert quality.epipolar["median_px"] <= 5.3   # today 4.67
    assert quality.epipolar["p90_px"] <= 14.5     # today 12.81


@needs_character()
def test_the_character_follows_the_captured_pose(quality_with_character):
    """How faithfully the skinning reproduces its input — it cannot see error
    already in the input, which is why it is only one of these assertions."""
    median = quality_with_character.retarget_pct_height["median_pct_height"]
    # today 1.47 (1.65 under COCO-17: most of that was the nose HEAD, which the
    # skull HEAD retires — 12.73 -> 1.54 % of height at HEAD alone)
    assert median <= 1.65, f"{median:.2f} % of body height"


def test_the_delivered_pose_reprojects_like_the_measurement(quality):
    """F01. The pose the app shows and exports must be no further from the 2D
    keypoints than the raw triangulation is.

    This was `xfail(strict=True)` until Phase 1: the causal EMA over frames
    that are discrete hand-posed shots put the delivered pose 4.24x (left) /
    6.17x (right) further from the keypoints than the measurement — 21.65 /
    17.18 px. It is a live gate now."""
    for cam in CAMERAS:
        ratio = quality.reproj_ratio(cam)
        # today 1.42 left / 1.40 right; was 4.24 / 6.17. Fixed at 1.5 by the
        # audit rather than set 10-15 % above the measurement: it is one of the
        # four promoted gates, and its bar is what the client was promised.
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
    # today 0.0 % of body height (bit-identical); 21.8 % before Phase 1. Fixed
    # at 3.0 by the audit — a promoted gate, not a measured threshold.
    assert pct <= 3.0, f"delivered pose is {pct:.1f} % of height off the fit"


def test_the_fixture_still_carries_the_pose_the_client_was_sent(delivered,
                                                                quality):
    """The two gates above are only worth something while SOMETHING still holds
    the August build's `fitted3d` — the smoothed, lagging pose the client was
    actually sent — to measure them against.

    `project.json` stopped being that the moment the fixture was regenerated on
    a fixed build (Phase 5b re-detected it with Halpe-26), so the delivered pose
    was lifted into `as_delivered.json` first and is copied forward untouched by
    every regeneration. If this number ever collapses towards zero, the two
    gates above have started measuring a fixed pose against a fixed pose and
    they, not this file, are what needs re-reading.
    """
    doc = json.loads((FIXTURE / "as_delivered.json").read_text())
    # missing joints are null in the file, not NaN, so they cannot be handed
    # straight to a float array
    sent = np.array([[[np.nan if v is None else v for v in xyz]
                      for xyz in doc["frames"][f.frame_id]]
                     for f in delivered.frames], dtype=float)
    shown = np.stack([f.fitted3d for f in delivered.frames])
    worst = float(np.nanmax(np.linalg.norm(sent - shown, axis=2)))
    pct = 100.0 * worst / quality.subject_height_m
    # today 19.6 % of height: the lag the client complained about, now measured
    # across a change of HEAD convention too (21.5 % before Phase 5b)
    assert pct >= 10.0, (
        f"the pose the client was sent is only {pct:.1f} % of height from what "
        f"this build delivers — was as_delivered.json regenerated from a fixed "
        f"build?")

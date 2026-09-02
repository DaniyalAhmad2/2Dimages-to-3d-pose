"""The cross-view gate must reject hallucinations, not good data.

When it over-fires the result reads as the detector failing: gaps in the 2D
views and a sparse 3D preview. That is exactly what a flat 30 px tolerance did
on 3072x4080 phone captures — 0.7% of image height, which threw away 38% of a
real take.

It is a MASK (`Frame.rejected`), not a deletion: the observations stay in
`frame.kp2d`, drawn and draggable, and the mask is re-derived from them on
every recompute. It used to NaN them in place, which `save_project` then made
permanent — see `test_a_bad_rig_does_not_destroy_kp2d`.

Everything that exercises the gate runs on both rigs: the wide symmetric one
blown up to phone resolution, and the client's genuinely asymmetric close-range
rig, whose two images differ 2:1 in pixel scale.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.pipeline import (
    CalibratedRig, epipolar_threshold, triangulate_project, validate_cross_view,
)
from tests.synth import (
    cameras, close_range_two_cam, default_two_cam, project as project_points,
)
from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics


def _upscaled_symmetric(scale=4.0):
    """The 720p synthetic rig blown up to phone-sized images."""
    geo = default_two_cam()
    K = geo["K"].copy() * scale
    K[2, 2] = 1.0
    size = (int(geo["size"][0] * scale), int(geo["size"][1] * scale))
    return {**geo, "K": K, "size": size}


@pytest.fixture(params=["wide symmetric at phone resolution",
                        "close asymmetric (client rig)"])
def geo(request):
    return (_upscaled_symmetric() if request.param.startswith("wide")
            else close_range_two_cam())


def _rig_and_project(geo, n_frames=2):
    """A consistent two-view capture of `geo`'s subject."""
    cams = cameras(geo)
    intr = {c: Intrinsics(K=cams[c][0], dist=cams[c][1], image_size=cams[c][2])
            for c in (CAM_LEFT, CAM_RIGHT)}
    rig = CalibratedRig(intr[CAM_LEFT], intr[CAM_RIGHT],
                        Extrinsics(*geo["left"]), Extrinsics(*geo["right"]))

    gt = geo["subject"]
    data = ProjectData(name="cv")
    for i in range(n_frames):
        f = Frame(frame_id=f"{i:04d}")
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.kp2d[cam] = project_points(gt, cams[cam][0], cams[cam][1], *geo[cam])
            f.scores[cam] = np.full(NUM_JOINTS, 0.9)
        data.frames.append(f)
    return rig, data, gt


def test_consistent_observations_survive_at_phone_resolution(geo):
    """Regression guard: with a flat 30 px tolerance this dropped a third of a
    real take, which read to the user as 'half the keypoints not detected'."""
    rig, data, _ = _rig_and_project(geo)
    dropped = validate_cross_view(data, rig)
    assert dropped == 0, f"{dropped} good observations rejected"
    assert not np.isnan(data.frames[0].kp2d[CAM_LEFT]).any()


def test_the_tolerance_scales_with_the_image():
    small, _, _ = _rig_and_project(default_two_cam())
    big, _, _ = _rig_and_project(_upscaled_symmetric(4.0))
    assert epipolar_threshold(big) > 3.0 * epipolar_threshold(small)
    # and still reproduces roughly the historical 30 px at ~1080p
    hd = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(1920, 1080))
    assert 25.0 < epipolar_threshold(CalibratedRig(hd, hd, *[Extrinsics(
        R=np.eye(3), t=np.zeros(3))] * 2)) < 40.0


def test_triangulate_reports_what_it_threw_away(geo):
    """The count has to reach the UI, or a calibration problem masquerades as
    a detection problem with nothing to tell them apart."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][int(Joint.LEFT_ANKLE)] = 0.3
    assert triangulate_project(data, rig) == 1


def test_the_rejection_note_only_fires_when_it_matters():
    """One note, owned by the pipeline that owns the policy, so the import
    dialog and the recalculate status line cannot drift apart."""
    from pose3d.pipeline import rejection_note
    assert rejection_note(0, 10) == ""
    assert rejection_note(5, 10) == ""                   # 3%: normal occlusion
    note = rejection_note(60, 10)                        # 40%: something is wrong
    assert "40%" in note and "calibration" in note.lower()


def test_an_uncalibrated_import_still_reports_a_count():
    """Regression guard: `dropped` used to be bound only inside the
    `if rig is not None` branch, so importing without a calibration raised
    NameError into the dialog's blanket handler and showed "Import failed"."""
    from pose3d.pipeline import rejection_note
    assert rejection_note(0, 0) == ""      # no frames, no calibration, no crash


def test_a_hallucinated_joint_is_still_dropped(geo):
    """The gate's actual job: one view putting the ankle on the knee."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][int(Joint.LEFT_ANKLE)] = 0.3      # the weaker view

    dropped = validate_cross_view(data, rig)
    assert dropped == 1
    assert f.rejected[CAM_LEFT][int(Joint.LEFT_ANKLE)]
    assert not f.rejected[CAM_RIGHT][int(Joint.LEFT_ANKLE)]


# --- the gate is a mask, not a deletion (F06) -------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "client_take"


def _damaged_left(rig, deg=12.0, axis=(0, 1, 0)):
    """The same rig with the LEFT camera's orientation wrong by `deg`.

    12 deg is the audit's measured arming point for the destructive gate: it
    dropped 0 observations up to 5 deg, 12 at 8 deg and 185 of 780 at 12 deg
    on the client take. It is not a hypothetical — taking the wrong IPPE
    branch for a single tag is worth 63.7 deg, and this rig is solved from one
    tag.
    """
    from tests.synth import rot_about
    dR = rot_about(axis, deg)
    return CalibratedRig(
        rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
        Extrinsics(R=dR @ rig.ext[CAM_LEFT].R, t=dR @ rig.ext[CAM_LEFT].t),
        rig.ext[CAM_RIGHT])


def test_a_bad_rig_does_not_destroy_kp2d(tmp_path):
    """Damage -> save -> reload -> recompute with the good rig: EVERY
    observation comes back.

    Before this, it recovered 0 of them. The gate wrote NaN into `frame.kp2d`
    and 0.0 into `frame.scores`, `save_project` persisted that, and nothing in
    the app could undo it: "Recalculate 3D" re-ran the same gate over the
    already emptied 2D, and the NaN'd joints were not even drawn, so the user
    could not drag them back or tell the loss from a detection failure. Only a
    full re-detection restored them, and nothing said so.

    On the client's own rig and 2D (the committed fixture), a 12 deg error
    about the left camera's vertical rejects 347 of its 778 observations.
    """
    from pose3d.core.io_project import load_project, save_project
    from pose3d.quality import load_rig

    data = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    before = {(i, c): data.frames[i].kp2d[c].copy()
              for i in range(len(data.frames)) for c in (CAM_LEFT, CAM_RIGHT)}
    n_obs = sum(int(np.isfinite(v).all(1).sum()) for v in before.values())

    dropped = triangulate_project(data, _damaged_left(rig))
    assert dropped > 100, f"only {dropped} rejected; the damage must bite"
    # ...and every one of them is a MASK, not a deletion
    assert sum(int(f.rejected[c].sum()) for f in data.frames
               for c in (CAM_LEFT, CAM_RIGHT)) == dropped
    for (i, c), xy in before.items():
        assert np.array_equal(data.frames[i].kp2d[c], xy, equal_nan=True)

    save_project(data, tmp_path)
    reloaded = load_project(tmp_path)
    triangulate_project(reloaded, rig)          # the GOOD rig, as recompute does

    after = sum(int(np.isfinite(f.kp2d[c]).all(1).sum())
                for f in reloaded.frames for c in (CAM_LEFT, CAM_RIGHT))
    assert after == n_obs, f"{n_obs - after} of {n_obs} observations lost"
    assert not any(f.rejected[c].any() for f in reloaded.frames
                   for c in (CAM_LEFT, CAM_RIGHT)), "stale mask survived"
    # and the 3D the take had before the damage is back, joint for joint
    clean = load_project(FIXTURE)
    triangulate_project(clean, rig)
    for a, b in zip(clean.frames, reloaded.frames):
        assert np.array_equal(a.pose3d, b.pose3d, equal_nan=True)


def test_the_mask_decides_the_3d_and_nothing_else(geo):
    """A rejected observation must not reach the triangulation — and must not
    leave the 2D arrays either."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    ankle = int(Joint.LEFT_ANKLE)
    f.kp2d[CAM_LEFT][ankle] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][ankle] = 0.3

    assert triangulate_project(data, rig) == 1
    assert f.rejected[CAM_LEFT][ankle] and not f.rejected[CAM_RIGHT][ankle]
    assert np.isfinite(f.kp2d[CAM_LEFT][ankle]).all()   # still there, still drawn
    assert f.scores[CAM_LEFT][ankle] == 0.3             # and still its own score
    assert np.isnan(f.pose3d[ankle]).all()              # but no 3D from it


def test_the_mask_is_re_derived_not_accumulated(geo):
    """Put the hallucinated point back where it belongs and the rejection
    goes away — the mask describes the 2D as it is now, every time."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    ankle = int(Joint.LEFT_ANKLE)
    good = f.kp2d[CAM_LEFT][ankle].copy()
    f.kp2d[CAM_LEFT][ankle] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][ankle] = 0.3
    assert triangulate_project(data, rig) == 1

    f.kp2d[CAM_LEFT][ankle] = good
    assert triangulate_project(data, rig) == 0
    assert not any(f.rejected[c].any() for f in data.frames
                   for c in (CAM_LEFT, CAM_RIGHT))
    assert not np.isnan(f.pose3d[ankle]).any()


def test_a_hand_placed_point_is_not_left_flagged(geo):
    """`Frame.set_kp` is the user overruling the gate; the purple dot must not
    outlive the point it was measured on."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    ankle = int(Joint.LEFT_ANKLE)
    f.kp2d[CAM_LEFT][ankle] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][ankle] = 0.3
    triangulate_project(data, rig)
    assert f.rejected[CAM_LEFT][ankle]

    f.set_kp(CAM_LEFT, ankle, 10.0, 20.0, score=1.0, corrected=True)
    assert not f.rejected[CAM_LEFT][ankle]

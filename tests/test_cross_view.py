"""The cross-view gate must reject hallucinations, not good data.

It NaNs observations out of `frame.kp2d`, so when it over-fires the result is
indistinguishable from the detector failing: gaps in the 2D views and an empty
3D preview. That is exactly what a flat 30 px tolerance did on 3072x4080 phone
captures — 0.7% of image height, which threw away 38% of a real take.

Everything that exercises the gate runs on both rigs: the wide symmetric one
blown up to phone resolution, and the client's genuinely asymmetric close-range
rig, whose two images differ 2:1 in pixel scale.
"""
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


def test_a_hallucinated_joint_is_still_dropped(geo):
    """The gate's actual job: one view putting the ankle on the knee."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][int(Joint.LEFT_ANKLE)] = 0.3      # the weaker view

    dropped = validate_cross_view(data, rig)
    assert dropped == 1
    assert np.isnan(f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)]).all()
    assert not np.isnan(f.kp2d[CAM_RIGHT][int(Joint.LEFT_ANKLE)]).any()


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

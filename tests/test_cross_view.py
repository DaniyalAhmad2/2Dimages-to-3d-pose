"""The cross-view gate must reject hallucinations, not good data.

It NaNs observations out of `frame.kp2d`, so when it over-fires the result is
indistinguishable from the detector failing: gaps in the 2D views and an empty
3D preview. That is exactly what a flat 30 px tolerance did on 3072x4080 phone
captures — 0.7% of image height, which threw away 38% of a real take.
"""
import numpy as np
import pytest

from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.pipeline import (
    CalibratedRig, epipolar_threshold, triangulate_project, validate_cross_view,
)
from tests.synth import default_two_cam, project as project_points, sample_skeleton_3d
from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics


def _rig_and_project(scale=1.0, n_frames=2):
    """A consistent two-view capture; `scale` blows the image up to phone size."""
    geo = default_two_cam()
    K = geo["K"].copy() * scale
    K[2, 2] = 1.0
    size = (int(geo["size"][0] * scale), int(geo["size"][1] * scale))
    intr = Intrinsics(K=K, dist=geo["dist"], image_size=size)
    rig = CalibratedRig(intr, intr, Extrinsics(*geo["left"]), Extrinsics(*geo["right"]))

    gt = sample_skeleton_3d()
    data = ProjectData(name="cv")
    for i in range(n_frames):
        f = Frame(frame_id=f"{i:04d}")
        for cam, ext in ((CAM_LEFT, geo["left"]), (CAM_RIGHT, geo["right"])):
            f.kp2d[cam] = project_points(gt, K, geo["dist"], *ext)
            f.scores[cam] = np.full(NUM_JOINTS, 0.9)
        data.frames.append(f)
    return rig, data, gt


def test_consistent_observations_survive_at_phone_resolution():
    """Regression guard: with a flat 30 px tolerance this dropped a third of a
    real take, which read to the user as 'half the keypoints not detected'."""
    rig, data, _ = _rig_and_project(scale=4.0)      # ~4x bigger images
    dropped = validate_cross_view(data, rig)
    assert dropped == 0, f"{dropped} good observations rejected"
    assert not np.isnan(data.frames[0].kp2d[CAM_LEFT]).any()


def test_the_tolerance_scales_with_the_image():
    small, _, _ = _rig_and_project(scale=1.0)
    big, _, _ = _rig_and_project(scale=4.0)
    assert epipolar_threshold(big) > 3.0 * epipolar_threshold(small)
    # and still reproduces roughly the historical 30 px at ~1080p
    hd = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(1920, 1080))
    assert 25.0 < epipolar_threshold(CalibratedRig(hd, hd, *[Extrinsics(
        R=np.eye(3), t=np.zeros(3))] * 2)) < 40.0


def test_a_hallucinated_joint_is_still_dropped():
    """The gate's actual job: one view putting the ankle on the knee."""
    rig, data, _ = _rig_and_project(scale=4.0)
    f = data.frames[0]
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][int(Joint.LEFT_ANKLE)] = 0.3      # the weaker view

    dropped = validate_cross_view(data, rig)
    assert dropped == 1
    assert np.isnan(f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)]).all()
    assert not np.isnan(f.kp2d[CAM_RIGHT][int(Joint.LEFT_ANKLE)]).any()


def test_triangulate_reports_what_it_threw_away():
    """The count has to reach the UI, or a calibration problem masquerades as
    a detection problem with nothing to tell them apart."""
    rig, data, _ = _rig_and_project(scale=4.0)
    f = data.frames[0]
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][int(Joint.LEFT_ANKLE)] = 0.3
    assert triangulate_project(data, rig) == 1


def test_the_import_note_only_fires_when_it_matters():
    from pose3d.ui.import_dialog import _rejection_note
    assert _rejection_note(0, 10) == ""
    assert _rejection_note(5, 10) == ""                  # 3%: normal occlusion
    note = _rejection_note(60, 10)                       # 40%: something is wrong
    assert "40%" in note and "calibration" in note.lower()

"""The manual-correction path and the batch path are one computation.

Both now call `pipeline.fit_frame` with targets from
`pipeline.bone_length_targets`. Before this, `ProjectModel` kept its own
verbatim copy of the target computation plus a session cache, and the batch
path smoothed afterwards while the drag path did not — so re-placing a joint
on the pixel it already sat on moved the whole pose (median 4.92 mm, max
25.65 mm = 21.8 % of body height, on 15 of 15 joints).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS, Frame, ProjectData
from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.geometry.triangulate import triangulate_one
from pose3d.pipeline import CalibratedRig, fit_project, triangulate_project
from tests.synth import default_two_cam, project as project_points, rot_about, \
    sample_skeleton_3d

pytest.importorskip("PySide6")

from pose3d.ui.model import ProjectModel  # noqa: E402


def _take(n=6):
    """A calibrated project of `n` frames of a moving subject, reconstructed
    exactly as the app reconstructs an import."""
    geo = default_two_cam()
    intr = Intrinsics(K=geo["K"], dist=geo["dist"], image_size=geo["size"])
    rig = CalibratedRig(intr, intr, Extrinsics(*geo["left"]),
                        Extrinsics(*geo["right"]))
    gt = sample_skeleton_3d()
    centre = gt.mean(0)

    data = ProjectData(name="fit")
    for t in range(n):
        pose = (gt - centre) @ rot_about((0, 0, 1), 4.0 * t).T + centre \
            + np.array([0.0, 0.0, 0.02 * t])
        f = Frame(frame_id=f"{t:04d}",
                  images={CAM_LEFT: "left.png", CAM_RIGHT: "right.png"})
        for cam, key in ((CAM_LEFT, "left"), (CAM_RIGHT, "right")):
            xy = project_points(pose, geo["K"], geo["dist"], *geo[key])
            # NECK and PELVIS are not detected: skeleton.derive_joints makes
            # them the PIXEL midpoints of the shoulders/hips, which is not the
            # projection of the 3D midpoint. The fixture has to do the same or
            # it is not testing the pose the app actually holds.
            xy[int(Joint.NECK)] = 0.5 * (xy[int(Joint.LEFT_SHOULDER)]
                                         + xy[int(Joint.RIGHT_SHOULDER)])
            xy[int(Joint.PELVIS)] = 0.5 * (xy[int(Joint.LEFT_HIP)]
                                           + xy[int(Joint.RIGHT_HIP)])
            f.kp2d[cam] = xy
            f.scores[cam] = np.full(NUM_JOINTS, 0.9)
        data.frames.append(f)
    triangulate_project(data, rig)
    fit_project(data)
    return data, rig


def test_zero_pixel_drag_is_a_no_op():
    """Re-placing every joint on the pixel it already sits on must change
    nothing at all. Today: median 4.92 mm, max 25.65 mm, 15/15 joints."""
    data, rig = _take()
    worst = 0.0
    for idx in range(len(data.frames)):
        model = ProjectModel(data, rig)
        model.set_frame(idx)
        f = model.frame()
        before = f.fitted3d.copy()
        for cam in CAMERAS:
            for j in range(NUM_JOINTS):
                xy = f.kp2d[cam][j]
                model.set_joint_2d(cam, j, float(xy[0]), float(xy[1]))
        moved = np.linalg.norm(f.fitted3d - before, axis=1)
        worst = max(worst, float(np.nanmax(moved)))
    assert worst < 5e-5, f"{1000 * worst:.4f} mm"      # < 0.05 mm


def test_a_drag_lands_on_the_same_pose_the_batch_path_would():
    """F10 by construction: one fit function, one set of targets."""
    data, rig = _take()
    model = ProjectModel(data, rig)
    model.set_frame(2)
    f = model.frame()
    xy = f.kp2d[CAM_LEFT][int(Joint.LEFT_WRIST)]
    model.set_joint_2d(CAM_LEFT, int(Joint.LEFT_WRIST),
                       float(xy[0]) + 25.0, float(xy[1]) - 12.0)
    after_drag = f.fitted3d.copy()

    fit_project(data)                                  # the batch path

    assert np.allclose(f.fitted3d, after_drag, atol=1e-9)


def test_derived_joint_follows_a_shoulder_drag():
    """NECK is the midpoint of the shoulders, so it has to move with one.

    Today it does not: a 60 px shoulder drag leaves NECK 30 px stale in 2D and
    1.9-3.3 mm out in 3D, and the bone fit is then solved against a pose the
    user can see is contradictory.
    """
    data, rig = _take()
    model = ProjectModel(data, rig)
    model.set_frame(1)
    f = model.frame()

    for parent, derived, other in (
            (Joint.LEFT_SHOULDER, Joint.NECK, Joint.RIGHT_SHOULDER),
            (Joint.LEFT_HIP, Joint.PELVIS, Joint.RIGHT_HIP)):
        xy = f.kp2d[CAM_LEFT][int(parent)]
        model.set_joint_2d(CAM_LEFT, int(parent), float(xy[0]) + 60.0,
                           float(xy[1]))

        want2d = 0.5 * (f.kp2d[CAM_LEFT][int(parent)]
                        + f.kp2d[CAM_LEFT][int(other)])
        assert np.allclose(f.kp2d[CAM_LEFT][int(derived)], want2d), derived.name

        want3d = triangulate_one(
            want2d, f.kp2d[CAM_RIGHT][int(derived)],
            rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
            rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
        err = float(np.linalg.norm(f.pose3d[int(derived)] - want3d))
        assert err < 2e-4, f"{derived.name} 3D off by {1000 * err:.3f} mm"


def test_one_undo_reverses_the_derived_joint_too():
    data, rig = _take()
    model = ProjectModel(data, rig)
    model.set_frame(1)
    f = model.frame()
    before2d = f.kp2d[CAM_LEFT][int(Joint.NECK)].copy()
    before3d = f.fitted3d.copy()

    xy = f.kp2d[CAM_LEFT][int(Joint.LEFT_SHOULDER)]
    model.set_joint_2d(CAM_LEFT, int(Joint.LEFT_SHOULDER),
                       float(xy[0]) + 60.0, float(xy[1]))
    assert not np.allclose(f.kp2d[CAM_LEFT][int(Joint.NECK)], before2d)

    model.undo()

    assert np.allclose(f.kp2d[CAM_LEFT][int(Joint.NECK)], before2d)
    assert np.allclose(f.fitted3d, before3d, atol=1e-9)


def test_a_hand_placed_derived_joint_is_not_overwritten():
    """A correction outranks the derivation: if the user put the neck
    somewhere, dragging a shoulder must not move it back."""
    data, rig = _take()
    model = ProjectModel(data, rig)
    model.set_frame(1)
    f = model.frame()
    neck = f.kp2d[CAM_LEFT][int(Joint.NECK)]
    model.set_joint_2d(CAM_LEFT, int(Joint.NECK),
                       float(neck[0]) + 8.0, float(neck[1]) + 3.0)
    placed = f.kp2d[CAM_LEFT][int(Joint.NECK)].copy()

    xy = f.kp2d[CAM_LEFT][int(Joint.LEFT_SHOULDER)]
    model.set_joint_2d(CAM_LEFT, int(Joint.LEFT_SHOULDER),
                       float(xy[0]) + 60.0, float(xy[1]))

    assert np.allclose(f.kp2d[CAM_LEFT][int(Joint.NECK)], placed)


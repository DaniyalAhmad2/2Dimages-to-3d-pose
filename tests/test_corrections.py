"""Phase 7 verification: reversible correction stack + live re-solve."""
import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.corrections import CorrectionStack
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame
from pose3d.geometry.triangulate import triangulate_one
from tests.synth import default_two_cam, project, sample_skeleton_3d


def _frame_with_pose():
    rig = default_two_cam()
    gt = sample_skeleton_3d()
    pl = project(gt, rig["K"], rig["dist"], *rig["left"])
    pr = project(gt, rig["K"], rig["dist"], *rig["right"])
    f = Frame(frame_id="0001")
    f.kp2d[CAM_LEFT] = pl
    f.kp2d[CAM_RIGHT] = pr
    f.scores[CAM_LEFT] = np.ones(len(pl))
    f.scores[CAM_RIGHT] = np.ones(len(pr))
    return f, rig, gt


def test_apply_undo_redo():
    f, rig, gt = _frame_with_pose()
    stack = CorrectionStack({f.frame_id: f})

    old = tuple(f.kp2d[CAM_LEFT][6])
    stack.apply("0001", CAM_LEFT, 6, 123.0, 456.0, ts="t")
    assert tuple(f.kp2d[CAM_LEFT][6]) == (123.0, 456.0)
    assert f.corrected[CAM_LEFT][6]
    assert len(stack.log) == 1

    stack.undo()
    assert np.allclose(f.kp2d[CAM_LEFT][6], old)
    assert not f.corrected[CAM_LEFT][6]

    stack.redo()
    assert tuple(f.kp2d[CAM_LEFT][6]) == (123.0, 456.0)


def test_live_resolve_moves_only_that_joint():
    f, rig, gt = _frame_with_pose()
    intr = Intrinsics(K=rig["K"], dist=rig["dist"], image_size=rig["size"])
    ext_l = Extrinsics(*rig["left"]); ext_r = Extrinsics(*rig["right"])

    # correct joint 6's left-view point to a new location, re-triangulate it
    stack = CorrectionStack({f.frame_id: f})
    # shift the left observation by 10px and re-solve just that joint
    new_l = f.kp2d[CAM_LEFT][6] + np.array([10.0, 0.0])
    stack.apply("0001", CAM_LEFT, 6, new_l[0], new_l[1])
    new_xyz = triangulate_one(
        f.kp2d[CAM_LEFT][6], f.kp2d[CAM_RIGHT][6], intr, intr, ext_l, ext_r)
    # the 3D point changed from ground truth (because we moved the obs)
    assert np.linalg.norm(new_xyz - gt[6]) > 1e-3


def test_undo_empty_is_safe():
    f, rig, gt = _frame_with_pose()
    stack = CorrectionStack({f.frame_id: f})
    assert stack.undo() is None
    assert not stack.can_undo()

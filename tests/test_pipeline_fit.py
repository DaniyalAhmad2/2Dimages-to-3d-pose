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
from pose3d.detect.base import Detection
from pose3d.detect.base import KeypointDetector
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


def _with_a_one_frame_dropout(t=2, joint=int(Joint.LEFT_WRIST)):
    """A take where one joint of frame `t` is unseen in the LEFT view, so it
    cannot triangulate and the gap fill invents it from frames t-1 and t+1."""
    data, rig = _take()
    data.frames[t].kp2d[CAM_LEFT][joint] = np.nan
    data.frames[t].scores[CAM_LEFT][joint] = 0.0
    triangulate_project(data, rig)
    fit_project(data)
    assert np.isnan(data.frames[t].pose3d[joint]).all()   # the measurement
    assert data.frames[t].filled[joint]                   # the fit's input
    return data, rig


def test_a_drag_elsewhere_keeps_the_filled_joint():
    """The fill lives in the fit's INPUT, not in `pose3d`, so the live path has
    to rebuild it. Editing any other joint used to be able to drop it and move
    the whole pose."""
    t, joint = 2, int(Joint.LEFT_WRIST)
    data, rig = _with_a_one_frame_dropout(t, joint)
    model = ProjectModel(data, rig)
    model.set_frame(t)
    f = model.frame()
    before = f.fitted3d[joint].copy()

    xy = f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)]
    model.set_joint_2d(CAM_LEFT, int(Joint.LEFT_ANKLE),
                       float(xy[0]), float(xy[1]))

    assert f.filled[joint]
    assert np.isnan(f.pose3d[joint]).all()                # still no measurement
    assert np.allclose(f.fitted3d[joint], before, atol=1e-9)


def test_the_live_path_fills_the_gap_exactly_as_the_batch_path_does():
    """The same computation, gap fill included: a zero-pixel drag on the frame
    with the dropout must reproduce the batch fit bit for bit."""
    t, joint = 2, int(Joint.LEFT_WRIST)
    data, rig = _with_a_one_frame_dropout(t, joint)
    batch = data.frames[t].fitted3d.copy()

    model = ProjectModel(data, rig)
    model.set_frame(t)
    f = model.frame()
    xy = f.kp2d[CAM_RIGHT][int(Joint.HEAD)]
    model.set_joint_2d(CAM_RIGHT, int(Joint.HEAD), float(xy[0]), float(xy[1]))

    assert np.allclose(f.fitted3d, batch, atol=1e-9, equal_nan=True)


def test_placing_the_missing_point_by_hand_makes_it_a_measurement_again():
    """A correction that gives the joint back must clear the interpolated flag:
    it is an observation now, and the accuracy panel, the 3D view and the
    export all read that flag."""
    t, joint = 2, int(Joint.LEFT_WRIST)
    data, rig = _with_a_one_frame_dropout(t, joint)
    model = ProjectModel(data, rig)
    model.set_frame(t)
    f = model.frame()

    # the user puts the point back in the view that lost it (roughly where the
    # neighbouring frame had it — a hand correction, not a re-detection)
    xy = data.frames[t - 1].kp2d[CAM_LEFT][joint]
    model.set_joint_2d(CAM_LEFT, joint, float(xy[0]), float(xy[1]))

    assert not f.filled[joint]                     # an observation again
    assert np.isfinite(f.pose3d[joint]).all()
    assert np.isfinite(f.fitted3d[joint]).all()


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


# --- Phase 1b: re-detection must not discard hand corrections -------------

class _ShiftedDetector(KeypointDetector):
    """Returns every joint 40 px away from where it was — i.e. it disagrees
    with the user about all of them."""

    def __init__(self, base_xy):
        self._xy = np.asarray(base_xy, float) + 40.0

    def detect(self, image_bgr):
        return Detection(xy=self._xy.copy(),
                         scores=np.full(NUM_JOINTS, 0.5))


def test_redetect_keeps_corrections():
    """"Run Detection" used to overwrite every hand-corrected point while
    leaving its `corrected` flag True — 0 of 5 survived, and all 5 flags lied.
    """
    from pose3d.pipeline import detect_project

    data, rig = _take(n=2)
    f = data.frames[0]
    corrected = [int(Joint.HEAD), int(Joint.LEFT_WRIST), int(Joint.RIGHT_WRIST),
                 int(Joint.LEFT_ANKLE), int(Joint.RIGHT_ANKLE)]
    for j in corrected:
        f.set_kp(CAM_LEFT, j, 11.0 + j, 22.0 + j, score=1.0, corrected=True)
    placed = {j: f.kp2d[CAM_LEFT][j].copy() for j in corrected}

    detector = _ShiftedDetector(f.kp2d[CAM_LEFT])
    detector_xy = detector._xy.copy()
    detect_project(data, detector, lambda p: np.zeros((4, 4, 3), np.uint8))

    survived = sum(bool(np.allclose(f.kp2d[CAM_LEFT][j], placed[j]))
                   for j in corrected)
    assert survived == len(corrected)
    lying = sum(bool(f.corrected[CAM_LEFT][j])
                and not np.allclose(f.kp2d[CAM_LEFT][j], placed[j])
                for j in corrected)
    assert lying == 0
    # ... while everything NOT corrected did take the detector's new value
    free = [j for j in range(NUM_JOINTS) if j not in corrected]
    assert np.allclose(f.kp2d[CAM_LEFT][free], detector_xy[free])


def test_redetect_can_be_asked_to_overwrite_corrections():
    from pose3d.pipeline import detect_project

    data, _ = _take(n=1)
    f = data.frames[0]
    f.set_kp(CAM_LEFT, int(Joint.HEAD), 1.0, 2.0, score=1.0, corrected=True)
    detect_project(data, _ShiftedDetector(f.kp2d[CAM_LEFT]),
                   lambda p: np.zeros((4, 4, 3), np.uint8),
                   respect_corrections=False)
    assert not np.allclose(f.kp2d[CAM_LEFT][int(Joint.HEAD)], (1.0, 2.0))


def test_detect_project_reports_progress():
    from pose3d.pipeline import detect_project

    data, _ = _take(n=3)
    seen = []
    detect_project(data, _ShiftedDetector(data.frames[0].kp2d[CAM_LEFT]),
                   lambda p: np.zeros((4, 4, 3), np.uint8),
                   on_frame=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 3), (2, 3), (3, 3)]


# --- Phase 1b: the fix reaches an existing take, once, and says so --------

def test_a_legacy_project_is_recomputed_on_open_and_says_what_moved():
    from pose3d.core.project import PIPELINE_VERSION
    from pose3d.geometry.bonefit import smooth_temporal

    data, rig = _take()
    # pretend the previous build wrote this: fitted3d carries the causal lag
    lagged = smooth_temporal(np.stack([f.fitted3d for f in data.frames]), 0.6)
    for f, pose in zip(data.frames, lagged):
        f.fitted3d = pose.copy()
    data.pipeline_version = 0

    model = ProjectModel(data, rig)
    note = model.upgrade_pipeline()

    assert data.pipeline_version == PIPELINE_VERSION
    assert "mm median" in note and "solver v" in note
    assert not np.allclose(data.frames[3].fitted3d, lagged[3])
    # ... and opening it again does nothing
    assert model.upgrade_pipeline() == ""


def test_the_stored_pose_can_be_restored_in_session():
    from pose3d.geometry.bonefit import smooth_temporal

    data, rig = _take()
    lagged = smooth_temporal(np.stack([f.fitted3d for f in data.frames]), 0.6)
    for f, pose in zip(data.frames, lagged):
        f.fitted3d = pose.copy()
    data.pipeline_version = 0

    model = ProjectModel(data, rig)
    model.upgrade_pipeline()
    assert model.restore_stored_pose()

    for f, pose in zip(data.frames, lagged):
        assert np.allclose(f.fitted3d, pose)
    assert not model.restore_stored_pose()      # nothing left to restore


def test_a_project_with_no_calibration_is_left_alone():
    """Recomputing needs a rig. Without one the stored pose is untouched and
    the user is told why, rather than silently getting nothing.

    And it must stay UNSTAMPED: the take still carries the old pipeline's
    pose, so it still needs the migration. Stamping it on the strength of a
    note the user may never act on would, the moment they loaded a calibration
    and saved, freeze the lagged pose for good.
    """
    from pose3d.core.project import PIPELINE_VERSION

    data, rig = _take()
    stored = [f.fitted3d.copy() for f in data.frames]
    data.pipeline_version = 0

    model = ProjectModel(data, None)
    note = model.upgrade_pipeline()

    assert "calibration" in note
    for f, pose in zip(data.frames, stored):
        assert np.allclose(f.fitted3d, pose)
    assert data.pipeline_version == 0

    # ... so once the calibration is loaded, the correction still happens
    note = ProjectModel(data, rig).upgrade_pipeline()
    assert "mm median" in note
    assert data.pipeline_version == PIPELINE_VERSION


def test_restoring_the_stored_pose_takes_the_version_stamp_back_with_it():
    """Restore puts the previous build's pose back, so the file must stop
    claiming to hold the new one — otherwise saving after a Restore freezes
    the pose the client complained about and the offer never returns."""
    from pose3d.core.project import PIPELINE_VERSION
    from pose3d.geometry.bonefit import smooth_temporal

    data, rig = _take()
    lagged = smooth_temporal(np.stack([f.fitted3d for f in data.frames]), 0.6)
    for f, pose in zip(data.frames, lagged):
        f.fitted3d = pose.copy()
    data.pipeline_version = 0

    model = ProjectModel(data, rig)
    model.upgrade_pipeline()
    assert data.pipeline_version == PIPELINE_VERSION

    assert model.restore_stored_pose()

    assert data.pipeline_version == 0
    # and a fresh open of that project offers the correction again
    assert "mm median" in ProjectModel(data, rig).upgrade_pipeline()


def test_upgrade_does_nothing_to_a_current_project():
    data, rig = _take()
    stored = [f.fitted3d.copy() for f in data.frames]
    model = ProjectModel(data, rig)
    assert model.upgrade_pipeline() == ""
    for f, pose in zip(data.frames, stored):
        assert np.allclose(f.fitted3d, pose)


def test_the_recompute_banner_names_the_face_point_migration():
    """A project made before face keypoints existed cannot grow them on open —
    the recompute has no detector. So the banner that already explains the
    pose change also names the one action that fixes the head."""
    data, rig = _take()
    data.pipeline_version = 0
    assert not np.isfinite(data.frames[0].head2d[CAM_LEFT]).any()

    note = ProjectModel(data, rig).upgrade_pipeline()

    assert "Re-detect face points only" in note
    assert "mm median" in note              # still says what moved


def test_a_take_that_already_has_face_points_gets_no_hint():
    data, rig = _take()
    data.pipeline_version = 0
    for f in data.frames:
        for cam in CAMERAS:
            f.head2d[cam][0] = (10.0, 20.0)

    note = ProjectModel(data, rig).upgrade_pipeline()

    assert "Re-detect face points" not in note


def test_the_no_calibration_note_carries_the_hint_too():
    data, _ = _take()
    data.pipeline_version = 0
    note = ProjectModel(data, None).upgrade_pipeline()
    assert "calibration" in note and "Re-detect face points only" in note


def test_redetect_face_points_says_nothing_happened_when_it_did_not():
    """The migration action is the only route to face keypoints, so it must
    not report success on a build whose detector has none to give — the head
    would go on riding the neck while the status bar said it had been fixed.
    """
    data, rig = _take(n=2)
    model = ProjectModel(data, rig)
    said = []
    model.statusMessage.connect(said.append)

    # a detector with body keypoints only (Detection.head_xy defaults to None)
    model.redetect_head(_ShiftedDetector(data.frames[0].kp2d[CAM_LEFT]),
                        lambda p: np.zeros((4, 4, 3), np.uint8))

    assert "does not produce face points" in said[-1]
    assert "re-detected" not in said[-1]
    for f in data.frames:
        for cam in CAMERAS:
            assert np.isnan(f.head2d[cam]).all()

"""Face keypoints ride alongside the canonical joints, never inside them.

`NUM_JOINTS` stays 15: a dozen modules reshape to it, and the bone fit sizes
its variable vector at NUM_JOINTS*3. The face points exist only to orient the
head, so they are stored, triangulated and consumed on a parallel path.
"""
import numpy as np
import pytest

from pose3d.core.io_project import load_project, save_project
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
from pose3d.core.skeleton import (
    COCO17_INDEX, HEAD_KP_NAMES, NUM_HEAD_KP, NUM_JOINTS, extract_head,
)


def test_extract_head_reads_the_same_indices_in_both_models():
    """COCO-17 and Halpe-26 share indices 0-4, so one extractor serves both."""
    kp = np.arange(26 * 2, dtype=float).reshape(26, 2)
    sc = np.linspace(0, 1, 26)
    for n_kp in (17, 26):
        xy, s = extract_head(kp[:n_kp], sc[:n_kp])
        assert xy.shape == (NUM_HEAD_KP, 2) and s.shape == (NUM_HEAD_KP,)
        for i, name in enumerate(HEAD_KP_NAMES):
            assert np.array_equal(xy[i], kp[COCO17_INDEX[name]])


def test_extract_head_copies_rather_than_views():
    """The detector masks rejected points in place; that must not scribble on
    the caller's raw model output."""
    kp = np.zeros((17, 2)); sc = np.ones(17)
    xy, _ = extract_head(kp, sc)
    xy[0] = np.nan
    assert not np.isnan(kp).any()


def test_the_canonical_arrays_keep_their_shape():
    f = Frame(frame_id="0000")
    assert f.kp2d[CAM_LEFT].shape == (NUM_JOINTS, 2)
    assert f.pose3d.shape == (NUM_JOINTS, 3)
    assert f.head2d[CAM_LEFT].shape == (NUM_HEAD_KP, 2)
    assert f.head3d.shape == (NUM_HEAD_KP, 3)


def test_head_keypoints_round_trip(tmp_path):
    p = ProjectData(name="head")
    f = Frame(frame_id="0000")
    f.head2d[CAM_LEFT] = np.arange(NUM_HEAD_KP * 2, dtype=float).reshape(-1, 2)
    f.head_scores[CAM_RIGHT] = np.linspace(0.1, 0.9, NUM_HEAD_KP)
    f.head3d = np.arange(NUM_HEAD_KP * 3, dtype=float).reshape(-1, 3)
    p.frames.append(f)
    save_project(p, tmp_path)

    q = load_project(tmp_path)
    g = q.frames[0]
    assert np.allclose(g.head2d[CAM_LEFT], f.head2d[CAM_LEFT])
    assert np.allclose(g.head_scores[CAM_RIGHT], f.head_scores[CAM_RIGHT])
    assert np.allclose(g.head3d, f.head3d)


def test_a_project_saved_before_head_keypoints_still_loads(tmp_path):
    """The load loop indexes fd["kp2d"] directly, so the new keys had to be
    read with .get — otherwise every existing project raises KeyError."""
    import json
    p = ProjectData(name="legacy")
    p.frames.append(Frame(frame_id="0000"))
    save_project(p, tmp_path)

    doc = json.loads((tmp_path / "project.json").read_text())
    for fd in doc["frames"]:                      # strip the new keys entirely
        for k in ("head2d", "head_scores", "head3d"):
            fd.pop(k, None)
    (tmp_path / "project.json").write_text(json.dumps(doc))

    q = load_project(tmp_path)                    # must not raise
    g = q.frames[0]
    assert g.head3d.shape == (NUM_HEAD_KP, 3)
    assert np.isnan(g.head3d).all()
    assert np.isnan(g.head2d[CAM_LEFT]).all()


def test_detection_head_fields_are_optional():
    """The manual detector has no notion of a face."""
    from pose3d.detect.base import Detection
    d = Detection(xy=np.zeros((NUM_JOINTS, 2)), scores=np.ones(NUM_JOINTS))
    assert d.head_xy is None and d.head_scores is None


def test_triangulation_fills_head3d_without_touching_pose3d():
    from tests.synth import default_two_cam, project as project_points, sample_skeleton_3d
    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.calib.intrinsics import Intrinsics
    from pose3d.geometry.triangulate import fundamental_matrix
    from pose3d.pipeline import CalibratedRig, triangulate_project

    geo = default_two_cam()
    intr = Intrinsics(K=geo["K"], dist=geo["dist"], image_size=geo["size"])
    rig = CalibratedRig(intr, intr, Extrinsics(*geo["left"]), Extrinsics(*geo["right"]))

    gt = sample_skeleton_3d()
    head_gt = gt[:NUM_HEAD_KP] + np.array([0.0, 0.0, 0.3])   # any 5 known points
    data = ProjectData(name="tri")
    f = Frame(frame_id="0000")
    for cam, ext in ((CAM_LEFT, geo["left"]), (CAM_RIGHT, geo["right"])):
        f.kp2d[cam] = project_points(gt, geo["K"], geo["dist"], *ext)
        f.scores[cam] = np.full(NUM_JOINTS, 0.9)
        f.head2d[cam] = project_points(head_gt, geo["K"], geo["dist"], *ext)
    data.frames.append(f)

    dropped = triangulate_project(data, rig)
    assert data.frames[0].pose3d.shape == (NUM_JOINTS, 3)
    assert np.allclose(data.frames[0].head3d, head_gt, atol=1e-6)

    # A second frame identical to the first except that its RIGHT nose sits
    # 80 px off the epipolar line its LEFT twin defines — the occlusion
    # hallucination the canonical joints have been gated against since the
    # cross-view check existed, on a point that was never gated at all. The
    # offset runs along the line's own normal, so 80 px IS the distance.
    bad = Frame(frame_id="0001")
    for cam, ext in ((CAM_LEFT, geo["left"]), (CAM_RIGHT, geo["right"])):
        bad.kp2d[cam] = project_points(gt, geo["K"], geo["dist"], *ext)
        bad.scores[cam] = np.full(NUM_JOINTS, 0.9)
        bad.head2d[cam] = project_points(head_gt, geo["K"], geo["dist"], *ext)
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    a, b, _ = F @ np.array([*bad.head2d[CAM_LEFT][0], 1.0])
    bad.head2d[CAM_RIGHT][0] += 80.0 * np.array([a, b]) / np.hypot(a, b)
    before2d = {c: bad.head2d[c].copy() for c in (CAM_LEFT, CAM_RIGHT)}
    data.frames.append(bad)

    # the face gate is not a keypoint rejection: it writes no mask and is not
    # counted, so the number the user is told about does not move
    assert triangulate_project(data, rig) == dropped
    assert np.isnan(bad.head3d[0]).all(), "the disagreeing nose kept its 3D"
    assert np.isfinite(bad.head3d[1:]).all(), "the four agreeing points were gated too"
    assert np.allclose(data.frames[0].head3d, head_gt, atol=1e-6)
    for c in (CAM_LEFT, CAM_RIGHT):
        assert np.allclose(bad.head2d[c], before2d[c]), "the gate wrote into head2d"
        assert not np.asarray(bad.rejected[c]).any(), "the gate wrote a joint mask"


# --- editing: the loop the user actually closes -----------------------------

def _model_with_heads():
    """A calibrated two-camera project whose frames carry face keypoints."""
    from tests.synth import default_two_cam, project as project_points, sample_skeleton_3d
    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.calib.intrinsics import Intrinsics
    from pose3d.pipeline import CalibratedRig, triangulate_project
    from pose3d.ui.model import ProjectModel

    geo = default_two_cam()
    intr = Intrinsics(K=geo["K"], dist=geo["dist"], image_size=geo["size"])
    rig = CalibratedRig(intr, intr, Extrinsics(*geo["left"]), Extrinsics(*geo["right"]))
    gt = sample_skeleton_3d()
    neck = gt[8]                              # NECK (see Joint)
    head_gt = np.array([                      # nose, eyes, ears around it
        neck + [0.00, -0.12, 0.22],
        neck + [-0.03, -0.10, 0.24], neck + [0.03, -0.10, 0.24],
        neck + [-0.07, 0.00, 0.22], neck + [0.07, 0.00, 0.22]])
    data = ProjectData(name="edit")
    f = Frame(frame_id="0000")
    for cam, ext in ((CAM_LEFT, geo["left"]), (CAM_RIGHT, geo["right"])):
        f.kp2d[cam] = project_points(gt, geo["K"], geo["dist"], *ext)
        f.scores[cam] = np.full(NUM_JOINTS, 0.9)
        f.head2d[cam] = project_points(head_gt, geo["K"], geo["dist"], *ext)
        f.head_scores[cam] = np.full(NUM_HEAD_KP, 0.9)
    data.frames.append(f)
    triangulate_project(data, rig)
    return ProjectModel(data, rig)


def test_dragging_the_nose_updates_head3d_with_undo():
    from pose3d.core.project import CAM_LEFT as L

    m = _model_with_heads()
    f = m.frame()
    before3d = f.head3d.copy()
    before2d = f.head2d[L].copy()

    nose = NUM_JOINTS + 0                       # face point 0, offset convention
    x, y = f.head2d[L][0]
    # 40 px ALONG the epipolar line (this rig's lines run near-horizontal):
    # a correction the two views still agree about, so the gate keeps it
    m.set_joint_2d(L, nose, x + 40.0, y)

    assert not np.allclose(f.head2d[L][0], before2d[0])
    assert not np.allclose(f.head3d[0], before3d[0]), "head3d not re-triangulated"
    assert np.isfinite(f.head3d).all(), "an agreeing pair lost its 3D"
    assert np.allclose(f.kp2d[L], m.frame().kp2d[L]), "canonical joints touched"

    m.undo()
    assert np.allclose(f.head2d[L][0], before2d[0])
    assert np.allclose(f.head3d, before3d), "undo left the head somewhere else"


def test_dragging_a_face_point_off_the_epipolar_line_drops_its_3d():
    """The drag path is gated exactly as the recompute is.

    A drag that the two views cannot both be right about must reach the same
    `head3d` as the next full recompute would — otherwise the head is oriented
    by a point the recompute is about to delete, and the 3D view changes under
    the user for no edit they made.
    """
    from pose3d.core.project import CAM_LEFT as L

    m = _model_with_heads()
    f = m.frame()
    x, y = f.head2d[L][0]
    m.set_joint_2d(L, NUM_JOINTS + 0, x, y + 200.0)      # across the lines

    assert np.isnan(f.head3d[0]).all(), "a pair 200 px apart was triangulated"
    assert np.isfinite(f.head3d[1:]).all(), "the untouched face points were gated too"
    assert np.allclose(f.head2d[L][0], [x, y + 200.0]), "the gate wrote into head2d"


def test_dragging_the_nose_moves_the_face_nose_too():
    """The canonical HEAD dot and the nose face point are one detection; a
    drag must reach both, or the head's orientation ignores the user."""
    from pose3d.core.project import CAM_LEFT as L
    from pose3d.core.skeleton import Joint

    m = _model_with_heads()
    f = m.frame()
    nose3d_before = f.head3d[0].copy()
    x, y = f.kp2d[L][int(Joint.HEAD)]
    m.set_joint_2d(L, int(Joint.HEAD), x + 35.0, y - 10.0)

    assert np.allclose(f.head2d[L][0], f.kp2d[L][int(Joint.HEAD)])
    assert not np.allclose(f.head3d[0], nose3d_before)


def test_nose_sync_leaves_legacy_projects_alone():
    from pose3d.core.project import CAM_LEFT as L
    from pose3d.core.skeleton import Joint

    m = _model_with_heads()
    f = m.frame()
    for cam in f.head2d:                        # simulate a legacy project
        f.head2d[cam][:] = np.nan
        f.head_scores[cam][:] = np.nan
    f.head3d[:] = np.nan
    x, y = f.kp2d[L][int(Joint.HEAD)]
    m.set_joint_2d(L, int(Joint.HEAD), x + 20.0, y)
    assert np.isnan(f.head2d[L]).all(), "sync invented face points"


def _face_panel():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from pose3d.ui.camera_view import CameraPanel

    return CameraPanel("left", "LEFT CAMERA")


def _face_pose(panel, head_source, head_mode, head=None):
    """Draw one frame with all five face points and return {id: visible}."""
    xy = np.tile(np.arange(NUM_JOINTS, dtype=float)[:, None], (1, 2)) * 10 + 5
    if head is None:
        head = np.tile(
            np.arange(NUM_HEAD_KP, dtype=float)[:, None], (1, 2)) * 8 + 200
    panel.view.set_pose(xy, np.full(NUM_JOINTS, 0.9), head_xy=head,
                        head_source=head_source, head_mode=head_mode)
    return {it.joint_id: it.isVisible() for it in panel.view._face}


def test_face_items_are_drawn_and_editable():
    """All five face points exist as items; which of them the user SEES is
    decided per frame by the two conventions, and by nothing else.

    The nose dot is drawn only under the skull convention, where the nose and
    the canonical HEAD are two different detections: under the nose convention
    the HEAD dot IS the nose (`_resolve_joint` syncs the two), so a second dot
    on top of it would be the same point drawn twice, draggable to two
    different places. Eyes and ears are drawn only in Face mode, the only mode
    in which they orient anything — in Nose mode they are still detected and
    still stored, they simply do not steer the character.
    """
    from pose3d.core.skeleton import HEAD_KP_INDEX
    from pose3d.ui.camera_view import FACE_KP_IDS

    p = _face_panel()
    assert len(p.view._face) == NUM_HEAD_KP, "all five face points are items"
    assert [it.joint_id for it in p.view._face] == list(FACE_KP_IDS)
    assert FACE_KP_IDS[0] == NUM_JOINTS + HEAD_KP_INDEX["nose"]

    nose_id = NUM_JOINTS + HEAD_KP_INDEX["nose"]
    eyes_ears = [NUM_JOINTS + k for k in range(1, NUM_HEAD_KP)]

    # the full matrix: head_source decides the nose dot, head_mode the rest
    for source in ("nose", "skull"):
        for mode in ("nose", "face"):
            vis = _face_pose(p, source, mode)
            assert vis[nose_id] is (source == "skull"), (source, mode)
            for jid in eyes_ears:
                assert vis[jid] is (mode == "face"), (source, mode, jid)

    # tooltips say what each dot does, and they differ: the nose turns the
    # head in both modes, the eyes and ears only in Face mode
    tips = {it.joint_id: it.toolTip() for it in p.view._face}
    assert "turns the character's head" in tips[nose_id]
    for jid in eyes_ears:
        assert "orient the character's head (Face mode)" in tips[jid]


def test_a_face_point_without_a_detection_is_never_drawn():
    """NaN beats both conventions: a face point this view has no detection for
    is not on the image, so no mode may put a draggable dot at (nan, nan) —
    and `set_show_joints(True)`, which knows nothing about either convention,
    may not resurrect one either."""
    p = _face_panel()
    head = np.tile(np.arange(NUM_HEAD_KP, dtype=float)[:, None], (1, 2)) * 8 + 200
    head[2] = np.nan                            # a hidden eye stays hidden

    vis = _face_pose(p, "skull", "face", head)
    assert not vis[NUM_JOINTS + 2]
    assert vis[NUM_JOINTS + 1] and vis[NUM_JOINTS + 3] and vis[NUM_JOINTS + 4]

    p.view.set_show_joints(False)
    p.view.set_show_joints(True)
    vis = {it.joint_id: it.isVisible() for it in p.view._face}
    assert not vis[NUM_JOINTS + 2], "the joints toggle resurrected a NaN dot"

    # and the same toggle may not resurrect a dot the MODE hides either
    _face_pose(p, "nose", "nose", head)
    p.view.set_show_joints(True)
    vis = {it.joint_id: it.isVisible() for it in p.view._face}
    assert not any(vis.values()), "the joints toggle overrode Nose mode"


def test_frame_change_delivers_head3d_to_the_3d_view():
    """Regression guard: _on_frame_changed called set_pose without head3d, so
    the character's head snapped back to riding the neck on frame change."""
    import inspect
    from pose3d.ui import main_window
    src = inspect.getsource(main_window.MainWindow._on_frame_changed)
    assert "head3d" in src

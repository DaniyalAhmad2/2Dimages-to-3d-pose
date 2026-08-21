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

    triangulate_project(data, rig)
    assert data.frames[0].pose3d.shape == (NUM_JOINTS, 3)
    assert np.allclose(data.frames[0].head3d, head_gt, atol=1e-6)

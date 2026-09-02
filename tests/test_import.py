"""Tests for the image importer + calibration resolver (upload/ArUco/warning)."""
import cv2
import numpy as np
import pytest

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.calib.resolve import resolve_calibration
from pose3d.core.importer import build_project, match_frames
from pose3d.core.project import CAM_LEFT, CAM_RIGHT
from tests.test_extrinsics import _cam, _look_at, _render_marker


# --- importer matching -----------------------------------------------------

def test_match_numeric():
    left = ["left_0002.jpg", "left_0001.jpg", "left_0003.jpg"]
    right = ["right_0001.jpg", "right_0003.jpg", "right_0002.jpg"]
    pairs = match_frames(left, right)
    stems = [(l.stem, r.stem) for l, r in pairs]
    assert stems == [("left_0001", "right_0001"),
                     ("left_0002", "right_0002"),
                     ("left_0003", "right_0003")]


def test_match_l_r_convention():
    pairs = match_frames(["L_1.png", "L_2.png"], ["R_1.png", "R_2.png"])
    assert [(l.name, r.name) for l, r in pairs] == [("L_1.png", "R_1.png"),
                                                    ("L_2.png", "R_2.png")]


def test_match_fallback_sorted_when_no_numbers():
    pairs = match_frames(["a.jpg", "b.jpg"], ["x.jpg", "y.jpg"])
    assert [(l.name, r.name) for l, r in pairs] == [("a.jpg", "x.jpg"),
                                                    ("b.jpg", "y.jpg")]


def test_build_project_copies_images(tmp_path):
    # make dummy image files
    ld = tmp_path / "L"; rd = tmp_path / "R"; ld.mkdir(); rd.mkdir()
    left, right = [], []
    for i in (1, 2):
        lp = ld / f"left_{i}.png"; rp = rd / f"right_{i}.png"
        cv2.imwrite(str(lp), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(str(rp), np.zeros((8, 8, 3), np.uint8))
        left.append(lp); right.append(rp)
    proj = build_project(left, right, name="T", copy_into=tmp_path / "proj")
    assert len(proj.frames) == 2
    for f in proj.frames:
        assert (tmp_path / "proj" / "images").exists()
        assert f.images[CAM_LEFT].startswith(str(tmp_path / "proj"))


# --- calibration resolver --------------------------------------------------

def _two_cam_marker_project(tmp_path, marker_len=0.30, with_markers=True):
    """Build a 1-frame project whose images show (or don't) a shared ArUco tag."""
    intr = _cam()
    eye_l = np.array([-0.6, -1.3, 0.2]); eye_r = np.array([0.6, -1.3, 0.2])
    Rl, tl = _look_at(eye_l); Rr, tr = _look_at(eye_r)
    if with_markers:
        img_l = _render_marker(intr.K, intr.dist, Rl, tl, 0, marker_len, (0, 0, 0))
        img_r = _render_marker(intr.K, intr.dist, Rr, tr, 0, marker_len, (0, 0, 0))
    else:
        img_l = np.full((720, 1280), 255, np.uint8)
        img_r = np.full((720, 1280), 255, np.uint8)
    ld = tmp_path / "l.png"; rd = tmp_path / "r.png"
    cv2.imwrite(str(ld), img_l); cv2.imwrite(str(rd), img_r)
    proj = build_project([ld], [rd], name="cal")
    return proj, intr, (eye_l, eye_r)


def _loader(path):
    return cv2.imread(str(path))


def test_resolve_uses_uploaded_calibration(tmp_path):
    proj, intr, _ = _two_cam_marker_project(tmp_path, with_markers=False)
    el = Extrinsics(R=np.eye(3), t=np.zeros(3))
    er = Extrinsics(R=np.eye(3), t=np.array([1.0, 0, 0]))
    res = resolve_calibration(proj, _loader, intr_left=intr, intr_right=intr,
                              ext_left=el, ext_right=er)
    assert res.ok and res.status == "uploaded"


def test_resolve_estimates_extrinsics_from_aruco(tmp_path):
    proj, intr, (eye_l, eye_r) = _two_cam_marker_project(tmp_path, marker_len=0.30)
    res = resolve_calibration(proj, _loader, marker_length=0.30,
                              intr_left=intr, intr_right=intr)
    assert res.ok, res.message
    assert res.status == "aruco"
    # recovered camera centres should match the synthetic eyes
    assert np.linalg.norm(res.rig.ext[CAM_LEFT].camera_center - eye_l) < 0.05
    assert np.linalg.norm(res.rig.ext[CAM_RIGHT].camera_center - eye_r) < 0.05


def test_resolve_warns_when_no_calibration_and_no_aruco(tmp_path):
    proj, intr, _ = _two_cam_marker_project(tmp_path, with_markers=False)
    res = resolve_calibration(proj, _loader, intr_left=intr, intr_right=intr)
    assert not res.ok
    assert res.status == "failed"
    assert "not successful" in res.message.lower()


def test_resolve_approximates_intrinsics_when_missing(tmp_path):
    proj, intr, _ = _two_cam_marker_project(tmp_path, marker_len=0.30)
    res = resolve_calibration(proj, _loader, marker_length=0.30)  # no intrinsics
    assert res.ok and res.approximate
    assert res.status == "aruco"


def test_resolve_autodetects_6x6_on_client_images():
    """Regression: the client's real tags are DICT_6X6, not the old 4x4 default."""
    from pathlib import Path
    a, b = Path("assets/1_10.jpg"), Path("assets/2_10.jpg")
    if not (a.exists() and b.exists()):
        pytest.skip("client sample images not present")
    proj = build_project([a], [b], name="RealAruco")
    res = resolve_calibration(proj, _loader, marker_length=0.05)
    assert res.ok, res.message           # must find the shared 6x6 marker (id 14)
    assert res.status == "aruco"


def test_import_calibrate_save_reload(tmp_path):
    """Full import path: build -> ArUco calibrate -> save -> reload with rig."""
    from pose3d.calib.resolve import save_rig
    from pose3d.core.io_project import save_project
    import pose3d.app as app

    proj, intr, _ = _two_cam_marker_project(tmp_path, marker_len=0.30)
    folder = tmp_path / "proj"
    # re-point frames into a saved project folder
    proj = build_project([f.images[CAM_LEFT] for f in proj.frames],
                         [f.images[CAM_RIGHT] for f in proj.frames],
                         name="E2E", copy_into=folder)
    res = resolve_calibration(proj, _loader, marker_length=0.30,
                              intr_left=intr, intr_right=intr)
    assert res.ok
    save_rig(res.rig, folder / "calibration")
    save_project(proj, folder)

    model = app.build_model(str(folder))
    assert model.rig is not None
    assert model.project_dir == str(folder)
    assert len(model.project.frames) == len(proj.frames)


# --- head keypoints survive an import -------------------------------------

def test_import_populates_head_keypoints(tmp_path):
    """Import Images used to run its own detection loop that wrote only
    kp2d/scores, silently dropping Detection.head_xy — so every project ever
    made through the wizard was headless and the character's head just rode
    the neck. The dialog now calls pipeline.detect_project, the one loop.
    """
    from pose3d.core.skeleton import NUM_HEAD_KP, NUM_JOINTS
    from pose3d.detect.base import Detection
    from pose3d.geometry.triangulate import triangulate_points
    from pose3d.pipeline import detect_project

    proj, intr, _ = _two_cam_marker_project(tmp_path, marker_len=0.30)
    res = resolve_calibration(proj, _loader, marker_length=0.30,
                              intr_left=intr, intr_right=intr)
    assert res.ok, res.message

    gt3d = np.array([[0.0, 0.0, 0.2], [-0.03, 0.02, 0.24], [0.03, 0.02, 0.24],
                     [-0.05, 0.0, 0.22], [0.05, 0.0, 0.22]])
    from tests.test_extrinsics import _look_at
    import cv2 as _cv2

    def _project(pts, R, t):
        rvec, _ = _cv2.Rodrigues(R)
        img, _ = _cv2.projectPoints(pts, rvec, t.reshape(3, 1), intr.K, intr.dist)
        return img.reshape(-1, 2)

    rig = res.rig
    per_cam = {c: _project(gt3d, rig.ext[c].R, rig.ext[c].t)
               for c in (CAM_LEFT, CAM_RIGHT)}

    class _Det:
        def __init__(self):
            self._i = 0

        def detect(self, image_bgr):
            cam = CAM_LEFT if self._i % 2 == 0 else CAM_RIGHT
            self._i += 1
            return Detection(xy=np.zeros((NUM_JOINTS, 2)),
                             scores=np.full(NUM_JOINTS, 0.9),
                             head_xy=per_cam[cam].copy(),
                             head_scores=np.full(NUM_HEAD_KP, 0.9))

    detect_project(proj, _Det(), _loader)

    f = proj.frames[0]
    for c in (CAM_LEFT, CAM_RIGHT):
        assert np.isfinite(f.head2d[c]).all()
        assert np.isfinite(f.head_scores[c]).all()

    f.head3d = triangulate_points(
        f.head2d[CAM_LEFT], f.head2d[CAM_RIGHT],
        rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
        rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    assert np.isfinite(f.head3d).all()
    assert np.max(np.linalg.norm(f.head3d - gt3d, axis=1)) < 0.01


def test_import_dialog_calls_the_pipeline_detection_loop():
    """Regression guard for the deleted inline loop: the dialog must not grow
    its own copy of detection again."""
    import inspect
    from pose3d.ui import import_dialog
    src = inspect.getsource(import_dialog.ImportDialog._process)
    assert "detect_project(" in src
    assert "det.detect(" not in src

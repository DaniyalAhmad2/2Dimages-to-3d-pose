"""Phase 2 verification: ArUco extrinsics recover known camera geometry.

We render ArUco markers into a synthetic camera at a known pose, then check
that estimate_extrinsics recovers that pose (camera center within tolerance).
"""
import cv2
import numpy as np

from pose3d.calib.extrinsics import (
    estimate_extrinsics, marker_object_points,
)
from pose3d.calib.intrinsics import Intrinsics


def _render_marker(K, dist, R, t, marker_id, marker_len, world_center, size=(1280, 720)):
    """Render one ArUco marker (in world at world_center) into an image."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = np.full((size[1], size[0]), 255, np.uint8)
    # marker world corners (TL,TR,BR,BL) around world_center in z=0 plane
    obj = marker_object_points(marker_len) + np.asarray(world_center, np.float32)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(obj, rvec, t.reshape(3, 1), K, dist)
    proj = proj.reshape(-1, 2).astype(np.float32)
    # generate the marker bitmap and warp it onto the projected quad
    S = 400
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, S)
    src = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    Hmat = cv2.getPerspectiveTransform(src, proj)
    warped = cv2.warpPerspective(marker_img, Hmat, size, borderValue=255,
                                 flags=cv2.INTER_NEAREST)
    ones = np.full((S, S), 255, np.uint8)
    mask = cv2.warpPerspective(ones, Hmat, size, borderValue=0,
                               flags=cv2.INTER_NEAREST)
    img[mask > 0] = warped[mask > 0]
    return img


def _cam(f=900.0, w=1280, h=720):
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    return Intrinsics(K=K, dist=np.zeros((1, 5)), image_size=(w, h))


def _look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)
    return R, -R @ eye


def test_single_marker_extrinsics_recovers_center():
    intr = _cam()
    eye = np.array([0.3, -1.2, 0.25])
    R, t = _look_at(eye, (0, 0, 0))
    marker_len = 0.30   # large enough to project to ~100+ px and be detectable
    img = _render_marker(intr.K, intr.dist, R, t, marker_id=0,
                         marker_len=marker_len, world_center=(0, 0, 0))
    ext = estimate_extrinsics(img, intr, marker_length=marker_len)
    # single-marker mode: marker frame IS world; recovered center ~ eye
    assert np.linalg.norm(ext.camera_center - eye) < 0.02, ext.camera_center


def test_projection_matrix_shapes():
    intr = _cam()
    R, t = _look_at((0.4, -2.0, 0.3))
    from pose3d.calib.extrinsics import Extrinsics
    ext = Extrinsics(R=R, t=t)
    assert ext.P_normalized.shape == (3, 4)
    assert ext.projection_pixel(intr.K).shape == (3, 4)

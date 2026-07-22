"""Per-shot extrinsics from the scene ArUco tags.

Both cameras localize against the SAME ArUco tag world-frame, so they
automatically share one coordinate system. The known tag size gives real
metric scale, so triangulated bone lengths come out in real-world units.

Modern OpenCV: ArucoDetector class + solvePnP(SOLVEPNP_IPPE_SQUARE).
estimatePoseSingleMarkers is deprecated and NOT used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pose3d.calib.intrinsics import Intrinsics


@dataclass
class Extrinsics:
    """World->camera transform for one camera (world = ArUco tag frame)."""
    R: np.ndarray   # 3x3 rotation, X_cam = R @ X_world + t
    t: np.ndarray   # (3,) translation

    @property
    def P_normalized(self) -> np.ndarray:
        """3x4 projection [R|t] for use with NORMALIZED (undistorted) points."""
        return np.hstack([self.R, self.t.reshape(3, 1)])

    def projection_pixel(self, K: np.ndarray) -> np.ndarray:
        """3x4 projection K[R|t] for use with PIXEL points."""
        return K @ self.P_normalized

    @property
    def camera_center(self) -> np.ndarray:
        """Camera center in world coords: C = -R^T t."""
        return (-self.R.T @ self.t.reshape(3, 1)).ravel()


def marker_object_points(marker_length: float) -> np.ndarray:
    """3D corners of one ArUco marker in its own frame, centered at origin.

    Order MUST be TL, TR, BR, BL to match cv2 detectMarkers() corner order
    and the SOLVEPNP_IPPE_SQUARE convention. Marker lies in z=0 plane.
    """
    L = marker_length / 2.0
    return np.array([
        [-L,  L, 0.0],   # top-left
        [ L,  L, 0.0],   # top-right
        [ L, -L, 0.0],   # bottom-right
        [-L, -L, 0.0],   # bottom-left
    ], dtype=np.float32)


def make_detector(dictionary_id: int = cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()          # constructor, not create()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_markers(image: np.ndarray, detector=None):
    """Return (corners, ids) with ids as a flat list (may be empty)."""
    if detector is None:
        detector = make_detector()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    ids_flat = [] if ids is None else ids.ravel().tolist()
    return corners, ids_flat


def estimate_extrinsics(
    image: np.ndarray,
    intr: Intrinsics,
    marker_length: float,
    tag_world_positions: dict[int, np.ndarray] | None = None,
    detector=None,
) -> Extrinsics:
    """Estimate world->camera pose from the visible ArUco tags.

    Two modes:
    - Multi-tag board (tag_world_positions given): assemble object points for
      all seen tags in the shared world frame and solvePnP once (robust,
      recommended for the 4-tag backdrop).
    - Single tag (tag_world_positions None): use the first detected marker's
      own frame as the world frame via SOLVEPNP_IPPE_SQUARE.
    """
    corners, ids = detect_markers(image, detector)
    if not ids:
        raise ValueError("No ArUco markers detected; cannot estimate extrinsics")

    if tag_world_positions is None:
        # single-marker: that marker defines the world origin
        obj = marker_object_points(marker_length)
        img_pts = corners[0].reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            obj, img_pts, intr.K, intr.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            raise ValueError("solvePnP failed on single marker")
    else:
        # multi-tag: build correspondences across all recognized tags
        obj_all: list[np.ndarray] = []
        img_all: list[np.ndarray] = []
        half = marker_object_points(marker_length)  # local corners TL,TR,BR,BL
        for c, mid in zip(corners, ids):
            if mid not in tag_world_positions:
                continue
            center = np.asarray(tag_world_positions[mid], dtype=np.float32)
            obj_all.append(half + center)             # translate to world
            img_all.append(c.reshape(4, 2).astype(np.float32))
        if len(obj_all) < 1:
            raise ValueError("No known tags visible for multi-tag extrinsics")
        obj = np.concatenate(obj_all, axis=0)
        img_pts = np.concatenate(img_all, axis=0)
        flags = (cv2.SOLVEPNP_IPPE_SQUARE if len(obj_all) == 1
                 else cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, intr.K, intr.dist, flags=flags)
        if not ok:
            raise ValueError("solvePnP failed on multi-tag set")
        # refine
        rvec, tvec = cv2.solvePnPRefineLM(
            obj, img_pts, intr.K, intr.dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    return Extrinsics(R=R, t=tvec.ravel())

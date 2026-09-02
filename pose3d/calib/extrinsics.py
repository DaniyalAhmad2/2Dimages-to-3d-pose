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


@dataclass
class MarkerSolve:
    """One of IPPE_SQUARE's two solutions for a single square marker.

    A planar square seen by a pinhole camera has TWO poses that project to the
    same four corners (they differ by a flip about an axis in the marker
    plane). Only the corner-pixel noise separates them, so a solve is only
    trustworthy when the branches' reprojection errors differ a lot; `err` is
    what makes that judgeable, and it is thrown away by `cv2.solvePnP`, which
    silently returns the lower-error branch.
    """
    R: np.ndarray      # 3x3, X_cam = R @ X_marker + t
    t: np.ndarray      # (3,)
    err: float         # reprojection rms of the four corners (px)

    @property
    def extrinsics(self) -> Extrinsics:
        return Extrinsics(R=self.R, t=self.t)


def estimate_extrinsics_for_marker(
    corners, ids, target_id: int, intr: Intrinsics, marker_length: float,
) -> list[MarkerSolve]:
    """BOTH IPPE poses of one marker, best (lowest reprojection error) first.

    The marker's own frame is the world frame, so two cameras calling this with
    the same target_id end up in a shared coordinate system. Returns [] if
    target_id was not detected or the solve failed.

    Returning both branches rather than the winner is the point: on the
    client's take the ratio err[1]/err[0] is 8.7-13.8 for the tags whose pose
    is real and 1.26 for the one that is bent, and that ratio is the only
    signal that separates them. Callers score it.
    """
    ids = list(ids)
    if target_id not in ids:
        return []
    idx = ids.index(target_id)
    obj = marker_object_points(marker_length)
    img_pts = np.asarray(corners[idx], np.float64).reshape(4, 2)
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj.astype(np.float64), img_pts, intr.K, intr.dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for i in range(int(n)):
        R, _ = cv2.Rodrigues(rvecs[i])
        out.append(MarkerSolve(R=R, t=np.asarray(tvecs[i], float).ravel(),
                               err=float(np.asarray(errs[i]).ravel()[0])))
    return sorted(out, key=lambda s: s.err)


def estimate_extrinsics(
    image: np.ndarray,
    intr: Intrinsics,
    marker_length: float,
    detector=None,
) -> Extrinsics:
    """World->camera pose from the FIRST ArUco tag detected in `image`.

    That tag's own frame is the world frame. There is deliberately no
    multi-tag mode: assembling several tags into one object-point set requires
    knowing their layout, and assuming they are coplanar and identically
    rotated is wrong by up to 179 deg on a backdrop where the tags are taped by
    hand (it put the camera centre 527-1020 mm out on the client's take).
    `resolve.resolve_calibration` uses one scored tag instead.
    """
    corners, ids = detect_markers(image, detector)
    if not ids:
        raise ValueError("No ArUco markers detected; cannot estimate extrinsics")
    solves = estimate_extrinsics_for_marker(
        corners, ids, ids[0], intr, marker_length)
    if not solves:
        raise ValueError("solvePnP failed on single marker")
    return solves[0].extrinsics

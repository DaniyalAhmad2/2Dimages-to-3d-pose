"""Two-view DLT triangulation.

Consistency rule (never mix): we undistort points to NORMALIZED coordinates
and triangulate with P = [R | t] (no K). Distortion is always removed first;
triangulatePoints assumes a linear pinhole model.
"""
from __future__ import annotations

import cv2
import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics


def undistort_normalized(pts: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Undistort Nx2 pixel points to Nx2 normalized coords (P omitted)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts, intr.K, intr.dist)   # P omitted -> normalized
    return norm.reshape(-1, 2)


def triangulate_points(
    pts_left: np.ndarray, pts_right: np.ndarray,
    intr_left: Intrinsics, intr_right: Intrinsics,
    ext_left: Extrinsics, ext_right: Extrinsics,
) -> np.ndarray:
    """Triangulate matched Nx2 pixel points into Nx3 world coordinates.

    NaN in either view's row yields a NaN 3D row (joint not reconstructible).
    """
    pts_left = np.asarray(pts_left, float).reshape(-1, 2)
    pts_right = np.asarray(pts_right, float).reshape(-1, 2)
    n = pts_left.shape[0]
    out = np.full((n, 3), np.nan, dtype=float)

    valid = ~(np.isnan(pts_left).any(1) | np.isnan(pts_right).any(1))
    if not valid.any():
        return out

    nl = undistort_normalized(pts_left[valid], intr_left)
    nr = undistort_normalized(pts_right[valid], intr_right)

    P1 = ext_left.P_normalized      # [R|t], normalized points
    P2 = ext_right.P_normalized
    pts4 = cv2.triangulatePoints(P1, P2, nl.T, nr.T)   # 4xM
    xyz = (pts4[:3] / pts4[3]).T                        # Mx3
    out[valid] = xyz
    return out


def triangulate_one(
    pt_left: np.ndarray, pt_right: np.ndarray,
    intr_left: Intrinsics, intr_right: Intrinsics,
    ext_left: Extrinsics, ext_right: Extrinsics,
) -> np.ndarray:
    """Triangulate a single joint (used for live re-solve on drag)."""
    xyz = triangulate_points(
        np.asarray(pt_left).reshape(1, 2), np.asarray(pt_right).reshape(1, 2),
        intr_left, intr_right, ext_left, ext_right)
    return xyz[0]


def fundamental_matrix(intr_left, intr_right, ext_left, ext_right) -> np.ndarray:
    """Fundamental matrix F such that x_R^T F x_L = 0 for undistorted pixels.

    Built from the two cameras' calibration (world->camera R,t + K).
    """
    R_rel = ext_right.R @ ext_left.R.T
    t_rel = ext_right.t.reshape(3) - R_rel @ ext_left.t.reshape(3)
    tx = np.array([[0, -t_rel[2], t_rel[1]],
                   [t_rel[2], 0, -t_rel[0]],
                   [-t_rel[1], t_rel[0], 0]], dtype=float)
    E = tx @ R_rel
    F = np.linalg.inv(intr_right.K).T @ E @ np.linalg.inv(intr_left.K)
    return F


def epipolar_distance(pt_left, pt_right, intr_left, intr_right,
                      ext_left, ext_right) -> float:
    """Symmetric epipolar (Sampson) distance in px between matched observations.

    Points are undistorted to pixel coords first, then checked against F.
    Large distance => the two views cannot be seeing the same 3D point
    (e.g. one view hallucinated an occluded joint).
    """
    pt_left = np.asarray(pt_left, float).reshape(2)
    pt_right = np.asarray(pt_right, float).reshape(2)
    if np.isnan(pt_left).any() or np.isnan(pt_right).any():
        return float("nan")
    # undistort to pixel coords (P = K)
    ul = cv2.undistortPoints(pt_left.reshape(1, 1, 2), intr_left.K,
                             intr_left.dist, P=intr_left.K).reshape(2)
    ur = cv2.undistortPoints(pt_right.reshape(1, 1, 2), intr_right.K,
                             intr_right.dist, P=intr_right.K).reshape(2)
    F = fundamental_matrix(intr_left, intr_right, ext_left, ext_right)
    xl = np.array([ul[0], ul[1], 1.0])
    xr = np.array([ur[0], ur[1], 1.0])
    Fxl = F @ xl
    Ftxr = F.T @ xr
    num = float(xr @ Fxl) ** 2
    den = Fxl[0] ** 2 + Fxl[1] ** 2 + Ftxr[0] ** 2 + Ftxr[1] ** 2
    return float(np.sqrt(num / den)) if den > 1e-12 else float("nan")


def reprojection_error(
    xyz: np.ndarray, pts_pixel: np.ndarray,
    intr: Intrinsics, ext: Extrinsics,
) -> np.ndarray:
    """Per-point reprojection error (px) of 3D points into one view.

    NaN 3D or NaN observation -> NaN error.
    """
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    pts_pixel = np.asarray(pts_pixel, float).reshape(-1, 2)
    err = np.full(xyz.shape[0], np.nan)
    valid = ~(np.isnan(xyz).any(1) | np.isnan(pts_pixel).any(1))
    if not valid.any():
        return err
    rvec, _ = cv2.Rodrigues(ext.R)
    proj, _ = cv2.projectPoints(
        xyz[valid], rvec, ext.t.reshape(3, 1), intr.K, intr.dist)
    proj = proj.reshape(-1, 2)
    err[valid] = np.linalg.norm(proj - pts_pixel[valid], axis=1)
    return err

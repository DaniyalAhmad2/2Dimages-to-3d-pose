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

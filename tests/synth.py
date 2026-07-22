"""Synthetic two-camera rig with known ground truth.

Used across calibration/geometry tests AND as the controlled baseline for the
dataset validation. Because we know the exact 3D points and camera matrices,
we can measure triangulation error to machine precision.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_intrinsics(w=1280, h=720, f=900.0):
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1.0]], dtype=float)
    dist = np.zeros((1, 5), dtype=float)
    return K, dist, (w, h)


def look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    """Return world->camera (R, t) for a camera at `eye` looking at `target`.

    OpenCV camera convention: +z forward, +x right, +y down.
    """
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    z = target - eye
    z = z / np.linalg.norm(z)          # forward
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)          # right
    y = np.cross(z, x)                 # down
    R = np.stack([x, y, z], axis=0)    # world->camera rows
    t = -R @ eye
    return R, t


def project(P3d, K, dist, R, t):
    """Project Nx3 world points to Nx2 pixels through K[R|t] (+ distortion)."""
    P3d = np.asarray(P3d, float).reshape(-1, 3)
    rvec, _ = cv2.Rodrigues(R)
    img, _ = cv2.projectPoints(P3d, rvec, t.reshape(3, 1), K, dist)
    return img.reshape(-1, 2)


def default_two_cam(baseline_deg=80.0, radius=3.0, height=1.0):
    """A left/right camera pair ~baseline_deg apart looking at the origin."""
    half = np.radians(baseline_deg / 2.0)
    left_eye = (-radius * np.sin(half), -radius * np.cos(half), height)
    right_eye = (radius * np.sin(half), -radius * np.cos(half), height)
    K, dist, size = make_intrinsics()
    Rl, tl = look_at(left_eye, (0, 0, height))
    Rr, tr = look_at(right_eye, (0, 0, height))
    return {
        "K": K, "dist": dist, "size": size,
        "left": (Rl, tl), "right": (Rr, tr),
    }


def sample_skeleton_3d():
    """A plausible standing pose in canonical Joint order (17 joints), metres.

    Order matches pose3d.core.skeleton.Joint (incl. feet).
    """
    return np.array([
        [0.00, 0.00, 1.70],   # HEAD
        [0.00, 0.00, 1.50],   # NECK
        [-0.18, 0.00, 1.48],  # LEFT_SHOULDER
        [0.18, 0.00, 1.48],   # RIGHT_SHOULDER
        [-0.20, 0.02, 1.20],  # LEFT_ELBOW
        [0.20, 0.02, 1.20],   # RIGHT_ELBOW
        [-0.22, 0.05, 0.95],  # LEFT_WRIST
        [0.22, 0.05, 0.95],   # RIGHT_WRIST
        [0.00, 0.00, 0.95],   # PELVIS
        [-0.10, 0.00, 0.95],  # LEFT_HIP
        [0.10, 0.00, 0.95],   # RIGHT_HIP
        [-0.11, 0.02, 0.52],  # LEFT_KNEE
        [0.11, 0.02, 0.52],   # RIGHT_KNEE
        [-0.12, 0.03, 0.08],  # LEFT_ANKLE
        [0.12, 0.03, 0.08],   # RIGHT_ANKLE
        [-0.12, 0.18, 0.03],  # LEFT_FOOT (toe, forward +y)
        [0.12, 0.18, 0.03],   # RIGHT_FOOT
    ], dtype=float)

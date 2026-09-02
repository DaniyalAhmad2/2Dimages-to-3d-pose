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
    """A left/right camera pair ~baseline_deg apart looking at the origin.

    A SYMMETRIC rig: one K, one image size, 3 m away from a standing adult.
    Keep it — it is the controlled baseline the dataset validation compares
    against — but never test only on it: see `close_range_two_cam`.
    """
    half = np.radians(baseline_deg / 2.0)
    left_eye = (-radius * np.sin(half), -radius * np.cos(half), height)
    right_eye = (radius * np.sin(half), -radius * np.cos(half), height)
    K, dist, size = make_intrinsics()
    Rl, tl = look_at(left_eye, (0, 0, height))
    Rr, tr = look_at(right_eye, (0, 0, height))
    return {
        "K": K, "dist": dist, "size": size,
        "left": (Rl, tl), "right": (Rr, tr),
        "subject": sample_skeleton_3d(),
    }


def close_range_two_cam(baseline=0.56, dist=0.53, f_left=4080,
                        size_left=(3072, 4080), f_right=2048,
                        size_right=(1536, 2048), subject_height=0.12):
    """The client's rig: two phones either side of a hand-sized mannequin.

    ASYMMETRIC by exactly 2:1 — different sensor, different focal length,
    different pixel scale in each view. Every geometry test used to pass one
    `Intrinsics` for both cameras at a 3.9 m baseline and 3 m range, so a bug
    that assumes a single K, or that sizes a pixel tolerance from one image and
    applies it to the other, was structurally invisible.

    Defaults are the measured client take: 0.56 m baseline, 0.53 m range,
    3072x4080 at f=4080 left and 1536x2048 at f=2048 right.
    """
    half = baseline / 2.0
    depth = float(np.sqrt(max(dist ** 2 - half ** 2, 1e-9)))
    centre = subject_height / 2.0
    Kl, dist_l, sl = make_intrinsics(size_left[0], size_left[1], f_left)
    Kr, dist_r, sr = make_intrinsics(size_right[0], size_right[1], f_right)
    Rl, tl = look_at((-half, -depth, centre), (0, 0, centre))
    Rr, tr = look_at((half, -depth, centre), (0, 0, centre))
    return {
        "K_left": Kl, "dist_left": dist_l, "size_left": sl,
        "K_right": Kr, "dist_right": dist_r, "size_right": sr,
        "left": (Rl, tl), "right": (Rr, tr),
        "subject": sample_skeleton_3d(height=subject_height),
    }


def cameras(geo) -> dict[str, tuple]:
    """{cam: (K, dist, image_size)} for either rig shape.

    A symmetric rig reports the same triple twice, so a test can be written
    once and parametrised over both without caring which it got.
    """
    if "K" in geo:
        return {"left": (geo["K"], geo["dist"], geo["size"]),
                "right": (geo["K"], geo["dist"], geo["size"])}
    return {"left": (geo["K_left"], geo["dist_left"], geo["size_left"]),
            "right": (geo["K_right"], geo["dist_right"], geo["size_right"])}


def sample_skeleton_3d(height=1.7):
    """A plausible standing pose in canonical Joint order (15 joints), metres.

    Order matches pose3d.core.skeleton.Joint. `height` scales the whole figure
    about the ground plane, so the same pose serves both the 1.7 m human of
    the synthetic rig and the client's 12 cm mannequin.
    """
    pose = np.array([
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
    ], dtype=float)
    return pose * (height / 1.7)


def rot_about(axis, deg):
    """Rotation matrix of `deg` about `axis` (Rodrigues).

    Shared so the orientation and retarget suites cannot drift apart on what
    "rotate the subject 15 degrees" means.
    """
    a = np.radians(deg)
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)

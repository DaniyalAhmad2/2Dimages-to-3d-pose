"""Shared pose orientation helpers.

The 3D view and the Blender export must agree on how a reconstructed pose is
turned upright, otherwise the exported character faces/leans differently from
the live preview. Both import from here so there is a single source of truth.
"""
from __future__ import annotations

import numpy as np

from pose3d.core.skeleton import Joint, NUM_JOINTS


def upright_matrix(axis: int, sign: float) -> np.ndarray:
    """3x3 matrix mapping world coords to upright view coords (up-axis -> +Z).

    Guaranteed to be a proper ROTATION (det=+1): one horizontal axis is flipped
    when the naive axis-permutation would be a reflection, so the figure is
    never left/right mirrored (a raised left hand stays a left hand).
    """
    others = [i for i in range(3) if i != axis]
    perm_parity = -1.0 if axis == 1 else 1.0
    hx = sign * perm_parity
    M = np.zeros((3, 3))
    M[0, others[0]] = hx
    M[1, others[1]] = 1.0
    M[2, axis] = sign
    return M


def _frame_up(pose3d: np.ndarray, valid: np.ndarray):
    """Unit up-vector for one pose (head minus the lowest available body joint),
    or None if it can't be determined."""
    pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
    head = pose3d[int(Joint.HEAD)]
    if np.isnan(head).any():
        head = np.nanmean(pose3d[[int(Joint.NECK), int(Joint.HEAD)]], axis=0)
    ref = None
    for idxs in ([Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE],
                 [Joint.LEFT_KNEE, Joint.RIGHT_KNEE],
                 [Joint.PELVIS],
                 [Joint.LEFT_HIP, Joint.RIGHT_HIP]):
        pts = pose3d[[int(i) for i in idxs]]
        if np.isnan(pts).all():
            continue
        cand = np.nanmean(pts, axis=0)
        if not np.isnan(cand).any():
            ref = cand
            break
    if ref is None or np.isnan(head).any():
        return None
    d = head - ref
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def detect_vertical(pose3d: np.ndarray, valid: np.ndarray):
    """Find the world up-axis from head vs the lowest available body joint.

    Ankles can be dropped (occlusion gating), so fall back through
    knees -> pelvis -> hips to keep the figure upright.
    """
    pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
    up = _frame_up(pose3d, valid)
    if up is not None:
        axis = int(np.argmax(np.abs(up)))
        return axis, float(np.sign(up[axis]) or 1.0)
    vpts = pose3d[valid]
    return int(np.argmax(vpts.max(0) - vpts.min(0))), 1.0


def sequence_up(poses: np.ndarray):
    """Average unit up-vector over a whole pose sequence.

    Individual frames share whatever tilt the reconstruction's world frame has
    (e.g. a calibration board that wasn't perfectly level), plus the subject's
    own per-frame lean. Averaging cancels the (zero-mean) genuine lean and leaves
    the consistent world tilt, which `de_tilt_matrix` then removes. Returns None
    if no frame yields an up-vector.
    """
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    ups = []
    for p in poses:
        u = _frame_up(p, ~np.isnan(p).any(1))
        if u is not None:
            ups.append(u)
    if not ups:
        return None
    m = np.mean(ups, axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-9 else None


def de_tilt_matrix(up: np.ndarray) -> np.ndarray:
    """Minimal proper rotation (3x3) mapping the up-vector onto +Z.

    Rotates only in the plane containing `up` and +Z, so it removes the world's
    forward/side tilt WITHOUT spinning the figure's facing or mirroring it (a
    raised left hand stays a left hand). For an already-upright sequence this is
    ~identity.
    """
    up = np.asarray(up, float)
    up = up / (np.linalg.norm(up) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    c = float(np.dot(up, z))
    s = np.linalg.norm(v)
    if c < -0.999999:                      # pointing straight down: flip about X
        return np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

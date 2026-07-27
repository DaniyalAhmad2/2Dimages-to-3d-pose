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


def detect_vertical(pose3d: np.ndarray, valid: np.ndarray):
    """Find the world up-axis from head vs the lowest available body joint.

    Ankles can be dropped (occlusion gating), so fall back through
    knees -> pelvis -> hips to keep the figure upright.
    """
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
    if ref is not None and not np.isnan(head).any():
        diff = head - ref
        axis = int(np.argmax(np.abs(diff)))
        return axis, float(np.sign(diff[axis]) or 1.0)
    vpts = pose3d[valid]
    return int(np.argmax(vpts.max(0) - vpts.min(0))), 1.0

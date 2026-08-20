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
    """Unit up-vector for one pose (top of the body minus the lowest available
    body joint), or None if it can't be determined.

    The top reference is the NECK (shoulder midpoint — on the body axis), not
    HEAD: HEAD is the NOSE in the default pipeline, which sits forward of the
    body axis and was measured to tip the estimated up ~10 deg forward on a
    real take, leaning the whole de-tilted scene.
    """
    pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
    head = pose3d[int(Joint.NECK)]
    if np.isnan(head).any():
        head = pose3d[int(Joint.HEAD)]
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


# A pose-estimated up further than this from every world axis means the
# calibration frame is arbitrary and cannot be trusted as a gravity reference.
_AXIS_SNAP_MAX = np.radians(35.0)


def resolve_up(poses):
    """The vertical to display and export against: (unit vector, source).

    source is "axis" or "estimated" (or (None, "estimated") when no frame
    yields an up-vector at all).

    The world frame comes from the calibration markers, and when those are
    taped square — on a wall or flat on the floor — the frame's AXES are
    gravity-true even though none of them is nominally "up" (measured takes
    were 60-90 deg off +Z). So: estimate the subject's average body line,
    and if a signed world axis lies within _AXIS_SNAP_MAX of it, use that
    axis. Unlike levelling on the body line itself, this preserves the
    subject's genuine lean — per frame AND held across the whole take, single
    -frame projects included — which body-line levelling silently erased.

    The cost, accepted and visible in the UI: markers taped crooked by some
    angle tilt the scene by that angle (indistinguishable from genuine lean).
    When no axis is close the frame is treated as arbitrary and the body-line
    average itself is used, which normalises average lean — exactly the old
    behaviour.
    """
    bu = sequence_up(poses)
    if bu is None:
        return None, "estimated"
    axis = int(np.argmax(np.abs(bu)))
    snapped = np.zeros(3)
    snapped[axis] = np.sign(bu[axis]) or 1.0
    cos = float(np.clip(np.dot(bu, snapped), -1.0, 1.0))
    if np.arccos(cos) <= _AXIS_SNAP_MAX:
        return snapped, "axis"
    return bu, "estimated"


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

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

    This is not a true gravity reference — lean held across a whole take is
    normalised away, and a single-frame project is forced upright. Both
    alternatives were measured and are worse on this rig:

    * The calibration world frame is NOT gravity-aligned. Its axes come from
      whichever ArUco tag `resolve_calibration` picked, and the tags are taped
      at arbitrary rotations: three markers in one image of the client's take
      disagreed about "up" by 6, 92 and 89 degrees. Snapping to the nearest
      world axis inherited that and tilted the figure ~26 degrees forward.
    * The cameras' own up-vectors are a gravity proxy, but disagreed with the
      body line by 18 degrees on the same take.

    Both become viable once extrinsics are solved from the full tag layout
    rather than one arbitrary tag.
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


# A recorded vertical is three proxies averaged (geometry/gravity.py); their
# pairwise spread is its uncertainty and is 8-14 deg on the client's rig. Above
# this the proxies are not describing the same axis, so the subject's own body
# line is the better answer.
RECORDED_UP_MAX_SPREAD_DEG = 20.0


def take_up(poses, recorded=None):
    """The up-vector a whole take should be levelled on.

    `recorded` is (up, source, spread_deg) as written at calibration time and
    read back by `calib.resolve.load_world_up`, or None. It is preferred when
    present and confident, because it is a property of the ROOM: it keeps a
    lean held for the whole take, which `sequence_up` cannot (it averages the
    lean away by construction).

    Returns (up, source, spread_deg) — `source` is a label for the UI and
    `spread_deg` is None when the answer came from the subject. A project
    calibrated before the vertical was recorded gets exactly today's answer,
    bit for bit.
    """
    if recorded is not None:
        up, source, spread = recorded
        if up is not None and spread is not None \
                and spread <= RECORDED_UP_MAX_SPREAD_DEG:
            up = np.asarray(up, float).ravel()
            n = float(np.linalg.norm(up))
            if n > 1e-9:
                return up / n, source, float(spread)
    if poses is None:
        return None, "subject", None
    return sequence_up(poses), "subject", None


def de_tilt_matrix(up: np.ndarray) -> np.ndarray:
    """Minimal proper rotation (3x3) mapping the up-vector onto +Z.

    Rotates only in the plane containing `up` and +Z, so it removes the world's
    forward/side tilt WITHOUT spinning the figure's facing or mirroring it (a
    raised left hand stays a left hand). For an already-upright sequence this is
    ~identity.

    ONE constant rotation is applied to the whole take, whatever `up` came
    from. That is the guarantee the levelling rests on: the subject's per-frame
    lean (10.83 deg median, 21.10 deg max on the client's take) passes through
    untouched to 1.5e-11 deg, and only the take-wide tilt moves. Anything that
    made this per-frame would be flattening the performance.
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

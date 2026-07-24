"""Build a smooth, human-like body surface around a 3D skeleton.

Instead of discrete capsules/spheres (which read as a wooden dummy), we define
a soft signed-distance field: a tapered capsule per bone (thick torso, slimmer
limbs) plus a rounded head, combined with a smooth-union so the parts blend
organically. Marching cubes extracts one connected surface — a "shrink-wrapped"
body silhouette. Pure numpy + PyMCubes so it is unit-testable and Qt-free.
"""
from __future__ import annotations

import numpy as np

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint

# bone radius as a fraction of the figure height (thicker torso, slimmer limbs)
_BONE_R = {
    (Joint.PELVIS, Joint.NECK): 0.135,      # torso/spine — the bulk
    (Joint.PELVIS, Joint.LEFT_HIP): 0.10,
    (Joint.PELVIS, Joint.RIGHT_HIP): 0.10,
    (Joint.NECK, Joint.LEFT_SHOULDER): 0.075,
    (Joint.NECK, Joint.RIGHT_SHOULDER): 0.075,
    (Joint.NECK, Joint.HEAD): 0.055,        # neck
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW): 0.052,
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW): 0.052,
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST): 0.040,
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST): 0.040,
    (Joint.LEFT_HIP, Joint.LEFT_KNEE): 0.080,
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE): 0.080,
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE): 0.058,
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE): 0.058,
}
_DEFAULT_BONE_R = 0.05
# extra rounded blobs at some joints (head, hands, feet, shoulders/hips)
_JOINT_R = {
    Joint.HEAD: 0.115,       # head
    Joint.LEFT_WRIST: 0.045, Joint.RIGHT_WRIST: 0.045,
    Joint.LEFT_ANKLE: 0.05, Joint.RIGHT_ANKLE: 0.05,
    Joint.LEFT_SHOULDER: 0.075, Joint.RIGHT_SHOULDER: 0.075,
    Joint.LEFT_HIP: 0.09, Joint.RIGHT_HIP: 0.09,
    Joint.PELVIS: 0.11,
}


def _capsule_sdf(P, a, b, r):
    ab = b - a
    denom = float(ab @ ab) + 1e-9
    t = np.clip(((P - a) @ ab) / denom, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(P - proj, axis=1) - r


def _sphere_sdf(P, c, r):
    return np.linalg.norm(P - c, axis=1) - r


def _smin(a, b, k):
    """Smooth minimum (polynomial) — blends shapes organically."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def human_body_mesh(pose3d, resolution: int = 46):
    """Return (verts, faces) of a smooth body surface, or (None, None).

    pose3d: (NUM_JOINTS, 3) in world/view coords (NaN allowed). Vertices come
    back in the same coordinate space as pose3d.
    """
    import mcubes

    pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
    valid = ~np.isnan(pose3d).any(1)
    if valid.sum() < 4:
        return None, None

    vpts = pose3d[valid]
    lo = vpts.min(0); hi = vpts.max(0)
    height = float(np.linalg.norm(hi - lo)) or 1.0
    k = height * 0.06                        # blend smoothness

    # segments (bones with both ends present) + joint blobs
    segs = []
    for a, b in BONES:
        ia, ib = int(a), int(b)
        if valid[ia] and valid[ib]:
            r = _BONE_R.get((a, b), _DEFAULT_BONE_R) * height
            segs.append((pose3d[ia], pose3d[ib], r))
    blobs = []
    for j, r in _JOINT_R.items():
        if valid[int(j)]:
            blobs.append((pose3d[int(j)], r * height))
    if not segs and not blobs:
        return None, None

    pad = 0.16 * height
    lo = lo - pad; hi = hi + pad
    res = int(resolution)
    xs = np.linspace(lo[0], hi[0], res)
    ys = np.linspace(lo[1], hi[1], res)
    zs = np.linspace(lo[2], hi[2], res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    sdf = np.full(P.shape[0], 1e6)
    for a, b, r in segs:
        sdf = _smin(sdf, _capsule_sdf(P, a, b, r), k)
    for c, r in blobs:
        sdf = _smin(sdf, _sphere_sdf(P, c, r), k)

    field = (-sdf).reshape(res, res, res)    # inside > 0
    verts, faces = mcubes.marching_cubes(field, 0.0)
    if len(verts) == 0:
        return None, None
    # marching-cubes verts are in [0,res-1] index space -> back to world
    scale = (hi - lo) / (res - 1)
    verts = lo[None, :] + verts * scale[None, :]
    return verts.astype(np.float32), faces.astype(np.int32)

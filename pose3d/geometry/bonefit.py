"""Bone-length constrained fit + temporal smoothing.

Raw two-view triangulation is jittery and its bone lengths wobble frame to
frame. We enforce a fixed-length skeleton (the target rig) so the output is
usable animation rather than noise. This is the step the client explicitly
asked for (joint/bone-length constraints mapped to a Mixamo rig).

Approach: given raw 3D joints (some may be NaN), find joint positions that
(a) stay close to the observed raw positions and (b) make each bone match its
target length. Solved as a soft least-squares problem with SciPy, rooted at
the pelvis. Occluded joints (NaN) are driven purely by the bone-length term
from their parent along the previous/observed direction.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint


def measure_bone_lengths(poses3d: np.ndarray) -> dict[tuple[int, int], float]:
    """Median bone length across a sequence of 3D poses (T, NUM_JOINTS, 3)."""
    poses3d = np.asarray(poses3d, float).reshape(-1, NUM_JOINTS, 3)
    lengths: dict[tuple[int, int], float] = {}
    for a, b in BONES:
        d = poses3d[:, int(b)] - poses3d[:, int(a)]
        seg = np.linalg.norm(d, axis=1)
        seg = seg[~np.isnan(seg)]
        lengths[(int(a), int(b))] = float(np.median(seg)) if seg.size else 0.0
    return lengths


# metres; used when no measured rig is supplied
_FALLBACK_LENGTHS: dict[tuple[Joint, Joint], float] = {
    (Joint.PELVIS, Joint.NECK): 0.55,
    (Joint.NECK, Joint.HEAD): 0.20,
    (Joint.NECK, Joint.LEFT_SHOULDER): 0.18,
    (Joint.NECK, Joint.RIGHT_SHOULDER): 0.18,
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW): 0.28,
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW): 0.28,
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST): 0.25,
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST): 0.25,
    (Joint.PELVIS, Joint.LEFT_HIP): 0.10,
    (Joint.PELVIS, Joint.RIGHT_HIP): 0.10,
    (Joint.LEFT_HIP, Joint.LEFT_KNEE): 0.43,
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE): 0.43,
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE): 0.44,
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE): 0.44,
}


def fallback_bone_lengths() -> dict[tuple[int, int], float]:
    return {(int(a), int(b)): v for (a, b), v in _FALLBACK_LENGTHS.items()}


def fit_bone_lengths(
    raw3d: np.ndarray,
    bone_lengths: dict[tuple[int, int], float],
    data_weight: float = 1.0,
    bone_weight: float = 5.0,
    fill_missing: bool = True,
) -> np.ndarray:
    """Fit one frame's joints to fixed bone lengths.

    Parameters
    ----------
    raw3d : (NUM_JOINTS, 3) triangulated joints (NaN allowed for occluded).
    bone_lengths : target length per (parent, child) bone.
    data_weight : pull toward observed positions.
    bone_weight : enforce bone lengths (higher = stiffer skeleton).
    fill_missing : if True, occluded joints are placed by the bone constraints;
        if False, joints with no observation stay NaN (not invented) — used so
        a joint dropped from both/one view is genuinely absent from the 3D.

    Returns fitted (NUM_JOINTS, 3).
    """
    raw3d = np.asarray(raw3d, float).reshape(NUM_JOINTS, 3)
    observed = ~np.isnan(raw3d).any(1)

    if not observed.any():
        # Nothing was seen in this frame, so there is nothing to fit toward.
        # Solving anyway would either fail outright or, with fill_missing,
        # invent a whole skeleton out of the bone lengths alone.
        return raw3d.copy()

    # initial guess: observed points as-is; missing filled from parent + bone
    x0 = raw3d.copy()
    centroid = np.nanmean(raw3d, axis=0)
    for j in range(NUM_JOINTS):
        if not observed[j]:
            x0[j] = centroid
    x0 = np.nan_to_num(x0, nan=0.0)

    bones = [(int(a), int(b), bone_lengths[(int(a), int(b))]) for a, b in BONES]

    def residuals(x):
        pts = x.reshape(NUM_JOINTS, 3)
        res = []
        # data term (only observed joints)
        for j in range(NUM_JOINTS):
            if observed[j]:
                res.extend(data_weight * (pts[j] - raw3d[j]))
        # bone-length term
        for a, b, L in bones:
            d = np.linalg.norm(pts[b] - pts[a])
            res.append(bone_weight * (d - L))
        return np.asarray(res)

    # Levenberg-Marquardt is faster but refuses an under-determined problem:
    # it needs residuals >= variables. Variables are fixed at NUM_JOINTS*3
    # (45), residuals are 3 per OBSERVED joint plus one per bone, so lm needs
    # about 11 of the 15 joints present. Sparse frames are routine — the
    # detector is unreliable on the client's grey mannequin and
    # validate_cross_view deliberately drops inconsistent observations — and
    # scipy's ValueError used to abort the whole import. Fall back to trf,
    # which handles the under-determined case; the un-observed joints are then
    # only loosely pinned by the bone terms, which is exactly what
    # `fill_missing` already governs.
    n_res = 3 * int(observed.sum()) + len(bones)
    method = "lm" if n_res >= NUM_JOINTS * 3 else "trf"
    sol = least_squares(residuals, x0.ravel(), method=method, max_nfev=200)
    fitted = sol.x.reshape(NUM_JOINTS, 3)
    if not fill_missing:
        fitted[~observed] = np.nan     # do not invent un-observed joints
    return fitted


def smooth_temporal(poses3d: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Light causal EMA smoothing over a (T, NUM_JOINTS, 3) sequence.

    alpha weights the current frame; 1.0 = no smoothing. NaN-safe.
    """
    poses3d = np.asarray(poses3d, float).reshape(-1, NUM_JOINTS, 3)
    out = poses3d.copy()
    for t in range(1, out.shape[0]):
        prev, cur = out[t - 1], poses3d[t]
        blend = alpha * cur + (1 - alpha) * prev
        # where current is NaN keep prev; where prev is NaN keep current
        blend = np.where(np.isnan(cur), prev, blend)
        blend = np.where(np.isnan(prev), cur, blend)
        out[t] = blend
    return out

"""Sanity checks on a calibration, so a bad one is reported rather than silently
skewing every reconstruction.

Neither of the faults below stops the pipeline producing a 3D pose. They just
make it wrong in ways that look like the character is at fault: a tilted world
frame reads as the figure leaning, and guessed intrinsics skew the triangulated
skeleton asymmetrically.
"""
from __future__ import annotations

import numpy as np

# The cameras are hand-held or on tripods, so their "up" is close to gravity.
# If the calibration's Z axis disagrees with that by more than this, the target
# almost certainly was not lying flat.
_TILT_WARN_DEG = 30.0


def world_up_tilt(rig) -> float | None:
    """Angle (degrees) between the calibration's Z axis and true vertical.

    Estimated from the cameras themselves: whoever shot the take held them
    roughly upright, so their up-vectors approximate gravity.
    """
    ups = []
    for e in rig.ext.values():
        R = np.asarray(e.R, float)
        ups.append(R.T @ np.array([0.0, -1.0, 0.0]))   # camera up, in world
    if not ups:
        return None
    m = np.mean(ups, axis=0)
    n = np.linalg.norm(m)
    if n < 1e-9:
        return None
    # ±Z are equally acceptable: the target may simply be face-down
    return float(np.degrees(np.arccos(min(1.0, abs(float(m[2] / n))))))


def looks_assumed(intr) -> bool:
    """True if these intrinsics look guessed from the image size rather than
    measured: focal == the larger image dimension, principal point at the exact
    centre, and no distortion at all."""
    K = np.asarray(intr.K, float)
    dist = np.asarray(intr.dist, float).ravel()
    w, h = intr.image_size
    f_guess = float(max(w, h))
    return (np.allclose(dist, 0.0)
            and abs(K[0, 0] - f_guess) < 1.0 and abs(K[1, 1] - f_guess) < 1.0
            and abs(K[0, 2] - w / 2.0) < 1.0 and abs(K[1, 2] - h / 2.0) < 1.0)


def check_rig(rig) -> list[str]:
    """Warnings about a calibration that will distort the 3D reconstruction."""
    if rig is None:
        return []
    msgs: list[str] = []

    tilt = world_up_tilt(rig)
    if tilt is not None and tilt > _TILT_WARN_DEG:
        # Not fatal by itself: the view levels against the nearest world AXIS
        # (see pose3d.geometry.orient.resolve_up), which is gravity-true as
        # long as the markers were taped square. What still matters is that a
        # crooked board tilts the figure by exactly its own crookedness.
        msgs.append(
            f"Calibration's nominal up is {tilt:.0f}° off vertical (board on a "
            f"wall, or not flat). The view levels against the nearest world "
            f"axis instead — accurate if the markers are square; a crooked "
            f"board tilts the figure by the same amount.")

    assumed = [cam for cam, k in rig.intr.items() if looks_assumed(k)]
    if assumed:
        who = " and ".join(sorted(assumed))
        msgs.append(
            f"Camera intrinsics for {who} were assumed from the image size, not "
            f"measured. Depth and limb angles will be skewed — shoot a "
            f"checkerboard with each camera to calibrate them.")

    sizes = {cam: tuple(k.image_size) for cam, k in rig.intr.items()}
    if len(set(sizes.values())) > 1:
        pretty = ", ".join(f"{c} {w}x{h}" for c, (w, h) in sorted(sizes.items()))
        msgs.append(
            f"The two cameras differ in resolution ({pretty}), so they are "
            f"different devices or settings. Each needs its own calibration.")
    return msgs

"""What the sidebar says about a calibration, and whether it is a PROBLEM.

None of the things reported here stops the pipeline producing a 3D pose, and
most of them are not faults at all — they are properties of the kit the client
owns. Two phones of different resolutions, with no intrinsics files, is his
setup, not his mistake; a marker tag taped to a wall at an arbitrary rotation
is how the world frame is meant to be established. Reporting each of those as
a warning turned every correct import amber and told a non-technical user that
output he had no reason to distrust could not be trusted — which is how P2
("all of the detections seem to be a very low accuracy score") restarts.

So a line here is a NOTE unless it is a problem, and a note says the one thing
the user can actually do about it. A problem is a calibration that did not
work: no tag seen by both cameras, a failed solve, a gate the rig did not pass
— and those arrive from elsewhere as plain strings, which is why anything that
does not declare itself a note counts as one (`is_problem`).
"""
from __future__ import annotations

import numpy as np

from pose3d.geometry.orient import RECORDED_UP_MAX_SPREAD_DEG

# The cameras are hand-held or on tripods, so their "up" is close to gravity.
# If the calibration's Z axis disagrees with that by more than this, the target
# almost certainly was not lying flat.
_TILT_WARN_DEG = 30.0

#: The one action any of these notes implies. There is no calibration screen
#: in the app and the client owns no checkerboard: a photo set he can shoot
#: with the cameras he has is the only thing he could send.
_ACTION = "For tighter depth, send a one-time calibration photo set."


class CalibrationNote(str):
    """One line for the calibration panel, and whether it is a problem.

    A `str` subclass so that every caller that already joins, inserts, prints
    or tests these goes on working unchanged — including the window, which
    inserts its own plain-string `rig_error` into the same list, and which is
    another task's file this pass.
    """

    __slots__ = ("severity",)

    def __new__(cls, text: str, severity: str = "note"):
        self = super().__new__(cls, text)
        self.severity = severity
        return self


def is_problem(item) -> bool:
    """Is this line a problem rather than a note?

    Anything that does not declare itself a note IS one: a bare string comes
    from a caller that has not been taught the difference (the window's
    `rig_error`, a solve's failure message), and the safe reading of an
    unlabelled line about a calibration is that something went wrong.
    """
    return getattr(item, "severity", "problem") != "note"


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


def recorded_up_in_use(world_up) -> bool:
    """Is the vertical recorded at calibration the one the view actually uses?

    The same test `orient.take_up` applies — asked of that constant, not
    paraphrased — because the sidebar's job is to describe what the view and
    the export DO. Claiming a recorded vertical is in use when `take_up` has
    refused it for a 25 deg spread tells the user a lean held for the whole
    take is preserved while it is being normalised away in both.
    """
    return (world_up is not None and world_up[0] is not None
            and world_up[2] is not None
            and float(world_up[2]) <= RECORDED_UP_MAX_SPREAD_DEG)


def check_rig(rig, world_up=None) -> list[CalibrationNote]:
    """What the calibration panel should say about this rig.

    `world_up` is the recorded vertical, (up, source, spread_deg), when the
    project has one — it decides whether the world frame's tilt costs anything
    at all, so it decides whether there is anything to say.
    """
    if rig is None:
        return []
    msgs: list[CalibrationNote] = []

    tilt = world_up_tilt(rig)
    if tilt is not None and tilt > _TILT_WARN_DEG and not recorded_up_in_use(
            world_up):
        # The world frame's up comes from one ArUco tag, and tags taped at
        # different rotations define different ups, so the tilt is never on
        # its own a reason to distrust the reconstruction — which is why there
        # is nothing here at all when the view is levelling on a recorded
        # vertical. What is worth a line is what the view levels on INSTEAD,
        # and what THAT costs.
        why = ""
        if world_up is not None and world_up[0] is not None \
                and world_up[2] is not None:
            why = (f" A vertical was recorded at calibration (from the "
                   f"{world_up[1]}), but its estimates disagree by "
                   f"±{world_up[2]:.0f}° — wider than the "
                   f"{RECORDED_UP_MAX_SPREAD_DEG:.0f}° the 3D view will "
                   f"accept — so it is not being used.")
        msgs.append(CalibrationNote(
            f"The calibration's own up is {tilt:.0f}° off vertical, which is "
            f"normal: the world frame comes from one marker tag and a tag's "
            f"rotation is arbitrary. The 3D view levels on the subject "
            f"instead, so a lean held for the whole take will read as "
            f"upright.{why}"))

    assumed = [cam for cam, k in rig.intr.items() if looks_assumed(k)]
    if assumed:
        who = " and ".join(sorted(assumed))
        msgs.append(CalibrationNote(
            f"The {who} camera's lens was estimated from the photo size — no "
            f"calibration files were supplied, which is the normal way to use "
            f"this app. The pose is still reconstructed from the tags; depth "
            f"and limb angles carry a little extra error. {_ACTION}"
            if len(assumed) == 1 else
            f"The {who} cameras' lenses were estimated from the photo size — "
            f"no calibration files were supplied, which is the normal way to "
            f"use this app. The pose is still reconstructed from the tags; "
            f"depth and limb angles carry a little extra error. {_ACTION}"))

    sizes = {cam: tuple(k.image_size) for cam, k in rig.intr.items()}
    if len(set(sizes.values())) > 1:
        pretty = ", ".join(f"{c} {w}x{h}" for c, (w, h) in sorted(sizes.items()))
        msgs.append(CalibrationNote(
            f"The two cameras shot at different sizes ({pretty}), so they are "
            f"different devices or settings. Each is measured on its own, so "
            f"this is not a fault; the same one-time calibration photo set "
            f"would tighten both."))
    return msgs

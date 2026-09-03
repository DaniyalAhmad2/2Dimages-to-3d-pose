"""Load a calibrated two-camera rig from a calibration folder.

One implementation, deliberately Qt-free. `pose3d.app` and `pose3d.quality`
both used to carry their own copy of this — `app` because the UI opens a
project folder, `quality` because it must stay importable from a headless CLI
and from pytest, where importing `pose3d.app` pulls in PySide6 at module
scope. Two copies of the same file-format contract is one copy too many: the
day the folder grows a file, only one of them would learn about it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics


def check_extrinsics(cam: str, d) -> tuple:
    """(R, t) for one camera, or raise ValueError naming what is wrong.

    A 2x2 `R` used to build a perfectly happy `CalibratedRig` and only explode
    inside triangulation, frames later, as a numpy broadcast error nobody could
    read. The shape and the orthonormality are cheap and they are the two ways
    a hand-edited or externally produced extrinsics.json goes wrong.

    It lives HERE, not in the UI, because the app and every headless caller
    (`quality.load_rig`, `tools/measure_take.py`, the tests) must fail on the
    same file for the same reason. It was the app's alone, so the malformed
    file the UI explained in a sentence still exploded unreadably in the CLI.
    """
    try:
        R = np.asarray(d["R"], float)
        t = np.asarray(d["t"], float).ravel()
    except (TypeError, ValueError) as e:
        raise ValueError(f"{cam} camera: R/t are not numbers ({e})") from e
    if R.shape != (3, 3):
        raise ValueError(
            f"{cam} camera: R is {'x'.join(str(n) for n in R.shape)}, "
            f"it must be 3x3")
    if t.shape != (3,):
        raise ValueError(
            f"{cam} camera: t has {t.size} numbers, it must have 3")
    if not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError(f"{cam} camera: R or t contains NaN/inf")
    off = float(np.abs(R @ R.T - np.eye(3)).max())
    if off > 1e-3 or float(np.linalg.det(R)) < 0.0:
        raise ValueError(
            f"{cam} camera: R is not a rotation (R·Rᵀ is {off:.3g} off the "
            f"identity, det {float(np.linalg.det(R)):.3f}) — every "
            f"reconstruction from it would be skewed")
    return R, t


def load_rig(calib_dir):
    """`pipeline.CalibratedRig` from a calibration folder.

    Raises if the folder is incomplete or malformed — see `load_rig_or_none`
    for the caller that has to carry on regardless. `check_extrinsics` is what
    turns "malformed" into a sentence: the same one `app.load_rig_with_reason`
    shows, so the CLI and the app fail identically on the same file.
    """
    from pose3d.pipeline import CalibratedRig

    calib_dir = Path(calib_dir)
    il = Intrinsics.load(calib_dir / "left_intrinsics.json")
    ir = Intrinsics.load(calib_dir / "right_intrinsics.json")
    ext = json.loads(
        (calib_dir / "extrinsics.json").read_text(encoding="utf-8"))
    left = check_extrinsics("left", ext["left"])
    right = check_extrinsics("right", ext["right"])
    return CalibratedRig(il, ir, Extrinsics(R=left[0], t=left[1]),
                         Extrinsics(R=right[0], t=right[1]))


def load_rig_or_none(calib_dir):
    """The rig, or None when this project has no usable calibration yet.

    The app opens such projects on purpose: the user imports images first and
    calibrates afterwards, and the window has to come up either way.
    """
    try:
        return load_rig(calib_dir)
    except Exception:
        return None

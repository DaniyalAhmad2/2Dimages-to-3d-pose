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


def load_rig(calib_dir):
    """`pipeline.CalibratedRig` from a calibration folder.

    Raises if the folder is incomplete or malformed — see `load_rig_or_none`
    for the caller that has to carry on regardless.
    """
    from pose3d.pipeline import CalibratedRig

    calib_dir = Path(calib_dir)
    il = Intrinsics.load(calib_dir / "left_intrinsics.json")
    ir = Intrinsics.load(calib_dir / "right_intrinsics.json")
    ext = json.loads((calib_dir / "extrinsics.json").read_text())
    el = Extrinsics(R=np.array(ext["left"]["R"], float),
                    t=np.array(ext["left"]["t"], float))
    er = Extrinsics(R=np.array(ext["right"]["R"], float),
                    t=np.array(ext["right"]["t"], float))
    return CalibratedRig(il, ir, el, er)


def load_rig_or_none(calib_dir):
    """The rig, or None when this project has no usable calibration yet.

    The app opens such projects on purpose: the user imports images first and
    calibrates afterwards, and the window has to come up either way.
    """
    try:
        return load_rig(calib_dir)
    except Exception:
        return None

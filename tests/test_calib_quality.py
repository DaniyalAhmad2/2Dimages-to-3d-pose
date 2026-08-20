"""Calibration faults must be reported, not silently skew every reconstruction.

Both faults below still produce a 3D pose — a wrong one — and they look like the
character is at fault, so the app has to name them.
"""
import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.calib.quality import check_rig, looks_assumed, world_up_tilt
from pose3d.pipeline import CalibratedRig
from pose3d.core.project import CAM_LEFT, CAM_RIGHT


def _intr(w=1920, h=1080, f=None, dist=None, measured=False):
    f = float(max(w, h)) if f is None else f
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    d = np.zeros((1, 5)) if dist is None else np.asarray(dist, float).reshape(1, -1)
    return Intrinsics(K=K, dist=d, image_size=(w, h), rms=0.4 if measured else 0.0)


def _upright_cam(pos=(0.0, -3.0, 1.5)):
    """A camera standing upright in a world whose +Z really is up."""
    fwd = np.array([0.0, 1.0, 0.0])            # looking along +Y
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    R = np.stack([right, -up, fwd])            # world -> camera
    t = -R @ np.asarray(pos, float)
    return Extrinsics(R=R, t=t)


def _tipped_cam(pos=(0.0, -3.0, 1.5)):
    """The same camera, but the calibration board stood upright, so the world's
    Z axis points sideways instead of up."""
    e = _upright_cam(pos)
    # rotate the world 90 deg about X: world +Z now lies horizontal
    W = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    return Extrinsics(R=e.R @ W.T, t=e.t)


def _rig(ext, intr_l, intr_r):
    return CalibratedRig(intr_l, intr_r, ext[0], ext[1])


def test_level_calibration_reports_nothing():
    rig = _rig((_upright_cam((-0.3, -3, 1.5)), _upright_cam((0.3, -3, 1.5))),
               _intr(dist=[0.1, -0.02, 0, 0, 0], f=1500, measured=True),
               _intr(dist=[0.1, -0.02, 0, 0, 0], f=1500, measured=True))
    assert world_up_tilt(rig) < 5.0
    assert check_rig(rig) == []


def test_world_frame_not_vertical_is_reported():
    """The world frame's up comes from one arbitrarily-rotated marker tag, so
    the user has to be told the 3D view is levelling on the subject instead —
    and what that costs (a lean held all take reads as upright)."""
    rig = _rig((_tipped_cam((-0.3, -3, 1.5)), _tipped_cam((0.3, -3, 1.5))),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True))
    assert world_up_tilt(rig) > 60.0
    msgs = " ".join(check_rig(rig)).lower()
    assert "off vertical" in msgs
    assert "levels on the subject" in msgs


def test_assumed_intrinsics_are_reported():
    """Focal guessed from the image size, principal point dead centre, no
    distortion — the signature of intrinsics that were never measured."""
    assert looks_assumed(_intr())                       # f == max(w, h)
    assert not looks_assumed(_intr(f=1500, dist=[0.1, -0.02, 0, 0, 0]))
    rig = _rig((_upright_cam(), _upright_cam()), _intr(), _intr())
    msgs = " ".join(check_rig(rig)).lower()
    assert "assumed" in msgs and "checkerboard" in msgs


def test_mismatched_resolutions_are_reported():
    rig = _rig((_upright_cam(), _upright_cam()),
               _intr(3072, 4080, f=4000, dist=[0.1, 0, 0, 0, 0], measured=True),
               _intr(1536, 2048, f=2000, dist=[0.1, 0, 0, 0, 0], measured=True))
    msgs = " ".join(check_rig(rig)).lower()
    assert "resolution" in msgs


def test_no_rig_is_not_an_error():
    assert check_rig(None) == []

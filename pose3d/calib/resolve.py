"""Resolve a CalibratedRig for an imported project.

Priority, per the requirement:
  1. Use uploaded intrinsics + extrinsics when present.
  2. If extrinsics are missing, estimate them from ArUco markers shared between
     the two views (a common marker becomes the world frame).
  3. If intrinsics are missing, fall back to an approximate pinhole model from
     the image size (flagged as approximate — coplanar tags cannot recover
     intrinsics reliably).
  4. If no calibration was uploaded and no ArUco markers can be detected in any
     image pair, report failure: "calibration was not successful".
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pose3d.calib.extrinsics import (
    Extrinsics, detect_markers, estimate_extrinsics_for_marker, make_detector,
)
from pose3d.calib.intrinsics import Intrinsics, focal_from_exif
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, ProjectData
from pose3d.pipeline import CalibratedRig

# ArUco dictionaries to try, most likely first (the client's tags are 6x6).
_DICT_NAMES = [
    "DICT_6X6_250", "DICT_4X4_50", "DICT_5X5_250",
    "DICT_APRILTAG_36h11", "DICT_ARUCO_ORIGINAL",
]
DEFAULT_DICTS = [getattr(cv2.aruco, n) for n in _DICT_NAMES
                 if hasattr(cv2.aruco, n)]


def save_rig(rig: CalibratedRig, calib_dir) -> None:
    """Persist a rig to <calib_dir> in the format app._load_rig expects."""
    import json
    from pathlib import Path
    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    rig.intr[CAM_LEFT].save(calib_dir / "left_intrinsics.json")
    rig.intr[CAM_RIGHT].save(calib_dir / "right_intrinsics.json")
    (calib_dir / "extrinsics.json").write_text(json.dumps({
        "left": {"R": rig.ext[CAM_LEFT].R.tolist(),
                 "t": rig.ext[CAM_LEFT].t.tolist()},
        "right": {"R": rig.ext[CAM_RIGHT].R.tolist(),
                  "t": rig.ext[CAM_RIGHT].t.tolist()},
    }, indent=2))


def load_extrinsics_json(path):
    """Load an uploaded extrinsics file: {'left':{R,t},'right':{R,t}}."""
    import json
    from pathlib import Path
    d = json.loads(Path(path).read_text())
    return (Extrinsics(R=np.array(d["left"]["R"], float), t=np.array(d["left"]["t"], float)),
            Extrinsics(R=np.array(d["right"]["R"], float), t=np.array(d["right"]["t"], float)))


@dataclass
class CalibrationResult:
    ok: bool
    rig: CalibratedRig | None
    status: str          # 'uploaded' | 'aruco' | 'failed'
    message: str
    approximate: bool = False   # True if intrinsics were guessed from image size


def _approx_intrinsics(image: np.ndarray, path=None) -> Intrinsics:
    """Pinhole model for a camera we were never given a calibration for.

    The focal comes from the photo's EXIF when it is there, and only otherwise
    from the old f≈max(w,h) guess. That guess is badly wrong on a phone — a
    Pixel 10 Pro shooting 3072x4080 is ~2830 px, not 4080 — and an over-long
    focal warps triangulated depth, which shows up as the figure leaning by an
    amount that changes with where it stands. Principal point stays at the
    centre and distortion stays zero either way; only a checkerboard can give
    those.
    """
    h, w = image.shape[:2]
    f = focal_from_exif(path) if path is not None else None
    source = "exif" if f else "assumed"
    if not f:
        f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=float)
    return Intrinsics(K=K, dist=np.zeros((1, 5), dtype=float),
                      image_size=(w, h), source=source)


def resolve_calibration(
    project: ProjectData,
    load_image,
    marker_length: float = 0.05,
    intr_left: Intrinsics | None = None,
    intr_right: Intrinsics | None = None,
    ext_left: Extrinsics | None = None,
    ext_right: Extrinsics | None = None,
    dictionaries: list[int] | None = None,
) -> CalibrationResult:
    if not project.frames:
        return CalibrationResult(False, None, "failed", "No frames to calibrate.")

    approximate = False

    # --- intrinsics ---
    if intr_left is None or intr_right is None:
        f0 = project.frames[0]
        img_l = load_image(f0.images[CAM_LEFT])
        img_r = load_image(f0.images[CAM_RIGHT])
        if img_l is None or img_r is None:
            return CalibrationResult(False, None, "failed",
                                     "Could not read the first image pair.")
        intr_left = intr_left or _approx_intrinsics(img_l, f0.images[CAM_LEFT])
        intr_right = intr_right or _approx_intrinsics(img_r, f0.images[CAM_RIGHT])
        # "approximate" still means "no checkerboard"; EXIF is much closer but
        # still has no distortion model, so the flag stays set either way.
        approximate = True

    # --- extrinsics from upload ---
    if ext_left is not None and ext_right is not None:
        rig = CalibratedRig(intr_left, intr_right, ext_left, ext_right)
        msg = "Using uploaded calibration."
        if approximate:
            msg += " Intrinsics were approximated from image size."
        return CalibrationResult(True, rig, "uploaded", msg, approximate)

    # --- extrinsics from ArUco (find a marker seen by BOTH cameras) ---
    # The marker dictionary is auto-detected: try each candidate and use the
    # first that yields a marker common to both views (the client's tags are
    # 6x6, but this also handles 4x4/5x5/AprilTag boards).
    dictionaries = dictionaries or DEFAULT_DICTS
    detectors = [(d, make_detector(d)) for d in dictionaries]
    for frame in project.frames:
        img_l = load_image(frame.images[CAM_LEFT])
        img_r = load_image(frame.images[CAM_RIGHT])
        if img_l is None or img_r is None:
            continue
        for dict_id, detector in detectors:
            cl, idl = detect_markers(img_l, detector)
            cr, idr = detect_markers(img_r, detector)
            common = sorted(set(idl) & set(idr))
            for tid in common:
                el = estimate_extrinsics_for_marker(cl, idl, tid, intr_left, marker_length)
                er = estimate_extrinsics_for_marker(cr, idr, tid, intr_right, marker_length)
                if el is not None and er is not None:
                    rig = CalibratedRig(intr_left, intr_right, el, er)
                    msg = (f"Calibration estimated from ArUco marker {tid} "
                           f"(frame {frame.frame_id}).")
                    if approximate:
                        msg += (" Intrinsics are approximate — upload a one-time "
                                "calibration for metric accuracy.")
                    return CalibrationResult(True, rig, "aruco", msg, approximate)

    return CalibrationResult(
        False, None, "failed",
        "Calibration was not successful: no calibration was uploaded and no "
        "ArUco markers common to both cameras were detected in the images.",
        approximate)

"""One-time per-camera intrinsic calibration (checkerboard or ChArUco).

Intrinsics (K, distortion) must be estimated ONCE per physical camera and
reused across sessions. The scene ArUco tags are coplanar, which is fine for
extrinsics but insufficient for intrinsics, hence the dedicated capture.

WEBCAM GOTCHA: autofocus MUST be disabled/locked. If focus drifts between the
calibration capture and later shots, the focal length changes and this
calibration is invalidated. Each camera (L and R may differ) needs its own K.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Intrinsics:
    """Pinhole intrinsics for one camera."""
    K: np.ndarray          # 3x3 camera matrix
    dist: np.ndarray       # (1, N) distortion coeffs
    image_size: tuple[int, int]  # (width, height)
    rms: float = 0.0       # reprojection RMS from calibration (px)
    # How the focal length was obtained, worst to best:
    #   "assumed" — guessed from the image size; wrong by tens of percent
    #   "exif"    — from the camera's 35mm-equivalent focal length; close
    #   "measured"— checkerboard/ChArUco calibration; exact, with distortion
    # Drives which warning the sidebar shows, and nothing else.
    source: str = "measured"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "image_size": list(self.image_size),
            "rms": float(self.rms),
            "source": self.source,
        }, indent=2))

    @staticmethod
    def load(path: str | Path) -> "Intrinsics":
        d = json.loads(Path(path).read_text())
        return Intrinsics(
            K=np.array(d["K"], dtype=float),
            dist=np.array(d["dist"], dtype=float),
            image_size=tuple(d["image_size"]),
            rms=float(d.get("rms", 0.0)),
            # projects written before this field existed hold whatever the
            # import produced, which was always the image-size guess
            source=str(d.get("source", "measured")),
        )


# A 35mm frame's diagonal, the reference for "35mm-equivalent focal length".
_FRAME35_DIAG_MM = 43.266615


def focal_from_exif(image_path) -> float | None:
    """Focal length in PIXELS from a photo's EXIF, or None.

    Phones record `FocalLengthIn35mmFilm`: the focal a 35mm camera would need
    for the same diagonal field of view. Rescaling it by this image's diagonal
    recovers the pixel focal length, which is a far better starting point than
    guessing f = max(width, height) — for a Pixel 10 Pro (24mm-equivalent,
    3072x4080) the guess is ~45% too long, and an over-long focal warps
    triangulated depth.
    """
    _F35, _EXIF_IFD = 41989, 0x8769
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
            exif = im.getexif()
            # Real cameras put FocalLengthIn35mmFilm in the Exif sub-IFD, not
            # the top-level one getexif() returns — a Pixel's IFD0 holds only
            # make/model/orientation. Check both: PIL writes it to IFD0 when
            # round-tripping, so files produced by other tools can have it
            # there too.
            f35 = exif.get(_F35) if exif else None
            if f35 is None and exif:
                f35 = exif.get_ifd(_EXIF_IFD).get(_F35)
    except Exception:
        return None                                # unreadable/no EXIF: caller falls back
    try:
        f35 = float(f35)
    except (TypeError, ValueError):
        return None
    if not (f35 > 0) or not (w > 0 and h > 0):
        return None
    return f35 * float(np.hypot(w, h)) / _FRAME35_DIAG_MM


def calibrate_checkerboard(
    images: list[np.ndarray],
    pattern_size: tuple[int, int],
    square_size: float = 1.0,
) -> Intrinsics:
    """Calibrate one camera from checkerboard images.

    Parameters
    ----------
    images : list of BGR or gray frames of the checkerboard at varied
             depths/tilts (the "hold the board, move it around" capture).
    pattern_size : (cols, rows) of INNER corners.
    square_size : real-world square edge length (metric); scales tvecs only,
                  intrinsics are scale-independent.
    """
    cols, rows = pattern_size
    # object points: (0,0,0),(1,0,0),... scaled by square_size, z=0
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners)

    if len(objpoints) < 3:
        raise ValueError(
            f"Need >=3 valid checkerboard views, got {len(objpoints)}")

    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    return Intrinsics(K=K, dist=dist, image_size=image_size, rms=float(rms),
                      source="measured")

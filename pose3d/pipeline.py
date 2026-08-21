"""End-to-end pose reconstruction pipeline over a project.

Ties the layers together: detect per view -> triangulate -> bone-length fit ->
temporal smoothing. Kept independent of Qt so it runs headless (dataset
validation, CLI, tests) and is called by the UI's recompute.
"""
from __future__ import annotations

import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, ProjectData
from pose3d.core.skeleton import NUM_JOINTS
from pose3d.detect.base import KeypointDetector
from pose3d.geometry.bonefit import (
    fallback_bone_lengths, fit_bone_lengths, measure_bone_lengths,
    smooth_temporal,
)
from pose3d.geometry.triangulate import epipolar_distance, triangulate_points


class CalibratedRig:
    """Intrinsics + extrinsics for the two-camera rig."""

    def __init__(self, intr_l: Intrinsics, intr_r: Intrinsics,
                 ext_l: Extrinsics, ext_r: Extrinsics):
        self.intr = {CAM_LEFT: intr_l, CAM_RIGHT: intr_r}
        self.ext = {CAM_LEFT: ext_l, CAM_RIGHT: ext_r}


def detect_project(project: ProjectData, detector: KeypointDetector,
                   load_image) -> None:
    """Populate each frame's 2D keypoints/scores via the detector.

    load_image(path) -> BGR ndarray. Mutates project in place.
    """
    for frame in project.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            img = load_image(frame.images[cam])
            det = detector.detect(img)
            frame.kp2d[cam] = det.xy
            frame.scores[cam] = det.scores
            if det.head_xy is not None:
                frame.head2d[cam] = det.head_xy
                frame.head_scores[cam] = det.head_scores


# Epipolar tolerance as a fraction of the image DIAGONAL. A flat 30 px was
# implicitly tuned for ~1080p; on the client's 3072x4080 phone captures that
# is 0.7% of image height, tight enough to cut through the middle of a
# legitimate distribution and silently delete a third of every pose. This
# reproduces ~30 px at 1080p and scales with the sensor.
_EPI_THR_FRAC = 0.014


def epipolar_threshold(rig: CalibratedRig) -> float:
    """Pixels of epipolar disagreement tolerated, scaled to the image size."""
    w, h = rig.intr[CAM_LEFT].image_size
    return float(_EPI_THR_FRAC * np.hypot(w, h))


def validate_cross_view(project: ProjectData, rig: CalibratedRig,
                        epi_thr: float | None = None) -> int:
    """Drop 2D observations that are geometrically inconsistent across views.

    When a joint is occluded/out-of-frame in one camera, the detector often
    hallucinates it (e.g. an ankle collapsed onto the knee). Such a point can
    never correspond to the same 3D location the other camera sees, so its
    epipolar distance is large. For every joint present in BOTH views, if the
    epipolar distance exceeds `epi_thr` px, the observation in the LOWER-
    confidence view is dropped (set to NaN) — so it is neither drawn nor
    triangulated (the 3D point then drops out too, since a joint needs both
    views). User-corrected joints are trusted and never auto-dropped.

    Returns the number of observations dropped.
    """
    if epi_thr is None:
        epi_thr = epipolar_threshold(rig)
    dropped = 0
    for frame in project.frames:
        for j in range(NUM_JOINTS):
            pl = frame.kp2d[CAM_LEFT][j]
            pr = frame.kp2d[CAM_RIGHT][j]
            if np.isnan(pl).any() or np.isnan(pr).any():
                continue
            e = epipolar_distance(pl, pr, rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                                  rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
            if np.isnan(e) or e <= epi_thr:
                continue
            sl = frame.scores[CAM_LEFT][j]
            sr = frame.scores[CAM_RIGHT][j]
            # drop the worse (lower-confidence) view, unless it was hand-corrected
            drop_left = np.nan_to_num(sl) <= np.nan_to_num(sr)
            cam = CAM_LEFT if drop_left else CAM_RIGHT
            if frame.corrected[cam][j]:
                cam = CAM_RIGHT if drop_left else CAM_LEFT  # try the other view
                if frame.corrected[cam][j]:
                    continue                                 # both corrected: keep
            frame.kp2d[cam][j] = np.nan
            frame.scores[cam][j] = 0.0
            dropped += 1
    return dropped


# Above this share of keypoints rejected, the cause is the calibration rather
# than the detector, and the user has no way to tell those apart from the
# symptoms (gaps in the 2D views, a sparse 3D pose).
_REJECT_NOTE_FRAC = 0.15


def rejection_note(dropped: int, n_frames: int) -> str:
    """One sentence for the user about keypoints the cross-view check threw
    away, or "" when the amount is unremarkable."""
    total = max(1, n_frames * NUM_JOINTS)
    frac = dropped / total
    if frac < _REJECT_NOTE_FRAC:
        return ""
    return (f"{dropped} keypoints ({frac:.0%}) were rejected because the two "
            f"views disagree about where they are. The detector found them; "
            f"the calibration is what says they cannot both be right. Expect "
            f"gaps in the 2D views and a sparse 3D pose — recalibrating with "
            f"the markers clearly visible in both cameras is what fixes it.")


def triangulate_project(project: ProjectData, rig: CalibratedRig,
                        validate: bool = True) -> int:
    """Fill each frame's raw pose3d from its two 2D views.

    By default first drops cross-view-inconsistent observations (occlusion
    hallucinations) so they don't corrupt the 3D pose. Returns how many were
    dropped, so callers can tell the user — a bad calibration rejects good
    detections wholesale, which looks exactly like a detection failure.
    """
    dropped = 0
    if validate:
        dropped = validate_cross_view(project, rig)
    for frame in project.frames:
        frame.pose3d = triangulate_points(
            frame.kp2d[CAM_LEFT], frame.kp2d[CAM_RIGHT],
            rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
            rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
        # the face points ride the same geometry; they are not cross-view
        # validated or bone-fitted, they only orient the head
        frame.head3d = triangulate_points(
            frame.head2d[CAM_LEFT], frame.head2d[CAM_RIGHT],
            rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
            rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    return dropped


def fit_project(project: ProjectData, bone_lengths=None,
                smooth: bool = True, alpha: float = 0.6) -> None:
    """Bone-length fit every frame, then optional temporal smoothing."""
    raw = np.stack([f.pose3d for f in project.frames]) \
        if project.frames else np.zeros((0, NUM_JOINTS, 3))
    if bone_lengths is None:
        measured = measure_bone_lengths(raw)
        # if a bone was never observed, fall back to a default proportion
        fb = fallback_bone_lengths()
        bone_lengths = {k: (v if v > 1e-6 else fb[k]) for k, v in measured.items()}

    # One awkward frame must never lose the whole take: fall back to its raw
    # triangulation and carry on. The import dialog wraps this in a blanket
    # except, so anything raised here used to surface as "Import failed" with
    # every other frame's work discarded.
    per_frame, failed, first_error = [], 0, None
    for f in project.frames:
        try:
            per_frame.append(
                fit_bone_lengths(f.pose3d, bone_lengths, fill_missing=False))
        except Exception as e:
            # A bad calibration makes this systematic, not sporadic — printing
            # a traceback per frame would bury the log in hundreds of copies.
            failed += 1
            first_error = first_error or f"{type(e).__name__}: {e}"
            per_frame.append(np.asarray(f.pose3d, float))
    if failed:
        print(f"bone fit fell back to the raw triangulation on {failed}/"
              f"{len(project.frames)} frames; first was {first_error}")
    fitted = np.stack(per_frame) if per_frame else raw
    if smooth and len(fitted) > 1:
        fitted = smooth_temporal(fitted, alpha=alpha)
    for f, pose in zip(project.frames, fitted):
        f.fitted3d = pose


def run_full(project: ProjectData, detector: KeypointDetector,
             rig: CalibratedRig, load_image, smooth: bool = True) -> None:
    """Detect -> triangulate -> fit for the whole project."""
    detect_project(project, detector, load_image)
    triangulate_project(project, rig)
    fit_project(project, smooth=smooth)

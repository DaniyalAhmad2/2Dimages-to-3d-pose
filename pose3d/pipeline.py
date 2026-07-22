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


def validate_cross_view(project: ProjectData, rig: CalibratedRig,
                        epi_thr: float = 30.0) -> int:
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


def triangulate_project(project: ProjectData, rig: CalibratedRig,
                        validate: bool = True) -> None:
    """Fill each frame's raw pose3d from its two 2D views.

    By default first drops cross-view-inconsistent observations (occlusion
    hallucinations) so they don't corrupt the 3D pose.
    """
    if validate:
        validate_cross_view(project, rig)
    for frame in project.frames:
        frame.pose3d = triangulate_points(
            frame.kp2d[CAM_LEFT], frame.kp2d[CAM_RIGHT],
            rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
            rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])


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

    fitted = np.stack([
        fit_bone_lengths(f.pose3d, bone_lengths, fill_missing=False)
        for f in project.frames]) if project.frames else raw
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

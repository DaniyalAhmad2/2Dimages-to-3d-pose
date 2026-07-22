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
from pose3d.geometry.triangulate import triangulate_points


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


def triangulate_project(project: ProjectData, rig: CalibratedRig) -> None:
    """Fill each frame's raw pose3d from its two 2D views."""
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
        fit_bone_lengths(f.pose3d, bone_lengths) for f in project.frames]) \
        if project.frames else raw
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

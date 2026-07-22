"""Qt signal hub wrapping the pure ProjectData + pipeline.

Panels connect to this; a joint drag flows:
    JointItem.itemChange -> set_joint_2d -> re-triangulate that joint ->
    re-fit -> emit pose3dChanged/accuracyChanged -> 3D + accuracy panels redraw.
Panels never call each other directly.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

from pose3d.core.corrections import CorrectionStack
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS, ProjectData
from pose3d.core.skeleton import NUM_JOINTS
from pose3d.geometry.bonefit import (
    fallback_bone_lengths, fit_bone_lengths, measure_bone_lengths,
)
from pose3d.geometry.triangulate import reprojection_error, triangulate_one
from pose3d.pipeline import CalibratedRig


class ProjectModel(QObject):
    frameChanged = Signal(int)                 # current frame index
    joint2dChanged = Signal(str, int)          # cam, joint (after edit)
    pose3dChanged = Signal(object)             # (NUM_JOINTS,3) fitted pose
    accuracyChanged = Signal(object)           # (NUM_JOINTS,) reproj error px
    historyChanged = Signal()                  # undo/redo availability

    statusMessage = Signal(str)                # user-facing status text

    def __init__(self, project: ProjectData, rig: CalibratedRig | None = None,
                 project_dir=None):
        super().__init__()
        self.project = project
        self.rig = rig
        self.project_dir = project_dir
        self.current = 0
        self.auto_recalc = True
        self.stack = CorrectionStack({f.frame_id: f for f in project.frames})
        self._bone_lengths = None

    # --- whole-project recompute / detection / save ---
    def recompute_all(self) -> None:
        """Re-triangulate + re-fit every frame from the current 2D points."""
        if self.rig is None:
            self.statusMessage.emit("No calibration loaded — cannot recompute 3D")
            return
        from pose3d.pipeline import fit_project, triangulate_project
        triangulate_project(self.project, self.rig)
        self._bone_lengths = None
        fit_project(self.project, smooth=True)
        self.set_frame(self.current)
        self.statusMessage.emit(
            f"Recalculated 3D for {len(self.project.frames)} frames")

    def redetect_all(self, detector, load_image) -> None:
        """Re-run the detector on every frame, then recompute 3D."""
        if detector is None:
            self.statusMessage.emit("No detector available in this build")
            return
        from pose3d.pipeline import detect_project
        self.statusMessage.emit("Running detection…")
        detect_project(self.project, detector, load_image)
        self.recompute_all()
        self.statusMessage.emit(
            f"Detection complete ({len(self.project.frames)} frames)")

    def save(self) -> None:
        from pose3d.core.io_project import save_project
        self.project.corrections = list(self.stack.log)
        if not self.project_dir:
            self.statusMessage.emit("No project folder set — use Save As")
            return
        save_project(self.project, self.project_dir)
        self.statusMessage.emit(
            f"Saved {len(self.project.corrections)} corrections to "
            f"{self.project_dir}")

    # --- navigation ---
    def set_frame(self, idx: int) -> None:
        idx = max(0, min(idx, len(self.project.frames) - 1))
        self.current = idx
        self.frameChanged.emit(idx)
        f = self.frame()
        self.pose3dChanged.emit(f.fitted3d)
        self.accuracyChanged.emit(self._accuracy(idx))

    def frame(self):
        return self.project.frames[self.current]

    # --- editing ---
    def set_joint_2d(self, cam: str, joint: int, x: float, y: float) -> None:
        f = self.frame()
        self.stack.apply(f.frame_id, cam, joint, x, y)
        self.joint2dChanged.emit(cam, joint)
        if self.auto_recalc:
            self._resolve_joint(joint)
        self.historyChanged.emit()

    def undo(self) -> None:
        e = self.stack.undo()
        if e is not None:
            self._resolve_joint(e.joint)
            self.joint2dChanged.emit(e.cam, e.joint)
        self.historyChanged.emit()

    def redo(self) -> None:
        e = self.stack.redo()
        if e is not None:
            self._resolve_joint(e.joint)
            self.joint2dChanged.emit(e.cam, e.joint)
        self.historyChanged.emit()

    # --- geometry ---
    def _resolve_joint(self, joint: int) -> None:
        if self.rig is None:
            return
        f = self.frame()
        pl = f.kp2d[CAM_LEFT][joint]
        pr = f.kp2d[CAM_RIGHT][joint]
        xyz = triangulate_one(
            pl, pr, self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
            self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])
        f.pose3d[joint] = xyz
        # re-fit this frame to keep bone lengths consistent
        if self._bone_lengths is None:
            self._compute_bone_lengths()
        f.fitted3d = fit_bone_lengths(f.pose3d, self._bone_lengths)
        self.pose3dChanged.emit(f.fitted3d)
        self.accuracyChanged.emit(self._accuracy(self.current))

    def _compute_bone_lengths(self):
        raw = np.stack([f.pose3d for f in self.project.frames])
        measured = measure_bone_lengths(raw)
        fb = fallback_bone_lengths()
        self._bone_lengths = {
            k: (v if v > 1e-6 else fb[k]) for k, v in measured.items()}

    def _accuracy(self, idx: int) -> np.ndarray:
        f = self.project.frames[idx]
        if self.rig is None:
            return np.full(NUM_JOINTS, np.nan)
        errs = []
        for cam in CAMERAS:
            errs.append(reprojection_error(
                f.pose3d, f.kp2d[cam], self.rig.intr[cam], self.rig.ext[cam]))
        return np.nanmean(np.stack(errs), axis=0)

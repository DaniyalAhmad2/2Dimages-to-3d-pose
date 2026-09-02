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
from pose3d.core.skeleton import Joint, NUM_JOINTS
from pose3d.geometry.triangulate import reprojection_error, triangulate_one
from pose3d.pipeline import CalibratedRig, bone_length_targets, fit_frame

HEAD_JOINT = int(Joint.HEAD)

# Joints the detector does not see but derives as a midpoint of two it does
# (skeleton.derive_joints). Dragging a shoulder therefore has to move the neck
# with it, or the pose keeps a neck the user can see is in the wrong place and
# the bone fit is solved against a contradiction.
_DERIVED_FROM: dict[int, tuple[int, int]] = {
    int(Joint.NECK): (int(Joint.LEFT_SHOULDER), int(Joint.RIGHT_SHOULDER)),
    int(Joint.PELVIS): (int(Joint.LEFT_HIP), int(Joint.RIGHT_HIP)),
}
_DERIVED_OF: dict[int, int] = {
    parent: derived
    for derived, parents in _DERIVED_FROM.items() for parent in parents
}


class ProjectModel(QObject):
    frameChanged = Signal(int)                 # current frame index
    joint2dChanged = Signal(str, int)          # cam, joint (after edit)
    pose3dChanged = Signal(object, object, object)   # fitted pose, face kp,
                                                    # per-joint filled flags
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

    # --- whole-project recompute / detection / save ---
    def recompute_all(self) -> None:
        """Re-triangulate + re-fit every frame from the current 2D points."""
        if self.rig is None:
            self.statusMessage.emit("No calibration loaded — cannot recompute 3D")
            return
        from pose3d.pipeline import (
            fit_project, rejection_note, triangulate_project)
        dropped = triangulate_project(self.project, self.rig)
        smoothing = self.project.smoothing
        report = fit_project(self.project, smooth=smoothing != "none",
                             alpha=_smoothing_alpha(smoothing))
        self.set_frame(self.current)
        msg = f"Recalculated 3D for {len(self.project.frames)} frames"
        notes = [n for n in (rejection_note(dropped, len(self.project.frames)),
                             report.note()) if n]
        self.statusMessage.emit(" — ".join([msg, *notes]))

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
        self.pose3dChanged.emit(f.fitted3d, f.head3d, f.filled)
        self.accuracyChanged.emit(self._accuracy(idx))

    def frame(self):
        return self.project.frames[self.current]

    # --- editing ---
    def set_joint_2d(self, cam: str, joint: int, x: float, y: float) -> None:
        f = self.frame()
        self.stack.apply(f.frame_id, cam, joint, x, y)
        self.joint2dChanged.emit(cam, joint)
        if self.auto_recalc:
            self._resolve_joint(joint, cam)
        self.historyChanged.emit()

    def undo(self) -> None:
        e = self.stack.undo()
        if e is not None:
            self._resolve_joint(e.joint, e.cam)
            self.joint2dChanged.emit(e.cam, e.joint)
        self.historyChanged.emit()

    def redo(self) -> None:
        e = self.stack.redo()
        if e is not None:
            self._resolve_joint(e.joint, e.cam)
            self.joint2dChanged.emit(e.cam, e.joint)
        self.historyChanged.emit()

    # --- geometry ---
    def _resolve_joint(self, joint: int, cam: str | None = None) -> None:
        """Re-triangulate one edited point and re-fit this frame.

        `joint >= NUM_JOINTS` addresses face keypoint `joint - NUM_JOINTS`
        (the camera views and the correction stack share this convention).
        `cam` is the view whose 2D was edited; derived joints are re-derived
        there. None means "in every view", used for a plain re-solve.
        """
        if self.rig is None:
            return
        f = self.frame()
        if joint >= NUM_JOINTS:
            k = joint - NUM_JOINTS
            f.head3d[k] = triangulate_one(
                f.head2d[CAM_LEFT][k], f.head2d[CAM_RIGHT][k],
                self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
                self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])
            # face points have no bones: no re-fit, just re-orient the head
            self.pose3dChanged.emit(f.fitted3d, f.head3d, f.filled)
            self.accuracyChanged.emit(self._accuracy(self.current))
            return
        if joint == HEAD_JOINT:
            # The canonical HEAD and the nose face point are the same physical
            # detection, so a nose drag must move both — otherwise the head's
            # orientation (built from nose + ears) ignores the drag entirely.
            # Synced here rather than as a second stack edit, so one Ctrl+Z
            # reverses the whole drag (undo re-resolves and re-syncs).
            for c in (CAM_LEFT, CAM_RIGHT):
                if not np.isnan(f.head2d[c]).all():     # cam has face points
                    f.head2d[c][0] = f.kp2d[c][joint]
            f.head3d[0] = triangulate_one(
                f.head2d[CAM_LEFT][0], f.head2d[CAM_RIGHT][0],
                self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
                self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])
        touched = [joint] + self._sync_derived(f, joint, cam)
        for j in touched:
            self._retriangulate(f, j)
        self._refit_frame(f)
        self.pose3dChanged.emit(f.fitted3d, f.head3d, f.filled)
        self.accuracyChanged.emit(self._accuracy(self.current))

    def _sync_derived(self, f, joint: int, cam: str | None) -> list[int]:
        """Move NECK/PELVIS with the shoulder/hip that defines them.

        Same reasoning (and same undo behaviour) as the HEAD/nose sync above:
        the derived point is not an independent measurement, it IS the
        midpoint, so re-deriving it here rather than as a second stack edit
        keeps one Ctrl+Z reversing the whole drag. A derived point the user
        has placed by hand in that view is left alone — their correction
        outranks the derivation.
        """
        if self.project.keypoint_model != "coco17":
            return []                    # halpe26 detects neck/pelvis natively
        derived = _DERIVED_OF.get(int(joint))
        if derived is None:
            return []
        a, b = _DERIVED_FROM[derived]
        moved = False
        for c in (CAMERAS if cam is None else (cam,)):
            if f.corrected[c][derived]:
                continue
            pa, pb = f.kp2d[c][a], f.kp2d[c][b]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            f.kp2d[c][derived] = (pa + pb) / 2.0
            f.scores[c][derived] = min(f.scores[c][a], f.scores[c][b])
            moved = True
        return [derived] if moved else []

    def _retriangulate(self, f, joint: int) -> None:
        xyz = triangulate_one(
            f.kp2d[CAM_LEFT][joint], f.kp2d[CAM_RIGHT][joint],
            self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
            self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])
        if np.isnan(xyz).any() and f.filled[joint]:
            # Only one view has this joint, so there is nothing to triangulate
            # — and the interpolated value from the neighbouring frames is
            # still the best 3D estimate there is. Dropping it here would make
            # editing any other joint in the frame move the whole pose.
            return
        f.pose3d[joint] = xyz
        # this point now comes from its own two observations, whatever it was
        f.filled[joint] = False

    def _refit_frame(self, f) -> None:
        """The SAME fit the batch path runs, on one frame."""
        targets, _ = bone_length_targets(self.project)
        try:
            f.fitted3d = fit_frame(f.pose3d, targets)
        except Exception as e:
            # Qt swallows exceptions raised in a slot, so a fit that failed on
            # a sparse frame would make the drag look like it did nothing.
            # Showing the raw triangulation is better than showing nothing.
            f.fitted3d = np.asarray(f.pose3d, float)
            self.statusMessage.emit(
                f"Bone fit failed on this frame ({type(e).__name__}); "
                f"showing the raw triangulation")

    def _accuracy(self, idx: int) -> np.ndarray:
        f = self.project.frames[idx]
        if self.rig is None:
            return np.full(NUM_JOINTS, np.nan)
        errs = []
        for cam in CAMERAS:
            errs.append(reprojection_error(
                f.pose3d, f.kp2d[cam], self.rig.intr[cam], self.rig.ext[cam]))
        import warnings
        with warnings.catch_warnings():   # all-NaN joint -> quiet nanmean
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.stack(errs), axis=0)


def _smoothing_alpha(smoothing: str, default: float = 0.6) -> float:
    """Alpha out of a project's smoothing setting ("ema0.6" -> 0.6)."""
    try:
        return float(smoothing.removeprefix("ema"))
    except ValueError:
        return default

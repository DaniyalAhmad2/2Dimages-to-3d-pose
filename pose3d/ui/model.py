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
from pose3d.core.project import (
    CAM_LEFT, CAM_RIGHT, CAMERAS, PIPELINE_VERSION, ProjectData,
)
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
    # fitted pose, face keypoints, per-joint gap-filled flags
    pose3dChanged = Signal(object, object, object)
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
        # set by upgrade_pipeline() when a legacy project is recomputed on open
        self.migration_note = ""
        self._stored_fitted3d = None
        self._stored_version = None

    # --- pipeline version migration ---
    def upgrade_pipeline(self) -> str:
        """Bring a project written by an older pipeline up to date, once.

        The fix for the smoothing defect changes what the client sees the
        moment they open the take they complained about, so it has to happen
        without a button press — and it has to say, in real numbers, what
        moved and offer the stored pose back. Nothing is written to disk: the
        recompute lives in memory until the user saves.

        Returns the sentence to show, or "" when nothing was done.
        """
        p = self.project
        if not p.frames or p.pipeline_version >= PIPELINE_VERSION:
            return ""
        if self.rig is None:
            # Deliberately NOT stamped: nothing was recomputed, so this take
            # still carries the old pipeline's pose and still needs the
            # migration. Stamping it here would mark it corrected on the
            # strength of a sentence the user may never act on, and once they
            # loaded a calibration and saved, the offer would never come back.
            self.migration_note = (_NO_RIG_NOTE + _head_hint(p)).strip()
            return self.migration_note
        stored = np.stack([np.asarray(f.fitted3d, float) for f in p.frames])
        self._stored_version = p.pipeline_version
        self.recompute_all()
        p.pipeline_version = PIPELINE_VERSION
        self._stored_fitted3d = stored
        now = np.stack([np.asarray(f.fitted3d, float) for f in p.frames])
        self.migration_note = (_recompute_note(stored, now)
                               + _head_hint(p)).strip()
        return self.migration_note

    def restore_stored_pose(self) -> bool:
        """Put back the pose as the previous build had saved it (this session
        only — nothing was overwritten on disk)."""
        if self._stored_fitted3d is None:
            return False
        for f, pose in zip(self.project.frames, self._stored_fitted3d):
            f.fitted3d = pose.copy()
        self._stored_fitted3d = None
        if self._stored_version is not None:
            # What is on screen is the old pipeline's pose again, so the file
            # must not go on claiming otherwise: saving now keeps this a legacy
            # take, and the next open offers the correction (and the numbers)
            # again instead of freezing the pose the client complained about.
            self.project.pipeline_version = self._stored_version
            self._stored_version = None
        self.migration_note = ""
        self.set_frame(self.current)
        self.statusMessage.emit(
            "Restored the pose stored by the previous build. Recalculate 3D "
            "to go back to the corrected one.")
        return True

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
            f"Detection complete ({len(self.project.frames)} frames); "
            f"hand-corrected points were kept")

    def redetect_head(self, detector, load_image) -> None:
        """Re-run the detector for the FACE keypoints only.

        The migration path for a project made before face keypoints existed:
        it gives the character's head something to be oriented by without
        touching the body pose or a single hand correction.
        """
        if detector is None:
            self.statusMessage.emit("No detector available in this build")
            return
        if self.rig is None:
            self.statusMessage.emit("No calibration loaded — cannot recompute 3D")
            return
        from pose3d.pipeline import detect_project
        from pose3d.geometry.triangulate import triangulate_points
        self.statusMessage.emit("Re-detecting face points…")
        wrote = detect_project(self.project, detector, load_image,
                               fields="head")
        if not wrote:
            # A build whose detector has no face points (the manual detector,
            # or an RTMPose bundle without the face model) writes nothing —
            # saying "re-detected" here would be a success message for work
            # that did not happen, and the head would go on riding the neck.
            self.statusMessage.emit(
                "This build's detector does not produce face points, so "
                "nothing was changed — the head keeps its nose-pitch estimate")
            return
        for f in self.project.frames:
            f.head3d = triangulate_points(
                f.head2d[CAM_LEFT], f.head2d[CAM_RIGHT],
                self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
                self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])
        self.set_frame(self.current)
        self.statusMessage.emit(
            f"Face points re-detected on {len(self.project.frames)} frames; "
            f"the body pose and every correction were left alone")

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
    def _resolve_joint(self, joint: int, cam: str) -> None:
        """Re-triangulate one edited point and re-fit this frame.

        `joint >= NUM_JOINTS` addresses face keypoint `joint - NUM_JOINTS`
        (the camera views and the correction stack share this convention).
        `cam` is the view whose 2D was edited; a derived joint is re-derived
        in that view only, since that is the only one whose parents moved.
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

    def _sync_derived(self, f, joint: int, cam: str) -> list[int]:
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
        if f.corrected[cam][derived]:
            return []
        pa, pb = f.kp2d[cam][a], f.kp2d[cam][b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            return []
        f.kp2d[cam][derived] = (pa + pb) / 2.0
        f.scores[cam][derived] = min(f.scores[cam][a], f.scores[cam][b])
        return [derived]

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


_NO_RIG_NOTE = (
    "This project was made by an earlier build whose 3D lagged one frame "
    "behind the keypoints. It has no calibration loaded, so the stored pose "
    "has been left exactly as it was — load or re-estimate the calibration "
    "and press Recalculate 3D to correct it.")


_HEAD_HINT = (
    " This take also has no face points, so the head keeps its old "
    "nose-pitch guess — run Tools ▸ \"Re-detect face points only\" to orient "
    "it from the eyes and ears (your body pose and every correction are left "
    "untouched).")


def _head_hint(project: ProjectData) -> str:
    """The one line the recompute banner adds when the take predates the face
    keypoints. Recompute cannot invent them — it has no detector — so the
    migration action has to be named where the user is already looking."""
    for f in project.frames:
        for cam in CAMERAS:
            if np.isfinite(np.asarray(f.head2d[cam], float)).any():
                return ""
    return _HEAD_HINT


def _recompute_note(stored: np.ndarray, now: np.ndarray) -> str:
    """The recompute-on-open sentence, with this take's own numbers."""
    d = np.linalg.norm(now - stored, axis=2)
    d = d[np.isfinite(d)]
    if not d.size:
        return ""
    med, mx = float(np.median(d)), float(d.max())
    height = _body_height(now)
    pct = ""
    if np.isfinite(height) and height > 1e-9:
        pct = (f" ({100.0 * med / height:.0f} % / {100.0 * mx / height:.0f} % "
               f"of the figure's height)")
    return (f"Recomputed with solver v{PIPELINE_VERSION}: each frame now "
            f"follows its own keypoints; the previous build blended every "
            f"frame with the one before it — the pose moved "
            f"{1000.0 * med:.1f} mm median, {1000.0 * mx:.1f} mm max{pct}.")


def _body_height(poses: np.ndarray) -> float:
    """Median vertical extent of the de-tilted pose — the scale the client
    actually perceives. Same definition the audit measured against."""
    from pose3d.geometry.orient import de_tilt_matrix, sequence_up
    up = sequence_up(poses)
    if up is None:
        return float("nan")
    upright = poses @ de_tilt_matrix(up).T
    spans = [float(p[v, 2].max() - p[v, 2].min())
             for p, v in ((q, ~np.isnan(q).any(1)) for q in upright)
             if v.sum() >= 2]
    return float(np.median(spans)) if spans else float("nan")


def _smoothing_alpha(smoothing: str, default: float = 0.6) -> float:
    """Alpha out of a project's smoothing setting ("ema0.6" -> 0.6)."""
    try:
        return float(smoothing.removeprefix("ema"))
    except ValueError:
        return default

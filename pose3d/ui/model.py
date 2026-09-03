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
from pose3d.core.skeleton import (
    DERIVED_MIDPOINT_PARENTS, Joint, NUM_JOINTS, derived_joints,
)
from pose3d.geometry.triangulate import (
    fundamental_matrix, reprojection_error, triangulate_one)
from pose3d.pipeline import (
    CalibratedRig, bone_length_targets, cross_view_rejection, fill_frame_gaps,
    fit_frame, gated_kp2d, per_image_allowances, revalidate_joint,
    triangulate_face,
)

HEAD_JOINT = int(Joint.HEAD)

# Cached in place of a `TakeQuality` when the measurement raised: distinct
# from None (= "not measured yet"), so one failure is reported once per
# invalidation instead of on every sidebar refresh.
_QUALITY_FAILED = object()

# Per-joint states the camera views paint. A joint with no 3D is not a joint
# with a poor one, and the two used to be drawn identically.
STATE_OK = "ok"
STATE_NOT_MEASURED = "not_measured"
STATE_REJECTED = "rejected"


class RejectedState(str):
    """`STATE_REJECTED`, carrying the two numbers that make it a diagnosis.

    It IS the string — every `== STATE_REJECTED`, every `RAG_COLORS[...]`
    lookup and every `in HOLLOW_STATES` goes on working — with the measured
    disagreement and the gate it failed attached, so the camera view can name
    them in the tooltip. A purple dot the user cannot get a number out of is
    just a new kind of silence; a second channel from the model to the view
    would be a signature change in every caller in between.
    """
    __slots__ = ("px", "gate")

    def __new__(cls, px: float, gate: float):
        self = super().__new__(cls, STATE_REJECTED)
        self.px = float(px)
        self.gate = float(gate)
        return self


# Joints a layout does not measure but derives as the midpoint of two it does.
# Dragging a shoulder therefore has to move the neck with it, or the pose keeps
# a neck the user can see is in the wrong place and the bone fit is solved
# against a contradiction. WHICH joints those are is per project and comes from
# `skeleton.derived_joints`; this is only the parent pairing, which is the same
# under both layouts.
_DERIVED_FROM: dict[int, tuple[int, int]] = {
    int(joint): (int(a), int(b))
    for joint, (a, b) in DERIVED_MIDPOINT_PARENTS.items()
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
    # {cam: {"measured"|"delivered": (NUM_JOINTS,) residual / figure height}}
    accuracyChanged = Signal(object)
    qualityChanged = Signal()                  # take-wide numbers moved
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
        # bone targets for the live re-solve: the median of every bone over the
        # whole take, which is O(frames x bones) to measure and cannot change
        # between two drags (a drag moves ONE joint of one frame; the median
        # over 26 frames does not follow it). Measured once per recompute and
        # dropped whenever the take's 3D changes underneath it.
        self._bone_targets = None
        # why there is no rig, when the folder held one but it could not be
        # used. Empty for "genuinely uncalibrated" (see app._load_rig).
        self.rig_error = ""
        self._figure_h_px = None
        self._quality = None
        self._epi_thr = None

    # --- pipeline version migration ---
    def upgrade_pipeline(self) -> str:
        """Bring a project written by an older pipeline up to date, once.

        The fix for the smoothing defect changes what the client sees the
        moment they open the take they complained about, so it has to happen
        without a button press — and it has to say, in real numbers, what
        moved and offer the stored pose back. The POSE lives in memory until
        the user saves — nothing in the project file is touched here.

        One exception, and it is in the calibration rather than the project:
        the recompute re-decides the SENSE of the recorded vertical from the
        poses that now exist, and `calib.resolve.finalize_world_up` rewrites
        `calibration/extrinsics.json` (and `report.json`) when that sign
        actually flips. A folder that cannot be written costs only that
        re-check, reported as a note (`recompute_all`).

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
        self._bone_targets = None
        self.invalidate_readouts()
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
        self._bone_targets = None
        write_note = ""
        dropped = triangulate_project(self.project, self.rig)
        smoothing = self.project.smoothing
        report = fit_project(self.project, smooth=smoothing != "none",
                             alpha=_smoothing_alpha(smoothing))
        if self.project_dir:
            # the recorded vertical's SENSE is decided by the poses, and the
            # poses have just changed (see calib.resolve.finalize_world_up).
            # A no-op unless the answer actually moved.
            from pathlib import Path

            from pose3d.calib.resolve import finalize_world_up
            try:
                finalize_world_up(self.project,
                                  Path(self.project_dir) / "calibration")
            except OSError as e:
                # This is the ONE thing a recompute writes, and it writes into
                # the project folder — which can be read-only (a copy off a
                # share, an antivirus lock). Raising here would escape a Qt
                # slot, or — through app.build_model -> upgrade_pipeline —
                # abort startup before any window exists. The recompute itself
                # is done and correct in memory; only the calibration's
                # recorded SIGN stays as it was on disk, so say that and carry
                # on rather than losing the recompute to a failed write.
                write_note = (
                    f"the recorded vertical could not be re-checked "
                    f"(the calibration folder could not be written: "
                    f"{type(e).__name__}: {e})")
        self.invalidate_readouts()
        self.set_frame(self.current)
        # pulled, not pushed: measuring the whole take is not free, and it is
        # wasted work when nothing is listening
        self.qualityChanged.emit()
        msg = f"Recalculated 3D for {len(self.project.frames)} frames"
        notes = [n for n in (rejection_note(dropped, len(self.project.frames)),
                             report.note(), write_note) if n]
        self.statusMessage.emit(" — ".join([msg, *notes]))

    # --- scale ---
    def measured_subject_height(self) -> float:
        """Height of the reconstructed subject, in the calibration's units.

        The same definition as everywhere else in the repo: the median
        vertical extent of the de-tilted delivered pose
        (`pose3d.quality.subject_height`).
        """
        from pose3d.quality import subject_height
        frames = self.project.frames
        if not frames:
            return float("nan")
        return subject_height(np.stack([f.fitted3d for f in frames]))

    def set_scale_from_height(self, real_height_m: float) -> float | None:
        """Rescale the calibration so the subject comes out `real_height_m`.

        The reconstruction is only as correctly SIZED as the marker edge
        length that was typed in at import: get that wrong and every distance,
        every export and the bundled character's fit are wrong by one common
        factor, while every consistency check in the app still reads perfect
        (a similarity transform changes no reprojection, no epipolar distance
        and no bone-length spread). Nothing in the pipeline can detect it, so
        the only fix is a distance the client measures.

        Scaling the world by `s` means `t -> s*t` for both cameras: a point
        `X` images at `R X + t`, so `R (sX) + s t = s (R X + t)` is the same
        ray. Every reconstructed point therefore scales by exactly `s` and no
        image measurement changes. The marker length is scaled with it so a
        later recalibration from the same tags reproduces this size rather
        than silently reverting to the old one.

        Returns the factor applied, or None when there is nothing to scale.
        """
        current = self.measured_subject_height()
        if not (np.isfinite(current) and current > 1e-9):
            self.statusMessage.emit(
                "No reconstructed pose to measure — cannot set the scale")
            return None
        if not (np.isfinite(real_height_m) and real_height_m > 1e-9):
            self.statusMessage.emit("Enter a real distance greater than zero")
            return None
        factor = float(real_height_m) / float(current)
        self.rescale_calibration(factor)
        self.statusMessage.emit(
            f"Rescaled the calibration by {factor:.4f}x: the subject now "
            f"reconstructs {100.0 * float(real_height_m):.1f} cm tall "
            f"(was {100.0 * current:.1f} cm)")
        return factor

    def rescale_calibration(self, factor: float) -> None:
        """Multiply the rig's translations (and marker length) by `factor`.

        Re-persists the calibration folder and recomputes, so the 3D view,
        the export and the next open all agree about the new size.
        """
        if self.rig is None:
            self.statusMessage.emit("No calibration loaded — nothing to scale")
            return
        factor = float(factor)
        for cam in CAMERAS:
            self.rig.ext[cam].t = np.asarray(
                self.rig.ext[cam].t, float) * factor
        self._persist_rig(factor)
        self.recompute_all()

    def _persist_rig(self, factor: float) -> None:
        """Write the rescaled rig back, keeping the calibration's provenance.

        `save_rig` alone would drop report.json's evidence, so the report is
        read, its one length-valued field scaled, and handed back. The
        recorded vertical is a DIRECTION, which a scale cannot touch: it is
        carried through the same report (`save_rig` copies it into
        extrinsics.json), so the folder is written once rather than written
        and then patched. A project with no report.json — calibrated before
        Phase 6 — has nothing to carry it in, and only that case re-opens the
        file.

        Neither read is swallowed. "There is no report.json" is a fact about
        an older project and stays quiet; "report.json is there and is not
        readable" is a fault, and dropping the recorded vertical or the tag
        provenance without a word is how a calibration silently loses its
        evidence.
        """
        if not self.project_dir:
            self.statusMessage.emit(
                "No project folder set — the new scale was applied to this "
                "session only. Use Save As to keep it.")
            return
        import json
        from pathlib import Path

        from pose3d.calib.resolve import save_rig
        calib_dir = Path(self.project_dir) / "calibration"
        ext_path = calib_dir / "extrinsics.json"
        up_keys = ("world_up", "world_up_source", "world_up_spread_deg")
        before = self._read_json(
            ext_path,
            "The calibration's extrinsics.json could not be read ({reason}), "
            "so any vertical recorded in it is not carried into the rescaled "
            "calibration.")
        report = self._read_json(
            calib_dir / "report.json",
            "The calibration's report.json could not be read ({reason}), so "
            "the rescaled calibration is saved without its marker provenance.")
        if report is not None and report.get("marker_length_m") is not None:
            report["marker_length_m"] = float(
                report["marker_length_m"]) * factor
            report["marker_length_source"] = (
                f"rescaled in-app by {factor:.4f}x from a measured distance")
        if report is not None and report.get("world_up") is None and before:
            for key in up_keys:
                if key in before:
                    report[key] = before[key]
        try:
            save_rig(self.rig, calib_dir, report)
            if report is None and before:
                # No report to carry the vertical, so put it back by hand.
                doc = json.loads(ext_path.read_text())
                restored = {k: before[k] for k in up_keys
                            if k in before and k not in doc}
                if restored:
                    ext_path.write_text(json.dumps(doc | restored, indent=2))
        except Exception as e:
            self.statusMessage.emit(
                f"Could not save the rescaled calibration "
                f"({type(e).__name__}: {e})")

    def _read_json(self, path, complaint: str):
        """Parse a calibration side-file: dict, or None when it is absent.

        `complaint` is emitted (with `{reason}` filled in) when the file EXISTS
        but cannot be read — the case the old blanket `except Exception` made
        indistinguishable from "this project predates that file".
        """
        import json
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None                   # older project: nothing to keep
        except Exception as e:
            self.statusMessage.emit(
                complaint.format(reason=f"{type(e).__name__}: {e}"))
            return None

    def redetect_all(self, detector, load_image) -> None:
        """Re-run the detector on every frame, then recompute 3D."""
        if detector is None:
            self.statusMessage.emit("No detector available in this build")
            return
        from pose3d.pipeline import detect_project
        self.statusMessage.emit("Running detection…")
        detect_project(self.project, detector, load_image)
        self.recompute_all()          # invalidates the take-wide readouts
        self.statusMessage.emit(
            f"Detection complete ({len(self.project.frames)} frames); "
            f"hand-corrected points were kept")

    def redetect_head(self, detector, load_image) -> None:
        """Re-run the detector for the nose and the other face points only.

        The migration path for a project made before face keypoints existed:
        it gives the character's head something to turn toward — the nose in
        both modes, the whole face in Face mode — without touching the body
        pose or a single hand correction.
        """
        if detector is None:
            self.statusMessage.emit("No detector available in this build")
            return
        if self.rig is None:
            self.statusMessage.emit("No calibration loaded — cannot recompute 3D")
            return
        from pose3d.pipeline import detect_project
        self.statusMessage.emit("Re-detecting the nose and face points…")
        wrote = detect_project(self.project, detector, load_image,
                               fields="head")
        if not wrote:
            # A build whose detector has no face points (the manual detector,
            # or an RTMPose bundle without the face model) writes nothing —
            # saying "re-detected" here would be a success message for work
            # that did not happen, and the head would go on riding the neck.
            self.statusMessage.emit(
                "This build's detector does not produce face points (the nose, "
                "the eyes and the ears), so nothing was changed — the head "
                "keeps its nose-pitch estimate")
            return
        # the same gate the batch recompute applies, from the same take-wide
        # threshold: a re-detect must not leave face points a recompute would
        # refuse (nor refuse ones it would keep)
        epi_thr = self.epipolar_gate()
        F = fundamental_matrix(
            self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
            self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])
        allow = per_image_allowances(self.rig)
        for f in self.project.frames:
            triangulate_face(f, self.rig, epi_thr, F, allow)
        self.set_frame(self.current)
        self.statusMessage.emit(
            f"Nose and face points re-detected on "
            f"{len(self.project.frames)} frames; the body pose and every "
            f"correction were left alone")

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
        # the take-wide numbers now describe a 2D that no longer exists; drop
        # them so the next reader recomputes rather than showing a stale row.
        # (The figure height and the cross-view gate `_epi_thr` are medians
        # over the WHOLE take and one dragged point cannot move them, so they
        # deliberately survive — recomputing the gate per drag costs a pass
        # over every pair in the take, 15 ms on the 26-frame client take and
        # linear in its length. The row the sidebar recomputes and the number
        # the tooltip has cached therefore still agree; test_cross_view.py::
        # test_a_drag_does_not_split_the_gate_in_two is the guard.)
        self._quality = None
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
        # measured BEFORE the edit, so a drag and its undo fit against the
        # same targets and land in the same place
        self._targets()
        f = self.frame()
        if joint >= NUM_JOINTS:
            # THE CROSS-VIEW GATE APPLIES HERE TOO, exactly as it does to a
            # dragged canonical joint in `_retriangulate`: one helper, so a
            # drag and the next recompute cannot reach different `head3d`.
            # The whole face is re-derived rather than the one point dragged —
            # five triangulations and five verdicts, and no branch that could
            # leave the other four judged by an older rig.
            triangulate_face(f, self.rig, self.epipolar_gate())
            # face points have no bones and never change the character's
            # dimensions: no re-fit, just re-orient the rigid neck+head chain
            # (the nose turns it in both modes, the ears only in Face mode)
            self.pose3dChanged.emit(f.fitted3d, f.head3d, f.filled)
            self.accuracyChanged.emit(self._accuracy(self.current))
            return
        if joint == HEAD_JOINT and self._head_is_the_nose():
            # Under the nose convention the canonical HEAD and the nose face
            # point ARE the same physical detection, so a drag must move both —
            # otherwise the head's orientation (the nose in both modes, the
            # nose + ears face basis in Face mode) ignores it entirely. That
            # is also why the camera views draw no separate nose dot under
            # this convention: the HEAD dot IS it. Synced here rather than as
            # a second stack edit, so one Ctrl+Z reverses the whole drag (undo
            # re-resolves and re-syncs). Under the skull convention they are
            # two different detections ~86 px apart on frame 0 of the client
            # take, and copying one onto the other would teleport the nose onto
            # the skull vertex — silently, into head2d, which is persisted and
            # is what the head basis is built from. That is the convention
            # under which the nose does get its own draggable dot.
            for c in (CAM_LEFT, CAM_RIGHT):
                if not np.isnan(f.head2d[c]).all():     # cam has face points
                    f.head2d[c][0] = f.kp2d[c][joint]
            # and through the same gate as any other face edit: the synced
            # nose is a cross-view pair like the rest, and a drag that pulls
            # it off its epipolar line must lose its 3D here too
            triangulate_face(f, self.rig, self.epipolar_gate())
        touched = [joint] + self._sync_derived(f, joint, cam)
        for j in touched:
            self._retriangulate(f, j)
        self._refit_frame(f)
        self.pose3dChanged.emit(f.fitted3d, f.head3d, f.filled)
        self.accuracyChanged.emit(self._accuracy(self.current))

    def _head_is_the_nose(self) -> bool:
        """Is this project's canonical HEAD the same point as the nose?

        `head_source` is the project's own record of which convention it was
        detected under ("nose" = COCO-17's nose, "skull" = Halpe-26's skull
        vertex), and it is the ONLY thing that may answer this: a project
        written before the key existed defaults to "nose", which is what it
        was. Deciding it from the detector layout instead would make the
        answer depend on a policy the project does not record.
        """
        return self.project.head_source == "nose"

    def _sync_derived(self, f, joint: int, cam: str) -> list[int]:
        """Move NECK/PELVIS with the shoulder/hip that defines them.

        Same reasoning (and same undo behaviour) as the HEAD/nose sync above:
        the derived point is not an independent measurement, it IS the
        midpoint, so re-deriving it here rather than as a second stack edit
        keeps one Ctrl+Z reversing the whole drag. A derived point the user
        has placed by hand in that view is left alone — their correction
        outranks the derivation.

        WHICH joints are derived is the project's layout policy
        (`skeleton.derived_joints`), never the layout's name: under the shipped
        `HALPE26_POLICY` both layouts derive NECK and PELVIS, and an early
        return on `keypoint_model != "coco17"` left a dragged shoulder with a
        28 px stale NECK — triangulated and bone-fitted from a pose the user
        can see is contradictory, and different from what the batch path
        computes from the same 2D.
        """
        derived = _DERIVED_OF.get(int(joint))
        if derived is None or Joint(derived) not in derived_joints(
                self.project.keypoint_model):
            return []
        a, b = _DERIVED_FROM[derived]
        if f.corrected[cam][derived]:
            return []
        pa, pb = f.kp2d[cam][a], f.kp2d[cam][b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            return []
        f.kp2d[cam][derived] = (pa + pb) / 2.0
        # nanmin, not min: min(nan, 0.5) is nan while min(0.5, nan) is 0.5, so
        # a plain min made the derived point's confidence depend on which
        # parent happens to be listed first. Both parents unscored leaves it
        # NaN — the 2D above is real either way.
        pair = np.array([f.scores[cam][a], f.scores[cam][b]], float)
        f.scores[cam][derived] = (float(np.nanmin(pair))
                                  if np.isfinite(pair).any() else np.nan)
        return [derived]

    def _retriangulate(self, f, joint: int) -> None:
        # THE CROSS-VIEW GATE APPLIES HERE TOO, or the live re-solve and the
        # batch recompute reach different poses from the same 2D. It is
        # re-derived for this joint first (the drag may have reconciled the
        # two views, or created the disagreement), then applied to the pair
        # that gets triangulated, which is exactly what triangulate_project
        # does with the whole take. Without it a drag in one view revived a
        # joint whose OTHER view the gate had already refused: the 3D view and
        # the export got a point the tooltip was calling "not triangulated",
        # and the next recompute deleted it again.
        #
        # A hand-placed point still wins: `validate_cross_view` never rejects
        # a `corrected` view, so the loser of a disagreement the user created
        # is the view they did not touch.
        revalidate_joint(f, joint, self.rig, self.epipolar_gate())
        kp = gated_kp2d(f)
        # pose3d is the MEASUREMENT: it takes whatever the two views now say,
        # NaN included. An interpolated value never lives here — _refit_frame
        # rebuilds the fill (and the flag) from the neighbouring frames right
        # after, so a joint only one view can see still reaches the fit.
        f.pose3d[joint] = triangulate_one(
            kp[CAM_LEFT][joint], kp[CAM_RIGHT][joint],
            self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
            self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])

    def _targets(self) -> dict:
        """Bone-length targets for the live re-solve, measured once per
        recompute.

        The target is the MEDIAN of one bone over the whole take, so moving
        one joint of one frame cannot legitimately move it — and re-measuring
        14 bones across every frame on every mouse release, undo and redo made
        the answer depend on the order the edits arrived in (a drag and its
        undo fitted against fractionally different targets). Dropped whenever
        the take's 3D is recomputed underneath it.
        """
        if self._bone_targets is None:
            self._bone_targets, _ = bone_length_targets(self.project)
        return self._bone_targets

    def _refit_frame(self, f) -> None:
        """The SAME fit the batch path runs, on one frame — and on the two
        frames whose fill was interpolated FROM it.

        Including the gap fill: the batch path fits `fill_gaps`'s copy, so the
        live path fits this frame's copy from `fill_frame_gaps`, or a drag
        would silently drop a joint the recompute poses.

        The neighbours matter because the fill reads them: if this drag makes
        the cross-view gate reject a joint (or reinstates one), a neighbouring
        frame that had interpolated that joint across a one-frame dropout is
        now holding a fill the batch path would no longer invent — its
        `filled` flag and its `fitted3d` would stay stale until the next whole
        recompute, and the live and batch paths would disagree about which
        joints were invented. `fill_frame_gaps` reads only the immediate
        neighbours' `pose3d`, so re-running it either side is enough and it
        cannot cascade further; the refit only runs when the flags actually
        moved, so an ordinary drag costs one fit as before.
        """
        # by identity: Frame is a dataclass full of arrays, so list.index()
        # would compare them elementwise and raise
        idx = next(i for i, g in enumerate(self.project.frames) if g is f)
        self._fit_one(idx)
        for nb in (idx - 1, idx + 1):
            if not 0 <= nb < len(self.project.frames):
                continue
            before = np.array(self.project.frames[nb].filled, copy=True)
            # re-derives the neighbour's flags in place from today's pose3d
            infill = fill_frame_gaps(self.project, nb)
            if not np.array_equal(before, self.project.frames[nb].filled):
                self._fit_one(nb, infill)

    def _fit_one(self, idx: int, infill=None) -> None:
        """Fit frame `idx` from its gap-filled input (computed if not given)."""
        f = self.project.frames[idx]
        if infill is None:
            infill = fill_frame_gaps(self.project, idx)
        try:
            f.fitted3d = fit_frame(infill, self._targets())
        except Exception as e:
            # Qt swallows exceptions raised in a slot, so a fit that failed on
            # a sparse frame would make the drag look like it did nothing.
            # Showing the raw triangulation is better than showing nothing.
            f.fitted3d = np.asarray(infill, float)
            self.statusMessage.emit(
                f"Bone fit failed on frame {f.frame_id} "
                f"({type(e).__name__}); showing the raw triangulation")

    # --- readouts ---
    def figure_h_px(self) -> dict[str, float]:
        """Each camera's median figure height in pixels, over the whole take.

        THE denominator for every residual this class reports. A pixel is not
        a unit anyone can compare: the subject stands 776 px tall in the left
        image of the client's take and 407 px in the right, so the same error
        reads 1.9x worse on the right, and a different phone changes both.
        Computed once per take and invalidated whenever the 2D changes.
        """
        if self._figure_h_px is None:
            from pose3d.quality import figure_height_px
            frames = self.project.frames
            self._figure_h_px = {
                c: (figure_height_px(np.stack([f.kp2d[c] for f in frames]))
                    if frames else float("nan"))
                for c in CAMERAS}
        return self._figure_h_px

    def invalidate_readouts(self) -> None:
        """Drop the cached take-wide numbers (figure height, take quality).

        Called wherever the 2D or the 3D changes wholesale. A single joint drag
        does NOT invalidate the figure height: it is a median over 26 frames
        and one point cannot move it, and recomputing it per drag would make
        every correction O(take)."""
        self._figure_h_px = None
        self._quality = None
        self._epi_thr = None

    def quality(self):
        """`pose3d.quality.TakeQuality` for the whole take, or None.

        Computed once per recompute — it is a take-wide measurement, not a
        per-frame one — and cached, because the sidebar reads it on every
        refresh and it walks every frame twice.

        A FAILURE is cached too. Without that, a take the measurement cannot
        handle re-ran the whole two-pass walk and re-emitted the same status
        message on every sidebar refresh — which is every frame change — so
        the one take that cannot be measured is also the one that runs the
        measurement most often.
        """
        if self.rig is None or not self.project.frames:
            return None
        if self._quality is _QUALITY_FAILED:
            return None
        if self._quality is None:
            from pose3d.quality import take_quality
            try:
                self._quality = take_quality(self.project, self.rig)
            except Exception as e:      # never let a readout break the app
                self._quality = _QUALITY_FAILED
                self.statusMessage.emit(
                    f"Could not measure this take ({type(e).__name__}: {e})")
                return None
        return self._quality

    def _accuracy(self, idx: int) -> dict:
        """Per-camera, per-stage normalised reprojection residual for a frame.

        `{cam: {"measured": (NUM_JOINTS,), "delivered": (NUM_JOINTS,)}}`, each
        divided by that camera's own `figure_h_px`.

        Two stages, never averaged over cameras:
          "measured"  — `Frame.pose3d`, the raw triangulation: do the two views
                        agree? This is the only one a keypoint drag can move.
          "delivered" — `Frame.fitted3d`, the pose the 3D view shows and the
                        export writes: what you are actually looking at.
        Their ratio is the permanent regression detector. The build the client
        complained about delivered a pose 4.2x/6.2x further from the keypoints
        than the measurement and no readout in the app could say so, because
        this function scored only `pose3d` and then averaged the two cameras
        into one number that hid which view disagreed.
        """
        blank = np.full(NUM_JOINTS, np.nan)
        if self.rig is None:
            return {c: {"measured": blank.copy(), "delivered": blank.copy()}
                    for c in CAMERAS}
        f = self.project.frames[idx]
        fh = self.figure_h_px()
        out = {}
        for cam in CAMERAS:
            denom = fh.get(cam, float("nan"))
            denom = denom if np.isfinite(denom) and denom > 1e-9 else np.nan
            per = {}
            for stage, pose in (("measured", f.pose3d),
                                ("delivered", f.fitted3d)):
                err = reprojection_error(pose, f.kp2d[cam],
                                         self.rig.intr[cam], self.rig.ext[cam])
                per[stage] = np.asarray(err, float) / denom
            out[cam] = per
        return out

    def epipolar_gate(self) -> float:
        """The cross-view gate this take is being judged by, in px.

        Cached with the other take-wide numbers and dropped by
        `invalidate_readouts`: the camera views ask for it on every frame
        change, and it must be the SAME number the gate itself used or a
        purple dot would quote a threshold nothing was measured against.

        NaN without a calibration: there is no gate, because nothing was
        gated.
        """
        if self.rig is None:
            return float("nan")
        if self._epi_thr is None:
            from pose3d.pipeline import epipolar_threshold
            self._epi_thr = float(
                epipolar_threshold(self.rig, self.project))
        return self._epi_thr

    def joint_states(self, idx: int) -> dict[str, list[str]]:
        """Per-camera, per-joint state: why a joint has no number.

        "not measured" and "rejected by the cross-view check" are different
        facts and used to be drawn the same. A rejection is now a flag the
        gate wrote (`Frame.rejected`, re-derived on every recompute), so this
        reports what actually happened rather than inferring it. It ALSO
        rejects a pair the gate would refuse but carries no flag yet — the
        same verdict the next recompute will record, from the same
        `pipeline.cross_view_rejection` — because between a hand edit and a
        recompute the mask is stale for whatever the edit touched, and a joint
        may still be holding 3D built from a point that has since moved. It
        is that ONE function and not a paraphrase of it, so the dot cannot
        call a joint rejected that the gate has decided to keep.

        A rejected state carries the numbers (`RejectedState.px` / `.gate`) so
        the tooltip can name them.
        """
        f = self.project.frames[idx]
        states = {c: [STATE_OK] * NUM_JOINTS for c in CAMERAS}
        if self.rig is None:
            return states
        thr = self.epipolar_gate()
        F = fundamental_matrix(
            self.rig.intr[CAM_LEFT], self.rig.intr[CAM_RIGHT],
            self.rig.ext[CAM_LEFT], self.rig.ext[CAM_RIGHT])
        allow = per_image_allowances(self.rig)
        for j in range(NUM_JOINTS):
            flagged = {c: bool(f.rejected[c][j]) for c in CAMERAS}
            # No short-circuit on "it already has 3D": that 3D can be older
            # than the 2D under it. With auto-recalc off, a drag (or an undo)
            # moves a keypoint and clears its flag while `pose3d` keeps the
            # value the old point produced, and the joint read OK right up to
            # the recompute that refused it. 15 verdicts per frame change is
            # under a millisecond.
            # The WHOLE rule, not part of it: `cross_view_rejection` is the
            # function that writes the mask, asked without letting it write.
            # Checking the Sampson distance alone missed the per-image half
            # 6.2 added; checking `cross_view_verdict` alone missed the other
            # end of the rule, that a pair corrected in BOTH views is kept —
            # so the joints the user had already fixed by hand were painted
            # "not triangulated" while their 3D sat in the export.
            e, loser = cross_view_rejection(f, j, self.rig, F, thr, allow)
            if any(flagged.values()) or loser is not None:
                state = RejectedState(e, thr)
            elif np.isnan(f.pose3d[j]).any():
                state = STATE_NOT_MEASURED
            else:
                continue
            for c in CAMERAS:
                states[c][j] = state
        return states


def worst_per_joint(errors, stage: str):
    """The worse of the two cameras, per joint — never their mean.

    `errors` is one frame of `ProjectModel._accuracy`. A joint the left camera
    places well and the right does not is a joint with a problem, and the mean
    of the two says it is half a problem.

    This and `frame_stat` are the two statistics that decide the timeline band
    and the gauge, so they live beside the residuals they reduce rather than as
    private helpers on the window: they are facts about a project, and a test
    should not have to reach into a QMainWindow to compute one.
    """
    stack = [np.asarray(errors[c][stage], float) for c in errors]
    if not stack:
        return np.full(0, np.nan)
    # fmax, not nanmax: NaN-tolerant, and all-NaN gives NaN without the
    # "all-NaN slice" warning nanmax raises on a joint neither view saw.
    return np.fmax.reduce(np.stack(stack), axis=0)


def frame_stat(per_joint):
    """One number for a frame: the MEDIAN joint, not the worst.

    The worst of 15 joints is a max over 15 samples; on a good take it is red
    almost every frame, which is how the old timeline managed to be red 21
    times out of 26 (and green never) and tell the user nothing.
    """
    a = np.asarray(per_joint, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


_NO_RIG_NOTE = (
    "This project was made by an earlier build whose 3D lagged one frame "
    "behind the keypoints. It has no calibration loaded, so the stored pose "
    "has been left exactly as it was — load or re-estimate the calibration "
    "and press Recalculate 3D to correct it.")


_HEAD_HINT = (
    " This take also has no face points, so the head keeps its old "
    "nose-pitch guess — run Tools ▸ \"Re-detect face points only\" so the "
    "character's head can turn toward the detected nose (or the face, in Face "
    "mode). Your body pose and every correction are left untouched.")


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

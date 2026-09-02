"""End-to-end pose reconstruction pipeline over a project.

Ties the layers together: detect per view -> triangulate -> fill one-frame
gaps -> bone-length fit. Kept independent of Qt so it runs headless (dataset
validation, CLI, tests) and is called by the UI's recompute.

`fit_frame` is the single fit: both the batch path (`fit_project`) and the
live manual-correction path (`ui.model.ProjectModel._resolve_joint`) call it
with targets from `bone_length_targets`, so a drag and a recompute cannot
disagree by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def _detector_keypoint_model(detector: KeypointDetector) -> str:
    """Which joint layout this detector emits.

    Read off the detector rather than declared by it: only RTMPose has the
    choice today (`feet=True` selects Halpe-26, which detects neck/pelvis
    natively instead of deriving them as midpoints), and a detector that grows
    an explicit attribute is honoured first.
    """
    declared = getattr(detector, "keypoint_model", None)
    if declared:
        return str(declared)
    return "halpe26" if getattr(detector, "feet", False) else "coco17"


def detect_project(project: ProjectData, detector: KeypointDetector,
                   load_image, on_frame=None, respect_corrections: bool = True,
                   fields: str = "all") -> int:
    """Populate each frame's 2D keypoints/scores via the detector.

    load_image(path) -> BGR ndarray. Mutates project in place.

    on_frame(i, n) is called after each frame, for a progress dialog.

    respect_corrections keeps hand-placed points: a correction is a human
    saying the detector was wrong there, and re-running detection used to
    overwrite every one of them while leaving the `corrected` flag set — so
    the UI went on claiming a point had been corrected by hand after the
    correction had been thrown away.

    fields="head" writes only the face keypoints (head2d/head_scores), leaving
    kp2d, scores and corrected untouched. That is the migration path for a
    project made before face keypoints existed: its body pose and its
    corrections survive, and the head stops riding the neck.

    Returns how many (frame, camera) face-keypoint sets the detector actually
    supplied — 0 when it returns none, which is the difference between "the
    face points were re-detected" and "this build's detector has no face
    points to give", and the caller must not report the first as the second.
    """
    if fields not in ("all", "head"):
        raise ValueError(f"fields must be 'all' or 'head', not {fields!r}")
    if fields == "all":
        project.keypoint_model = _detector_keypoint_model(detector)
    heads = 0
    n = len(project.frames)
    for i, frame in enumerate(project.frames):
        for cam in (CAM_LEFT, CAM_RIGHT):
            img = load_image(frame.images[cam])
            det = detector.detect(img)
            if fields == "all":
                keep = frame.corrected[cam] if respect_corrections \
                    else np.zeros(NUM_JOINTS, bool)
                frame.kp2d[cam] = np.where(keep[:, None], frame.kp2d[cam],
                                           det.xy)
                frame.scores[cam] = np.where(keep, frame.scores[cam],
                                             det.scores)
            if det.head_xy is not None:
                frame.head2d[cam] = det.head_xy
                frame.head_scores[cam] = det.head_scores
                heads += 1
        if on_frame is not None:
            on_frame(i + 1, n)
    return heads


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
    for frame in project.frames:
        # A gap-fill flag describes the 3D that is about to be replaced. Reset
        # before anything else touches the take, so a flag can never outlive
        # the fill that set it (a joint recovered by a re-detect or a hand
        # correction must come back as a measurement, not stay "interpolated").
        frame.filled[:] = False
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


def fill_gaps(project: ProjectData, max_gap: int = 1) -> tuple[np.ndarray, int]:
    """The bone fit's INPUT: pose3d with one-frame dropouts interpolated.

    Returns `(poses, n_filled)` — a (frames, joints, 3) COPY, never the
    frames' own arrays — and flags what it invented in `Frame.filled`.
    `Frame.pose3d` is not touched: it is the measurement, and it stays NaN
    where the cameras saw nothing, so every "measured" number in
    `pose3d.quality` (bone-length CV, symmetry, reprojection of the raw pose)
    is computed on observations only. Interpolated joints reach the user
    through `fitted3d`, flagged.

    A joint that is missing for a single frame but present either side is
    almost always a momentary detection failure, not the joint leaving the
    scene. Interpolating linearly between the two neighbours (for the default
    max_gap=1, their midpoint) restores a continuous limb without inventing
    anything the take does not contain: the value is symmetric — no lag, no
    forward leak, unlike the stale value the old smoother carried across a
    dropout — and the flag tells the 3D view, the camera views, the
    reconstructed-joint count and the export it was interpolated. Gaps longer
    than `max_gap`, and gaps that run off either end of the take, stay NaN — a
    visible hole is the honest answer there.

    Re-runnable by construction: it reads only the raw measurement, so the
    flags always describe the 2D as it is now.
    """
    frames = project.frames
    if not frames:
        return np.zeros((0, NUM_JOINTS, 3)), 0
    poses = np.stack([np.asarray(f.pose3d, float) for f in frames])
    for f in frames:
        f.filled[:] = False
    filled = 0
    for j in range(NUM_JOINTS):
        present = [not np.isnan(poses[t, j]).any() for t in range(len(frames))]
        t = 0
        while t < len(frames):
            if present[t]:
                t += 1
                continue
            run = t
            while run < len(frames) and not present[run]:
                run += 1
            # flanked by observations on both sides, and short enough?
            if t > 0 and run < len(frames) and (run - t) <= max_gap:
                a, b = poses[t - 1, j], poses[run, j]
                for k in range(t, run):
                    w = (k - t + 1) / (run - t + 1)
                    poses[k, j] = (1 - w) * a + w * b
                    frames[k].filled[j] = True
                    filled += 1
            t = run
    return poses, filled


def fill_frame_gaps(project: ProjectData, index: int) -> np.ndarray:
    """The bone fit's input for ONE frame, and that frame's `filled` flags.

    The live path's half of `fill_gaps`: a drag re-solves a single frame, so
    the fill for that frame is recomputed here from its neighbours' raw
    `pose3d` rather than left as whatever the last whole-take fill decided.
    Same rule as the batch fill at the default `max_gap=1` — a joint missing
    here and present in both neighbours becomes their midpoint — so a drag and
    a recompute cannot disagree about which joints were invented.
    """
    frames = project.frames
    f = frames[index]
    poses = np.asarray(f.pose3d, float).copy()
    f.filled[:] = False
    if 0 < index < len(frames) - 1:
        a = np.asarray(frames[index - 1].pose3d, float)
        b = np.asarray(frames[index + 1].pose3d, float)
        for j in range(NUM_JOINTS):
            if not np.isnan(poses[j]).any():
                continue
            if np.isnan(a[j]).any() or np.isnan(b[j]).any():
                continue
            poses[j] = 0.5 * (a[j] + b[j])
            f.filled[j] = True
    return poses


@dataclass
class FitReport:
    """What the batch fit had to compromise on, for the user to see.

    Previously print()ed, which in a windowed build goes to a log file nobody
    opens — so a take where the fit fell back on half its frames looked
    identical to a clean one.
    """
    failed: int = 0                  # frames that fell back to the raw pose
    first_error: str | None = None
    fallback_bones: int = 0          # bones never observed -> default length
    gaps_filled: int = 0             # joint-frames interpolated and flagged

    def note(self) -> str:
        """One user-facing sentence, or "" when there is nothing to say."""
        parts = []
        if self.failed:
            parts.append(
                f"the bone fit could not solve {self.failed} frame(s) and "
                f"shows their raw triangulation instead ({self.first_error})")
        if self.fallback_bones:
            parts.append(
                f"{self.fallback_bones} bone(s) were never seen in this take, "
                f"so a default body proportion was used for them")
        if self.gaps_filled:
            parts.append(
                f"{self.gaps_filled} joint(s) were missing for a single frame "
                f"and were interpolated from the frames either side — they are "
                f"drawn hollow")
        return "; ".join(parts)


def bone_length_targets(project: ProjectData) -> tuple[dict, int]:
    """(target length per bone, how many fell back to a default proportion).

    Measured as the median over the whole take — the subject's own skeleton,
    not a generic body — with a default proportion only where a bone was never
    observed at all. The single source of these numbers: the batch fit and the
    live manual-correction re-solve both call this, so they cannot drift.
    """
    raw = np.stack([f.pose3d for f in project.frames]) if project.frames \
        else np.zeros((0, NUM_JOINTS, 3))
    measured = measure_bone_lengths(raw)
    fb = fallback_bone_lengths()
    n_fallback = sum(1 for v in measured.values() if v <= 1e-6)
    return {k: (v if v > 1e-6 else fb[k]) for k, v in measured.items()}, n_fallback


def fit_frame(pose3d: np.ndarray, bone_lengths: dict) -> np.ndarray:
    """Bone-fit ONE frame's triangulation. The whole fit, for every caller.

    Unobserved joints stay NaN: a joint the cameras did not see is absent,
    not guessed.
    """
    return fit_bone_lengths(pose3d, bone_lengths, fill_missing=False)


def fit_project(project: ProjectData, bone_lengths=None,
                smooth: bool = False, alpha: float = 0.6) -> FitReport:
    """Bone-length fit every frame, over the gap-filled input.

    The fit's input is `fill_gaps`'s copy, so a one-frame dropout is posed
    (and flagged) in `fitted3d` while `pose3d` keeps the hole the cameras
    actually left. Bone targets are measured from the RAW pose for the same
    reason: an interpolated joint is not an observation of a bone length.

    Smoothing is OFF by default and opt-in per project: see
    `bonefit.smooth_temporal` for what a temporal filter costs on a take of
    discrete hand-posed frames.
    """
    infill, n_filled = fill_gaps(project)
    report = FitReport(gaps_filled=n_filled)
    if bone_lengths is None:
        bone_lengths, report.fallback_bones = bone_length_targets(project)

    # One awkward frame must never lose the whole take: fall back to its raw
    # triangulation and carry on. The import dialog wraps this in a blanket
    # except, so anything raised here used to surface as "Import failed" with
    # every other frame's work discarded.
    per_frame = []
    for i in range(len(project.frames)):
        try:
            per_frame.append(fit_frame(infill[i], bone_lengths))
        except Exception as e:
            report.failed += 1
            report.first_error = report.first_error or f"{type(e).__name__}: {e}"
            per_frame.append(np.asarray(infill[i], float))
    fitted = np.stack(per_frame) if per_frame \
        else np.zeros((0, NUM_JOINTS, 3))
    if smooth and len(fitted) > 1:
        fitted = smooth_temporal(fitted, alpha=alpha)
    for f, pose in zip(project.frames, fitted):
        f.fitted3d = pose
    return report


def run_full(project: ProjectData, detector: KeypointDetector,
             rig: CalibratedRig, load_image, smooth: bool = False) -> FitReport:
    """Detect -> triangulate -> fill gaps -> fit for the whole project."""
    detect_project(project, detector, load_image)
    triangulate_project(project, rig)
    return fit_project(project, smooth=smooth)

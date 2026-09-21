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

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, ProjectData
from pose3d.core.skeleton import (
    CORE_INDEX, DERIVED_MIDPOINT_PARENTS, EXTREMITY_JOINTS, JOINT_NAMES,
    Joint, NUM_HEAD_KP, NUM_JOINTS, derived_joints,
)
from pose3d.detect.base import KeypointDetector
from pose3d.geometry.bonefit import (
    fallback_bone_lengths, fit_bone_lengths, measure_bone_lengths,
    reference_from_measured, smooth_temporal,
)
from pose3d.geometry.triangulate import (
    epipolar_distance, fundamental_matrix, triangulate_points,
)

#: `CORE_INDEX` as a membership test, for the per-joint loops below.
_CORE_SET: frozenset[int] = frozenset(CORE_INDEX)


class Cancelled(Exception):
    """The user stopped a long job before it finished.

    Defined here, in the Qt-free half, and re-exported by `ui.worker`, so the
    GUI and the pipeline raise and catch ONE class. It is not a failure: what
    it guarantees is that whatever raised it committed nothing.
    """


class CalibratedRig:
    """Intrinsics + extrinsics for the two-camera rig."""

    def __init__(self, intr_l: Intrinsics, intr_r: Intrinsics,
                 ext_l: Extrinsics, ext_r: Extrinsics):
        self.intr = {CAM_LEFT: intr_l, CAM_RIGHT: intr_r}
        self.ext = {CAM_LEFT: ext_l, CAM_RIGHT: ext_r}


def read_frame_image(frame, cam: str, load_image, skipped=None):
    """This frame's image for `cam`, or None with the reason recorded.

    A photo that cannot be read costs its own PAIR and nothing else — the
    rule for every pass over a take's photographs, stated once here because
    two passes each having their own was the defect: the calibration skipped
    a 0-byte OneDrive placeholder and the detection right after it raised on
    the same file, so the import ended with "Import failed" and no project
    written, after the copy and after the calibration.

    Every failure of the injected loader is caught, not just `ImageReadError`:
    the loader is a parameter (the GUI passes `read_image`, the tools pass
    `cv2.imread`, tests pass doubles), and a frame is the unit of loss
    whatever it raises. A loader that ANSWERS None instead of raising is the
    same fact — `cv2.imread` does that on a path it cannot encode in the
    machine's ANSI code page — and `frame.images.get(cam)` returning None is
    the same kind of accident, which used to be worse: `read_image(None)`
    dies in `Path(None)` with a TypeError that names nothing.

    `skipped` is an optional list; each entry is
    `{"frame", "camera", "reason"}` — what the user is shown, so it must be
    the reason this run actually MEASURED and not a fabricated one.
    """
    path = frame.images.get(cam)

    def note(reason):
        if skipped is not None:
            skipped.append({"frame": frame.frame_id, "camera": cam,
                            "reason": reason})

    if path is None:
        note("this frame has no photo for that camera")
        return None
    try:
        img = load_image(path)
    except Exception as e:
        note(str(e) or f"{type(e).__name__}")
        return None
    if img is None:
        note(f"'{path}' could not be read")
    return img


def summarise_unreadable(skipped, what: str) -> str:
    """One sentence naming the frames `what` could not read ("" for none).

    The sentence states only the fact every caller shares — these frames had
    a photo that could not be read — and `what` states the COST, because the
    cost differs: the calibration loses the whole frame (it needs both views
    to see the same tag), while the detection keeps the view it could read
    and leaves the other one missing. "N image pair(s) were skipped"
    overstated that second case to the one person who has to decide whether
    to re-import.
    """
    if not skipped:
        return ""
    frames = sorted({s["frame"] for s in skipped}, key=_frame_order)
    shown = ", ".join(frames[:5]) + (", …" if len(frames) > 5 else "")
    return (f"{len(frames)} frame(s) had a photo that could not be read "
            f"({shown}); {what}. "
            f"The first was: {skipped[0]['reason']}")


def _frame_order(frame_id):
    """Sort key for frame ids: numerically when they are numbers.

    The ids the importer writes are zero-padded ("0007"), where lexicographic
    and numeric order agree — but a project whose frames are named "9" and
    "10" would be listed 10 before 9, in a sentence whose whole job is to let
    the user find the photo.
    """
    s = str(frame_id)
    return (0, int(s), "") if s.isdigit() else (1, 0, s)


def _detector_keypoint_model(detector: KeypointDetector) -> str:
    """Which joint layout this detector emits.

    Read off the detector rather than declared by it: only RTMPose has the
    choice today (`feet=True` selects Halpe-26, which detects a skull-vertex
    HEAD where COCO-17 has only the nose), and a detector that grows an
    explicit attribute is honoured first.

    The layout name does NOT say which joints are derived: Halpe-26 also
    detects a neck and a hip, but `skeleton.HALPE26_POLICY` does not take
    them, so under both layouts NECK and PELVIS are the 2D midpoints of the
    shoulders/hips. Ask `skeleton.derived_joints(keypoint_model)` for that,
    never this string.
    """
    declared = getattr(detector, "keypoint_model", None)
    if declared:
        return str(declared)
    return "halpe26" if getattr(detector, "feet", False) else "coco17"


# The parents of every joint the midpoint convention derives, in CANONICAL
# indices. `skeleton.DERIVED_MIDPOINT_PARENTS` is the pairing; WHICH of them a
# given project actually derives is `skeleton.derived_joints(keypoint_model)`,
# asked per project and never inferred from the layout's name.
_DERIVED_PARENTS: dict[int, tuple[int, int]] = {
    int(joint): (int(a), int(b))
    for joint, (a, b) in DERIVED_MIDPOINT_PARENTS.items()
}


def derive_midpoint(frame, cam: str, derived: int) -> bool:
    """Put ONE derived joint back on its parents' midpoint, in ONE view.

    THE midpoint rule, for every path that has to restore it: the live drag
    (`ui.model._sync_derived`, which re-derives the one joint whose parent
    moved) and the batch paths (`derive_midpoints`, below). A derived joint is
    not an independent measurement — it IS the midpoint — so a NECK that
    contradicts the shoulders it is made of is triangulated and bone-fitted
    from a pose the user can see is wrong, and the two paths must not be able
    to disagree about what it should be.

    Returns True when it wrote the point. False, leaving the joint alone, when

    * the user placed it by hand (`corrected`): their correction outranks the
      derivation, here as everywhere;
    * either parent is missing: the midpoint of a NaN is not an improvement on
      whatever is there.
    """
    parents = _DERIVED_PARENTS.get(int(derived))
    if parents is None or frame.corrected[cam][derived]:
        return False
    a, b = parents
    pa, pb = frame.kp2d[cam][a], frame.kp2d[cam][b]
    if np.isnan(pa).any() or np.isnan(pb).any():
        return False
    frame.kp2d[cam][derived] = (pa + pb) / 2.0
    # nanmin, not min: min(nan, 0.5) is nan while min(0.5, nan) is 0.5, so a
    # plain min made the derived point's confidence depend on which parent
    # happens to be listed first. Both parents unscored leaves it NaN — the 2D
    # above is real either way.
    pair = np.array([frame.scores[cam][a], frame.scores[cam][b]], float)
    frame.scores[cam][derived] = (float(np.nanmin(pair))
                                  if np.isfinite(pair).any() else np.nan)
    return True


def derive_midpoints(project: ProjectData) -> int:
    """Restore the midpoint rule across the whole take. Returns how many moved.

    Every BATCH path that rewrites 2D has to run this, because every one of
    them can leave a derived joint contradicting its parents: a re-detect
    keeps the hand-corrected shoulder and writes the detector's own neck over
    the synced one (half the correction out), and a plain recompute follows a
    drag made with "Auto Recalculate 3D" off, or before the project was
    calibrated, where the live re-solve never ran at all. The contradictory 2D
    is then triangulated, bone-fitted, drawn and saved.

    A hand-placed derived joint is left alone (see `derive_midpoint`), and a
    layout that DETECTS these joints rather than deriving them is left alone
    entirely — `skeleton.derived_joints` is asked, not the layout's name.
    """
    joints = sorted(int(j) for j in derived_joints(project.keypoint_model))
    moved = 0
    for frame in project.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            for j in joints:
                moved += int(derive_midpoint(frame, cam, j))
    return moved


def detect_project(project: ProjectData, detector: KeypointDetector,
                   load_image, on_frame=None, respect_corrections: bool = True,
                   fields: str = "all", on_progress=None, cancelled=None,
                   skipped=None) -> int:
    """Populate each frame's 2D keypoints/scores via the detector.

    load_image(path) -> BGR ndarray. Mutates project in place.

    on_frame(i, n) is called after each frame, for a progress dialog.

    on_progress(done, total, label) is the same fact in the shape
    `ui.worker.run_job` hands a job, so a UI call site passes its `report`
    straight through.

    cancelled() is polled between frames; when it turns True the detection
    raises `Cancelled` and NOTHING is written. That is why the loop stages its
    answers and commits them, together with the provenance triple, in one
    post-loop step: a take half-detected under a new layout, still carrying
    the old `head_source`, is a project that lies about its own 2D.

    skipped is an optional list collecting every (frame, camera) whose photo
    could not be read, in the shape `read_frame_image` records — the same
    shape `calib.resolve` reports its own skips in. A photo that cannot be
    read costs that view of that frame and nothing else; it never stops the
    detection, because one 0-byte placeholder used to abort the whole import
    with nothing written.

    respect_corrections keeps hand-placed points: a correction is a human
    saying the detector was wrong there, and re-running detection used to
    overwrite every one of them while leaving the `corrected` flag set — so
    the UI went on claiming a point had been corrected by hand after the
    correction had been thrown away.

    fields="head" writes only the face keypoints (head2d/head_scores), leaving
    kp2d, scores and corrected untouched. That is the migration path for a
    project made before face keypoints existed: its body pose and its
    corrections survive, and the head stops riding the neck.

    All three facts about the detection — the layout (`keypoint_model`), the
    head convention (`head_source`) and the model that produced it
    (`detector`) — are recorded HERE, from the detector itself. They used to
    be left to each caller: the provenance string was written out at three
    call sites, and a caller that forgot `head_source` produced a halpe26
    project posed under the nose convention, i.e. the ~45 deg nose offset
    applied to a skull point — the one double-correction the whole
    `head_source` mechanism exists to prevent. A caller that knows better may
    still override them afterwards.

    Returns how many (frame, camera) face-keypoint sets the detector actually
    supplied — 0 when it returns none, which is the difference between "the
    face points were re-detected" and "this build's detector has no face
    points to give", and the caller must not report the first as the second.
    """
    if fields not in ("all", "head"):
        raise ValueError(f"fields must be 'all' or 'head', not {fields!r}")
    heads = 0
    n = len(project.frames)
    staged: list[tuple] = []          # (frame, cam, what to write)
    for i, frame in enumerate(project.frames):
        if cancelled is not None and cancelled():
            raise Cancelled()
        for cam in (CAM_LEFT, CAM_RIGHT):
            img = read_frame_image(frame, cam, load_image, skipped)
            if img is None:
                # This photo costs its own view of this frame and nothing
                # else. It used to cost the take: `read_image` RAISES where
                # `cv2.imread` returned None — 0-byte OneDrive placeholders
                # being the case it was written for — and the raise reached
                # the import dialog's outer handler, which reported "Import
                # failed" and wrote no project at all. The calibration right
                # before it had already learned to skip the same file.
                #
                # Nothing is staged for it, so the frame keeps whatever 2D it
                # had — NaN on a fresh import, which the views draw as a
                # missing joint the user can place by hand. Inventing a pose
                # for a photo nobody could read is the one thing worse.
                continue
            det = detector.detect(img)
            body = None
            if fields == "all":
                keep = frame.corrected[cam] if respect_corrections \
                    else np.zeros(NUM_JOINTS, bool)
                # The detector's own answer, kept whole and separately: the
                # working arrays below merge it with the user's corrections,
                # and this is the only record of what the detector said. It is
                # written once per detection and never again.
                body = (np.asarray(det.xy, float).copy(),
                        np.asarray(det.scores, float).copy(),
                        np.where(keep[:, None], frame.kp2d[cam], det.xy),
                        np.where(keep, frame.scores[cam], det.scores))
            head = None
            if det.head_xy is not None:
                # copied like the body arrays above: a detector that reuses an
                # output buffer would otherwise have every frame's face points
                # pointing at the same array — its own.
                head = (np.asarray(det.head_xy, float).copy(),
                        np.asarray(det.head_scores, float).copy())
                heads += 1
            staged.append((frame, cam, body, head))
        if on_frame is not None:
            on_frame(i + 1, n)
        if on_progress is not None:
            on_progress(i + 1, n, f"Detecting keypoints — frame {i + 1} of {n}")

    # committed only now: every frame, or none of them
    if fields == "all":
        project.keypoint_model = _detector_keypoint_model(detector)
        project.head_source = getattr(detector, "head_source", None) or "nose"
        project.detector = (getattr(detector, "provenance", None)
                            or type(detector).__name__)
    for frame, cam, body, head in staged:
        if body is not None:
            (frame.kp2d_raw[cam], frame.scores_raw[cam],
             frame.kp2d[cam], frame.scores[cam]) = body
        if head is not None:
            # a hand-placed face point is kept exactly as a hand-placed body
            # joint is: the detector does not overrule the user, and
            # `redetect_head` reports that every correction was left alone
            head_xy, head_sc = head
            keep = (np.asarray(frame.head_corrected[cam], bool)
                    if respect_corrections else np.zeros(NUM_HEAD_KP, bool))
            frame.head2d[cam] = np.where(keep[:, None], frame.head2d[cam],
                                         head_xy)
            frame.head_scores[cam] = np.where(keep, frame.head_scores[cam],
                                              head_sc)
    if fields == "all":
        # The merge above is exactly where the midpoint rule breaks: a kept
        # hand correction on a SHOULDER sits beside the detector's own NECK,
        # which is the midpoint of the shoulders the user rejected. Restored
        # here, from the layout just recorded, so the take is never
        # triangulated from a neck that contradicts its shoulders. A neck the
        # user placed by hand is `corrected` and survives both steps.
        derive_midpoints(project)
    return heads


# Epipolar tolerance as a fraction of the image DIAGONAL. A flat 30 px was
# implicitly tuned for ~1080p; on the client's 3072x4080 phone captures that
# is 0.7% of image height, tight enough to cut through the middle of a
# legitimate distribution and silently delete a third of every pose. This
# reproduces ~30 px at 1080p and scales with the sensor.
_EPI_THR_FRAC = 0.014


# The data-driven gate: k x a robust scale of the take's OWN Sampson
# distribution, floored so a near-perfect take does not gate itself to
# nothing, and capped at _EPI_THR_FRAC of the SMALLER image's diagonal.
# k = 6 measured on the client take: median 4.91 px -> 29.4 px, which still
# clears that take's own tail (max 29.38 px, 0 of 388 pairs rejected) while
# being 2.4x tighter than the 71.5 px the image-only rule gave. k = 5 gives
# 24.6 px and clips the tail; the floor is what k = 5 could not supply.
_EPI_K = 6.0
_EPI_FLOOR_PX = 25.0


def per_image_allowances(rig: CalibratedRig) -> dict[str, float]:
    """Point-to-line disagreement tolerated IN EACH IMAGE, in its own pixels.

    _EPI_THR_FRAC of that camera's own diagonal — 71.5 px on the client's
    3072x4080 left view, 35.8 px on its 1536x2048 right one. The Sampson
    distance the gate's other half judges is (d_L^-2 + d_R^-2)^-1/2: it is
    below BOTH per-image distances and is dominated by the lower-resolution
    one, so on a 2:1 mixed rig it is not in either image's pixels. These two
    numbers are, and each is measured against the image it belongs to.
    """
    return {c: float(_EPI_THR_FRAC * np.hypot(*rig.intr[c].image_size))
            for c in (CAM_LEFT, CAM_RIGHT)}


def point_line_distances(pt_left, pt_right, F) -> tuple[float, float]:
    """Per-image point-to-epipolar-line distances (d_L, d_R) in px.

    Sampson is one number for the pair; these two say how far the observation
    sits from its partner's epipolar line IN EACH IMAGE, which is the only
    form comparable across an asymmetric rig once each is divided by its own
    image diagonal. `F` is the rig's fundamental matrix, passed in because it
    is a constant of the rig and rebuilding it per pair is 388 matrix
    inversions on the client take.

    The ONE implementation: `pose3d.quality._point_line_px` is a thin alias
    for this (it used to be a verbatim copy) — the gate and the metric that
    reports on the gate must not be able to disagree about what "how far
    apart are these two views" means, and
    test_cross_view.py::test_the_gate_and_the_metric_measure_the_same_thing
    is the guard on that.
    """
    pt_left = np.asarray(pt_left, float).reshape(2)
    pt_right = np.asarray(pt_right, float).reshape(2)
    if np.isnan(pt_left).any() or np.isnan(pt_right).any():
        return float("nan"), float("nan")
    xl = np.array([pt_left[0], pt_left[1], 1.0])
    xr = np.array([pt_right[0], pt_right[1], 1.0])
    num = abs(float(xr @ F @ xl))
    lr = F @ xl                      # epipolar line of xl, in the RIGHT image
    ll = F.T @ xr                    # epipolar line of xr, in the LEFT image
    nl = float(np.hypot(ll[0], ll[1]))
    nr = float(np.hypot(lr[0], lr[1]))
    return (num / nl if nl > 1e-12 else float("nan"),
            num / nr if nr > 1e-12 else float("nan"))


def epipolar_gate(median_px: float, ceiling_px: float) -> float:
    """The gate from a take's own median disagreement. THE formula.

    Exposed so the sidebar can state the gate from the distribution it already
    has (`pose3d.quality`'s epipolar block) instead of keeping a second copy
    of the arithmetic; `epipolar_threshold` is this same formula fed from a
    project.
    """
    if not np.isfinite(median_px):
        return float(ceiling_px)
    return float(min(max(_EPI_K * float(median_px), _EPI_FLOOR_PX),
                     float(ceiling_px)))


def epipolar_threshold(rig: CalibratedRig,
                       project: ProjectData | None = None) -> float:
    """Pixels of epipolar disagreement tolerated by the cross-view gate.

    Data-driven when a project is supplied: clip(6 x median Sampson over the
    CORE joints, 25 px, 1.4 % of the SMALLER image's diagonal) — sized from
    the core set and applied to every joint, see the loop below. The
    image-only rule this replaces
    was 71.5 px on the client take against an observed median of 4.91 px —
    14.6x the data, 3x its p99, and provably inert: it rejected 0 of 388 pairs
    and caught the ankle-on-knee hallucination its own docstring names in only
    15 of 26 frames. It was also scaled from the LEFT image alone, so on this
    2:1 rig it was 2.79 % of the right image's diagonal, and it stayed silent
    through ~7 deg of extrinsic error.

    Without a project there is no distribution to measure, so this returns the
    ceiling — the widest the gate may ever be. That is the honest answer for a
    caller holding a rig and nothing else, and it is still tighter than the
    old rule whenever the low-resolution camera is not the left one.
    """
    ceiling = min(per_image_allowances(rig).values())
    if project is None or not project.frames:
        return float(ceiling)
    # SIZED from the core set, APPLIED to all 17. A big toe is the noisiest
    # point the detector produces — small, often motion-blurred, frequently
    # half out of frame — and this median decides which CORE keypoints get
    # dropped. On the client take, Run Detection alone moved the threshold
    # 28.049 -> 28.998 px with nothing else changed. The toes are still gated
    # (`validate_cross_view` judges every joint); they just do not buy the
    # body a looser gate first.
    dists = []
    for frame in project.frames:
        for j in CORE_INDEX:
            e = epipolar_distance(
                frame.kp2d[CAM_LEFT][j], frame.kp2d[CAM_RIGHT][j],
                rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
            if np.isfinite(e):
                dists.append(e)
    if not dists:
        return float(ceiling)
    return epipolar_gate(float(np.median(dists)), ceiling)


def cross_view_verdict(pt_left, pt_right, rig: CalibratedRig, F,
                       epi_thr: float, allow: dict[str, float],
                       ) -> tuple[float, bool]:
    """THE cross-view test for one pair of observations: `(sampson_px, bad)`.

    Two halves — the Sampson distance against the take's gate, and the
    point-to-line distance in EACH image against 1.4 % of that image's own
    diagonal — and `bad` is True when EITHER is exceeded. Never both: d_R is
    systematically the smaller of the two (median 5.57 px vs 10.21 px on the
    client rig), so requiring both to exceed their own 1.4 % would need
    d_R > 35.8 px when its measured maximum is 33.8 — strictly more permissive
    than the rule it replaces, which is the opposite of the intent.

    One function because every caller asks the same question:
    `cross_view_rejection` turns this verdict into the view that loses (and
    from there `validate_cross_view` writes the mask, `revalidate_joint`
    re-derives one entry of it after a drag, and `ui.model.joint_states`
    explains a joint the mask has not caught up with yet), and
    `quality.take_quality` reports on the answer. When those disagree the
    user is told a joint was rejected for a reason it was not, or shown a
    green dot on a joint the next recompute will refuse. The Sampson distance
    is returned as well as the verdict because the tooltip names the number.
    """
    pt_left = np.asarray(pt_left, float)
    pt_right = np.asarray(pt_right, float)
    if np.isnan(pt_left).any() or np.isnan(pt_right).any():
        return float("nan"), False
    e = epipolar_distance(pt_left, pt_right,
                          rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                          rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    d_l, d_r = point_line_distances(pt_left, pt_right, F)
    bad = bool((np.isfinite(e) and e > epi_thr)
               or (np.isfinite(d_l) and d_l > allow[CAM_LEFT])
               or (np.isfinite(d_r) and d_r > allow[CAM_RIGHT]))
    return float(e), bad


def cross_view_rejection(frame, j: int, rig: CalibratedRig, F,
                         epi_thr: float, allow: dict[str, float],
                         ) -> tuple[float, str | None]:
    """`(sampson_px, the view the gate would refuse)` — or `None` for a keep.

    The WHOLE rule, and it writes nothing. `cross_view_verdict` answers only
    the first half of it ("do these two views disagree past the gate?"); the
    second half is who loses, and its last clause is that a pair whose BOTH
    views were placed by hand is KEPT however far apart they look — the user
    has overruled the gate, and the joint goes on to be triangulated, fitted
    and exported.

    It is a separate, pure function because two callers need the same answer
    and only one of them may write: `_judge` sets the mask from it, and
    `ui.model.joint_states` uses it to explain a joint whose mask is stale.
    When those two knew different rules, exactly the joints the gate had
    decided to trust — the hand-corrected ones — came back purple with a
    tooltip claiming they were not triangulated, permanently, while their 3D
    was in the export.
    """
    e, bad = cross_view_verdict(frame.kp2d[CAM_LEFT][j],
                                frame.kp2d[CAM_RIGHT][j], rig, F, epi_thr,
                                allow)
    if not bad:
        return e, None
    sl = frame.scores[CAM_LEFT][j]
    sr = frame.scores[CAM_RIGHT][j]
    # drop the worse (lower-confidence) view, unless it was hand-corrected
    drop_left = np.nan_to_num(sl) <= np.nan_to_num(sr)
    cam = CAM_LEFT if drop_left else CAM_RIGHT
    if frame.corrected[cam][j]:
        cam = CAM_RIGHT if drop_left else CAM_LEFT      # try the other view
        if frame.corrected[cam][j]:
            return e, None                              # both corrected: keep
    return e, cam


def _judge(frame, j: int, rig: CalibratedRig, F, epi_thr: float,
           allow: dict[str, float]) -> int:
    """Write `frame.rejected[*][j]` from this frame's current 2D. 0 or 1.

    The caller has already cleared both flags for `j`; this only ever sets
    one, on the view `cross_view_rejection` names.
    """
    _, cam = cross_view_rejection(frame, j, rig, F, epi_thr, allow)
    if cam is None:
        return 0
    frame.rejected[cam][j] = True
    return 1


def revalidate_joint(frame, joint: int, rig: CalibratedRig,
                     epi_thr: float) -> None:
    """Re-derive the mask for ONE joint of ONE frame, after its 2D moved.

    The live re-solve's half of `validate_cross_view`, and it exists so the
    live path and the batch path cannot reach different poses from the same
    2D. `ui.model._retriangulate` calls this before triangulating: a drag
    that reconciles the two views clears the flag on both, a drag that does
    not leaves the pair refused exactly as the next recompute would refuse
    it, and a hand-placed point still wins because `frame.corrected` is
    never auto-rejected.

    The gate `epi_thr` is the take-wide one the caller is holding: a median
    over every pair in the take, which one dragged point cannot move.
    """
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    for c in (CAM_LEFT, CAM_RIGHT):
        frame.rejected[c][joint] = False
    if np.isfinite(epi_thr):
        _judge(frame, int(joint), rig, F, float(epi_thr),
               per_image_allowances(rig))


def validate_cross_view(project: ProjectData, rig: CalibratedRig,
                        epi_thr: float | None = None) -> int:
    """Mask 2D observations that are geometrically inconsistent across views.

    When a joint is occluded/out-of-frame in one camera, the detector often
    hallucinates it (e.g. an ankle collapsed onto the knee). Such a point can
    never correspond to the same 3D location the other camera sees, so its
    epipolar distance is large. Every joint present in BOTH views is put
    through `cross_view_verdict` (the Sampson distance against `epi_thr`, and
    the point-to-line distance in EACH image against 1.4 % of that image's
    own diagonal, rejected when EITHER is exceeded) and the loser is the
    observation in the LOWER-confidence view, marked
    `frame.rejected[cam][j] = True`. It is then not
    triangulated (the 3D point drops out too, since a joint needs both views),
    but it is still drawn, still draggable, and still in the file.
    User-corrected joints are trusted and never auto-rejected.

    NON-DESTRUCTIVE, and that is the whole point: this used to write NaN into
    `kp2d` and 0.0 into `scores`, which `io_project` then persisted. A rig
    wrong by ~12 deg rejects a quarter of a take that way, and fixing the
    calibration recovered NOTHING — the observations were gone from the file,
    hidden in the 2D views, and only a full re-detection could bring them
    back. The mask is rebuilt from scratch here on every call, so a recompute
    with a better rig reinstates every observation it no longer objects to.

    Returns the number of CORE observations rejected. Every rejection is
    masked, toes included — the mask is what colours a handle — but the count
    is the one the user is told (`rejection_note`, whose denominator is the
    core set, and `quality.gap_stats["rejected"]`, which is core-only for the
    same reason). A take whose toes the gate refuses must not read as a worse
    calibration than the same take with its feet out of frame.
    """
    if epi_thr is None:
        epi_thr = epipolar_threshold(rig, project)
    allow = per_image_allowances(rig)
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    dropped = 0
    for frame in project.frames:
        # Re-derived, never accumulated: a mask left over from the last rig
        # would go on rejecting joints this one is happy with, which is the
        # destructive behaviour again with an extra step.
        for c in (CAM_LEFT, CAM_RIGHT):
            frame.rejected[c][:] = False
        for j in range(NUM_JOINTS):
            # judged on every joint, counted on the core ones
            n = _judge(frame, j, rig, F, epi_thr, allow)
            dropped += n if j in _CORE_SET else 0
    return dropped


def gated_kp2d(frame) -> dict[str, np.ndarray]:
    """This frame's 2D with the rejected observations removed — a COPY.

    The gate's verdict applied to the arithmetic and nowhere else: everything
    that reads `Frame.kp2d` keeps seeing what the detector and the user put
    there, and only triangulation is denied the observations the gate objects
    to. The LIVE path goes through here too (`ui.model._retriangulate` calls
    `revalidate_joint` and then this), so a drag and a recompute cannot reach
    different 3D from the same 2D. What a drag buys the user is not an
    exemption from the gate but a re-judgement of the pair: `Frame.set_kp`
    clears the mask, `frame.corrected` protects the point they placed, and
    the view they did not touch is the one that loses if the two still
    disagree.
    """
    out = {}
    for cam in (CAM_LEFT, CAM_RIGHT):
        xy = np.array(frame.kp2d[cam], dtype=float, copy=True)
        xy[np.asarray(frame.rejected[cam], bool)] = np.nan
        out[cam] = xy
    return out


def face_protect(frame, head_source: str) -> tuple[int, ...]:
    """Which face points this frame's hand corrections put beyond the gate.

    Under the COCO-17 convention the canonical HEAD **is** face point 0 — one
    physical detection judged by two gates, and they must not disagree about
    it. `cross_view_rejection`'s last clause keeps a body pair whose BOTH views
    were hand-placed: the user has overruled the gate and the joint goes on to
    be triangulated, fitted and exported. The face gate had no such clause, so
    a HEAD corrected in both views kept `pose3d[HEAD]` (the neck went on aiming
    at the user's point) while `head3d[0]` was NaN'd and the nose roll switched
    off for that frame — on exactly the frame the user worked hardest on, with
    no nose dot drawn under this convention to explain it.

    BOTH views, never either: that is the joint gate's rule, and it is what
    keeps the two answers equal. With one view corrected the joint gate drops
    the OTHER view and `pose3d[HEAD]` goes NaN too, so protecting the nose
    there would re-create the same disagreement pointing the other way.

    Under the skull convention the HEAD joint and the nose are different
    detections (~86 px apart on frame 0 of the client take) and `corrected`
    says nothing about the nose, so nothing is protected.

    The JOINT's flag and not the nose's own `head_corrected`, even though the
    face points now have one: this gate is about the pair the CANONICAL HEAD
    forms, and it must return the same answer as `cross_view_rejection` gives
    that joint or the two disagree about one detection — which is the defect
    it was written for. Under this convention a HEAD drag sets both flags
    anyway (`ui.model._resolve_joint` syncs the nose with its provenance), so
    the two readings agree on every path a user can take.

    A separate, pure function because all four callers of `triangulate_face`
    must reach the same `head3d` from the same 2D, and this is the only thing
    they would otherwise each have to re-derive.
    """
    if head_source != "nose":
        return ()
    j = int(Joint.HEAD)
    if frame.corrected[CAM_LEFT][j] and frame.corrected[CAM_RIGHT][j]:
        return (0,)
    return ()


def triangulate_face(frame, rig: CalibratedRig, epi_thr: float,
                     F=None, allow: dict[str, float] | None = None,
                     protect: Iterable[int] = ()) -> None:
    """Fill this frame's `head3d`, NaN where the two views disagree.

    The face points reach the character exactly as the canonical joints reach
    the bone fit, so they are judged by the same rule: `cross_view_verdict`,
    the same gate, the same per-image allowances. A nose the detector put on
    the ear in one view triangulates to a point metres from the head, and the
    head is then aimed at it — the failure the cross-view check has always
    caught for a knee, on the one path that never asked.

    What it does NOT do is write a mask. `Frame.rejected` is a per-JOINT array
    the bone fit and the camera views read; the face points have no entry in
    it and no dot colour to explain one — their one override is `protect`,
    which the nose borrows from the HEAD joint it IS under the COCO-17
    convention (see `face_protect`). They DO keep a `head_corrected` flag of
    their own, but that records who placed a point, not whether the two views
    agree about where it is, and a gate must not be overruled by the first of
    those. The verdict lives only in the 3D: a refused pair is NaN, and the
    character falls back to the neck's own aim for that frame. `head2d` is
    left exactly as the detector and the user wrote it, so the next
    calibration — or a drag that reconciles the pair — reinstates the point
    with no state to undo.

    ONE function because three callers must reach the same `head3d` from the
    same 2D: the batch recompute, the face re-detect, and the drag path. When
    the drag path had its own arithmetic, dragging a face point revived a
    3D the next recompute deleted.

    `protect` names face indices left UNGATED — the user's override, since the
    gate cannot see one. Ask `face_protect` for it rather than assembling it
    per caller: the whole point is that every caller passes the same set for
    the same frame. Empty by default, so a caller that does not know this
    project's head convention cannot accidentally claim one.

    `F` and `allow` are derived from `rig` when omitted; pass them when
    looping over a take, where they are the same for every frame.
    """
    if F is None:
        F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                               rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    if allow is None:
        allow = per_image_allowances(rig)
    kept = {int(k) for k in protect}
    head3d = triangulate_points(
        frame.head2d[CAM_LEFT], frame.head2d[CAM_RIGHT],
        rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
        rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    for k in range(NUM_HEAD_KP):
        if k in kept:
            continue
        _, bad = cross_view_verdict(frame.head2d[CAM_LEFT][k],
                                    frame.head2d[CAM_RIGHT][k],
                                    rig, F, epi_thr, allow)
        if bad:
            head3d[k] = np.nan
    frame.head3d = head3d


# Above this share of keypoints rejected, the cause is the calibration rather
# than the detector, and the user has no way to tell those apart from the
# symptoms (gaps in the 2D views, a sparse 3D pose).
_REJECT_NOTE_FRAC = 0.15


def rejection_note(dropped: int, n_frames: int) -> str:
    """One sentence for the user about keypoints the cross-view check threw
    away, or "" when the amount is unremarkable.

    The denominator is the core set: a toe the detector never produced was
    never rejected either, and the percentage the client reads must not move
    because the skeleton grew a joint.
    """
    total = max(1, n_frames * len(CORE_INDEX))
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

    By default first MASKS cross-view-inconsistent observations (occlusion
    hallucinations) so they don't corrupt the 3D pose, and triangulates a
    masked COPY — the mask is a verdict about this rig, and it must not reach
    the 2D the user can see and drag. Returns how many were rejected, so
    callers can tell the user: a bad calibration rejects good detections
    wholesale, which looks exactly like a detection failure.

    `validate=False` means "do not RE-derive the mask", not "do not gate":
    whatever mask the frames already carry is still applied, because it is
    part of the frame and `gated_kp2d` is the only route to the arithmetic.
    On a project loaded from disk that mask was written by a previous
    session's rig, so pass False only when you know the mask is current or
    empty (the tests that use it triangulate freshly loaded fixtures, which
    carry none).

    The FACE points are gated on every call whichever way `validate` goes:
    their verdict is not a mask (see `triangulate_face`), so there is nothing
    of a previous rig's to preserve or destroy — it is re-derived from this
    rig here, as it is on the drag path and the face re-detect. The one thing
    that does carry over is the user's own override, read off `corrected` by
    `face_protect`, so a recompute cannot delete a nose a drag was right to
    keep.
    """
    dropped = 0
    for frame in project.frames:
        # A gap-fill flag describes the 3D that is about to be replaced. Reset
        # before anything else touches the take, so a flag can never outlive
        # the fill that set it (a joint recovered by a re-detect or a hand
        # correction must come back as a measurement, not stay "interpolated").
        frame.filled[:] = False
    # measured once for the take, not once per frame — and once for BOTH
    # gates: the joints and the face points must be judged by the same number
    # or a nose could be kept by a threshold no knee was ever measured against
    epi_thr = epipolar_threshold(rig, project)
    if validate:
        dropped = validate_cross_view(project, rig, epi_thr)
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    allow = per_image_allowances(rig)
    for frame in project.frames:
        kp = gated_kp2d(frame)
        frame.pose3d = triangulate_points(
            kp[CAM_LEFT], kp[CAM_RIGHT],
            rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
            rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
        # the face points ride the same geometry and the same cross-view gate;
        # they are not bone-fitted, they only orient the head. The project is
        # what knows whether the nose and the HEAD joint are one detection, so
        # the override the joint gate honours is passed in from here.
        triangulate_face(frame, rig, epi_thr, F, allow,
                         face_protect(frame, project.head_source))
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

    The dependency runs BOTH ways: a drag that loses (or reinstates) an
    observation also changes what the two NEIGHBOURING frames may interpolate,
    since their fill reads this frame's `pose3d`. The live caller
    (`ui.model.ProjectModel._refit_frame`) therefore re-runs this for
    `index ± 1` as well and re-fits either of them whose flags moved. Reading
    only the immediate neighbours is what bounds that: it cannot cascade.
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
    # bones never observed in this take, by name -> a default proportion was
    # used. Named, not counted: "2 bones fell back" does not tell the user
    # whether it was the two collarbones (harmless) or both thighs (the whole
    # lower body posed off a table).
    fallback_bones: list[str] = field(default_factory=list)
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
                f"{len(self.fallback_bones)} bone(s) were never seen in this "
                f"take ({', '.join(self.fallback_bones)}), so a default body "
                f"proportion — scaled to this subject — was used for them")
        if self.gaps_filled:
            parts.append(
                f"{self.gaps_filled} joint(s) were missing for a single frame "
                f"and were interpolated from the frames either side — they are "
                f"drawn hollow")
        return "; ".join(parts)


def bone_length_targets(project: ProjectData) -> tuple[dict, list[str]]:
    """(target length per bone, the names of the bones that fell back).

    Measured as the median over the whole take — the subject's own skeleton,
    not a generic body — with a default proportion only where a bone was never
    observed at all. The single source of these numbers: the batch fit and the
    live manual-correction re-solve both call this, so they cannot drift.

    The defaults are PROPORTIONS of the subject's own measured spine, not a
    1.75 m adult's centimetres. A fallback length sits in the least-squares as
    a residual whether or not the joint it belongs to is being solved for, so
    an unmeasurable bone used to drag every observed joint around it toward a
    skeleton 14x the size of the client's mannequin.

    The REPORT names core bones only. A bone into an extremity is unmeasurable
    on every take detected before toe points existed, so listing it would put
    two lines the user can do nothing about in front of every project they
    own. The fit is unchanged: those bones still get their fallback target,
    and a toe no view saw is frozen out of the solve anyway.
    """
    raw = np.stack([f.pose3d for f in project.frames]) if project.frames \
        else np.zeros((0, NUM_JOINTS, 3))
    measured = measure_bone_lengths(raw)
    fb = fallback_bone_lengths(reference_from_measured(measured))
    extremity = {int(j) for j in EXTREMITY_JOINTS}
    fallen = [f"{JOINT_NAMES[a]}-{JOINT_NAMES[b]}"
              for (a, b), v in measured.items()
              if v <= 1e-6 and int(b) not in extremity]
    return {k: (v if v > 1e-6 else fb[k]) for k, v in measured.items()}, fallen


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

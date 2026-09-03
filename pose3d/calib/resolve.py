"""Resolve a CalibratedRig for an imported project.

Priority, per the requirement:
  1. Use uploaded intrinsics + extrinsics when present.
  2. If extrinsics are missing, estimate them from ArUco markers shared between
     the two views (a common marker becomes the world frame).
  3. If intrinsics are missing, fall back to an approximate pinhole model from
     the image size (flagged as approximate — coplanar tags cannot recover
     intrinsics reliably).
  4. If no calibration was uploaded and no ArUco markers can be detected in any
     image pair, report failure: "calibration was not successful", naming which
     camera saw which tags in how many frames.

Step 2 is not first-hit any more. Every frame is searched, every tag is scored
(`score_tags` / `admit_tags` / `score_branch_pairs`), and the choice plus the
evidence for it is written to `calibration/report.json` by `save_rig` — enough
to rebuild the rig bit for bit (`rebuild_extrinsics_from_report`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pose3d.calib.extrinsics import (
    Extrinsics, detect_markers, estimate_extrinsics_for_marker, make_detector,
)
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS, ProjectData
from pose3d.pipeline import CalibratedRig

# ArUco dictionaries to try, most likely first (the client's tags are 6x6).
_DICT_NAMES = [
    "DICT_6X6_250", "DICT_4X4_50", "DICT_5X5_250",
    "DICT_APRILTAG_36h11", "DICT_ARUCO_ORIGINAL",
]
DEFAULT_DICTS = [getattr(cv2.aruco, n) for n in _DICT_NAMES
                 if hasattr(cv2.aruco, n)]
DICT_NAME_BY_ID = {getattr(cv2.aruco, n): n for n in _DICT_NAMES
                   if hasattr(cv2.aruco, n)}

# --- admission thresholds -------------------------------------------------
# A tag is admitted only if its IPPE reprojection rms stays under this and
# within 3x the median tag's. Measured on the client's take: tags 14/15/17 sit
# at 0.23/0.48/0.82 px and the tag stuck over a curved sweep (13) at 2.41 px —
# no overlap, and the discriminator is planarity, not viewing angle (tag 13 is
# seen at an unremarkable 33.5 deg).
TAG_RMS_MAX_PX = 1.5
TAG_RMS_MEDIAN_FACTOR = 3.0
# A square marker has two IPPE poses. Believe the better one only when it is
# this much better; medians are 8.7-13.8 for the tags whose pose is real and
# 1.26 for tag 13.
BRANCH_RATIO_MIN = 2.0
# The scene is static, so the world tag must hold still in each camera. Above
# these the pose is moving by more than solve noise does (measured over the
# client's take: 1.02 deg and 8.4 mm, both of which are noise).
MOTION_WARN_DEG = 2.0
MOTION_WARN_MM = 30.0


def _to_jsonable(o):
    # bool before int: True is an int subclass, and "moved": 1 in a provenance
    # file reads as a count
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    return o


def save_rig(rig: CalibratedRig, calib_dir, report: dict | None = None) -> None:
    """Persist a rig to <calib_dir> in the format app._load_rig expects.

    `report` is the provenance record from `resolve_calibration`; when given it
    is written as report.json and its recorded vertical is copied into
    extrinsics.json, where the view and the export read it.
    """
    calib_dir = Path(calib_dir)
    calib_dir.mkdir(parents=True, exist_ok=True)
    rig.intr[CAM_LEFT].save(calib_dir / "left_intrinsics.json")
    rig.intr[CAM_RIGHT].save(calib_dir / "right_intrinsics.json")
    doc = {
        "left": {"R": rig.ext[CAM_LEFT].R.tolist(),
                 "t": rig.ext[CAM_LEFT].t.tolist()},
        "right": {"R": rig.ext[CAM_RIGHT].R.tolist(),
                  "t": rig.ext[CAM_RIGHT].t.tolist()},
    }
    if report and report.get("world_up") is not None:
        doc["world_up"] = _to_jsonable(report["world_up"])
        doc["world_up_source"] = report.get("world_up_source")
        doc["world_up_spread_deg"] = _to_jsonable(
            report.get("world_up_spread_deg"))
    (calib_dir / "extrinsics.json").write_text(json.dumps(doc, indent=2),
                                               encoding="utf-8")
    if report is not None:
        (calib_dir / "report.json").write_text(
            json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")


def load_extrinsics_json(path):
    """Load an uploaded extrinsics file: {'left':{R,t},'right':{R,t}}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return (Extrinsics(R=np.array(d["left"]["R"], float), t=np.array(d["left"]["t"], float)),
            Extrinsics(R=np.array(d["right"]["R"], float), t=np.array(d["right"]["t"], float)))


def load_world_up(calib_dir):
    """The recorded vertical from a calibration folder, or None.

    Returns (up, source, spread_deg). None for a project calibrated before the
    vertical was recorded — those keep levelling on the subject, unchanged.

    `spread_deg` is None when the file does not carry one, or carries null.
    Reading that as 0.0 would turn a file that says nothing about its
    uncertainty into the most confident vertical the app can hold, and
    `orient.take_up` would then prefer it over the subject's body line; an
    unknown spread must be refused, so a hand-edited or externally produced
    extrinsics.json cannot smuggle in a vertical nothing validated.
    """
    try:
        d = json.loads(
            (Path(calib_dir) / "extrinsics.json").read_text(encoding="utf-8"))
        up = d["world_up"]
    except Exception:
        return None
    up = np.asarray(up, float).ravel()
    n = float(np.linalg.norm(up))
    if up.shape != (3,) or not np.isfinite(n) or n < 1e-9:
        return None
    spread = d.get("world_up_spread_deg")
    try:
        spread = None if spread is None else float(spread)
    except (TypeError, ValueError):
        spread = None
    if spread is not None and not np.isfinite(spread):
        spread = None
    return (up / n, str(d.get("world_up_source") or "recorded"), spread)


# Recorded in the report so the sense of the vertical is provenance too: a
# reader can tell an axis a body decided from one a camera guessed at.
SIGN_FROM_POSES = "NECK above ANKLE, from the triangulated poses"
SIGN_FROM_CAMERA_UP = "camera up proxy (no triangulated poses were available)"


def finalize_world_up(project, calib_dir) -> tuple | None:
    """Re-decide the SENSE of the recorded vertical from triangulated poses.

    `resolve_calibration` has to run BEFORE detection and triangulation — it is
    what makes triangulation possible — so at import time no frame has a
    `fitted3d` yet and `estimate_world_up` has to settle the sign from the
    camera-up proxy: "the phones were held the right way up". That proxy is
    wrong for exactly one rig, and it is a rig people build: both cameras
    mounted rotated ~180 deg. `estimate_world_up` then returns [0, 0, -1] where
    the truth is [0, 0, 1], and every recalibrated project renders and exports
    upside down.

    So the sign is decided twice. This runs after triangulation, re-applies the
    NECK-above-ANKLE test to the poses that now exist, and rewrites
    extrinsics.json / report.json in place. It only ever FLIPS the axis — the
    direction estimated from the rig geometry is untouched — so nothing about
    the take's lean can leak into the vertical.

    Returns the finalised (up, source, spread_deg), or None when there is
    nothing to decide (no recorded vertical, or no pose with both a NECK and an
    ankle). Files are rewritten only when something actually changed, so
    re-opening a project is not a disk write.
    """
    from pose3d.geometry.gravity import sign_from_poses

    calib_dir = Path(calib_dir)
    ext_path = calib_dir / "extrinsics.json"
    try:
        doc = json.loads(ext_path.read_text(encoding="utf-8"))
        raw_up = doc["world_up"]
    except Exception:
        return None
    up = np.asarray(raw_up, float).ravel()
    n = float(np.linalg.norm(up))
    if up.shape != (3,) or not np.isfinite(n) or n < 1e-9:
        return None
    up = up / n

    frames = getattr(project, "frames", None) or []
    poses = [np.asarray(f.fitted3d, float) for f in frames
             if getattr(f, "fitted3d", None) is not None]
    if not poses:
        return None
    sign = sign_from_poses(up, np.stack(poses))
    if sign is None:                      # the poses cannot tell: leave it be
        return None

    flipped = sign < 0
    if flipped:
        up = -up
        doc["world_up"] = _to_jsonable(up)
    changed = flipped

    rep_path = calib_dir / "report.json"
    try:
        report = json.loads(rep_path.read_text(encoding="utf-8"))
    except Exception:
        report = None
    if report is not None:
        if flipped:
            report["world_up"] = _to_jsonable(up)
            report["world_up_candidates"] = {
                k: _to_jsonable(-np.asarray(v, float))
                for k, v in (report.get("world_up_candidates") or {}).items()}
        if report.get("world_up_sign_source") != SIGN_FROM_POSES:
            report["world_up_sign_source"] = SIGN_FROM_POSES
            changed = True
        if changed:
            rep_path.write_text(json.dumps(_to_jsonable(report), indent=2),
                                encoding="utf-8")
    if flipped:
        ext_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return load_world_up(calib_dir)


@dataclass
class CalibrationResult:
    ok: bool
    rig: CalibratedRig | None
    status: str          # 'uploaded' | 'aruco' | 'failed'
    message: str
    approximate: bool = False   # True if intrinsics were guessed from image size
    report: dict | None = None  # provenance; save_rig writes it to report.json


def _approx_intrinsics(image: np.ndarray) -> Intrinsics:
    """Rough pinhole model from image size: f≈max(w,h), principal point=centre.

    Physically this is a guess with no basis — the EXIF 35mm-equivalent says a
    Pixel 10 Pro shooting 3072x4080 is ~3570 px, not 4080. It is nonetheless
    what we use, because it is what MEASURES better: with extrinsics solved
    from a single planar marker using the same K, f=4080 gives a
    self-consistent stereo pair on the client's captures (median body epipolar
    4.91 px, nothing rejected by validate_cross_view) while the EXIF pair
    scores 10.9 px and the best focal a sweep could find scores 5.37.

    `intrinsics.focal_from_exif` is kept and tested to record that the number
    it used to return was computed WRONG (no DigitalZoomRatio), so that the
    revert of the EXIF focal is not re-litigated as "the approach was wrong".
    Two intrinsics stay unrecoverable from tags and must come from a
    checkerboard: distortion (k1 = 0.03 moves a tag corner 10.9 px but the
    subject 0.27 px, so the tags cannot fit it usefully) and the principal
    point (assumed centred; ±200 px costs 19 px of epipolar error).
    """
    h, w = image.shape[:2]
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=float)
    return Intrinsics(K=K, dist=np.zeros((1, 5), dtype=float),
                      image_size=(w, h), source="assumed")


# --------------------------------------------------------------------------
# tag observations
# --------------------------------------------------------------------------
def detect_all_tags(project: ProjectData, load_image, dictionaries=None):
    """Detect every tag in every frame of both cameras.

    Returns (dictionary_id, observations) where observations is
    {frame_id: {cam: {tag_id: (4,2) corners}}}. The dictionary chosen is the
    one seen in most frames by both cameras (ties broken by total detections,
    then by the order of DEFAULT_DICTS), so a stray false positive from
    another dictionary cannot win. (dictionary_id is None when nothing was
    detected at all.)

    Every image is read once and shown to every candidate detector: reading
    3072x4080 JPEGs is what costs, not detecting in them.
    """
    dictionaries = list(dictionaries or DEFAULT_DICTS)
    detectors = [(d, make_detector(d)) for d in dictionaries]
    obs = {d: {} for d in dictionaries}
    for frame in project.frames:
        images = {}
        for cam in CAMERAS:
            images[cam] = load_image(frame.images.get(cam))
        if any(img is None for img in images.values()):
            continue
        for dict_id, detector in detectors:
            per_cam = {}
            for cam in CAMERAS:
                corners, ids = detect_markers(images[cam], detector)
                per_cam[cam] = {int(i): np.asarray(c, float).reshape(4, 2)
                                for c, i in zip(corners, ids)}
            obs[dict_id][frame.frame_id] = per_cam

    def rank(d):
        frames = obs[d]
        common = sum(1 for f in frames.values()
                     if set(f[CAM_LEFT]) & set(f[CAM_RIGHT]))
        total = sum(len(f[cam]) for f in frames.values() for cam in CAMERAS)
        return (common, total)

    best = max(dictionaries, key=lambda d: (rank(d), -dictionaries.index(d)))
    if rank(best) == (0, 0):
        return None, {}
    return best, obs[best]


def observation_counts(observations) -> dict:
    """{cam: {"frames": n, "tags": {tag: n frames}, "empty_frames": n}}."""
    out = {}
    for cam in CAMERAS:
        tags: dict[int, int] = {}
        empty = 0
        for per_cam in observations.values():
            seen = per_cam.get(cam, {})
            if not seen:
                empty += 1
            for tid in seen:
                tags[tid] = tags.get(tid, 0) + 1
        out[cam] = {"frames": len(observations), "tags": dict(sorted(tags.items())),
                    "empty_frames": empty}
    return out


def failure_message(observations) -> str:
    """Name which camera saw which tags in how many frames.

    "No markers were detected" is not actionable; the client's right camera saw
    one tag in 11 of 26 frames and nothing in the other 15, and the fix is to
    move that camera.
    """
    counts = observation_counts(observations)
    n = len(observations)
    parts = []
    for cam in CAMERAS:
        c = counts[cam]
        if not c["tags"]:
            parts.append(f"{cam} camera saw no tags in {n}/{n} frames")
            continue
        tags = ", ".join(f"tag {t} in {k}/{n} frames"
                         for t, k in c["tags"].items())
        empty = (f" and nothing in {c['empty_frames']}"
                 if c["empty_frames"] else "")
        parts.append(f"{cam} camera saw {tags}{empty}")
    return ("Calibration was not successful: no calibration was uploaded and no "
            "ArUco tag was seen by both cameras in the same frame. "
            + "; ".join(parts)
            + " — move the cameras so at least two tags are in both frames.")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _solves(observations, frame_id, cam, tag_id, intr, marker_length):
    corners = observations.get(frame_id, {}).get(cam, {}).get(tag_id)
    if corners is None:
        return []
    return estimate_extrinsics_for_marker([corners], [tag_id], tag_id, intr,
                                          marker_length)


def _rot_angle(A, B) -> float:
    return float(np.degrees(np.arccos(
        np.clip((np.trace(np.asarray(A).T @ np.asarray(B)) - 1) / 2, -1, 1))))


def score_tags(observations, intr, marker_length) -> dict:
    """Per tag: IPPE reprojection rms and branch ratio, pooled over frames.

    `intr` is {cam: Intrinsics}. The rms is the planarity evidence — a tag that
    is not flat cannot be fitted by a square, whichever pose it is given — and
    the ratio is the branch evidence.
    """
    per: dict[int, dict] = {}
    for frame_id, per_cam in observations.items():
        for cam in CAMERAS:
            for tag_id in per_cam.get(cam, {}):
                s = _solves(observations, frame_id, cam, tag_id, intr[cam],
                            marker_length)
                if not s:
                    continue
                d = per.setdefault(tag_id, {"rms": [], "ratio": {c: [] for c in CAMERAS},
                                            "n": 0})
                d["rms"].append(s[0].err)
                d["n"] += 1
                if len(s) > 1:
                    d["ratio"][cam].append(s[1].err / max(s[0].err, 1e-9))
    out = {}
    for tag_id, d in sorted(per.items()):
        out[tag_id] = {
            "n_observations": d["n"],
            "rms_px": float(np.median(d["rms"])),
            "rms_max_px": float(np.max(d["rms"])),
            "branch_ratio": {c: (float(np.median(d["ratio"][c]))
                                 if d["ratio"][c] else None) for c in CAMERAS},
        }
    return out


def admit_tags(tag_scores: dict) -> tuple[list[int], dict[int, str]]:
    """Split tags into admitted / rejected-with-reason on planarity alone.

    Two thresholds, because either alone is fooled: an absolute cap (a set of
    uniformly bad tags must not admit itself) and 3x the median tag (a rig with
    good optics must not admit its worst tag just because 1.5 px is generous).
    """
    if not tag_scores:
        return [], {}
    med = float(np.median([v["rms_px"] for v in tag_scores.values()]))
    limit = min(TAG_RMS_MAX_PX, TAG_RMS_MEDIAN_FACTOR * med)
    admitted, rejected = [], {}
    for tag_id, v in sorted(tag_scores.items()):
        if v["rms_px"] <= limit:
            admitted.append(tag_id)
        else:
            rejected[tag_id] = (
                f"IPPE reprojection rms {v['rms_px']:.2f} px exceeds "
                f"{limit:.2f} px (cap {TAG_RMS_MAX_PX:.2f} px, "
                f"{TAG_RMS_MEDIAN_FACTOR:.0f}x the median tag's "
                f"{med:.2f} px) — the tag is not flat")
    return admitted, rejected


def score_branch_pairs(observations, frames, tag_id, intr, marker_length) -> dict:
    """Score all four (left branch, right branch) pairs for one tag.

    Both cameras are scored as ONE choice, never separately: two cameras
    picking the wrong branch do not cancel. On the client's take L1R1 returns a
    0.554 m baseline — indistinguishable from the correct pair's — while its
    relative pose is 104 deg out, so no baseline or plausibility check can
    catch it. What does catch it is that IPPE prefers branch 0 by 8.7x (left)
    and 13.4x (right): a pair is admissible only where each camera's own branch
    ratio does not contradict the branch that pair uses.

    Returns {(bl, br): {...}} with `admissible`, the cross-frame relative-pose
    spread and the median baseline for each pair.
    """
    ratios = {cam: [] for cam in CAMERAS}
    solves = {}
    for frame_id in frames:
        s = {cam: _solves(observations, frame_id, cam, tag_id, intr[cam],
                          marker_length) for cam in CAMERAS}
        if not (s[CAM_LEFT] and s[CAM_RIGHT]):
            continue
        solves[frame_id] = s
        for cam in CAMERAS:
            if len(s[cam]) > 1:
                ratios[cam].append(s[cam][1].err / max(s[cam][0].err, 1e-9))
    med_ratio = {cam: (float(np.median(ratios[cam])) if ratios[cam] else None)
                 for cam in CAMERAS}

    out = {}
    for bl in (0, 1):
        for br in (0, 1):
            rels, bases = [], []
            for s in solves.values():
                if len(s[CAM_LEFT]) <= bl or len(s[CAM_RIGHT]) <= br:
                    continue
                a, b = s[CAM_LEFT][bl], s[CAM_RIGHT][br]
                R = b.R @ a.R.T
                t = b.t - R @ a.t
                rels.append(R)
                bases.append(float(np.linalg.norm(t)))
            if not rels:
                continue
            spread = max((_rot_angle(rels[i], rels[j])
                          for i in range(len(rels))
                          for j in range(i + 1, len(rels))), default=0.0)
            # a branch other than the best one is only admissible where that
            # camera's IPPE could not tell the two apart
            ok = True
            for cam, b in ((CAM_LEFT, bl), (CAM_RIGHT, br)):
                r = med_ratio[cam]
                if b != 0 and (r is None or r >= BRANCH_RATIO_MIN):
                    ok = False
            out[(bl, br)] = {
                "admissible": ok,
                "n_frames": len(rels),
                "relpose_spread_deg": float(spread),
                "baseline_m": float(np.median(bases)),
            }
    return {"pairs": out, "branch_ratio": med_ratio}


def pick_branch_pair(pairs: dict) -> tuple:
    """Which (left branch, right branch) pair to calibrate from.

    Admissible pairs first (see `score_branch_pairs`), then MOST FRAMES, then
    lowest cross-frame relative-pose spread, then the lowest index so the
    choice is reproducible.

    Frame count has to come first because the spreads are not comparable
    across pairs scored on different numbers of frames: the spread is a
    maximum over pairwise angles, so a pair present in a single frame has no
    pairs to take a maximum over and scores 0.0 — the best possible value, on
    no evidence whatsoever. Healthy rigs never reach the tie-break at all
    (branch 1 is inadmissible wherever a camera's median IPPE ratio separates
    the branches, leaving {(0, 0)}), but on an ambiguous rig it would be
    decided by whichever wrong pair was seen least.
    """
    ok = {k: v for k, v in pairs.items() if v["admissible"]} or pairs
    return min(ok, key=lambda k: (-ok[k]["n_frames"],
                                  ok[k]["relpose_spread_deg"], k))


def camera_motion_check(observations, frames, tag_id, intr, marker_length,
                        reference: dict) -> dict:
    """Has a camera moved during the take?

    The tags are static, so each camera's pose relative to the world tag must
    be constant. Nothing else in the app notices a camera that was nudged
    between frames, and the reconstruction just quietly bends.

    Per camera: the largest deviation from the median pose over the frames that
    saw the world tag, in degrees and mm. The branch is picked per frame as the
    one closest to the chosen calibration pose, so a branch flip is not
    reported as motion.
    """
    out = {}
    for cam in CAMERAS:
        ref = reference[cam]
        Rs, Cs = [], []
        for frame_id in frames:
            s = _solves(observations, frame_id, cam, tag_id, intr[cam],
                        marker_length)
            if not s:
                continue
            pick = min(s, key=lambda x: _rot_angle(x.R, ref.R))
            Rs.append(pick.R)
            Cs.append((-pick.R.T @ pick.t.reshape(3, 1)).ravel())
        if not Rs:
            out[cam] = {"n_frames": 0, "max_rotation_deg": None,
                        "max_centre_mm": None, "moved": None}
            continue
        Cs = np.asarray(Cs)
        med_c = np.median(Cs, axis=0)
        med_R = min(Rs, key=lambda R: sum(_rot_angle(R, B) for B in Rs))
        rot = max(_rot_angle(R, med_R) for R in Rs)
        cen = float(np.max(np.linalg.norm(Cs - med_c, axis=1))) * 1000.0
        out[cam] = {
            "n_frames": len(Rs),
            "max_rotation_deg": float(rot),
            "max_centre_mm": cen,
            "moved": bool(rot > MOTION_WARN_DEG or cen > MOTION_WARN_MM),
        }
    return out


@dataclass
class RigSolution:
    """The chosen calibration plus everything that justifies the choice."""
    ok: bool
    ext: dict = field(default_factory=dict)      # cam -> Extrinsics
    report: dict = field(default_factory=dict)
    message: str = ""


def solve_rig_from_observations(
    observations, frame_order, intr, marker_length: float,
    dictionary: str | None = None, poses=None,
) -> RigSolution:
    """Pick a world tag, a world frame and an IPPE branch pair, with evidence.

    `observations` is {frame_id: {cam: {tag_id: corners}}} (see
    `detect_all_tags`), `frame_order` fixes which frame wins ties, `intr` is
    {cam: Intrinsics}.

    The world frame is the FIRST frame in `frame_order` in which the chosen tag
    is seen by both cameras. Choosing the "best" frame instead would be a
    different calibration for every re-import as detections shift by a corner
    pixel; a rule the user can predict is worth more than a fraction of a
    pixel, and the pose is checked against every other frame anyway (the
    relative-pose spread is in the report).
    """
    tag_scores = score_tags(observations, intr, marker_length)
    admitted, rejected = admit_tags(tag_scores)

    # tags common to both cameras in at least one frame, with those frames
    common: dict[int, list[str]] = {}
    for frame_id in frame_order:
        per_cam = observations.get(frame_id)
        if not per_cam:
            continue
        for tag_id in sorted(set(per_cam.get(CAM_LEFT, {}))
                             & set(per_cam.get(CAM_RIGHT, {}))):
            common.setdefault(tag_id, []).append(frame_id)

    usable = [t for t in admitted if t in common]
    if not usable:
        blocked = sorted(set(common) - set(usable))
        if blocked:
            # a tag can be in `common` and in neither list: its solve failed,
            # so it was never scored and so never admitted or rejected. Saying
            # "were rejected — ." names no reason at all.
            reasons = [rejected.get(t, "no usable pose could be solved from "
                                       "its corners") for t in blocked]
            msg = ("Calibration was not successful: the only tags both cameras "
                   f"saw ({', '.join(str(t) for t in blocked)}) were rejected — "
                   + "; ".join(reasons) + ".")
        else:
            msg = failure_message(observations)
        return RigSolution(False, {}, {
            "dictionary": dictionary,
            "marker_length_m": float(marker_length),
            "per_tag_rms_px": {t: v["rms_px"] for t, v in tag_scores.items()},
            "tags_admitted": admitted,
            "tags_rejected_with_reason": rejected,
            "observation_counts": observation_counts(observations),
        }, msg)

    # Most frames seen by both cameras wins: cross-frame agreement is the only
    # check on the branch choice, and it needs frames. Ties by lower rms.
    world_tag = max(usable, key=lambda t: (len(common[t]),
                                           -tag_scores[t]["rms_px"], -t))
    frames = common[world_tag]
    world_frame = frames[0]

    scored = score_branch_pairs(observations, frames, world_tag, intr,
                                marker_length)
    pairs = scored["pairs"]
    branch = pick_branch_pair(pairs)

    solves = {cam: _solves(observations, world_frame, cam, world_tag,
                           intr[cam], marker_length) for cam in CAMERAS}
    picked = {CAM_LEFT: solves[CAM_LEFT][branch[0]],
              CAM_RIGHT: solves[CAM_RIGHT][branch[1]]}
    ext = {cam: picked[cam].extrinsics for cam in CAMERAS}

    # tag layout in world coords, for the vertical: every admitted tag the LEFT
    # camera sees in the world frame, mapped through that camera's pose.
    layout = {}
    R_w, t_w = ext[CAM_LEFT].R, ext[CAM_LEFT].t
    for tag_id in admitted:
        s = _solves(observations, world_frame, CAM_LEFT, tag_id,
                    intr[CAM_LEFT], marker_length)
        if s:
            layout[tag_id] = (R_w.T @ s[0].R, R_w.T @ (s[0].t - t_w))

    from pose3d.geometry.gravity import estimate_world_up, sign_from_poses
    rig = CalibratedRig(intr[CAM_LEFT], intr[CAM_RIGHT], ext[CAM_LEFT],
                        ext[CAM_RIGHT])
    up, spread, cands = estimate_world_up(rig, layout, poses)
    # did the BODY settle the sense, or only the camera-up proxy? At import
    # time it is always the proxy (poses is None until triangulation has run);
    # finalize_world_up revisits it and rewrites this.
    decided_by_poses = (up is not None and poses is not None
                        and sign_from_poses(up, poses) is not None)

    centres = {cam: ext[cam].camera_center for cam in CAMERAS}
    axes = {cam: ext[cam].R.T @ np.array([0.0, 0.0, 1.0]) for cam in CAMERAS}
    report = {
        "solver": "IPPE_SQUARE single tag, branch scored across frames",
        "dictionary": dictionary,
        "world_tag_id": int(world_tag),
        "world_frame_id": world_frame,
        "marker_length_m": float(marker_length),
        "branch_index": {CAM_LEFT: int(branch[0]), CAM_RIGHT: int(branch[1])},
        "per_camera_ippe_ratio": scored["branch_ratio"],
        "branch_pair_scores": {f"L{a}R{b}": v for (a, b), v in pairs.items()},
        "per_tag_rms_px": {t: v["rms_px"] for t, v in tag_scores.items()},
        "tags_admitted": admitted,
        "tags_rejected_with_reason": rejected,
        "observation_counts": observation_counts(observations),
        # world-tag observations the branch scoring weighed (2 per frame that
        # both cameras saw it in); the final pose comes from one of them
        "n_observations_used": 2 * len(frames),
        "n_frames_with_a_common_tag": len(frames),
        "residual_rms_px": {cam: picked[cam].err for cam in CAMERAS},
        "baseline_m": float(np.linalg.norm(centres[CAM_LEFT] - centres[CAM_RIGHT])),
        "convergence_deg": float(np.degrees(np.arccos(np.clip(
            float(np.dot(axes[CAM_LEFT], axes[CAM_RIGHT])), -1, 1)))),
        "relpose_spread_deg": pairs[branch]["relpose_spread_deg"],
        "camera_motion_check": camera_motion_check(
            observations, list(frame_order), world_tag, intr, marker_length,
            picked),
        "focal_source_per_camera": {cam: intr[cam].source for cam in CAMERAS},
        "image_size": {cam: list(intr[cam].image_size) for cam in CAMERAS},
        "world_up": (None if up is None else up.tolist()),
        "world_up_source": (None if up is None else " + ".join(cands)),
        "world_up_spread_deg": (None if up is None or spread is None
                                else float(spread)),
        "world_up_candidates": {k: v.tolist() for k, v in cands.items()},
        # whichever of the two rules settled the SENSE of that axis; the
        # camera-up proxy is only a stand-in until finalize_world_up can ask
        # the triangulated poses
        "world_up_sign_source": (SIGN_FROM_POSES if decided_by_poses
                                 else SIGN_FROM_CAMERA_UP),
    }
    msg = (f"Calibration estimated from ArUco marker {world_tag} "
           f"(frame {world_frame}).")
    return RigSolution(True, ext, report, msg)


def rebuild_extrinsics_from_report(report: dict, observations, intr):
    """Re-derive the extrinsics from report.json and the tag corners.

    This is what makes the report a provenance record rather than a summary:
    the world tag, world frame, marker length and branch index in it are
    exactly the inputs the solve used, so replaying them reproduces the same
    two poses bit for bit.
    """
    tag_id = int(report["world_tag_id"])
    frame_id = str(report["world_frame_id"])
    length = float(report["marker_length_m"])
    out = {}
    for cam in CAMERAS:
        s = _solves(observations, frame_id, cam, tag_id, intr[cam], length)
        b = int(report["branch_index"][cam])
        if len(s) <= b:
            raise ValueError(f"report names branch {b} for {cam}, but the "
                             f"corners give {len(s)} solution(s)")
        out[cam] = s[b].extrinsics
    return out


def resolve_calibration(
    project: ProjectData,
    load_image,
    marker_length: float = 0.05,
    intr_left: Intrinsics | None = None,
    intr_right: Intrinsics | None = None,
    ext_left: Extrinsics | None = None,
    ext_right: Extrinsics | None = None,
    dictionaries: list[int] | None = None,
) -> CalibrationResult:
    if not project.frames:
        return CalibrationResult(False, None, "failed", "No frames to calibrate.")

    approximate = False

    # --- intrinsics ---
    if intr_left is None or intr_right is None:
        f0 = project.frames[0]
        img_l = load_image(f0.images[CAM_LEFT])
        img_r = load_image(f0.images[CAM_RIGHT])
        if img_l is None or img_r is None:
            return CalibrationResult(False, None, "failed",
                                     "Could not read the first image pair.")
        intr_left = intr_left or _approx_intrinsics(img_l)
        intr_right = intr_right or _approx_intrinsics(img_r)
        approximate = True

    # --- extrinsics from upload ---
    if ext_left is not None and ext_right is not None:
        rig = CalibratedRig(intr_left, intr_right, ext_left, ext_right)
        msg = "Using uploaded calibration."
        if approximate:
            msg += " Intrinsics were approximated from image size."
        return CalibrationResult(True, rig, "uploaded", msg, approximate)

    # --- extrinsics from ArUco ---
    dict_id, observations = detect_all_tags(project, load_image, dictionaries)
    if not observations:
        return CalibrationResult(
            False, None, "failed",
            "Calibration was not successful: no calibration was uploaded and "
            "no ArUco tags were detected in any image.", approximate)

    intr = {CAM_LEFT: intr_left, CAM_RIGHT: intr_right}
    poses = np.stack([f.fitted3d for f in project.frames])
    sol = solve_rig_from_observations(
        observations, [f.frame_id for f in project.frames], intr,
        marker_length, DICT_NAME_BY_ID.get(dict_id),
        poses=(poses if np.isfinite(poses).any() else None))
    if not sol.ok:
        return CalibrationResult(False, None, "failed", sol.message,
                                 approximate, sol.report)

    rig = CalibratedRig(intr_left, intr_right, sol.ext[CAM_LEFT],
                        sol.ext[CAM_RIGHT])
    msg = sol.message
    if approximate:
        msg += (" Intrinsics are approximate — upload a one-time "
                "calibration for metric accuracy.")
    return CalibrationResult(True, rig, "aruco", msg, approximate, sol.report)

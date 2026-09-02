"""Objective, ground-truth-free accuracy metrics for one take.

The client's subject is a rigid-limbed artist mannequin, so every bone length is
CONSTANT by construction and any spread across frames is pipeline error — no
ground truth needed. The body keypoints play no part in the calibration either,
so their epipolar disagreement is an independent test of the rig.

This module is the single place those numbers are defined, so the tests, the
CLI (`tools/measure_take.py`) and the UI cannot drift apart on what "% of body
height" or "reprojection error" means. It is deliberately Qt-free and free of
import side effects: the character rig is loaded only when a `Character` is
actually passed in.

The implementations of `bone_length_stats`, `body_epipolar`, `reprojection` and
`retarget_error` are ported verbatim from the September 2026 audit harness
(`docs/audit-2026-09/prototypes/metrics.py`) and the roll measurement from
`prototypes/roll_prototype.py`, so every number stays directly comparable with
`docs/audit-2026-09/wf_baseline.md`.

One denominator: every "% of height" in the repo is a fraction of
`TakeQuality.subject_height_m`, the median z-extent of the de-tilted delivered
pose (0.1178 m on the client take).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pose3d import pipeline as pl
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS, ProjectData
from pose3d.core.skeleton import BONES, JOINT_NAMES, NUM_JOINTS, Joint
from pose3d.geometry.bonefit import measure_bone_lengths
from pose3d.geometry.character import PoseUnavailable
from pose3d.geometry.orient import de_tilt_matrix, sequence_up
from pose3d.geometry.triangulate import (
    epipolar_distance, fundamental_matrix, reprojection_error,
)

# Human-readable name per canonical bone (audit table T1's row labels).
BONE_NAMES: dict[tuple[int, int], str] = {
    (int(Joint.PELVIS), int(Joint.NECK)): "spine (pelvis-neck)",
    (int(Joint.NECK), int(Joint.HEAD)): "neck-head",
    (int(Joint.NECK), int(Joint.LEFT_SHOULDER)): "neck-Lshoulder",
    (int(Joint.NECK), int(Joint.RIGHT_SHOULDER)): "neck-Rshoulder",
    (int(Joint.LEFT_SHOULDER), int(Joint.LEFT_ELBOW)): "L upper arm",
    (int(Joint.RIGHT_SHOULDER), int(Joint.RIGHT_ELBOW)): "R upper arm",
    (int(Joint.LEFT_ELBOW), int(Joint.LEFT_WRIST)): "L forearm",
    (int(Joint.RIGHT_ELBOW), int(Joint.RIGHT_WRIST)): "R forearm",
    (int(Joint.PELVIS), int(Joint.LEFT_HIP)): "pelvis-Lhip",
    (int(Joint.PELVIS), int(Joint.RIGHT_HIP)): "pelvis-Rhip",
    (int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE)): "L thigh",
    (int(Joint.RIGHT_HIP), int(Joint.RIGHT_KNEE)): "R thigh",
    (int(Joint.LEFT_KNEE), int(Joint.LEFT_ANKLE)): "L shin",
    (int(Joint.RIGHT_KNEE), int(Joint.RIGHT_ANKLE)): "R shin",
}

# (label, left bone, right bone) — the same limb on both sides of one rigid
# mannequin, so the two lengths should be equal.
SYMMETRY_PAIRS: list[tuple[str, tuple[int, int], tuple[int, int]]] = [
    ("thigh", (int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE)),
     (int(Joint.RIGHT_HIP), int(Joint.RIGHT_KNEE))),
    ("shin", (int(Joint.LEFT_KNEE), int(Joint.LEFT_ANKLE)),
     (int(Joint.RIGHT_KNEE), int(Joint.RIGHT_ANKLE))),
    ("upper arm", (int(Joint.LEFT_SHOULDER), int(Joint.LEFT_ELBOW)),
     (int(Joint.RIGHT_SHOULDER), int(Joint.RIGHT_ELBOW))),
    ("forearm", (int(Joint.LEFT_ELBOW), int(Joint.LEFT_WRIST)),
     (int(Joint.RIGHT_ELBOW), int(Joint.RIGHT_WRIST))),
]

# (label, proximal, mid, distal) for the bend-plane comparison.
BEND_CHAINS: list[tuple[str, Joint, Joint, Joint]] = [
    ("L elbow", Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    ("R elbow", Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    ("L knee", Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    ("R knee", Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
]

# Rig role -> the captured (root, mid, end) joints whose cross product is that
# bone's anatomical bend-plane normal. These are the eight limb bones whose
# roll about their own aim axis is undefined in today's retarget (F02).
ROLL_BONES: dict[str, tuple[Joint, Joint, Joint]] = {
    "upper_arm.L": (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    "forearm.L": (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    "upper_arm.R": (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    "forearm.R": (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    "thigh.L": (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    "shin.L": (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    "thigh.R": (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    "shin.R": (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
}

FOOT_ROLES = ("foot.L", "foot.R")


def _u(v):
    return np.asarray(v, float) / (np.linalg.norm(v) + 1e-12)


def _nanstat(a, fn) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else float("nan")


def _pct(v, total) -> float:
    """v as a percentage of `total`, NaN-safe."""
    return (float(100.0 * v / total)
            if np.isfinite(v) and np.isfinite(total) and total > 1e-12
            else float("nan"))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_rig(calib_dir):
    """`pipeline.CalibratedRig` from a calibration folder.

    Deliberately the same body as `pose3d.app._load_rig`, duplicated rather
    than imported: `pose3d.app` pulls in PySide6 at module scope, and this
    module has to stay importable from a headless CLI and from pytest.
    """
    import json
    from pathlib import Path

    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.calib.intrinsics import Intrinsics

    calib_dir = Path(calib_dir)
    il = Intrinsics.load(calib_dir / "left_intrinsics.json")
    ir = Intrinsics.load(calib_dir / "right_intrinsics.json")
    ext = json.loads((calib_dir / "extrinsics.json").read_text())
    el = Extrinsics(R=np.array(ext["left"]["R"], float),
                    t=np.array(ext["left"]["t"], float))
    er = Extrinsics(R=np.array(ext["right"]["R"], float),
                    t=np.array(ext["right"]["t"], float))
    return pl.CalibratedRig(il, ir, el, er)


# ---------------------------------------------------------------------------
# the one denominator
# ---------------------------------------------------------------------------
def de_tilt_rotation(poses: np.ndarray) -> np.ndarray:
    """The 3x3 the 3D view and the Blender export both apply to a take.

    Identity when no frame yields an up-vector, so a degenerate take still
    measures rather than raising.
    """
    up = sequence_up(np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3))
    return np.eye(3) if up is None else de_tilt_matrix(up)


def de_tilted(poses: np.ndarray) -> np.ndarray:
    """`poses` rotated upright exactly as the 3D view and the export do."""
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    return poses @ de_tilt_rotation(poses).T


def subject_height(poses: np.ndarray) -> float:
    """Median z-extent of the de-tilted pose, in metres.

    THE denominator for every "% of body height" in the repo. Not the full 3D
    span (which includes the subject's motion across the take) and not the
    nose-to-ankle distance; a per-frame vertical extent, taken at the median.
    """
    heights = []
    for p in de_tilted(poses):
        v = ~np.isnan(p).any(1)
        if v.sum() >= 2:
            heights.append(float(p[v, 2].max() - p[v, 2].min()))
    return float(np.median(heights)) if heights else float("nan")


def figure_height_px(kp2d: np.ndarray) -> float:
    """Median over frames of the 2D bounding-box height of one camera's finite
    keypoints, in pixels.

    The two cameras are 2:1 apart in resolution, so a pixel error means twice
    as much in the right view as in the left. Every px figure is reported
    against this per-camera denominator as well as raw.
    """
    kp2d = np.asarray(kp2d, float).reshape(-1, NUM_JOINTS, 2)
    heights = []
    for f in kp2d:
        ys = f[np.isfinite(f).all(1), 1]
        if ys.size >= 2:
            heights.append(float(ys.max() - ys.min()))
    return float(np.median(heights)) if heights else float("nan")


# ---------------------------------------------------------------------------
# geometry metrics (ported from the audit harness)
# ---------------------------------------------------------------------------
def bone_length_stats(poses: np.ndarray) -> dict:
    """Per canonical bone: median / std / CV across frames, plus L/R symmetry.

    Measured per FRAME through `bonefit.measure_bone_lengths` — the same
    function the fit uses to pick its targets — so the metric and the fit
    cannot disagree about what a bone is. A rigid mannequin gives std ~ 0.
    """
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    per_frame = [measure_bone_lengths(p[None]) for p in poses]

    bones, med = {}, {}
    for a, b in BONES:
        key = (int(a), int(b))
        # measure_bone_lengths reports an unobservable bone as 0.0
        seg = np.array([f[key] for f in per_frame], float)
        seg = seg[np.isfinite(seg) & (seg > 0.0)]
        m = float(np.median(seg)) if seg.size else float("nan")
        s = float(np.std(seg)) if seg.size else float("nan")
        med[key] = m
        bones[key] = {
            "name": BONE_NAMES[key],
            "n": int(seg.size),
            "median_m": m,
            "std_m": s,
            "cv_pct": _pct(s, m),
            "min_m": float(seg.min()) if seg.size else float("nan"),
            "max_m": float(seg.max()) if seg.size else float("nan"),
        }
    sym = {}
    for label, lkey, rkey in SYMMETRY_PAIRS:
        lm, rm = med.get(lkey, np.nan), med.get(rkey, np.nan)
        ok = np.isfinite(lm) and np.isfinite(rm) and rm > 1e-9
        sym[label] = {
            "left_m": lm, "right_m": rm,
            "ratio_L_over_R": float(lm / rm) if ok else float("nan"),
            "asym_pct": (float(100.0 * abs(lm - rm) / ((lm + rm) / 2))
                         if ok else float("nan")),
        }
    finite = [v["cv_pct"] for v in bones.values() if np.isfinite(v["cv_pct"])]
    return {
        "bones": bones,
        "symmetry": sym,
        "median_cv_pct": float(np.median(finite)) if finite else float("nan"),
        "max_cv_pct": float(np.max(finite)) if finite else float("nan"),
    }


def _point_line_px(pt_left, pt_right, rig) -> tuple[float, float]:
    """Per-image point-to-epipolar-line distances (d_L, d_R) in px.

    Sampson is one number for the pair; these two say how far the observation
    sits from its partner's epipolar line IN EACH IMAGE, which is the only form
    comparable across an asymmetric rig once each is divided by its own
    image diagonal.
    """
    pt_left = np.asarray(pt_left, float).reshape(2)
    pt_right = np.asarray(pt_right, float).reshape(2)
    if np.isnan(pt_left).any() or np.isnan(pt_right).any():
        return float("nan"), float("nan")
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    xl = np.array([pt_left[0], pt_left[1], 1.0])
    xr = np.array([pt_right[0], pt_right[1], 1.0])
    num = abs(float(xr @ F @ xl))
    lr = F @ xl                      # epipolar line of xl, in the RIGHT image
    ll = F.T @ xr                    # epipolar line of xr, in the LEFT image
    nl = math.hypot(ll[0], ll[1])
    nr = math.hypot(lr[0], lr[1])
    return (num / nl if nl > 1e-12 else float("nan"),
            num / nr if nr > 1e-12 else float("nan"))


def body_epipolar(kp2d: dict[str, np.ndarray], rig) -> dict:
    """Per-joint Sampson epipolar error (px) over frames seen in BOTH views.

    The body keypoints played no part in the calibration, so this is an
    independent test of the extrinsics/intrinsics — but only of their mutual
    consistency: a focal error shared by both cameras largely cancels in F.
    """
    L = np.asarray(kp2d[CAM_LEFT], float)
    R = np.asarray(kp2d[CAM_RIGHT], float)
    per, allv = {}, []
    d_left, d_right = [], []
    for j in range(NUM_JOINTS):
        vals = []
        for t in range(L.shape[0]):
            e = epipolar_distance(L[t, j], R[t, j],
                                  rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                                  rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
            if np.isfinite(e):
                vals.append(e)
                dl, dr = _point_line_px(L[t, j], R[t, j], rig)
                d_left.append(dl)
                d_right.append(dr)
        allv.extend(vals)
        per[JOINT_NAMES[j]] = {
            "n": len(vals),
            "median_px": _nanstat(vals, np.median),
            "p90_px": (float(np.percentile(vals, 90)) if vals else float("nan")),
            "max_px": _nanstat(vals, np.max),
        }
    diag = {c: float(math.hypot(*rig.intr[c].image_size)) for c in CAMERAS}
    per_image = {}
    for cam, vals in ((CAM_LEFT, d_left), (CAM_RIGHT, d_right)):
        per_image[cam] = {
            "median_px": _nanstat(vals, np.median),
            "p90_px": (float(np.percentile(vals, 90)) if vals else float("nan")),
            "max_px": _nanstat(vals, np.max),
            "image_diag_px": diag[cam],
            "median_pct_diag": _pct(_nanstat(vals, np.median), diag[cam]),
            "p90_pct_diag": _pct(
                float(np.percentile(vals, 90)) if vals else float("nan"),
                diag[cam]),
        }
    thr = float(pl.epipolar_threshold(rig))
    return {
        "per_joint": per,
        "per_image": per_image,
        "median_px": _nanstat(allv, np.median),
        "p90_px": (float(np.percentile(allv, 90)) if allv else float("nan")),
        "p99_px": (float(np.percentile(allv, 99)) if allv else float("nan")),
        "max_px": _nanstat(allv, np.max),
        "n_pairs": len(allv),
        "image_diag_px": diag[CAM_LEFT],
        "median_pct_diag": _pct(_nanstat(allv, np.median), diag[CAM_LEFT]),
        "threshold_px": thr,
        "frac_over_threshold": (float(np.mean(np.asarray(allv) > thr))
                                if allv else float("nan")),
    }


def reprojection(poses: np.ndarray, kp2d: dict[str, np.ndarray], rig,
                 figure_h_px: dict[str, float] | None = None) -> dict:
    """Per-joint, per-camera median reprojection error (px and % of figure).

    Weak by construction for a freshly triangulated point, which always
    reprojects near its own observations. It bites for poses CHANGED after
    triangulation — the bone fit and the smoother — which is exactly what
    "delivered vs measured" compares.
    """
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    out = {}
    for cam in CAMERAS:
        fh = (figure_h_px or {}).get(cam, float("nan"))
        obs = np.asarray(kp2d[cam], float)
        errs = np.full((poses.shape[0], NUM_JOINTS), np.nan)
        for t in range(poses.shape[0]):
            errs[t] = reprojection_error(poses[t], obs[t],
                                         rig.intr[cam], rig.ext[cam])
        per = {}
        for j in range(NUM_JOINTS):
            col = errs[:, j]
            col = col[np.isfinite(col)]
            m = float(np.median(col)) if col.size else float("nan")
            per[JOINT_NAMES[j]] = {
                "n": int(col.size),
                "median_px": m,
                "max_px": float(col.max()) if col.size else float("nan"),
                "median_pct_figure": _pct(m, fh),
            }
        flat = errs[np.isfinite(errs)]
        med = float(np.median(flat)) if flat.size else float("nan")
        out[cam] = {
            "per_joint": per,
            "median_px": med,
            "p90_px": (float(np.percentile(flat, 90)) if flat.size
                       else float("nan")),
            "max_px": float(flat.max()) if flat.size else float("nan"),
            "figure_h_px": fh,
            "median_pct_figure": _pct(med, fh),
        }
    return out


def gap_stats(project: ProjectData) -> dict:
    """Where the take has no 3D, and how it got that way.

    `rejected` counts 2D observations that are absent from the stored project
    (the cross-view gate NaNs them in place, so a dropped observation and one
    the detector never found look the same afterwards). `filled` counts joints
    a gap fill invented; it is 0 until Phase 1 adds `Frame.filled`.
    """
    missing, rejected, filled = [], 0, 0
    for f in project.frames:
        p = np.asarray(f.pose3d, float).reshape(NUM_JOINTS, 3)
        for j in range(NUM_JOINTS):
            if np.isnan(p[j]).all():
                missing.append((f.frame_id, JOINT_NAMES[j]))
        for c in CAMERAS:
            rejected += int(np.isnan(np.asarray(f.kp2d[c], float)).any(1).sum())
        flags = getattr(f, "filled", None)
        if flags is not None:
            filled += int(np.count_nonzero(flags))
    total = max(1, len(project.frames) * NUM_JOINTS)
    return {
        "missing": missing,
        "n_missing": len(missing),
        "missing_pct": 100.0 * len(missing) / total,
        "rejected": rejected,
        "filled": filled,
    }


# ---------------------------------------------------------------------------
# character metrics (need the bundled rig)
# ---------------------------------------------------------------------------
def _angle_deg(u, v) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if not (nu > 1e-12 and nv > 1e-12):
        return float("nan")
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def retarget_error(character, up: np.ndarray, height: float, scale: float,
                   head3d_up: np.ndarray | None = None) -> dict:
    """How far the posed character's joints land from the captured joints.

    `character` must already be `fit_to_subject`-ed on `up`; `up` is the
    de-tilted pose sequence, which is what `main_window` hands the 3D view.
    Distances are reported against the ONE body-height denominator.

    This measures how faithfully the skinning reproduces its input, not
    absolute accuracy: it cannot see error already present in `up`.
    """
    up = np.asarray(up, float).reshape(-1, NUM_JOINTS, 3)
    dists = np.full((up.shape[0], NUM_JOINTS), np.nan)
    bend = {label: [] for label, *_ in BEND_CHAINS}
    n_posed = 0
    for t, p in enumerate(up):
        valid = ~np.isnan(p).any(1)
        if not valid.any():
            continue
        hp = None
        if head3d_up is not None:
            h = np.asarray(head3d_up[t], float)
            hp = None if np.isnan(h).all() else h
        try:
            J = character.posed_joints(p, valid, hp)
        except PoseUnavailable:
            # no hips in this frame: the rig has no root, so there is no
            # retarget error to measure here. Not an error, just no sample.
            continue
        if J is None:
            continue
        n_posed += 1
        J = np.asarray(J, float)
        d = np.linalg.norm(J - p, axis=1)
        d[~valid] = np.nan
        dists[t] = d
        for label, a, m, b in BEND_CHAINS:
            ia, im, ib = int(a), int(m), int(b)
            if not (valid[ia] and valid[im] and valid[ib]):
                continue
            if np.isnan(J[[ia, im, ib]]).any():
                continue
            ang = _angle_deg(np.cross(p[im] - p[ia], p[ib] - p[im]),
                             np.cross(J[im] - J[ia], J[ib] - J[im]))
            if np.isfinite(ang):
                bend[label].append(ang)

    per = {}
    for j in range(NUM_JOINTS):
        col = dists[:, j]
        col = col[np.isfinite(col)]
        per[JOINT_NAMES[j]] = {
            "n": int(col.size),
            "median_m": float(np.median(col)) if col.size else float("nan"),
            "median_pct_height": _pct(
                float(np.median(col)) if col.size else float("nan"), height),
            "max_pct_height": _pct(
                float(col.max()) if col.size else float("nan"), height),
        }
    flat = dists[np.isfinite(dists)]
    bend_stats = {
        label: {"n": len(vals),
                "median_deg": float(np.median(vals)) if vals else float("nan"),
                "max_deg": float(np.max(vals)) if vals else float("nan")}
        for label, vals in bend.items()}
    allbend = [v for vals in bend.values() for v in vals]
    return {
        "n_frames_posed": n_posed,
        "character_scale": scale,
        "per_joint": per,
        "median_m": float(np.median(flat)) if flat.size else float("nan"),
        "median_pct_height": _pct(
            float(np.median(flat)) if flat.size else float("nan"), height),
        "p90_pct_height": _pct(
            float(np.percentile(flat, 90)) if flat.size else float("nan"),
            height),
        "bend_plane": bend_stats,
        "bend_plane_median_deg": (float(np.median(allbend)) if allbend
                                  else float("nan")),
        "bend_plane_max_deg": (float(np.max(allbend)) if allbend
                               else float("nan")),
    }


def _rest_bend_references(character) -> dict[int, np.ndarray]:
    """Each limb bone's bend-plane normal in the REST pose, per bone index.

    This is the reference a bone "carries": rotate it by the bone's posed
    rotation and it says where the retarget put the limb's bend plane. Today
    nothing constrains it, which is precisely what `roll_error` measures.
    """
    rj = character.rest_joints()
    refs = {}
    for role, (ja, jm, jb) in ROLL_BONES.items():
        b = character.role.get(role)
        if b is None:
            continue
        n = np.cross(rj[int(jm)] - rj[int(ja)], rj[int(jb)] - rj[int(jm)])
        if np.linalg.norm(n) < 1e-9 or np.isnan(n).any():
            # straight at rest: fall back to the rig's own rest bend pole
            ub = character.role.get(role.replace("forearm", "upper_arm")
                                    .replace("shin", "thigh"))
            pole = character._rest_pole.get(ub, np.array([0, 1.0, 0]))
            n = np.cross(rj[int(jb)] - rj[int(ja)], pole)
        refs[b] = _u(n)
    return refs


def limb_metrics(character, up: np.ndarray, height: float) -> dict:
    """Roll error per limb bone, sole tilt per foot, and the ground datum.

    Roll error is the angle ABOUT each bone's aim axis between the rest bend
    reference the bone carries and the captured bend-plane normal (F02). Every
    positional metric in the repo is structurally blind to it: a bone can aim
    at exactly the right joint while twisted 50 deg about that aim.

    Sole tilt is the posed foot bone's axis off horizontal; the ground datum is
    the captured ankle's height above the character's lowest vertex, as % of
    body height — what makes the figure bob against a fixed grid.
    """
    up = np.asarray(up, float).reshape(-1, NUM_JOINTS, 3)
    refs = _rest_bend_references(character)
    roll = {role: [] for role in ROLL_BONES if role in character.role}
    sole = {role: [] for role in FOOT_ROLES if role in character.role}
    ground = []

    for p in up:
        valid = ~np.isnan(p).any(1)
        if not valid.any():
            continue
        skin, _, _, Rz = character._skin_matrices(p, valid, None)
        if skin is None:
            continue

        for role in roll:
            b = character.role[role]
            ja, jm, jb = ROLL_BONES[role]
            if not all(valid[int(x)] for x in (ja, jm, jb)):
                continue
            axis = _u(skin[b][:3, :3] @ (character.tail[b] - character.head[b]))
            n = np.cross(p[int(jm)] - p[int(ja)], p[int(jb)] - p[int(jm)])
            if np.linalg.norm(n) < 1e-9:
                continue
            tgt = _u(Rz @ n)
            tgt = _u(tgt - np.dot(tgt, axis) * axis)
            cur = skin[b][:3, :3] @ refs[b]
            cur = _u(cur - np.dot(cur, axis) * axis)
            roll[role].append(abs(np.degrees(np.arctan2(
                float(np.dot(np.cross(cur, tgt), axis)),
                float(np.dot(cur, tgt))))))

        for role in sole:
            b = character.role[role]
            rest = _u(character.tail[b] - character.head[b])
            axis = _u(skin[b][:3, :3] @ rest)
            sole[role].append(float(np.degrees(np.arcsin(
                min(1.0, abs(float(axis[2])))))))

        ankles = [p[int(j), 2] for j in (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)
                  if valid[int(j)]]
        if ankles:
            verts, _, _ = character.pose_and_joints(p, valid, None)
            if verts is not None:
                ground.append(_pct(min(ankles) - float(verts[:, 2].min()),
                                   height))

    def stats(vals):
        return {"n": len(vals),
                "median_deg": float(np.median(vals)) if vals else float("nan"),
                "max_deg": float(np.max(vals)) if vals else float("nan")}

    roll_stats = {role: stats(v) for role, v in roll.items()}
    finite_roll = [s["median_deg"] for s in roll_stats.values()
                   if np.isfinite(s["median_deg"])]
    return {
        "roll_error_deg": roll_stats,
        "roll_error_median_deg": (float(np.median(finite_roll)) if finite_roll
                                  else float("nan")),
        "roll_error_max_deg": _nanstat(
            [s["max_deg"] for s in roll_stats.values()], np.max),
        "sole_tilt_deg": {role: stats(v) for role, v in sole.items()},
        "ground_datum_pct": {
            "n": len(ground),
            "median": float(np.median(ground)) if ground else float("nan"),
            "peak_to_peak": (float(np.max(ground) - np.min(ground)) if ground
                             else float("nan")),
        },
    }


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------
@dataclass
class TakeQuality:
    """Everything measurable about one take without ground truth.

    `measured` always means the raw triangulation (`Frame.pose3d`) and
    `delivered` the pose the app actually shows and exports (`Frame.fitted3d`).
    """
    n_frames: int
    frame_ids: list[str]
    subject_height_m: float
    figure_h_px: dict[str, float]
    reproj: dict[str, dict[str, dict]]
    epipolar: dict
    bone_cv: dict
    symmetry: dict
    gaps: dict
    # character metrics; None when no Character was supplied
    retarget_pct_height: dict | None = None
    roll_error_deg: dict | None = None
    sole_tilt_deg: dict | None = None
    ground_datum_pct: dict | None = None

    def reproj_ratio(self, cam: str) -> float:
        """delivered / measured median reprojection error in one camera.

        1.0 means the pose that ships is as close to the keypoints as the
        measurement is. Today's shipped pipeline is 4.2x (left) / 6.2x (right)
        because of the causal EMA smoother.
        """
        m = self.reproj[cam]["measured"]["median_px"]
        d = self.reproj[cam]["delivered"]["median_px"]
        return float(d / m) if np.isfinite(m) and m > 1e-9 else float("nan")

    def to_dict(self) -> dict:
        """JSON-ready: NaN -> None, tuple keys -> "a-b" strings."""
        return _jsonable({f: getattr(self, f) for f in self.__dataclass_fields__})


def _jsonable(o):
    if isinstance(o, dict):
        return {(f"{k[0]}-{k[1]}" if isinstance(k, tuple) else str(k)):
                _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(float(o)) else float(o)
    return o


def _stack(project: ProjectData, attr: str) -> np.ndarray:
    if not project.frames:
        return np.zeros((0, NUM_JOINTS, 3))
    return np.stack([getattr(f, attr) for f in project.frames])


def take_quality(project: ProjectData, rig, character=None) -> TakeQuality:
    """Measure one take. `rig` is a `pipeline.CalibratedRig`.

    Pass a `Character` (already constructed, not yet fitted — this fits it to
    the take) to add the retarget, roll, sole-tilt and ground-datum metrics.
    """
    measured = _stack(project, "pose3d")
    delivered = _stack(project, "fitted3d")
    if not np.isfinite(delivered).any():
        # a project that was never bone-fitted still has to be measurable
        delivered = measured
    kp2d = {c: np.stack([f.kp2d[c] for f in project.frames])
            if project.frames else np.zeros((0, NUM_JOINTS, 2))
            for c in CAMERAS}

    height = subject_height(delivered)
    figure_h = {c: figure_height_px(kp2d[c]) for c in CAMERAS}
    rep_m = reprojection(measured, kp2d, rig, figure_h)
    rep_d = reprojection(delivered, kp2d, rig, figure_h)
    blen = bone_length_stats(measured)          # RAW: the fit flattens this

    q = TakeQuality(
        n_frames=len(project.frames),
        frame_ids=[f.frame_id for f in project.frames],
        subject_height_m=height,
        figure_h_px=figure_h,
        reproj={c: {"measured": rep_m[c], "delivered": rep_d[c]}
                for c in CAMERAS},
        epipolar=body_epipolar(kp2d, rig),
        bone_cv=blen,
        symmetry=blen["symmetry"],
        gaps=gap_stats(project),
    )
    if character is None:
        return q

    R = de_tilt_rotation(delivered)
    up = delivered @ R.T
    scale = character.fit_to_subject(up)
    head3d = _stack(project, "head3d") if project.frames else None
    head3d = head3d @ R.T if head3d is not None and np.isfinite(head3d).any() \
        else None
    q.retarget_pct_height = retarget_error(character, up, height, scale, head3d)
    limbs = limb_metrics(character, up, height)
    q.roll_error_deg = limbs["roll_error_deg"] | {
        "ALL": {"median_deg": limbs["roll_error_median_deg"],
                "max_deg": limbs["roll_error_max_deg"]}}
    q.sole_tilt_deg = limbs["sole_tilt_deg"]
    q.ground_datum_pct = limbs["ground_datum_pct"]
    return q


# ---------------------------------------------------------------------------
# sidebar wording
# ---------------------------------------------------------------------------
# A left/right limb-length difference this size is worth naming. NOT 3 %:
# a Monte-Carlo over this take's own keypoint noise puts the p99 of a
# perfectly symmetric subject at 4.24 %, so 3 % would flag a rigid mannequin
# most of the time. On the Panoptic ground-truth reference RTMPose alone
# produces 6.5 % thigh asymmetry where the ground truth has 1.4 %, which is
# why the wording below blames a KEYPOINT and never the calibration.
SYMMETRY_FLAG_PCT = 5.0


def symmetry_notes(q: "TakeQuality",
                   threshold_pct: float = SYMMETRY_FLAG_PCT) -> list[str]:
    """One sentence per limb whose two sides measure differently (F34).

    The same moulded limb on both sides of one rigid mannequin must measure
    the same, so a difference is reconstruction error. It is NOT evidence
    about the calibration: a shared rig error moves both sides together, and
    the detector's own left/right confusion produces this exact signature on
    ground-truth data. So the sentence names the endpoint the two views place
    least consistently, gives both endpoints' epipolar residuals so the claim
    can be checked, and asks the user to look at that keypoint in both images.
    """
    notes: list[str] = []
    per_joint = (q.epipolar or {}).get("per_joint", {})

    def residual(joint: int) -> float:
        v = (per_joint.get(JOINT_NAMES[int(joint)]) or {}).get("median_px")
        return float("nan") if v is None else float(v)

    for label, lkey, rkey in SYMMETRY_PAIRS:
        asym = (q.symmetry.get(label) or {}).get("asym_pct")
        asym = float("nan") if asym is None else float(asym)
        if not np.isfinite(asym) or asym < threshold_pct:
            continue
        # the DISTAL end of each bone is the keypoint that actually moved
        ends = {"L": int(lkey[1]), "R": int(rkey[1])}
        res = {k: residual(j) for k, j in ends.items()}
        if all(np.isfinite(v) for v in res.values()):
            worse, better = ("L", "R") if res["L"] >= res["R"] else ("R", "L")
            detail = (f"the two views place {JOINT_NAMES[ends[worse]]} less "
                      f"consistently than {JOINT_NAMES[ends[better]]} "
                      f"({res[worse]:.1f} px vs {res[better]:.1f} px off the "
                      f"epipolar line)")
        else:
            detail = (f"check {JOINT_NAMES[ends['L']]} and "
                      f"{JOINT_NAMES[ends['R']]}")
        notes.append(
            f"L/R {label} lengths differ by {asym:.1f} % on a subject whose "
            f"two sides are identical — {detail}. Check that keypoint in "
            f"both images.")
    return notes

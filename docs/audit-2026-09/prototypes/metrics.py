"""Objective, ground-truth-free accuracy metrics for a pose3d take.

Reusable module + CLI. Import has no side effects (no Qt, no GUI, no writes).

Why these metrics: the client's subject is a rigid-limbed artist mannequin held
still by a clamp, so every bone length is CONSTANT by construction. Any spread
in a reconstructed bone length across frames is pipeline error, with no ground
truth needed. The body keypoints are also not used by the calibration, so their
epipolar disagreement is an independent test of the rig.

Usage (from the repo root):
    cd /media/athena/hd3/Projects/pose3d-tool && .venv/bin/python \
        <this file> <project_dir> [--calib DIR] \
        [--source raw|fitted|refit|refit-nosmooth] [--json out.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import os
import sys

import numpy as np

# This module lives outside the repo, so Python's implicit sys.path[0] (the
# script's own folder) does not find the package. Add the repo root: the cwd
# when the CLI is run from it (the documented way), else $POSE3D_REPO, else
# the first ancestor of the cwd that holds pose3d/__init__.py.
def _add_repo_to_path():
    cands = [os.environ.get("POSE3D_REPO"), os.getcwd()]
    cands += [str(p) for p in Path(os.getcwd()).resolve().parents]
    for c in cands:
        if c and (Path(c) / "pose3d" / "__init__.py").exists():
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None


_add_repo_to_path()

from pose3d.core.io_project import load_project  # noqa: E402
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS
from pose3d.core.skeleton import BONES, JOINT_NAMES, NUM_JOINTS, Joint
from pose3d.geometry.triangulate import (
    epipolar_distance, reprojection_error, triangulate_points,
)
from pose3d import pipeline as pl

SOURCES = ("raw", "fitted", "refit", "refit-nosmooth")

BONE_NAMES = {
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

# (label, left bone, right bone) for the limb-symmetry check
SYMMETRY_PAIRS = [
    ("thigh", (int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE)),
     (int(Joint.RIGHT_HIP), int(Joint.RIGHT_KNEE))),
    ("shin", (int(Joint.LEFT_KNEE), int(Joint.LEFT_ANKLE)),
     (int(Joint.RIGHT_KNEE), int(Joint.RIGHT_ANKLE))),
    ("upper arm", (int(Joint.LEFT_SHOULDER), int(Joint.LEFT_ELBOW)),
     (int(Joint.RIGHT_SHOULDER), int(Joint.RIGHT_ELBOW))),
    ("forearm", (int(Joint.LEFT_ELBOW), int(Joint.LEFT_WRIST)),
     (int(Joint.RIGHT_ELBOW), int(Joint.RIGHT_WRIST))),
]

# (label, proximal, mid, distal) for the bend-plane comparison
BEND_CHAINS = [
    ("L elbow", Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    ("R elbow", Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    ("L knee", Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    ("R knee", Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_rig(calib_dir):
    """CalibratedRig from a calibration folder.

    Same body as pose3d.app._load_rig (pose3d/app.py:81) but without importing
    pose3d.app, which pulls in PySide6 at module scope (pose3d/app.py:16).
    """
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


def load_take(project_dir, calib_dir=None) -> dict:
    """Everything a metric needs, as plain arrays plus the live ProjectData."""
    project_dir = Path(project_dir)
    project = load_project(project_dir)
    calib_dir = Path(calib_dir) if calib_dir else project_dir / "calibration"
    rig = load_rig(calib_dir)

    T = len(project.frames)
    kp2d = {c: np.stack([f.kp2d[c] for f in project.frames]) for c in CAMERAS}
    scores = {c: np.stack([f.scores[c] for f in project.frames]) for c in CAMERAS}
    head2d = {c: np.stack([f.head2d[c] for f in project.frames]) for c in CAMERAS}
    return {
        "project_dir": str(project_dir),
        "calib_dir": str(calib_dir),
        "project": project,
        "n_frames": T,
        "frame_ids": [f.frame_id for f in project.frames],
        "images": {c: [f.images.get(c, "") for f in project.frames]
                   for c in CAMERAS},
        "kp2d": kp2d,
        "scores": scores,
        "head2d": head2d,
        "head3d": np.stack([f.head3d for f in project.frames]) if T else None,
        "rig": rig,
        "pose3d": np.stack([f.pose3d for f in project.frames]) if T
        else np.zeros((0, NUM_JOINTS, 3)),
        "fitted3d": np.stack([f.fitted3d for f in project.frames]) if T
        else np.zeros((0, NUM_JOINTS, 3)),
        "corrections": len(project.corrections),
    }


# --------------------------------------------------------------------------
# triangulation
# --------------------------------------------------------------------------
def triangulate(kp2d, rig) -> np.ndarray:
    """(T,15,3) plain DLT per frame, no cross-view validation."""
    L, R = np.asarray(kp2d[CAM_LEFT], float), np.asarray(kp2d[CAM_RIGHT], float)
    out = np.full((L.shape[0], NUM_JOINTS, 3), np.nan)
    for t in range(L.shape[0]):
        out[t] = triangulate_points(
            L[t], R[t], rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
            rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    return out


def triangulate_validated(project, rig):
    """(poses, dropped, kp2d_after) with pipeline.validate_cross_view applied
    to a DEEP COPY of the project first (the shipped default path)."""
    p = copy.deepcopy(project)
    dropped = pl.validate_cross_view(p, rig)
    kp = {c: np.stack([f.kp2d[c] for f in p.frames]) for c in CAMERAS}
    return triangulate(kp, rig), dropped, kp


def refit(project, smooth=True):
    """(T,15,3) = pipeline.fit_project on a deep copy of the stored pose3d."""
    p = copy.deepcopy(project)
    pl.fit_project(p, smooth=smooth)
    return np.stack([f.fitted3d for f in p.frames]) if p.frames \
        else np.zeros((0, NUM_JOINTS, 3))


def poses_for_source(take, source: str) -> np.ndarray:
    if source == "raw":
        return take["pose3d"]
    if source == "fitted":
        return take["fitted3d"]
    if source == "refit":
        return refit(take["project"], smooth=True)
    if source == "refit-nosmooth":
        return refit(take["project"], smooth=False)
    raise ValueError(f"unknown source {source!r}; want one of {SOURCES}")


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _nanstat(a, fn):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else float("nan")


def bone_length_stats(poses) -> dict:
    """Per canonical bone: median / std / std-as-%-of-median across frames,
    plus left/right symmetry ratios. A rigid mannequin gives std ~ 0."""
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    bones, med = {}, {}
    for a, b in BONES:
        key = (int(a), int(b))
        seg = np.linalg.norm(poses[:, int(b)] - poses[:, int(a)], axis=1)
        seg = seg[np.isfinite(seg)]
        m = float(np.median(seg)) if seg.size else float("nan")
        s = float(np.std(seg)) if seg.size else float("nan")
        med[key] = m
        bones[key] = {
            "name": BONE_NAMES[key],
            "n": int(seg.size),
            "median_m": m,
            "std_m": s,
            "cv_pct": float(100.0 * s / m) if seg.size and m > 1e-9
            else float("nan"),
            "min_m": float(seg.min()) if seg.size else float("nan"),
            "max_m": float(seg.max()) if seg.size else float("nan"),
        }
    sym = {}
    for label, lkey, rkey in SYMMETRY_PAIRS:
        lm, rm = med.get(lkey, np.nan), med.get(rkey, np.nan)
        ratio = float(lm / rm) if np.isfinite(lm) and np.isfinite(rm) \
            and rm > 1e-9 else float("nan")
        sym[label] = {
            "left_m": lm, "right_m": rm, "ratio_L_over_R": ratio,
            "asym_pct": float(100.0 * abs(lm - rm) / ((lm + rm) / 2))
            if np.isfinite(ratio) else float("nan"),
        }
    finite_cv = [v["cv_pct"] for v in bones.values() if np.isfinite(v["cv_pct"])]
    return {
        "bones": bones,
        "symmetry": sym,
        "median_cv_pct": float(np.median(finite_cv)) if finite_cv
        else float("nan"),
        "max_cv_pct": float(np.max(finite_cv)) if finite_cv else float("nan"),
    }


def body_epipolar(kp2d, rig) -> dict:
    """Per-joint median Sampson epipolar error (px) over frames seen in BOTH
    views. The body keypoints played no part in the calibration, so this is an
    independent test of the extrinsics/intrinsics."""
    L, R = np.asarray(kp2d[CAM_LEFT], float), np.asarray(kp2d[CAM_RIGHT], float)
    per = {}
    allv = []
    for j in range(NUM_JOINTS):
        vals = []
        for t in range(L.shape[0]):
            e = epipolar_distance(L[t, j], R[t, j],
                                  rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                                  rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
            if np.isfinite(e):
                vals.append(e)
        allv.extend(vals)
        per[JOINT_NAMES[j]] = {
            "n": len(vals),
            "median_px": _nanstat(vals, np.median),
            "p90_px": (float(np.percentile(vals, 90)) if vals else float("nan")),
            "max_px": _nanstat(vals, np.max),
        }
    w, h = rig.intr[CAM_LEFT].image_size
    diag = float(math.hypot(w, h))
    return {
        "per_joint": per,
        "overall_median_px": _nanstat(allv, np.median),
        "overall_p90_px": (float(np.percentile(allv, 90)) if allv
                           else float("nan")),
        "n_pairs": len(allv),
        "image_diag_px": diag,
        "overall_median_pct_diag": (100.0 * _nanstat(allv, np.median) / diag),
        "epipolar_threshold_px": float(pl.epipolar_threshold(rig)),
        "frac_over_threshold": (
            float(np.mean(np.asarray(allv) > pl.epipolar_threshold(rig)))
            if allv else float("nan")),
    }


def reprojection(poses, kp2d, rig) -> dict:
    """Per-joint, per-camera median reprojection error (px).

    Weak by construction: a two-view DLT point always reprojects near its own
    observations. It only bites for poses that were CHANGED after
    triangulation (the bone fit, the smoother)."""
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    out = {}
    for cam in CAMERAS:
        obs = np.asarray(kp2d[cam], float)
        errs = np.full((poses.shape[0], NUM_JOINTS), np.nan)
        for t in range(poses.shape[0]):
            errs[t] = reprojection_error(poses[t], obs[t],
                                         rig.intr[cam], rig.ext[cam])
        per = {}
        for j in range(NUM_JOINTS):
            col = errs[:, j]
            col = col[np.isfinite(col)]
            per[JOINT_NAMES[j]] = {
                "n": int(col.size),
                "median_px": float(np.median(col)) if col.size else float("nan"),
                "max_px": float(col.max()) if col.size else float("nan"),
            }
        flat = errs[np.isfinite(errs)]
        out[cam] = {
            "per_joint": per,
            "overall_median_px": float(np.median(flat)) if flat.size
            else float("nan"),
            "overall_p90_px": float(np.percentile(flat, 90)) if flat.size
            else float("nan"),
        }
    return out


def _angle_deg(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if not (nu > 1e-12 and nv > 1e-12):
        return float("nan")
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def retarget_error(poses, head3d=None) -> dict:
    """How far the bundled character's joints land from the captured joints.

    Poses the character EXACTLY as the app does (pose3d/ui/main_window.py:495
    and pose3d/geometry/character.py:328/702):
        R  = de_tilt_matrix(sequence_up(poses))
        up = poses @ R.T
        Character().fit_to_subject(up)  # once, whole take
        Character().posed_joints(up[t], valid[t])  # per frame
    Distances are reported as a % of body height (z-extent of the captured
    up-pose), which is the scale the client actually perceives.
    """
    from pose3d.geometry.character import Character
    from pose3d.geometry.orient import de_tilt_matrix, sequence_up

    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    up_vec = sequence_up(poses)
    if up_vec is None:
        return {"error": "sequence_up returned None (no usable frame)"}
    R = de_tilt_matrix(up_vec)
    up = poses @ R.T

    ch = Character()
    scale = ch.fit_to_subject(up)

    heights = []
    for p in up:
        v = ~np.isnan(p).any(1)
        if v.sum() >= 2:
            heights.append(float(p[v, 2].max() - p[v, 2].min()))
    height = float(np.median(heights)) if heights else float("nan")

    dists = np.full((up.shape[0], NUM_JOINTS), np.nan)
    bend = {label: [] for label, *_ in BEND_CHAINS}
    n_posed = 0
    for t, p in enumerate(up):
        valid = ~np.isnan(p).any(1)
        if not valid.any():
            continue
        hp = None
        if head3d is not None:
            h = np.asarray(head3d[t], float)
            hp = None if np.isnan(h).all() else h @ R.T
        try:
            J = ch.posed_joints(p, valid, hp)
        except Exception:
            J = None
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
            nc = np.cross(p[im] - p[ia], p[ib] - p[im])
            nr = np.cross(J[im] - J[ia], J[ib] - J[im])
            ang = _angle_deg(nc, nr)
            if np.isfinite(ang):
                bend[label].append(ang)

    per = {}
    for j in range(NUM_JOINTS):
        col = dists[:, j]
        col = col[np.isfinite(col)]
        per[JOINT_NAMES[j]] = {
            "n": int(col.size),
            "median_m": float(np.median(col)) if col.size else float("nan"),
            "median_pct_height": float(100.0 * np.median(col) / height)
            if col.size and np.isfinite(height) and height > 1e-9
            else float("nan"),
            "max_pct_height": float(100.0 * col.max() / height)
            if col.size and np.isfinite(height) and height > 1e-9
            else float("nan"),
        }
    flat = dists[np.isfinite(dists)]
    bend_stats = {}
    for label, vals in bend.items():
        bend_stats[label] = {
            "n": len(vals),
            "median_deg": float(np.median(vals)) if vals else float("nan"),
            "max_deg": float(np.max(vals)) if vals else float("nan"),
        }
    allbend = [v for vals in bend.values() for v in vals]
    return {
        "height_m": height,
        "character_scale": scale,
        "n_frames_posed": n_posed,
        "per_joint": per,
        "overall_median_m": float(np.median(flat)) if flat.size
        else float("nan"),
        "overall_median_pct_height": float(100.0 * np.median(flat) / height)
        if flat.size and np.isfinite(height) and height > 1e-9
        else float("nan"),
        "overall_p90_pct_height": float(100.0 * np.percentile(flat, 90) / height)
        if flat.size and np.isfinite(height) and height > 1e-9
        else float("nan"),
        "bend_plane": bend_stats,
        "bend_plane_overall_median_deg": float(np.median(allbend)) if allbend
        else float("nan"),
        "bend_plane_overall_max_deg": float(np.max(allbend)) if allbend
        else float("nan"),
    }


def pose_delta(a, b) -> dict:
    """Per-joint mean |a - b| in mm over frames where both are finite."""
    a = np.asarray(a, float).reshape(-1, NUM_JOINTS, 3)
    b = np.asarray(b, float).reshape(-1, NUM_JOINTS, 3)
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=2) * 1000.0
    per = {}
    for j in range(NUM_JOINTS):
        col = d[:, j]
        col = col[np.isfinite(col)]
        per[JOINT_NAMES[j]] = {
            "n": int(col.size),
            "mean_mm": float(col.mean()) if col.size else float("nan"),
            "max_mm": float(col.max()) if col.size else float("nan"),
        }
    flat = d[np.isfinite(d)]
    return {"per_joint": per,
            "overall_mean_mm": float(flat.mean()) if flat.size else float("nan"),
            "overall_max_mm": float(flat.max()) if flat.size else float("nan")}


def missing_stats(kp2d, poses) -> dict:
    """How much of the take actually exists."""
    out = {}
    for c in CAMERAS:
        a = np.asarray(kp2d[c], float)
        out[c + "_kp_present_pct"] = float(
            100.0 * np.mean(~np.isnan(a).any(2)))
    p = np.asarray(poses, float)
    out["joints3d_present_pct"] = float(100.0 * np.mean(~np.isnan(p).any(2)))
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def summarise(take, source, poses, blen, epi, rep, ret, extra=None) -> str:
    """Markdown report for one (take, source)."""
    L = []
    A = L.append
    A(f"### {Path(take['project_dir']).name} — source `{source}` "
      f"({take['n_frames']} frames)")
    A("")
    A("**Bone lengths across frames** (rigid mannequin: a perfect pipeline "
      "gives std = 0)")
    A("")
    A("| bone | n | median (m) | std (m) | std/median % | min (m) | max (m) |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for (a, b) in [(int(x), int(y)) for x, y in BONES]:
        v = blen["bones"][(a, b)]
        A(f"| {v['name']} | {v['n']} | {_f(v['median_m'],4)} | "
          f"{_f(v['std_m'],4)} | {_f(v['cv_pct'],1)} | {_f(v['min_m'],4)} | "
          f"{_f(v['max_m'],4)} |")
    A(f"| **median / max over bones** |  |  |  | "
      f"**{_f(blen['median_cv_pct'],1)} / {_f(blen['max_cv_pct'],1)}** |  |  |")
    A("")
    A("**Left/right symmetry** (same mannequin limb both sides)")
    A("")
    A("| limb | left (m) | right (m) | L/R | asym % |")
    A("|---|---:|---:|---:|---:|")
    for label, *_ in SYMMETRY_PAIRS:
        s = blen["symmetry"][label]
        A(f"| {label} | {_f(s['left_m'],4)} | {_f(s['right_m'],4)} | "
          f"{_f(s['ratio_L_over_R'],3)} | {_f(s['asym_pct'],1)} |")
    A("")
    A(f"**Body epipolar (Sampson, px)** — threshold in use "
      f"{_f(epi['epipolar_threshold_px'],1)} px, image diag "
      f"{_f(epi['image_diag_px'],0)} px")
    A("")
    A("| joint | n | median px | p90 px | max px |")
    A("|---|---:|---:|---:|---:|")
    for name in JOINT_NAMES:
        v = epi["per_joint"][name]
        A(f"| {name} | {v['n']} | {_f(v['median_px'],1)} | "
          f"{_f(v['p90_px'],1)} | {_f(v['max_px'],1)} |")
    A(f"| **ALL** | {epi['n_pairs']} | **{_f(epi['overall_median_px'],1)}** | "
      f"{_f(epi['overall_p90_px'],1)} |  |")
    A(f"\nOverall median = {_f(epi['overall_median_pct_diag'],3)} % of the "
      f"image diagonal; {_f(100*epi['frac_over_threshold'],1)} % of pairs "
      f"exceed the pipeline threshold.")
    A("")
    A("**Reprojection (px)**")
    A("")
    A("| joint | left n | left median | left max | right n | right median | "
      "right max |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name in JOINT_NAMES:
        l, r = rep[CAM_LEFT]["per_joint"][name], rep[CAM_RIGHT]["per_joint"][name]
        A(f"| {name} | {l['n']} | {_f(l['median_px'],2)} | {_f(l['max_px'],2)} "
          f"| {r['n']} | {_f(r['median_px'],2)} | {_f(r['max_px'],2)} |")
    A(f"| **ALL** |  | **{_f(rep[CAM_LEFT]['overall_median_px'],2)}** |  |  | "
      f"**{_f(rep[CAM_RIGHT]['overall_median_px'],2)}** |  |")
    A("")
    if "error" in ret:
        A(f"**Retargeting**: {ret['error']}")
    else:
        A(f"**Retargeting onto the bundled character** — subject height "
          f"{_f(ret['height_m'],4)} m, uniform scale "
          f"{_f(ret['character_scale'],4)}, "
          f"{ret['n_frames_posed']}/{take['n_frames']} frames posed")
        A("")
        A("| joint | n | median (m) | median % height | max % height |")
        A("|---|---:|---:|---:|---:|")
        for name in JOINT_NAMES:
            v = ret["per_joint"][name]
            A(f"| {name} | {v['n']} | {_f(v['median_m'],4)} | "
              f"{_f(v['median_pct_height'],1)} | "
              f"{_f(v['max_pct_height'],1)} |")
        A(f"| **ALL** |  | **{_f(ret['overall_median_m'],4)}** | "
          f"**{_f(ret['overall_median_pct_height'],1)}** | p90 "
          f"{_f(ret['overall_p90_pct_height'],1)} |")
        A("")
        A("**Bend-plane normal, captured vs character (deg)**")
        A("")
        A("| chain | n | median deg | max deg |")
        A("|---|---:|---:|---:|")
        for label, *_ in BEND_CHAINS:
            v = ret["bend_plane"][label]
            A(f"| {label} | {v['n']} | {_f(v['median_deg'],1)} | "
              f"{_f(v['max_deg'],1)} |")
        A(f"| **ALL** |  | **{_f(ret['bend_plane_overall_median_deg'],1)}** | "
          f"{_f(ret['bend_plane_overall_max_deg'],1)} |")
    if extra:
        A("")
        A(extra)
    return "\n".join(L)


def evaluate(take, source, with_retarget=True) -> dict:
    poses = poses_for_source(take, source)
    blen = bone_length_stats(poses)
    epi = body_epipolar(take["kp2d"], take["rig"])
    rep = reprojection(poses, take["kp2d"], take["rig"])
    ret = retarget_error(poses, take.get("head3d")) if with_retarget else {}
    res = {
        "source": source,
        "n_frames": take["n_frames"],
        "missing": missing_stats(take["kp2d"], poses),
        "bone_lengths": blen,
        "epipolar": epi,
        "reprojection": rep,
        "retarget": ret,
    }
    res["markdown"] = summarise(take, source, poses, blen, epi, rep, ret)
    return res


def _jsonable(o):
    if isinstance(o, dict):
        return {(f"{k[0]}-{k[1]}" if isinstance(k, tuple) else str(k)):
                _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_dir")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--source", default="fitted", choices=list(SOURCES))
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-retarget", action="store_true")
    args = ap.parse_args(argv)

    take = load_take(args.project_dir, args.calib)
    res = evaluate(take, args.source, with_retarget=not args.no_retarget)
    print(res["markdown"])
    if args.json:
        Path(args.json).write_text(json.dumps(_jsonable(res), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

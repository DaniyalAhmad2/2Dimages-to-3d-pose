"""Follow-ups: smoothing order, live-edit derived-joint staleness, drag sim."""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np

BASE = "/tmp/claude-1000/-media-athena-hd3-Projects-pose3d-tool/0fcb8c9e-52c1-483d-9e0a-80896b078fa5/scratchpad/baseline"
OUT = Path("/tmp/claude-1000/-media-athena-hd3-Projects-pose3d-tool/0fcb8c9e-52c1-483d-9e0a-80896b078fa5/scratchpad/find-pipeline")
sys.path.insert(0, BASE)
import metrics as M

from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS
from pose3d.core.skeleton import BONES, JOINT_NAMES, NUM_JOINTS, Joint
from pose3d.geometry.bonefit import (
    fallback_bone_lengths, fit_bone_lengths, measure_bone_lengths, smooth_temporal)
from pose3d.geometry.triangulate import triangulate_points, triangulate_one
from pose3d import pipeline as PL

R = {}
def jf(x, n=4):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jf(v, n) for v in np.asarray(x).tolist()]
    try: return round(float(x), n)
    except Exception: return x

take = M.load_take("workspace/pose3d_projects/Imported_Session")
rig, project, raw, fitted = take["rig"], take["project"], take["pose3d"], take["fitted3d"]
kp2d, T, fids = take["kp2d"], take["n_frames"], take["frame_ids"]
HEIGHT = 0.1178
nosm = M.refit(project, smooth=False)

def summar(po, label):
    bs = M.bone_length_stats(po)
    cvs = [v["cv_pct"] for v in bs["bones"].values() if np.isfinite(v["cv_pct"])]
    rep = M.reprojection(po, kp2d, rig)
    dd = np.linalg.norm(po - nosm, axis=2); dd = dd[np.isfinite(dd)]
    jit = np.linalg.norm(np.diff(po, axis=0), axis=2); jit = jit[np.isfinite(jit)]
    return {"bone_cv_median_pct": jf(np.median(cvs), 3),
            "bone_cv_max_pct": jf(max(cvs), 3),
            "reproj_median_px": {c: jf(rep[c]["overall_median_px"], 2) for c in CAMERAS},
            "vs_fit_only_mean_mm": jf(dd.mean() * 1000, 2),
            "interframe_motion_median_mm": jf(np.median(jit) * 1000, 2)}

# ---- A: ordering. smooth the RAW first, then fit (fit has the last word) ----
tgt = measure_bone_lengths(raw)
fb = fallback_bone_lengths()
bl = {k: (v if v > 1e-6 else fb[k]) for k, v in tgt.items()}

raw_sm = smooth_temporal(raw, alpha=0.6)
pre = np.stack([fit_bone_lengths(raw_sm[t], bl, fill_missing=False) for t in range(T)])
post = M.refit(project, smooth=True)          # shipped: fit then smooth
R["A_ordering"] = {
    "fit_only (smooth=False)": summar(nosm, "fit"),
    "shipped fit_then_smooth": summar(post, "post"),
    "smooth_then_fit": summar(pre, "pre"),
}

# ---- B: live-edit drag simulation, Qt-free reimplementation of
#         ProjectModel._resolve_joint (pose3d/ui/model.py:120-171) ----------
def resolve_joint(p, t, joint, bone_lengths):
    """Exactly what _resolve_joint does, minus the Qt signals."""
    f = p.frames[t]
    pl_, pr_ = f.kp2d[CAM_LEFT][joint], f.kp2d[CAM_RIGHT][joint]
    f.pose3d[joint] = triangulate_one(pl_, pr_, rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                                      rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    f.fitted3d = fit_bone_lengths(f.pose3d, bone_lengths, fill_missing=False)
    return f.fitted3d

# B1: a zero-pixel drag on frame 0014 (worst smoothing frame)
res = {}
for t in (13, 19):  # frames 0014 and 0020
    p = copy.deepcopy(project)
    before = np.array(p.frames[t].fitted3d)
    after = resolve_joint(p, t, int(Joint.LEFT_WRIST), bl)
    dd = np.linalg.norm(after - before, axis=1)
    res[fids[t]] = {"drag_px": 0.0,
                    "dragged_joint": "LEFT_WRIST",
                    "dragged_joint_moved_mm": jf(dd[int(Joint.LEFT_WRIST)] * 1000, 2),
                    "other_joints_median_mm": jf(np.nanmedian(np.delete(dd, int(Joint.LEFT_WRIST))) * 1000, 2),
                    "other_joints_max_mm": jf(np.nanmax(np.delete(dd, int(Joint.LEFT_WRIST))) * 1000, 2),
                    "max_pct_height": jf(np.nanmax(dd) / HEIGHT * 100, 2),
                    "worst_joint": JOINT_NAMES[int(np.nanargmax(dd))]}
R["B1_zero_px_drag"] = res

# B2: neighbour discontinuity — after the drag, frame t is unsmoothed while
# t-1 / t+1 are still smoothed. Measure the inter-frame jump before/after.
p = copy.deepcopy(project)
t = 13
def step(arr, t):
    return (np.nanmedian(np.linalg.norm(arr[t] - arr[t - 1], axis=1)),
            np.nanmedian(np.linalg.norm(arr[t + 1] - arr[t], axis=1)))
seq_before = np.array(fitted)
b_prev, b_next = step(seq_before, t)
after = resolve_joint(p, t, int(Joint.LEFT_WRIST), bl)
seq_after = seq_before.copy(); seq_after[t] = after
a_prev, a_next = step(seq_after, t)
R["B2_neighbour_discontinuity_mm"] = {
    "frame": fids[t],
    "before_prev_mm": jf(b_prev * 1000, 2), "before_next_mm": jf(b_next * 1000, 2),
    "after_prev_mm": jf(a_prev * 1000, 2), "after_next_mm": jf(a_next * 1000, 2),
    "ratio_prev": jf(a_prev / b_prev, 2), "ratio_next": jf(a_next / b_next, 2)}

# B3: DERIVED JOINT STALENESS. Drag LEFT_HIP 60 px in the left view and see
# whether PELVIS (the retarget root) follows.
DRAG_PX = 60.0
for jname, jidx, derived, parents in (
        ("LEFT_HIP", Joint.LEFT_HIP, Joint.PELVIS, (Joint.LEFT_HIP, Joint.RIGHT_HIP)),
        ("LEFT_SHOULDER", Joint.LEFT_SHOULDER, Joint.NECK,
         (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER))):
    p = copy.deepcopy(project)
    f = p.frames[0]
    f.kp2d[CAM_LEFT][int(jidx)] = f.kp2d[CAM_LEFT][int(jidx)] + np.array([DRAG_PX, 0.0])
    f.corrected[CAM_LEFT][int(jidx)] = True
    app = resolve_joint(p, 0, int(jidx), bl)      # what the app does today

    # what it SHOULD do: the derived midpoint follows its parent
    q = copy.deepcopy(project)
    g = q.frames[0]
    g.kp2d[CAM_LEFT][int(jidx)] = g.kp2d[CAM_LEFT][int(jidx)] + np.array([DRAG_PX, 0.0])
    g.kp2d[CAM_LEFT][int(derived)] = 0.5 * (g.kp2d[CAM_LEFT][int(parents[0])]
                                            + g.kp2d[CAM_LEFT][int(parents[1])])
    resolve_joint(q, 0, int(jidx), bl)
    correct = resolve_joint(q, 0, int(derived), bl)

    dd = np.linalg.norm(app - correct, axis=1)
    R[f"B3_stale_derived_{jname}"] = {
        "drag_px": DRAG_PX,
        "derived_joint": derived.name,
        "derived_2d_stale_px": jf(np.linalg.norm(
            f.kp2d[CAM_LEFT][int(derived)]
            - 0.5 * (f.kp2d[CAM_LEFT][int(parents[0])] + f.kp2d[CAM_LEFT][int(parents[1])])), 2),
        "derived_3d_error_mm": jf(dd[int(derived)] * 1000, 2),
        "derived_3d_error_pct_height": jf(dd[int(derived)] / HEIGHT * 100, 2),
        "whole_pose_median_error_mm": jf(np.nanmedian(dd) * 1000, 2),
        "whole_pose_max_error_mm": jf(np.nanmax(dd) * 1000, 2),
    }

# B4: sensitivity — mm of 3D per px of 2D drag, so the numbers scale
p = copy.deepcopy(project)
f0 = p.frames[0]
base = triangulate_one(f0.kp2d[CAM_LEFT][int(Joint.LEFT_HIP)],
                       f0.kp2d[CAM_RIGHT][int(Joint.LEFT_HIP)],
                       rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                       rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
moved = triangulate_one(f0.kp2d[CAM_LEFT][int(Joint.LEFT_HIP)] + np.array([10.0, 0]),
                        f0.kp2d[CAM_RIGHT][int(Joint.LEFT_HIP)],
                        rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                        rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
R["B4_mm_per_px"] = jf(np.linalg.norm(moved - base) * 1000 / 10.0, 4)

# ---- C: what a single-view ray-constrained placement would buy (H7) -------
# The 2 single-view joints today become NaN then get frozen by the smoother.
R["C_single_view"] = {
    "pairs": [["0012", "RIGHT_KNEE"], ["0021", "LEFT_ANKLE"]],
    "frozen_at_prev_frame_error_vs_next_real_mm": [20.21, 38.5],
    "pct_height": [jf(20.21 / (HEIGHT * 1000) * 100, 1), jf(38.5 / (HEIGHT * 1000) * 100, 1)],
}

# ---- D: how much does the fit's asymmetric target cost? ------------------
sym_bl = dict(bl)
for label, lk, rk in M.SYMMETRY_PAIRS:
    m = 0.5 * (bl[lk] + bl[rk]); sym_bl[lk] = m; sym_bl[rk] = m
sym_fit = np.stack([fit_bone_lengths(raw[t], sym_bl, fill_missing=False) for t in range(T)])
dd = np.linalg.norm(sym_fit - nosm, axis=2); dd = dd[np.isfinite(dd)]
rep_s = M.reprojection(sym_fit, kp2d, rig)
rep_n = M.reprojection(nosm, kp2d, rig)
R["D_symmetric_targets"] = {
    "pose_change_median_mm": jf(np.median(dd) * 1000, 3),
    "pose_change_max_mm": jf(dd.max() * 1000, 3),
    "reproj_median_px_symmetric": {c: jf(rep_s[c]["overall_median_px"], 3) for c in CAMERAS},
    "reproj_median_px_asymmetric": {c: jf(rep_n[c]["overall_median_px"], 3) for c in CAMERAS},
}

(OUT / "probe2.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))

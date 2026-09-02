"""Auditor experiments for the triangulation / fit / smoothing / live-edit dimension.

Run:  cd /media/athena/hd3/Projects/pose3d-tool && MPLBACKEND=Agg .venv/bin/python \
      /tmp/.../find-pipeline/pipeline_probe.py
"""
from __future__ import annotations

import copy, json, sys, time
from pathlib import Path

import numpy as np

BASE = "/tmp/claude-1000/-media-athena-hd3-Projects-pose3d-tool/0fcb8c9e-52c1-483d-9e0a-80896b078fa5/scratchpad/baseline"
OUT = Path("/tmp/claude-1000/-media-athena-hd3-Projects-pose3d-tool/0fcb8c9e-52c1-483d-9e0a-80896b078fa5/scratchpad/find-pipeline")
sys.path.insert(0, BASE)
import metrics as M  # noqa: E402

from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS  # noqa: E402
from pose3d.core.skeleton import BONES, JOINT_NAMES, NUM_JOINTS, Joint  # noqa: E402
from pose3d.geometry.bonefit import (  # noqa: E402
    fallback_bone_lengths, fit_bone_lengths, measure_bone_lengths, smooth_temporal,
)
from pose3d.geometry.triangulate import triangulate_points, undistort_normalized  # noqa: E402
from pose3d import pipeline as PL  # noqa: E402
import cv2  # noqa: E402

TAKE_DIR = "workspace/pose3d_projects/Imported_Session"
R = {}


def jf(x, n=4):
    if isinstance(x, (list, tuple)):
        return [jf(v, n) for v in x]
    try:
        return round(float(x), n)
    except Exception:
        return x


take = M.load_take(TAKE_DIR)
rig = take["rig"]
project = take["project"]
raw = take["pose3d"]
fitted = take["fitted3d"]
fids = take["frame_ids"]
T = take["n_frames"]
kp2d = take["kp2d"]
scores = take["scores"]

# subject height reference: z-extent of de-tilted pose used by baseline = 0.118 m
HEIGHT = 0.1178

nosm = M.refit(project, smooth=False)
sm = M.refit(project, smooth=True)

R["setup"] = {"frames": T, "height_m": HEIGHT,
              "fitted_equals_refit_max_m": jf(np.nanmax(np.abs(fitted - sm)), 6)}

# =====================================================================
# H1  smoothing displacement
# =====================================================================
d = np.linalg.norm(sm - nosm, axis=2)          # (T,15) metres
per_joint = np.nanmedian(d, axis=0)
per_joint_max = np.nanmax(d, axis=0)
per_frame = np.nanmedian(d, axis=1)
flat = d[np.isfinite(d)]
worst = np.unravel_index(np.nanargmax(np.where(np.isfinite(d), d, -1)), d.shape)
order_j = np.argsort(-np.nan_to_num(per_joint))
order_f = np.argsort(-np.nan_to_num(per_frame))
R["H1_smoothing"] = {
    "median_mm": jf(np.median(flat) * 1000, 2),
    "mean_mm": jf(flat.mean() * 1000, 2),
    "p90_mm": jf(np.percentile(flat, 90) * 1000, 2),
    "max_mm": jf(flat.max() * 1000, 2),
    "median_pct_height": jf(np.median(flat) / HEIGHT * 100, 2),
    "p90_pct_height": jf(np.percentile(flat, 90) / HEIGHT * 100, 2),
    "max_pct_height": jf(flat.max() / HEIGHT * 100, 2),
    "worst_joint_frame": [JOINT_NAMES[worst[1]], fids[worst[0]], jf(d[worst] * 1000, 2)],
    "top_joints_median_mm": {JOINT_NAMES[j]: jf(per_joint[j] * 1000, 2) for j in order_j[:6]},
    "top_joints_max_mm": {JOINT_NAMES[j]: jf(per_joint_max[j] * 1000, 2) for j in order_j[:6]},
    "top_frames_median_mm": {fids[t]: jf(per_frame[t] * 1000, 2) for t in order_f[:6]},
    "frac_over_5pct_height": jf((flat > 0.05 * HEIGHT).mean(), 3),
    "frac_over_10pct_height": jf((flat > 0.10 * HEIGHT).mean(), 3),
}

# EMA lag: is this stop-motion? inter-frame raw motion
mo = np.linalg.norm(np.diff(raw, axis=0), axis=2)
mo = mo[np.isfinite(mo)]
R["H1_motion"] = {
    "interframe_raw_median_mm": jf(np.median(mo) * 1000, 2),
    "interframe_raw_p90_mm": jf(np.percentile(mo, 90) * 1000, 2),
    "interframe_pct_height_median": jf(np.median(mo) / HEIGHT * 100, 2),
    "ema_steady_state_lag_frames": jf((1 - 0.6) / 0.6, 3),
    "predicted_lag_mm": jf(np.median(mo) * 1000 * (1 - 0.6) / 0.6, 2),
}

# alpha sweep: bone CV and smoothing displacement vs alpha
sweep = {}
for a in (1.0, 0.9, 0.8, 0.6, 0.4):
    p = copy.deepcopy(project)
    PL.fit_project(p, smooth=(a < 1.0), alpha=a)
    po = np.stack([f.fitted3d for f in p.frames])
    bs = M.bone_length_stats(po)
    cvs = [v["cv_pct"] for v in bs["bones"].values() if np.isfinite(v["cv_pct"])]
    dd = np.linalg.norm(po - nosm, axis=2)
    dd = dd[np.isfinite(dd)]
    rep = M.reprojection(po, kp2d, rig)
    sweep[str(a)] = {"bone_cv_median_pct": jf(np.median(cvs), 2),
                     "bone_cv_max_pct": jf(max(cvs), 2),
                     "disp_mean_mm": jf(dd.mean() * 1000, 2),
                     "reproj_median_px": {c: jf(rep[c]["overall_median_px"], 2) for c in CAMERAS}}
R["H1_alpha_sweep"] = sweep

# =====================================================================
# EXTRA: does smooth_temporal resurrect NaN joints the fit refused to invent?
# =====================================================================
nan_nosm = np.isnan(nosm).any(2)
nan_sm = np.isnan(sm).any(2)
resurrected = nan_nosm & ~nan_sm
idx = np.argwhere(resurrected)
R["EXTRA_nan_resurrection"] = {
    "n_nan_nosmooth": int(nan_nosm.sum()),
    "n_nan_smoothed": int(nan_sm.sum()),
    "n_resurrected": int(resurrected.sum()),
    "which": [[fids[t], JOINT_NAMES[j]] for t, j in idx],
    "raw_nan": [[fids[t], JOINT_NAMES[j]] for t, j in np.argwhere(np.isnan(raw).any(2))],
}
# for a resurrected joint: how far is the invented point from the last real one,
# and what does its bone length become?
res_detail = []
for t, j in idx:
    prev = sm[t - 1, j] if t > 0 else None
    res_detail.append({
        "frame": fids[t], "joint": JOINT_NAMES[j],
        "invented_xyz": jf(sm[t, j].tolist(), 5),
        "equals_prev_frame": bool(prev is not None and np.allclose(sm[t, j], prev)),
        "dist_to_next_real_mm": jf(
            np.linalg.norm(sm[t, j] - nosm[t + 1, j]) * 1000, 2)
        if t + 1 < T and np.isfinite(nosm[t + 1, j]).all() else None,
    })
R["EXTRA_nan_resurrection"]["detail"] = res_detail

# =====================================================================
# H2  nose / HEAD displaced by the fixed NECK-HEAD bone
# =====================================================================
dfit = np.linalg.norm(nosm - raw, axis=2)      # fit-only displacement (no smoothing)
pj = np.nanmedian(dfit, axis=0)
R["H2_fit_displacement_mm"] = {JOINT_NAMES[j]: jf(pj[j] * 1000, 2) for j in range(NUM_JOINTS)}
R["H2_head"] = {
    "HEAD_median_mm": jf(pj[int(Joint.HEAD)] * 1000, 2),
    "HEAD_max_mm": jf(np.nanmax(dfit[:, int(Joint.HEAD)]) * 1000, 2),
    "other_joints_median_mm": jf(np.median(np.delete(pj, int(Joint.HEAD))) * 1000, 2),
    "rank_of_HEAD": int(np.argsort(-pj).tolist().index(int(Joint.HEAD))) + 1,
}
# neck->nose raw length variation (does the "bone" actually change?)
seg = np.linalg.norm(raw[:, int(Joint.HEAD)] - raw[:, int(Joint.NECK)], axis=1)
seg = seg[np.isfinite(seg)]
R["H2_neck_head_raw"] = {"median_m": jf(np.median(seg), 5), "std_m": jf(seg.std(), 5),
                         "cv_pct": jf(100 * seg.std() / np.median(seg), 2),
                         "min_m": jf(seg.min(), 5), "max_m": jf(seg.max(), 5),
                         "range_pct": jf(100 * (seg.max() - seg.min()) / np.median(seg), 2)}
# what the fit costs the nose vs its own observation, per frame
R["H2_head_worst_frames_mm"] = {fids[t]: jf(dfit[t, int(Joint.HEAD)] * 1000, 2)
                                for t in np.argsort(-np.nan_to_num(dfit[:, int(Joint.HEAD)]))[:5]}

# =====================================================================
# H3  raw bone length std + L/R asymmetry
# =====================================================================
bs_raw = M.bone_length_stats(raw)
R["H3_raw_bones"] = {v["name"]: {"median_mm": jf(v["median_m"] * 1000, 2),
                                 "std_mm": jf(v["std_m"] * 1000, 2),
                                 "cv_pct": jf(v["cv_pct"], 2),
                                 "range_mm": jf((v["max_m"] - v["min_m"]) * 1000, 2)}
                     for v in bs_raw["bones"].values()}
R["H3_symmetry_raw"] = {k: {"left_mm": jf(v["left_m"] * 1000, 2),
                            "right_mm": jf(v["right_m"] * 1000, 2),
                            "asym_pct": jf(v["asym_pct"], 2),
                            "diff_mm": jf(abs(v["left_m"] - v["right_m"]) * 1000, 2)}
                        for k, v in bs_raw["symmetry"].items()}
# what target lengths the fit actually used
tgt = measure_bone_lengths(raw)
fb = fallback_bone_lengths()
R["H3_targets_used"] = {M.BONE_NAMES[k]: jf(v * 1000, 2) for k, v in tgt.items()}
R["H3_symmetric_target_would_be"] = {}
for label, lk, rk in M.SYMMETRY_PAIRS:
    R["H3_symmetric_target_would_be"][label] = {
        "L_mm": jf(tgt[lk] * 1000, 2), "R_mm": jf(tgt[rk] * 1000, 2),
        "mean_mm": jf((tgt[lk] + tgt[rk]) / 2 * 1000, 2),
        "each_moves_mm": jf(abs(tgt[lk] - tgt[rk]) / 2 * 1000, 2)}

# =====================================================================
# H4  fallback lengths
# =====================================================================
R["H4_fallback"] = {
    "any_measured_zero": {M.BONE_NAMES[k]: jf(v, 6) for k, v in tgt.items() if v <= 1e-6},
    "n_measured_zero": int(sum(1 for v in tgt.values() if v <= 1e-6)),
    "fallback_thigh_m": fb[(int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE))],
    "measured_thigh_m": jf(tgt[(int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE))], 5),
    "ratio_fallback_over_measured": jf(
        fb[(int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE))]
        / tgt[(int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE))], 1),
}
# simulate: NaN the left shin in every frame -> that bone measures 0 -> fallback 0.44 m
p = copy.deepcopy(project)
for f in p.frames:
    f.pose3d[int(Joint.LEFT_ANKLE)] = np.nan
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = np.nan
PL.fit_project(p, smooth=False)
sim = np.stack([f.fitted3d for f in p.frames])
tgt_sim = measure_bone_lengths(np.stack([f.pose3d for f in p.frames]))
blen_sim = {k: (v if v > 1e-6 else fb[k]) for k, v in tgt_sim.items()}
d_sim = np.linalg.norm(sim - nosm, axis=2)
R["H4_sim_missing_left_shin"] = {
    "L_shin_target_used_m": jf(blen_sim[(int(Joint.LEFT_KNEE), int(Joint.LEFT_ANKLE))], 4),
    "is_fallback": bool(tgt_sim[(int(Joint.LEFT_KNEE), int(Joint.LEFT_ANKLE))] <= 1e-6),
    "other_joints_moved_median_mm": jf(np.nanmedian(d_sim) * 1000, 2),
    "other_joints_moved_max_mm": jf(np.nanmax(d_sim) * 1000, 2),
    "worst_joint": JOINT_NAMES[int(np.nanargmax(np.nanmedian(d_sim, axis=0)))],
    "per_joint_median_mm": {JOINT_NAMES[j]: jf(np.nanmedian(d_sim[:, j]) * 1000, 2)
                            for j in range(NUM_JOINTS)},
    "L_knee_moved_median_mm": jf(np.nanmedian(d_sim[:, int(Joint.LEFT_KNEE)]) * 1000, 2),
    "subject_height_after_m": jf(
        np.nanmax(sim[..., :].reshape(-1, 3), axis=0)
        - np.nanmin(sim[..., :].reshape(-1, 3), axis=0), 4).tolist()
    if True else None,
}
# also: fill_missing=True variant (what the *live edit* / default arg would do)
one = fit_bone_lengths(p.frames[0].pose3d, blen_sim, fill_missing=True)
R["H4_sim_fill_missing_true"] = {
    "L_ankle_placed_at_dist_from_knee_m": jf(
        np.linalg.norm(one[int(Joint.LEFT_ANKLE)] - one[int(Joint.LEFT_KNEE)]), 4),
    "subject_span_m": jf(float(np.nanmax(np.linalg.norm(
        one[:, None] - one[None, :], axis=2))), 4),
}

# =====================================================================
# H5  derived NECK/PELVIS: 2D midpoint triangulated vs 3D midpoint
# =====================================================================
def midpoint_check(poses, kp):
    out = {}
    for name, dj, (a, b) in (("NECK", Joint.NECK, (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)),
                             ("PELVIS", Joint.PELVIS, (Joint.LEFT_HIP, Joint.RIGHT_HIP))):
        p3 = poses[:, int(dj)]
        mid = 0.5 * (poses[:, int(a)] + poses[:, int(b)])
        dd = np.linalg.norm(p3 - mid, axis=1)
        dd = dd[np.isfinite(dd)]
        span = np.linalg.norm(poses[:, int(a)] - poses[:, int(b)], axis=1)
        span = span[np.isfinite(span)]
        # also verify the stored 2D really is the pixel midpoint
        mid2d = {c: np.linalg.norm(kp[c][:, int(dj)]
                                   - 0.5 * (kp[c][:, int(a)] + kp[c][:, int(b)]), axis=1)
                 for c in CAMERAS}
        out[name] = {
            "median_mm": jf(np.median(dd) * 1000, 3),
            "max_mm": jf(dd.max() * 1000, 3),
            "pct_height_median": jf(np.median(dd) / HEIGHT * 100, 3),
            "pct_of_segment_span": jf(100 * np.median(dd) / np.median(span), 2),
            "segment_span_mm": jf(np.median(span) * 1000, 2),
            "stored_2d_is_pixel_midpoint_maxpx": {
                c: jf(np.nanmax(mid2d[c]), 4) for c in CAMERAS},
        }
    return out


R["H5_midpoint"] = midpoint_check(raw, kp2d)

# how big could it get?  depth difference between the two shoulders
zl = raw[:, int(Joint.LEFT_SHOULDER)]
zr = raw[:, int(Joint.RIGHT_SHOULDER)]
cam_c = -rig.ext[CAM_LEFT].R.T @ rig.ext[CAM_LEFT].t.reshape(3)
dl = np.linalg.norm(zl - cam_c, axis=1)
dr = np.linalg.norm(zr - cam_c, axis=1)
R["H5_depth_spread"] = {
    "shoulder_depth_diff_median_mm": jf(np.nanmedian(np.abs(dl - dr)) * 1000, 2),
    "shoulder_depth_diff_max_mm": jf(np.nanmax(np.abs(dl - dr)) * 1000, 2),
    "camera_distance_m": jf(np.nanmedian(dl), 3),
    "relative_depth_spread_pct": jf(100 * np.nanmedian(np.abs(dl - dr)) / np.nanmedian(dl), 3),
}

# =====================================================================
# H6  live edit: the jump on a 0-px drag
# =====================================================================
bl = {k: (v if v > 1e-6 else fb[k]) for k, v in tgt.items()}
jumps = []
for t, f in enumerate(project.frames):
    single = fit_bone_lengths(f.pose3d, bl, fill_missing=False)
    dd = np.linalg.norm(single - fitted[t], axis=1)
    jumps.append(dd)
jumps = np.array(jumps)
fl = jumps[np.isfinite(jumps)]
R["H6_zero_px_drag_jump"] = {
    "median_mm": jf(np.median(fl) * 1000, 2),
    "mean_mm": jf(fl.mean() * 1000, 2),
    "p90_mm": jf(np.percentile(fl, 90) * 1000, 2),
    "max_mm": jf(fl.max() * 1000, 2),
    "median_pct_height": jf(np.median(fl) / HEIGHT * 100, 2),
    "max_pct_height": jf(fl.max() / HEIGHT * 100, 2),
    "per_frame_max_mm": {fids[t]: jf(np.nanmax(jumps[t]) * 1000, 2)
                         for t in np.argsort(-np.nan_to_num(np.nanmax(jumps, axis=1)))[:5]},
    "note": "distance between stored fitted3d (smoothed) and the single-frame "
            "unsmoothed refit ProjectModel._resolve_joint writes on any drag",
}
# and: the WHOLE pose changes, not only the dragged joint
R["H6_joints_moved_by_one_drag"] = {
    "median_joints_moving_gt_1mm": jf(np.median((jumps > 0.001).sum(1)), 1),
    "of_total_joints": NUM_JOINTS,
}
# _accuracy uses pose3d (raw), the view shows fitted3d
acc_raw = M.reprojection(raw, kp2d, rig)
acc_fit = M.reprojection(fitted, kp2d, rig)
R["H6_accuracy_gauge_mismatch_px"] = {
    "gauge_shows_raw_median": {c: jf(acc_raw[c]["overall_median_px"], 2) for c in CAMERAS},
    "displayed_pose_fitted_median": {c: jf(acc_fit[c]["overall_median_px"], 2) for c in CAMERAS},
    "ratio": jf(acc_fit[CAM_LEFT]["overall_median_px"]
                / acc_raw[CAM_LEFT]["overall_median_px"], 2),
}
# derived-joint staleness: dragging a shoulder does not move NECK's 2D
R["H6_derived_stale"] = {
    "code": "ProjectModel._resolve_joint re-triangulates ONLY the dragged joint index; "
            "NECK/PELVIS 2D are derive_joints midpoints of the shoulders/hips and are "
            "never recomputed after an edit",
    "neck_shift_for_a_10px_shoulder_drag_px": 5.0,
    "median_shoulder_span_px_left": jf(np.nanmedian(np.linalg.norm(
        kp2d[CAM_LEFT][:, int(Joint.LEFT_SHOULDER)]
        - kp2d[CAM_LEFT][:, int(Joint.RIGHT_SHOULDER)], axis=1)), 1),
}

# =====================================================================
# H7  joints present in exactly one view
# =====================================================================
nl = np.isnan(kp2d[CAM_LEFT]).any(2)
nr = np.isnan(kp2d[CAM_RIGHT]).any(2)
one_view = (nl ^ nr)
R["H7_single_view"] = {
    "total_obs": int(T * NUM_JOINTS),
    "n_exactly_one_view": int(one_view.sum()),
    "pct": jf(100 * one_view.sum() / (T * NUM_JOINTS), 2),
    "which": [[fids[t], JOINT_NAMES[j], "left_only" if nr[t, j] else "right_only"]
              for t, j in np.argwhere(one_view)],
    "n_neither_view": int((nl & nr).sum()),
    "n_raw3d_nan": int(np.isnan(raw).any(2).sum()),
}
R["H7_validate_would_drop_now"] = int(M.triangulate_validated(project, rig)[1])

# =====================================================================
# H8  DLT vs optimal (Sampson-corrected) vs midpoint triangulation
# =====================================================================
def tri_midpoint(pl_, pr_, rig):
    """Mid-point-of-common-perpendicular triangulation."""
    out = np.full((pl_.shape[0], 3), np.nan)
    ok = ~(np.isnan(pl_).any(1) | np.isnan(pr_).any(1))
    if not ok.any():
        return out
    nlz = undistort_normalized(pl_[ok], rig.intr[CAM_LEFT])
    nrz = undistort_normalized(pr_[ok], rig.intr[CAM_RIGHT])
    El, Er = rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT]
    Cl = -El.R.T @ El.t.reshape(3)
    Cr = -Er.R.T @ Er.t.reshape(3)
    res = []
    for a, b in zip(nlz, nrz):
        d1 = El.R.T @ np.array([a[0], a[1], 1.0]); d1 /= np.linalg.norm(d1)
        d2 = Er.R.T @ np.array([b[0], b[1], 1.0]); d2 /= np.linalg.norm(d2)
        w = Cl - Cr
        a11 = 1.0; a12 = -d1 @ d2; a22 = -1.0
        b1 = -d1 @ w; b2 = -d2 @ w
        Amat = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
        bvec = np.array([-(d1 @ w), -(d2 @ w)])
        s, t_ = np.linalg.solve(Amat, bvec)
        res.append(0.5 * ((Cl + s * d1) + (Cr + t_ * d2)))
    out[ok] = np.array(res)
    return out


def tri_optimal(pl_, pr_, rig):
    """cv2.correctMatches (Hartley-Sturm optimal correction) + DLT."""
    from pose3d.geometry.triangulate import fundamental_matrix
    out = np.full((pl_.shape[0], 3), np.nan)
    ok = ~(np.isnan(pl_).any(1) | np.isnan(pr_).any(1))
    if not ok.any():
        return out
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    p1 = pl_[ok].reshape(1, -1, 2).astype(np.float64)
    p2 = pr_[ok].reshape(1, -1, 2).astype(np.float64)
    n1, n2 = cv2.correctMatches(F, p1, p2)
    out[ok] = triangulate_points(n1.reshape(-1, 2), n2.reshape(-1, 2),
                                 rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                                 rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    return out


mid = np.stack([tri_midpoint(kp2d[CAM_LEFT][t], kp2d[CAM_RIGHT][t], rig) for t in range(T)])
opt = np.stack([tri_optimal(kp2d[CAM_LEFT][t], kp2d[CAM_RIGHT][t], rig) for t in range(T)])
def cmp(x, name):
    dd = np.linalg.norm(x - raw, axis=2); dd = dd[np.isfinite(dd)]
    bs = M.bone_length_stats(x)
    cvs = [v["cv_pct"] for v in bs["bones"].values() if np.isfinite(v["cv_pct"])]
    rep = M.reprojection(x, kp2d, rig)
    return {"vs_DLT_median_mm": jf(np.median(dd) * 1000, 4),
            "vs_DLT_max_mm": jf(dd.max() * 1000, 4),
            "bone_cv_median_pct": jf(np.median(cvs), 3),
            "reproj_median_px": {c: jf(rep[c]["overall_median_px"], 3) for c in CAMERAS}}


bs0 = M.bone_length_stats(raw)
cv0 = [v["cv_pct"] for v in bs0["bones"].values() if np.isfinite(v["cv_pct"])]
rep0 = M.reprojection(raw, kp2d, rig)
R["H8_triangulators"] = {
    "DLT_baseline": {"bone_cv_median_pct": jf(np.median(cv0), 3),
                     "reproj_median_px": {c: jf(rep0[c]["overall_median_px"], 3) for c in CAMERAS}},
    "midpoint": cmp(mid, "midpoint"),
    "optimal_correctMatches": cmp(opt, "optimal"),
}

# =====================================================================
# H9  pelvis/root displacement + solve time lm vs trf
# =====================================================================
dp = np.linalg.norm(nosm - raw, axis=2)
R["H9_root"] = {
    "PELVIS_median_mm": jf(np.nanmedian(dp[:, int(Joint.PELVIS)]) * 1000, 2),
    "PELVIS_max_mm": jf(np.nanmax(dp[:, int(Joint.PELVIS)]) * 1000, 2),
    "all_joints_median_mm": jf(np.nanmedian(dp) * 1000, 2),
    "centroid_shift_median_mm": jf(np.nanmedian(np.linalg.norm(
        np.nanmean(nosm, axis=1) - np.nanmean(raw, axis=1), axis=1)) * 1000, 2),
}
t0 = time.perf_counter()
for f in project.frames:
    fit_bone_lengths(f.pose3d, bl, fill_missing=False)
t_all = (time.perf_counter() - t0) / T
# force trf by nulling joints
sparse = project.frames[0].pose3d.copy()
sparse[[2, 3, 4, 5, 6]] = np.nan
t0 = time.perf_counter()
for _ in range(10):
    fit_bone_lengths(sparse, bl, fill_missing=False)
t_trf = (time.perf_counter() - t0) / 10
R["H9_timing"] = {"per_frame_ms_default": jf(t_all * 1000, 2),
                  "per_frame_ms_sparse_trf": jf(t_trf * 1000, 2),
                  "n_variables": NUM_JOINTS * 3,
                  "constraints": len(BONES)}
# does the fit honour the bone targets it was given?
bs_n = M.bone_length_stats(nosm)
R["H9_bone_target_residual"] = {
    v["name"]: jf((v["median_m"] - tgt[k]) * 1000, 3)
    for k, v in bs_n["bones"].items()}
R["H9_bone_weight"] = {"data_weight": 1.0, "bone_weight": 5.0,
                       "test_uses": 20.0,
                       "max_abs_target_residual_mm": jf(max(
                           abs(v) for v in R["H9_bone_target_residual"].values()), 3)}

# =====================================================================
# EXTRA: destructiveness of validate_cross_view (2D is overwritten in place)
# =====================================================================
p = copy.deepcopy(project)
tot = []
for i in range(3):
    tot.append(PL.validate_cross_view(p, rig))
R["EXTRA_validate_repeat_drops"] = tot
R["EXTRA_epi_threshold_px"] = jf(PL.epipolar_threshold(rig), 2)
epi = M.body_epipolar(kp2d, rig)
R["EXTRA_epipolar"] = {"median_px": jf(epi["overall_median_px"], 2),
                       "max_px": jf(max(v["max_px"] for v in epi["per_joint"].values() if np.isfinite(v["max_px"])), 2),
                       "threshold_px": jf(PL.epipolar_threshold(rig), 2),
                       "headroom_x": jf(PL.epipolar_threshold(rig) / max(v["max_px"] for v in epi["per_joint"].values() if np.isfinite(v["max_px"])), 2)}

# =====================================================================
# EXTRA: derive_joints has no confidence gate -> NECK/PELVIS from bad parents
# =====================================================================
R["EXTRA_scores"] = {}
for c in CAMERAS:
    s = scores[c]
    R["EXTRA_scores"][c] = {
        "median_all": jf(np.nanmedian(s), 3),
        "per_joint_median": {JOINT_NAMES[j]: jf(np.nanmedian(s[:, j]), 3)
                             for j in range(NUM_JOINTS)},
        "frac_below_0.35": jf(float((s < 0.35).mean()), 3),
        "frac_below_0.60": jf(float((s < 0.60).mean()), 3),
    }

(OUT / "probe.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))

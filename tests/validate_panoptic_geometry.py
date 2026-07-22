"""Validate the two-camera geometry core on REAL CMU Panoptic data.

For each frame:
  1. Load GT 3D (real human pose, cm) from Panoptic hdPose3d.
  2. Project into two real HD cameras using Panoptic's real calibration
     (K, distortion, R, t) -> 2D pixel observations (the ideal detector).
  3. Run OUR pipeline (undistort -> DLT triangulate) with the same calibration.
  4. Compare recovered 3D vs GT 3D -> per-joint error (mm).
  5. Repeat with realistic pixel noise to emulate detector error.
  6. Run bone-length fit over the noisy sequence and report stabilisation.

This exercises real intrinsics/distortion/extrinsics and real human motion.
Run: .venv/bin/python -m tests.validate_panoptic_geometry
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose3d.datasets.panoptic import (
    frame_index, list_pose_frames, load_camera, load_pose3d,
)
from pose3d.geometry.bonefit import (
    fallback_bone_lengths, fit_bone_lengths, measure_bone_lengths,
    smooth_temporal,
)
from pose3d.geometry.triangulate import reprojection_error, triangulate_points
from pose3d.core.skeleton import JOINT_NAMES, NUM_JOINTS
from tests.synth import project  # cv2 projectPoints wrapper

DATA = Path(__file__).resolve().parent.parent / "data"
SEQ = "171204_pose1_sample"
CALIB = DATA / f"calibration_{SEQ}.json"
POSE_DIR = DATA / "hdPose3d" / "hdPose3d_stage1_coco19"
# Two HD cameras with a wide baseline (like the client's L/R 2-cam rig)
CAM_L, CAM_R = "00_00", "00_12"


def _project(pts3d, cam):
    valid = ~np.isnan(pts3d).any(1)
    out = np.full((pts3d.shape[0], 2), np.nan)
    if valid.any():
        out[valid] = project(pts3d[valid], cam.intr.K, cam.intr.dist,
                              cam.ext.R, cam.ext.t)
    return out


def main():
    cam_l = load_camera(CALIB, CAM_L)
    cam_r = load_camera(CALIB, CAM_R)
    print(f"Loaded Panoptic cameras {CAM_L} & {CAM_R}")
    print(f"  {CAM_L}: f=({cam_l.intr.K[0,0]:.1f},{cam_l.intr.K[1,1]:.1f}) "
          f"dist={cam_l.intr.dist.ravel()[:3]}")
    baseline = np.linalg.norm(cam_l.ext.camera_center - cam_r.ext.camera_center)
    print(f"  baseline between cameras: {baseline:.1f} cm")

    frames = list_pose_frames(POSE_DIR)
    gts, obs_l, obs_r = [], [], []
    for fp in frames:
        gt = load_pose3d(fp)
        if np.isnan(gt).all():
            continue
        gts.append(gt)
        obs_l.append(_project(gt, cam_l))
        obs_r.append(_project(gt, cam_r))
    gts = np.array(gts)
    print(f"\nFrames with a body: {len(gts)}")

    # --- 1. PERFECT observations ---
    errs = []
    for gt, pl, pr in zip(gts, obs_l, obs_r):
        rec = triangulate_points(pl, pr, cam_l.intr, cam_r.intr,
                                 cam_l.ext, cam_r.ext)
        e = np.linalg.norm(rec - gt, axis=1)   # cm
        errs.append(e)
    errs = np.array(errs)
    print("\n=== [A] Ideal 2D (perfect detector) ===")
    print(f"  mean 3D error: {np.nanmean(errs)*10:.3f} mm   "
          f"max: {np.nanmax(errs)*10:.3f} mm")

    # --- 2. NOISY observations (emulate detector error) ---
    rng = np.random.default_rng(0)
    for noise_px in (2.0, 5.0):
        errs_n = []
        for gt, pl, pr in zip(gts, obs_l, obs_r):
            pln = pl + rng.normal(0, noise_px, pl.shape)
            prn = pr + rng.normal(0, noise_px, pr.shape)
            rec = triangulate_points(pln, prn, cam_l.intr, cam_r.intr,
                                     cam_l.ext, cam_r.ext)
            errs_n.append(np.linalg.norm(rec - gt, axis=1))
        errs_n = np.array(errs_n)
        print(f"\n=== [B] {noise_px:.0f}px Gaussian 2D noise ===")
        print(f"  mean 3D error: {np.nanmean(errs_n)*10:.1f} mm   "
              f"median: {np.nanmedian(errs_n)*10:.1f} mm")

    # --- 3. Bone-fit stabilisation on noisy sequence ---
    raw_seq = []
    for gt, pl, pr in zip(gts, obs_l, obs_r):
        pln = pl + rng.normal(0, 5.0, pl.shape)
        prn = pr + rng.normal(0, 5.0, pr.shape)
        raw_seq.append(triangulate_points(pln, prn, cam_l.intr, cam_r.intr,
                                          cam_l.ext, cam_r.ext))
    raw_seq = np.array(raw_seq)
    gt_lengths = measure_bone_lengths(gts)
    fb = fallback_bone_lengths()
    lengths = {k: (v if v > 1e-6 else fb[k]) for k, v in gt_lengths.items()}
    fitted = np.array([fit_bone_lengths(p, lengths) for p in raw_seq])
    fitted = smooth_temporal(fitted, alpha=0.5)

    def bone_len_std(seq):
        from pose3d.core.skeleton import BONES
        stds = []
        for a, b in BONES:
            d = np.linalg.norm(seq[:, int(b)] - seq[:, int(a)], axis=1)
            stds.append(np.nanstd(d))
        return np.mean(stds)

    print("\n=== [C] Bone-length fit + smoothing (5px noise) ===")
    print(f"  raw bone-length jitter (std):    {bone_len_std(raw_seq)*10:.1f} mm")
    print(f"  fitted bone-length jitter (std): {bone_len_std(fitted)*10:.1f} mm")
    raw_3d_err = np.nanmean([np.linalg.norm(r - g, axis=1)
                             for r, g in zip(raw_seq, gts)]) * 10
    fit_3d_err = np.nanmean([np.linalg.norm(r - g, axis=1)
                             for r, g in zip(fitted, gts)]) * 10
    print(f"  raw 3D error:    {raw_3d_err:.1f} mm")
    print(f"  fitted 3D error: {fit_3d_err:.1f} mm")

    # summary JSON for the report
    summary = {
        "cameras": [CAM_L, CAM_R],
        "baseline_cm": float(baseline),
        "frames": int(len(gts)),
        "ideal_mean_mm": float(np.nanmean(errs) * 10),
        "ideal_max_mm": float(np.nanmax(errs) * 10),
    }
    out = DATA / "geometry_validation_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

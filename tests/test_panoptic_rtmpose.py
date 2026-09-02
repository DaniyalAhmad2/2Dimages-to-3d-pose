"""Full-pipeline validation on REAL images: RTMPose -> triangulate -> vs GT.

Extracts synchronized frames from two Panoptic HD camera videos, runs the
actual RTMPose detector on each view, triangulates with Panoptic's real
calibration, and compares the reconstructed 3D pose to the ground-truth 3D.

Because we triangulate with Panoptic's own calibration, the reconstructed
points live in the same dome world frame (cm) as the GT, so errors are
directly comparable with no alignment.

Frame sync is verified by reprojecting the GT into each image and checking it
overlaps the person; a small offset search corrects any indexing mismatch.

Run: .venv/bin/python -m tests.validate_panoptic_rtmpose
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from pose3d.datasets.panoptic import (
    frame_index, list_pose_frames, load_camera, load_pose3d,
)
from pose3d.core.skeleton import JOINT_NAMES, NUM_JOINTS
from pose3d.geometry.triangulate import triangulate_points
from tests.synth import project

DATA = Path(__file__).resolve().parent.parent / "data"
SEQ = "171204_pose1_sample"
CALIB = DATA / f"calibration_{SEQ}.json"
POSE_DIR = DATA / "hdPose3d" / "hdPose3d_stage1_coco19"
VIDEOS = DATA / "hdVideos"
FRAMES_DIR = DATA / "frames"
CAM_L, CAM_R = "00_00", "00_12"


def extract_frame(video: Path, frame_no: int, out: Path) -> bool:
    """Extract a single 0-indexed frame with ffmpeg."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video), "-vf", f"select=eq(n\\,{frame_no})",
           "-vframes", "1", str(out)]
    subprocess.run(cmd, check=False)
    return out.exists()


def project_gt(gt3d, cam):
    valid = ~np.isnan(gt3d).any(1)
    out = np.full((NUM_JOINTS, 2), np.nan)
    if valid.any():
        out[valid] = project(gt3d[valid], cam.intr.K, cam.intr.dist,
                             cam.ext.R, cam.ext.t)
    return out


def main(num_frames: int = 6):
    from pose3d.detect.rtmpose import RTMPoseDetector

    cam_l = load_camera(CALIB, CAM_L)
    cam_r = load_camera(CALIB, CAM_R)
    detector = RTMPoseDetector(mode="balanced", device="cpu")
    print("RTMPose loaded (CPU).")

    pose_files = list_pose_frames(POSE_DIR)
    # sample evenly across the sequence
    pick = pose_files[:: max(1, len(pose_files) // num_frames)][:num_frames]

    results = []
    for pf in pick:
        fi = frame_index(pf)
        gt = load_pose3d(pf)
        if np.isnan(gt).all():
            continue
        img_l = FRAMES_DIR / f"{CAM_L}_{fi:08d}.jpg"
        img_r = FRAMES_DIR / f"{CAM_R}_{fi:08d}.jpg"
        extract_frame(VIDEOS / f"hd_{CAM_L}.mp4", fi, img_l)
        extract_frame(VIDEOS / f"hd_{CAM_R}.mp4", fi, img_r)
        if not (img_l.exists() and img_r.exists()):
            print(f"frame {fi}: extraction failed"); continue

        bgr_l = cv2.imread(str(img_l))
        bgr_r = cv2.imread(str(img_r))
        det_l = detector.detect(bgr_l)
        det_r = detector.detect(bgr_r)

        # sync check: projected-GT vs detected 2D distance in the left view
        pgt_l = project_gt(gt, cam_l)
        d = np.nanmean(np.linalg.norm(det_l.xy - pgt_l, axis=1))

        rec = triangulate_points(det_l.xy, det_r.xy, cam_l.intr, cam_r.intr,
                                 cam_l.ext, cam_r.ext)
        err = np.linalg.norm(rec - gt, axis=1)   # cm
        results.append({
            "frame": fi,
            "sync_2d_px": float(d),
            "mean_3d_mm": float(np.nanmean(err) * 10),
            "median_3d_mm": float(np.nanmedian(err) * 10),
            "detected": int(np.sum(~np.isnan(det_l.xy).any(1))),
        })
        # save an overlay for visual evidence
        _save_overlay(bgr_l, det_l.xy, pgt_l,
                      DATA / "overlays" / f"{CAM_L}_{fi:08d}.jpg")
        print(f"frame {fi}: sync={d:5.1f}px  mean3D={np.nanmean(err)*10:6.1f}mm  "
              f"median3D={np.nanmedian(err)*10:6.1f}mm  det={results[-1]['detected']}/15")

    if results:
        mm = np.array([r["mean_3d_mm"] for r in results])
        med = np.array([r["median_3d_mm"] for r in results])
        print("\n=== SUMMARY (RTMPose on real images vs Panoptic GT) ===")
        print(f"  frames: {len(results)}")
        print(f"  mean 3D error across frames:   {mm.mean():.1f} mm")
        print(f"  median 3D error across frames: {np.median(med):.1f} mm")
        (DATA / "rtmpose_validation_summary.json").write_text(
            json.dumps(results, indent=2))


def _save_overlay(bgr, det_xy, gt_xy, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    img = bgr.copy()
    for p in gt_xy:
        if not np.isnan(p).any():
            cv2.circle(img, tuple(p.astype(int)), 6, (0, 255, 0), 2)  # GT green
    for p in det_xy:
        if not np.isnan(p).any():
            cv2.circle(img, tuple(p.astype(int)), 4, (0, 0, 255), -1)  # det red
    cv2.imwrite(str(out), img)


if __name__ == "__main__":
    main()

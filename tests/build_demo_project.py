"""Build a real demo project from Panoptic + render a dashboard screenshot.

Extracts frames from two Panoptic HD cameras, runs the full pipeline
(RTMPose detect -> triangulate with real calibration -> bone-fit), saves a
loadable project, then renders the dashboard to a PNG (offscreen).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import cv2
import numpy as np

from pose3d.core.io_project import save_project
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
from pose3d.datasets.panoptic import (
    frame_index, list_pose_frames, load_camera, load_pose3d,
)
from pose3d.detect.rtmpose import RTMPoseDetector
from pose3d.pipeline import CalibratedRig, fit_project, triangulate_project

DATA = Path(__file__).resolve().parent.parent / "data"
SEQ = "171204_pose1_sample"
CALIB = DATA / f"calibration_{SEQ}.json"
POSE_DIR = DATA / "hdPose3d" / "hdPose3d_stage1_coco19"
VIDEOS = DATA / "hdVideos"
CAM_L, CAM_R = "00_00", "00_12"
PROJ = DATA / "demo_project"
N = 10


def extract(video, frame_no, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return
    cv2.setNumThreads(0)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
           "-vf", f"select=eq(n\\,{frame_no})", "-vframes", "1", str(out)]
    import subprocess
    subprocess.run(cmd, check=False)


def main():
    cam_l = load_camera(CALIB, CAM_L)
    cam_r = load_camera(CALIB, CAM_R)
    rig = CalibratedRig(cam_l.intr, cam_r.intr, cam_l.ext, cam_r.ext)
    detector = RTMPoseDetector(mode="balanced", device="cpu")

    pose_files = list_pose_frames(POSE_DIR)
    pick = pose_files[:: max(1, len(pose_files) // N)][:N]

    project = ProjectData(name="Panoptic_171204_pose1", fps=30,
                          calibration_ref="calibration")
    img_dir = PROJ / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for pf in pick:
        fi = frame_index(pf)
        il = img_dir / f"left_{fi:04d}.jpg"
        ir = img_dir / f"right_{fi:04d}.jpg"
        extract(VIDEOS / f"hd_{CAM_L}.mp4", fi, il)
        extract(VIDEOS / f"hd_{CAM_R}.mp4", fi, ir)
        bl, br = cv2.imread(str(il)), cv2.imread(str(ir))
        if bl is None or br is None:
            continue
        dl, dr = detector.detect(bl), detector.detect(br)
        f = Frame(frame_id=f"{fi:04d}",
                  images={CAM_LEFT: str(il), CAM_RIGHT: str(ir)})
        f.kp2d[CAM_LEFT], f.scores[CAM_LEFT] = dl.xy, dl.scores
        f.kp2d[CAM_RIGHT], f.scores[CAM_RIGHT] = dr.xy, dr.scores
        project.frames.append(f)
        print(f"frame {fi}: detected L={np.sum(~np.isnan(dl.xy).any(1))}/15")

    triangulate_project(project, rig)
    fit_project(project, smooth=True)
    save_project(project, PROJ)
    print(f"Saved demo project ({len(project.frames)} frames) to {PROJ}")

    # export the animation to BVH/FBX/mp4 from the fitted poses
    from pose3d.export.blender_export import export_animation
    poses = np.stack([f.fitted3d for f in project.frames]) / 100.0  # cm->m
    res = export_animation(poses, PROJ / "export", name="demo", fps=30,
                           render_video=True, timeout=300)
    print(f"Export ok={res.ok} bvh={bool(res.bvh)} fbx={bool(res.fbx)} "
          f"mp4={bool(res.mp4)}")

    _screenshot(project, rig)


def _screenshot(project, rig):
    from PySide6.QtWidgets import QApplication
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    from pose3d.app import load_stylesheet

    app = QApplication.instance() or QApplication([])
    load_stylesheet(app)
    model = ProjectModel(project, rig)
    win = MainWindow(model)
    win.resize(1500, 900)
    win.show()
    app.processEvents()
    for _ in range(5):
        app.processEvents()
    out = DATA / "dashboard_screenshot.png"
    win.grab().save(str(out))
    print(f"Saved screenshot to {out}")


if __name__ == "__main__":
    main()

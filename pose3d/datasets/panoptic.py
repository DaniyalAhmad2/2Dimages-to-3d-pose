"""CMU Panoptic loader: real dome calibration + 3D pose ground truth.

Used to validate the two-camera geometry pipeline against real camera
calibration (real intrinsics + distortion + extrinsics) and real human 3D
motion. Panoptic 3D points are in centimetres in the dome world frame; camera
model is x = K (R X + t) with OpenCV-order distCoef [k1,k2,p1,p2,k3].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.skeleton import NUM_JOINTS, Joint

# Panoptic COCO19 index -> canonical Joint
COCO19_TO_CANONICAL: dict[int, Joint] = {
    1: Joint.HEAD,            # Nose
    0: Joint.NECK,            # Neck
    3: Joint.LEFT_SHOULDER,
    9: Joint.RIGHT_SHOULDER,
    4: Joint.LEFT_ELBOW,
    10: Joint.RIGHT_ELBOW,
    5: Joint.LEFT_WRIST,
    11: Joint.RIGHT_WRIST,
    2: Joint.PELVIS,          # BodyCenter (mid-hip)
    6: Joint.LEFT_HIP,
    12: Joint.RIGHT_HIP,
    7: Joint.LEFT_KNEE,
    13: Joint.RIGHT_KNEE,
    8: Joint.LEFT_ANKLE,
    14: Joint.RIGHT_ANKLE,
}


@dataclass
class PanopticCamera:
    name: str
    intr: Intrinsics
    ext: Extrinsics


def load_camera(calib_path: str | Path, cam_name: str) -> PanopticCamera:
    """Load one camera (e.g. '00_00') as Intrinsics + Extrinsics."""
    data = json.loads(Path(calib_path).read_text(encoding="utf-8"))
    for c in data["cameras"]:
        if c["name"] == cam_name:
            K = np.array(c["K"], dtype=float)
            dist = np.array(c["distCoef"], dtype=float).reshape(1, -1)
            R = np.array(c["R"], dtype=float)
            t = np.array(c["t"], dtype=float).reshape(3)
            w, h = c["resolution"]
            return PanopticCamera(
                name=cam_name,
                intr=Intrinsics(K=K, dist=dist, image_size=(w, h)),
                ext=Extrinsics(R=R, t=t))
    raise KeyError(f"camera {cam_name} not found in {calib_path}")


def load_pose3d(json_path: str | Path, body_index: int = 0) -> np.ndarray:
    """Return canonical (NUM_JOINTS, 3) GT in cm, NaN where a joint is absent."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out = np.full((NUM_JOINTS, 3), np.nan, dtype=float)
    if not data.get("bodies"):
        return out
    body = data["bodies"][body_index]
    joints = np.array(body["joints19"], dtype=float).reshape(-1, 4)  # x,y,z,conf
    for c19, canon in COCO19_TO_CANONICAL.items():
        if c19 < len(joints):
            out[int(canon)] = joints[c19, :3]
    return out


def list_pose_frames(pose_dir: str | Path) -> list[Path]:
    return sorted(Path(pose_dir).glob("body3DScene_*.json"))


def frame_index(pose_json: Path) -> int:
    """Extract the HD frame number from a pose filename."""
    return int(pose_json.stem.split("_")[-1])

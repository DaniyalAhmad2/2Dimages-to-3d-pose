"""Canonical skeleton definition and COCO-17 -> canonical mapping.

This is the single source of truth for joint identity in the whole app.
Detectors emit COCO-17; we map to a canonical joint set that ADDS the
derived joints the UI/mockup needs (neck, pelvis) and uses nose as the
single head point (per client: one head point, not a 5-point face mesh).

COCO-17 has no neck / pelvis / head-top, so those are derived by the
standard midpoint convention (same as OpenPose BODY_25 neck/mid-hip):
    head   = nose
    neck   = midpoint(left_shoulder, right_shoulder)
    pelvis = midpoint(left_hip, right_hip)
Derived-joint confidence = min(parent confidences) (conservative).
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


# --- COCO-17 keypoint order (rtmlib / RTMPose default output) ---------------
COCO17_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO17_INDEX: dict[str, int] = {n: i for i, n in enumerate(COCO17_NAMES)}


class Joint(IntEnum):
    """Canonical joint set used everywhere downstream of detection.

    Feet (indices 15, 16) are appended so existing joint indices are unchanged.
    """
    HEAD = 0          # nose (COCO) or native head (Halpe26)
    NECK = 1          # derived midpoint(shoulders) (COCO) or native (Halpe26)
    LEFT_SHOULDER = 2
    RIGHT_SHOULDER = 3
    LEFT_ELBOW = 4
    RIGHT_ELBOW = 5
    LEFT_WRIST = 6
    RIGHT_WRIST = 7
    PELVIS = 8        # derived midpoint(hips) (COCO) or native (Halpe26)
    LEFT_HIP = 9
    RIGHT_HIP = 10
    LEFT_KNEE = 11
    RIGHT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_ANKLE = 14
    LEFT_FOOT = 15    # Halpe26 left big toe (NaN if the model has no feet)
    RIGHT_FOOT = 16   # Halpe26 right big toe


NUM_JOINTS = len(Joint)
JOINT_NAMES: list[str] = [j.name for j in Joint]

# Bones as (parent, child) canonical-joint pairs. Parent is the joint the
# child rotates about; used for bone-length fit and for FK export driving.
BONES: list[tuple[Joint, Joint]] = [
    (Joint.PELVIS, Joint.NECK),
    (Joint.NECK, Joint.HEAD),
    (Joint.NECK, Joint.LEFT_SHOULDER),
    (Joint.NECK, Joint.RIGHT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.PELVIS, Joint.LEFT_HIP),
    (Joint.PELVIS, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.LEFT_ANKLE, Joint.LEFT_FOOT),
    (Joint.RIGHT_ANKLE, Joint.RIGHT_FOOT),
]

# Direct COCO-17 index for each canonical joint that maps 1:1 (derived = None).
_DIRECT_FROM_COCO: dict[Joint, int] = {
    Joint.HEAD: COCO17_INDEX["nose"],
    Joint.LEFT_SHOULDER: COCO17_INDEX["left_shoulder"],
    Joint.RIGHT_SHOULDER: COCO17_INDEX["right_shoulder"],
    Joint.LEFT_ELBOW: COCO17_INDEX["left_elbow"],
    Joint.RIGHT_ELBOW: COCO17_INDEX["right_elbow"],
    Joint.LEFT_WRIST: COCO17_INDEX["left_wrist"],
    Joint.RIGHT_WRIST: COCO17_INDEX["right_wrist"],
    Joint.LEFT_HIP: COCO17_INDEX["left_hip"],
    Joint.RIGHT_HIP: COCO17_INDEX["right_hip"],
    Joint.LEFT_KNEE: COCO17_INDEX["left_knee"],
    Joint.RIGHT_KNEE: COCO17_INDEX["right_knee"],
    Joint.LEFT_ANKLE: COCO17_INDEX["left_ankle"],
    Joint.RIGHT_ANKLE: COCO17_INDEX["right_ankle"],
}

# Mixamo bone name per canonical joint (the joint that is the *head* of the
# Mixamo bone). Used by the Blender export mapping. HEAD/PELVIS map to the
# rig root/head bones; confirm exact naming against the target rig (client Q5).
MIXAMO_BONE: dict[Joint, str] = {
    Joint.PELVIS: "mixamorig:Hips",
    Joint.NECK: "mixamorig:Neck",
    Joint.HEAD: "mixamorig:Head",
    Joint.LEFT_SHOULDER: "mixamorig:LeftArm",
    Joint.RIGHT_SHOULDER: "mixamorig:RightArm",
    Joint.LEFT_ELBOW: "mixamorig:LeftForeArm",
    Joint.RIGHT_ELBOW: "mixamorig:RightForeArm",
    Joint.LEFT_WRIST: "mixamorig:LeftHand",
    Joint.RIGHT_WRIST: "mixamorig:RightHand",
    Joint.LEFT_HIP: "mixamorig:LeftUpLeg",
    Joint.RIGHT_HIP: "mixamorig:RightUpLeg",
    Joint.LEFT_KNEE: "mixamorig:LeftLeg",
    Joint.RIGHT_KNEE: "mixamorig:RightLeg",
    Joint.LEFT_ANKLE: "mixamorig:LeftFoot",
    Joint.RIGHT_ANKLE: "mixamorig:RightFoot",
    Joint.LEFT_FOOT: "mixamorig:LeftToeBase",
    Joint.RIGHT_FOOT: "mixamorig:RightToeBase",
}


def derive_joints(
    coco_xy: np.ndarray, coco_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map COCO-17 keypoints to the canonical joint set.

    Parameters
    ----------
    coco_xy : (17, 2) float array of pixel coords in COCO-17 order.
    coco_scores : (17,) float array of per-keypoint confidence in [0, 1].

    Returns
    -------
    xy : (NUM_JOINTS, 2) float array in canonical Joint order.
    scores : (NUM_JOINTS,) float array; derived joints use min(parents).
    """
    coco_xy = np.asarray(coco_xy, dtype=float).reshape(17, 2)
    coco_scores = np.asarray(coco_scores, dtype=float).reshape(17)

    # NaN-init so unmapped joints (feet — COCO-17 has none) are excluded from
    # triangulation rather than triangulated at (0, 0).
    xy = np.full((NUM_JOINTS, 2), np.nan, dtype=float)
    scores = np.zeros(NUM_JOINTS, dtype=float)

    for joint, coco_idx in _DIRECT_FROM_COCO.items():
        xy[joint] = coco_xy[coco_idx]
        scores[joint] = coco_scores[coco_idx]

    ls, rs = COCO17_INDEX["left_shoulder"], COCO17_INDEX["right_shoulder"]
    lh, rh = COCO17_INDEX["left_hip"], COCO17_INDEX["right_hip"]

    xy[Joint.NECK] = (coco_xy[ls] + coco_xy[rs]) / 2.0
    scores[Joint.NECK] = min(coco_scores[ls], coco_scores[rs])

    xy[Joint.PELVIS] = (coco_xy[lh] + coco_xy[rh]) / 2.0
    scores[Joint.PELVIS] = min(coco_scores[lh], coco_scores[rh])

    return xy, scores


# --- Halpe-26 (RTMPose BodyWithFeet) -> canonical --------------------------
# Halpe26 order: 0-16 = COCO17, 17 head, 18 neck, 19 hip(pelvis),
# 20 L big toe, 21 R big toe, 22 L small toe, 23 R small toe, 24 L heel, 25 R heel
HALPE26_TO_CANONICAL: dict[int, Joint] = {
    17: Joint.HEAD,           # native head
    18: Joint.NECK,           # native neck
    5: Joint.LEFT_SHOULDER,
    6: Joint.RIGHT_SHOULDER,
    7: Joint.LEFT_ELBOW,
    8: Joint.RIGHT_ELBOW,
    9: Joint.LEFT_WRIST,
    10: Joint.RIGHT_WRIST,
    19: Joint.PELVIS,         # native mid-hip
    11: Joint.LEFT_HIP,
    12: Joint.RIGHT_HIP,
    13: Joint.LEFT_KNEE,
    14: Joint.RIGHT_KNEE,
    15: Joint.LEFT_ANKLE,
    16: Joint.RIGHT_ANKLE,
    20: Joint.LEFT_FOOT,      # left big toe
    21: Joint.RIGHT_FOOT,     # right big toe
}


def map_halpe26(kp: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map Halpe-26 keypoints to the canonical joint set (incl. feet).

    kp : (26, 2), scores : (26,). All canonical joints are direct (no
    derivation) since Halpe26 provides native neck/pelvis/head + feet.
    """
    kp = np.asarray(kp, dtype=float).reshape(26, 2)
    scores = np.asarray(scores, dtype=float).reshape(26)
    xy = np.full((NUM_JOINTS, 2), np.nan, dtype=float)
    sc = np.zeros(NUM_JOINTS, dtype=float)
    for h_idx, joint in HALPE26_TO_CANONICAL.items():
        xy[int(joint)] = kp[h_idx]
        sc[int(joint)] = scores[h_idx]
    return xy, sc


# --- RAG (red/amber/green) confidence banding ------------------------------
RAG_GREEN_MIN = 0.60
RAG_AMBER_MIN = 0.35


def rag_status(score: float) -> str:
    """Return 'green' | 'amber' | 'red' for a per-joint confidence.

    Bands (tune on real data): green >= 0.60, amber 0.35-0.60, red < 0.35.
    """
    if score >= RAG_GREEN_MIN:
        return "green"
    if score >= RAG_AMBER_MIN:
        return "amber"
    return "red"

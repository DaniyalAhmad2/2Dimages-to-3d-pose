"""Canonical skeleton definition and detector -> canonical mappings.

This is the single source of truth for joint identity in the whole app. Two
detector layouts reach it, and each has its own mapping:

* Halpe-26 (`map_halpe26`) is what the app detects with today
  (`detect.rtmpose.USE_HALPE26`). It carries a native head point — the skull
  vertex — which is taken as `Joint.HEAD`, while NECK and PELVIS stay 2D
  midpoints; see `map_halpe26` for why the three are not decided together.
* COCO-17 (`derive_joints`) is the other layout the detector can run, and what
  every project imported before that switch was flipped was detected with. It
  has no neck / pelvis / head-top at all, so those are derived by the standard
  midpoint convention (same as OpenPose BODY_25 neck/mid-hip):
      head   = nose
      neck   = midpoint(left_shoulder, right_shoulder)
      pelvis = midpoint(left_hip, right_hip)

Derived-joint confidence = min(parent confidences) (conservative).

Which of the two produced a project decides how its HEAD may be used
downstream: a skull-vertex HEAD sits on the head's axis, a nose does not.
That is recorded per project as `head_source` (see `HEAD_SOURCE`) and read by
`pose3d.geometry.character`, so the two conventions can never be corrected
for twice.
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

# The face keypoints, kept ALONGSIDE the canonical joints rather than inside
# them. One nose cannot carry head orientation — the reconstructed nose
# direction moved only ~20 deg across a take in which the head visibly turned
# far more — but two ears plus a nose give a full 3-axis head basis. The ears
# are detected at least as reliably as the nose (0.91/0.93 vs 0.71 on the
# client's mannequin). Indices 0-4 are identical in COCO-17 and Halpe-26, so
# one extractor serves both models.
HEAD_KP_NAMES: list[str] = ["nose", "left_eye", "right_eye",
                            "left_ear", "right_ear"]
NUM_HEAD_KP = len(HEAD_KP_NAMES)
HEAD_KP_INDEX: dict[str, int] = {n: i for i, n in enumerate(HEAD_KP_NAMES)}


def extract_head(kp: np.ndarray, scores: np.ndarray):
    """(NUM_HEAD_KP, 2), (NUM_HEAD_KP,) face keypoints from a raw model output.

    `kp`/`scores` are the detector's own layout (COCO-17 or Halpe-26); only
    indices 0-4 are read, which mean the same thing in both.
    """
    kp = np.asarray(kp, dtype=float)
    scores = np.asarray(scores, dtype=float)
    idx = [COCO17_INDEX[n] for n in HEAD_KP_NAMES]
    return kp[idx, :2].copy(), scores[idx].copy()


class Joint(IntEnum):
    """Canonical joint set used everywhere downstream of detection.

    COCO-17 based; feet are intentionally excluded (they are unreliable when
    the subject's feet are near/outside the frame).
    """
    HEAD = 0          # Halpe-26: the skull vertex; COCO-17: the nose
    NECK = 1          # derived: midpoint(shoulders)
    LEFT_SHOULDER = 2
    RIGHT_SHOULDER = 3
    LEFT_ELBOW = 4
    RIGHT_ELBOW = 5
    LEFT_WRIST = 6
    RIGHT_WRIST = 7
    PELVIS = 8        # derived: midpoint(hips)
    LEFT_HIP = 9
    RIGHT_HIP = 10
    LEFT_KNEE = 11
    RIGHT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_ANKLE = 14


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
}


def derive_joints(
    coco_xy: np.ndarray, coco_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map COCO-17 keypoints to the canonical joint set.

    COCO-17 has no head-top, neck or pelvis, so HEAD is the nose and the other
    two are midpoints. `map_halpe26` is the layout the app detects through
    today (see `detect.rtmpose.USE_HALPE26`); this one still runs every project
    imported before that switch was flipped, and is the one-line rollback.

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
# Halpe26 order: 0-16 = COCO17, 17 head (skull vertex), 18 neck, 19 hip
# (pelvis), 20 L big toe, 21 R big toe, 22 L small toe, 23 R small toe,
# 24 L heel, 25 R heel.
HALPE26_NAMES: list[str] = COCO17_NAMES + [
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]
HALPE26_INDEX: dict[str, int] = {n: i for i, n in enumerate(HALPE26_NAMES)}
NUM_HALPE26 = len(HALPE26_NAMES)

# The joints Halpe-26 and COCO-17 agree about, index for index. HEAD, NECK and
# PELVIS are deliberately absent: those three are the policy (see map_halpe26).
_DIRECT_FROM_HALPE26: dict[Joint, int] = {
    joint: idx for joint, idx in _DIRECT_FROM_COCO.items()
    if joint is not Joint.HEAD
}

# Which Halpe index each policy takes for the three joints that have a choice.
# "native" is the model's own point; "derived" is the 2D midpoint convention
# `derive_joints` uses; "nose" is COCO's head point.
_HALPE26_POLICY_INDEX: dict[Joint, dict[str, int]] = {
    Joint.HEAD: {"native": HALPE26_INDEX["head"],   # the skull vertex
                 "nose": HALPE26_INDEX["nose"]},
    Joint.NECK: {"native": HALPE26_INDEX["neck"]},
    Joint.PELVIS: {"native": HALPE26_INDEX["hip"]},
}
# Parents of the two joints the midpoint convention derives.
_HALPE26_DERIVED_FROM: dict[Joint, tuple[int, int]] = {
    Joint.NECK: (HALPE26_INDEX["left_shoulder"], HALPE26_INDEX["right_shoulder"]),
    Joint.PELVIS: (HALPE26_INDEX["left_hip"], HALPE26_INDEX["right_hip"]),
}

# What a given HEAD policy means anatomically — the value persisted in
# project.json as `head_source`, which is what tells the retarget whether the
# canonical HEAD is a point on the skull axis or the nose (see
# pose3d.geometry.character).
HEAD_SOURCE: dict[str, str] = {"native": "skull", "nose": "nose"}

#: The shipped policy, and `map_halpe26`'s defaults (bound from here, so this
#: dict is the single place the app's choice lives). Reverting HEAD to the
#: COCO convention — the rollback for this whole change — is one edit here.
HALPE26_POLICY: dict[str, str] = {
    "head": "native", "neck": "derived", "pelvis": "derived"}

#: What `head_source` a project detected under HALPE26_POLICY gets.
HALPE26_HEAD_SOURCE: str = HEAD_SOURCE[HALPE26_POLICY["head"]]


def map_halpe26(kp: np.ndarray, scores: np.ndarray,
                head: str = HALPE26_POLICY["head"],
                neck: str = HALPE26_POLICY["neck"],
                pelvis: str = HALPE26_POLICY["pelvis"],
                ) -> tuple[np.ndarray, np.ndarray]:
    """Map Halpe-26 keypoints to the canonical joint set, under a POLICY.

    The three joints Halpe detects natively are the only ones worth arguing
    about, and the argument does not come out the same way for all three, so
    each is a separate knob rather than one "use Halpe" flag:

    * ``head="native"`` takes Halpe's own head point (the skull vertex).
      Measured on the client take that is the single largest available win:
      HEAD retarget error 12.65 -> 3.93 % of body height with no face
      keypoints, and the neck-head bone-length CV on the rigid mannequin
      9.53 -> 3.85 %, which was the worst bone in the whole baseline.
      ``head="nose"`` is the COCO-17 convention and the one-line rollback.
    * ``neck="derived"``/``pelvis="derived"`` keep the per-view 2D midpoints,
      because Halpe's native neck is NOT the shoulder midpoint: taking it
      regresses the neck-Lshoulder bone CV 5.15 -> 8.13 %. The midpoints cost
      essentially nothing geometrically (triangulating the midpoint of the
      projections rather than projecting the 3D midpoint is 0.25 mm at NECK
      and 0.08 mm at PELVIS) and are better conditioned (derived NECK
      epipolar 1.51 px against its parents' 3.86).

    Feet (Halpe 20-25) are intentionally not mapped: the canonical joint set
    has no foot joints.

    Parameters
    ----------
    kp : (26, 2) float array of pixel coords in Halpe-26 order.
    scores : (26,) float array of per-keypoint confidence in [0, 1].
    head : "native" (skull vertex) or "nose".
    neck, pelvis : "derived" (2D midpoint of the parents) or "native".

    Returns
    -------
    xy : (NUM_JOINTS, 2) float array in canonical Joint order.
    scores : (NUM_JOINTS,) float array; derived joints use min(parents).
    """
    kp = np.asarray(kp, dtype=float).reshape(NUM_HALPE26, 2)
    scores = np.asarray(scores, dtype=float).reshape(NUM_HALPE26)
    policy = {Joint.HEAD: head, Joint.NECK: neck, Joint.PELVIS: pelvis}

    xy = np.full((NUM_JOINTS, 2), np.nan, dtype=float)
    sc = np.zeros(NUM_JOINTS, dtype=float)
    for joint, idx in _DIRECT_FROM_HALPE26.items():
        xy[int(joint)] = kp[idx]
        sc[int(joint)] = scores[idx]

    for joint, choice in policy.items():
        if choice == "derived":
            parents = _HALPE26_DERIVED_FROM.get(joint)
            if parents is None:
                raise ValueError(
                    f"{joint.name} has no midpoint convention to derive it "
                    f"from; use {sorted(_HALPE26_POLICY_INDEX[joint])}")
            a, b = parents
            xy[int(joint)] = (kp[a] + kp[b]) / 2.0
            sc[int(joint)] = min(scores[a], scores[b])
            continue
        idx = _HALPE26_POLICY_INDEX[joint].get(choice)
        if idx is None:
            allowed = sorted(set(_HALPE26_POLICY_INDEX[joint])
                             | ({"derived"} if joint in _HALPE26_DERIVED_FROM
                                else set()))
            raise ValueError(f"{joint.name} policy must be one of {allowed}, "
                             f"not {choice!r}")
        xy[int(joint)] = kp[idx]
        sc[int(joint)] = scores[idx]
    return xy, sc


# --- what each layout DERIVES, as opposed to detects ------------------------
#: The canonical parents of every joint the midpoint convention derives, in
#: CANONICAL indices — `map_halpe26` and `derive_joints` each hold the same
#: pairing in their own detector's indices, and this is the form anything
#: working on an already-mapped pose needs (the UI's manual correction, above
#: all: dragging a shoulder has to move the neck that IS its midpoint).
DERIVED_MIDPOINT_PARENTS: dict[Joint, tuple[Joint, Joint]] = {
    Joint.NECK: (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    Joint.PELVIS: (Joint.LEFT_HIP, Joint.RIGHT_HIP),
}

#: COCO-17 has no neck or pelvis keypoint at all, so both are always derived.
_COCO17_DERIVED: frozenset[Joint] = frozenset(DERIVED_MIDPOINT_PARENTS)


def derived_joints(keypoint_model: str | None) -> frozenset[Joint]:
    """Which canonical joints this layout DERIVES as a 2D midpoint.

    The question "is this joint a measurement or an arithmetic consequence of
    two others?" has one answer per layout and it must be asked HERE, of the
    policy, not inferred from the layout's name. Halpe-26 detects a neck and a
    hip natively, but `HALPE26_POLICY` does not take them (its native neck is
    not the shoulder midpoint and regresses the neck-Lshoulder bone CV 5.15 ->
    8.13 % — see `map_halpe26`), so under the shipped policy BOTH layouts
    derive NECK and PELVIS. Reading `keypoint_model != "coco17"` instead is
    what let the manual-correction path stop re-deriving them the moment the
    detector switch was flipped: a dragged shoulder left a 28 px stale NECK
    that was then triangulated and bone-fitted.

    An unknown or missing layout is treated as COCO-17, which is what every
    project written before `keypoint_model` existed was detected with.
    """
    if keypoint_model != "halpe26":
        return _COCO17_DERIVED
    return frozenset(
        joint for joint, key in ((Joint.HEAD, "head"), (Joint.NECK, "neck"),
                                 (Joint.PELVIS, "pelvis"))
        if HALPE26_POLICY[key] == "derived")


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

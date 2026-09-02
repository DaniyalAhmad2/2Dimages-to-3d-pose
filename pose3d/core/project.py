"""Pure-data project model (no Qt dependency).

Holds everything about a capture session: frames, per-view 2D keypoints +
confidence, the triangulated 3D pose, and the correction log. The Qt signal
hub (ui layer) wraps an instance of this and emits on mutation, so this stays
importable in headless tests and the geometry/detection pipeline.

Missing observations are represented as NaN so array shapes stay uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose3d.core.skeleton import NUM_HEAD_KP, NUM_JOINTS

# Canonical camera keys. Two-camera rig per the brief (left / right).
CAM_LEFT = "left"
CAM_RIGHT = "right"
CAMERAS = (CAM_LEFT, CAM_RIGHT)


def _nan_xy() -> np.ndarray:
    return np.full((NUM_JOINTS, 2), np.nan, dtype=float)


def _nan_scores() -> np.ndarray:
    return np.full((NUM_JOINTS,), np.nan, dtype=float)


def _nan_xyz() -> np.ndarray:
    return np.full((NUM_JOINTS, 3), np.nan, dtype=float)


def _nan_head_xy() -> np.ndarray:
    return np.full((NUM_HEAD_KP, 2), np.nan, dtype=float)


def _nan_head_scores() -> np.ndarray:
    return np.full((NUM_HEAD_KP,), np.nan, dtype=float)


def _nan_head_xyz() -> np.ndarray:
    return np.full((NUM_HEAD_KP, 3), np.nan, dtype=float)


@dataclass
class Frame:
    """One matched pair of images and all derived pose data for it."""
    frame_id: str
    images: dict[str, str] = field(default_factory=dict)          # cam -> path
    kp2d: dict[str, np.ndarray] = field(
        default_factory=lambda: {c: _nan_xy() for c in CAMERAS})
    scores: dict[str, np.ndarray] = field(
        default_factory=lambda: {c: _nan_scores() for c in CAMERAS})
    pose3d: np.ndarray = field(default_factory=_nan_xyz)          # (NUM_JOINTS,3)
    fitted3d: np.ndarray = field(default_factory=_nan_xyz)        # bone-fit result
    # per-(cam,joint) flag: True if the point was hand-corrected by the user.
    corrected: dict[str, np.ndarray] = field(
        default_factory=lambda: {c: np.zeros(NUM_JOINTS, bool) for c in CAMERAS})
    # Face keypoints (nose/eyes/ears), kept PARALLEL to the canonical arrays so
    # every `reshape(NUM_JOINTS, ...)` downstream stays true. They orient the
    # head and nothing else: no bones, no bone-length fit, not hand-editable.
    head2d: dict[str, np.ndarray] = field(
        default_factory=lambda: {c: _nan_head_xy() for c in CAMERAS})
    head_scores: dict[str, np.ndarray] = field(
        default_factory=lambda: {c: _nan_head_scores() for c in CAMERAS})
    head3d: np.ndarray = field(default_factory=_nan_head_xyz)

    def set_kp(self, cam: str, joint: int, x: float, y: float,
               score: float = 1.0, corrected: bool = False) -> None:
        self.kp2d[cam][joint] = (x, y)
        self.scores[cam][joint] = score
        self.corrected[cam][joint] = corrected

    def set_head_kp(self, cam: str, k: int, x: float, y: float,
                    score: float = 1.0) -> None:
        """Face keypoint k (nose/eyes/ears). No corrected flag: face points
        are never auto-dropped, so there is nothing to protect them from."""
        self.head2d[cam][k] = (x, y)
        self.head_scores[cam][k] = score


@dataclass
class Correction:
    """A single reversible 2D edit (also persisted to SQLite for v2)."""
    frame_id: str
    cam: str
    joint: int
    old_xy: tuple[float, float]
    new_xy: tuple[float, float]
    ts: str  # ISO timestamp (caller-provided; core stays clock-free)


@dataclass
class ProjectData:
    """Root project object: metadata + ordered frames + calibration ref."""
    name: str = "Untitled"
    fps: int = 30
    frames: list[Frame] = field(default_factory=list)
    # Calibration is stored as a separate file set; we keep only the folder
    # name/reference here. Loaded lazily by the geometry layer.
    calibration_ref: str | None = None
    corrections: list[Correction] = field(default_factory=list)
    # What the numbers in this project mean, recorded at import time because
    # nothing downstream can recover them: the ArUco tag edge the extrinsics
    # were scaled by (metres), and which detector produced the 2D keypoints.
    # None on a project imported before they were recorded.
    marker_length: float | None = None
    keypoint_model: str | None = None

    def frame_ids(self) -> list[str]:
        return [f.frame_id for f in self.frames]

    def get_frame(self, frame_id: str) -> Frame | None:
        for f in self.frames:
            if f.frame_id == frame_id:
                return f
        return None

"""Swappable keypoint-detector interface.

The rest of the pipeline (triangulation, fit, export) depends only on this
interface, so the detection method can change (markerless model today, red-dot
marker detector later, a fine-tuned model in v2) without touching downstream
layers. Detectors emit results in the canonical Joint order already, with a
RAG status per joint.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from pose3d.core.skeleton import NUM_HEAD_KP, NUM_JOINTS, rag_status


@dataclass
class Detection:
    """Per-view detection result in canonical Joint order."""
    xy: np.ndarray       # (NUM_JOINTS, 2) pixel coords (NaN if not found)
    scores: np.ndarray   # (NUM_JOINTS,) confidence in [0, 1]
    # Face keypoints (nose/eyes/ears), carried alongside rather than inside the
    # canonical set so NUM_JOINTS stays 15. None from detectors that have no
    # notion of a face (e.g. the manual one). See skeleton.extract_head.
    head_xy: np.ndarray | None = None       # (NUM_HEAD_KP, 2)
    head_scores: np.ndarray | None = None   # (NUM_HEAD_KP,)

    def rag(self) -> list[str]:
        return [rag_status(float(s)) if not np.isnan(s) else "red"
                for s in self.scores]

    def __post_init__(self):
        assert self.xy.shape == (NUM_JOINTS, 2), self.xy.shape
        assert self.scores.shape == (NUM_JOINTS,), self.scores.shape
        if self.head_xy is not None:
            assert self.head_xy.shape == (NUM_HEAD_KP, 2), self.head_xy.shape
            assert self.head_scores.shape == (NUM_HEAD_KP,), \
                self.head_scores.shape


class KeypointDetector(ABC):
    """Detect canonical-order 2D keypoints in a single image."""

    #: What this detector's canonical HEAD point IS: "nose" (the COCO-17
    #: convention, and what the manual detector marks) or "skull" (a point on
    #: the head's axis, e.g. Halpe-26's own head keypoint). It is part of the
    #: interface rather than a detail of one detector because the retarget
    #: corrects a nose HEAD for its ~45 deg forward offset and must NOT correct
    #: a skull one, so the project has to record which it holds
    #: (`ProjectData.head_source`, `pose3d.geometry.character`). "nose" is the
    #: safe default: it is what every detector emitted before the key existed.
    head_source: str = "nose"

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> Detection:
        """Return a Detection for the primary subject in the image."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

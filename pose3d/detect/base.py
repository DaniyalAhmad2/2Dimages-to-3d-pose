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

from pose3d.core.skeleton import NUM_JOINTS, rag_status


@dataclass
class Detection:
    """Per-view detection result in canonical Joint order."""
    xy: np.ndarray       # (NUM_JOINTS, 2) pixel coords (NaN if not found)
    scores: np.ndarray   # (NUM_JOINTS,) confidence in [0, 1]

    def rag(self) -> list[str]:
        return [rag_status(float(s)) if not np.isnan(s) else "red"
                for s in self.scores]

    def __post_init__(self):
        assert self.xy.shape == (NUM_JOINTS, 2), self.xy.shape
        assert self.scores.shape == (NUM_JOINTS,), self.scores.shape


class KeypointDetector(ABC):
    """Detect canonical-order 2D keypoints in a single image."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> Detection:
        """Return a Detection for the primary subject in the image."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__

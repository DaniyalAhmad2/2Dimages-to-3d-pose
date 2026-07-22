"""Manual 'detector': returns user-supplied keypoints via the same interface.

Lets the correction workflow (and tests) feed known/edited keypoints through
the identical KeypointDetector contract that RTMPose uses, so downstream code
never special-cases the source of a keypoint.
"""
from __future__ import annotations

import numpy as np

from pose3d.core.skeleton import NUM_JOINTS
from pose3d.detect.base import Detection, KeypointDetector


class ManualDetector(KeypointDetector):
    def __init__(self, xy: np.ndarray, scores: np.ndarray | None = None):
        xy = np.asarray(xy, float).reshape(NUM_JOINTS, 2)
        if scores is None:
            scores = np.where(np.isnan(xy).any(1), 0.0, 1.0)
        self._det = Detection(xy=xy, scores=np.asarray(scores, float))

    def detect(self, image_bgr: np.ndarray) -> Detection:
        return self._det

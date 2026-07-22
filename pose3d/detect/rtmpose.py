"""Markerless human pose detection via rtmlib (RTMPose, ONNX).

Pure onnxruntime; no mmpose/mmcv/torch. Uses the Halpe-26 model
(BodyWithFeet), which detects FEET (big/small toe, heel) plus native
neck/pelvis/head — mapped to the canonical joint set. Falls back to the
COCO-17 Body model (no feet) if requested.

Picks the highest-confidence person when several are detected.

CAVEAT (client-known): RTMPose is trained on real people and is unreliable on
the grey mannequin test rig and on extreme inverted poses. That is exactly why
the manual-correction path is first-class. Detector is swappable via base.py.
"""
from __future__ import annotations

import numpy as np

from pose3d.core.skeleton import NUM_JOINTS, derive_joints, map_halpe26
from pose3d.detect.base import Detection, KeypointDetector


class RTMPoseDetector(KeypointDetector):
    def __init__(self, mode: str = "balanced", device: str = "cpu",
                 backend: str = "onnxruntime", feet: bool = True):
        self.feet = feet
        if feet:
            from rtmlib import BodyWithFeet
            self._model = BodyWithFeet(mode=mode, backend=backend, device=device)
            self._map = map_halpe26          # 26 keypoints (Halpe26)
        else:
            from rtmlib import Body
            self._model = Body(mode=mode, backend=backend, device=device)
            self._map = derive_joints        # 17 keypoints (COCO-17)
        self.mode = mode
        self.device = device

    def detect(self, image_bgr: np.ndarray) -> Detection:
        keypoints, scores = self._model(image_bgr)   # (N,K,2), (N,K)
        keypoints = np.asarray(keypoints, float)
        scores = np.asarray(scores, float)
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return _empty_detection()
        best = int(np.argmax(scores.mean(axis=1)))   # most confident person
        cxy, cscore = self._map(keypoints[best], scores[best])
        return Detection(xy=cxy, scores=cscore)


def _empty_detection() -> Detection:
    return Detection(
        xy=np.full((NUM_JOINTS, 2), np.nan),
        scores=np.zeros(NUM_JOINTS))

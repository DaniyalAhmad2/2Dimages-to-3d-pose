"""Markerless human pose detection via rtmlib (RTMPose, ONNX).

Pure onnxruntime; no mmpose/mmcv/torch. Emits COCO-17, which we map to the
canonical joint set (deriving neck/pelvis/head). Picks the highest-confidence
person when several are detected.

CAVEAT (client-known): RTMPose is trained on real people and is unreliable on
the grey mannequin test rig and on extreme inverted poses. That is exactly why
the manual-correction path is first-class. Detector is swappable via base.py.
"""
from __future__ import annotations

import numpy as np

from pose3d.core.skeleton import derive_joints
from pose3d.detect.base import Detection, KeypointDetector


class RTMPoseDetector(KeypointDetector):
    def __init__(self, mode: str = "balanced", device: str = "cpu",
                 backend: str = "onnxruntime"):
        # imported lazily so the rest of the app imports without rtmlib present
        from rtmlib import Body
        self._body = Body(mode=mode, backend=backend, device=device)
        self.mode = mode
        self.device = device

    def detect(self, image_bgr: np.ndarray) -> Detection:
        keypoints, scores = self._body(image_bgr)   # (N,17,2), (N,17)
        keypoints = np.asarray(keypoints, float)
        scores = np.asarray(scores, float)
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return _empty_detection()
        # pick the most confident person
        best = int(np.argmax(scores.mean(axis=1)))
        cxy, cscore = derive_joints(keypoints[best], scores[best])
        return Detection(xy=cxy, scores=cscore)


def _empty_detection() -> Detection:
    from pose3d.core.skeleton import NUM_JOINTS
    return Detection(
        xy=np.full((NUM_JOINTS, 2), np.nan),
        scores=np.zeros(NUM_JOINTS))

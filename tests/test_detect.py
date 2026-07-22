"""Phase 3 verification: detector interface + RTMPose smoke (skips if absent)."""
import importlib.util

import numpy as np
import pytest

from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.detect.base import Detection
from pose3d.detect.manual import ManualDetector


def test_detection_rag():
    xy = np.zeros((NUM_JOINTS, 2))
    scores = np.full(NUM_JOINTS, 0.9)
    scores[Joint.LEFT_WRIST] = 0.4    # amber
    scores[Joint.RIGHT_ANKLE] = 0.1   # red
    d = Detection(xy=xy, scores=scores)
    rag = d.rag()
    assert rag[Joint.HEAD] == "green"
    assert rag[Joint.LEFT_WRIST] == "amber"
    assert rag[Joint.RIGHT_ANKLE] == "red"


def test_manual_detector_roundtrips():
    xy = np.arange(NUM_JOINTS * 2, dtype=float).reshape(NUM_JOINTS, 2)
    det = ManualDetector(xy)
    out = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert np.allclose(out.xy, xy)
    assert out.scores.shape == (NUM_JOINTS,)


def test_manual_detector_nan_scores_zero():
    xy = np.full((NUM_JOINTS, 2), np.nan)
    xy[0] = (1.0, 2.0)
    det = ManualDetector(xy)
    out = det.detect(np.zeros((10, 10, 3), np.uint8))
    assert out.scores[0] == 1.0
    assert out.scores[1] == 0.0


@pytest.mark.skipif(importlib.util.find_spec("rtmlib") is None,
                    reason="rtmlib not installed")
def test_rtmpose_shapes():
    """Smoke: RTMPose returns canonical-shaped output on a blank image."""
    from pose3d.detect.rtmpose import RTMPoseDetector
    det = RTMPoseDetector(mode="lightweight", device="cpu")
    out = det.detect(np.zeros((256, 192, 3), np.uint8))
    assert out.xy.shape == (NUM_JOINTS, 2)
    assert out.scores.shape == (NUM_JOINTS,)


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None,
                    reason="onnxruntime not installed")
def test_onnxruntime_cpu_provider():
    import onnxruntime as ort
    assert "CPUExecutionProvider" in ort.get_available_providers()

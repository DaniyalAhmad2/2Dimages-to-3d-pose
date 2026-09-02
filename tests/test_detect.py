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


# --- head-only re-detection (the migration path for a headless project) ---

class _HeadDetector:
    """A detector whose body keypoints disagree with everything already
    stored, so anything it is allowed to touch is obvious."""

    def __init__(self):
        from pose3d.core.skeleton import NUM_HEAD_KP
        self.xy = np.full((NUM_JOINTS, 2), 999.0)
        self.scores = np.full(NUM_JOINTS, 0.11)
        self.head_xy = np.arange(NUM_HEAD_KP * 2, dtype=float).reshape(-1, 2)
        self.head_scores = np.full(NUM_HEAD_KP, 0.8)

    def detect(self, image_bgr):
        return Detection(xy=self.xy.copy(), scores=self.scores.copy(),
                         head_xy=self.head_xy.copy(),
                         head_scores=self.head_scores.copy())


def _one_frame_project():
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
    p = ProjectData(name="head")
    f = Frame(frame_id="0000",
              images={CAM_LEFT: "l.png", CAM_RIGHT: "r.png"})
    for cam in (CAM_LEFT, CAM_RIGHT):
        f.kp2d[cam] = np.arange(NUM_JOINTS * 2, dtype=float).reshape(-1, 2)
        f.scores[cam] = np.full(NUM_JOINTS, 0.5)
    f.corrected[CAM_LEFT][int(Joint.HEAD)] = True
    p.frames.append(f)
    return p


def test_detect_head_only_touches_nothing_else():
    """"Re-detect face points only" is how a project made before the head
    feature existed gets a head — it must not disturb the body pose it was
    hand-corrected into."""
    from pose3d.core.project import CAM_LEFT, CAMERAS
    from pose3d.pipeline import detect_project

    p = _one_frame_project()
    f = p.frames[0]
    body_before = {c: f.kp2d[c].copy() for c in CAMERAS}
    scores_before = {c: f.scores[c].copy() for c in CAMERAS}

    detect_project(p, _HeadDetector(), lambda path: np.zeros((4, 4, 3), np.uint8),
                   fields="head")

    for c in CAMERAS:
        assert np.array_equal(f.kp2d[c], body_before[c])
        assert np.array_equal(f.scores[c], scores_before[c])
        assert not np.isnan(f.head2d[c]).any()          # ... and it DID write
        assert not np.isnan(f.head_scores[c]).any()
    assert f.corrected[CAM_LEFT][int(Joint.HEAD)]


def test_detect_rejects_an_unknown_field_set():
    from pose3d.pipeline import detect_project
    with pytest.raises(ValueError):
        detect_project(_one_frame_project(), _HeadDetector(),
                       lambda path: None, fields="body")

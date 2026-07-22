"""Phase 1 verification: canonical skeleton mapping + RAG banding."""
import numpy as np

from pose3d.core.skeleton import (
    COCO17_INDEX, NUM_JOINTS, BONES, Joint, derive_joints, rag_status,
)


def _fake_coco():
    # deterministic distinct coords per keypoint: idx -> (idx*10, idx*10+1)
    xy = np.array([[i * 10.0, i * 10.0 + 1.0] for i in range(17)])
    scores = np.linspace(0.1, 0.95, 17)
    return xy, scores


def test_derive_head_is_nose():
    xy, sc = _fake_coco()
    cxy, csc = derive_joints(xy, sc)
    assert np.allclose(cxy[Joint.HEAD], xy[COCO17_INDEX["nose"]])
    assert csc[Joint.HEAD] == sc[COCO17_INDEX["nose"]]


def test_derive_neck_is_shoulder_midpoint():
    xy, sc = _fake_coco()
    cxy, csc = derive_joints(xy, sc)
    ls, rs = COCO17_INDEX["left_shoulder"], COCO17_INDEX["right_shoulder"]
    assert np.allclose(cxy[Joint.NECK], (xy[ls] + xy[rs]) / 2.0)
    # derived confidence = min of parents
    assert csc[Joint.NECK] == min(sc[ls], sc[rs])


def test_derive_pelvis_is_hip_midpoint():
    xy, sc = _fake_coco()
    cxy, csc = derive_joints(xy, sc)
    lh, rh = COCO17_INDEX["left_hip"], COCO17_INDEX["right_hip"]
    assert np.allclose(cxy[Joint.PELVIS], (xy[lh] + xy[rh]) / 2.0)
    assert csc[Joint.PELVIS] == min(sc[lh], sc[rh])


def test_direct_joints_copied_exactly():
    xy, sc = _fake_coco()
    cxy, csc = derive_joints(xy, sc)
    assert np.allclose(cxy[Joint.LEFT_WRIST], xy[COCO17_INDEX["left_wrist"]])
    assert np.allclose(cxy[Joint.RIGHT_ANKLE], xy[COCO17_INDEX["right_ankle"]])


def test_output_shapes():
    xy, sc = _fake_coco()
    cxy, csc = derive_joints(xy, sc)
    assert cxy.shape == (NUM_JOINTS, 2)
    assert csc.shape == (NUM_JOINTS,)


def test_bones_reference_valid_joints():
    for a, b in BONES:
        assert 0 <= int(a) < NUM_JOINTS and 0 <= int(b) < NUM_JOINTS
    # a tree over NUM_JOINTS joints has NUM_JOINTS-1 edges
    assert len(BONES) == NUM_JOINTS - 1


def test_rag_bands():
    assert rag_status(0.9) == "green"
    assert rag_status(0.60) == "green"
    assert rag_status(0.5) == "amber"
    assert rag_status(0.35) == "amber"
    assert rag_status(0.2) == "red"

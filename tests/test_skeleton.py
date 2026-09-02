"""Canonical skeleton mapping (COCO-17 and Halpe-26) + RAG banding."""
import numpy as np
import pytest

from pose3d.core.skeleton import (
    BONES, COCO17_INDEX, DERIVED_MIDPOINT_PARENTS, HALPE26_HEAD_SOURCE,
    HALPE26_INDEX, HALPE26_POLICY, HEAD_SOURCE, NUM_JOINTS, Joint,
    derive_joints, derived_joints, map_halpe26, rag_status,
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


# --- Halpe-26 mapping, and the policy that decides three of its joints -----

def _fake_halpe():
    """Deterministic distinct coords per keypoint: idx -> (idx*10, idx*10+1)."""
    xy = np.array([[i * 10.0, i * 10.0 + 1.0] for i in range(26)])
    scores = np.linspace(0.1, 0.95, 26)
    return xy, scores


def test_map_halpe26():
    """The shipped policy: skull-vertex HEAD, derived NECK/PELVIS, no feet."""
    xy, sc = _fake_halpe()
    cxy, csc = map_halpe26(xy, sc)

    assert cxy.shape == (NUM_JOINTS, 2)
    assert csc.shape == (NUM_JOINTS,)

    # HEAD is Halpe's own head point (index 17, the skull vertex), NOT the nose
    assert np.allclose(cxy[Joint.HEAD], xy[HALPE26_INDEX["head"]])
    assert csc[Joint.HEAD] == sc[HALPE26_INDEX["head"]]
    assert not np.allclose(cxy[Joint.HEAD], xy[HALPE26_INDEX["nose"]])

    # NECK/PELVIS stay the 2D midpoints, with min-of-parents confidence
    ls, rs = HALPE26_INDEX["left_shoulder"], HALPE26_INDEX["right_shoulder"]
    lh, rh = HALPE26_INDEX["left_hip"], HALPE26_INDEX["right_hip"]
    assert np.allclose(cxy[Joint.NECK], (xy[ls] + xy[rs]) / 2.0)
    assert csc[Joint.NECK] == min(sc[ls], sc[rs])
    assert np.allclose(cxy[Joint.PELVIS], (xy[lh] + xy[rh]) / 2.0)
    assert csc[Joint.PELVIS] == min(sc[lh], sc[rh])

    # every other joint is copied index for index, from the same indices
    # COCO-17 uses — this is what makes the two layouts comparable at all
    for name, joint in (("left_wrist", Joint.LEFT_WRIST),
                        ("right_ankle", Joint.RIGHT_ANKLE),
                        ("left_shoulder", Joint.LEFT_SHOULDER),
                        ("right_hip", Joint.RIGHT_HIP)):
        assert np.allclose(cxy[joint], xy[COCO17_INDEX[name]])
        assert csc[joint] == sc[COCO17_INDEX[name]]

    # the six foot keypoints have nowhere to go in the canonical set, and
    # must not have leaked into some joint
    for foot in ("left_big_toe", "right_big_toe", "left_small_toe",
                 "right_small_toe", "left_heel", "right_heel"):
        assert not (cxy == xy[HALPE26_INDEX[foot]]).all(axis=1).any()


def test_head_neck_pelvis_policy():
    """The three joints Halpe detects natively are three separate decisions.

    HEAD native is the measured win (12.65 -> 3.93 % of body height); native
    NECK is the measured regression (neck-Lshoulder bone CV 5.15 -> 8.13 %),
    which is why one flag for all three would be wrong.
    """
    xy, sc = _fake_halpe()

    # the shipped policy is what the defaults are bound to, so a rollback in
    # HALPE26_POLICY moves the defaults with it
    assert HALPE26_POLICY == {"head": "native", "neck": "derived",
                              "pelvis": "derived"}
    assert np.allclose(map_halpe26(xy, sc)[0],
                       map_halpe26(xy, sc, **HALPE26_POLICY)[0], equal_nan=True)

    # HEAD: native = the skull vertex, nose = the COCO convention (rollback)
    assert np.allclose(map_halpe26(xy, sc, head="native")[0][Joint.HEAD],
                       xy[HALPE26_INDEX["head"]])
    nose = map_halpe26(xy, sc, head="nose")
    assert np.allclose(nose[0][Joint.HEAD], xy[HALPE26_INDEX["nose"]])
    assert nose[1][Joint.HEAD] == sc[HALPE26_INDEX["nose"]]
    # with head="nose" the mapping agrees with COCO-17's, joint for joint
    assert np.allclose(nose[0], derive_joints(xy[:17], sc[:17])[0],
                       equal_nan=True)

    # NECK/PELVIS: native is available but not shipped
    native = map_halpe26(xy, sc, neck="native", pelvis="native")[0]
    assert np.allclose(native[Joint.NECK], xy[HALPE26_INDEX["neck"]])
    assert np.allclose(native[Joint.PELVIS], xy[HALPE26_INDEX["hip"]])
    assert not np.allclose(native[Joint.NECK],
                           map_halpe26(xy, sc)[0][Joint.NECK])

    # HEAD has no midpoint convention, and a typo must not silently pick one
    for kwargs in ({"head": "derived"}, {"head": "middle"},
                   {"neck": "midpoint"}, {"pelvis": "hip"}):
        with pytest.raises(ValueError):
            map_halpe26(xy, sc, **kwargs)


def test_head_source_names_what_the_head_point_is():
    """`head_source` is the fact the retarget reads; it must follow the policy
    rather than being set independently of it."""
    assert HEAD_SOURCE["native"] == "skull"
    assert HEAD_SOURCE["nose"] == "nose"
    assert HALPE26_HEAD_SOURCE == HEAD_SOURCE[HALPE26_POLICY["head"]]


def test_which_joints_a_layout_derives_is_the_policy_not_the_name():
    """`derived_joints` is the one answer to "is this joint arithmetic?".

    Both shipped layouts derive NECK and PELVIS as the shoulder/hip midpoints
    — Halpe-26 detects a neck and a hip natively but `HALPE26_POLICY` does not
    take them — so anything that decides from the layout's NAME gets the
    shipped configuration wrong. `ui.model._sync_derived` did, and stopped
    maintaining the midpoints in the manual-correction path the moment the
    detector switch was flipped.
    """
    coco = derived_joints("coco17")
    assert coco == {Joint.NECK, Joint.PELVIS} == set(DERIVED_MIDPOINT_PARENTS)
    assert derived_joints("halpe26") == coco         # under HALPE26_POLICY
    # an unrecorded layout is the legacy one, which is what it was
    assert derived_joints(None) == coco
    assert derived_joints("") == coco

    # and it FOLLOWS the policy: taking Halpe's native neck drops it out
    saved = dict(HALPE26_POLICY)
    try:
        HALPE26_POLICY["neck"] = "native"
        assert derived_joints("halpe26") == {Joint.PELVIS}
        assert derived_joints("coco17") == coco      # COCO has no such choice
    finally:
        HALPE26_POLICY.clear()
        HALPE26_POLICY.update(saved)

    # the parents are canonical joints, and the pairing is the midpoint the
    # two mappings build in their own detectors' indices
    assert DERIVED_MIDPOINT_PARENTS[Joint.NECK] == (Joint.LEFT_SHOULDER,
                                                    Joint.RIGHT_SHOULDER)
    assert DERIVED_MIDPOINT_PARENTS[Joint.PELVIS] == (Joint.LEFT_HIP,
                                                      Joint.RIGHT_HIP)
    xy, sc = _fake_halpe()
    mapped = map_halpe26(xy, sc)[0]
    for joint, (a, b) in DERIVED_MIDPOINT_PARENTS.items():
        assert np.allclose(mapped[int(joint)],
                           0.5 * (mapped[int(a)] + mapped[int(b)]))

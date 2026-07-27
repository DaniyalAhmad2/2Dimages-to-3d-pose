"""The bundled character skins to a pose (live LBS) and stays aligned to it."""
import numpy as np
import pytest

from pose3d.config import character_blend

pytest.importorskip("numpy")

_HAVE = character_blend() is not None
from pose3d.core.skeleton import Joint
from tests.synth import sample_skeleton_3d


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_character_skins_to_pose():
    from pose3d.geometry.character import Character
    pose = sample_skeleton_3d()
    valid = ~np.isnan(pose).any(1)
    verts, faces = Character().pose(pose, valid)
    assert verts is not None and len(verts) > 500
    assert not np.isnan(verts).any()
    # character occupies roughly the pose's vertical span, at its position
    assert abs(verts[:, 2].min() - pose[:, 2].min()) < 0.3 * (pose[:, 2].max() - pose[:, 2].min())
    assert abs(verts[:, 2].max() - pose[:, 2].max()) < 0.4 * (pose[:, 2].max() - pose[:, 2].min())


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_character_raises_correct_side():
    """Raising the LEFT wrist must lift the character's LEFT-side vertices."""
    from pose3d.geometry.character import Character
    ch = Character()
    base = sample_skeleton_3d()
    valid = ~np.isnan(base).any(1)
    v0, _ = ch.pose(base, valid)
    raised = base.copy()
    raised[int(Joint.LEFT_ELBOW)] = [-0.30, 0, 1.45]
    raised[int(Joint.LEFT_WRIST)] = [-0.30, 0, 1.75]
    v1, _ = ch.pose(raised, valid)
    # left-side verts (x<0, matching LEFT_SHOULDER at -x) should rise on average
    left = v0[:, 0] < -0.05
    assert (v1[left, 2].mean() - v0[left, 2].mean()) > 0.05


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_character_handles_sparse_pose():
    from pose3d.geometry.character import Character
    pose = sample_skeleton_3d()
    pose[[Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE, Joint.LEFT_WRIST]] = np.nan
    verts, faces = Character().pose(pose, ~np.isnan(pose).any(1))
    assert verts is not None and not np.isnan(verts).any()


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_bones_stay_connected_when_posed():
    """Bones joined in the rest rig must stay joined once posed.

    Regression guard: bones used to be placed ABSOLUTELY between their two
    keypoints, ignoring the parent, so e.g. the spine drifted ~0.9 units off the
    hips and the mesh weighted across that joint was torn open at the waist.
    """
    from pose3d.geometry.character import Character
    ch = Character()
    base = sample_skeleton_3d()
    piv = base[int(Joint.PELVIS)].copy()

    def rotX(deg):
        a = np.radians(deg)
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    for lean in (0, 15, 35):                      # upright through a strong lean
        pose = (base - piv) @ rotX(lean).T + piv
        valid = ~np.isnan(pose).any(1)
        skin, *_ = ch._skin_matrices(pose, valid)
        for b in range(len(ch.parent)):
            p = ch.parent[b]
            if p < 0 or np.linalg.norm(ch.head[b] - ch.tail[p]) > 1e-6:
                continue                          # not joined in the rest rig
            child_head = skin[b][:3, :3] @ ch.head[b] + skin[b][:3, 3]
            parent_tail = skin[p][:3, :3] @ ch.tail[p] + skin[p][:3, 3]
            gap = float(np.linalg.norm(child_head - parent_tail))
            assert gap < 1e-6, (
                f"{ch.bone_names[b]} detached from {ch.bone_names[p]} "
                f"by {gap:.3f} at lean={lean}deg")


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_pose_bone_matrices_reproduce_lbs():
    """The bone matrices sent to Blender must reproduce the live-view skinning
    exactly (setting pose_bone.matrix = M drives the identical deform), so the
    export matches the 3D preview pose-for-pose."""
    from pose3d.geometry.character import Character
    ch = Character()
    pose = sample_skeleton_3d(); valid = ~np.isnan(pose).any(1)
    verts, _ = ch.pose(pose, valid)              # live-view mesh (up_pose space)

    # rebuild the mesh from the exported bone matrices via Blender's deform
    # relation: skin[b] = M_posed[b] @ rest[b]^-1
    bm = ch.pose_bone_matrices(pose, valid)
    skin = np.stack([np.asarray(bm[n]) @ np.linalg.inv(ch.rest[i])
                     for i, n in enumerate(ch.bone_names)])
    out = np.zeros((len(ch.verts0), 3))
    for k in range(ch.w_idx.shape[1]):
        M = skin[ch.w_idx[:, k]]
        out += ch.w_val[:, k][:, None] * np.einsum("vij,vj->vi", M, ch.vh)[:, :3]
    # pose() maps rig space back to up_pose space; do the same to compare
    _, pelvis, scale, Rz = ch._skin_matrices(pose, valid)
    out = pelvis + (Rz.T @ (out - ch.hips_world).T).T / scale
    assert np.abs(out - verts).max() < 1e-4

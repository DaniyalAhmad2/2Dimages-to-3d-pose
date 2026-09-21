"""The bundled character skins to a pose (live LBS) and stays aligned to it."""
import numpy as np
import pytest

from tests.gates import needs_character
from pose3d.core.skeleton import Joint
from tests.synth import sample_skeleton_3d


@needs_character()
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


@needs_character()
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


@needs_character()
def test_character_handles_sparse_pose():
    from pose3d.geometry.character import Character
    pose = sample_skeleton_3d()
    pose[[Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE, Joint.LEFT_WRIST]] = np.nan
    verts, faces = Character().pose(pose, ~np.isnan(pose).any(1))
    assert verts is not None and not np.isnan(verts).any()


@needs_character()
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


@needs_character()
def test_the_foot_aims_at_the_toe_when_it_is_seen():
    """One point per foot: the posed foot bone points from the ankle toward
    the captured big toe (pitch and yaw measured; roll from the leg's bend
    plane), rotation only, rest length kept."""
    from pose3d.geometry.character import Character
    ch = Character()
    pose = sample_skeleton_3d()
    ch.fit_to_subject(pose[None])
    valid = ~np.isnan(pose).any(1)
    j = ch.posed_joints(pose, valid)
    for ankle, toe in ((Joint.LEFT_ANKLE, Joint.LEFT_TOE),
                       (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE)):
        want = pose[toe] - j[ankle]                 # from the posed ankle
        got = j[toe] - j[ankle]
        cos = np.dot(want, got) / (np.linalg.norm(want) * np.linalg.norm(got))
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 1.0, (ankle, toe)
    # the foot keeps the rig's own length: the toe is a bone tail, not the
    # point. `rest_joints` is RIG space and `posed_joints` the caller's, and
    # `_from_rig` DIVIDES by the fitted scale, so the rest length comes back
    # over it (`_scale` is rig units per pose unit, ~8.5 on this rig).
    rest = ch.rest_joints()
    for ankle, toe in ((Joint.LEFT_ANKLE, Joint.LEFT_TOE),
                       (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE)):
        assert np.linalg.norm(j[toe] - j[ankle]) == pytest.approx(
            np.linalg.norm(rest[toe] - rest[ankle]) / ch._scale, rel=1e-6)


@needs_character()
def test_a_missing_toe_leaves_the_foot_exactly_as_before():
    """No toe: the foot rides the shin's matrix verbatim, which is what every
    take without feet got before the toes existed."""
    from pose3d.geometry.character import Character
    ch = Character()
    pose = sample_skeleton_3d()
    pose[[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = np.nan
    valid = ~np.isnan(pose).any(1)
    skin = ch._skin_matrices(pose, valid)[0]
    for foot, shin in (("foot.L", "shin.L"), ("foot.R", "shin.R")):
        assert np.array_equal(skin[ch.role[foot]], skin[ch.role[shin]]), \
            "an unseen toe leaves the foot on the shin's matrix bit for bit"
    j = ch.posed_joints(pose, valid)
    assert np.isfinite(j[Joint.LEFT_TOE]).all(), "the toe is read off the foot's tail"


@needs_character()
def test_a_foot_edge_weighs_one_in_the_scale_fit():
    """An EXTREMITY edge weighs 1 in the uniform-scale fit, whatever its
    parent weighs.

    The legs weigh double there — feet not reaching the floor is the mismatch
    the eye picks up first — and an edge takes the larger of its two ends'
    weights, so the ankle→toe edge inherited the ankle's 2. A foot is not a
    leg: one point each, often cropped, and it must not pull the whole
    character's size around. Design section 3: "the two foot edges enter the
    least squares with weight 1 (not the ankles' 2)".
    """
    from pose3d.geometry.bonefit import measure_bone_lengths
    from pose3d.geometry.character import Character, _SCALE_WEIGHTS
    ch = Character()
    pose = sample_skeleton_3d()
    # feet far longer than the rig's, so the weight they carry is legible in
    # the fitted scale
    for ankle, toe in ((Joint.LEFT_ANKLE, Joint.LEFT_TOE),
                       (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE)):
        pose[toe] = pose[ankle] + [0.0, 0.45, 0.0]
    got = ch.fit_to_subject(pose[None])

    sub = measure_bone_lengths(pose[None])
    rig = ch.rig_bone_lengths()
    toes = (int(Joint.LEFT_TOE), int(Joint.RIGHT_TOE))

    def fitted(foot_w):
        """The same weighted least squares with the foot edges at `foot_w`."""
        num = den = 0.0
        for (a, b), rig_len in rig.items():
            s = sub.get((a, b))
            if not s or not np.isfinite(s):
                continue
            w = foot_w if b in toes else max(_SCALE_WEIGHTS.get(a, 1.0),
                                             _SCALE_WEIGHTS.get(b, 1.0))
            num += w * s * rig_len
            den += w * s * s
        return num / den

    assert got == pytest.approx(fitted(1.0), rel=1e-12)
    assert got != pytest.approx(fitted(2.0), rel=1e-9), \
        "the foot edge is still carrying the ankle's weight"


@needs_character()
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

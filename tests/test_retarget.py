"""The character mimics the capture without ever deforming.

These guard the inversion: the rig's bone lengths are inviolable and motion
transfers as rotation only, with two-bone IK putting the end effectors as close
to the captured joints as fixed-length limbs allow.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.config import character_blend
from pose3d.core.skeleton import BONES, Joint
from tests.synth import sample_skeleton_3d

_HAVE = character_blend() is not None
pytestmark = pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")


def _ch():
    from pose3d.geometry.character import Character
    return Character()


def _lean(pose, deg):
    a = np.radians(deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    piv = pose[int(Joint.PELVIS)].copy()
    return (pose - piv) @ R.T + piv


def _poses():
    """A spread of poses: upright, leaning, limbs moved, sparse."""
    base = sample_skeleton_3d()
    out = [base, _lean(base, 15), _lean(base, 35)]
    reach = base.copy()
    reach[int(Joint.LEFT_ELBOW)] = [-0.30, 0.10, 1.45]
    reach[int(Joint.LEFT_WRIST)] = [-0.30, 0.20, 1.75]
    out.append(reach)
    sparse = base.copy()
    sparse[[int(Joint.LEFT_ANKLE), int(Joint.RIGHT_WRIST)]] = np.nan
    out.append(sparse)
    return out


def _height(p):
    v = p[~np.isnan(p).any(1)]
    return float(v[:, 2].max() - v[:, 2].min())


# --- the headline guarantee ------------------------------------------------

def test_no_bone_is_ever_scaled():
    """Every skin matrix is a pure rotation and every bone keeps its length."""
    ch = _ch()
    for pose in _poses():
        valid = ~np.isnan(pose).any(1)
        skin, *_ = ch._skin_matrices(pose, valid)
        for b in range(len(skin)):
            R = skin[b][:3, :3]
            assert np.abs(R.T @ R - np.eye(3)).max() < 1e-9, ch.bone_names[b]
            assert abs(np.linalg.det(R) - 1.0) < 1e-9, ch.bone_names[b]
            rest_len = np.linalg.norm(ch.tail[b] - ch.head[b])
            if rest_len > 1e-9:
                posed = np.linalg.norm(R @ (ch.tail[b] - ch.head[b]))
                assert abs(posed - rest_len) < 1e-9, ch.bone_names[b]


def test_no_bone_left_at_rest():
    """Every bone is actually posed — none stays frozen at the origin.

    Regression guard: helper bones parented to an orphan used to keep an
    identity matrix and sit frozen in front of the body in every export.
    """
    ch = _ch()
    pose = _lean(sample_skeleton_3d(), 25)
    skin, *_ = ch._skin_matrices(pose, ~np.isnan(pose).any(1))
    for b in range(len(skin)):
        assert not np.allclose(skin[b], np.eye(4)), \
            f"{ch.bone_names[b]} was never posed"


# --- IK --------------------------------------------------------------------

def _subject_from_rig(ch):
    """A 'subject' whose proportions are exactly the rig's, so every target is
    reachable and the retarget should be near-exact."""
    return ch._from_rig(ch.rest_joints(), np.zeros(3), 1.0, np.eye(3))


def test_ik_reaches_target_when_in_range():
    ch = _ch()
    sub = _subject_from_rig(ch)
    valid = ~np.isnan(sub).any(1)
    ch.fit_to_subject(sub[None])
    got = ch.posed_joints(sub, valid)
    for j in (Joint.LEFT_WRIST, Joint.RIGHT_WRIST,
              Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE):
        assert np.linalg.norm(got[int(j)] - sub[int(j)]) < 1e-6, j.name


def test_posed_joints_match_captured_when_proportions_match():
    """With a matching build, every joint lands within 1% of body height."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    valid = ~np.isnan(sub).any(1)
    ch.fit_to_subject(sub[None])
    got = ch.posed_joints(sub, valid)
    h = _height(sub)
    for j in range(len(got)):
        if np.isnan(sub[j]).any() or np.isnan(got[j]).any():
            continue
        assert np.linalg.norm(got[j] - sub[j]) / h < 0.01, Joint(j).name


def test_ik_falls_short_gracefully_out_of_range():
    """An unreachable target leaves the limb straight, short by exactly the
    difference — the closest a fixed-length limb can get."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    pose = sub.copy()
    # fling the wrist far beyond arm's reach, straight out to the side
    sh = pose[int(Joint.LEFT_SHOULDER)]
    pose[int(Joint.LEFT_WRIST)] = sh + np.array([-10.0, 0.0, 0.0])
    pose[int(Joint.LEFT_ELBOW)] = sh + np.array([-1.0, 0.0, 0.2])
    valid = ~np.isnan(pose).any(1)
    got = ch.posed_joints(pose, valid)

    shoulder = got[int(Joint.LEFT_SHOULDER)]
    elbow = got[int(Joint.LEFT_ELBOW)]
    wrist = got[int(Joint.LEFT_WRIST)]
    ua = np.linalg.norm(elbow - shoulder)
    fa = np.linalg.norm(wrist - elbow)
    # fully extended: the two segments are collinear
    assert np.linalg.norm((wrist - shoulder)) == pytest.approx(ua + fa, rel=1e-6)
    # and it points at the target
    to_target = pose[int(Joint.LEFT_WRIST)] - shoulder
    to_wrist = wrist - shoulder
    cos = np.dot(to_target, to_wrist) / (np.linalg.norm(to_target) * np.linalg.norm(to_wrist))
    assert cos > 0.999


def test_ik_uses_captured_bend_plane():
    """Mirroring the captured knee mirrors the character's knee."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    hip = sub[int(Joint.LEFT_HIP)]
    ankle = sub[int(Joint.LEFT_ANKLE)]
    axis = (ankle - hip) / np.linalg.norm(ankle - hip)

    def knee_side(offset):
        p = sub.copy()
        mid = (hip + ankle) / 2.0
        p[int(Joint.LEFT_KNEE)] = mid + offset
        got = ch.posed_joints(p, ~np.isnan(p).any(1))
        k = got[int(Joint.LEFT_KNEE)] - hip
        return k - np.dot(k, axis) * axis          # component off the axis

    off = np.array([0.0, 0.25, 0.0])
    fwd, back = knee_side(off), knee_side(-off)
    assert np.dot(fwd, off) > 0, "knee did not follow the captured bend"
    assert np.dot(back, off) < 0, "knee did not mirror with the capture"


def test_ik_pole_degenerate_falls_back_to_rest_bend():
    """A collinear or missing mid-joint must not produce NaN or a random flip."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    hip = sub[int(Joint.LEFT_HIP)]
    ankle = sub[int(Joint.LEFT_ANKLE)]

    collinear = sub.copy()
    collinear[int(Joint.LEFT_KNEE)] = (hip + ankle) / 2.0     # exactly on the axis
    got = ch.posed_joints(collinear, ~np.isnan(collinear).any(1))
    assert not np.isnan(got).any()

    missing = sub.copy()
    missing[int(Joint.LEFT_KNEE)] = np.nan
    got2 = ch.posed_joints(missing, ~np.isnan(missing).any(1))
    assert not np.isnan(got2[int(Joint.LEFT_ANKLE)]).any()


# --- stability and fallbacks ----------------------------------------------

def test_scale_is_constant_across_frames():
    """The character must not change size when joints drop out mid-take.

    Regression guard: the scale used to be recomputed per frame from whichever
    joints were visible, so the figure popped whenever the ankles vanished.
    """
    ch = _ch()
    base = sample_skeleton_3d()
    seq = np.stack([base, base, base])
    seq[1, [int(Joint.LEFT_ANKLE), int(Joint.RIGHT_ANKLE)]] = np.nan
    ch.fit_to_subject(seq)

    scales, heights = [], []
    for p in seq:
        valid = ~np.isnan(p).any(1)
        scales.append(ch._skin_matrices(p, valid)[2])
        verts, _, _ = ch.pose_and_joints(p, valid)
        heights.append(float(verts[:, 2].max() - verts[:, 2].min()))

    assert len(set(scales)) == 1, f"scale varied across frames: {scales}"
    # identical poses must give an identical figure
    assert heights[0] == heights[2]
    # the sparse frame's legs pose differently (no ankle to aim at), but the
    # FIGURE must not resize: per-frame scaling used to inflate it by ~37% here
    assert abs(heights[1] - heights[0]) / heights[0] < 0.01, heights


def test_pelvis_fallback_drives_torso():
    """A missing PELVIS must not freeze the torso.

    Regression guard: the pelvis was recovered from the hip midpoint, but the
    torso targets re-read Joint.PELVIS directly and came back empty, leaving
    the whole upper body at rest while the legs animated.
    """
    ch = _ch()
    pose = _lean(sample_skeleton_3d(), 30)
    pose[int(Joint.PELVIS)] = np.nan          # hips still present
    valid = ~np.isnan(pose).any(1)
    skin, *_ = ch._skin_matrices(pose, valid)
    for role in ("spine", "chest"):
        b = ch.role[role]
        assert not np.allclose(skin[b][:3, :3], np.eye(3)), \
            f"{role} stayed at rest with a missing pelvis"


def test_sparse_pose_still_skins():
    ch = _ch()
    pose = sample_skeleton_3d()
    pose[[int(Joint.LEFT_ANKLE), int(Joint.RIGHT_ANKLE),
          int(Joint.LEFT_WRIST)]] = np.nan
    verts, faces, joints = ch.pose_and_joints(pose, ~np.isnan(pose).any(1))
    assert verts is not None and not np.isnan(verts).any()
    assert joints is not None and not np.isnan(joints).any()


# --- the drawn skeleton ----------------------------------------------------

def test_posed_joints_sit_inside_the_mesh():
    """The overlay must never float outside the character."""
    ch = _ch()
    for pose in _poses():
        valid = ~np.isnan(pose).any(1)
        verts, _, joints = ch.pose_and_joints(pose, valid)
        lo, hi = verts.min(0), verts.max(0)
        span = float(np.linalg.norm(hi - lo))
        for j in range(len(joints)):
            if np.isnan(joints[j]).any():
                continue
            assert (joints[j] >= lo - 0.02 * span).all(), Joint(j).name
            assert (joints[j] <= hi + 0.02 * span).all(), Joint(j).name


# --- the rig itself --------------------------------------------------------

def test_bake_reproduces_the_shipped_asset(tmp_path):
    """tools/bake_character.py must rebuild the asset the app actually ships.

    The rig cannot be swapped safely unless the bake is known-good on a known
    input — otherwise a bake bug and a rig problem look identical. Vertex
    positions, bones and joints must come back exactly; the mesh is allowed a
    hair of slack because quad triangulation can legitimately split either way.
    """
    import subprocess
    from pose3d.config import blender_binary, character_blend
    from pose3d.geometry.character import Character

    blend = character_blend()
    exe = blender_binary()
    if not (blend and Path(exe).exists()):
        pytest.skip("Blender binary or character.blend not available")

    out = tmp_path / "baked.npz"
    script = Path(__file__).resolve().parent.parent / "tools" / "bake_character.py"
    r = subprocess.run([exe, "--background", blend, "--python", str(script),
                        "--", "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert out.exists(), f"bake produced nothing\n{r.stdout[-2000:]}"

    shipped = np.load(Path(blend).with_suffix(".npz"), allow_pickle=True)
    baked = np.load(out, allow_pickle=True)
    assert np.abs(shipped["verts"] - baked["verts"]).max() == 0
    assert np.abs(shipped["head"] - baked["head"]).max() == 0
    assert np.abs(shipped["tail"] - baked["tail"]).max() == 0
    assert (shipped["parent"] == baked["parent"]).all()
    assert [str(x) for x in shipped["bone_names"]] == [str(x) for x in baked["bone_names"]]

    pose = sample_skeleton_3d()
    valid = ~np.isnan(pose).any(1)
    va, _, ja = Character(Path(blend).with_suffix(".npz")).pose_and_joints(pose, valid)
    vb, _, jb = Character(out).pose_and_joints(pose, valid)
    h = float(va[:, 2].max() - va[:, 2].min())
    assert np.abs(va - vb).max() / h < 1e-3
    assert np.abs(ja - jb).max() / h < 1e-6


@pytest.mark.xfail(reason="the bundled rig is stylised (thigh:shank 0.675); "
                          "this passes once a human-proportioned rig is baked in",
                   strict=False)
def test_rig_proportions_are_human():
    """The rig must be shaped like a person, or fixed-length retargeting can
    never put the knees and elbows where the subject's are."""
    ch = _ch()
    rig = ch.rig_bone_lengths()
    thigh = rig[(int(Joint.LEFT_HIP), int(Joint.LEFT_KNEE))]
    shank = rig[(int(Joint.LEFT_KNEE), int(Joint.LEFT_ANKLE))]
    upper = rig[(int(Joint.LEFT_SHOULDER), int(Joint.LEFT_ELBOW))]
    fore = rig[(int(Joint.LEFT_ELBOW), int(Joint.LEFT_WRIST))]
    assert 0.90 <= thigh / shank <= 1.15, f"thigh:shank = {thigh / shank:.3f}"
    assert 1.15 <= upper / fore <= 1.40, f"upper_arm:forearm = {upper / fore:.3f}"

"""The character mimics the capture without ever deforming.

These guard the inversion: the rig's bone lengths are inviolable and motion
transfers as rotation only. Every bone aims at its own captured joint, so
joint DIRECTIONS track the capture exactly — dragging a knee moves the rig's
knee — while end effectors land within the fitted proportion mismatch. Two-bone
IK remains as the occlusion fallback, recovering a limb from its end effector
when the mid joint is missing.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.core.skeleton import BONES, Joint
from tests.gates import needs_character
from tests.synth import rot_about as _rot_about, sample_skeleton_3d

pytestmark = needs_character()


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


def test_end_effectors_land_close_when_in_range():
    """Direction-first retargeting still puts wrists and ankles essentially on
    the captured points for a matching build. (The old IK contract demanded
    1e-6 exactness; per-bone aiming instead accumulates the sub-0.2% seam
    between bone axes and joint read-back points, which is invisible and the
    price of the knee actually following the capture.)"""
    ch = _ch()
    sub = _subject_from_rig(ch)
    valid = ~np.isnan(sub).any(1)
    ch.fit_to_subject(sub[None])
    got = ch.posed_joints(sub, valid)
    h = _height(sub)
    for j in (Joint.LEFT_WRIST, Joint.RIGHT_WRIST,
              Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE):
        assert np.linalg.norm(got[int(j)] - sub[int(j)]) / h < 0.005, j.name


def test_posed_joints_match_captured_when_proportions_match():
    """With a matching build, every joint lands within 1% of body height.

    HEAD is exempted with a wider band: the read-back is the head-bone mid
    (a skull point), the capture is nose-convention, and the neck's bias
    clamp deliberately holds a skull-convention subject at rest — a ~1.2%
    offset by construction, not a defect.
    """
    ch = _ch()
    sub = _subject_from_rig(ch)
    valid = ~np.isnan(sub).any(1)
    ch.fit_to_subject(sub[None])
    got = ch.posed_joints(sub, valid)
    h = _height(sub)
    for j in range(len(got)):
        if np.isnan(sub[j]).any() or np.isnan(got[j]).any():
            continue
        band = 0.025 if j == int(Joint.HEAD) else 0.01
        assert np.linalg.norm(got[j] - sub[j]) / h < band, Joint(j).name


def test_arm_follows_the_captured_elbow_direction():
    """The defect this exists to prevent: the captured mid joint must set the
    limb's actual bend, not merely its bend plane. Under IK, dragging the
    elbow only rotated the bend plane and the arm stayed at whatever angle the
    shoulder->wrist distance implied."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    pose = sub.copy()
    sh = pose[int(Joint.LEFT_SHOULDER)]
    # a sharply bent arm: elbow out to the side, wrist back up near the shoulder
    pose[int(Joint.LEFT_ELBOW)] = sh + np.array([-0.25, 0.05, -0.10])
    pose[int(Joint.LEFT_WRIST)] = sh + np.array([-0.05, 0.15, 0.10])
    got = ch.posed_joints(pose, ~np.isnan(pose).any(1))

    want = pose[int(Joint.LEFT_ELBOW)] - sh
    have = got[int(Joint.LEFT_ELBOW)] - got[int(Joint.LEFT_SHOULDER)]
    cos = np.dot(want, have) / (np.linalg.norm(want) * np.linalg.norm(have))
    assert cos > 0.995, f"upper arm ignored the captured elbow (cos={cos:.4f})"


def test_ik_falls_short_gracefully_out_of_range():
    """With the MID JOINT OCCLUDED, IK recovers the limb from the end effector:
    an unreachable target leaves the limb straight, short by exactly the
    difference — the closest a fixed-length limb can get."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    pose = sub.copy()
    # fling the wrist far beyond arm's reach; the elbow is not captured, so
    # the chain must be recovered from the wrist alone (the IK path)
    sh = pose[int(Joint.LEFT_SHOULDER)]
    pose[int(Joint.LEFT_WRIST)] = sh + np.array([-10.0, 0.0, 0.0])
    pose[int(Joint.LEFT_ELBOW)] = np.nan
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

    # proportional to the limb, so the test means the same thing on any rig
    off = np.array([0.0, 0.25 * np.linalg.norm(ankle - hip), 0.0])
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


# --- the neck --------------------------------------------------------------

def _nose_convention(ch, sub):
    """Move the subject's HEAD point from the rig's skull convention to the
    nose convention real captures use: ~45 deg forward of the torso line."""
    from pose3d.geometry.character import _NOSE_PITCH
    p = sub.copy()
    neck = p[int(Joint.NECK)]
    torso = neck - p[int(Joint.PELVIS)]
    right = p[int(Joint.RIGHT_SHOULDER)] - p[int(Joint.LEFT_SHOULDER)]
    d = p[int(Joint.HEAD)] - neck
    cur = np.arccos(np.clip(np.dot(d / np.linalg.norm(d),
                                   torso / np.linalg.norm(torso)), -1, 1))
    # rotate whichever way about the shoulder axis actually lands the nose at
    # the anatomical angle — "forward" depends on which way the rig faces
    t_hat = torso / np.linalg.norm(torso)
    best = None
    for sign in (1.0, -1.0):
        cand = _rot_about(right, sign * np.degrees(_NOSE_PITCH - cur)) @ d
        th = np.arccos(np.clip(np.dot(cand / np.linalg.norm(cand), t_hat), -1, 1))
        if best is None or abs(th - _NOSE_PITCH) < best[0]:
            best = (abs(th - _NOSE_PITCH), cand)
    p[int(Joint.HEAD)] = neck + best[1]
    return p


def _neck_rotation_vs_chest(ch, pose):
    skin, *_ = ch._skin_matrices(pose, ~np.isnan(pose).any(1))
    Rn = skin[ch.role["neck"]][:3, :3]
    Rc = skin[ch.role["chest"]][:3, :3]
    rel = Rn @ Rc.T
    return np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))


def test_neck_is_at_rest_for_a_neutral_nose():
    """A nose held at the anatomical neutral must leave the neck at rest —
    the bias correction exists so real captures do not stare at the floor."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    assert _neck_rotation_vs_chest(ch, _nose_convention(ch, sub)) < 6.0


def test_nodding_the_nose_bends_the_neck():
    """Regression guard for the welded neck: neck/head were in no driving
    table, so the head inherited the chest verbatim and dragging the nose did
    nothing at all."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    neutral = _nose_convention(ch, sub)

    nod = neutral.copy()
    neck = nod[int(Joint.NECK)]
    right = nod[int(Joint.RIGHT_SHOULDER)] - nod[int(Joint.LEFT_SHOULDER)]
    torso = neck - nod[int(Joint.PELVIS)]
    t_hat = torso / np.linalg.norm(torso)
    d = nod[int(Joint.HEAD)] - neck
    # nod FORWARD: chin toward chest, i.e. the angle off the torso line grows.
    # Which rotation sign does that depends on the rig's facing, so pick it.
    cands = [_rot_about(right, s * 25.0) @ d for s in (1.0, -1.0)]
    nod[int(Joint.HEAD)] = neck + max(
        cands, key=lambda c: np.arccos(np.clip(
            np.dot(c / np.linalg.norm(c), t_hat), -1, 1)))

    before = _neck_rotation_vs_chest(ch, neutral)
    after = _neck_rotation_vs_chest(ch, nod)
    assert after - before > 15.0, (
        f"neck barely moved for a 25 deg nod ({before:.1f} -> {after:.1f} deg)")

    # and the posed head PITCHES by a comparable angle — measured as the
    # read-back head's angle off the torso line, since the read-back (a skull
    # point) and the target (a nose) sit at different phase angles and their
    # displacement chords legitimately differ in direction
    def head_pitch(pose):
        got = ch.posed_joints(pose, ~np.isnan(pose).any(1))
        v = got[int(Joint.HEAD)] - got[int(Joint.NECK)]
        return np.degrees(np.arccos(np.clip(
            np.dot(v / np.linalg.norm(v), t_hat), -1, 1)))

    dp = head_pitch(nod) - head_pitch(neutral)
    assert 12.0 < dp < 40.0, f"posed head pitched {dp:.1f} deg for a 25 deg nod"


def test_missing_head_leaves_the_neck_inherited():
    """No head point -> the neck rides the chest exactly as before the fix."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    pose = sub.copy()
    pose[int(Joint.HEAD)] = np.nan
    skin, *_ = ch._skin_matrices(pose, ~np.isnan(pose).any(1))
    assert np.allclose(skin[ch.role["neck"]], skin[ch.role["chest"]])
    assert not np.isnan(ch.posed_joints(pose, ~np.isnan(pose).any(1))).any()


def test_knee_dragged_to_the_hip_folds_the_thigh():
    """The reported defect, verbatim: the captured knee moved almost onto the
    hip, and the rig's knee stayed half bent because IK only read the
    hip->ankle distance. The thigh must now point where the captured knee is."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    pose = sub.copy()
    hip = pose[int(Joint.LEFT_HIP)]
    knee = pose[int(Joint.LEFT_KNEE)]
    # 80% of the way from the knee to the hip, pulled forward so the fold
    # direction is unambiguous
    target = hip + 0.2 * (knee - hip) + np.array([0.0, 0.35, 0.05])
    pose[int(Joint.LEFT_KNEE)] = target
    got = ch.posed_joints(pose, ~np.isnan(pose).any(1))

    want = target - hip
    have = got[int(Joint.LEFT_KNEE)] - got[int(Joint.LEFT_HIP)]
    cos = np.dot(want, have) / (np.linalg.norm(want) * np.linalg.norm(have))
    assert cos > 0.99, f"thigh did not follow the dragged knee (cos={cos:.4f})"


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
    # The sparse frame's legs pose differently (no ankle to aim at), which moves
    # the silhouette by a percent or so. What must not happen is the FIGURE
    # RESIZING: per-frame scaling used to inflate it by ~37% here, so this band
    # separates a pose difference from a scale pop by an order of magnitude.
    assert abs(heights[1] - heights[0]) / heights[0] < 0.03, heights


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
    # The textbook 1.27 is measured from the acromion. Rigs that put the
    # shoulder at the humeral head (MakeHuman does) read nearer 1.0 for the same
    # anatomy, so the band has to admit both conventions while still rejecting
    # a stylised rig — the one we replaced was 0.83.
    assert 0.90 <= upper / fore <= 1.40, f"upper_arm:forearm = {upper / fore:.3f}"

"""The character mimics the capture without ever deforming.

These guard the inversion: the rig's bone lengths are inviolable and motion
transfers as rotation only. Every bone aims at its own captured joint, so
joint DIRECTIONS track the capture exactly — dragging a knee moves the rig's
knee — while end effectors land within the fitted proportion mismatch. Two-bone
IK remains as the occlusion fallback, recovering a limb from its end effector
when the mid joint is missing.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from pose3d.core.skeleton import BONES, Joint
from tests.gates import needs_character
from tests.synth import rot_about as _rot_about, sample_skeleton_3d

pytestmark = needs_character()


def _ch(**kw):
    from pose3d.geometry.character import Character
    return Character(**kw)


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


#: The COMMITTED client take — the same 26 frames and the same calibration as
#: `workspace/pose3d_projects/Imported_Session`, re-detected with the shipping
#: detector (see tests/fixtures/regen_client_take.py). It is read instead of
#: the workspace copy because that copy is gitignored client data: every gate
#: that reached for it passed on this machine and SKIPPED everywhere else,
#: which is a gate that does not exist.
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "client_take"


@lru_cache(maxsize=1)
def _fixture_take():
    from pose3d.core.io_project import load_project
    from pose3d.geometry.orient import de_tilt_matrix, sequence_up
    p = load_project(_FIXTURE)
    poses = np.stack([f.fitted3d for f in p.frames])
    return poses @ de_tilt_matrix(sequence_up(poses)).T, p.head_source


def fixture_poses():
    """The client take's de-tilted delivered poses. Real capture noise.

    The fixture's stored `fitted3d` rather than a fresh reconstruction: what
    the tests below measure is the RETARGET onto those poses, so the input is
    held still on purpose and a pipeline regression is left to
    tests/test_client_regression.py, which recomputes the same fixture and
    measures it. (The two agree to 2e-5 of a body height today anyway.)

    A copy per call — the take is cached, and a caller that leans a pose must
    not lean it for everyone.
    """
    return _fixture_take()[0].copy()


def fixture_head_source():
    """What the fixture's canonical HEAD point IS ("skull" today).

    A `Character` measured against these poses must be told: the retarget
    corrects a nose HEAD for its ~45 deg forward offset and must not correct a
    skull one.
    """
    return _fixture_take()[1]


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

    HEAD keeps a wider band: the read-back is the head-bone MID (a point
    inside the skull) while the capture's HEAD joint is the nose, so the two
    are offset by construction rather than by error.
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

def _head_pts(ch, sub, right=None, fwd=None, up=None, scale=0.12):
    """Face keypoints (nose, eyes, ears) for a head with a given orientation.

    Built around the captured NECK so the basis is the only variable: `right`
    is the ear-to-ear axis, `fwd` where the face points.
    """
    neck = sub[int(Joint.NECK)]
    torso = neck - sub[int(Joint.PELVIS)]
    up = up if up is not None else torso / np.linalg.norm(torso)
    right = right if right is not None else np.array([1.0, 0.0, 0.0])
    fwd = fwd if fwd is not None else np.cross(up, right)
    right = right / np.linalg.norm(right)
    fwd = fwd / np.linalg.norm(fwd)
    centre = neck + up * scale * 2.0
    # COCO's left_* are the SUBJECT's left, and `right` points to their right,
    # so the left ear/eye sit at -right.
    return np.array([
        centre + fwd * scale,                                # nose
        centre + fwd * scale * 0.8 - right * scale * 0.3,    # left eye
        centre + fwd * scale * 0.8 + right * scale * 0.3,    # right eye
        centre - right * scale * 0.5,                        # left ear
        centre + right * scale * 0.5,                        # right ear
    ])


def _head_dirs(ch, pose, head_pts):
    """(right, forward) of the POSED head bone, in capture space."""
    skin, pelvis, scale, Rz = ch._skin_matrices(
        pose, ~np.isnan(pose).any(1), head_pts)
    R = skin[ch.role["head"]][:3, :3]
    rest = ch._rest_head_basis()
    world = R @ rest                       # rest basis carried into the pose
    return Rz.T @ world[:, 0], Rz.T @ world[:, 1]


def test_the_head_follows_the_face_keypoints():
    """The headline fix: two ears carry an orientation a lone nose cannot.

    Turning the head about the torso axis must turn the character's head — the
    case the old single-point aim collapsed to almost nothing (the captured
    nose direction moved ~20 deg across a take where the head visibly turned
    far more).
    """
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    torso = sub[int(Joint.NECK)] - sub[int(Joint.PELVIS)]
    axis = torso / np.linalg.norm(torso)

    straight = _head_pts(ch, sub)
    turned = _head_pts(ch, sub, right=_rot_about(axis, 40.0) @ np.array([1., 0, 0]))

    _, f0 = _head_dirs(ch, sub, straight)
    _, f1 = _head_dirs(ch, sub, turned)
    turn = np.degrees(np.arccos(np.clip(np.dot(f0, f1), -1, 1)))
    assert 30.0 < turn < 50.0, f"head turned {turn:.1f} deg for a 40 deg turn"


def test_the_head_is_not_flipped():
    """Absolute orientation, not just relative.

    Regression guard: `_head_basis` took the ear axis pointing to the
    subject's LEFT while `_rest_head_basis` took the shoulder line pointing
    RIGHT. Each basis was individually right-handed, so the composition was a
    clean 180 deg flip — the head pointed DOWN into the neck and the read-back
    HEAD joint collapsed onto NECK. Every angle-between-two-poses check passes
    under a consistent flip, which is exactly why this one measures against
    the capture instead.
    """
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    valid = ~np.isnan(sub).any(1)
    pts = _head_pts(ch, sub)

    with_pts = ch.posed_joints(sub, valid, pts)
    without = ch.posed_joints(sub, valid)
    d_with = np.linalg.norm(with_pts[int(Joint.HEAD)] - with_pts[int(Joint.NECK)])
    d_without = np.linalg.norm(without[int(Joint.HEAD)] - without[int(Joint.NECK)])
    assert d_with > 0.5 * d_without, (
        f"head collapsed toward the neck ({d_with:.4f} vs {d_without:.4f}) "
        "— the head basis is probably flipped")

    # and the head bone still points broadly up the body, not back down it
    torso = sub[int(Joint.NECK)] - sub[int(Joint.PELVIS)]
    up = torso / np.linalg.norm(torso)
    head_axis = with_pts[int(Joint.HEAD)] - with_pts[int(Joint.NECK)]
    assert np.dot(head_axis / np.linalg.norm(head_axis), up) > 0.3


def test_the_ear_axis_points_the_same_way_as_the_shoulders():
    """The two bases must share a handedness convention; this pins it."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    right_shoulder = sub[int(Joint.RIGHT_SHOULDER)] - sub[int(Joint.LEFT_SHOULDER)]
    basis = ch._head_basis(_head_pts(ch, sub, right=right_shoulder))
    assert basis is not None
    cos = np.dot(basis[:, 0], right_shoulder / np.linalg.norm(right_shoulder))
    assert cos > 0.9, f"ear axis opposes the shoulder line (cos={cos:.3f})"


def test_the_head_follows_a_nod():
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    right = np.array([1.0, 0.0, 0.0])

    level = _head_pts(ch, sub)
    _, f0 = _head_dirs(ch, sub, level)
    nodded = _head_pts(ch, sub, fwd=_rot_about(right, 25.0) @ f0)
    _, f1 = _head_dirs(ch, sub, nodded)
    nod = np.degrees(np.arccos(np.clip(np.dot(f0, f1), -1, 1)))
    assert 15.0 < nod < 35.0, f"head pitched {nod:.1f} deg for a 25 deg nod"


def test_eyes_stand_in_when_an_ear_is_hidden():
    """A turned head hides one ear; the eyes carry the same lateral axis."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    full = _head_pts(ch, sub)
    no_ears = full.copy()
    no_ears[3:] = np.nan
    assert ch._head_basis(full) is not None
    assert ch._head_basis(no_ears) is not None, "eyes should substitute"
    # and with neither pair there is nothing to orient from
    bare = full.copy(); bare[1:] = np.nan
    assert ch._head_basis(bare) is None


def test_without_face_keypoints_the_nose_still_bends_the_neck():
    """The legacy guarantee, re-asserted: projects saved before face
    keypoints existed (and the manual detector) drive the neck by aiming at
    the bias-corrected canonical HEAD. Regression guard — routing the neck to
    the ear midpoint alone made this resolve to nothing, so the neck was
    welded to the chest again on legacy data."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    neck = sub[int(Joint.NECK)]
    torso = neck - sub[int(Joint.PELVIS)]
    t_hat = torso / np.linalg.norm(torso)
    right = sub[int(Joint.RIGHT_SHOULDER)] - sub[int(Joint.LEFT_SHOULDER)]

    # place the nose at the anatomical neutral, then nod it forward
    d = sub[int(Joint.HEAD)] - neck
    from pose3d.geometry.character import _NOSE_PITCH
    cur = np.arccos(np.clip(np.dot(d / np.linalg.norm(d), t_hat), -1, 1))
    cands = [_rot_about(right, s * np.degrees(_NOSE_PITCH - cur)) @ d
             for s in (1.0, -1.0)]
    neutral_d = max(cands, key=lambda c: np.arccos(np.clip(
        np.dot(c / np.linalg.norm(c), t_hat), -1, 1)))
    neutral = sub.copy(); neutral[int(Joint.HEAD)] = neck + neutral_d
    nod_d = max([_rot_about(right, s * 25.0) @ neutral_d for s in (1.0, -1.0)],
                key=lambda c: np.arccos(np.clip(
                    np.dot(c / np.linalg.norm(c), t_hat), -1, 1)))
    nod = sub.copy(); nod[int(Joint.HEAD)] = neck + nod_d

    def neck_rot(pose):
        skin, *_ = ch._skin_matrices(pose, ~np.isnan(pose).any(1))
        rel = skin[ch.role["neck"]][:3, :3] @ skin[ch.role["chest"]][:3, :3].T
        return np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))

    assert neck_rot(nod) - neck_rot(neutral) > 15.0, (
        "the neck ignored a 25 deg nose nod with no face keypoints")


def test_no_face_keypoints_leaves_the_head_riding_the_neck():
    """Old projects and the manual detector supply none; behaviour must be
    exactly what it was before face keypoints existed."""
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    valid = ~np.isnan(sub).any(1)
    a, *_ = ch._skin_matrices(sub, valid)
    b, *_ = ch._skin_matrices(sub, valid, None)
    assert np.allclose(a, b)
    assert np.allclose(a[ch.role["head"]], a[ch.role["neck"]])


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


# --- roll: the bone's spin about its own aim -------------------------------
# `_align` gives the MINIMAL rotation, which leaves a bone's spin about its own
# aim axis as a numerical accident. These pin it to anatomy instead: the limb's
# captured bend plane for the arms and legs, the captured hip and shoulder
# lines for the torso. The defect is purely angular — every positional metric
# in this file is blind to it — so these tests are the only net it has.

def _roll_gauge(ch, pose, role, gauge=None, tgt=None):
    """(roll error vs the captured bend plane, roll angle off `gauge`), in deg.

    Both are measured the way the audit did: project the bone's carried rest
    reference and the target perpendicular to the POSED bone axis and take the
    signed angle between them about that axis.

    `tgt` overrides the captured bend normal with one the caller worked out for
    itself, in pose space. Without it this helper reads `_BEND_REF` and calls
    `Character._hemisphere`, so it shares the joint table and the sign
    convention with the code under test and cannot catch either being wrong —
    which is why the headline 40/60 case supplies its own.
    """
    from pose3d.geometry.character import _BEND_REF, Character, _proj_perp
    valid = ~np.isnan(pose).any(1)
    skin, _, _, Rz = ch._skin_matrices(pose, valid)
    b = ch.role[role]
    R = skin[b][:3, :3]
    ax = R @ (ch.tail[b] - ch.head[b])
    ax = ax / np.linalg.norm(ax)
    cur = _proj_perp(R @ ch._rest_ref[b], ax)

    def signed(x):
        return np.degrees(np.arctan2(float(np.dot(np.cross(x, cur), ax)),
                                     float(np.dot(x, cur))))

    if tgt is None:
        ja, jm, jb, line = _BEND_REF[role]
        tgt = Character._hemisphere(
            Rz @ np.cross(pose[int(jm)] - pose[int(ja)],
                          pose[int(jb)] - pose[int(jm)]),
            Rz @ (pose[int(line[1])] - pose[int(line[0])]))
    else:
        tgt = Rz @ np.asarray(tgt, float)
    tgt = None if tgt is None else _proj_perp(tgt, ax)
    err = None if tgt is None else -signed(tgt)
    return err, (None if gauge is None else signed(_proj_perp(gauge, ax)))


def _bent_left_arm(ch, bend_deg, twist_deg):
    """A subject with the rig's own proportions whose left elbow is bent
    `bend_deg` off straight, in a plane rolled `twist_deg` about the upper arm.
    """
    from pose3d.geometry.character import _perp
    sub = _subject_from_rig(ch)
    sh, el = sub[int(Joint.LEFT_SHOULDER)], sub[int(Joint.LEFT_ELBOW)]
    wr = sub[int(Joint.LEFT_WRIST)]
    axis = (el - sh) / np.linalg.norm(el - sh)
    side = _rot_about(axis, twist_deg) @ _perp(axis)
    a = np.radians(bend_deg)
    sub[int(Joint.LEFT_WRIST)] = el + np.linalg.norm(wr - el) * (
        np.cos(a) * axis + np.sin(a) * side)
    return sub


# The bend-plane normal of `_bent_left_arm(ch, 40, 60)` on the bundled rig,
# worked out ONCE from the synthetic construction and written down here so the
# test does not re-derive its own answer through `_BEND_REF` and
# `Character._hemisphere`: unit(cross(elbow - shoulder, wrist - elbow)), signed
# onto the shoulder line's side. Swap the rig and this must be re-derived,
# which is the point of hard-coding it.
_BEND_NORMAL_40_60 = np.array([-0.385651, 0.891343, -0.238289])


def test_limb_roll_matches_the_captured_bend_plane():
    """The upper arm's spin is a MEASUREMENT: the elbow is a hinge, so the
    plane the arm bends in fixes it. Bend the synthetic elbow 40 deg in a plane
    rolled 60 deg about the upper arm and the posed bone must carry its rest
    bend reference onto that plane."""
    ch = _ch()
    sub = _bent_left_arm(ch, 40.0, 60.0)

    # the literal really is this pose's bend normal: perpendicular to both limb
    # segments and on the shoulder line's side. Arithmetic only — no character
    # code, so a wrong joint table or a flipped hemisphere in the production
    # helpers cannot hide inside it.
    up = sub[int(Joint.LEFT_ELBOW)] - sub[int(Joint.LEFT_SHOULDER)]
    fore = sub[int(Joint.LEFT_WRIST)] - sub[int(Joint.LEFT_ELBOW)]
    line = sub[int(Joint.RIGHT_SHOULDER)] - sub[int(Joint.LEFT_SHOULDER)]
    assert abs(np.dot(_BEND_NORMAL_40_60, up / np.linalg.norm(up))) < 1e-5
    assert abs(np.dot(_BEND_NORMAL_40_60, fore / np.linalg.norm(fore))) < 1e-5
    assert np.dot(_BEND_NORMAL_40_60, line) > 0

    ch.fit_to_subject(sub[None])
    err, _ = _roll_gauge(ch, sub, "upper_arm.L", tgt=_BEND_NORMAL_40_60)
    assert abs(err) < 2.0, f"roll is {err:.1f} deg off the captured bend plane"


def test_roll_is_continuous_across_a_straightening_limb():
    """A limb that straightens loses its bend plane, so the roll must FADE, not
    switch off. Sweeping 60 -> 0 deg of bend one degree at a time, no step may
    move the roll more than 15 deg; the unweighted correction jumps 55.1."""
    ch = _ch()
    ch.fit_to_subject(_subject_from_rig(ch)[None])
    gauge = np.array([0.0, 0.0, 1.0])
    rolls = [_roll_gauge(ch, _bent_left_arm(ch, float(d), 60.0),
                         "upper_arm.L", gauge)[1]
             for d in range(60, -1, -1)]
    step = np.abs(np.diff(np.degrees(np.unwrap(np.radians(rolls)))))
    assert step.max() <= 15.0, f"roll jumps {step.max():.1f} deg in one degree of bend"


def test_roll_does_not_move_the_canonical_joints(monkeypatch):
    """What makes the limb roll safe to ship: it spins each bone about its own
    aim, and every limb child's head sits ON that aim, so not one canonical
    joint moves. (The TORSO reference deliberately does move the hip line —
    that is the point of it — so it stays on in both halves here.)"""
    import pose3d.geometry.character as C
    with_roll = [_ch().posed_joints(p, ~np.isnan(p).any(1)) for p in _poses()]
    monkeypatch.setattr(C, "_BEND_REF", {})
    without = [_ch().posed_joints(p, ~np.isnan(p).any(1)) for p in _poses()]
    for a, b in zip(with_roll, without):
        assert np.allclose(a, b, atol=1e-9, equal_nan=True), \
            f"limb roll moved a joint by {np.nanmax(np.abs(a - b)):.2e}"


def test_roll_weight_zero_restores_the_minimal_rotation(monkeypatch):
    """The documented rollback: setting the weight to 0 must reproduce the
    pre-roll matrices exactly, not approximately.

    "Bit for bit" carries ONE documented exception, asserted at the bottom of
    this test: `_align`'s antipodal branch keeps the rest reference whatever
    the weight is, because there the pre-roll answer was an arbitrary pick that
    flipped a bone's roll by 163.8 deg for a 1 deg wobble (F43). No synthetic
    pose here puts a bone at c < -0.999999, so the loop above would pass either
    way — hence the explicit `_align` check.
    """
    import pose3d.geometry.character as C
    monkeypatch.setattr(C, "_ROLL_WEIGHT", 0.0)
    off = _ch()
    monkeypatch.setattr(C, "_BEND_REF", {})
    monkeypatch.setattr(C, "_LINE_REF", {})
    bare = _ch()
    for p in _poses():
        valid = ~np.isnan(p).any(1)
        assert np.abs(off._skin_matrices(p, valid)[0]
                      - bare._skin_matrices(p, valid)[0]).max() == 0.0

    # the exception, stated as a fact rather than left in a comment
    a = np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    delta = np.abs(C._align(a, -a, ref=ref) - C._align(a, -a)).max()
    assert delta > 0.5, ("the antipodal branch no longer depends on the rest "
                         "reference — update the _ROLL_WEIGHT rollback note")


def test_hip_line_follows_capture(monkeypatch):
    """The pelvis used to keep whatever spin the torso aim left it with, so the
    character's hips faced the rig's way, not the subject's."""
    import pose3d.geometry.character as C
    ch = _ch()
    sub = _subject_from_rig(ch)
    ch.fit_to_subject(sub[None])
    pelvis = sub[int(Joint.PELVIS)]
    twist = _rot_about([0, 0, 1.0], 25.0)
    for j in (Joint.LEFT_HIP, Joint.RIGHT_HIP, Joint.LEFT_KNEE, Joint.RIGHT_KNEE,
              Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE):
        sub[int(j)] = twist @ (sub[int(j)] - pelvis) + pelvis

    def hip_error(c):
        got = c.posed_joints(sub, ~np.isnan(sub).any(1))
        a = got[int(Joint.RIGHT_HIP)] - got[int(Joint.LEFT_HIP)]
        b = sub[int(Joint.RIGHT_HIP)] - sub[int(Joint.LEFT_HIP)]
        return np.degrees(np.arccos(np.clip(float(
            np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))), -1, 1)))

    assert hip_error(ch) <= 2.0                 # measured 0.0 deg
    monkeypatch.setattr(C, "_LINE_REF", {})
    assert hip_error(_ch()) > 15.0, "the test cannot see the hip reference"


def test_align_is_deterministic_near_180():
    """At exactly 180 deg every axis perpendicular to the bone maps it onto its
    target, so the arbitrary pick flipped the roll by 163.8 deg for a 1 deg
    wobble. With a reference the choice is the same on both sides."""
    from pose3d.geometry.character import _align, _proj_perp
    a = np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 1.0, 0.0])            # a rest roll reference
    gauge = np.array([1.0, 0.0, 0.0])
    rolls = []
    for deg in (179.0, 180.0, 181.0):
        b = _rot_about([1.0, 0, 0], deg) @ a
        cur = _proj_perp(_align(a, b, ref=ref) @ ref, b)
        g = _proj_perp(gauge, b)
        rolls.append(np.degrees(np.arctan2(
            float(np.dot(np.cross(g, cur), b)), float(np.dot(g, cur)))))
    step = np.abs(np.diff(np.degrees(np.unwrap(np.radians(rolls)))))
    assert step.max() < 5.0, f"roll flips {step.max():.1f} deg across 180"


def _roll_error_undirected(ch, pose, role):
    """|roll error| for one bone on one frame, in deg, as an UNDIRECTED angle.

    The bend plane is a PLANE, so a normal and its negative describe the same
    one: taking |dot| makes this measurement independent of the hemisphere rule
    the production code uses to sign it, which is what keeps this a check of
    the roll rather than a restatement of the convention. Returns (bend, err),
    or None when the frame cannot supply the plane.
    """
    from pose3d.geometry.character import _BEND_REF, _proj_perp, _unit
    valid = ~np.isnan(pose).any(1)
    ja, jm, jb, _ = _BEND_REF[role]
    if not all(valid[int(x)] for x in (ja, jm, jb)):
        return None
    v1 = _unit(pose[int(jm)] - pose[int(ja)])
    v2 = _unit(pose[int(jb)] - pose[int(jm)])
    if v1 is None or v2 is None:
        return None
    bend = np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)), -1.0, 1.0)))
    skin, _, _, Rz = ch._skin_matrices(pose, valid)
    b = ch.role[role]
    R = skin[b][:3, :3]
    ax = _unit(R @ (ch.tail[b] - ch.head[b]))
    cur = _proj_perp(R @ ch._rest_ref[b], ax)
    tgt = _proj_perp(Rz @ np.cross(pose[int(jm)] - pose[int(ja)],
                                   pose[int(jb)] - pose[int(jm)]), ax)
    if cur is None or tgt is None:
        return None
    # atan2, not arccos: near zero arccos(1 - eps) loses half its digits and
    # would read 1e-6 deg where the roll is exact to 1e-14.
    return bend, float(np.degrees(np.arctan2(
        float(np.linalg.norm(np.cross(cur, tgt))),
        abs(float(np.dot(cur, tgt))))))


def test_roll_error_is_zero_where_the_bend_plane_exists():
    """The RESTATED ship gate for the roll (controller ruling 1/2a).

    The brief asked for <= 5 deg median / <= 15 deg max roll error over every
    frame, together with a 20/40 deg ramp that withholds the roll on nearly
    straight limbs — two requirements that cannot both hold on a take whose
    knees are often straight. The gate as restated applies where the bend plane
    actually exists (bend >= 40 deg, full weight), and there the error must be
    ZERO, not merely small: the bone is rotated onto the captured plane
    exactly. The residual on the ramped frames is recorded in
    docs/audit-2026-09/phase2_metrics.json (`roll_error_ramped_frames_deg`,
    up to 50.8 deg on upper_arm.L at 22.7 deg of elbow bend) and deliberately
    not gated. Continuity is covered by the straightening sweep above; the
    brief's max consecutive-frame change is a metric, not a gate, because the
    CAPTURED bend normal itself moves up to 55.7 deg between these discrete
    stop-motion poses.

    Measured on the COMMITTED client take, so this runs everywhere: it used to
    read the gitignored workspace copy and skip on every other checkout, i.e.
    one of Phase 2's two headline angular fixes had no gate anywhere CI runs.
    """
    from pose3d.geometry.character import _BEND_REF, _ROLL_BEND_FULL_DEG
    poses = fixture_poses()
    ch = _ch(head_source=fixture_head_source())
    ch.fit_to_subject(poses)
    worst = 0.0
    for role in _BEND_REF:
        measured = [_roll_error_undirected(ch, p, role) for p in poses]
        errs = [err for m in measured if m is not None
                for bend, err in [m] if bend >= _ROLL_BEND_FULL_DEG]
        assert errs, f"{role}: the take never bends this limb past full weight"
        # measured: worst median 6.6e-15 deg, worst max 2.9e-14 deg
        # over the eight bones — zero to floating point, not merely small
        # (7.1e-15 / 2.6e-14 on the COCO-17 workspace copy this used to read)
        assert float(np.median(errs)) <= 1e-6, f"{role} median"
        assert float(np.max(errs)) <= 1e-6, f"{role} max"
        worst = max(worst, float(np.max(errs)))
    assert worst <= 1e-6


def test_a_missing_wrist_falls_back_to_weight_zero():
    """A limb's bend plane needs all three of its joints, so a NaN wrist takes
    the roll away for that frame and the bone snaps back to the minimal
    rotation. That discontinuity is ACCEPTED and stateless (controller ruling
    3): fading it in would need caller-owned temporal state, which desyncs the
    scrubbed 3D view from the once-through Blender export. This pins the
    fallback — the matrices are exactly the reference-free ones, nothing
    raises, and the view and the export still agree — and the size of the pop.
    """
    import pose3d.geometry.character as C
    ch = _ch()
    sub = _bent_left_arm(ch, 60.0, 60.0)          # well past the 40 deg ramp
    ch.fit_to_subject(sub[None])
    gone = sub.copy()
    gone[int(Joint.LEFT_WRIST)] = np.nan
    valid = ~np.isnan(gone).any(1)
    b = ch.role["upper_arm.L"]

    got = ch._skin_matrices(gone, valid)[0]       # must not raise
    assert not np.isnan(got[b]).any()

    # the roll this frame keeps and the roll it loses, both measured before
    # anything is monkeypatched
    gauge = np.array([0.0, 0.0, 1.0])
    _, held = _roll_gauge(ch, sub, "upper_arm.L", gauge)

    # the fallback, stated exactly: the upper arm's matrix on the gappy frame
    # IS its reference-free matrix on the intact one, bit for bit. (The wrist
    # is downstream of this bone, so nothing else about it can differ.)
    saved = C._BEND_REF
    try:
        C._BEND_REF = {}
        bare = _ch()
        bare.fit_to_subject(sub[None])
        want = bare._skin_matrices(sub, ~np.isnan(sub).any(1))[0]
    finally:
        C._BEND_REF = saved
    assert np.abs(got[b] - want[b]).max() == 0.0

    # the pop it costs, measured against the same frame with the wrist present
    from pose3d.geometry.character import _proj_perp, _unit
    R = got[b][:3, :3]
    ax = _unit(R @ (ch.tail[b] - ch.head[b]))
    cur = _proj_perp(R @ ch._rest_ref[b], ax)
    g = _proj_perp(gauge, ax)
    lost = np.degrees(np.arctan2(float(np.dot(np.cross(g, cur), ax)),
                                 float(np.dot(g, cur))))
    pop = abs((held - lost + 180.0) % 360.0 - 180.0)
    # measured 70.8 deg here, ~50 deg worst case on the client take. Bounded,
    # not banned: this test exists so the discontinuity is a decision on the
    # record rather than a surprise.
    assert pop <= 80.0, f"the validity gate now pops {pop:.1f} deg"

    # and the export still matches the view on that frame
    mats = ch.pose_bone_matrices(gone, valid)
    skin = np.array([np.array(mats[n]) @ np.linalg.inv(ch.rest[i])
                     for i, n in enumerate(ch.bone_names)])
    _, pelvis, scale, Rz = ch._skin_matrices(gone, valid)
    assert np.allclose(ch._from_rig(ch._joints_from_skin(skin), pelvis, scale, Rz),
                       ch.posed_joints(gone, valid), atol=1e-9, equal_nan=True)


# --- one skinning for the view and the export ------------------------------

def test_view_and_export_share_one_skinning():
    """`pose_bone_matrices` is what Blender is driven with; run it back through
    Blender's own deform (matrix @ rest^-1) and it must reproduce the joints the
    3D view draws. Anything else and the export stops matching the preview.

    What this pins is the DOCUMENTED BLENDER DEFORM FORMULA — that
    `pose_bone_matrices` returns `skin @ rest`, so setting `pose_bone.matrix`
    to it reproduces the view's skinning. It is not a cross-check of two
    independent code paths: nothing here runs `pose3d/export/*`, and the
    round-trip is an identity for any implementation with that property. The
    export path itself is covered by tests/test_export_smoke.py.
    """
    ch = _ch()
    for p in list(_poses()) + list(fixture_poses()):
        valid = ~np.isnan(p).any(1)
        mats = ch.pose_bone_matrices(p, valid)
        skin = np.array([np.array(mats[n]) @ np.linalg.inv(ch.rest[b])
                         for b, n in enumerate(ch.bone_names)])
        _, pelvis, scale, Rz = ch._skin_matrices(p, valid)
        got = ch._from_rig(ch._joints_from_skin(skin), pelvis, scale, Rz)
        want = ch.posed_joints(p, valid)
        assert np.allclose(got, want, atol=1e-9, equal_nan=True)


def test_the_exported_matrices_are_the_view_through_one_similarity():
    """The SHIPPING configuration: `keep_root_motion=True`.

    `test_view_and_export_share_one_skinning` above pins the rig-space form,
    which is the rollback and is what `_character_bone_frames` (and so
    test_orient_up / test_gap_fill) reads. The delivered file is written the
    other way — in the de-tilted capture frame — and that path had no
    numpy-level guard at all: only the Blender-gated export tests covered it,
    so on a machine without Blender nothing checked it.

    What must hold, for the WHOLE take at once: the joints the exporter writes
    are the joints the 3D view draws, through ONE similarity — the fitted
    scale, no rotation, and the constant offset onto the take's pelvis. One
    for the take, so a per-frame yaw (round 1's `Rz @ (pelvis - ref) * scale`
    placement, which left the character facing forward while the subject
    turned) cannot hide in it.
    """
    from pose3d.geometry.character import take_pelvis_ref
    ch = _ch()
    poses = np.stack(list(_poses()) + list(fixture_poses()))
    ch.fit_to_subject(poses)
    ref = take_pelvis_ref(poses)
    assert ref is not None

    got, want = [], []
    for p in poses:
        valid = ~np.isnan(p).any(1)
        mats, al = ch.pose_bone_matrices(p, valid, None, keep_root_motion=True,
                                         pelvis_ref=ref, return_alignment=True)
        skin = np.array([np.array(mats[n]) @ np.linalg.inv(ch.rest[b])
                         for b, n in enumerate(ch.bone_names)])
        # `skin` here is the EXPORTED matrix's deform, so the joints it carries
        # are already in the capture frame; no `_from_rig` and no per-frame
        # anything is allowed to intervene
        j = ch._joints_from_skin(skin)
        view = ch.posed_joints(p, valid)
        m = np.isfinite(j).all(1) & np.isfinite(view).all(1)
        got.append(j[m]); want.append(view[m])
        assert al.scale == ch._scale
    got, want = np.concatenate(got), np.concatenate(want)
    resid = got - want * ch._scale
    assert np.abs(resid - resid.mean(0)).max() <= 1e-9 * ch.rig_h, \
        f"the export is not the view: {np.abs(resid - resid.mean(0)).max():.9f}"
    # ...and the constant IS the take's pelvis mapped onto the rig's hips
    assert np.allclose(resid.mean(0), ch.hips_world - ref * ch._scale, atol=1e-9)

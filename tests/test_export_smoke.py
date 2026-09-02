"""Phase 5 verification: Blender headless export produces BVH/FBX(/mp4).

Skipped automatically if the Blender binary is not present. Uses a short
synthetic 5-frame motion.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.config import blender_binary, character_blend
from pose3d.core.skeleton import Joint
from pose3d.export.blender_export import export_animation
from tests import bvh_util
from tests.gates import needs_blender, needs_character, needs_video_render
from tests.synth import rot_about, sample_skeleton_3d

_BLENDER = blender_binary()


def _motion(n=5):
    base = sample_skeleton_3d()
    seq = []
    for i in range(n):
        p = base.copy()
        # swing arms a little each frame so there is animation
        p[6, 0] -= 0.02 * i     # left wrist
        p[7, 0] += 0.02 * i     # right wrist
        seq.append(p)
    return np.stack(seq)


@needs_blender()
def test_export_bvh_fbx(tmp_path):
    res = export_animation(_motion(), tmp_path, name="t", fps=30,
                           render_video=False, timeout=300)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    assert res.bvh and res.bvh.stat().st_size > 0
    assert res.fbx and res.fbx.stat().st_size > 0


_CHARACTER = character_blend()


@needs_blender()
@needs_character()
def test_export_retargets_character(tmp_path):
    """The skinned character FBX — what the client imports. No rendering, so
    this holds even where Blender has no usable OpenGL."""
    res = export_animation(_motion(), tmp_path, name="c", fps=24,
                           render_video=False, timeout=500, character=_CHARACTER)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    # a real skinned character FBX is much larger than a stick figure
    assert res.fbx and res.fbx.stat().st_size > 100_000


@needs_blender()
@needs_character()
@needs_video_render()
def test_rendered_video_shows_the_posed_character(tmp_path):
    """Separate from the FBX above on purpose: rendering is the one part that
    legitimately cannot run on a machine without OpenGL, and it must not be
    able to mask a broken character export."""
    res = export_animation(_motion(), tmp_path, name="c", fps=24,
                           render_video=True, timeout=500, character=_CHARACTER)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    import cv2
    cap = cv2.VideoCapture(str(res.mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, frame = cap.read(); cap.release()
    assert ok and float(frame.std()) > 3.0   # posed character is visible


@needs_blender()
@needs_character()
def test_export_produces_hierarchical_mocap_rig(tmp_path):
    """The character export must be a real armature: a nested bone hierarchy
    with rotation animation. Regression guard — driving the rig by flattening it
    (every bone unparented, keyed in world space) gave a bone list with no
    hierarchy at all, so the FBX/BVH were unusable as mocap."""
    res = export_animation(_motion(), tmp_path, name="m", fps=24,
                           render_video=False, timeout=400, character=_CHARACTER)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    assert res.fbx and res.fbx.stat().st_size > 100_000

    head = res.bvh.read_text().split("MOTION")[0]
    depth = maxd = 0
    for line in head.splitlines():
        depth += line.count("{") - line.count("}")
        maxd = max(maxd, depth)
    # a flattened rig nests only ROOT -> JOINT (depth 2); a real skeleton chains
    # hips -> thigh -> shin -> foot -> ...
    assert maxd > 4, f"BVH hierarchy is flat (max nesting depth {maxd})"
    assert "End Site" in head


@needs_blender()
@needs_video_render()
def test_export_with_video(tmp_path):
    res = export_animation(_motion(), tmp_path, name="v", fps=30,
                           render_video=True, timeout=600)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    assert res.mp4 and res.mp4.stat().st_size > 0
    # the frame must contain the rendered skeleton, not a blank background
    # (armatures don't render; regression guard for the empty-mp4 bug)
    import cv2
    cap = cv2.VideoCapture(str(res.mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, "could not read a rendered frame"
    assert float(frame.std()) > 2.0, "rendered frame is blank (no geometry)"


# --- the delivered file itself, parsed (no Blender) ------------------------
# Everything below reads tests/fixtures/imported_session.bvh.gz, the BVH the
# client actually received for the take in tests/fixtures/client_take. Reading
# it needs nothing installed, so these run on every machine and every CI job —
# which is the point: the export path's defects (F13) survived because its only
# tests were "Blender exited 0 and the file is not empty".
#
# NOTE for Phase 3: that file was exported on 2026-07-28, one day before commit
# b83c91e swapped the bundled rig for the current 19-bone one. It therefore
# still carries the old rig's fingers and orphan IK helper bones. The pinned
# hips below are unchanged on today's rig; the helper-bone translation is not —
# a fresh export has none, so Phase 3 must re-record that fact from a new file
# (tools/check_export_fidelity.py prints it) rather than trust the constant.

DELIVERED_POSES = 26                  # the client take's 26 photographed poses
DELIVERED_FRAMES = 772                # 1 + 25*30 + 21, at 30 fps
RIG_HEIGHT = 14.4228                  # Character().rig_h for the bundled rig

# The largest non-hips position channel in the DELIVERED file, as a % of the
# CURRENT rig's height: an orphan IK helper bone carried the root motion that
# `to_rig` refused to give the hips. A historical fact about a file exported
# before the rig swap, so it is asserted as history; Phase 3's live gate is
# MAX_NON_HIPS_TRANSLATION_PCT on a fresh export.
DELIVERED_HELPER_TRANSLATION_PCT = 42.5   # 42.49 %, on shin.R.001

# Phase 3's gate, on a FRESH export: no bone but the hips may carry meaningful
# translation. Measured 0.00003 % on the current 19-bone rig, which has no
# helper bones at all; 1 % leaves room for a replacement rig whose helpers are
# pinned at rest rather than left to float.
MAX_NON_HIPS_TRANSLATION_PCT = 1.0

# Phase 3's headline gate, restated where it can fail: forward kinematics off
# the delivered BVH against `Character.posed_joints` — the CAPTURE's own space,
# which is what the 3D view draws — after ONE global similarity fit for the
# whole take. One fit, because a fit per keyframe re-places and re-orients
# every frame and so forgives precisely the defect this is looking for.
#
# The round-1 export was measured against the RIG-space pose it had itself
# written, which passes by construction; against `posed_joints` it measured
# 6.357 % median / 16.313 % max of body height on the client take
# (docs/audit-2026-09/phase3_metrics.json, label "round1"). That residual was
# the per-frame yaw `_skin_matrices` removes and the export never put back.
MAX_VIEW_DEVIATION_PCT = 1.0
ROUND1_CAPTURE_SPACE_MAX_PCT = 16.313     # what this gate has to catch

# ...and the same defect read off the file's own orientation: the exported
# figure's facing must TRACK the capture's. The constant angle between them is
# a rig convention and is removed; what is left is drift.
#
# Measured two ways, because on a REAL subject the hips and the shoulders
# twist against each other and the hips bone follows the hip line, not the
# shoulder line. On the client take (docs/audit-2026-09/phase3_metrics.json,
# label "after"): the file's own shoulder line tracks the captured shoulder
# line to 0.196 deg; the hips bone tracks the captured HIP line to 3.049 deg
# (the retarget's own roll fit) and the captured SHOULDER line only to 21.3
# deg, which is the subject's own 38.8 deg of torso twist and not the
# export's. Round 1 had the exported figure not turning AT ALL — 26.9 deg of
# drift on the same take — so either measure catches it. The synthetic fixture
# turns rigidly, so hips and shoulders are locked and both read ~0 there.
MAX_FACING_DRIFT_DEG = 3.0


@pytest.fixture(scope="module")
def delivered():
    from tests import bvh_util
    return bvh_util.parse(bvh_util.DELIVERED_BVH)


def test_the_delivered_file_is_one_stepped_frame_run_per_pose(delivered):
    from tests import bvh_util
    assert delivered.n_frames == DELIVERED_FRAMES
    assert delivered.n_frames == bvh_util.expected_frames(DELIVERED_POSES)
    assert delivered.fps == pytest.approx(30.0, abs=0.01)


def test_each_captured_pose_is_held_exactly(delivered):
    """The stop-motion hold must be a hold: 22 rows of the same numbers, not
    a slow drift that reads as the figure breathing.

    The run length is MEASURED off the file, not restated from the schedule
    constant that generated it — comparing `last - first` against `HOLD` when
    `last` came from `HOLD` says nothing about the file at all.
    """
    for i, (first, last) in enumerate(bvh_util.stepped_holds(DELIVERED_POSES)):
        row = delivered.motion[first]
        run = 0
        while (first + run < delivered.n_frames
               and np.array_equal(delivered.motion[first + run], row)):
            run += 1
        assert run == bvh_util.HOLD + 1, \
            f"pose {i} holds for {run} rows, not {bvh_util.HOLD + 1}"
        assert first + run == last + 1


def test_the_hips_never_move_in_the_delivered_file(delivered):
    """F13: `to_rig` pins the pelvis at the rig's rest hips on every frame, so
    the figure's 116 %-of-height travel across the take is simply absent from
    the file the client imports."""
    hips = delivered.positions("hips")
    assert hips.shape == (delivered.n_frames, 3)
    assert np.array_equal(hips, np.repeat(hips[:1], len(hips), axis=0))


def test_the_in_betweens_do_not_overshoot_the_poses_they_connect(delivered):
    """The eased transitions are fine and must stay fine: a bone that swings
    past a captured pose before settling onto it is a visible artefact and
    would be the first thing blamed for 'the character does not follow'."""
    from tests import bvh_util
    overshoot = bvh_util.channel_overshoot(delivered, DELIVERED_POSES)
    assert float(overshoot.max()) <= 5.0    # today 3.795 deg


def test_a_helper_bone_carried_the_root_motion_in_the_delivered_file(delivered):
    """HISTORICAL, and pinned as history rather than as a live bound.

    The translation the hips did not get had to go somewhere, and it landed on
    an orphan IK helper bone — motion no importer will use. That file predates
    the rig swap, so the 42.5 % below divides an OLD rig's channel by the
    CURRENT rig's height and is not a number today's export can be gated on;
    `test_helper_bones_are_pinned` gates a fresh export instead. What this
    still asserts is the shape of the defect, on the file the client has.
    """
    ranges = {}
    for j in delivered.joints:
        p = delivered.positions(j.name)
        if p.shape[1] == 3 and j.name != "hips":
            ranges[j.name] = float(np.max(p.max(0) - p.min(0)))
    name, worst = max(ranges.items(), key=lambda kv: kv[1])
    pct = 100.0 * worst / RIG_HEIGHT
    assert name == "shin.R.001"
    assert 0.99 * DELIVERED_HELPER_TRANSLATION_PCT <= pct \
        <= 1.01 * DELIVERED_HELPER_TRANSLATION_PCT, f"{name} carries {pct:.2f} %"


@needs_character()
def test_the_recorded_rig_height_is_still_the_bundled_rigs():
    """RIG_HEIGHT above is a number, not a measurement, so say out loud which
    rig it came from and fail if that rig is replaced."""
    from pose3d.geometry.character import Character
    assert Character().rig_h == pytest.approx(RIG_HEIGHT, abs=0.001)


# --- Phase 3: a FRESH export, which is what the client will now receive -----
# Everything above this line reads the file the client already has. These read
# a file this checkout produces, because that is the only way to assert on the
# current rig and the current placement rule — and because the delivered file
# predates the rig swap, so its numbers cannot gate today's code.

def _travelling_motion(n=6, step=0.06, turn_deg=8.0):
    """A take that walks AND turns: the subject moves +x while rotating.

    Both halves are load-bearing and both were missing before:

    * `_motion` above moves two wrists and nothing else, so it cannot tell an
      export that keeps the subject's travel from one that pins it away —
      which is the first half of F13.
    * without the turn, the shoulder line's yaw is constant, `Rz` is constant,
      and comparing the file against the RIG-space pose and against the
      CAPTURE-space pose give the same answer. A fixture like that passes
      `test_bvh_keyframes_match_the_view` however the export is written, and
      it is exactly what hid the 36.97 deg of unrepresented yaw on the client
      take for a whole round. `turn_deg` per pose puts a comparable span
      (0-40 deg over six poses) into the fixture.
    """
    base = sample_skeleton_3d()
    pelvis = base[int(Joint.PELVIS)].copy()
    seq = []
    for i in range(n):
        p = base.copy()
        p[6, 0] -= 0.02 * i     # left wrist
        p[7, 0] += 0.02 * i     # right wrist
        # turn the whole figure about the vertical through its own pelvis...
        R = rot_about((0, 0, 1), turn_deg * i)
        p = (p - pelvis) @ R.T + pelvis
        p[:, 0] += step * i     # ...and walk it
        seq.append(p)
    return np.stack(seq)


def _placement(seq):
    """(character fitted to the take, upright poses, pelvis_ref) as the export
    computes them — the reference the file has to reproduce."""
    from pose3d.geometry.character import Character, take_pelvis_ref
    from pose3d.geometry.orient import de_tilt_matrix, take_up
    up, _source, _spread = take_up(seq, None)
    R = de_tilt_matrix(up).T if up is not None else np.eye(3)
    upright = np.asarray(seq, float) @ R
    ch = Character()
    ch.fit_to_subject(upright)
    return ch, upright, take_pelvis_ref(upright)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """One fresh export of a travelling take, parsed: (seq, Bvh).

    Module-scoped because it costs a Blender run and every assertion below is
    about the same file.
    """
    out = tmp_path_factory.mktemp("fresh")
    seq = _travelling_motion()
    res = export_animation(seq, out, name="fresh", fps=30, render_video=False,
                           timeout=500, character=_CHARACTER)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    return seq, bvh_util.parse(res.bvh)


@needs_blender()
@needs_character()
def test_one_frame_per_pose_is_default(fresh):
    """Export frame k+1 IS photograph k. The stepped schedule put pose k at
    frame 1 + 30k, which nothing downstream could know without being told."""
    seq, bvh = fresh
    assert bvh.n_frames == len(seq)
    assert bvh_util.keyframe_rows(bvh, len(seq)) == [(i, i) for i in range(len(seq))]


@needs_blender()
@needs_character()
def test_root_motion_reaches_bvh(fresh, tmp_path):
    """F13: the hips translation in the file must be the subject's own travel.

    Measured as the DISTANCE the hips move between the first and last captured
    pose, against the same distance in the capture through the character's
    fitted scale.

    A distance is the right measure because the exported hips are
    `(pelvis_k - pelvis_ref) * scale` — the capture's own frame, no per-frame
    rotation — so it equals the captured distance exactly, on a take that
    turns as much as on one that does not. (Round 1 wrote
    `Rz_k @ (pelvis_k - pelvis_ref) * scale`, where that equality held only
    while `Rz` was constant, which was true of the old straight-ahead fixture
    and false of the client take. The docstring said "invariant to the yaw the
    retarget applies"; it was not.)
    """
    seq, bvh = fresh
    ch, upright, ref = _placement(seq)
    from pose3d.geometry.character import take_pelvis_ref
    pel = np.stack([take_pelvis_ref(p[None]) for p in upright])
    expected = float(np.linalg.norm(pel[-1] - pel[0])) * ch._scale
    assert expected > 0.1 * ch.rig_h, "the fixture take does not travel"

    rows = bvh_util.keyframe_rows(bvh, len(seq))
    hips = bvh.positions("hips")
    got = float(np.linalg.norm(hips[rows[-1][0]] - hips[rows[0][0]]))
    assert abs(got - expected) <= 0.01 * expected, \
        f"the file carries {got:.4f} rig units of hips travel, not {expected:.4f}"

    # ...and with the flag off, exactly none: the rollback switch, thrown.
    off = export_animation(seq, tmp_path, name="off", fps=30,
                           render_video=False, timeout=500,
                           character=_CHARACTER, keep_root_motion=False)
    assert off.ok, off.stderr[-2000:]
    h0 = bvh_util.parse(off.bvh).positions("hips")
    assert np.array_equal(h0, np.repeat(h0[:1], len(h0), axis=0))


@needs_blender()
@needs_character()
def test_helper_bones_are_pinned(fresh):
    """No bone but the hips may carry meaningful translation.

    A plain assertion, not an xfail: the delivered file put 42.5 % of rig
    height on `shin.R.001`, and a bone the app never drives must be keyframed
    at its rest transform — pinned, not omitted, because dropping a bone
    changes the exported armature's topology.
    """
    _seq, bvh = fresh
    ranges = {}
    for j in bvh.joints:
        p = bvh.positions(j.name)
        if p.shape[1] == 3 and j.name != "hips":
            ranges[j.name] = float(np.max(p.max(0) - p.min(0)))
    name, worst = max(ranges.items(), key=lambda kv: kv[1])
    pct = 100.0 * worst / RIG_HEIGHT
    # measured 0.00003 % on the bundled 19-bone rig
    assert pct <= MAX_NON_HIPS_TRANSLATION_PCT, f"{name} carries {pct:.3f} %"


@needs_blender()
@needs_character()
def test_bvh_keyframes_match_the_view(fresh):
    """THE gate: the delivered file holds the pose the 3D VIEW shows.

    Forward kinematics off every captured keyframe of the BVH, against
    `Character.posed_joints` — the capture's own space, which is what the view
    draws — after ONE global similarity fit (one scale, one rotation, one
    translation) for the whole take.

    One fit for the take, not one per keyframe, is the whole point: a per-frame
    fit re-places and re-orients each frame and would forgive both halves of
    "the character does not follow the keypoints". So would comparing against
    the rig-space matrices the exporter itself wrote, which is what round 1
    did: it passes however the export is written, and it passed while the
    capture-space deviation was 16.313 % of body height.

    On the client take this measures 0.0000 % of body height with this
    round's export (`phase3_metrics.json`, label "after"), against 6.357 %
    median / 16.313 % max for round 1's.
    """
    seq, bvh = fresh
    ch, upright, ref = _placement(seq)
    names = [b.name for b in bvh.joints]
    mapping = {j: bvh.index(ch.bone_names[b])
               for j, (b, which) in ch._joint_src.items()
               if which == "head" and ch.bone_names[b] in names}
    assert len(mapping) >= 10

    src, dst = [], []
    for k, (row, _last) in enumerate(bvh_util.keyframe_rows(bvh, len(seq))):
        valid = ~np.isnan(upright[k]).any(1)
        app = ch.posed_joints(upright[k], valid)
        fk = bvh.forward_kinematics(row)
        for j, bi in mapping.items():
            if valid[j] and np.isfinite(app[j]).all():
                src.append(fk[bi]); dst.append(app[j])
    err = bvh_util.similarity_error(np.asarray(src), np.asarray(dst))
    height = float(np.median([np.ptp(p[~np.isnan(p).any(1), 2]) for p in upright]))
    pct = 100.0 * float(err.max()) / height
    assert pct <= MAX_VIEW_DEVIATION_PCT, \
        f"{pct:.4f} % of body height (gate {MAX_VIEW_DEVIATION_PCT} %)"


@needs_blender()
@needs_character()
def test_the_exported_character_turns_with_the_subject(fresh):
    """The other half of the same gate, read off the file's own rotations.

    `_skin_matrices` turns every frame's shoulder line onto the rig's rest
    facing; the live view undoes that (`_from_rig`'s `Rz.T`) and the export
    used not to, so the subject turned on screen and stood rigidly forward in
    the delivered file — 36.97 deg of it on the client take. The angle between
    the exported hips' facing and the captured shoulder line is a rig
    convention, so it is the DRIFT in that angle that is asserted.
    """
    seq, bvh = fresh
    ch, upright, _ref = _placement(seq)
    hips = bvh.index("hips")
    ls, rs = int(Joint.LEFT_SHOULDER), int(Joint.RIGHT_SHOULDER)
    names = [b.name for b in bvh.joints]

    def _shoulder_bone(j):
        b, which = ch._joint_src[j]
        assert which == "head" and ch.bone_names[b] in names
        return bvh.index(ch.bone_names[b])

    bl, br = _shoulder_bone(ls), _shoulder_bone(rs)
    hips_yaw, file_yaw, cap_yaw = [], [], []
    for k, (row, _last) in enumerate(bvh_util.keyframe_rows(bvh, len(seq))):
        d = upright[k][rs] - upright[k][ls]
        cap_yaw.append(np.arctan2(d[1], d[0]))
        R = bvh.world_rotations(row)[hips]
        hips_yaw.append(np.arctan2(R[1, 0], R[0, 0]))
        fk = bvh.forward_kinematics(row)
        e = fk[br] - fk[bl]
        file_yaw.append(np.arctan2(e[1], e[0]))

    cap = np.degrees(np.unwrap(cap_yaw))
    assert np.ptp(cap) > 4 * MAX_FACING_DRIFT_DEG, \
        "the fixture does not turn, so this asserts nothing"
    for what, track in (("hips bone", hips_yaw),
                        ("exported shoulder line", file_yaw)):
        got = np.degrees(np.unwrap(track))
        drift = (got - cap) - (got - cap).mean()
        assert np.abs(drift).max() <= MAX_FACING_DRIFT_DEG, \
            (f"the {what} turns through {np.ptp(got):.2f} deg while the "
             f"subject turns through {np.ptp(cap):.2f}: drift "
             f"{np.abs(drift).max():.2f} deg")


@needs_blender()
@needs_character()
def test_no_quaternion_flips(fresh):
    """Consecutive keyframes must stay in one hemisphere: a flip is a bone
    spinning the long way round between two poses. Measured 0 pairs."""
    seq, bvh = fresh
    rows = [a for a, _b in bvh_util.keyframe_rows(bvh, len(seq))]
    flips = 0
    for j in bvh.joints:
        order = [c for c in j.channels if c.endswith("rotation")]
        if not order:
            continue
        r = bvh.rotations(j.name)
        keys = [bvh_util.euler_to_quat(r[a], order) for a in rows]
        flips += sum(1 for q0, q1 in zip(keys, keys[1:])
                     if float(np.dot(q0, q1)) < 0)
    assert flips == 0


@needs_blender()
@needs_character()
def test_inbetween_stays_on_the_arc(tmp_path):
    """The eased transitions are fine and must stay fine.

    Only the stepped schedule HAS in-betweens — that is the point of one frame
    per pose — so this exports the mp4's schedule explicitly rather than
    quietly measuring nothing.
    """
    seq = _travelling_motion()
    res = export_animation(seq, tmp_path, name="st", fps=30, render_video=False,
                           timeout=500, character=_CHARACTER,
                           schedule="stepped")
    assert res.ok, res.stderr[-2000:]
    bvh = bvh_util.parse(res.bvh)
    rows = bvh_util.keyframe_rows(bvh, len(seq))
    assert bvh.n_frames == bvh_util.expected_frames(len(seq))
    over = bvh_util.channel_overshoot(bvh, len(seq), holds=rows)
    assert float(over.max()) <= 5.0        # delivered file measured 3.795 deg


@needs_blender()
@needs_character()
def test_the_stepped_schedule_writes_a_frame_map(tmp_path):
    """`<name>_frames.json` says which export frames a captured pose occupies —
    written only where the 1 + 30k stride is real, since on the default
    schedule it would restate `k -> k+1`."""
    import json
    seq = _travelling_motion(n=4)
    res = export_animation(seq, tmp_path, name="fm", fps=30, render_video=False,
                           timeout=500, character=_CHARACTER,
                           schedule="stepped")
    assert res.ok, res.stderr[-2000:]
    doc = json.loads((tmp_path / "fm_frames.json").read_text())
    assert doc["0"] == [1, 1 + bvh_util.HOLD]
    assert doc["1"][0] == 1 + bvh_util.SEGMENT

    plain = export_animation(seq, tmp_path, name="pl", fps=30,
                             render_video=False, timeout=500,
                             character=_CHARACTER)
    assert plain.ok, plain.stderr[-2000:]
    assert not (tmp_path / "pl_frames.json").exists()


def test_missing_character_asset_reports(tmp_path):
    """F31: a missing asset must be a FAILED export naming the asset, not a
    different animation shipped under the same name.

    No Blender needed, and deliberately so: the refusal happens before the
    subprocess is launched, which is the only way it can be certain that
    nothing was written.
    """
    missing = tmp_path / "no_such_character.blend"
    res = export_animation(_motion(), tmp_path, name="x", fps=30,
                           render_video=False, character=str(missing))
    assert res.ok is False
    assert res.reason == "character_asset_missing"
    assert "no_such_character.blend" in res.message
    assert not list(tmp_path.glob("x.*"))


def test_an_unknown_schedule_is_refused(tmp_path):
    """`schedule` used to be normalised by `"stepped" if s == "stepped" else
    "one_per_pose"`, so a typo silently selected the default — which in a
    phase about never substituting one output for another is the wrong
    direction to fail in."""
    with pytest.raises(ValueError) as e:
        export_animation(_motion(), tmp_path, name="z", fps=30,
                         render_video=False, schedule="one-per-pose")
    assert "one_per_pose" in str(e.value)
    assert not list(tmp_path.glob("z.*"))


def test_the_substitute_export_needs_an_explicit_opt_in(tmp_path):
    """The fallback still exists — it is simply no longer the default answer to
    an exception. Asking for it is a decision the caller makes."""
    missing = tmp_path / "gone.blend"
    res = export_animation(_motion(), tmp_path, name="y", fps=30,
                           render_video=False, character=str(missing),
                           allow_fallback=True, blender="/nonexistent/blender")
    # it got as far as trying to run Blender instead of refusing outright,
    # and the result says out loud that what it would have written is not the
    # character — the caller has to decide what to tell the user
    assert res.reason == "blender_missing"
    assert res.substituted is True

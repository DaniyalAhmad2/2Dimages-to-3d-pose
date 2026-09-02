"""Phase 5 verification: Blender headless export produces BVH/FBX(/mp4).

Skipped automatically if the Blender binary is not present. Uses a short
synthetic 5-frame motion.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.config import blender_binary, character_blend
from pose3d.export.blender_export import export_animation
from tests.gates import needs_blender, needs_character, needs_video_render
from tests.synth import sample_skeleton_3d

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

# The largest non-hips position channel in the delivered file, as a % of rig
# height: an orphan IK helper bone carries the root motion that `to_rig` refuses
# to give the hips. Phase 3 gives the hips real root motion and pins the helper
# bones at rest; set this to 1.0 then and the assertion below is Phase 3's gate
# unchanged.
MAX_NON_HIPS_TRANSLATION_PCT = 42.5   # today 42.49 %, on shin.R.001


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
    """The stop-motion hold must be a hold: 21 frames of the same numbers, not
    a slow drift that reads as the figure breathing."""
    from tests import bvh_util
    for i, (first, last) in enumerate(bvh_util.stepped_holds(DELIVERED_POSES)):
        block = delivered.motion[first:last + 1]
        assert last - first == bvh_util.HOLD
        assert np.array_equal(block, np.repeat(block[:1], len(block), axis=0)), \
            f"pose {i}'s hold drifts by {np.abs(block - block[0]).max()}"


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


def test_a_helper_bone_carries_the_root_motion(delivered):
    """The translation the hips do not get has to go somewhere: it lands on an
    orphan IK helper bone, which is not motion any importer will use."""
    ranges = {}
    for j in delivered.joints:
        p = delivered.positions(j.name)
        if p.shape[1] == 3 and j.name != "hips":
            ranges[j.name] = float(np.max(p.max(0) - p.min(0)))
    name, worst = max(ranges.items(), key=lambda kv: kv[1])
    pct = 100.0 * worst / RIG_HEIGHT
    assert pct <= MAX_NON_HIPS_TRANSLATION_PCT, f"{name} carries {pct:.1f} %"
    if MAX_NON_HIPS_TRANSLATION_PCT > 1.0:
        # still today's file, so also pin the fact rather than just its bound
        assert name == "shin.R.001"
        assert pct >= 0.99 * MAX_NON_HIPS_TRANSLATION_PCT, f"{pct:.2f} %"


@needs_character()
def test_the_recorded_rig_height_is_still_the_bundled_rigs():
    """RIG_HEIGHT above is a number, not a measurement, so say out loud which
    rig it came from and fail if that rig is replaced."""
    from pose3d.geometry.character import Character
    assert Character().rig_h == pytest.approx(RIG_HEIGHT, abs=0.001)

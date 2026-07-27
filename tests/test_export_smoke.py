"""Phase 5 verification: Blender headless export produces BVH/FBX(/mp4).

Skipped automatically if the Blender binary is not present. Uses a short
synthetic 5-frame motion.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.config import blender_binary
from pose3d.export.blender_export import export_animation
from tests.synth import sample_skeleton_3d

_BLENDER = blender_binary()
_HAVE_BLENDER = Path(_BLENDER).exists()


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


@pytest.mark.skipif(not _HAVE_BLENDER, reason="Blender binary not found")
def test_export_bvh_fbx(tmp_path):
    res = export_animation(_motion(), tmp_path, name="t", fps=30,
                           render_video=False, timeout=300)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    assert res.bvh and res.bvh.stat().st_size > 0
    assert res.fbx and res.fbx.stat().st_size > 0


from pose3d.config import character_blend
_CHARACTER = character_blend()


@pytest.mark.skipif(not (_HAVE_BLENDER and _CHARACTER),
                    reason="bundled character not present")
def test_export_retargets_character(tmp_path):
    res = export_animation(_motion(), tmp_path, name="c", fps=24,
                           render_video=True, timeout=500, character=_CHARACTER)
    assert res.ok, f"rc={res.returncode}\nSTDERR:\n{res.stderr[-2000:]}"
    # a real skinned character FBX is much larger than a stick figure
    assert res.fbx and res.fbx.stat().st_size > 100_000
    import cv2
    cap = cv2.VideoCapture(str(res.mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, frame = cap.read(); cap.release()
    assert ok and float(frame.std()) > 3.0   # posed character is visible


@pytest.mark.skipif(not (_HAVE_BLENDER and _CHARACTER),
                    reason="bundled character not present")
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


@pytest.mark.skipif(not _HAVE_BLENDER, reason="Blender binary not found")
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

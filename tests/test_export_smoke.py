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


_CHARACTER = "/home/athena/Downloads/65-lowpolyboy/final low poly character  rigged.blend"


@pytest.mark.skipif(not (_HAVE_BLENDER and Path(_CHARACTER).exists()),
                    reason="character .blend not present")
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

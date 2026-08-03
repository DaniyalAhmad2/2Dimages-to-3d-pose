"""Skip locally, fail in CI.

Blender, the character rig and the ONNX weights are all optional on a developer
machine, so the tests that need them are skipped when they are absent. On a CI
runner that is exactly wrong: the entire export path — the product's output —
would report green having never run. These gates skip by default and fail when
the corresponding POSE3D_REQUIRE_* variable is set, which the Windows workflows
do after installing each dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _gate(available: bool, env: str, what: str):
    if available:
        return pytest.mark.skipif(False, reason="")
    if os.environ.get(env) == "1":
        # Turned into a failure by pytest_runtest_setup in conftest.py; a mark
        # cannot fail a test by itself.
        return pytest.mark.gate_missing(what=what, env=env)
    return pytest.mark.skipif(
        True, reason=f"{what} — set {env}=1 to make this a failure instead")


def _have_blender() -> bool:
    from pose3d.config import blender_binary
    exe = blender_binary()
    if Path(exe).is_file():
        return True
    import shutil
    return shutil.which(exe) is not None


def needs_blender():
    return _gate(_have_blender(), "POSE3D_REQUIRE_BLENDER",
                 "Blender binary not found")


def needs_character():
    from pose3d.config import character_blend
    return _gate(character_blend() is not None, "POSE3D_REQUIRE_ASSETS",
                 "bundled character asset missing")


def needs_video_render():
    """Rendering the preview mp4 drives EEVEE through whatever OpenGL the
    machine has, which is a much stronger requirement than writing BVH/FBX.

    A GPU-less Windows runner has no GL and no software fallback: Blender dies
    with EXCEPTION_ACCESS_VIOLATION after failing to find WGL extensions. The
    container ships Mesa precisely so this works, and a developer machine has a
    real GPU, so this runs by default and only the environments that genuinely
    cannot render opt out with POSE3D_NO_VIDEO=1.
    """
    return _gate(os.environ.get("POSE3D_NO_VIDEO") != "1",
                 "POSE3D_REQUIRE_VIDEO",
                 "POSE3D_NO_VIDEO=1: no OpenGL for headless Blender here")


def needs_weights(mode: str = "balanced"):
    from pose3d.detect import models
    return _gate(models.resolve(mode=mode) is not None,
                 "POSE3D_REQUIRE_WEIGHTS",
                 f"{mode} ONNX pose weights not staged")

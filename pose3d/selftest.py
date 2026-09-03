"""Prove an assembled build actually works, before a client sees it.

The container and the Windows bundle are put together by completely different
machinery, but they can fail in exactly the same ways: Blender not found or too
old, the character rig missing from the bundle, pose weights absent so the first
detection tries to download, no usable OpenGL for the 3D view, or an export that
dies inside Blender. Checking both deliveries with one piece of code is the only
way those checks stay in step.

    python -m pose3d.selftest        # a checkout or the container
    Pose3D.exe --selftest            # the shipped bundle

Exit status is 0 only if every required check passed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from pose3d.config import MIN_BLENDER_VERSION, blender_binary, character_blend
from pose3d.runtime import IS_FROZEN, IS_WINDOWS, in_container, subprocess_kwargs

#: A short standing pose, and the same pose with one arm raised, so the export
#: has actual movement to bake rather than two identical frames.
_POSE = np.array([
    [0.00, 0.00, 1.70], [0.00, 0.00, 1.50], [-0.18, 0.00, 1.48],
    [0.18, 0.00, 1.48], [-0.20, 0.02, 1.20], [0.20, 0.02, 1.20],
    [-0.22, 0.05, 0.95], [0.22, 0.05, 0.95], [0.00, 0.00, 0.95],
    [-0.10, 0.00, 0.95], [0.10, 0.00, 0.95], [-0.11, 0.02, 0.52],
    [0.11, 0.02, 0.52], [-0.12, 0.03, 0.08], [0.12, 0.03, 0.08]])


class Skip(Exception):
    """Raised by a check that does not apply to this build."""


class Degraded(Exception):
    """The build works, but something optional is missing. Not a failure."""


# --- the checks ------------------------------------------------------------

def check_blender() -> str:
    exe = blender_binary()
    out = subprocess.run([exe, "--version"], capture_output=True, timeout=180,
                         **subprocess_kwargs()).stdout
    ver = (out or "").splitlines()[0].strip()
    parts = ver.split()[1].split(".")
    found = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    if found < MIN_BLENDER_VERSION:
        raise AssertionError(
            f"export needs Blender {MIN_BLENDER_VERSION[0]}.x or newer, "
            f"found {ver} at {exe}")
    return f"{ver}  ({exe})"


def check_character_rig() -> str:
    from pose3d.geometry.character import Character
    blend = character_blend()
    if not (blend and Path(blend).exists()):
        raise AssertionError("character.blend is missing from this build")
    ch = Character()                      # loads character.npz alongside it
    return f"{len(ch.verts0)} verts, {len(ch.rest)} bones"


def check_pose_weights_offline() -> str:
    """The client may have no network, and a windowed build cannot survive
    rtmlib's download path at all — it writes progress to a stderr that does
    not exist there.

    BOTH pose models are required, whichever one the app is currently set to
    detect with (detect.rtmpose.USE_HALPE26). Checking only one of them is how
    the Halpe-26 checkpoint came to be missing from every bundle ever built,
    and requiring both means flipping that switch needs no build change.
    """
    from pose3d.detect import models
    found = []
    for feet in (True, False):
        w = models.resolve(feet=feet)
        if w is None:
            looked = "\n".join(f"    {d}" for d in models.search_dirs())
            want = ", ".join(models.required_files(feet=feet))
            raise AssertionError(
                f"the {'Halpe-26' if feet else 'COCO-17'} pose weights are not "
                "in this build, so the first detection would try to download "
                f"them ({want}). Looked in:\n{looked}")
        found.append(w)
    return (f"{Path(found[0].pose).name} + {Path(found[1].pose).name} "
            f"+ {Path(found[0].det).name}")


def check_qt_opengl() -> str:
    """The 3D view is a GLViewWidget, so it needs a real GL context — the
    single most likely thing to be missing from a frozen or headless build."""
    if not os.environ.get("DISPLAY") and not IS_WINDOWS and not sys.platform == "darwin":
        raise Skip("no DISPLAY")

    # Two different failures live here, and conflating them is what makes this
    # check either useless or unpassable. An import error means the 3D view is
    # missing from the build — a packaging defect, always fatal. A context or
    # draw-call error means this machine has no usable OpenGL, which is a
    # property of the hardware, not the build: a GPU-less CI runner cannot
    # provide one at all. POSE3D_NO_GL says so explicitly, and only then does
    # the second kind degrade. On the client's machine, where `--selftest` is
    # the thing that diagnoses a black 3D view, it stays fatal.
    try:
        from PySide6.QtWidgets import QApplication

        from pose3d.core.skeleton import NUM_JOINTS
        from pose3d.ui.view3d import View3D
    except ImportError as e:
        raise AssertionError(f"the 3D view is missing from this build: {e}") from e

    try:
        app = QApplication.instance() or QApplication([])
        v = View3D()
        v.resize(320, 240)
        v.show()
        app.processEvents()
        pose = np.zeros((NUM_JOINTS, 3))
        pose[:, 2] = np.linspace(0, 1.7, NUM_JOINTS)
        v.set_pose(pose)
        app.processEvents()

        # Constructing the widget is not proof of anything: Qt reports "Failed
        # to create context" on stderr and carries on, so a build with no
        # usable GL reaches this line looking healthy. Read back a real frame.
        if not v.isValid() or v.context() is None:
            raise AssertionError(
                "the 3D view has no OpenGL context — it would render nothing")
        img = v.grabFramebuffer()
        if img.isNull() or img.width() < 1 or img.height() < 1:
            raise AssertionError("the 3D view produced no frame")
    except Exception as e:
        if os.environ.get("POSE3D_NO_GL") == "1":
            raise Degraded(
                f"no usable OpenGL on this machine: {type(e).__name__}: "
                f"{str(e).splitlines()[0] if str(e) else ''}") from e
        raise
    return f"GL context valid, rendered {img.width()}x{img.height()}"


def _export(render_video: bool, timeout: int):
    from pose3d.core.skeleton import Joint
    from pose3d.export.blender_export import export_animation
    moved = _POSE.copy()
    moved[int(Joint.LEFT_WRIST)] = [-0.30, 0.0, 1.75]
    out = Path(tempfile.mkdtemp(prefix="pose3d-selftest-"))
    res = export_animation(np.stack([_POSE, moved]), out, name="selftest",
                           fps=24, render_video=render_video, timeout=timeout)
    sizes = {e: (out / f"selftest.{e}").stat().st_size
             for e in ("bvh", "fbx", "mp4") if (out / f"selftest.{e}").exists()}
    return res, sizes


def check_export() -> str:
    """The animation data itself — what the client imports. Must work."""
    res, sizes = _export(render_video=False, timeout=600)
    if not res.ok:
        raise AssertionError(
            f"rc={res.returncode}\n{(res.stderr or res.stdout or '')[-1500:]}")
    missing = {"bvh", "fbx"} - set(sizes)
    if missing:
        raise AssertionError(f"export produced no {', '.join(sorted(missing))} "
                             f"(got {sorted(sizes)})")
    return ", ".join(f"{k} {v // 1024}KB" for k, v in sorted(sizes.items()))


def check_video() -> str:
    """The preview mp4 — nice to have, and the one thing that legitimately
    fails on a machine without usable OpenGL.

    Kept as a separate Blender run rather than folded into the export above,
    because the two have completely different risk: writing BVH and FBX is
    arithmetic and takes seconds, while rendering drives EEVEE through whatever
    GL the machine has. The container ships Mesa for exactly this; a GPU-less
    CI runner has neither a GPU nor software GL, and on a busy machine llvmpipe
    can miss any timeout you pick. Splitting them means a slow or absent
    renderer downgrades the report instead of burying the fact that the
    animation data was fine.
    """
    res, sizes = _export(render_video=True, timeout=900)
    if not res.ok:
        # The export has one launch path with stderr folded into stdout, so the
        # reason a render failed usually arrives on stdout and stderr is empty.
        detail = res.stderr or res.stdout or ""
        raise Degraded(f"no preview video: rc={res.returncode} "
                       f"{detail.splitlines()[0] if detail else ''}")
    if "mp4" not in sizes:
        raise Degraded("Blender rendered no mp4")
    return f"mp4 {sizes['mp4'] // 1024}KB"


# --- driver ----------------------------------------------------------------

def checks(video: bool = True) -> list[tuple[str, object]]:
    """(name, callable) — in the order a first run depends on them."""
    out = [
        ("Blender present and new enough", check_blender),
        ("character rig assets", check_character_rig),
        ("pose weights available offline", check_pose_weights_offline),
        ("Qt + OpenGL 3D view", check_qt_opengl),
        ("export BVH + FBX", check_export),
    ]
    if video:
        out.append(("render preview video", check_video))
    return out


def run(video: bool = True, out=None, strict: bool | None = None) -> int:
    """strict: treat a degraded check as a failure.

    The container ships Mesa specifically so headless Blender can render, so
    there a missing video is a broken image and docker/build.sh sets this. A
    Windows CI runner has no GPU and no software GL to fall back on, so there
    the same result is expected and must not block a release.
    """
    out = out or sys.stdout
    if strict is None:
        strict = os.environ.get("POSE3D_REQUIRE_VIDEO") == "1"
    where = "container" if in_container() else ("frozen" if IS_FROZEN else "source")
    print(f"Pose3D self-test — {sys.platform}, {where}, "
          f"python {sys.version.split()[0]}", file=out)

    failed, skipped, degraded = [], [], []
    for name, fn in checks(video):
        try:
            detail = fn()
            print(f"  PASS  {name}" + (f" — {detail}" if detail else ""), file=out)
        except Skip as e:
            skipped.append(name)
            print(f"  SKIP  {name} — {e}", file=out)
        except Degraded as e:
            (failed if strict else degraded).append(name)
            print(f"  {'FAIL' if strict else 'WARN'}  {name} — {e}", file=out)
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name} — {type(e).__name__}: {e}", file=out)

    print(file=out)
    if degraded:
        print(f"Degraded (not fatal): {', '.join(degraded)}", file=out)
    if skipped:
        print(f"Skipped: {', '.join(skipped)}", file=out)
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=out)
        return 1
    print("All required checks passed.", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    strict = True if "--require-video" in argv else None
    return run(video="--no-video" not in argv, strict=strict)


if __name__ == "__main__":
    raise SystemExit(main())

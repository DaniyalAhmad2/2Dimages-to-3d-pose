"""Prove an assembled build actually works, before a client sees it.

The container and the Windows bundle are put together by completely different
machinery, but they can fail in exactly the same ways: Blender not found or too
old, the character rig missing from the bundle, pose weights absent so the first
detection tries to download, no usable OpenGL for the 3D view, or an export that
dies inside Blender. Checking both deliveries with one piece of code is the only
way those checks stay in step.

    python -m pose3d.selftest        # a checkout or the container
    Pose3D.exe --selftest            # the shipped bundle (writes to the log)
    Pose3D-diagnose.exe --selftest   # the same, in a console the client sees

Exit status is 0 only if every required check passed.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from pose3d.config import MIN_BLENDER_VERSION, blender_binary, character_blend
from pose3d.runtime import (
    IS_FROZEN, IS_WINDOWS, app_dir, in_container, subprocess_kwargs)

#: A frame is "drawn" when this fraction of its pixels differ from the view's
#: background by more than GL_DIFF_CHANNEL in some channel. Small on purpose:
#: the figure occupies ~10 % of the readback on a working machine and the
#: grid alone is well over 1 %, so the bar only has to separate "something was
#: drawn" from "the buffer was cleared and handed back".
GL_DRAWN_FRACTION = 0.001
GL_DIFF_CHANNEL = 8

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
    """Which Blender the export would run, and whether it is new enough.

    The version is searched for anywhere in the output rather than taken as
    the second word of the first line. Blender prints warnings before it
    (a missing display, a locale it cannot set), and a `blender` that is not
    Blender at all prints something else entirely — both of which used to end
    this check with an IndexError, which reads as a bug in the self-test
    rather than as the wrong executable being on the path.
    """
    exe = blender_binary()
    out = subprocess.run([exe, "--version"], capture_output=True, timeout=180,
                         **subprocess_kwargs()).stdout or ""
    m = re.search(r"Blender\s+(\d+)\.(\d+)", out)
    if m is None:
        raise AssertionError(
            f"{exe} --version did not print a Blender version. It printed:\n"
            f"{out.strip()[:500] or '(nothing)'}")
    # the whole line the version sits on, so the report keeps the patch
    # level and the build string the match itself drops
    ver = out[out.rfind("\n", 0, m.start()) + 1:].splitlines()[0].strip()
    found = (int(m.group(1)), int(m.group(2)))
    if found < MIN_BLENDER_VERSION:
        raise AssertionError(
            f"export needs Blender {MIN_BLENDER_VERSION[0]}.x or newer, "
            f"found {ver} at {exe}")
    return f"{ver}  ({exe})"


def check_bundle_layout() -> str:
    """Is this folder still the bundle we shipped? `integrity.problems()`.

    The same inventory the app checks before Qt starts, run here so the
    release gate answers the question once, in CI, with the manifest that will
    actually be inside the delivered zip. A frozen build that cannot find its
    manifest FAILS rather than skips: that build checks nothing at startup,
    and a check that fails open on the client's machine is the silence this
    whole change exists to remove.
    """
    from pose3d import integrity
    if not IS_FROZEN:
        raise Skip("not a frozen bundle")
    manifest = Path(integrity.MANIFEST)
    if not manifest.is_file():
        raise AssertionError(
            f"this build has no bundle inventory at {manifest}, so the "
            "startup integrity check silently checks nothing. It ships as "
            "_internal/packaging/windows/manifest.json (pose3d.spec datas).")
    found = integrity.problems()
    if found:
        raise AssertionError("\n" + "\n".join(f"  {p}" for p in found))
    return f"matches {manifest}"


def _qt_plugin_dir() -> Path:
    """Where Qt loads its plugins from in THIS build (frozen or not)."""
    from PySide6.QtCore import QLibraryInfo
    return Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))


def _qt_image_formats() -> set[str]:
    from PySide6.QtGui import QImageReader
    return {bytes(f).decode("ascii", "replace").lower()
            for f in QImageReader.supportedImageFormats()}


def check_qt_plugins() -> str:
    """Qt's plugins are loaded by name at run time, so no import scan sees them.

    Two of them decide whether the app runs at all. `platforms/qwindows.dll`
    is loaded by the QApplication constructor, which aborts the process when
    it is missing — no Qt dialog can report that, and nothing in the bundle
    audit looked for it. `imageformats/qjpeg.dll` decides whether the client's
    photographs are pictures or grey rectangles.
    """
    plugins = _qt_plugin_dir()
    platforms = plugins / "platforms"
    if IS_WINDOWS:
        qwindows = platforms / "qwindows.dll"
        if not qwindows.is_file():
            raise AssertionError(
                f"the Qt platform plugin is missing: {qwindows}\n"
                "Without it the QApplication constructor aborts with "
                '"could not find or load the Qt platform plugin windows".')
        where = str(qwindows)
    else:
        names = sorted(p.name for p in platforms.glob("*") if p.is_file())
        if not names:
            raise AssertionError(f"no Qt platform plugin in {platforms}")
        where = f"{platforms} ({len(names)} platform plugins)"
    formats = _qt_image_formats()
    missing = sorted({"jpg", "png"} - formats)
    if missing:
        raise AssertionError(
            f"Qt cannot read {', '.join(missing)} in this build, so the "
            "imported photographs would not display. Found: "
            f"{', '.join(sorted(formats)) or '(no image formats at all)'}")
    return f"{where}; jpg + png readable"


def check_inference() -> str:
    """Run one real detection, on a synthetic frame.

    Every other check about detection is about files being present. This one
    loads both ONNX sessions through onnxruntime and runs the person detector
    over a 640x480 image, which is what makes a bundle without onnxruntime —
    or with an onnxruntime that cannot load its own provider DLLs —
    unshippable rather than discovered by the client on their first import.

    No person is planted in the frame and none is expected: what is being
    proved is that the models load, run and answer in the shape the pipeline
    consumes. A checkout with no weights staged skips; a frozen bundle with
    none fails, and does so WITHOUT constructing the detector, which would
    otherwise fall straight through to rtmlib's downloader.
    """
    try:
        import onnxruntime
    except ImportError as e:
        raise AssertionError(
            f"onnxruntime is not in this build, so nothing can detect: {e}"
        ) from e
    from pose3d.core.skeleton import NUM_JOINTS
    from pose3d.detect import models
    from pose3d.detect.rtmpose import USE_HALPE26, RTMPoseDetector

    if models.resolve(mode="balanced", feet=USE_HALPE26) is None:
        if not IS_FROZEN:
            raise Skip("no pose weights staged in this checkout")
        raise AssertionError(
            "the balanced pose weights are not in this bundle, so detection "
            "would try to download them (see the weights check above)")

    started = time.monotonic()
    det = RTMPoseDetector(mode="balanced", device="cpu")
    if det.bundled is not True:
        raise AssertionError(
            "the detector did not use the bundled weights — it would download "
            "them on the client's machine")
    image = np.zeros((480, 640, 3), np.uint8)
    image[:, :, 1] = np.linspace(0, 255, 640, dtype=np.uint8)   # not uniform
    result = det.detect(image)
    if (result.xy.shape != (NUM_JOINTS, 2)
            or result.scores.shape != (NUM_JOINTS,)):
        raise AssertionError(
            f"detection returned the wrong shape: xy {result.xy.shape}, "
            f"scores {result.scores.shape}, expected ({NUM_JOINTS}, 2) and "
            f"({NUM_JOINTS},)")
    return (f"onnxruntime {onnxruntime.__version__} "
            f"[{', '.join(onnxruntime.get_available_providers())}], "
            f"one 640x480 frame in {time.monotonic() - started:.1f} s")


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


def _background(view) -> np.ndarray:
    """The colour the 3D view clears to, as RGB 0-255.

    pyqtgraph keeps it in `opts['bgcolor']` as floats. Falling back to the
    frame's own top-left pixel rather than to black keeps the comparison
    honest if that key is ever renamed: a cleared buffer still matches its own
    corner, so the check would go red, not silently green.
    """
    rgba = getattr(view, "opts", {}).get("bgcolor")
    if rgba is None:
        return np.array([-1, -1, -1])            # sampled by _drawn_fraction
    return np.array([int(round(float(c) * 255)) for c in rgba[:3]])


def _drawn_fraction(image, background) -> float:
    """How much of a readback is not the background."""
    from PySide6.QtGui import QImage
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    h, w, stride = rgb.height(), rgb.width(), rgb.bytesPerLine()
    flat = np.frombuffer(rgb.constBits(), dtype=np.uint8)[:stride * h]
    pixels = flat.reshape(h, stride)[:, :w * 3].reshape(h, w, 3).astype(np.int16)
    if background[0] < 0:
        background = pixels[0, 0]
    diff = np.abs(pixels - np.asarray(background, dtype=np.int16)).max(axis=2)
    return float((diff > GL_DIFF_CHANNEL).mean())


def diagnose_exe() -> Path:
    """The console-mode twin of the app, built from the same Analysis.

    The windowed exe writes its output into pose3d-log.txt, where a client
    running --selftest sees nothing at all; this one prints to a real console.
    """
    return app_dir() / ("Pose3D-diagnose.exe" if IS_WINDOWS
                        else "pose3d-diagnose")


def _software_gl_child() -> tuple[bool | None, str]:
    """Re-run JUST the GL check in one child process forced onto software GL.

    (True | False | None, detail). None means the child could not be run at
    all, which is "software path untested" and never "software GL also
    failed" — the second is a diagnosis we would not have made, and it is the
    one that sends the client shopping for a graphics card.

    A child, not a retry in this process, and that is not a preference: Qt
    reads AA_UseSoftwareOpenGL when the QApplication is constructed and
    ignores it afterwards, and pyqtgraph draws through PyOpenGL's system
    opengl32.dll rather than Qt's opengl32sw.dll, so nothing this process can
    do after the fact changes which GL it is using.
    """
    if (os.environ.get("QT_OPENGL") == "software"
            or os.environ.get("POSE3D_GL", "").lower() == "software"):
        return None, "this process is already running with software OpenGL"
    if IS_FROZEN:
        cmd = [str(diagnose_exe()), "--selftest", "--gl-only", "--no-video"]
    else:
        cmd = [sys.executable, "-m", "pose3d.selftest", "--gl-only"]
    env = dict(os.environ, QT_OPENGL="software", POSE3D_GL="software")
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=300, env=env,
                             **subprocess_kwargs())
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"could not run {cmd[0]}: {type(e).__name__}: {e}"
    out = ((res.stdout or "") + (res.stderr or "")).strip()
    return res.returncode == 0, (out.splitlines()[-1] if out
                                 else f"rc={res.returncode}")


def _gl_failure(exc: Exception) -> Exception:
    """What to raise when the hardware attempt did not produce a frame.

    Fatal by default: on the client's machine `--selftest` is the thing that
    diagnoses a black 3D view. POSE3D_NO_GL only downgrades the verdict when
    software OpenGL failed too, because that is the GPU-less CI runner — a
    machine where the software path DID render is a working install with the
    wrong flag, and saying so is the whole point of spawning the child.
    """
    first = str(exc).splitlines()[0] if str(exc) else ""
    hardware = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    ok, detail = _software_gl_child()
    if ok is True:
        return Degraded(
            f"hardware OpenGL rendered nothing ({hardware}), but software "
            "OpenGL did. Start Pose3D with --software-gl, or use \"Restart "
            "with software 3D\" in the 3D panel")
    both = (f"no usable OpenGL on this machine: {hardware}; "
            + (f"software OpenGL also rendered nothing ({detail})" if ok is False
               else f"the software path is untested ({detail})"))
    if os.environ.get("POSE3D_NO_GL") == "1":
        return Degraded(both)
    return AssertionError(both)


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
    # provide one at all, and the software-GL child is what tells those apart.
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

        # And a frame is not a drawing. A driver that hands back the cleared
        # buffer passes every test above it while the client looks at a black
        # 3D card, which is exactly the report we got.
        drawn = _drawn_fraction(img, _background(v))
        if drawn <= GL_DRAWN_FRACTION:
            raise AssertionError(
                f"the 3D view returned a frame with nothing drawn in it "
                f"({drawn * 100:.3f} % of pixels differ from the background): "
                "the context exists, the driver renders nothing")
    except Exception as e:
        raise _gl_failure(e) from e
    return (f"GL context valid, rendered {img.width()}x{img.height()}, "
            f"{drawn * 100:.1f} % of the frame drawn")


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
        ("bundle matches its inventory", check_bundle_layout),
        ("Blender present and new enough", check_blender),
        ("character rig assets", check_character_rig),
        ("pose weights available offline", check_pose_weights_offline),
        ("detection actually runs", check_inference),
        ("Qt plugins", check_qt_plugins),
        ("Qt + OpenGL 3D view", check_qt_opengl),
        ("export BVH + FBX", check_export),
    ]
    if video:
        out.append(("render preview video", check_video))
    return out


def run(video: bool = True, out=None, strict: bool | None = None,
        gl_only: bool = False) -> int:
    """strict: treat a degraded check as a failure.

    The container ships Mesa specifically so headless Blender can render, so
    there a missing video is a broken image and docker/build.sh sets this. A
    Windows CI runner has no GPU and no software GL to fall back on, so there
    the same result is expected and must not block a release.

    gl_only: run nothing but the 3D check. This is what the software-OpenGL
    child process is asked to do, so it must not start Blender to answer it.
    """
    out = out or sys.stdout
    if strict is None:
        strict = os.environ.get("POSE3D_REQUIRE_VIDEO") == "1"
    where = "container" if in_container() else ("frozen" if IS_FROZEN else "source")
    print(f"Pose3D self-test — {sys.platform}, {where}, "
          f"python {sys.version.split()[0]}", file=out)

    todo = checks(video)
    if gl_only:
        todo = [(name, fn) for name, fn in todo if fn is check_qt_opengl]

    failed, skipped, degraded = [], [], []
    for name, fn in todo:
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
    return run(video="--no-video" not in argv, strict=strict,
               gl_only="--gl-only" in argv)


if __name__ == "__main__":
    raise SystemExit(main())

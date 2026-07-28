"""In-container smoke test: everything the client's first run depends on.

Run with:
    docker run --rm --entrypoint python pose3d:latest /app/docker/smoketest.py

Checks the pieces that fail silently or look like app bugs if they are wrong:
Blender version, the character rig assets, offline pose weights, the Qt/OpenGL
stack the 3D view needs, and a real end-to-end export.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

FAIL = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name} — {type(e).__name__}: {e}")


def blender():
    from pose3d.config import blender_binary
    exe = blender_binary()
    out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                         timeout=120).stdout
    ver = out.splitlines()[0].strip()
    major = int(ver.split()[1].split(".")[0])
    assert major >= 5, f"export needs Blender 5.x, found {ver}"
    return ver


def character_assets():
    from pose3d.config import character_blend
    from pose3d.geometry.character import Character
    blend = character_blend()
    assert blend and Path(blend).exists(), "character.blend missing"
    ch = Character()                      # loads character.npz
    return f"{len(ch.verts0)} verts, {len(ch.rest)} bones"


def pose_weights_offline():
    """Weights must be on disk: the client may have no internet."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    ckpt = cache / "rtmlib" / "hub" / "checkpoints"
    found = sorted(p.name for p in ckpt.glob("*.onnx"))
    assert found, f"no ONNX weights under {ckpt}"
    return f"{len(found)} models in {ckpt}"


def qt_opengl():
    """The 3D view subclasses GLViewWidget, so a real GL context is required."""
    from PySide6.QtWidgets import QApplication
    from pose3d.ui.view3d import View3D
    app = QApplication.instance() or QApplication([])
    v = View3D()
    v.resize(320, 240)
    v.show()
    app.processEvents()
    import numpy as np
    from pose3d.core.skeleton import NUM_JOINTS
    pose = np.zeros((NUM_JOINTS, 3))
    pose[:, 2] = np.linspace(0, 1.7, NUM_JOINTS)
    v.set_pose(pose)
    app.processEvents()
    return "GLViewWidget created and rendered a pose"


def end_to_end_export():
    import numpy as np
    from pose3d.export.blender_export import export_animation
    from pose3d.core.skeleton import Joint
    base = np.array([
        [0.00, 0.00, 1.70], [0.00, 0.00, 1.50], [-0.18, 0.00, 1.48],
        [0.18, 0.00, 1.48], [-0.20, 0.02, 1.20], [0.20, 0.02, 1.20],
        [-0.22, 0.05, 0.95], [0.22, 0.05, 0.95], [0.00, 0.00, 0.95],
        [-0.10, 0.00, 0.95], [0.10, 0.00, 0.95], [-0.11, 0.02, 0.52],
        [0.11, 0.02, 0.52], [-0.12, 0.03, 0.08], [0.12, 0.03, 0.08]])
    b = base.copy()
    b[int(Joint.LEFT_WRIST)] = [-0.30, 0, 1.75]
    out = Path(tempfile.mkdtemp())
    res = export_animation(np.stack([base, b]), out, name="smoke", fps=24,
                           render_video=True, timeout=900)
    assert res.ok, f"rc={res.returncode}\n{(res.stderr or res.stdout)[-1500:]}"
    sizes = {e: (out / f"smoke.{e}").stat().st_size
             for e in ("bvh", "fbx", "mp4") if (out / f"smoke.{e}").exists()}
    assert {"bvh", "fbx", "mp4"} <= set(sizes), f"missing outputs, got {sizes}"
    return ", ".join(f"{k} {v//1024}KB" for k, v in sizes.items())


print("Pose3D container smoke test")
print(f"  python {sys.version.split()[0]}  display={os.environ.get('DISPLAY')}")
check("Blender 5.x present", blender)
check("character rig assets", character_assets)
check("pose weights available offline", pose_weights_offline)
check("Qt + OpenGL 3D view", qt_opengl)
check("end-to-end export (BVH/FBX/MP4)", end_to_end_export)

print()
if FAIL:
    print(f"FAILED: {', '.join(FAIL)}")
    sys.exit(1)
print("All checks passed.")

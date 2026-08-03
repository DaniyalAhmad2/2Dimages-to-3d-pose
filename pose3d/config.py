"""App-wide configuration defaults (paths, external tools)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from pose3d.runtime import IS_WINDOWS, app_dir, exe_name

# Export needs Blender 5.x: the video writer sets image_settings.media_type,
# which does not exist in 4.x, and the BVH exporter signature changed too.
MIN_BLENDER_VERSION = (5, 0)


def character_blend() -> str | None:
    """The bundled rigged humanoid model used for all exports (fixed app asset)."""
    p = Path(__file__).parent / "assets" / "character.blend"
    return str(p) if p.exists() else None


def _bundled_blender_dirs() -> list[Path]:
    """Places a Blender shipped WITH the app could live.

    The Windows bundle puts it in a `blender/` folder beside the executable;
    the container installs it at /opt/blender.
    """
    out = [app_dir() / "blender", app_dir() / "blender" / "blender"]
    if not IS_WINDOWS:
        out.append(Path("/opt/blender"))
    return out


def _installed_blender_dirs() -> list[Path]:
    """Standard install locations, since Blender adds itself to neither PATH
    (Windows) nor any predictable prefix."""
    if not IS_WINDOWS:
        return []
    out: list[Path] = []
    for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(var)
        if not root:
            continue
        base = Path(root) / "Blender Foundation"
        if base.is_dir():
            # newest first, so Blender 5.x wins over a leftover 4.x
            out.extend(sorted((p for p in base.glob("Blender *") if p.is_dir()),
                              reverse=True))
    return out


def blender_binary() -> str:
    """Resolve the Blender executable.

    Order: POSE3D_BLENDER -> a Blender shipped with the app -> a standard
    install -> PATH. Falls back to the bare name so the caller reports a
    clear "not found" rather than a path from whatever machine built this.
    """
    env = os.environ.get("POSE3D_BLENDER")
    if env and Path(env).is_file():
        return env
    name = exe_name("blender")
    for d in _bundled_blender_dirs() + _installed_blender_dirs():
        cand = Path(d) / name
        if cand.is_file():
            return str(cand)
    found = shutil.which(name) or shutil.which("blender")
    if found:
        return found
    return name


def blender_available() -> bool:
    exe = blender_binary()
    return Path(exe).is_file() or shutil.which(exe) is not None

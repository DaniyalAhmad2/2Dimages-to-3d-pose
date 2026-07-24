"""App-wide configuration defaults (paths, external tools)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Local Blender 5.1.1 install (client machine ships its own; override via env).
DEFAULT_BLENDER = "/home/athena/Downloads/blender-5.1.1-linux-x64/blender"


def character_blend() -> str | None:
    """Path to a rigged humanoid .blend to retarget for export, or None.

    Set via the POSE3D_CHARACTER env var or chosen in the app (Settings).
    """
    p = os.environ.get("POSE3D_CHARACTER")
    return p if (p and Path(p).exists()) else None


def blender_binary() -> str:
    """Resolve the Blender executable: env override -> default -> PATH."""
    env = os.environ.get("POSE3D_BLENDER")
    if env and Path(env).exists():
        return env
    if Path(DEFAULT_BLENDER).exists():
        return DEFAULT_BLENDER
    found = shutil.which("blender")
    if found:
        return found
    return DEFAULT_BLENDER  # let the caller surface a clear error if missing

"""App-wide configuration defaults (paths, external tools)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Local Blender 5.1.1 install (client machine ships its own; override via env).
DEFAULT_BLENDER = "/home/athena/Downloads/blender-5.1.1-linux-x64/blender"


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

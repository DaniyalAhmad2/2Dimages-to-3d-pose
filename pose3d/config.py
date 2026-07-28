"""App-wide configuration defaults (paths, external tools)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# Where a bundled Blender is expected to live when the app ships as a container
# or a self-contained archive. Kept machine-independent on purpose — set
# POSE3D_BLENDER to point at any other install.
BUNDLED_BLENDER_DIRS = (
    "/opt/blender",                                   # container image
    str(Path(__file__).resolve().parent.parent / "blender"),   # archive layout
)

# Export needs Blender 5.x: the video writer sets image_settings.media_type,
# which does not exist in 4.x, and the BVH exporter signature changed too.
MIN_BLENDER_VERSION = (5, 0)


def character_blend() -> str | None:
    """The bundled rigged humanoid model used for all exports (fixed app asset)."""
    p = Path(__file__).parent / "assets" / "character.blend"
    return str(p) if p.exists() else None


def blender_binary() -> str:
    """Resolve the Blender executable.

    Order: POSE3D_BLENDER -> a Blender bundled with the app -> PATH. Falls back
    to the bare name so the caller surfaces a clear "not found" error rather
    than a path from whatever machine built this.
    """
    env = os.environ.get("POSE3D_BLENDER")
    if env and Path(env).exists():
        return env
    for d in BUNDLED_BLENDER_DIRS:
        cand = Path(d) / "blender"
        if cand.exists():
            return str(cand)
    found = shutil.which("blender")
    if found:
        return found
    return "blender"


def blender_available() -> bool:
    return Path(blender_binary()).exists() or shutil.which(blender_binary()) is not None

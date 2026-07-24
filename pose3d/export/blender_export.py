"""Host-side Blender export driver.

Serializes a per-frame 3D pose sequence to JSON and invokes the local Blender
binary headless to produce BVH + FBX + mp4. Blender runs blender_job.py.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pose3d.config import blender_binary
from pose3d.core.skeleton import BONES, JOINT_NAMES, MIXAMO_BONE, NUM_JOINTS

_JOB = Path(__file__).with_name("blender_job.py")


@dataclass
class ExportResult:
    bvh: Path | None
    fbx: Path | None
    mp4: Path | None
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and "POSE3D_EXPORT_OK" in self.stdout


def _poses_to_json(poses3d: np.ndarray, fps: int) -> dict:
    """poses3d: (T, NUM_JOINTS, 3); NaN -> null."""
    poses3d = np.asarray(poses3d, float).reshape(-1, NUM_JOINTS, 3)
    frames = []
    for pose in poses3d:
        frames.append([
            (None if np.isnan(p).any() else [float(p[0]), float(p[1]), float(p[2])])
            for p in pose])
    return {
        "fps": int(fps),
        "joint_names": JOINT_NAMES,
        "bones": [[int(a), int(b)] for a, b in BONES],
        "mixamo_names": {str(int(j)): MIXAMO_BONE[j] for j in MIXAMO_BONE},
        "frames": frames,
    }


def export_animation(
    poses3d: np.ndarray,
    out_dir: str | Path,
    name: str = "pose3d",
    fps: int = 30,
    render_video: bool = True,
    blender: str | None = None,
    timeout: int = 600,
    display_frame: int = 0,
    character: str | None = "__bundled__",
) -> ExportResult:
    if character == "__bundled__":
        from pose3d.config import character_blend
        character = character_blend()          # always the bundled model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = _poses_to_json(poses3d, fps)
    doc["display_frame"] = int(display_frame)   # which pose the turntable spins
    json_path = out_dir / f"{name}_poses.json"
    json_path.write_text(json.dumps(doc))

    blender = blender or blender_binary()
    # if a rigged character .blend is given, open it as the base file so the job
    # can retarget it; otherwise run with an empty scene (skeleton figure).
    use_char = bool(character) and Path(character).exists()
    base = [character] if use_char else []
    cmd = [blender, "--background", *base, "--python", str(_JOB), "--",
           "--in", str(json_path), "--out", str(out_dir),
           "--fps", str(fps), "--name", name]
    if use_char:
        cmd += ["--character", str(character)]
    if not render_video:
        cmd.append("--no-video")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _exists(ext):
        p = out_dir / f"{name}.{ext}"
        return p if p.exists() else None

    return ExportResult(
        bvh=_exists("bvh"), fbx=_exists("fbx"),
        mp4=_exists("mp4") if render_video else None,
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)

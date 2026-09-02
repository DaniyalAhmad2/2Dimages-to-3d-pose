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
from pose3d.runtime import subprocess_kwargs
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


def _character_bone_frames(poses3d: np.ndarray, display_frame: int,
                           head3d: np.ndarray | None = None,
                           filled: np.ndarray | None = None):
    """Per-frame posed bone matrices, computed with the SAME skinning the live
    3D view uses, so the exported character matches the preview pose-for-pose.

    `filled` is the (T, NUM_JOINTS) flag array from `pipeline.fill_gaps`. The
    export reads the same array the 3D view draws from, so the two can never
    disagree about what a dropout is: a one-frame gap the view shows filled
    (amber) is posed here too, and a longer gap the view shows as a hole stays
    absent here — which is what keeps Blender's hold-the-last-known-pose rule
    from quietly papering over a dropout the app is telling the user about.

    Returns (bone_frames, bone_names) or (None, None) if the character asset is
    unavailable — the Blender job then falls back to its aim-only retarget.
    """
    try:
        from pose3d.geometry.character import Character
        from pose3d.geometry.orient import (sequence_up, de_tilt_matrix,
                                            detect_vertical, upright_matrix)
    except Exception:
        return None, None
    try:
        ch = Character()
    except Exception:
        return None, None

    poses3d = np.asarray(poses3d, float).reshape(-1, NUM_JOINTS, 3)
    # orient exactly as the 3D view does, so the export matches the preview
    # pose-for-pose (see orient.sequence_up for why the calibration frame is
    # not used as the vertical reference)
    up = sequence_up(poses3d)
    if up is not None:
        R = de_tilt_matrix(up).T
    else:
        # fallback: single-frame axis detection
        axis, sign = 2, 1.0
        order = [display_frame] + [i for i in range(len(poses3d)) if i != display_frame]
        for i in order:
            if 0 <= i < len(poses3d):
                v = ~np.isnan(poses3d[i]).any(1)
                if v.any():
                    axis, sign = detect_vertical(poses3d[i], v)
                    break
        R = upright_matrix(axis, sign).T

    # size the character to this subject once, from the whole take — the same
    # fit the 3D view applies, so the export stays identical to the preview
    ch.fit_to_subject(poses3d @ R)

    bone_frames = []
    for i, pose in enumerate(poses3d):
        valid = ~np.isnan(pose).any(1)
        if filled is not None:
            valid |= np.asarray(filled[i], bool) & ~np.isnan(pose).any(1)
        if not valid.any():
            bone_frames.append(None); continue
        up = pose @ R                       # upright; centring is irrelevant here
        # the face keypoints take the same rotation, so the exported head is
        # oriented exactly as the preview shows it
        hp = None if head3d is None else np.asarray(head3d[i], float) @ R
        bone_frames.append(ch.pose_bone_matrices(up, valid, hp))
    if all(b is None for b in bone_frames):
        return None, None
    return bone_frames, ch.bone_names


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
    head3d: np.ndarray | None = None,
    filled: np.ndarray | None = None,
    on_line=None,
) -> ExportResult:
    if character == "__bundled__":
        from pose3d.config import character_blend
        character = character_blend()          # always the bundled model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = _poses_to_json(poses3d, fps)
    doc["display_frame"] = int(display_frame)   # which pose the turntable spins
    if character and Path(character).exists():
        # drive the rig with the exact skinning the live view uses
        bone_frames, bone_names = _character_bone_frames(
            poses3d, display_frame, head3d, filled)
        if bone_frames is not None:
            doc["bone_frames"] = bone_frames
            doc["bone_names"] = bone_names
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

    # A frozen GUI has no usable stdin for the child to inherit, its output is
    # UTF-8 whatever the machine's code page says, and on Windows a console
    # child flashes a black window over the UI. See pose3d.runtime.
    kw = subprocess_kwargs()
    try:
        if on_line is None:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        else:
            # stream Blender's output so the caller can show live progress
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=1, **kw)
            chunks = []
            for line in p.stdout:
                chunks.append(line)
                try:
                    on_line(line.rstrip())
                except Exception:
                    pass
            p.wait(timeout=timeout)
            stdout, stderr, rc = "".join(chunks), "", p.returncode
    except subprocess.TimeoutExpired:
        return ExportResult(
            bvh=None, fbx=None, mp4=None, returncode=124, stdout="",
            stderr=(f"Blender did not finish within {timeout} s and was stopped.\n\n"
                    "A long take can legitimately take a while to render; try "
                    "exporting without the video, or a shorter selection."))
    except FileNotFoundError:
        # must precede OSError, of which it is a subclass: say which binary is
        # missing and how to point at one, instead of a bare OSError
        return ExportResult(
            bvh=None, fbx=None, mp4=None, returncode=127, stdout="",
            stderr=(f"Blender was not found (tried: {blender}).\n\n"
                    "Export needs Blender 5.x. Install it and either put it on "
                    "PATH or set the POSE3D_BLENDER environment variable to the "
                    "blender executable."))
    except OSError as e:
        # e.g. POSE3D_BLENDER pointing at a folder, or a non-executable file
        return ExportResult(
            bvh=None, fbx=None, mp4=None, returncode=126, stdout="",
            stderr=(f"Could not run Blender at '{blender}'.\n\n{type(e).__name__}: {e}\n\n"
                    "Set POSE3D_BLENDER to the blender executable itself."))

    def _exists(ext):
        p = out_dir / f"{name}.{ext}"
        return p if p.exists() else None

    return ExportResult(
        bvh=_exists("bvh"), fbx=_exists("fbx"),
        mp4=_exists("mp4") if render_video else None,
        returncode=rc, stdout=stdout, stderr=stderr)

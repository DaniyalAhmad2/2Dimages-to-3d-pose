"""Host-side Blender export driver.

Serializes a per-frame 3D pose sequence to JSON and invokes the local Blender
binary headless to produce BVH + FBX + mp4. Blender runs blender_job.py.
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pose3d.config import blender_binary
from pose3d.runtime import subprocess_kwargs
from pose3d.core.skeleton import BONES, JOINT_NAMES, MIXAMO_BONE, NUM_JOINTS

_JOB = Path(__file__).with_name("blender_job.py")

# The two keyframe schedules. "one_per_pose" is the default for the delivered
# files (export frame k+1 IS photograph k); "stepped" is the 0.7 s hold /
# 0.3 s ease stop-motion the previews use, and which with keep_root_motion off
# reproduces the file the client already has.
SCHEDULES = frozenset({"one_per_pose", "stepped"})


# Why an export produced nothing, as a value rather than a swallowed exception.
# `_character_bone_frames` used to answer every failure with `return None, None`,
# and the Blender job then quietly wrote a DIFFERENT animation (an aim-only
# Damped-Track retarget, or a null-rooted stick figure) that the host reported as
# a successful export. Each reason below is a thing the user can act on.
FAILURE_MESSAGES = {
    "character_asset_missing":
        "The character model this export needs was not found ({detail}).\n\n"
        "Nothing was written: the app will not quietly export a different "
        "figure in its place. Reinstall the app, or re-run the export with "
        "the fallback explicitly allowed.",
    "character_import_failed":
        "The character posing code could not be loaded ({detail}).\n\n"
        "Nothing was written rather than exporting a different figure.",
    "character_load_failed":
        "The character model could not be opened ({detail}).\n\n"
        "Nothing was written rather than exporting a different figure.",
    "fill_flags_disagree":
        "This take's gap-fill flags do not match its 3D ({detail}).\n\n"
        "A joint is flagged as interpolated but has no position, so the "
        "export cannot tell what it would be writing. Nothing was written. "
        "Recalculate 3D and export again.",
    "no_posable_frame":
        "No frame of this take could be posed onto the character ({detail}).\n\n"
        "This normally means the 3D reconstruction is empty; the 3D preview "
        "will be empty too.",
    "blender_timeout": "{detail}",
    "blender_cancelled": "{detail}",
    "blender_missing": "{detail}",
    "blender_not_runnable": "{detail}",
    "blender_failed": "{detail}",
}


# How often the pump wakes to look at the clock and the cancel flag while
# Blender is saying nothing. Small enough that Cancel feels immediate, large
# enough that a 10-minute render costs a few thousand wakeups.
_POLL_S = 0.25
# How long each escalation of _stop is given before the next one.
_STOP_GRACE_S = 5


def _stop(proc) -> None:
    """End `proc` and reap it: terminate, then kill, then (Windows) taskkill.

    Reaping matters as much as killing. A zombie holds the pipe open, and
    "and was stopped" in the message the user reads is only true once the
    process object has a returncode.
    """
    for end_it in (proc.terminate, proc.kill):
        try:
            end_it()
        except OSError:
            pass                      # already gone
        try:
            proc.wait(timeout=_STOP_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue
    if sys.platform.startswith("win"):
        # Blender spawns children of its own; /T takes the tree with it
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=_STOP_GRACE_S)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.wait(timeout=_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        pass


def _pump(proc, chunks: list, on_line, timeout, idle_timeout, cancelled) -> str:
    """Read `proc`'s merged output until it ends, or until we stop it.

    Returns "" when Blender finished on its own, else the reason it was
    stopped. A daemon thread does the blocking read — Windows has no `select`
    on a pipe, so a reader that could be interrupted does not exist — and this
    loop watches three clocks: the overall deadline, an IDLE deadline (nothing
    said for `idle_timeout`, which is what a hung render looks like), and the
    caller's cancel.
    """
    lines: queue.Queue = queue.Queue()

    def read():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)           # EOF, whatever happened

    reader = threading.Thread(target=read, daemon=True)
    reader.start()

    started = last = time.monotonic()
    stopped = ""
    while True:
        now = time.monotonic()
        if cancelled is not None and cancelled():
            stopped = "cancelled"
        elif timeout is not None and now - started > timeout:
            stopped = "timeout"
        elif idle_timeout is not None and now - last > idle_timeout:
            stopped = "timeout"
        if stopped:
            _stop(proc)
            break
        try:
            line = lines.get(timeout=_POLL_S)
        except queue.Empty:
            continue
        if line is None:
            break
        last = time.monotonic()
        chunks.append(line)
        if on_line is not None:
            try:
                on_line(line.rstrip())
            except Exception:
                pass                  # a broken progress display is not a
                                      # reason to lose the export

    # Whatever the child managed to say before it was stopped is still the
    # best diagnostic there is, so let the reader finish (the child is dead by
    # now, so its pipe is at EOF) and take everything it left behind.
    reader.join(timeout=_STOP_GRACE_S)
    while True:
        try:
            line = lines.get_nowait()
        except queue.Empty:
            break
        if line is None:
            continue
        chunks.append(line)
    if not reader.is_alive():
        try:
            proc.stdout.close()       # not while the reader still holds it
        except OSError:
            pass

    if not stopped:
        left = None if timeout is None else max(
            1.0, timeout - (time.monotonic() - started))
        try:
            proc.wait(timeout=left)   # reap; EOF is not quite exit
        except subprocess.TimeoutExpired:
            _stop(proc)
            stopped = "timeout"
    return stopped


def _timeout_detail(timeout, idle_timeout) -> str:
    """Why a stopped export was stopped, naming only the deadlines there are.

    Either may be None, which means there is no deadline of that kind — and
    `f"{None:g}"` is a TypeError, so the old message died while reporting the
    timeout instead of returning it.
    """
    if idle_timeout is not None and timeout is not None:
        what = (f"produced nothing for {idle_timeout:g} s (or ran past "
                f"{timeout:g} s)")
    elif idle_timeout is not None:
        what = f"produced nothing for {idle_timeout:g} s"
    elif timeout is not None:
        what = f"ran past {timeout:g} s"
    else:                                 # no deadline: only _stop's own wait
        what = "did not finish"
    return (f"Blender {what} and was stopped.\n\nA long take can legitimately "
            "take a while to render; try exporting without the video, or a "
            "shorter selection.")


def _failure(reason: str, detail: str, returncode: int,
             note: str = "", stdout: str = "") -> "ExportResult":
    """The failure as a value. `stdout` is whatever the child did manage to
    say, which is the only diagnostic a killed Blender leaves behind."""
    msg = FAILURE_MESSAGES.get(reason, "{detail}").format(detail=detail)
    return ExportResult(bvh=None, fbx=None, mp4=None, returncode=returncode,
                        stdout=stdout, stderr=msg, reason=reason, message=msg,
                        fallback_note=note)


@dataclass
class ExportResult:
    bvh: Path | None
    fbx: Path | None
    mp4: Path | None
    returncode: int
    stdout: str
    stderr: str
    # Why it failed, as one of FAILURE_MESSAGES' keys, and the sentence to show.
    # None on success.
    reason: str | None = None
    message: str = ""
    # The fixed-camera render (the left camera's own pose), when one was asked
    # for. `mp4` stays the turntable, so the existing contract is unchanged.
    mp4_camera: Path | None = None
    # A substitution the HOST decided on (a missing asset with allow_fallback),
    # which never reaches Blender's stdout and so cannot be read back from it.
    fallback_note: str = ""

    @property
    def ok(self) -> bool:
        return (self.returncode == 0 and "POSE3D_EXPORT_OK" in self.stdout
                and "POSE3D_EXPORT_FAILED" not in self.stdout)

    @property
    def preview_failed(self) -> bool:
        """The files were written; a preview video was not.

        A distinct outcome from a failed export, and it has to be: rendering
        needs ffmpeg and a usable GL stack that the BVH and FBX do not, and
        reporting "export failed" over a preview would have the user throw
        away two correct files.
        """
        return "POSE3D_EXPORT_PREVIEW_FAILED" in self.stdout

    @property
    def substituted(self) -> bool:
        """True when Blender wrote something OTHER than the posed character.

        Only reachable with `allow_fallback=True`: it is what the opt-in buys,
        and the caller is expected to say so rather than present the file as
        the character export.
        """
        return bool(self.fallback_note) or (
            "POSE3D_EXPORT_FALLBACK" in self.stdout)


def _rig_camera(camera, M_view, scale, pelvis_ref, hips_world):
    """The capture camera, expressed in the posed rig's space.

    `camera` is {"K", "R", "t", "image_size"} in the ArUco world frame, i.e.
    X_cam = R @ X_world + t (`calib.extrinsics.Extrinsics`). The character is
    placed by the same similarity every joint goes through — de-tilt, uniform
    scale, offset onto the take's pelvis — so the camera goes through it too
    and ends up looking at the figure from exactly where the photograph was
    taken. That is what makes the fixed-camera mp4 comparable to the client's
    own photos frame for frame, which a turntable spin over the whole clip
    cannot be.

    ONE map, and no `Rz`: the exported character now lives in the de-tilted
    CAPTURE frame (`Character.export_transform`), so the same transform serves
    every frame. While the export was in rig space this had to pick one
    frame's yaw for a whole clip, and on the client take the character drifted
    away from the photograph by up to the take's 36.97 deg of turning as the
    clip ran.
    """
    K = np.asarray(camera["K"], float).reshape(3, 3)
    Rc = np.asarray(camera["R"], float).reshape(3, 3)
    tc = np.asarray(camera["t"], float).reshape(3)
    w, h = (int(v) for v in camera["image_size"])

    U = M_view                            # world rotation -> capture rotation
    centre_world = -Rc.T @ tc
    centre = hips_world + (M_view @ centre_world - pelvis_ref) * scale
    # OpenCV camera axes in world: +X right, +Y down, +Z forward. Blender's are
    # +X right, +Y up, -Z forward.
    axes = U @ Rc.T
    basis = np.column_stack([axes[:, 0], -axes[:, 1], -axes[:, 2]])
    M = np.eye(4)
    M[:3, :3] = basis
    M[:3, 3] = centre
    return {
        "matrix": M.tolist(),
        # 36 mm is Blender's default sensor width; the lens that reproduces
        # this K on it is f_px / width * sensor.
        "lens_mm": float(36.0 * K[0, 0] / w),
        "sensor_mm": 36.0,
        # Blender's shift is in units of the SENSOR WIDTH for both axes under
        # sensor_fit HORIZONTAL, and a positive shift moves the frame so the
        # optical axis lands earlier: checked against
        # bpy_extras.object_utils.world_to_camera_view, which puts an on-axis
        # point at (0.5 - shift_x) of the width and (0.5*h + shift_y*w) pixels
        # down. Both are 0 for the assumed pinhole the app usually has.
        "shift_x": float(0.5 - K[0, 2] / w),
        "shift_y": float((K[1, 2] / h - 0.5) * h / w),
        "image_size": [w, h],
    }


def _character_bone_frames(poses3d, display_frame, head3d=None, *,
                           filled=None, recorded_up=None):
    """Just the posed bone matrices and their names, or (None, None).

    The narrow view of `_character_document` below, kept because it is what
    "does the export pose the character exactly as the 3D view does?" reads —
    and it must stay answerable without the placement, the root offsets or the
    camera getting in the way. Root motion is OFF here for the same reason: it
    is placement, not pose. The export itself takes `_character_document`,
    which says WHY it failed instead of answering None twice.
    """
    frag, _reason = _character_document(
        poses3d, display_frame, head3d, filled=filled,
        recorded_up=recorded_up, keep_root_motion=False)
    if frag is None:
        return None, None
    return frag["bone_frames"], frag["bone_names"]


def _character_document(poses3d: np.ndarray, display_frame: int,
                        head3d: np.ndarray | None = None, *,
                        filled: np.ndarray | None = None,
                        recorded_up=None, keep_root_motion: bool = True,
                        camera=None):
    """Per-frame posed bone matrices, computed with the SAME skinning the live
    3D view uses, so the exported character matches the preview pose-for-pose.

    `filled` is the (T, NUM_JOINTS) flag array from `pipeline.fill_gaps`. The
    export reads the same array the 3D view draws from, so the two can never
    disagree about what a dropout is: a one-frame gap the view shows filled
    (amber) is posed here too, and a longer gap the view shows as a hole stays
    absent here — which is what keeps Blender's hold-the-last-known-pose rule
    from quietly papering over a dropout the app is telling the user about.

    One case is left where view and export still differ, and it is inherent:
    a frame with NO valid joint at all yields None here, and `blender_job`
    then holds the previous whole pose (`last`) while the 3D view draws
    nothing. An animation format has no way to express "no pose this frame" —
    omitting the keyframes holds the previous pose too — so this is stated
    rather than fixed.

    `keep_root_motion` writes the bones in the de-tilted CAPTURE frame instead
    of rig space (`Character.export_transform`), so the file carries both
    things the 3D view shows and the delivered file did not: the subject's
    travel away from the take's pelvis, and the subject's TURNING.
    `root_offsets` carries the translation half out separately, so the
    turntable render can pin the figure back in place without a second posing
    pass changing anything else.

    Returns (fragment, reason): the JSON fragment to merge into the export
    document, or None and a `FAILURE_MESSAGES` key saying why. A reason is
    never silently swallowed — that is what used to let a different retarget
    ship in the character's place (F31).
    """
    try:
        from pose3d.geometry.character import Character, take_pelvis_ref
        from pose3d.geometry.orient import (take_up, de_tilt_matrix,
                                            detect_vertical, upright_matrix)
    except Exception as e:
        return None, ("character_import_failed", f"{type(e).__name__}: {e}")
    try:
        ch = Character()
    except Exception as e:
        return None, ("character_load_failed", f"{type(e).__name__}: {e}")

    poses3d = np.asarray(poses3d, float).reshape(-1, NUM_JOINTS, 3)
    # orient exactly as the 3D view does, so the export matches the preview
    # pose-for-pose: the vertical recorded at calibration time when there is
    # one, else the subject's own body line (see orient.take_up)
    up, _source, _spread = take_up(poses3d, recorded_up)
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

    upright = poses3d @ R
    # size the character to this subject once, from the whole take — the same
    # fit the 3D view applies, so the export stays identical to the preview
    ch.fit_to_subject(upright)
    # ...and place it by the same take-wide rule the view uses (view3d
    # `_take_placement`), which is what root motion IS: one reference pelvis
    # for the sequence instead of re-centring on each frame's own.
    pelvis_ref = take_pelvis_ref(upright) if keep_root_motion else None
    if keep_root_motion and pelvis_ref is None:
        keep_root_motion = False

    bone_frames, root_offsets = [], []
    scale = None                # the take's uniform scale, from the first frame
    for i, pose in enumerate(upright):
        valid = ~np.isnan(pose).any(1)
        if filled is not None:
            # `fill_gaps` gives every joint it flags a value, and pose3d itself
            # stays raw — so a flag can never point at a hole here. Checked
            # rather than papered over with `valid |= filled & valid`, which
            # was a tautology dressed as a coupling.
            #
            # NOT an `assert`: `python -O` strips those, so the guard would be
            # absent from exactly the build a frozen bundle might use, and if
            # it fired it aborted the export with an AssertionError instead of
            # the typed reason this phase exists to give.
            bad = np.flatnonzero(np.asarray(filled[i], bool) & ~valid)
            if bad.size:
                return None, ("fill_flags_disagree",
                              f"frame {i}: joints {list(map(int, bad))}")
        if not valid.any():
            bone_frames.append(None); root_offsets.append(None); continue
        # the face keypoints take the same rotation, so the exported head is
        # oriented exactly as the preview shows it
        hp = None if head3d is None else np.asarray(head3d[i], float) @ R
        # one solve per frame: the matrices AND the placement they were built
        # with, so `root_offsets` cannot drift from what the bones carry
        mats, al = ch.pose_bone_matrices(
            pose, valid, hp, keep_root_motion=keep_root_motion,
            pelvis_ref=pelvis_ref, return_alignment=True)
        bone_frames.append(mats)
        if al is None:
            root_offsets.append(None); continue
        if scale is None:
            scale = al.scale
        root_offsets.append([float(v) for v in al.root_offset])
    if all(b is None for b in bone_frames):
        return None, ("no_posable_frame", f"{len(bone_frames)} frames")

    frag = {"bone_frames": bone_frames, "bone_names": ch.bone_names,
            "root_offsets": root_offsets,
            "keep_root_motion": bool(keep_root_motion)}
    if camera is not None and pelvis_ref is not None and scale is not None:
        try:
            frag["camera"] = _rig_camera(camera, R.T, scale,
                                         pelvis_ref, ch.hips_world)
        except Exception as e:
            # a missing or malformed camera must cost the take its extra
            # render, never the export itself
            print(f"fixed-camera render skipped: {type(e).__name__}: {e}")
    return frag, None


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
    recorded_up=None,
    on_line=None,
    keep_root_motion: bool = True,
    schedule: str = "one_per_pose",
    camera=None,
    allow_fallback: bool = False,
    idle_timeout: float | None = None,
    cancelled=None,
) -> ExportResult:
    """Write BVH + FBX (+ mp4) for a take, by posing the bundled character.

    `keep_root_motion` (default ON for the files) places every frame against
    the take's pelvis, so the subject's travel is in the exported hips instead
    of being pinned away; the turntable mp4 is rendered with it pinned back so
    the figure does not spin its way out of frame.

    `schedule` is `"one_per_pose"` (default: export frame k+1 IS photograph k)
    or `"stepped"`, the 0.7 s hold / 0.3 s ease stop-motion the mp4 still uses
    and which, together with `keep_root_motion=False`, reproduces the file the
    client already has.

    `camera` is an optional {"K", "R", "t", "image_size"} for the LEFT camera
    in the capture's world frame; given one, a second mp4 is rendered from that
    exact viewpoint, which is the comparison the client actually makes.

    `allow_fallback` is the ONLY way to receive a file that is not the posed
    character. Without it a missing asset or a failed retarget returns
    `ok is False` with a `reason`, instead of silently shipping a different
    animation under the same name (F31).

    `timeout` is the overall deadline, `idle_timeout` the one that catches a
    hung render (no output at all for that long), and `cancelled()` is polled
    while Blender runs so the user can stop it. Any of the three kills the
    child and reaps it before returning `blender_timeout` / `blender_cancelled`
    — the export used to promise "and was stopped" while nothing had been.

    Both deadlines default to what today's callers already had: `timeout=600`
    as before, and `idle_timeout=None`, because an idle deadline is a kill
    condition NO caller had. An FBX bake or a single heavy EEVEE frame can say
    nothing for five minutes and still be working. The callers that want it
    ask for it — the GUI export (`idle_timeout=300, timeout=None`, where the
    Cancel button is the real guard) and the self-test.
    """
    requested_bundled = character == "__bundled__"
    if requested_bundled:
        from pose3d.config import character_blend
        character = character_blend()          # always the bundled model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = _poses_to_json(poses3d, fps)
    doc["display_frame"] = int(display_frame)   # which pose the turntable spins
    # a typo used to select the default silently, which in a phase about not
    # substituting one output for another is the wrong direction to fail in
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {sorted(SCHEDULES)}, "
                         f"not {schedule!r}")
    doc["schedule"] = schedule

    want_character = requested_bundled or bool(character)
    have_character = bool(character) and Path(character).exists()
    fallback_note = ""
    if want_character and not have_character:
        detail = f"looked for {character}" if character else "no path configured"
        if not allow_fallback:
            return _failure("character_asset_missing", detail, 2)
        fallback_note = f"the character asset is missing ({detail})"
        print(f"POSE3D_EXPORT_FALLBACK: {fallback_note}")
    if have_character:
        # drive the rig with the exact skinning the live view uses
        frag, reason = _character_document(
            poses3d, display_frame, head3d, filled=filled,
            recorded_up=recorded_up, keep_root_motion=keep_root_motion,
            camera=camera)
        if frag is not None:
            doc.update(frag)
        elif not allow_fallback:
            return _failure(reason[0], reason[1], 3)
        else:
            fallback_note = f"{reason[0]}: {reason[1]}"
            print(f"POSE3D_EXPORT_FALLBACK: {fallback_note}")
    json_path = out_dir / f"{name}_poses.json"
    json_path.write_text(json.dumps(doc), encoding="utf-8")

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
    if allow_fallback:
        cmd.append("--allow-fallback")

    # A frozen GUI has no usable stdin for the child to inherit, its output is
    # UTF-8 whatever the machine's code page says, and on Windows a console
    # child flashes a black window over the UI. See pose3d.runtime.
    kw = subprocess_kwargs()
    # ONE launch path. There used to be two — a streaming one for the GUI and
    # a `subprocess.run` one for everything else — and only the second
    # honoured `timeout`, so on the path the client actually took a hung
    # Blender blocked in `for line in p.stdout` for ever with nothing to kill
    # it. stderr is merged into stdout because a single stream is a single
    # reader, and a second pipe nobody drains is another way to deadlock.
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1, **kw)
    except FileNotFoundError:
        # must precede OSError, of which it is a subclass: say which binary is
        # missing and how to point at one, instead of a bare OSError
        return _failure("blender_missing",
                        f"Blender was not found (tried: {blender}).\n\nExport "
                        "needs Blender 5.x. Install it and either put it on "
                        "PATH or set the POSE3D_BLENDER environment variable to "
                        "the blender executable.", 127, note=fallback_note)
    except OSError as e:
        # e.g. POSE3D_BLENDER pointing at a folder, or a non-executable file
        return _failure("blender_not_runnable",
                        f"Could not run Blender at '{blender}'.\n\n"
                        f"{type(e).__name__}: {e}\n\nSet POSE3D_BLENDER to the "
                        "blender executable itself.", 126,
                        note=fallback_note)

    chunks: list[str] = []
    stopped = _pump(proc, chunks, on_line, timeout, idle_timeout, cancelled)
    stdout, stderr, rc = "".join(chunks), "", proc.returncode
    if stopped == "timeout":
        return _failure("blender_timeout",
                        _timeout_detail(timeout, idle_timeout),
                        124, note=fallback_note, stdout=stdout)
    if stopped == "cancelled":
        return _failure("blender_cancelled",
                        "Export cancelled. Blender was stopped and nothing "
                        "was written.", 125, note=fallback_note, stdout=stdout)

    def _exists(stem, ext):
        p = out_dir / f"{stem}.{ext}"
        return p if p.exists() else None

    res = ExportResult(
        bvh=_exists(name, "bvh"), fbx=_exists(name, "fbx"),
        mp4=_exists(name, "mp4") if render_video else None,
        mp4_camera=_exists(f"{name}_camera", "mp4") if render_video else None,
        fallback_note=fallback_note,
        returncode=rc, stdout=stdout, stderr=stderr)
    if not res.ok:
        res.reason = "blender_failed"
        res.message = (stderr or stdout or "")[-1500:]
    return res

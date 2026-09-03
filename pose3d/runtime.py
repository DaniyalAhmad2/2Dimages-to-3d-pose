"""Where the app's environment differs: platform, packaging, container.

Everything platform-specific goes through here so the differences are in one
readable place rather than scattered as `sys.platform` checks. The app ships
three ways — from source, as a Linux container, and as a frozen Windows
bundle — and each answers these questions differently.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")
IS_FROZEN = bool(getattr(sys, "frozen", False))

#: Marker file next to the executable that pins software OpenGL for the
#: next start. Written by the 3D view's "Restart with software 3D"
#: control, read by `pose3d.app._configure_gl` before QApplication exists.
SOFTWARE_GL_MARKER = "use-software-gl"

_LOG_NAME = "pose3d-log.txt"
_LOG_ROLLED = "pose3d-log.1.txt"

#: Roll the log over at this size. A GUI that logs every Qt warning fills a
#: disk otherwise, and nobody is going to email us a 900 MB file.
LOG_ROTATE_BYTES = 5 * 1024 * 1024

# Where the log actually went. Decided once, by ensure_std_streams(); None
# means the devnull fallback was taken and there is no file to point at.
_LOG_FILE: "Path | None" = None
_LOG_DECIDED = False


def in_container() -> bool:
    """True inside the Docker image (set by docker/Dockerfile).

    Container-only behaviour keys off this rather than off "not Windows", so
    running from source on Linux behaves like a normal desktop app.
    """
    return os.environ.get("POSE3D_CONTAINER") == "1"


def app_dir() -> Path:
    """Directory the application lives in.

    Frozen, this is the folder holding the executable, so things shipped
    alongside it — Blender, the pose weights — are found relative to it.
    From source it is the repository root.
    """
    if IS_FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def exe_name(stem: str) -> str:
    """Executable file name for this platform ('blender' -> 'blender.exe')."""
    return f"{stem}.exe" if IS_WINDOWS else stem


def user_state_dir() -> Path:
    """Per-user folder for application state, when the install folder is not
    writable — which is any bundle extracted under Program Files."""
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "Pose3D"
        return Path.home() / "AppData" / "Local" / "Pose3D"
    return Path.home() / ".local" / "state" / "pose3d"


def _rotate(path: Path) -> None:
    """Roll `path` aside once it is big enough to be a nuisance."""
    try:
        if path.stat().st_size < LOG_ROTATE_BYTES:
            return
        path.replace(path.with_name(_LOG_ROLLED))
    except OSError:
        pass                             # a log we cannot roll is still a log


def _open_log() -> "tuple[object, Path | None] | None":
    """(stream, path) for the best log location available, or None."""
    for folder, make in ((app_dir(), False), (user_state_dir(), True)):
        path = folder / _LOG_NAME
        try:
            if make:
                folder.mkdir(parents=True, exist_ok=True)
            _rotate(path)
            return open(path, "a", encoding="utf-8", buffering=1), path
        except OSError:
            continue
    try:
        return open(os.devnull, "w", encoding="utf-8"), None
    except OSError:
        return None


def ensure_std_streams() -> None:
    """Guarantee sys.stdout/stderr exist.

    A PyInstaller windowed build has None for both. Plenty of library code
    writes to them without checking — rtmlib does while downloading model
    weights — and dies with AttributeError. Point them at a log file beside
    the executable so failures leave a trail; at the user's own state folder
    when that location is read-only (an install under Program Files); and at
    the null device when even that fails, because a missing log must not stop
    the app from running.
    """
    global _LOG_FILE, _LOG_DECIDED
    missing = [n for n in ("stdout", "stderr") if getattr(sys, n, None) is None]
    if not missing:
        return
    opened = _open_log()
    if opened is None:
        return
    sink, _LOG_FILE = opened
    _LOG_DECIDED = True
    for name in missing:
        setattr(sys, name, sink)


def log_path() -> "Path | None":
    """The log file that was actually opened, or None when there is none.

    It used to name app_dir()/pose3d-log.txt unconditionally — including after
    the devnull fallback, i.e. exactly in the case where the client was being
    asked to send us a file that had never been written.
    """
    if _LOG_DECIDED:
        return _LOG_FILE
    return app_dir() / _LOG_NAME


def subprocess_kwargs() -> dict:
    """Keyword arguments for launching a child process from the GUI.

    Covers three things that only bite once the app is a windowed executable:
    a frozen build has no valid stdin for the child to inherit; a console
    child would flash a black window over the UI on Windows; and the child's
    output is UTF-8 regardless of the machine's ANSI code page, so decoding it
    with the locale default fails on any non-ASCII path.
    """
    kw: dict = {
        "stdin": subprocess.DEVNULL,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if IS_WINDOWS:
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return kw

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

_LOG_NAME = "pose3d-log.txt"


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


def ensure_std_streams() -> None:
    """Guarantee sys.stdout/stderr exist.

    A PyInstaller windowed build has None for both. Plenty of library code
    writes to them without checking — rtmlib does while downloading model
    weights — and dies with AttributeError. Point them at a log file beside
    the executable so failures leave a trail, or at the null device if that
    location is not writable.
    """
    missing = [n for n in ("stdout", "stderr") if getattr(sys, n, None) is None]
    if not missing:
        return
    sink = None
    try:
        sink = open(app_dir() / _LOG_NAME, "a", encoding="utf-8", buffering=1)
    except OSError:
        try:
            sink = open(os.devnull, "w", encoding="utf-8")
        except OSError:
            return
    for name in missing:
        setattr(sys, name, sink)


def log_path() -> Path:
    """Where ensure_std_streams() would write, for support instructions."""
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

"""File pickers that behave sensibly when the app runs inside a container.

Two things differ from a normal desktop:

* There is no desktop portal, so Qt's "native" dialog silently degrades to a
  bare fallback. Asking for the Qt dialog outright gives the same, predictable,
  themeable dialog on every host instead.
* The app can only see folders that were shared with it. Browsing to somewhere
  the container cannot reach shows an empty list, which reads as "broken", so
  the shared folders are pinned in the sidebar and used as the starting point.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QFileDialog

from pose3d.runtime import in_container

# Where the user's own files are mounted. /workspace is writable and is where
# projects and exports go; /host is the folder the app was launched from,
# mounted read-only so source images can be browsed without copying them in.
WORKSPACE = Path(os.environ.get("POSE3D_WORKSPACE", "/workspace"))
HOST = Path(os.environ.get("POSE3D_HOST", "/host"))


def mounts_apply() -> bool:
    """Do /workspace and /host mean anything on this machine?

    Only inside the image, or where the user named them. Written as absolute
    POSIX paths they are DRIVE-RELATIVE on Windows — `Path("/workspace")` is
    `C:\\workspace` — so a client who happens to have a folder of that name
    got it pinned into the file dialog and used as the starting directory,
    under a hint explaining that the app can only see files somewhere it has
    never heard of.
    """
    return in_container() or bool(os.environ.get("POSE3D_WORKSPACE")
                                  or os.environ.get("POSE3D_HOST"))


def shared_folders() -> list[Path]:
    """Folders the app can actually read, most useful first."""
    out = [p for p in (WORKSPACE, HOST) if mounts_apply() and p.is_dir()]
    if not out:
        # Running natively: nothing is restricted, so this is only about where
        # to start. The Windows bundle ships a `workspace` folder next to the
        # exe for projects and exports; fall back to the user's home.
        from pose3d.runtime import app_dir
        local = app_dir() / "workspace"
        out = [local] if local.is_dir() else [Path.home()]
    return out


def default_dir() -> str:
    folders = shared_folders()
    return str(folders[0])


def is_writable(path) -> bool:
    """Can the app actually create a file here?

    Asking the operating system is not enough on Windows: os.access(W_OK) only
    reports the folder's read-only *attribute* and ignores ACLs entirely, so it
    answers True for C:\\Program Files. The export pre-flight guard would then
    pass and the export would die minutes later inside Blender — which is the
    exact failure that guard exists to prevent. So write something.
    """
    p = Path(path)
    if not p.is_dir():
        return False
    try:
        fd, name = tempfile.mkstemp(prefix=".pose3d-write-test-", dir=str(p))
    except OSError:
        return False
    os.close(fd)
    try:
        os.unlink(name)
    except OSError:                  # created but not removable; still writable
        pass
    return True


def writable_dir() -> str:
    """Where output can actually be written.

    /host is mounted read-only on purpose — it exists so source images can be
    browsed, not written over — so saving must start somewhere writable.
    """
    for p in shared_folders():
        if is_writable(p):
            return str(p)
    return str(Path.home())


def not_writable_message(path) -> str:
    """Why this folder cannot be written to, and where to put things instead."""
    if mounts_apply() and HOST.is_dir() and str(path).startswith(str(HOST)):
        return (f"'{path}' is read-only.\n\nThat folder is shared with the app "
                f"for reading your images only. Save to {WORKSPACE} instead — "
                f"it is the 'workspace' folder next to docker-compose.yml, so "
                f"anything written there appears on your machine straight away.")
    return (f"'{path}' is read-only, so nothing can be saved there.\n\n"
            f"Try {writable_dir()}.")


def _prep(dlg: QFileDialog) -> None:
    if in_container():
        # No desktop portal in the image, so Qt's "native" dialog degrades to a
        # bare fallback; asking for the Qt one outright gives a usable dialog.
        # Pinning the mounts matters too — browsing anywhere else shows an
        # empty list, which reads as a broken app rather than as "not shared".
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setSidebarUrls([QUrl.fromLocalFile(str(p)) for p in shared_folders()])
    # Everywhere else the OS dialog is the right one: it knows about drives,
    # network locations, OneDrive and the user's own Recent list, none of which
    # Qt's fallback can offer.


def _run(dlg: QFileDialog):
    return dlg.selectedFiles() if dlg.exec() else []


def open_files(parent, title: str, filt: str, start: str | None = None) -> list[str]:
    dlg = QFileDialog(parent, title, start or default_dir(), filt)
    dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
    _prep(dlg)
    return _run(dlg)


def open_file(parent, title: str, filt: str, start: str | None = None) -> str:
    dlg = QFileDialog(parent, title, start or default_dir(), filt)
    dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
    _prep(dlg)
    got = _run(dlg)
    return got[0] if got else ""


def existing_directory(parent, title: str, start: str | None = None) -> str:
    dlg = QFileDialog(parent, title, start or default_dir())
    dlg.setFileMode(QFileDialog.FileMode.Directory)
    dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
    _prep(dlg)
    got = _run(dlg)
    return got[0] if got else ""


def location_hint() -> str:
    """One line telling the user which folders the app can see."""
    folders = shared_folders()
    if WORKSPACE in folders and HOST in folders:
        return ("The app can only open files under the folder you launched it "
                "from. Exports and projects are saved to its 'workspace' "
                "subfolder.")
    if WORKSPACE in folders:
        return ("The app can only open files under the 'workspace' folder next "
                "to docker-compose.yml. Copy your camera images there.")
    return ""

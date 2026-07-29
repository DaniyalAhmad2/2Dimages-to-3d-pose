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
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QFileDialog

# Where the user's own files are mounted. /workspace is writable and is where
# projects and exports go; /host is the folder the app was launched from,
# mounted read-only so source images can be browsed without copying them in.
WORKSPACE = Path(os.environ.get("POSE3D_WORKSPACE", "/workspace"))
HOST = Path(os.environ.get("POSE3D_HOST", "/host"))


def shared_folders() -> list[Path]:
    """Folders the app can actually read, most useful first."""
    out = [p for p in (WORKSPACE, HOST) if p.is_dir()]
    if not out:                      # running natively, not in the container
        out = [Path.home()]
    return out


def default_dir() -> str:
    folders = shared_folders()
    return str(folders[0])


def _prep(dlg: QFileDialog) -> None:
    # the container has no portal; the Qt dialog is the one that actually works
    dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dlg.setSidebarUrls([QUrl.fromLocalFile(str(p)) for p in shared_folders()])


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

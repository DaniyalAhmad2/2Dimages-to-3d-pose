"""Application entry point.

Usage:
    python -m pose3d.app [PROJECT_FOLDER]

Loads a project (or an empty session), builds the dashboard, runs the Qt event
loop. The Import wizard (top-bar button) creates a new project from uploaded
images + calibration and opens it in a fresh window.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# keep window references so they are not garbage-collected
_WINDOWS: list = []


def apply_dark_theme(app: QApplication) -> None:
    """Make the app dark everywhere, not just where the stylesheet reaches.

    dark.qss covers the widgets we style by name, but Qt draws everything else
    — splitter handles, tool tips, menus, combo popups, disabled text, focus
    rings — from the *system* palette and with the host's native style. On a
    Windows machine set to light mode that means light chrome punched through a
    dark UI: the splitter handles vanished against the panels, and the accuracy
    gauge was filled with the native window colour behind near-white text.

    Fusion is the one style that honours a supplied palette identically on
    every platform, so pin both rather than inheriting whatever the desktop is
    set to. This is a dark-themed application by design; it should not change
    appearance with the user's OS setting.
    """
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor("#0d0f15"))
    p.setColor(QPalette.ColorRole.WindowText, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.Base, QColor("#0f1219"))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor("#12151d"))
    p.setColor(QPalette.ColorRole.Text, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.Button, QColor("#161a24"))
    p.setColor(QPalette.ColorRole.ButtonText, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#12151d"))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor("#e6e8ee"))
    p.setColor(QPalette.ColorRole.Highlight, QColor("#3d7bfd"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8a91a3"))
    p.setColor(QPalette.ColorRole.Link, QColor("#6ea8ff"))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#5a6072"))
    app.setPalette(p)

    qss = Path(__file__).parent / "ui" / "dark.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text())


# kept: older call sites (and tests) refer to this name
load_stylesheet = apply_dark_theme


def build_model(project_folder: str | None):
    from pose3d.core.io_project import load_project
    from pose3d.core.project import ProjectData
    from pose3d.ui.model import ProjectModel

    if project_folder and Path(project_folder).exists():
        project = load_project(project_folder)
        rig = _load_rig(Path(project_folder) / "calibration")
        return ProjectModel(project, rig, project_dir=project_folder)
    project = ProjectData(name="No project loaded")
    return ProjectModel(project, None)


def _load_rig(calib_dir: Path):
    """Load a CalibratedRig from a calibration folder if fully present."""
    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.calib.intrinsics import Intrinsics
    from pose3d.pipeline import CalibratedRig
    import json
    try:
        il = Intrinsics.load(calib_dir / "left_intrinsics.json")
        ir = Intrinsics.load(calib_dir / "right_intrinsics.json")
        ext = json.loads((calib_dir / "extrinsics.json").read_text())
        import numpy as np
        el = Extrinsics(R=np.array(ext["left"]["R"]), t=np.array(ext["left"]["t"]))
        er = Extrinsics(R=np.array(ext["right"]["R"]), t=np.array(ext["right"]["t"]))
        return CalibratedRig(il, ir, el, er)
    except Exception:
        return None


def open_project_window(project_folder: str | None):
    """Build a model + MainWindow for a project folder and show it."""
    from pose3d.ui.main_window import MainWindow
    model = build_model(project_folder)
    win = MainWindow(model, open_callback=open_project_window)
    _WINDOWS.append(win)
    # In the container the app IS the desktop, so fill the virtual screen rather
    # than floating a fixed-size window inside it. showMaximized() is no use
    # here: maximising is a window-manager job and the container runs none, so
    # set the geometry outright.
    if os.environ.get("POSE3D_MAXIMIZE", "0") == "1":
        from PySide6.QtWidgets import QApplication
        qapp = QApplication.instance()
        screen = qapp.primaryScreen() if qapp is not None else None
        if screen is not None:
            win.setGeometry(screen.availableGeometry())
    win.show()
    return win


def install_crash_handler() -> None:
    """Make an unhandled exception visible instead of silent.

    Started from a shortcut with no console, a crash otherwise just closes the
    window: the client sees the app "not open" and has nothing to send us. Log
    it and say where the log is.
    """
    import traceback

    from pose3d.runtime import log_path

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr, flush=True)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                QMessageBox.critical(
                    None, "Pose3D stopped",
                    f"{exc_type.__name__}: {exc}\n\n"
                    f"Details were written to:\n{log_path()}")
        except Exception:
            pass                        # a dialog must never mask the crash

    sys.excepthook = hook


def main():
    # A frozen windowed build has no console to inherit, so anything that
    # writes to stdout/stderr — rtmlib's download progress, a Qt warning, a
    # traceback — hits None and raises. Give them a file to land in first.
    from pose3d.runtime import ensure_std_streams
    ensure_std_streams()
    install_crash_handler()

    if "--selftest" in sys.argv[1:]:
        from pose3d.selftest import main as selftest
        sys.exit(selftest([a for a in sys.argv[1:] if a != "--selftest"]))

    folder = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    # Share one GL context across windows so pyqtgraph's cached shader programs
    # stay valid when the Import flow opens a second window (otherwise
    # glUseProgram raises GLError 1281 on the new context).
    from PySide6.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    # Containers ship no desktop environment, so Qt has no icon theme to fall
    # back on and file-dialog toolbar buttons render blank. Name one explicitly
    # when the image provides it.
    from PySide6.QtGui import QIcon
    if not QIcon.themeName() and os.path.isdir("/usr/share/icons/Adwaita"):
        QIcon.setThemeName("Adwaita")
    apply_dark_theme(app)
    open_project_window(folder)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

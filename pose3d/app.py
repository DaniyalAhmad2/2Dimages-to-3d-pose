"""Application entry point.

Usage:
    python -m pose3d.app [PROJECT_FOLDER]

Loads a project (or an empty session), builds the dashboard, runs the Qt event
loop. The Import wizard (top-bar button) creates a new project from uploaded
images + calibration and opens it in a fresh window.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

# keep window references so they are not garbage-collected
_WINDOWS: list = []


def load_stylesheet(app: QApplication) -> None:
    qss = Path(__file__).parent / "ui" / "dark.qss"
    if qss.exists():
        app.setStyleSheet(qss.read_text())


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
    win.show()
    return win


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    # Share one GL context across windows so pyqtgraph's cached shader programs
    # stay valid when the Import flow opens a second window (otherwise
    # glUseProgram raises GLError 1281 on the new context).
    from PySide6.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    load_stylesheet(app)
    open_project_window(folder)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

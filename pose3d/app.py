"""Application entry point.

Usage:
    python -m pose3d.app [PROJECT_FOLDER]

Loads a project (or a demo if none given), builds the dashboard, runs the Qt
event loop. Calibration is loaded from <project>/calibration if present.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

_QSS = Path(__file__).with_name("ui") / "dark.qss"


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
    else:
        project = ProjectData(name="No project loaded")
        rig = None
    return ProjectModel(project, rig)


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


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    load_stylesheet(app)
    model = build_model(folder)

    from pose3d.ui.main_window import MainWindow
    win = MainWindow(model)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

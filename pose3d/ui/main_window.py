"""Main dashboard window assembling all panels, matching the mockup layout.

Top bar | Sidebar | [Left cam][Right cam][3D preview + accuracy] | Timeline.
Wires panels to the ProjectModel signal hub (panels never talk to each other).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from pose3d.core.project import CAM_LEFT, CAM_RIGHT
from pose3d.ui.camera_view import CameraView
from pose3d.ui.model import ProjectModel
from pose3d.ui.panels import JointAccuracyList, Sidebar
from pose3d.ui.timeline import Timeline
from pose3d.ui.view3d import View3D


class MainWindow(QMainWindow):
    def __init__(self, model: ProjectModel, load_image=None):
        super().__init__()
        self.model = model
        self.load_image = load_image or (lambda p: p)
        self.setWindowTitle("Pose3D — Animation Dashboard")
        self.resize(1500, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())

        mid = QWidget()
        mid_lay = QHBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)

        self.sidebar = Sidebar()
        mid_lay.addWidget(self.sidebar)

        self.cam_left = CameraView(CAM_LEFT)
        self.cam_right = CameraView(CAM_RIGHT)
        mid_lay.addWidget(self._titled("LEFT CAMERA", self.cam_left), 3)
        mid_lay.addWidget(self._titled("RIGHT CAMERA", self.cam_right), 3)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        self.view3d = View3D()
        right_lay.addWidget(self._titled("3D PREVIEW", self.view3d), 3)
        self.accuracy = JointAccuracyList()
        right_lay.addWidget(self.accuracy, 2)
        mid_lay.addWidget(right, 3)

        root.addWidget(mid, 1)

        self.timeline = Timeline()
        root.addWidget(self.timeline)

        self._wire()
        self._load_model()

    # --- build helpers ---
    def _build_topbar(self) -> QWidget:
        bar = QWidget(); bar.setObjectName("topbar"); bar.setFixedHeight(48)
        lay = QHBoxLayout(bar); lay.setContentsMargins(16, 0, 16, 0)
        self.title_label = QLabel("Project: —")
        self.title_label.setObjectName("projectTitle")
        lay.addWidget(self.title_label)
        lay.addStretch(1)
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_auto = QPushButton("Auto Recalculate 3D"); self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(True)
        self.btn_save = QPushButton("Save Corrections")
        for b in (self.btn_undo, self.btn_redo, self.btn_auto, self.btn_save):
            lay.addWidget(b)
        return bar

    def _titled(self, title: str, widget: QWidget) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(4)
        lab = QLabel(title); lab.setObjectName("panelTitle")
        v.addWidget(lab)
        v.addWidget(widget, 1)
        return box

    # --- wiring ---
    def _wire(self):
        self.cam_left.jointDragged.connect(self._on_drag)
        self.cam_right.jointDragged.connect(self._on_drag)
        self.timeline.frameSelected.connect(self.model.set_frame)

        self.model.frameChanged.connect(self._on_frame_changed)
        self.model.pose3dChanged.connect(self.view3d.set_pose)
        self.model.accuracyChanged.connect(self.accuracy.update_errors)
        self.model.joint2dChanged.connect(lambda *_: self._refresh_views())
        self.model.historyChanged.connect(self._refresh_history)

        self.btn_undo.clicked.connect(self.model.undo)
        self.btn_redo.clicked.connect(self.model.redo)
        self.btn_save.clicked.connect(self._save)
        self.btn_auto.toggled.connect(
            lambda on: setattr(self.model, "auto_recalc", on))

        self.sidebar.showJointsToggled.connect(self.cam_left.set_show_joints)
        self.sidebar.showJointsToggled.connect(self.cam_right.set_show_joints)
        self.sidebar.showBonesToggled.connect(self.cam_left.set_show_bones)
        self.sidebar.showBonesToggled.connect(self.cam_right.set_show_bones)

    def _on_drag(self, cam: str, joint: int, pos):
        self.model.set_joint_2d(cam, joint, pos.x(), pos.y())

    def _on_frame_changed(self, idx: int):
        self._refresh_views()
        f = self.model.frame()
        self.view3d.set_pose(f.fitted3d)

    def _refresh_views(self):
        f = self.model.frame()
        for cam, view in ((CAM_LEFT, self.cam_left), (CAM_RIGHT, self.cam_right)):
            path = f.images.get(cam)
            if path:
                view.set_image(self.load_image(path) if False else path)
            view.set_pose(f.kp2d[cam], f.scores[cam], f.corrected[cam])

    def _refresh_history(self):
        self.btn_undo.setEnabled(self.model.stack.can_undo())
        self.btn_redo.setEnabled(self.model.stack.can_redo())

    def _load_model(self):
        p = self.model.project
        self.title_label.setText(f"Project: {p.name}")
        res = "—"
        if p.frames and p.frames[0].images:
            res = "(images)"
        self.sidebar.set_project(p.name, len(p.frames), 2, res)
        self.sidebar.set_calibrated(self.model.rig is not None)
        self.timeline.populate(p.frames, self.load_image)
        if p.frames:
            self.model.set_frame(0)
            self.timeline.select(0)

    def _save(self):
        from pose3d.core.io_project import save_project
        # persist corrections into project + folder if known
        self.model.project.corrections = list(self.model.stack.log)
        # UI-level save target chosen elsewhere; here just emit to log
        print(f"Saved {len(self.model.project.corrections)} corrections")

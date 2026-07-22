"""Main dashboard window — laid out to match the Animation Dashboard mockup.

Top bar | Sidebar | [Left cam][Right cam] + action row | right column
(3D preview + pose-accuracy gauge + joint accuracy + selected joint) | Timeline.
Panels talk only through the ProjectModel signal hub.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QToolButton,
    QVBoxLayout, QWidget,
)

from pose3d.core.project import CAM_LEFT, CAM_RIGHT
from pose3d.core.skeleton import JOINT_NAMES
from pose3d.ui.camera_view import CameraPanel
from pose3d.ui.model import ProjectModel
from pose3d.ui.panels import (
    JointAccuracyList, PoseAccuracyPanel, SelectedJointPanel, Sidebar,
)
from pose3d.ui.timeline import Timeline, TimelineHeader
from pose3d.ui.view3d import View3D


class MainWindow(QMainWindow):
    def __init__(self, model: ProjectModel, load_image=None, detector=None):
        super().__init__()
        self.model = model
        self.detector = detector
        if load_image is None:
            import cv2
            load_image = lambda p: cv2.imread(p)
        self.load_image = load_image
        self.setWindowTitle("Pose3D — Animation Dashboard")
        self.resize(1540, 920)
        self.statusBar().showMessage("Ready")

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        root.addWidget(self._build_topbar())

        mid = QWidget(); mid_lay = QHBoxLayout(mid)
        mid_lay.setContentsMargins(6, 6, 6, 6); mid_lay.setSpacing(6)

        self.sidebar = Sidebar()
        mid_lay.addWidget(self.sidebar)

        # centre column: the two cameras + the action button row
        self.cam_left = CameraPanel(CAM_LEFT, "LEFT CAMERA")
        self.cam_right = CameraPanel(CAM_RIGHT, "RIGHT CAMERA")
        centre = QWidget(); centre_lay = QVBoxLayout(centre)
        centre_lay.setContentsMargins(0, 0, 0, 0); centre_lay.setSpacing(6)
        cams = QHBoxLayout(); cams.setSpacing(6)
        cams.addWidget(self._panel(self.cam_left), 1)
        cams.addWidget(self._panel(self.cam_right), 1)
        centre_lay.addLayout(cams, 1)
        centre_lay.addWidget(self._build_action_row())
        mid_lay.addWidget(centre, 6)

        mid_lay.addWidget(self._build_right_column(), 3)
        root.addWidget(mid, 1)

        # timeline area with header + legend
        tl_area = QWidget(); tl_lay = QVBoxLayout(tl_area)
        tl_lay.setContentsMargins(10, 2, 10, 6); tl_lay.setSpacing(2)
        self.timeline_header = TimelineHeader()
        self.timeline = Timeline()
        tl_lay.addWidget(self.timeline_header)
        tl_lay.addWidget(self.timeline)
        root.addWidget(tl_area)

        self._wire()
        self._load_model()

    # --- build helpers ---
    def _panel(self, w):
        w.setObjectName("cardPanel")
        return w

    def _build_topbar(self):
        bar = QWidget(); bar.setObjectName("topbar"); bar.setFixedHeight(46)
        lay = QHBoxLayout(bar); lay.setContentsMargins(16, 0, 16, 0)
        self.title_label = QLabel("Project: —"); self.title_label.setObjectName("projectTitle")
        lay.addWidget(self.title_label)
        lay.addStretch(1)
        self.saved_label = QLabel("✓ Project Saved"); self.saved_label.setObjectName("savedLabel")
        lay.addWidget(self.saved_label)
        lay.addStretch(1)
        for t in ("⚙ Settings", "? Help"):
            b = QToolButton(); b.setText(t); b.setObjectName("topTool")
            lay.addWidget(b)
        return bar

    def _build_action_row(self):
        row = QWidget(); row.setObjectName("actionRow")
        lay = QHBoxLayout(row); lay.setContentsMargins(8, 4, 8, 4); lay.setSpacing(8)
        self.btn_undo = QPushButton("↩  Undo")
        self.btn_redo = QPushButton("↪  Redo")
        self.btn_auto = QPushButton("⟳  Auto Recalculate 3D"); self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(True); self.btn_auto.setObjectName("primaryBtn")
        self.btn_save = QPushButton("⬇  Save Corrections")
        lay.addStretch(1)
        for b in (self.btn_undo, self.btn_redo, self.btn_auto, self.btn_save):
            lay.addWidget(b)
        lay.addStretch(1)
        return row

    def _build_right_column(self):
        col = QWidget(); col.setFixedWidth(340)
        lay = QVBoxLayout(col); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)

        # 3D preview card with header
        card = QWidget(); card.setObjectName("cardPanel")
        cl = QVBoxLayout(card); cl.setContentsMargins(8, 8, 8, 8)
        head = QHBoxLayout()
        t = QLabel("3D PREVIEW"); t.setObjectName("panelTitle"); head.addWidget(t)
        head.addStretch(1)
        self.proj_combo = QComboBox(); self.proj_combo.addItems(["Perspective", "Orthographic"])
        head.addWidget(self.proj_combo)
        self.btn_full = QToolButton(); self.btn_full.setText("⤢"); self.btn_full.setObjectName("camTool")
        head.addWidget(self.btn_full)
        cl.addLayout(head)
        self.view3d = View3D()
        cl.addWidget(self.view3d, 1)
        lay.addWidget(card, 4)

        self.pose_acc = PoseAccuracyPanel(); self.pose_acc.setObjectName("cardPanel")
        lay.addWidget(self.pose_acc, 3)
        self.accuracy = JointAccuracyList(); self.accuracy.setObjectName("cardPanel")
        lay.addWidget(self.accuracy, 4)
        self.selected = SelectedJointPanel(); self.selected.setObjectName("cardPanel")
        lay.addWidget(self.selected, 2)
        return col

    # --- wiring ---
    def _wire(self):
        for panel in (self.cam_left, self.cam_right):
            panel.view.jointDragged.connect(self._on_drag)
            panel.view.jointPicked.connect(self._on_pick)
        self.timeline.frameSelected.connect(self.model.set_frame)

        self.model.frameChanged.connect(self._on_frame_changed)
        self.model.pose3dChanged.connect(self.view3d.set_pose)
        self.model.accuracyChanged.connect(self._on_accuracy)
        self.model.joint2dChanged.connect(lambda *_: self._refresh_views())
        self.model.historyChanged.connect(self._refresh_history)
        self.model.statusMessage.connect(
            lambda m: self.statusBar().showMessage(m, 6000))

        self.btn_undo.clicked.connect(self.model.undo)
        self.btn_redo.clicked.connect(self.model.redo)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_auto.toggled.connect(self._on_auto_toggled)
        self.btn_full.clicked.connect(self._toggle_fullscreen)
        self.proj_combo.currentTextChanged.connect(self.view3d.set_projection)

        self.sidebar.runDetection.connect(self._on_run_detection)
        self.sidebar.recalibrate.connect(self._on_recalibrate)
        self.sidebar.showJointsToggled.connect(self.cam_left.view.set_show_joints)
        self.sidebar.showJointsToggled.connect(self.cam_right.view.set_show_joints)
        self.sidebar.showBonesToggled.connect(self.cam_left.view.set_show_bones)
        self.sidebar.showBonesToggled.connect(self.cam_right.view.set_show_bones)
        self.sidebar.showBodyToggled.connect(self.view3d.set_show_body)

    # --- handlers ---
    def _on_drag(self, cam, joint, pos):
        self.model.set_joint_2d(cam, joint, pos.x(), pos.y())
        self._mark_unsaved()

    def _on_pick(self, cam, joint):
        f = self.model.frame()
        score = float(np.nanmax([f.scores[CAM_LEFT][joint], f.scores[CAM_RIGHT][joint]]))
        corrected = bool(f.corrected[CAM_LEFT][joint] or f.corrected[CAM_RIGHT][joint])
        self.selected.set_joint(JOINT_NAMES[joint], score, corrected)

    def _on_accuracy(self, errors):
        overall = self.accuracy.update_errors(errors)
        self.pose_acc.set_overall(overall)

    def _on_auto_toggled(self, on):
        self.model.auto_recalc = on
        self.statusBar().showMessage(f"Auto Recalculate 3D {'ON' if on else 'OFF'}", 4000)

    def _on_save(self):
        self.model.save()
        self.saved_label.setText("✓ Project Saved")

    def _mark_unsaved(self):
        self.saved_label.setText("● Unsaved changes")

    def _ensure_detector(self):
        from PySide6.QtWidgets import QApplication
        if self.detector is None:
            try:
                from pose3d.detect.rtmpose import RTMPoseDetector
                self.statusBar().showMessage("Loading RTMPose model…")
                QApplication.processEvents()
                self.detector = RTMPoseDetector(mode="balanced", device="cpu")
            except Exception as e:
                self.statusBar().showMessage(f"Detector unavailable: {e}", 8000)
        return self.detector

    def _on_run_detection(self):
        from PySide6.QtWidgets import QApplication
        det = self._ensure_detector()
        if det is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self.model.redetect_all(det, self.load_image)
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_views(); self._refresh_timeline_status()

    def _on_recalibrate(self):
        self.model.recompute_all()
        self._refresh_views(); self._refresh_timeline_status()

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # --- refresh ---
    def _on_frame_changed(self, idx):
        self._refresh_views()
        self.view3d.set_pose(self.model.frame().fitted3d)

    def _refresh_views(self):
        f = self.model.frame()
        import os
        for cam, panel in ((CAM_LEFT, self.cam_left), (CAM_RIGHT, self.cam_right)):
            path = f.images.get(cam)
            if path:
                panel.view.set_image(path)
                panel.set_filename(os.path.basename(path))
            panel.view.set_pose(f.kp2d[cam], f.scores[cam], f.corrected[cam])

    def _refresh_history(self):
        self.btn_undo.setEnabled(self.model.stack.can_undo())
        self.btn_redo.setEnabled(self.model.stack.can_redo())

    def _refresh_timeline_status(self):
        for i in range(len(self.model.project.frames)):
            errs = self.model._accuracy(i)
            worst = float(np.nanmax(errs)) if not np.all(np.isnan(errs)) else None
            status = ("red" if worst is None else
                      "green" if worst < 5 else "amber" if worst < 12 else "red")
            self.timeline.set_status(i, status)

    def _load_model(self):
        p = self.model.project
        self.title_label.setText(f"Project: {p.name}")
        res = "—"
        if p.frames and p.frames[0].images:
            res = "(images)"
        self.sidebar.set_project(p.name, len(p.frames), 2, res)
        self.sidebar.set_calibrated(self.model.rig is not None)
        self.timeline.populate(p.frames, self.load_image)
        self.timeline_header.set_count(len(p.frames))
        if p.frames:
            self.model.set_frame(0)
            self.timeline.select(0)
            self._refresh_timeline_status()
        self._refresh_history()

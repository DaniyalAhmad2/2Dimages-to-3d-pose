"""Main dashboard window — laid out to match the Animation Dashboard mockup.

Top bar | Sidebar | [Left cam][Right cam] + action row | right column
(3D preview + pose-accuracy gauge + joint accuracy + selected joint) | Timeline.
Panels talk only through the ProjectModel signal hub.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter,
    QToolButton, QVBoxLayout, QWidget,
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
    def __init__(self, model: ProjectModel, load_image=None, detector=None,
                 open_callback=None):
        super().__init__()
        self.model = model
        self.detector = detector
        self.open_callback = open_callback   # open_project_window(folder)
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
        self._root_lay = root

        root.addWidget(self._build_topbar())

        mid = QWidget(); mid_lay = QHBoxLayout(mid)
        self._mid = mid
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

        # cameras | right column are drag-resizable (grab the divider to
        # widen the 3D panel); the right column is itself a vertical splitter.
        self._hsplit = QSplitter(Qt.Orientation.Horizontal)
        self._hsplit.addWidget(centre)
        self._hsplit.addWidget(self._build_right_column())
        self._hsplit.setStretchFactor(0, 3)
        self._hsplit.setStretchFactor(1, 2)
        self._hsplit.setSizes([820, 500])
        mid_lay.addWidget(self._hsplit, 1)
        root.addWidget(mid, 1)

        # timeline area with header + legend
        tl_area = QWidget(); tl_lay = QVBoxLayout(tl_area)
        tl_lay.setContentsMargins(10, 2, 10, 6); tl_lay.setSpacing(2)
        self.timeline_header = TimelineHeader()
        self.timeline = Timeline()
        tl_lay.addWidget(self.timeline_header)
        tl_lay.addWidget(self.timeline)
        root.addWidget(tl_area)
        self._tl_area = tl_area
        self._fs_active = False

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
        self.btn_import = QPushButton("⬆  Import Images")
        self.btn_export = QPushButton("⬇  Export Results")
        lay.addWidget(self.btn_import); lay.addWidget(self.btn_export)
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
        col = QSplitter(Qt.Orientation.Vertical)
        col.setMinimumWidth(300)

        # 3D preview card with header (drag the splitter handles to resize)
        card = QWidget(); card.setObjectName("cardPanel")
        cl = QVBoxLayout(card); cl.setContentsMargins(8, 8, 8, 8)
        head = QHBoxLayout()
        t = QLabel("3D PREVIEW"); t.setObjectName("panelTitle"); head.addWidget(t)
        head.addStretch(1)
        self.proj_combo = QComboBox(); self.proj_combo.addItems(["Perspective", "Orthographic"])
        head.addWidget(self.proj_combo)
        self.btn_full = QToolButton(); self.btn_full.setText("⤢")
        self.btn_full.setObjectName("camTool")
        self.btn_full.setToolTip("Toggle full-window 3D view (Esc to exit)")
        head.addWidget(self.btn_full)
        cl.addLayout(head)
        self.view3d = View3D()
        self.view3d.setMinimumHeight(220)
        cl.addWidget(self.view3d, 1)
        self._view3d_card = card            # whole card (header+view) for fullscreen

        self.pose_acc = PoseAccuracyPanel(); self.pose_acc.setObjectName("cardPanel")
        self.accuracy = JointAccuracyList(); self.accuracy.setObjectName("cardPanel")
        self.selected = SelectedJointPanel(); self.selected.setObjectName("cardPanel")

        for w in (card, self.pose_acc, self.accuracy, self.selected):
            col.addWidget(w)
        col.setCollapsible(0, False)
        col.setSizes([460, 190, 240, 130])   # 3D gets the most room by default
        self._rightcol = col
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
        # a joint edit repositions overlays only — never reloads the image /
        # re-fits the view (doing so mid-drag re-enters itemChange -> recursion)
        self.model.joint2dChanged.connect(lambda *_: self._refresh_overlays())
        self.model.historyChanged.connect(self._refresh_history)
        self.model.statusMessage.connect(
            lambda m: self.statusBar().showMessage(m, 6000))

        self.btn_undo.clicked.connect(self.model.undo)
        self.btn_redo.clicked.connect(self.model.redo)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_auto.toggled.connect(self._on_auto_toggled)
        self.btn_full.clicked.connect(self._toggle_fullscreen)
        self.proj_combo.currentTextChanged.connect(self.view3d.set_projection)

        self.btn_import.clicked.connect(self._on_import)
        self.btn_export.clicked.connect(self._on_export)
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

    def _on_import(self):
        from pose3d.ui.import_dialog import ImportDialog
        dlg = ImportDialog(self)
        if dlg.exec() and dlg.result_folder:
            if self.open_callback is not None:
                self.open_callback(dlg.result_folder)   # opens a fresh window
                self.close()
            else:
                self.statusBar().showMessage(
                    f"Imported to {dlg.result_folder}", 8000)

    def _on_export(self):
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        import numpy as np
        from pose3d.core.skeleton import NUM_JOINTS
        frames = self.model.project.frames
        # how many 3D joints actually reconstructed, on average?
        per_frame = [int((~np.isnan(f.fitted3d).any(1)).sum()) for f in frames]
        total = sum(per_frame)
        if not frames or total == 0:
            QMessageBox.warning(
                self, "Nothing to export",
                "No 3D pose was reconstructed, so there is nothing to render.\n\n"
                "This usually means calibration failed or the two camera views "
                "disagree on every joint. Import a synced pair with valid "
                "calibration (or upload intrinsics) and try again.")
            return
        avg = total / len(frames)
        if avg < 5:   # too few joints to look like a figure
            go = QMessageBox.question(
                self, "Sparse reconstruction",
                f"Only about {avg:.0f} of {NUM_JOINTS} joints were reconstructed "
                "per frame, so the video will look almost empty.\n\n"
                "This is a calibration/data issue — commonly approximate "
                "intrinsics, or the two views being too different so joints get "
                "dropped as inconsistent. Export anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if go != QMessageBox.StandardButton.Yes:
                return
        out = QFileDialog.getExistingDirectory(self, "Export results to folder")
        if not out:
            return
        poses = np.stack([f.fitted3d for f in frames])   # native units; camera auto-frames
        from pose3d.export.blender_export import export_animation
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Exporting BVH/FBX/mp4 via Blender…")
        QApplication.processEvents()
        try:
            res = export_animation(poses, out, name=self.model.project.name,
                                   fps=self.model.project.fps, render_video=True,
                                   display_frame=self.model.current)
        finally:
            QApplication.restoreOverrideCursor()
        if res.ok:
            QMessageBox.information(
                self, "Export complete",
                f"Wrote:\n{res.bvh}\n{res.fbx}\n{res.mp4}")
        else:
            QMessageBox.critical(self, "Export failed",
                                 (res.stderr or "")[-1500:])

    def _toggle_fullscreen(self):
        """Toggle the 3D card filling the whole app window (in-app, not a popup)."""
        if not self._fs_active:
            # hide everything else; the 3D card expands to fill the window
            self._mid.hide()
            self._tl_area.hide()
            self._root_lay.insertWidget(1, self._view3d_card, 1)  # after topbar
            self._view3d_card.show()
            self._fs_active = True
        else:
            self._root_lay.removeWidget(self._view3d_card)
            self._rightcol.insertWidget(0, self._view3d_card)     # back to top
            self._mid.show()
            self._tl_area.show()
            self._fs_active = False
        # nudge the GL view to re-frame at the new size
        self.view3d.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._fs_active:
            self._toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    # --- refresh ---
    def _on_frame_changed(self, idx):
        self._refresh_views()
        self.view3d.set_pose(self.model.frame().fitted3d)

    def _refresh_views(self):
        """Full refresh: (re)load the frame images AND reposition overlays."""
        self._load_images()
        self._refresh_overlays()

    def _load_images(self):
        """Load the frame's images + filenames. Only on frame change — NOT on a
        joint edit (reloading + fitInView mid-drag re-enters itemChange)."""
        import os
        f = self.model.frame()
        for cam, panel in ((CAM_LEFT, self.cam_left), (CAM_RIGHT, self.cam_right)):
            path = f.images.get(cam)
            if path:
                panel.view.set_image(path)
                panel.set_filename(os.path.basename(path))

    def _refresh_overlays(self):
        """Reposition/recolour the joint overlays from the current model state."""
        f = self.model.frame()
        for cam, panel in ((CAM_LEFT, self.cam_left), (CAM_RIGHT, self.cam_right)):
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

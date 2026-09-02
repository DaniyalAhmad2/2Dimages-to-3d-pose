"""Main dashboard window — laid out to match the Animation Dashboard mockup.

Top bar | Sidebar | [Left cam][Right cam] + action row | right column
(3D preview + pose-accuracy gauge + joint accuracy) | Timeline. Per-joint
detail lives on the keypoints themselves: hovering one in a camera view names
it and shows its accuracy. Panels talk only through the ProjectModel signal
hub.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter,
    QToolButton, QVBoxLayout, QWidget,
)


class _ExportWorker(QThread):
    """Runs the Blender export off the UI thread, streaming progress."""
    status = Signal(str)
    finished_res = Signal(object)          # ExportResult or Exception

    def __init__(self, poses, out, name, fps, display_frame, head3d=None,
                 filled=None, recorded_up=None, camera=None):
        super().__init__()
        self._a = (poses, out, name, fps, display_frame, head3d, filled,
                   recorded_up, camera)

    def _on_line(self, line: str):
        if "Fra:" in line:
            try:
                fr = line.split("Fra:")[1].split()[0]
                self.status.emit(f"Rendering the animation… (frame {fr})")
            except Exception:
                self.status.emit("Rendering the animation…")
        elif "FBX export" in line:
            self.status.emit("Exporting the FBX character…")
        elif "BVH Exported" in line or "export_anim.bvh" in line:
            self.status.emit("Exporting BVH…")
        elif "bake" in line.lower():
            self.status.emit("Baking the pose animation…")

    def run(self):
        from pose3d.export.blender_export import export_animation
        poses, out, name, fps, df, head3d, filled, recorded_up, cam = self._a
        self.status.emit("Posing the character in Blender…")
        try:
            # The delivered file's defaults: the subject's travel kept (the
            # 3D view places the figure by the same take-wide rule), one
            # keyframe per photographed pose, and no substitute animation if
            # anything goes wrong. `camera` adds a preview rendered from the
            # LEFT camera's own pose, which is the comparison the client makes.
            res = export_animation(poses, out, name=name, fps=fps,
                                   render_video=True, display_frame=df,
                                   head3d=head3d, filled=filled,
                                   recorded_up=recorded_up, camera=cam,
                                   keep_root_motion=True,
                                   schedule="one_per_pose",
                                   allow_fallback=False,
                                   on_line=self._on_line)
        except Exception as e:      # surface any failure to the UI thread
            res = e
        self.finished_res.emit(res)

from pose3d.core.project import CAM_LEFT, CAM_RIGHT
from pose3d.ui.camera_view import CameraPanel
from pose3d.ui.model import ProjectModel, frame_stat, worst_per_joint
from pose3d.ui.panels import (
    JointAccuracyList, PoseAccuracyPanel, Sidebar,
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
        # The 3D view and the Blender export each build their own Character
        # and never see this project, so the head convention it was detected
        # under is published process-wide here, once, before anything is drawn.
        # Both then read the same value and stay pose-identical.
        from pose3d.geometry.character import set_default_head_source
        set_default_head_source(getattr(model.project, "head_source", "nose"))
        self.setWindowTitle("Pose3D — Animation Dashboard")
        self.resize(1540, 920)
        self.statusBar().showMessage("Ready")

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        self._root_lay = root

        self._build_menus()
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
        self._fs_split = self._fs_side = None
        self._migration_banner = None

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
        return bar

    def _build_menus(self):
        """The one menu action the dashboard needs: the migration path for a
        project made before face keypoints existed."""
        tools = self.menuBar().addMenu("&Tools")
        act = tools.addAction("Re-detect face points only")
        act.setToolTip("Detect the nose/eyes/ears again so the character's "
                       "head can be oriented, leaving the body pose and every "
                       "hand correction exactly as they are")
        act.triggered.connect(self._on_redetect_head)

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
        # Where a character failure surfaces. Everything the 3D view could not
        # do used to be swallowed, so an empty card was the only symptom.
        self.view3d_error = QLabel("")
        self.view3d_error.setWordWrap(True)
        self.view3d_error.setStyleSheet("color:#e0a33a; font-size:11px;")
        self.view3d_error.hide()
        cl.addWidget(self.view3d_error)
        self.view3d = View3D()
        self.view3d.setMinimumHeight(220)
        cl.addWidget(self.view3d, 1)
        self._view3d_card = card            # whole card (header+view) for fullscreen

        self.pose_acc = PoseAccuracyPanel(); self.pose_acc.setObjectName("cardPanel")
        self.accuracy = JointAccuracyList(); self.accuracy.setObjectName("cardPanel")

        for w in (card, self.pose_acc, self.accuracy):
            col.addWidget(w)
        col.setCollapsible(0, False)
        col.setSizes([460, 190, 300])   # 3D gets the most room by default
        self._rightcol = col
        return col

    # --- wiring ---
    def _wire(self):
        for panel in (self.cam_left, self.cam_right):
            panel.view.jointDragged.connect(self._on_drag)
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
        self.sidebar.setScale.connect(self._on_set_scale)
        self.view3d.characterError.connect(self._on_character_error)
        self.model.qualityChanged.connect(self._refresh_quality)
        self.sidebar.showJointsToggled.connect(self.cam_left.view.set_show_joints)
        self.sidebar.showJointsToggled.connect(self.cam_right.view.set_show_joints)
        self.sidebar.showBonesToggled.connect(self.cam_left.view.set_show_bones)
        self.sidebar.showBonesToggled.connect(self.cam_right.view.set_show_bones)
        self.sidebar.showBodyToggled.connect(self.view3d.set_show_body)
        self.sidebar.showCaptureToggled.connect(self.view3d.set_show_capture)

    # --- handlers ---
    def _on_drag(self, cam, joint, pos):
        self.model.set_joint_2d(cam, joint, pos.x(), pos.y())
        self._mark_unsaved()

    def _on_accuracy(self, errors):
        """`errors` is `{cam: {"measured"|"delivered": (NUM_JOINTS,)}}`.

        The residuals arrive normalised by each camera's own figure height, so
        the two views are directly comparable and neither is averaged into the
        other — averaging them is what used to hide the case that matters,
        one view agreeing and the other not.

        The gauge reads the DELIVERED pose, because that is the pose on screen
        and in the export. The per-joint dots and the JOINT ACCURACY list read
        the MEASURED one, because that is the residual a drag of the keypoint
        can actually drive to zero: banding a draggable point on a number the
        bone fit controls would make correcting it feel broken.
        """
        states = self.model.joint_states(self.model.current)
        worst = worst_per_joint(errors, "measured")
        self.accuracy.update_errors(worst)
        self.pose_acc.set_accuracy(
            {c: frame_stat(errors[c]["delivered"]) for c in errors},
            {c: frame_stat(errors[c]["measured"]) for c in errors})
        # the keypoints themselves are colour-banded by the same numbers, and
        # hovering one shows the figure — replaces the old SELECTED JOINT card
        for cam, panel in ((CAM_LEFT, self.cam_left),
                           (CAM_RIGHT, self.cam_right)):
            per = errors.get(cam, {})
            panel.set_accuracy(per.get("measured"), per.get("delivered"),
                               states.get(cam))

    def _on_character_error(self, message: str):
        self.view3d_error.setText(message)
        self.view3d_error.setToolTip(message)
        self.view3d_error.setVisible(bool(message))
        if message:
            self.statusBar().showMessage(message, 10000)

    def _on_set_scale(self, real_height_m: float):
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            applied = self.model.set_scale_from_height(real_height_m)
        finally:
            QApplication.restoreOverrideCursor()
        if applied is None:
            return
        self._apply_view_orientation()   # the character is sized to the take
        self._refresh_views(); self._refresh_timeline_status()
        self._refresh_calibration_status()
        self._mark_unsaved()

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
                # which pose model this is (COCO-17 / Halpe-26) is
                # RTMPoseDetector's own default: see detect.rtmpose.USE_HALPE26
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
            # BEFORE redetect_all, which re-poses and redraws as it goes: the
            # 2D is about to be replaced, so what HEAD means is replaced with
            # it, and nothing may be posed under the old convention.
            self._adopt_head_source(det)
            self.model.redetect_all(det, self.load_image)
        finally:
            QApplication.restoreOverrideCursor()
        self._apply_view_orientation()   # poses changed: re-fit the character
        self._refresh_views(); self._refresh_timeline_status()

    def _on_redetect_head(self):
        from PySide6.QtWidgets import QApplication
        det = self._ensure_detector()
        if det is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self.model.redetect_head(det, self.load_image)
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_views()

    def _adopt_head_source(self, det):
        """Record the head convention a re-detection is about to write.

        `redetect_all` replaces every 2D point, so a project detected as
        COCO-17 becomes a Halpe-26 one; leaving `head_source` behind would have
        the retarget correct a skull HEAD for the nose's forward offset.
        """
        from pose3d.geometry.character import set_default_head_source
        source = det.head_source
        self.model.project.head_source = source
        set_default_head_source(source)
        # The 3D view builds its Character once and keeps it; the Blender
        # export builds a fresh one per export and would read the new
        # convention immediately. Drop the cached one when the convention
        # actually moves, or the preview and the export would pose the same
        # frame differently — the one thing they may never do. It is rebuilt,
        # and re-fitted by _apply_view_orientation, on the next draw.
        view = getattr(self, "view3d", None)
        cached = getattr(view, "_character", None)
        if cached is not None and cached.head_source != source:
            view._character = None

    def _on_recalibrate(self):
        self.model.recompute_all()
        self._apply_view_orientation()   # poses changed: re-fit the character
        self._refresh_views(); self._refresh_timeline_status()
        self._refresh_calibration_status()

    def _recorded_vertical(self):
        """The vertical recorded in this project's calibration folder, or None.

        Read from disk every time rather than cached: the sidebar, the 3D view
        and the export must all state the SAME vertical, and a cache set as a
        side effect of drawing the view makes the export's answer depend on
        whether the view happened to be refreshed first.
        """
        from pathlib import Path

        from pose3d.calib.resolve import load_world_up
        if not self.model.project_dir:
            return None
        return load_world_up(Path(self.model.project_dir) / "calibration")

    def _refresh_calibration_status(self):
        from pose3d.calib.quality import check_rig
        warnings = list(check_rig(self.model.rig, self._recorded_vertical()))
        if self.model.rig_error:
            warnings.insert(0, self.model.rig_error)
        self.sidebar.set_calibrated(self.model.rig is not None, warnings)
        self._refresh_quality()

    def _calibration_report(self):
        """(this project's calibration/report.json or None, why not).

        A take calibrated before the report existed simply has none, and the
        marker row says "—" rather than inventing a tag id for it. A report
        that is THERE and unreadable is a different fact and gets a sentence:
        showing the same "—" for both is exactly the swallowing this phase is
        about, and the row would be quietly saying "no provenance recorded"
        about a file that records it.
        """
        import json
        from pathlib import Path
        if not self.model.project_dir:
            return None, ""
        path = (Path(self.model.project_dir) / "calibration" / "report.json")
        try:
            return json.loads(path.read_text()), ""
        except FileNotFoundError:
            return None, ""
        except Exception as e:
            return None, (f"calibration/report.json could not be read "
                          f"({type(e).__name__}) — the marker size and tag "
                          f"this calibration used cannot be shown")

    def _refresh_quality(self):
        """The sidebar rows that reprojection cannot see.

        Bone-length spread is ~7x more responsive than the gauge to a wrong
        camera pose but blind to a focal length shared by both cameras;
        epipolar disagreement moves on exactly that focal. Neither is a
        reprojection, which is weak by construction for a freshly triangulated
        point. They are here because between them they cover the gauge's
        blind spots.
        """
        from pose3d.quality import symmetry_notes
        q = self.model.quality()
        self.sidebar.set_quality(q, symmetry_notes(q) if q else ())
        report, report_error = self._calibration_report()
        baseline = None
        if self.model.rig is not None:
            centres = [self.model.rig.ext[c].camera_center
                       for c in (CAM_LEFT, CAM_RIGHT)]
            baseline = float(np.linalg.norm(centres[0] - centres[1]))
        self.sidebar.set_metric_facts(
            baseline_m=baseline,
            subject_height_m=None if q is None else q.subject_height_m,
            marker=report, marker_error=report_error)

    def _on_import(self):
        from PySide6.QtWidgets import QMessageBox
        try:
            from pose3d.ui.import_dialog import ImportDialog
            dlg = ImportDialog(self)
        except Exception as e:
            # Qt swallows exceptions raised inside a slot, so without this the
            # button just appears to do nothing and the traceback goes only to
            # the container log, where nobody is looking.
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Could not open the import dialog",
                f"{type(e).__name__}: {e}")
            return
        return self._run_import_dialog(dlg)

    def _run_import_dialog(self, dlg):
        if dlg.exec() and dlg.result_folder:
            if self.open_callback is not None:
                self.open_callback(dlg.result_folder)   # opens a fresh window
                self.close()
            else:
                self.statusBar().showMessage(
                    f"Imported to {dlg.result_folder}", 8000)

    def _on_export(self):
        from PySide6.QtWidgets import QApplication, QMessageBox
        from pose3d.ui import filedialog
        import numpy as np
        from pose3d.core.skeleton import NUM_JOINTS
        frames = self.model.project.frames
        # How many 3D joints actually RECONSTRUCTED, on average — a joint the
        # gap fill interpolated is posed and exported, but it is not something
        # the cameras saw, so it does not count towards "the two views gave us
        # a figure". Counting it here would let a take whose every other frame
        # is an interpolation report a full skeleton.
        per_frame = [int((~np.isnan(f.fitted3d).any(1)
                          & ~np.asarray(f.filled, bool)).sum())
                     for f in frames]
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
        out = filedialog.existing_directory(
            self, "Export results to folder", filedialog.writable_dir())
        if not out:
            return
        # Fail here, with an explanation, rather than minutes later inside
        # Blender with a bare errno from a read-only mount.
        if not filedialog.is_writable(out):
            QMessageBox.warning(self, "Cannot save there",
                                filedialog.not_writable_message(out))
            return
        poses = np.stack([f.fitted3d for f in frames])   # native units; camera auto-frames

        # run the (slow) Blender export on a worker thread with a live progress
        # dialog, so the UI stays responsive instead of looking frozen/crashed.
        from PySide6.QtWidgets import QProgressDialog
        prog = QProgressDialog("Preparing export…", None, 0, 0, self)
        prog.setWindowTitle("Exporting results")
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumWidth(420); prog.setMinimumDuration(0)
        prog.setCancelButton(None)         # a Blender render can't be safely killed
        prog.show()

        heads = np.stack([f.head3d for f in frames])
        # the same flags the 3D view draws amber, so the export agrees with
        # the preview about which joints were interpolated
        filled = np.stack([f.filled for f in frames])
        worker = _ExportWorker(poses, out, self.model.project.name,
                               self.model.project.fps, self.model.current,
                               head3d=heads, filled=filled,
                               recorded_up=self._recorded_vertical(),
                               camera=self._left_camera())
        self._export_worker = worker       # keep a reference
        worker.status.connect(prog.setLabelText)

        def _finished(res):
            prog.close()
            self._export_worker = None
            if isinstance(res, Exception):
                QMessageBox.critical(self, "Export failed", str(res))
            elif res.ok:
                items = [
                    ("Video — turntable", res.mp4),
                    ("Video — from the left camera's own position",
                     res.mp4_camera),
                    ("Character — rigged armature, matches the 3D view", res.fbx),
                    ("Motion capture", res.bvh)]
                body = "\n\n".join(f"{lbl}:\n{p}" for lbl, p in items if p)
                QMessageBox.information(self, "Export complete", "Wrote:\n\n" + body)
            else:
                # Say WHY, from the reason the export carries, instead of the
                # tail of Blender's log: nothing was written on purpose, and
                # the user needs to know that rather than guess.
                QMessageBox.critical(
                    self, "Export failed",
                    res.message or (res.stderr or res.stdout or "")[-1500:])

        worker.finished_res.connect(_finished)
        worker.start()

    def _left_camera(self):
        """The LEFT camera's intrinsics + pose, or None.

        Feeds the fixed-camera preview render: the same viewpoint the client's
        photographs were taken from, so "does the character follow the
        keypoints?" can be answered by looking. Absent calibration it is simply
        not rendered — never a guessed viewpoint presented as the real one.
        """
        import json
        from pathlib import Path
        try:
            calib = Path(self.model.project_dir) / "calibration"
            from pose3d.calib.intrinsics import Intrinsics
            intr = Intrinsics.load(calib / "left_intrinsics.json")
            ext = json.loads((calib / "extrinsics.json").read_text())["left"]
            return {"K": intr.K.tolist(), "R": ext["R"], "t": ext["t"],
                    "image_size": list(intr.image_size)}
        except Exception:
            return None

    def _toggle_fullscreen(self):
        """Toggle the 3D view filling the app window (in-app, not a popup).

        The cameras step aside, but the accuracy readouts move to the right of
        the 3D view and the timeline stays at the bottom — so frames can be
        stepped through and judged without leaving the large view.
        """
        if not self._fs_active:
            self._mid.hide()
            side = QWidget()
            sl = QVBoxLayout(side)
            sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(6)
            for w in (self.pose_acc, self.accuracy):
                sl.addWidget(w)
            sl.addStretch(1)

            split = QSplitter(Qt.Orientation.Horizontal)
            split.addWidget(self._view3d_card)
            split.addWidget(side)
            split.setStretchFactor(0, 4)
            split.setStretchFactor(1, 1)
            split.setSizes([1120, 340])
            split.setCollapsible(0, False)

            self._fs_split, self._fs_side = split, side
            self._root_lay.insertWidget(1, split, 1)      # after the topbar
            split.show()
            self._tl_area.show()                          # keep frames scrubbable
            self._fs_active = True
        else:
            self._root_lay.removeWidget(self._fs_split)
            # put the panels back in the right column, in their original order
            self._rightcol.insertWidget(0, self._view3d_card)
            for i, w in enumerate((self.pose_acc, self.accuracy), 1):
                self._rightcol.insertWidget(i, w)
            self._rightcol.setSizes([460, 190, 300])
            self._fs_side.deleteLater()
            self._fs_split.deleteLater()
            self._fs_split = self._fs_side = None
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
        f = self.model.frame()
        # head3d must ride along or the character's head snaps back to riding
        # the neck on every frame change
        self.view3d.set_pose(f.fitted3d, f.head3d, f.filled)

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
            panel.view.set_pose(f.kp2d[cam], f.scores[cam], f.corrected[cam],
                                head_xy=f.head2d[cam], filled=f.filled)

    def _refresh_history(self):
        self.btn_undo.setEnabled(self.model.stack.can_undo())
        self.btn_redo.setEnabled(self.model.stack.can_redo())

    def _refresh_timeline_status(self):
        """Band each frame on its MEDIAN joint residual, in figure heights.

        The cuts are the same two the dots and the gauge use (0.4 % and 1.0 %
        of the subject's height in the image), not the old literal 5 px and
        12 px — which on a 3072x4080 frame demanded the whole take reconstruct
        to under a thousandth of the figure, and painted 21 of the client
        take's 26 frames red with not one green. It now reads 2 green, 22
        amber, 2 red.

        It reads the MEASURED residual: the timeline is a navigation aid, and
        what it should point at is the frames whose two views disagree, which
        are the frames a keypoint correction can actually fix.
        """
        from pose3d.ui.panels import ACC_AMBER_FRAC, ACC_GREEN_FRAC
        for i in range(len(self.model.project.frames)):
            errs = self.model._accuracy(i)
            frac = frame_stat(worst_per_joint(errs, "measured"))
            status = ("red" if not np.isfinite(frac) else
                      "green" if frac < ACC_GREEN_FRAC else
                      "amber" if frac < ACC_AMBER_FRAC else "red")
            self.timeline.set_status(i, status)

    def _apply_view_orientation(self):
        """Orient the 3D view so "up" is trustworthy.

        The vertical recorded at calibration time when the project has one (it
        is a property of the room, so it keeps a lean held for the whole take),
        else the subject's own body line as before. The sidebar states which,
        and how uncertain it is.
        """
        from pose3d.geometry.orient import take_up, de_tilt_matrix
        recorded = self._recorded_vertical()  # the export reads it too
        frames = self.model.project.frames
        poses = [f.fitted3d for f in frames
                 if f.fitted3d is not None and not np.isnan(f.fitted3d).all()]
        R = None
        up, source, spread = take_up(np.stack(poses) if poses else None,
                                     recorded)
        if up is not None:
            R = de_tilt_matrix(up)
        self.sidebar.show_levelling_note(R is not None, source, spread,
                                         recorded)
        self.view3d.set_orientation(R)
        if poses:
            # size the character to this subject (same fit the export uses)
            self.view3d.fit_subject(np.stack(poses))

    def _show_migration_banner(self, note: str):
        """One dismissible line saying what changed on open, and offering the
        stored pose back. Nothing has been written to disk."""
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
        label = QLabel(note); label.setWordWrap(True)
        restore = QPushButton("Restore stored pose")
        dismiss = QPushButton("Dismiss")
        lay.addWidget(label, 1); lay.addWidget(restore); lay.addWidget(dismiss)
        self.statusBar().addWidget(bar, 1)
        self._migration_banner = bar

        def close_banner():
            self.statusBar().removeWidget(bar)
            bar.deleteLater()
            self._migration_banner = None

        def on_restore():
            if self.model.restore_stored_pose():
                self._apply_view_orientation()
                self._refresh_views(); self._refresh_timeline_status()
            close_banner()

        restore.clicked.connect(on_restore)
        dismiss.clicked.connect(close_banner)

    def _load_model(self):
        p = self.model.project
        # A project written by an older pipeline is brought up to date once,
        # here, with no button press — and says so, with this take's numbers.
        note = self.model.upgrade_pipeline() or self.model.migration_note
        self.title_label.setText(f"Project: {p.name}")
        res = "—"
        if p.frames and p.frames[0].images:
            res = "(images)"
        self.sidebar.set_project(p.name, len(p.frames), 2, res)
        self._refresh_calibration_status()
        self.timeline.populate(p.frames, self.load_image)
        self.timeline_header.set_count(len(p.frames))
        self._apply_view_orientation()
        if p.frames:
            self.model.set_frame(0)
            self.timeline.select(0)
            self._refresh_timeline_status()
        self._refresh_history()
        if note:
            self._show_migration_banner(note)

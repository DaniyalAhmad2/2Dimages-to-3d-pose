"""Import wizard: upload synced image pairs + calibration, then process.

Flow:
  1. Pick the left-camera and right-camera images (auto-matched by filename).
  2. Optionally upload intrinsics (per camera) and extrinsics. If extrinsics
     are omitted they are estimated from ArUco markers in the images; if
     intrinsics are omitted an approximate model is used. If no calibration is
     uploaded and no ArUco can be found, a "calibration not successful" warning
     is shown (the project still opens for 2D review).
  3. Process: detect keypoints, triangulate, bone-fit, save the project.

On success, ``result_folder`` holds the saved project folder.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from pose3d.calib.intrinsics import Intrinsics
from pose3d.calib.resolve import (
    finalize_world_up, load_extrinsics_json, resolve_calibration, save_rig,
)
from pose3d.core.importer import build_project, match_frames
from pose3d.core.io_project import save_project
from pose3d.core.names import safe_name
from pose3d.imageio import read_image
from pose3d.ui import filedialog, guard
from pose3d.ui.worker import run_job


class _FilePicker(QWidget):
    """A read-only line + Browse button for one or many files/a folder."""

    def __init__(self, kind: str, filt: str = ""):
        super().__init__()
        self.kind = kind          # 'files' | 'file' | 'dir'
        self.filt = filt
        self.paths: list[str] = []
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.line = QLineEdit(); self.line.setReadOnly(True)
        btn = QPushButton("Browse…"); btn.clicked.connect(self._browse)
        lay.addWidget(self.line, 1); lay.addWidget(btn)

    def _browse(self):
        if self.kind == "files":
            paths = filedialog.open_files(self, "Select images", self.filt)
            self.paths = paths
            self.line.setText(f"{len(paths)} file(s)" if paths else "")
        elif self.kind == "dir":
            d = filedialog.existing_directory(self, "Select folder")
            self.paths = [d] if d else []
            self.line.setText(d)
        else:
            p = filedialog.open_file(self, "Select file", self.filt)
            self.paths = [p] if p else []
            self.line.setText(p)

    def first(self):
        return self.paths[0] if self.paths else None



class ImportDialog(QDialog):
    def __init__(self, parent=None, projects_root: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Import Images & Calibration")
        self.setMinimumWidth(560)
        self._hint = filedialog.location_hint()
        self.result_folder: str | None = None
        self._detector = None
        self._projects_root = Path(projects_root or (Path.home() / "pose3d_projects"))

        root = QVBoxLayout(self)

        # Running in a container, the app can only read folders shared with it;
        # browsing anywhere else shows an empty list and looks broken.
        if self._hint:
            hint = QLabel(self._hint)
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#e0a33a; font-size:11px;")
            root.addWidget(hint)

        # --- images ---
        img_box = QGroupBox("Synced Images")
        form = QFormLayout(img_box)
        self.left_pick = _FilePicker("files", "Images (*.jpg *.jpeg *.png *.bmp)")
        self.right_pick = _FilePicker("files", "Images (*.jpg *.jpeg *.png *.bmp)")
        form.addRow("Left camera:", self.left_pick)
        form.addRow("Right camera:", self.right_pick)
        self.match_label = QLabel("")
        self.left_pick.line.textChanged.connect(self._update_match)
        self.right_pick.line.textChanged.connect(self._update_match)
        form.addRow("Matched pairs:", self.match_label)
        root.addWidget(img_box)

        # --- calibration ---
        cal_box = QGroupBox("Calibration (optional — else estimated from ArUco)")
        cform = QFormLayout(cal_box)
        self.intr_l = _FilePicker("file", "Intrinsics (*.json)")
        self.intr_r = _FilePicker("file", "Intrinsics (*.json)")
        self.extr = _FilePicker("file", "Extrinsics (*.json)")
        self.marker = QDoubleSpinBox()
        self.marker.setDecimals(3); self.marker.setRange(0.005, 2.0)
        self.marker.setValue(0.05); self.marker.setSuffix(" m")
        cform.addRow("Left intrinsics:", self.intr_l)
        cform.addRow("Right intrinsics:", self.intr_r)
        cform.addRow("Extrinsics:", self.extr)
        cform.addRow("ArUco marker size:", self.marker)
        root.addWidget(cal_box)

        # --- project ---
        proj_box = QGroupBox("Project")
        pform = QFormLayout(proj_box)
        self.name = QLineEdit("Imported_Session")
        self.out_pick = _FilePicker("dir")
        # Off by default, and it says why: a temporal filter is a video-rate
        # tool. On stop-motion frames there is no temporal signal to filter,
        # so it just drags every frame toward its neighbours.
        self.smooth_check = QCheckBox(
            "Smooth across frames (video-rate capture only; a stop-motion "
            "take has no temporal signal to filter)")
        self.smooth_check.setChecked(False)
        pform.addRow("Name:", self.name)
        pform.addRow("Save to folder:", self.out_pick)
        pform.addRow("Smoothing:", self.smooth_check)
        root.addWidget(proj_box)

        # --- buttons ---
        btns = QHBoxLayout(); btns.addStretch(1)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        self.process_btn = QPushButton("Process"); self.process_btn.setObjectName("primaryBtn")
        self.process_btn.clicked.connect(self._process)
        btns.addWidget(cancel); btns.addWidget(self.process_btn)
        root.addLayout(btns)

    def _update_match(self):
        left, right = self.left_pick.paths, self.right_pick.paths
        if left and right:
            n = len(match_frames(left, right))
            self.match_label.setText(f"{n} pair(s)")
        else:
            self.match_label.setText("")

    # --- processing ---
    def _process(self):
        """Run the import, phase by phase, with the GUI thread left alone.

        Every long phase here — copying the images, finding the markers,
        detecting, reconstructing — goes through `worker.run_job`, so the
        dialog stays alive and says where it is. It used to drive them all
        inline and pump the event loop by hand, which re-enters every slot in
        the app from the middle of this one.
        """
        left, right = self.left_pick.paths, self.right_pick.paths
        if not left or not right:
            guard.report_error(self, "Missing images",
                               "Please choose both left and right images.")
            return
        out_root = Path(self.out_pick.first() or self._projects_root)
        # `safe_name`, not `replace(" ", "_")`: a colon or a question mark in
        # the typed name is a folder Windows refuses to create, and the
        # refusal arrived after the whole dialog had been filled in.
        folder = out_root / safe_name(self.name.text())
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            guard.report_error(self, "Cannot create folder", str(e))
            return

        try:
            # No Cancel on the phases that cannot honour one: a button that
            # does nothing is the lie this phase exists to remove. What the
            # copy needed was to say WHICH pair it is stuck on — that is where
            # a virus scanner or a OneDrive placeholder holds it up.
            project = run_job(
                self, "Importing images", lambda report, cancelled:
                build_project(left, right, name=self.name.text(),
                              copy_into=folder, on_progress=report),
                cancellable=False)
            if isinstance(project, Exception):
                return                    # cancelled, or already reported
            if not project.frames:
                guard.report_error(self, "No pairs",
                                   "No image pairs matched.")
                return

            # calibration inputs
            il = Intrinsics.load(self.intr_l.first()) if self.intr_l.first() else None
            ir = Intrinsics.load(self.intr_r.first()) if self.intr_r.first() else None
            el = er = None
            if self.extr.first():
                el, er = load_extrinsics_json(self.extr.first())

            cal = run_job(
                self, "Resolving calibration", lambda report, cancelled:
                resolve_calibration(
                    project, read_image,
                    marker_length=self.marker.value(),
                    intr_left=il, intr_right=ir, ext_left=el, ext_right=er),
                cancellable=False)
            if isinstance(cal, Exception):
                return

            rig = cal.rig
            # what the numbers mean, kept with the project: the tag size the
            # extrinsics were scaled by is not recoverable from anything else
            project.marker_length = float(self.marker.value())
            if not cal.ok:
                cont = QMessageBox.warning(
                    self, "Calibration not successful",
                    cal.message + "\n\nOpen the project for 2D review without "
                    "3D reconstruction?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if cont != QMessageBox.StandardButton.Yes:
                    return
            elif cal.approximate:
                QMessageBox.information(self, "Calibration", cal.message)

            # detection with progress — the pipeline's own loop, so the face
            # keypoints (and anything else it learns to write) are not dropped
            # on the floor by a duplicate loop that only knew about kp2d.
            # keypoint_model / head_source / detector are all written by
            # detect_project from the detector itself: they are facts about
            # the detection, and a caller that forgot head_source produced a
            # halpe26 project posed under the nose convention.
            from pose3d.pipeline import detect_project

            def detect(report, cancelled):
                det = self._ensure_detector()      # loading the model is slow
                return detect_project(project, det, read_image,
                                      on_progress=report, cancelled=cancelled)

            # ... and this one IS cancellable: it is the long phase, and
            # `detect_project` commits every frame or none of them.
            detected = run_job(self, "Detecting keypoints", detect)
            if isinstance(detected, Exception):
                return                    # a cancel wrote nothing at all

            smooth = self.smooth_check.isChecked()
            project.smoothing = "ema0.6" if smooth else "none"

            # reconstruct if calibrated
            dropped, fit_report = 0, None
            if rig is not None:
                def reconstruct(report, cancelled):
                    from pose3d.pipeline import fit_project, triangulate_project
                    report(0, 0, "Triangulating every frame…")
                    n = triangulate_project(project, rig)
                    report(0, 0, "Fitting the skeleton to the take…")
                    got = fit_project(project, smooth=smooth)
                    save_rig(rig, folder / "calibration", cal.report)
                    # The calibration was resolved before any of the above, so
                    # its recorded vertical had no poses to take its SENSE
                    # from and fell back to "the phones were held upright".
                    # Now there are poses: settle it on the body and rewrite
                    # the two files.
                    finalize_world_up(project, folder / "calibration")
                    return n, got

                done = run_job(self, "Reconstructing 3D", reconstruct,
                               cancellable=False)
                if isinstance(done, Exception):
                    return
                dropped, fit_report = done

            save_project(project, folder)
            self.result_folder = str(folder)
            msg = f"Imported {len(project.frames)} frames.\n{cal.message}"
            from pose3d.pipeline import rejection_note
            notes = [rejection_note(dropped, len(project.frames))]
            if fit_report is not None:
                notes.append(fit_report.note())
            for note in notes:
                if note:
                    msg += "\n\n" + note
            QMessageBox.information(self, "Done", msg)
            self.accept()
        except Exception as e:                       # surface any failure cleanly
            import traceback
            traceback.print_exc()          # into the log the user is asked for
            guard.report_error(self, "Import failed",
                               f"{type(e).__name__}: {e}")

    def _ensure_detector(self):
        if self._detector is None:
            from pose3d.detect.rtmpose import RTMPoseDetector
            # which pose model this is (COCO-17 / Halpe-26) is
            # RTMPoseDetector's own default: see detect.rtmpose.USE_HALPE26
            self._detector = RTMPoseDetector(mode="balanced", device="cpu")
        return self._detector

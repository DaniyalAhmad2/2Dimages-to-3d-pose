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

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressDialog, QPushButton,
    QVBoxLayout, QWidget,
)

from pose3d.calib.intrinsics import Intrinsics
from pose3d.calib.resolve import (
    load_extrinsics_json, resolve_calibration, save_rig,
)
from pose3d.core.importer import build_project, match_frames
from pose3d.core.io_project import save_project
from pose3d.ui import filedialog


class _Cancelled(Exception):
    """The user pressed Cancel in the progress dialog."""


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
        left, right = self.left_pick.paths, self.right_pick.paths
        if not left or not right:
            QMessageBox.warning(self, "Missing images",
                                "Please choose both left and right images.")
            return
        out_root = Path(self.out_pick.first() or self._projects_root)
        folder = out_root / self.name.text().strip().replace(" ", "_")
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Cannot create folder", str(e)); return

        prog = QProgressDialog("Importing images…", "Cancel", 0, 0, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0); prog.show()

        try:
            project = build_project(left, right, name=self.name.text(),
                                    copy_into=folder)
            if not project.frames:
                prog.close()
                QMessageBox.warning(self, "No pairs", "No image pairs matched.")
                return

            # calibration inputs
            il = Intrinsics.load(self.intr_l.first()) if self.intr_l.first() else None
            ir = Intrinsics.load(self.intr_r.first()) if self.intr_r.first() else None
            el = er = None
            if self.extr.first():
                el, er = load_extrinsics_json(self.extr.first())

            prog.setLabelText("Resolving calibration…")
            _pe()
            cal = resolve_calibration(
                project, lambda p: cv2.imread(str(p)),
                marker_length=self.marker.value(),
                intr_left=il, intr_right=ir, ext_left=el, ext_right=er)

            rig = cal.rig
            if not cal.ok:
                prog.close()
                cont = QMessageBox.warning(
                    self, "Calibration not successful",
                    cal.message + "\n\nOpen the project for 2D review without "
                    "3D reconstruction?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if cont != QMessageBox.StandardButton.Yes:
                    return
                prog = QProgressDialog("Detecting…", "Cancel", 0, len(project.frames), self)
                prog.setWindowModality(Qt.WindowModality.WindowModal); prog.show()
            elif cal.approximate:
                QMessageBox.information(self, "Calibration", cal.message)

            # detection with progress — the pipeline's own loop, so the face
            # keypoints (and anything else it learns to write) are not dropped
            # on the floor by a duplicate loop that only knew about kp2d
            det = self._ensure_detector()
            prog.setMaximum(len(project.frames))
            prog.setLabelText("Detecting keypoints…")
            from pose3d.pipeline import detect_project

            def on_frame(i, n):
                if prog.wasCanceled():
                    raise _Cancelled()
                prog.setValue(i); _pe()

            try:
                detect_project(project, det, lambda p: cv2.imread(str(p)),
                               on_frame=on_frame)
            except _Cancelled:
                prog.close()
                return

            smooth = self.smooth_check.isChecked()
            project.smoothing = "ema0.6" if smooth else "none"

            # reconstruct if calibrated
            dropped, report = 0, None
            if rig is not None:
                prog.setLabelText("Reconstructing 3D…"); _pe()
                from pose3d.pipeline import fit_project, triangulate_project
                dropped = triangulate_project(project, rig)
                report = fit_project(project, smooth=smooth)
                save_rig(rig, folder / "calibration")

            save_project(project, folder)
            prog.close()
            self.result_folder = str(folder)
            msg = f"Imported {len(project.frames)} frames.\n{cal.message}"
            from pose3d.pipeline import rejection_note
            notes = [rejection_note(dropped, len(project.frames))]
            if report is not None:
                notes.append(report.note())
            for note in notes:
                if note:
                    msg += "\n\n" + note
            QMessageBox.information(self, "Done", msg)
            self.accept()
        except Exception as e:                       # surface any failure cleanly
            prog.close()
            QMessageBox.critical(self, "Import failed", str(e))

    def _ensure_detector(self):
        if self._detector is None:
            from pose3d.detect.rtmpose import RTMPoseDetector
            self._detector = RTMPoseDetector(mode="balanced", device="cpu")
        return self._detector


def _pe():
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()

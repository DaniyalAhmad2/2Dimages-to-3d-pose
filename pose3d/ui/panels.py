"""Sidebar, joint-accuracy list, and RAG legend panels."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from pose3d.core.skeleton import JOINT_NAMES, NUM_JOINTS, rag_status


def _section(title: str) -> QLabel:
    lab = QLabel(title)
    lab.setObjectName("sectionHeader")
    return lab


class Sidebar(QWidget):
    runDetection = Signal()
    recalibrate = Signal()
    showJointsToggled = Signal(bool)
    showBonesToggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(240)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)

        lay.addWidget(_section("PROJECT"))
        self.project_name = QLabel("Untitled")
        self.project_name.setObjectName("projectName")
        lay.addWidget(self.project_name)
        self.info = QLabel("")
        self.info.setObjectName("infoText")
        lay.addWidget(self.info)

        lay.addSpacing(8)
        lay.addWidget(_section("CALIBRATION"))
        self.calib_status = QLabel("Not calibrated")
        lay.addWidget(self.calib_status)
        btn_recal = QPushButton("Recalibrate 3D")
        btn_recal.clicked.connect(self.recalibrate)
        lay.addWidget(btn_recal)

        lay.addSpacing(8)
        lay.addWidget(_section("PROCESSING"))
        btn_run = QPushButton("Run Detection")
        btn_run.clicked.connect(self.runDetection)
        lay.addWidget(btn_run)

        lay.addSpacing(8)
        lay.addWidget(_section("DISPLAY"))
        self.cb_joints = QCheckBox("Show Joints"); self.cb_joints.setChecked(True)
        self.cb_bones = QCheckBox("Show Bones"); self.cb_bones.setChecked(True)
        self.cb_joints.toggled.connect(self.showJointsToggled)
        self.cb_bones.toggled.connect(self.showBonesToggled)
        lay.addWidget(self.cb_joints)
        lay.addWidget(self.cb_bones)

        lay.addStretch(1)

    def set_project(self, name: str, n_frames: int, n_cams: int, res: str):
        self.project_name.setText(name)
        self.info.setText(f"Frames: {n_frames}\nCameras: {n_cams}\nResolution: {res}")

    def set_calibrated(self, ok: bool):
        self.calib_status.setText("Calibrated" if ok else "Not calibrated")


class JointAccuracyList(QWidget):
    """Per-joint reprojection-error list, sorted worst-first (like the mockup)."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(_section("JOINT ACCURACY"))
        self.list = QListWidget()
        self.list.setObjectName("accuracyList")
        lay.addWidget(self.list)

    def update_errors(self, errors: np.ndarray):
        """errors: (NUM_JOINTS,) reprojection error px (NaN allowed)."""
        errors = np.asarray(errors, float).reshape(NUM_JOINTS)
        order = np.argsort(-np.nan_to_num(errors, nan=-1))  # worst first
        self.list.clear()
        for j in order:
            e = errors[j]
            # map error (px) to a pseudo-confidence for RAG colour
            txt = f"{JOINT_NAMES[j]:<16} {'--' if np.isnan(e) else f'{e:5.1f}px'}"
            item = QListWidgetItem(txt)
            if np.isnan(e):
                status = "red"
            elif e < 3:
                status = "green"
            elif e < 8:
                status = "amber"
            else:
                status = "red"
            item.setForeground(Qt.GlobalColor.white)
            from pose3d.ui.timeline import STATUS_COLORS
            item.setForeground(STATUS_COLORS[status])
            self.list.addItem(item)

"""Sidebar, Pose-Accuracy gauge, Joint-accuracy list, Selected-Joint panel.

Styled to match the Animation Dashboard mockup: sectioned sidebar with info
rows, a circular pose-accuracy gauge, a per-joint accuracy list (as %), and a
selected-joint detail panel.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from pose3d.core.skeleton import JOINT_NAMES, NUM_JOINTS

# accuracy banding (percent) -> colour, matching the mockup legend
ACC_HIGH, ACC_MED = 85.0, 70.0
COL_GREEN = QColor(78, 214, 122)
COL_AMBER = QColor(240, 190, 74)
COL_RED = QColor(235, 92, 92)
COL_PURPLE = QColor(170, 120, 240)


def accuracy_pct(err_px: float) -> float:
    """Map a reprojection error (px) to a 0-100 accuracy percentage."""
    if np.isnan(err_px):
        return float("nan")
    return float(np.clip(100.0 * np.exp(-err_px / 6.0), 0.0, 100.0))


def acc_color(pct: float) -> QColor:
    if np.isnan(pct):
        return COL_RED
    if pct >= ACC_HIGH:
        return COL_GREEN
    if pct >= ACC_MED:
        return COL_AMBER
    return COL_RED


def acc_label(pct: float) -> str:
    if np.isnan(pct):
        return "Low"
    return "High" if pct >= ACC_HIGH else ("Medium" if pct >= ACC_MED else "Low")


def _section(title: str) -> QLabel:
    lab = QLabel(title)
    lab.setObjectName("sectionHeader")
    return lab


class _InfoRow(QWidget):
    def __init__(self, label: str, value: str = ""):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        self._k = QLabel(label); self._k.setObjectName("infoKey")
        self._v = QLabel(value); self._v.setObjectName("infoVal")
        self._v.setAlignment(Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self._k); lay.addStretch(1); lay.addWidget(self._v)

    def set_value(self, v: str):
        self._v.setText(v)


class PoseAccuracyGauge(QWidget):
    """Circular gauge showing overall pose accuracy % + High/Medium/Low."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self._pct = float("nan")

    def set_value(self, pct: float):
        self._pct = pct
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height()) - 16
        rect = QRectF((self.width() - side) / 2, 8, side, side)
        # track
        p.setPen(QPen(QColor(40, 44, 56), 10))
        p.drawArc(rect, 0, 360 * 16)
        pct = 0.0 if np.isnan(self._pct) else self._pct
        col = acc_color(self._pct)
        arc_pen = QPen(col, 10)
        arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc_pen)
        # start at top (90deg), clockwise
        p.drawArc(rect, 90 * 16, -int(360 * 16 * pct / 100.0))
        # text
        p.setPen(QColor(235, 238, 245))
        f = QFont(); f.setPointSize(22); f.setBold(True); p.setFont(f)
        txt = "--" if np.isnan(self._pct) else f"{pct:.0f}%"
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)
        f2 = QFont(); f2.setPointSize(9); p.setFont(f2)
        p.setPen(col)
        sub = QRectF(rect.left(), rect.center().y() + 16, rect.width(), 20)
        p.drawText(sub, Qt.AlignmentFlag.AlignHCenter, acc_label(self._pct))


class PoseAccuracyPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.addWidget(_section("POSE ACCURACY"))
        self.gauge = PoseAccuracyGauge()
        lay.addWidget(self.gauge)
        for txt, col in (("High (95-100%)", COL_GREEN),
                         ("Medium (85-94%)", COL_AMBER),
                         ("Low (0-84%)", COL_RED)):
            row = QHBoxLayout()
            dot = QLabel("●"); dot.setStyleSheet(f"color: {col.name()};")
            row.addWidget(dot); row.addWidget(QLabel(txt)); row.addStretch(1)
            lay.addLayout(row)
        lay.addStretch(1)

    def set_overall(self, pct: float):
        self.gauge.set_value(pct)


class JointAccuracyList(QWidget):
    """Per-joint accuracy (%) list, worst-first, with coloured dots."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.addWidget(_section("JOINT ACCURACY"))
        self.list = QListWidget()
        self.list.setObjectName("accuracyList")
        lay.addWidget(self.list)

    def update_errors(self, errors: np.ndarray) -> float:
        errors = np.asarray(errors, float).reshape(NUM_JOINTS)
        pcts = np.array([accuracy_pct(e) for e in errors])
        order = np.argsort(np.nan_to_num(pcts, nan=-1))   # worst first
        self.list.clear()
        for j in order:
            pct = pcts[j]
            shown = "--" if np.isnan(pct) else f"{pct:3.0f}%"
            item = QListWidgetItem(f"  ●  {JOINT_NAMES[j]:<15}{shown:>6}")
            item.setForeground(acc_color(pct))
            self.list.addItem(item)
        valid = pcts[~np.isnan(pcts)]
        return float(valid.mean()) if valid.size else float("nan")


class SelectedJointPanel(QWidget):
    """Shows the currently-picked joint: name, confidence, status."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.addWidget(_section("SELECTED JOINT"))
        self.name = QLabel("—"); self.name.setObjectName("selJointName")
        lay.addWidget(self.name)
        self.conf = _InfoRow("Confidence", "—"); lay.addWidget(self.conf)
        self.status = _InfoRow("Status", "—"); lay.addWidget(self.status)
        lay.addStretch(1)

    def set_joint(self, name: str, score: float, corrected: bool):
        col = acc_color(100 if score >= 0.6 else (75 if score >= 0.35 else 20))
        self.name.setText(f"●  {name}")
        self.name.setStyleSheet(f"color: {col.name()}; font-size: 15px;")
        self.conf.set_value("--" if np.isnan(score) else f"{score:.2f}")
        self.status.set_value("Corrected" if corrected else
                              ("Detected" if score > 0 else "Missing"))


class Sidebar(QWidget):
    runDetection = Signal()
    recalibrate = Signal()
    showJointsToggled = Signal(bool)
    showBonesToggled = Signal(bool)
    showBodyToggled = Signal(bool)
    showCaptureToggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(232)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(6)

        lay.addWidget(_section("PROJECT"))
        self.project_name = QLabel("Untitled")
        self.project_name.setObjectName("projectName")
        lay.addWidget(self.project_name)
        self.row_frames = _InfoRow("Frames", "0"); lay.addWidget(self.row_frames)
        self.row_cams = _InfoRow("Cameras", "2 (Left, Right)"); lay.addWidget(self.row_cams)
        self.row_res = _InfoRow("Resolution", "—"); lay.addWidget(self.row_res)
        self.row_fps = _InfoRow("FPS (Export)", "30"); lay.addWidget(self.row_fps)
        self.row_status = _InfoRow("Status", "Ready"); lay.addWidget(self.row_status)

        lay.addSpacing(6)
        lay.addWidget(_section("CALIBRATION"))
        self.calib_status = QLabel("Not calibrated")
        lay.addWidget(self.calib_status)
        # problems that will skew the 3D silently unless they are surfaced
        self.calib_warn = QLabel("")
        self.calib_warn.setWordWrap(True)
        self.calib_warn.setStyleSheet("color:#e0a33a; font-size:11px;")
        self.calib_warn.hide()
        lay.addWidget(self.calib_warn)
        btn_recal = QPushButton("↻  Recalibrate 3D")
        btn_recal.clicked.connect(self.recalibrate)
        lay.addWidget(btn_recal)

        lay.addSpacing(6)
        lay.addWidget(_section("PROCESSING"))
        btn_run = QPushButton("▷  Run Detection")
        btn_run.clicked.connect(self.runDetection)
        lay.addWidget(btn_run)

        lay.addSpacing(6)
        lay.addWidget(_section("DISPLAY"))
        self.cb_joints = QCheckBox("Show Joints"); self.cb_joints.setChecked(True)
        self.cb_bones = QCheckBox("Show Bones"); self.cb_bones.setChecked(True)
        self.cb_body = QCheckBox("Show Body (3D)"); self.cb_body.setChecked(True)
        # the raw triangulated skeleton, for judging how closely the character
        # tracks the capture; off by default so the view shows one figure
        self.cb_capture = QCheckBox("Show Captured Skeleton")
        self.cb_capture.setChecked(False)
        self.cb_capture.setToolTip(
            "Overlay the skeleton measured from the cameras.\n"
            "The character has its own proportions, so the two differ where\n"
            "the subject's build differs from the model's.")
        self.cb_joints.toggled.connect(self.showJointsToggled)
        self.cb_bones.toggled.connect(self.showBonesToggled)
        self.cb_body.toggled.connect(self.showBodyToggled)
        self.cb_capture.toggled.connect(self.showCaptureToggled)
        lay.addWidget(self.cb_joints)
        lay.addWidget(self.cb_bones)
        lay.addWidget(self.cb_body)
        lay.addWidget(self.cb_capture)

        lay.addStretch(1)
        hint = QLabel("Drag joints in the 2D views to adjust. 3D updates "
                      "automatically when Auto Recalculate 3D is on.")
        hint.setObjectName("hintBox"); hint.setWordWrap(True)
        lay.addWidget(hint)

    def set_project(self, name, n_frames, n_cams, res):
        self.project_name.setText(name)
        self.row_frames.set_value(str(n_frames))
        self.row_res.set_value(res)

    def set_calibrated(self, ok: bool, warnings=()):
        warnings = list(warnings)
        if ok and warnings:
            self.calib_status.setText("⚠ Calibrated (with problems)")
            self.calib_status.setStyleSheet("color:#e0a33a;")
        else:
            self.calib_status.setText("✓ Calibrated" if ok else "Not calibrated")
            self.calib_status.setStyleSheet(
                f"color: {'#4ed67a' if ok else '#e65c5c'};")
        self.calib_warn.setText("\n\n".join(f"• {w}" for w in warnings))
        self.calib_warn.setToolTip("\n\n".join(warnings))
        self.calib_warn.setVisible(bool(warnings))

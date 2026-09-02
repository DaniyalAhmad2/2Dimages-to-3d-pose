"""Sidebar, Pose-Accuracy gauge, Joint-accuracy list.

Styled to match the Animation Dashboard mockup: sectioned sidebar with info
rows, a circular pose-accuracy gauge and a per-joint accuracy list (as %).
Per-joint detail on demand lives on the camera keypoints themselves (hover),
not in a separate panel.
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
# #cardPanel from dark.qss. Custom-painted widgets must fill this themselves —
# a stylesheet background is not drawn for a QWidget subclass that overrides
# paintEvent, so anything relying on it inherits the host's system colour.
COL_PANEL = QColor(15, 18, 25)
COL_TEXT = QColor(235, 238, 245)


def accuracy_pct(err_px: float) -> float:
    """Map a reprojection error (px) to a 0-100 accuracy percentage."""
    if np.isnan(err_px):
        return float("nan")
    return float(np.clip(100.0 * np.exp(-err_px / 6.0), 0.0, 100.0))


def acc_band(pct: float) -> str:
    """Accuracy band key: "green" | "amber" | "red".

    The single source of banding. `acc_color`, `acc_label` and the camera
    views' joint dots all derive from this, so a legend reword cannot silently
    change what colour a joint is drawn in.
    """
    if np.isnan(pct):
        return "red"
    return "green" if pct >= ACC_HIGH else ("amber" if pct >= ACC_MED else "red")


_BAND_COLORS = {"green": COL_GREEN, "amber": COL_AMBER, "red": COL_RED}
_BAND_LABELS = {"green": "High", "amber": "Medium", "red": "Low"}


def acc_color(pct: float) -> QColor:
    return _BAND_COLORS[acc_band(pct)]


def acc_label(pct: float) -> str:
    return _BAND_LABELS[acc_band(pct)]


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
        # Paint our own background rather than relying on whatever is behind.
        # Overriding paintEvent means Qt does not apply the stylesheet's
        # background to this subclass, so the widget was filled with the host's
        # native window colour — light, on a Windows machine in light mode —
        # and the near-white readout below became unreadable.
        p.fillRect(self.rect(), COL_PANEL)
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
        p.setPen(COL_TEXT)
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
        # the same bands acc_label() actually applies (ACC_HIGH/ACC_MED) —
        # this legend used to claim 95/85 while the code banded at 85/70
        for txt, col in (("High (85-100%)", COL_GREEN),
                         ("Medium (70-84%)", COL_AMBER),
                         ("Low (0-69%)", COL_RED)):
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


_SUBJECT_VERTICAL = (
    "Vertical: estimated from the subject, because this project has no "
    "recorded vertical (calibrated before it was recorded, or the estimates "
    "disagreed too much to use). A lean held through the whole take reads as "
    "upright.")


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
        # which vertical the 3D view/export is levelled against — the answer
        # to "is the model tilted, or is that what the images show?"
        self.vertical_ref = QLabel(_SUBJECT_VERTICAL)
        self.vertical_ref.setWordWrap(True)
        self.vertical_ref.setStyleSheet("color:#8a91a3; font-size:11px;")
        self.vertical_ref.hide()
        lay.addWidget(self.vertical_ref)
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

    def show_levelling_note(self, on: bool, source: str = "subject",
                            spread_deg: float | None = None,
                            recorded=None):
        """Say which vertical the 3D view levelled on, and how sure it is.

        A recorded vertical comes with a number (the spread between the proxies
        it averages) and the number is the point: "±14 deg" is a fact the user
        can weigh, "levelled" is not.

        `recorded` is the (up, source, spread) read from the calibration
        folder, whether or not the view used it. When it exists but carries no
        spread — one proxy survived, so nothing disagreed with it and nothing
        confirmed it either — the note says UNVERIFIED. It must never say
        "±0°": that is the wording for three estimates agreeing exactly, and
        printing it for one unchecked estimate sells the weakest evidence the
        app can hold as the strongest.
        """
        if source and source != "subject" and spread_deg is not None:
            self.vertical_ref.setText(
                f"Vertical: recorded at calibration from the {source}, "
                f"±{spread_deg:.0f}° between those estimates. The 3D view and "
                f"the export both use it, so a lean held all take stays a "
                f"lean.")
        elif (recorded is not None and recorded[0] is not None
                and recorded[2] is None):
            self.vertical_ref.setText(
                f"Vertical: estimated from the subject. A vertical was "
                f"recorded at calibration (from the {recorded[1]}) but is "
                f"unverified — only one estimate, with nothing to check it "
                f"against — so it is not used. A lean held through the whole "
                f"take reads as upright.")
        else:
            self.vertical_ref.setText(_SUBJECT_VERTICAL)
        self.vertical_ref.setVisible(bool(on))

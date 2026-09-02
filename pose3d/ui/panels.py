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
    QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from pose3d.core.project import CAM_LEFT, CAM_RIGHT
from pose3d.core.skeleton import JOINT_NAMES, NUM_JOINTS

# The accuracy scale, as (residual / figure height, percent) anchors.
#
# Reprojection error in RAW PIXELS is not comparable between the two cameras of
# this rig — the subject stands 776 px tall in the left image and 407 px in the
# right, so the same physical error reads 1.9x worse on the right — and it is
# not comparable between takes at all. Every band is therefore a fraction of
# THAT camera's figure height (`pose3d.quality.figure_height_px`).
#
# The old map was 100*exp(-err_px/6): green needed <= 0.98 px on a 3072x4080
# frame, which no real take reaches, so 70.6 % of the client take's joint dots
# were red, its timeline was 21 red frames out of 26 with no green one, and the
# colour said nothing. These anchors put 0.4 % of figure height at the green
# cut and 1.0 % at the amber cut, which is where a good two-camera
# reconstruction of this subject actually sits (it reads 30.7 % red, and the
# delivered pose bands amber at 71 % / 70 %).
ACC_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.000, 100.0),
    (0.004, 85.0),
    (0.010, 70.0),
    (0.030, 0.0),
)
# The two anchors that ARE the band edges, kept named so the timeline and the
# legend cannot drift from `acc_band`.
ACC_GREEN_FRAC = ACC_ANCHORS[1][0]
ACC_AMBER_FRAC = ACC_ANCHORS[2][0]
# ...and the same two cuts as PERCENTAGES, which is the side `acc_band` works
# in. Derived from the anchors (never typed twice), computed once at import:
# `acc_band` runs per joint per repaint, and it was rebuilding two numpy
# interpolations on every one of those calls.
ACC_GREEN_PCT = ACC_ANCHORS[1][1]
ACC_AMBER_PCT = ACC_ANCHORS[2][1]

COL_GREEN = QColor(78, 214, 122)
COL_AMBER = QColor(240, 190, 74)
COL_RED = QColor(235, 92, 92)
COL_PURPLE = QColor(170, 120, 240)
# #cardPanel from dark.qss. Custom-painted widgets must fill this themselves —
# a stylesheet background is not drawn for a QWidget subclass that overrides
# paintEvent, so anything relying on it inherits the host's system colour.
COL_PANEL = QColor(15, 18, 25)
COL_TEXT = QColor(235, 238, 245)


def accuracy_pct(err_px, figure_h_px=1.0):
    """Map a reprojection error to a 0-100 accuracy percentage.

    Piecewise-linear on `err_px / figure_h_px` through `ACC_ANCHORS`. Scalar in,
    scalar out; array in, array out.

    `figure_h_px` defaults to 1.0 so a caller that already holds a NORMALISED
    residual — everything downstream of `ProjectModel._accuracy`, which divides
    by each camera's own figure height so the two cameras are comparable — can
    pass it straight in.
    """
    xs = [a for a, _ in ACC_ANCHORS]
    ys = [b for _, b in ACC_ANCHORS]
    frac = np.asarray(err_px, float) / float(figure_h_px)
    pct = np.interp(frac, xs, ys, left=ys[0], right=ys[-1])
    pct = np.where(np.isnan(frac), np.nan, np.clip(pct, 0.0, 100.0))
    return float(pct) if pct.ndim == 0 else pct


def acc_band(pct: float) -> str:
    """Accuracy band key: "green" | "amber" | "red".

    The single source of banding. `acc_color`, `acc_label` and the camera
    views' joint dots all derive from this, so a legend reword cannot silently
    change what colour a joint is drawn in.
    """
    if np.isnan(pct):
        return "red"
    return ("green" if pct >= ACC_GREEN_PCT
            else ("amber" if pct >= ACC_AMBER_PCT else "red"))


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


class _StackedRow(QWidget):
    """An info row whose value is too long to sit beside its key.

    The sidebar is 232 px wide and "50 mm · tag 15 · frame 0007" is not going
    to share a line with a label. Stacked and word-wrapped, it fits at any
    width instead of pushing the column wider than the panel and eliding.
    """

    def __init__(self, label: str, value: str = ""):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(0)
        self._k = QLabel(label); self._k.setObjectName("infoKey")
        self._v = QLabel(value); self._v.setObjectName("infoVal")
        self._v.setWordWrap(True)
        lay.addWidget(self._k); lay.addWidget(self._v)

    def set_value(self, v: str):
        self._v.setText(v)


class PoseAccuracyGauge(QWidget):
    """Circular gauge: one arc per camera, never one averaged number.

    The two cameras are 2:1 apart in resolution and see the subject from
    different angles, so a single figure hides the case that matters — one view
    agreeing and the other not. Two arcs (outer = LEFT, inner = RIGHT) and two
    percentages; the word underneath is the WORSE of the two bands, because
    that is what the reconstruction is limited by.
    """

    # camera key -> arc inset in px. Outer ring is LEFT.
    ARCS = ((CAM_LEFT, 0.0), (CAM_RIGHT, 16.0))

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self._pct: dict[str, float] = {}

    def set_cameras(self, pct: dict):
        self._pct = {k: float(v) for k, v in dict(pct).items()}
        self.update()

    def _worst(self) -> float:
        vals = [v for v in self._pct.values() if not np.isnan(v)]
        if not vals:
            return float("nan")
        return min(vals)

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
        base = QRectF((self.width() - side) / 2, 8, side, side)
        for cam, inset in self.ARCS:
            rect = base.adjusted(inset, inset, -inset, -inset)
            p.setPen(QPen(QColor(40, 44, 56), 8))
            p.drawArc(rect, 0, 360 * 16)
            val = self._pct.get(cam, float("nan"))
            if np.isnan(val):
                continue
            arc_pen = QPen(acc_color(val), 8)
            arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(arc_pen)
            # start at top (90deg), clockwise
            p.drawArc(rect, 90 * 16, -int(360 * 16 * max(val, 0.0) / 100.0))

        inner = base.adjusted(30, 30, -30, -30)
        p.setPen(COL_TEXT)
        f = QFont(); f.setPointSize(16); f.setBold(True); p.setFont(f)
        shown = " / ".join(
            "--" if np.isnan(self._pct.get(cam, float("nan")))
            else f"{self._pct[cam]:.0f}"
            for cam, _ in self.ARCS)
        p.drawText(inner, Qt.AlignmentFlag.AlignCenter, f"{shown}%")
        worst = self._worst()
        f2 = QFont(); f2.setPointSize(9); p.setFont(f2)
        p.setPen(acc_color(worst))
        sub = QRectF(base.left(), base.center().y() + 14, base.width(), 20)
        p.drawText(sub, Qt.AlignmentFlag.AlignHCenter, acc_label(worst))
        f3 = QFont(); f3.setPointSize(8); p.setFont(f3)
        p.setPen(QColor(138, 145, 163))
        legend = QRectF(base.left(), base.center().y() + 30, base.width(), 18)
        p.drawText(legend, Qt.AlignmentFlag.AlignHCenter, "L / R")


class PoseAccuracyPanel(QWidget):
    """The gauge plus the raw numbers it was banded from.

    The banding is a judgement call and the first thresholds will be wrong, so
    the fraction of figure height each percentage came from is printed beside
    it — the client can see straight through the colour to the measurement.
    """

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.addWidget(_section("POSE ACCURACY"))
        self.gauge = PoseAccuracyGauge()
        lay.addWidget(self.gauge)
        self.row_delivered = _InfoRow("Pose shown", "—")
        self.row_delivered.setToolTip(
            "How far the pose you are looking at lands from the keypoints in "
            "each view, as a fraction of the subject's height in that image.")
        lay.addWidget(self.row_delivered)
        self.row_measured = _InfoRow("Measured", "—")
        self.row_measured.setToolTip(
            "The same for the raw triangulation, before the bone-length fit. "
            "The gap between the two rows is what the fit cost.")
        lay.addWidget(self.row_measured)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setObjectName("legendLabel")
        self.note.hide()
        lay.addWidget(self.note)
        # the same bands acc_band() actually applies, stated in the units they
        # are applied in — this legend used to claim percentages with no
        # denominator at all
        for txt, col in (
                (f"High — under {100 * ACC_GREEN_FRAC:.1f} % of figure height",
                 COL_GREEN),
                (f"Medium — under {100 * ACC_AMBER_FRAC:.1f} %", COL_AMBER),
                (f"Low — over {100 * ACC_AMBER_FRAC:.1f} %", COL_RED)):
            row = QHBoxLayout()
            dot = QLabel("●"); dot.setStyleSheet(f"color: {col.name()};")
            lab = QLabel(txt); lab.setObjectName("legendLabel")
            row.addWidget(dot); row.addWidget(lab); row.addStretch(1)
            lay.addLayout(row)
        lay.addStretch(1)

    def set_accuracy(self, delivered: dict, measured: dict | None = None):
        """`delivered`/`measured` are {cam: normalised residual} scalars.

        Both are fractions of that camera's figure height, so the two cameras
        are directly comparable and neither is averaged into the other.
        """
        pct = {cam: accuracy_pct(v) for cam, v in delivered.items()}
        self.gauge.set_cameras(pct)
        self.row_delivered.set_value(_frac_row(delivered))
        self.row_measured.set_value(
            "—" if measured is None else _frac_row(measured))
        self.note.setText(_fit_cost_note(measured, delivered))
        self.note.setVisible(bool(self.note.text()))


def _frac_row(fracs: dict) -> str:
    """"0.95 % / 0.99 %" — left then right, as % of figure height."""
    order = [cam for cam, _ in PoseAccuracyGauge.ARCS]
    parts = []
    for cam in order:
        v = fracs.get(cam, float("nan"))
        parts.append("—" if np.isnan(v) else f"{100.0 * v:.2f} %")
    return " / ".join(parts)


def _fit_cost_note(measured, delivered, warn_ratio: float = 1.2) -> str:
    """What the bone fit cost, in the client's own take, or "".

    The measured-vs-delivered ratio is the permanent regression detector for
    everything downstream of triangulation: the build the client complained
    about put the delivered pose 4.2x/6.2x further from the keypoints than the
    measurement and nothing on screen said so.
    """
    if not measured or not delivered:
        return ""
    ratios = []
    for cam, d in delivered.items():
        m = measured.get(cam, float("nan"))
        if np.isfinite(m) and m > 1e-9 and np.isfinite(d):
            ratios.append(d / m)
    if not ratios or max(ratios) < warn_ratio:
        return ""
    return (f"The pose shown sits {max(ratios):.1f}x further from the "
            f"keypoints than the measurement does — that is what holding the "
            f"character's bone lengths fixed costs on this take.")


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

    def update_errors(self, errors: np.ndarray) -> None:
        """`errors` are NORMALISED residuals (fraction of figure height).

        `ProjectModel._accuracy` divides by each camera's own figure height
        before it gets here, so a joint's number means the same thing in both
        views and in every take.

        Returns nothing: it used to return the mean accuracy for the gauge,
        and the gauge stopped taking one number for both cameras — a mean over
        joints and then over cameras is exactly the statistic that hid the
        client's problem.
        """
        errors = np.asarray(errors, float).reshape(NUM_JOINTS)
        pcts = np.asarray(accuracy_pct(errors), float)
        order = np.argsort(np.nan_to_num(pcts, nan=-1))   # worst first
        self.list.clear()
        for j in order:
            pct = pcts[j]
            shown = "--" if np.isnan(pct) else f"{pct:3.0f}%"
            item = QListWidgetItem(f"  ●  {JOINT_NAMES[j]:<15}{shown:>6}")
            item.setForeground(acc_color(pct))
            self.list.addItem(item)


_MARKER_TIP = (
    "The tag the world frame was built on, its printed edge length, and the "
    "frame it was solved in.")


_SUBJECT_VERTICAL = (
    "Vertical: estimated from the subject, because this project has no "
    "recorded vertical (calibrated before it was recorded, or the estimates "
    "disagreed too much to use). A lean held through the whole take reads as "
    "upright.")


class Sidebar(QWidget):
    runDetection = Signal()
    recalibrate = Signal()
    setScale = Signal(float)                # real subject height, in metres
    showJointsToggled = Signal(bool)
    showBonesToggled = Signal(bool)
    showBodyToggled = Signal(bool)
    showCaptureToggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(232)
        # The CALIBRATION section grew a section's worth of measurements, and
        # what it holds depends on the take (a symmetry note appears only when
        # a limb pair disagrees). Scroll rather than squeeze: a column that
        # compresses its labels to fit is how a number ends up elided to "5…"
        # on a laptop screen.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")
        scroll.viewport().setStyleSheet("background: transparent;")
        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self._scroll = scroll
        lay = QVBoxLayout(inner)
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

        # Facts the client can check against the scene with a tape measure.
        # A calibration can be self-consistent and still be the wrong SIZE —
        # the marker length is the only real-world dimension that enters it —
        # and these three rows are where that shows up.
        self.row_baseline = _InfoRow("Camera baseline", "—")
        self.row_baseline.setToolTip(
            "Distance between the two cameras, as the calibration has it. "
            "Measure it: if this is wrong, everything is wrong by the same "
            "factor.")
        lay.addWidget(self.row_baseline)
        self.row_height = _InfoRow("Subject height", "—")
        self.row_height.setToolTip(
            "Median vertical extent of the reconstructed pose, levelled — the "
            "denominator every '% of height' in this app is measured against.")
        lay.addWidget(self.row_height)
        self.row_marker = _StackedRow("Marker", "—")
        self.row_marker.setToolTip(_MARKER_TIP)
        lay.addWidget(self.row_marker)

        # --- set the scale from something actually measured ---------------
        # The whole reconstruction is only as correctly SIZED as the marker
        # length that was typed in at import. Rather than make the client
        # re-import to correct it, let them state the one distance they can
        # measure and rescale the calibration to match.
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 2, 0, 0); scale_row.setSpacing(4)
        self.scale_value = QDoubleSpinBox()
        self.scale_value.setRange(0.1, 1000.0)
        self.scale_value.setDecimals(1)
        self.scale_value.setSuffix(" cm")
        self.scale_value.setValue(11.8)
        # The last height this widget PRE-FILLED (from the reconstruction).
        # While the spinbox still holds it, a refresh may replace it; once the
        # user types their own measurement it is theirs, and pressing
        # "Recalculate 3D" before "Set scale" must not silently take it back.
        self._scale_prefill = self.scale_value.value()
        self.scale_value.setMaximumWidth(96)
        self.scale_value.setToolTip(
            "The subject's real height, measured. Applying it rescales the "
            "calibration (both camera translations and the marker length) so "
            "the reconstruction comes out this tall.")
        self.btn_scale = QPushButton("Set scale")
        self.btn_scale.setMaximumWidth(92)
        self.btn_scale.setToolTip(
            "Rescale the calibration so the reconstructed subject is exactly "
            "the height on the left, then recompute and re-save it.")
        self.btn_scale.clicked.connect(
            lambda: self.setScale.emit(self.scale_value.value() / 100.0))
        scale_row.addWidget(self.scale_value, 1)
        scale_row.addWidget(self.btn_scale)
        lay.addLayout(scale_row)

        # --- what the reconstruction says about itself --------------------
        # Reprojection cannot see any of these: it is weak by construction for
        # a freshly triangulated point, which always reprojects near its own
        # observations. Each of these three covers one of its blind spots.
        self.row_bone_cv = _StackedRow("Bone length spread", "—")
        self.row_bone_cv_caption = QLabel(
            "a rigid subject should read near 0 %")
        self.row_bone_cv_caption.setWordWrap(True)
        self.row_bone_cv_caption.setObjectName("legendLabel")
        self.row_bone_cv.setToolTip(
            "Spread of each bone's length across the take. The subject is "
            "rigid, so a rigid subject should read near 0 % — anything else "
            "is reconstruction error, and this moves on a wrong camera pose "
            "when the gauge barely does.")
        lay.addWidget(self.row_bone_cv)
        lay.addWidget(self.row_bone_cv_caption)
        self.row_epipolar = _StackedRow("View disagreement", "—")
        self.row_epipolar.setToolTip(
            "How far each keypoint sits from where the other camera says it "
            "must be, per image and as a fraction of that image's diagonal. "
            "The body keypoints play no part in the calibration, so this is "
            "an independent check on it — and it moves on a wrong focal "
            "length, which the bone spread does not.")
        lay.addWidget(self.row_epipolar)
        self.row_gate = _StackedRow("Cross-view gate", "—")
        self.row_gate.setToolTip(
            "How far the two views may disagree about one keypoint before it "
            "is refused (drawn purple, not triangulated). Sized from THIS "
            "take: six times its own median disagreement, floored at 25 px "
            "and capped at 1.4 % of the smaller image's diagonal. The fixed "
            "1.4 %-of-the-left-image rule it replaces was 71.5 px here — 14.6x "
            "the median, wide enough to pass a rig seven degrees out and to "
            "miss an ankle detected on the knee in 11 of 26 frames.")
        lay.addWidget(self.row_gate)
        self.row_symmetry = _InfoRow("L/R symmetry", "—")
        lay.addWidget(self.row_symmetry)
        self.symmetry_note = QLabel("")
        self.symmetry_note.setWordWrap(True)
        self.symmetry_note.setStyleSheet("color:#e0a33a; font-size:11px;")
        self.symmetry_note.hide()
        lay.addWidget(self.symmetry_note)

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

    # --- ruler-checkable facts ------------------------------------------
    def set_metric_facts(self, baseline_m=None, subject_height_m=None,
                         marker=None, marker_error: str = ""):
        """Baseline, reconstructed subject height, and the marker provenance.

        `marker` is the calibration report's dict (or None for a project
        calibrated before the report existed — those simply say "—" rather
        than inventing a tag id). `marker_error` is the sentence for a report
        that exists and could not be read, which is not the same fact and must
        not read as "no marker was recorded".
        """
        self.row_baseline.set_value(_cm(baseline_m))
        self.row_height.set_value(_cm(subject_height_m))
        if (subject_height_m is not None and np.isfinite(subject_height_m)
                and self.scale_value.value() == self._scale_prefill):
            self.scale_value.setValue(round(100.0 * subject_height_m, 1))
            self._scale_prefill = self.scale_value.value()
        self.row_marker.set_value(marker_error or _marker_text(marker))
        self.row_marker.setToolTip(marker_error or _MARKER_TIP)

    # --- what the reconstruction says about itself -----------------------
    def set_quality(self, q, notes=()):
        """Fill the three quality rows from a `pose3d.quality.TakeQuality`.

        `None` blanks them: a take with no calibration has no such numbers,
        and a stale row is worse than an empty one.
        """
        if q is None:
            for row in (self.row_bone_cv, self.row_epipolar, self.row_gate,
                        self.row_symmetry):
                row.set_value("—")
            self.symmetry_note.hide()
            return
        cv = q.bone_cv or {}
        self.row_bone_cv.set_value(
            f"{_pct1(cv.get('median_cv_pct'))} med / "
            f"{_pct1(cv.get('max_cv_pct'))} max")
        per_image = (q.epipolar or {}).get("per_image", {})
        parts = []
        for cam in (CAM_LEFT, CAM_RIGHT):
            d = per_image.get(cam) or {}
            px, frac = d.get("median_px"), d.get("median_pct_diag")
            if px is None or not np.isfinite(px):
                parts.append("—")
            else:
                parts.append(f"{px:.1f} px ({_pct1(frac, 2)})")
        self.row_epipolar.set_value(" / ".join(parts))
        self.row_gate.set_value(_gate_text(q.epipolar or {}))
        asym = [v.get("asym_pct") for v in (q.symmetry or {}).values()]
        asym = [a for a in asym if a is not None and np.isfinite(a)]
        self.row_symmetry.set_value(
            f"{max(asym):.1f} % worst" if asym else "—")
        notes = list(notes)
        self.symmetry_note.setText("\n\n".join(f"• {n}" for n in notes))
        self.symmetry_note.setToolTip("\n\n".join(notes))
        self.symmetry_note.setVisible(bool(notes))


def _gate_text(epi: dict) -> str:
    """The gate and the distribution it was sized from, in one line.

    `threshold_px` is the gate `pose3d.quality` counted `frac_over_threshold`
    against, which is the gate `validate_cross_view` runs — one number from
    one place, so the sidebar cannot quote a threshold nothing was measured
    against.
    """
    med, thr = epi.get("median_px"), epi.get("threshold_px")
    if med is None or thr is None or not np.isfinite(med) or not np.isfinite(thr):
        return "—"
    mx = epi.get("max_px")
    tail = f", max {mx:.1f}" if mx is not None and np.isfinite(mx) else ""
    return f"{thr:.0f} px — median {med:.1f}{tail} px"


def _cm(v) -> str:
    return ("—" if v is None or not np.isfinite(v)
            else f"{100.0 * float(v):.1f} cm")


def _pct1(v, nd: int = 1) -> str:
    return ("—" if v is None or not np.isfinite(v)
            else f"{float(v):.{nd}f} %")


def _marker_text(marker) -> str:
    """"50 mm · tag 15 · frame 0007" from a calibration report."""
    if not marker:
        return "—"
    bits = []
    length = marker.get("marker_length_m")
    if length is not None and np.isfinite(length):
        bits.append(f"{1000.0 * float(length):.0f} mm")
    tag = marker.get("world_tag_id")
    if tag is not None:
        bits.append(f"tag {tag}")
    frame = marker.get("world_frame_id")
    if frame:
        bits.append(f"frame {frame}")
    return " · ".join(bits) or "—"

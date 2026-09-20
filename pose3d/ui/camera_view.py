"""Camera panel: image + draggable joint overlay + bones.

QGraphicsView/Scene with a QGraphicsPixmapItem background and one draggable
JointItem per joint. Dragging emits (cam, joint, scene-pos); bones are
QGraphicsLineItems refreshed when either endpoint moves. RAG colouring per joint.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget,
)

from pose3d.core.skeleton import (
    BONES, HEAD_KP_NAMES, JOINT_NAMES, NUM_HEAD_KP, NUM_JOINTS)
from pose3d.ui.model import (
    STATE_NOT_MEASURED, STATE_OK, STATE_REJECTED)
from pose3d.ui.panels import (
    COL_AMBER, COL_GREEN, COL_PURPLE, COL_RED, acc_band, acc_label,
    accuracy_pct)

COL_GREY = QColor(140, 148, 166)

# Handle radii in SCREEN pixels, not image pixels: the items carry
# ItemIgnoresTransformations, so the client's 3072x4080 photograph can be
# fitted into a 400-px panel and the dot he has to hit is still ~20 px across.
# They were 6 IMAGE pixels, which drew at ~1.4 px on that photograph — "very
# very small, difficult to see" (2026-08-16), and a miss pans the image.
HANDLE_R = 10.0
# The face points stay secondary to the skeleton and sit close together
# around the head, so they are smaller — but still nothing like a pixel.
FACE_HANDLE_R = 7.0
#: The smallest on-screen hit target a handle may present, across. Stated
#: here so the test asserts the number the code promises.
MIN_GRAB_PX = 16.0

# Shared with the accuracy panels so a joint's dot, its tooltip and the
# JOINT ACCURACY list can never disagree about what "amber" means.
RAG_COLORS = {
    "green": COL_GREEN,
    "amber": COL_AMBER,
    "red": COL_RED,
    "corrected": COL_PURPLE,
    # drawn as a hollow ring, never a filled dot: this joint's 3D was
    # interpolated across a one-frame dropout, not measured here.
    "filled": COL_AMBER,
    # this keypoint exists in THIS view but produced no 3D at all — the other
    # camera has nothing to pair it with. A filled dot here used to be banded
    # off the detector's confidence, so a joint with no reconstruction
    # whatsoever was drawn green.
    "unmeasured": COL_GREY,
    # both views have a point but they disagree about where the joint is by
    # more than the geometry allows, so the pair was not triangulated. PURPLE,
    # not red: red is "measured, and badly" — a whole band of the accuracy
    # scale — and this joint has no measurement to be bad. It is the same
    # colour as a hand-corrected point because both are the geometry being
    # overruled, and the two are never confusable: corrected is a filled dot,
    # rejected a hollow ring.
    "rejected": COL_PURPLE,
    # the detector found nothing here at all, so this handle is a PLACEHOLDER
    # the user drags onto the limb, not a point anything saw. RED, which the
    # accuracy bands otherwise own, for two reasons: it is the colour the
    # timeline legend already gives the word "Missing", the client's own
    # vocabulary for this; and this is the one state whose entire point is
    # that he must FIND it on the photograph, which grey on a photograph
    # loses. It cannot be confused with a red band — that is a filled dot,
    # this is a DASHED ring, and no other state is dashed. (The argument the
    # file makes for painting "rejected" purple rather than red does not
    # reach here: a rejected joint has two measurements that disagree and is
    # the geometry's problem, while this one has no measurement at all and
    # nothing but the user can fix it.)
    "missing": COL_RED,
}

# States drawn as a hollow ring rather than a filled dot: none of them is a
# measurement of this frame, and none may look like one.
HOLLOW_STATES = ("filled", "unmeasured", "rejected")

# Hollow AND dashed: nothing was detected here, so the ring is not even
# reporting a position — it is an empty slot waiting to be dragged onto one.
DASHED_STATES = ("missing",)

# The per-joint states are imported from `pose3d.ui.model`, which produces
# them, rather than restated here: two independent copies of three string
# constants desync on a typo with nothing to catch it — the dots would simply
# stop being drawn as "rejected" and no test would notice.

# The face keypoints (nose, eyes, ears) that orient the character's head.
# Drawn smaller and in one fixed accent colour: they are not part of the
# skeleton, carry no accuracy banding, and exist to be nudged when the head
# points the wrong way. Their item ids are offset by NUM_JOINTS — the single
# convention the model and the correction stack share.
#
# An item exists for all five, but which of them the user SEES is decided per
# frame by `set_pose` from the project's two head conventions, never here:
# under the "nose" head_source the nose face point and the canonical HEAD dot
# are the SAME physical detection (`ProjectModel._resolve_joint` syncs them),
# so a second dot on top of it would be one point drawn twice and draggable
# to two places; and the eyes and ears steer nothing outside Face mode.
FACE_COLOR = QColor(94, 200, 245)
FACE_KP_IDS = tuple(range(NUM_JOINTS, NUM_JOINTS + NUM_HEAD_KP))


class _JointSignals(QObject):
    moved = Signal(int, QPointF)     # joint id, new scene pos (live, during drag)
    released = Signal(int, QPointF)  # joint id, final pos (commit on mouse-up)


class JointItem(QGraphicsEllipseItem):
    R = HANDLE_R

    def __init__(self, joint_id: int, radius: float | None = None):
        r = self.R if radius is None else radius
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.joint_id = joint_id
        self.signals = _JointSignals()
        # this joint was not detected in this view and the handle is an empty
        # slot to drag onto it — set by `CameraView.set_pose`, read by the
        # colouring and by the bones, which may not span a guess
        self.is_placeholder = False
        # The radius is in SCREEN pixels: the handle keeps its size whatever
        # the view transform is, so fitting a 4080-px photo into the panel no
        # longer shrinks the grab target to a pixel. `pos()` is unaffected —
        # it stays the joint's position in image coordinates, which is what
        # the drag reports and what the model stores.
        self.setFlag(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(10)
        self.setPen(QPen(QColor(20, 20, 20), 1))
        self.setCursor(Qt.CursorShape.SizeAllCursor)   # signals "draggable"
        self.set_status("green")

    def set_status(self, status: str):
        if status in DASHED_STATES:
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            pen = QPen(RAG_COLORS[status], 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            self.setPen(pen)
            return
        if status in HOLLOW_STATES:
            self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.setPen(QPen(RAG_COLORS[status], 2))
            return
        self.setBrush(QBrush(RAG_COLORS.get(status, RAG_COLORS["red"])))
        self.setPen(QPen(QColor(20, 20, 20), 1))

    def itemChange(self, change, value):
        # live signal only updates the bone lines in-view; the model is NOT
        # touched here (doing so mid-drag re-enters itemChange -> recursion).
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            self.signals.moved.emit(self.joint_id, value)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        self._press_pos = self.pos()      # the drag-commit test in mouseRelease
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # commit to the model only if the joint actually moved (not a click)
        start = getattr(self, "_press_pos", None)
        if start is not None and (self.pos() - start).manhattanLength() > 0.5:
            self.signals.released.emit(self.joint_id, self.pos())


class CameraView(QGraphicsView):
    """One camera's editable overlay."""
    jointDragged = Signal(str, int, QPointF)   # cam, joint, pos

    def __init__(self, cam: str):
        super().__init__()
        self.cam = cam
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.OpenHandCursor)   # hint: draggable to pan
        self._pixmap_item = None
        self._joints: list[JointItem] = []
        self._face: list[JointItem] = []       # nose/eyes/ears, ids NUM_JOINTS..
        # per-face-item "this frame's conventions say draw it" flag. Kept
        # because `set_show_joints` knows nothing about either convention: it
        # may only hide dots and un-hide the ones that were shown, never
        # resurrect one this frame's mode (or a NaN) took away.
        self._face_shown = [False] * NUM_HEAD_KP
        # the same flag for the canonical joints, and for the same reason. The
        # toggle used to ask the ITEM where it was (`not isnan(pos().x())`),
        # which can never be NaN: `set_pose` skips `setPos` for a joint the
        # cameras did not see, so the item keeps its last finite position and
        # the guard passed every time. Toggling joints off and on re-showed
        # every undetected joint at a stale position, where a drag would write
        # a hand correction out of nothing. True until the first `set_pose`,
        # which is what the old guard said for a never-posed item too.
        self._joint_shown = [True] * NUM_JOINTS
        # the last position THIS view was given for each joint, so a frame
        # the detector missed can offer the placeholder where the joint was
        # last actually seen rather than in the middle of the picture
        self._last_seen: dict[int, QPointF] = {}
        self._bones: list[QGraphicsLineItem] = []
        self._show_joints = True
        self._show_bones = True
        self._panning = False
        self._pan_start = None
        self._accuracy = None       # per-joint MEASURED residual, normalised
        self._delivered = None      # per-joint DELIVERED residual, normalised
        self._states = None         # per-joint state (see STATE_* above)
        self._scores = None         # per-joint detector confidence
        self._corrected = None      # per-joint hand-corrected flags
        self._filled = None         # per-joint gap-filled (3D interpolated)
        self._build_items()

    def _build_items(self):
        for a, b in BONES:
            line = QGraphicsLineItem()
            line.setPen(QPen(QColor(220, 220, 220, 180), 2))
            line.setZValue(5)
            self._scene.addItem(line)
            self._bones.append(line)
        for j in range(NUM_JOINTS):
            item = JointItem(j)
            item.signals.moved.connect(self._on_moved_live)       # bones only
            item.signals.released.connect(self._on_released)      # commit
            self._scene.addItem(item)
            self._joints.append(item)
        for jid in FACE_KP_IDS:
            item = JointItem(jid, radius=FACE_HANDLE_R)
            item.setBrush(QBrush(FACE_COLOR))
            k = jid - NUM_JOINTS
            # the nose turns the head in BOTH modes; the eyes and ears steer
            # it only in Face mode, and the tooltip says which is which
            what = ("turns the character's head" if HEAD_KP_NAMES[k] == "nose"
                    else "orient the character's head (Face mode)")
            item.setToolTip(f"<b>{HEAD_KP_NAMES[k].upper()}</b><br>{what}")
            item.signals.released.connect(self._on_released)      # commit
            item.setVisible(False)
            self._scene.addItem(item)
            self._face.append(item)

    def set_image(self, path: str):
        pm = QPixmap(path)
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pm)
            self._pixmap_item.setZValue(0)
        else:
            self._pixmap_item.setPixmap(pm)
        if not pm.isNull():
            self._scene.setSceneRect(QRectF(pm.rect()))
            # …but only while the view is still the one WE chose. A frame
            # step reloads the image, and re-fitting here threw away the
            # user's zoom and pan every single time — so correcting the same
            # joint across ten frames meant zooming and panning ten times.
            # `_zoomed` is the flag that already means "the user has taken
            # the view over"; `resizeEvent` and the Fit button honour it too.
            if not self._zoomed:
                self.fitInView(self._pixmap_item,
                               Qt.AspectRatioMode.KeepAspectRatio)

    def set_pose(self, xy: np.ndarray, scores: np.ndarray,
                 corrected: np.ndarray | None = None,
                 head_xy: np.ndarray | None = None,
                 filled: np.ndarray | None = None,
                 head_source: str | None = None,
                 head_mode: str = "nose"):
        """Place joints from (NUM_JOINTS,2) pixel coords + scores.

        `head_xy` is the optional (NUM_HEAD_KP,2) face keypoints, drawn as
        small draggable dots. `head_source` and `head_mode` are the project's
        two head conventions and they decide which of the five are drawn: the
        NOSE only under "skull", where it is a different detection from the
        canonical HEAD dot (under "nose" they are the same point, and drawing
        both would let the user drag one detection to two places); the eyes
        and ears only in Face mode, the only mode in which they orient
        anything. They stay detected and stored in both modes either way.

        `filled` flags joints whose 3D was interpolated across a one-frame
        dropout (pipeline.fill_gaps); they are drawn as hollow rings.
        """
        self._scores = np.asarray(scores, float)
        self._corrected = corrected
        self._filled = filled
        for item in self._face:
            k = item.joint_id - NUM_JOINTS
            q = None if head_xy is None else head_xy[k]
            wanted = (head_source == "skull" if HEAD_KP_NAMES[k] == "nose"
                      else head_mode == "face")
            if not wanted or q is None or np.isnan(q).any():
                self._face_shown[k] = False
                item.setVisible(False)
                continue
            self._face_shown[k] = True
            item.setVisible(self._show_joints)
            item.signals.blockSignals(True)
            item.setPos(float(q[0]), float(q[1]))
            item.signals.blockSignals(False)
        for j, item in enumerate(self._joints):
            p = xy[j]
            # suppress the move signal while we set position programmatically
            # (QGraphicsItem is not a QObject; the Signal lives on item.signals)
            item.signals.blockSignals(True)
            if np.isnan(p).any():
                # The detector found nothing for this joint HERE. Hiding it
                # was the same as deleting it: an invisible QGraphicsItem is
                # not hit-tested and a drag is the only way into a correction,
                # so the frames the tool exists for — occlusion, extreme poses
                # — were exactly the ones the user could not fix. Offer an
                # empty, dashed slot instead, where the joint was last seen in
                # this view (else the middle of the picture), and let him drag
                # it onto the limb: that drag commits through the ordinary
                # `released` path, so the model records it as a correction.
                self._joint_shown[j] = True
                item.is_placeholder = True
                item.setPos(self._placeholder_pos(j))
            else:
                self._joint_shown[j] = True
                item.is_placeholder = False
                item.setPos(float(p[0]), float(p[1]))
                self._last_seen[j] = QPointF(float(p[0]), float(p[1]))
            item.setVisible(self._show_joints)
            item.signals.blockSignals(False)
        self._apply_status()
        self._refresh_bones()

    def _placeholder_pos(self, j: int) -> QPointF:
        """Where to park the handle for a joint this view did not detect.

        The last position this view had for the joint — for a dropout mid-take
        that is the previous frame's, a few pixels from where the limb really
        is — else the middle of the image, the one point always on screen.
        """
        prev = self._last_seen.get(j)
        if prev is not None:
            return QPointF(prev)
        if self._pixmap_item is not None:
            r = self._pixmap_item.boundingRect()
            if r.width() and r.height():
                return r.center()
        return self._scene.sceneRect().center()

    def set_accuracy(self, errors, delivered=None, states=None) -> None:
        """Per-joint residuals for THIS view, normalised by figure height.

        `errors` is the MEASURED residual (the raw triangulation reprojected
        into this view) — the one a drag of this very keypoint can drive to
        zero, which is why it is what the dot is coloured by. `delivered` is
        the same for the pose actually shown in 3D, reported in the tooltip so
        the gap the bone fit costs is visible at the joint that pays it.
        `states` is the per-joint state array from `ProjectModel.joint_states`.
        """
        def arr(v):
            return None if v is None else np.asarray(v, float)
        self._accuracy = arr(errors)
        self._delivered = arr(delivered)
        self._states = None if states is None else list(states)
        self._apply_status()

    def _apply_status(self) -> None:
        """Colour each joint and set its hover tooltip.

        Accuracy (reprojection error, as a fraction of the subject's height in
        THIS image) is what the user is actually judging when correcting a
        pose, so it drives the colour. Hovering names the joint and gives the
        number, replacing the old SELECTED JOINT panel — the info appears
        where the user is already looking.
        """
        for j, item in enumerate(self._joints):
            if item.is_placeholder:
                # nothing was detected, so there is no accuracy, no
                # confidence and no band to report — only the one thing the
                # user can do about it
                item.set_status("missing")
                item.setToolTip(
                    f"<b>{JOINT_NAMES[j]}</b><br>"
                    f"<span style='color:{RAG_COLORS['missing'].name()};'>"
                    f"not detected — drag to place</span>")
                continue
            status, tip = self._joint_status(j)
            item.set_status(status)
            item.setToolTip(tip)

    def _joint_status(self, j: int) -> tuple[str, str]:
        """(band key, tooltip html) for one joint.

        A joint that produced no 3D is NOT banded: it is painted as its own
        state, because "no reconstruction" is a different fact from "a poor
        one" and the two used to be drawn identically — the detector's
        confidence stood in for an accuracy that did not exist, so a joint
        with no 3D at all could show a green dot at 90 % confidence.
        """
        name = JOINT_NAMES[j]
        state = (self._states[j] if self._states is not None
                 and j < len(self._states) else STATE_OK)
        err = float(self._accuracy[j]) if self._accuracy is not None else np.nan

        if state == STATE_REJECTED:
            status = "rejected"
            # The NUMBER is the diagnosis. A purple dot on its own is a new
            # kind of silence: the user cannot tell a hallucinated ankle from
            # a rig that is 12 deg out, and those want opposite responses
            # (drag the point vs recalibrate). `px`/`gate` ride on the state
            # itself (ui.model.RejectedState); a state without them still
            # renders, it just cannot say how far apart the views were.
            px, gate = getattr(state, "px", None), getattr(state, "gate", None)
            how_far = (f"by {px:.0f} px (gate {gate:.0f} px)"
                       if px is not None and gate is not None
                       and np.isfinite(px) and np.isfinite(gate)
                       else "by more than the calibration allows")
            detail = (f"rejected by the cross-view check — the two views "
                      f"disagree about where this joint is {how_far}, so it "
                      f"was not triangulated. The keypoints are still here: "
                      f"drag either one, or recalibrate")
        elif state == STATE_NOT_MEASURED or not np.isfinite(err):
            status = "unmeasured"
            detail = ("not measured — no 3D was reconstructed for this joint, "
                      "so there is no accuracy to report")
        else:
            pct = accuracy_pct(err)
            status = acc_band(pct)
            detail = (f"accuracy {pct:.0f}% ({acc_label(pct)}) — "
                      f"{100.0 * err:.2f}% of the figure's height in this view")

        # The extra lines FIRST, because two of them change what the joint is —
        # and the headline is coloured with the joint's final state. Built the
        # other way round, an interpolated or hand-corrected joint drew an
        # amber/blue dot while its tooltip's first line kept the accuracy
        # band's colour, so the dot and the words disagreed about what it is.
        more = []
        if self._delivered is not None and np.isfinite(self._delivered[j]):
            more.append(f"<span style='color:#8a91a3;'>pose shown: "
                        f"{100.0 * float(self._delivered[j]):.2f}% of height"
                        f"</span>")
        if self._scores is not None and np.isfinite(self._scores[j]):
            more.append(f"<span style='color:#8a91a3;'>detector confidence "
                        f"{100.0 * float(self._scores[j]):.0f}%</span>")
        if self._filled is not None and self._filled[j]:
            status = "filled"
            more.append(f"<span style='color:{RAG_COLORS['filled'].name()};'>"
                        f"3D interpolated — this joint was missing for one "
                        f"frame</span>")
        if self._corrected is not None and self._corrected[j]:
            status = "corrected"
            more.append(f"<span style='color:{RAG_COLORS['corrected'].name()};'>"
                        f"corrected by hand</span>")

        tip = (f"<b>{name}</b><br>"
               f"<span style='color:{RAG_COLORS[status].name()};'>{detail}</span>")
        return status, "<br>".join([tip, *more])

    def _on_moved_live(self, joint_id: int, pos: QPointF):
        # cheap live feedback during the drag: just redraw the bone lines
        self._refresh_bones()

    def _on_released(self, joint_id: int, pos: QPointF):
        # commit the final position to the model once, on mouse-up
        self.jointDragged.emit(self.cam, joint_id, pos)

    def _refresh_bones(self):
        for (a, b), line in zip(BONES, self._bones):
            ja, jb = self._joints[int(a)], self._joints[int(b)]
            # a bone to a placeholder would draw a limb out of a guess: the
            # handle is parked where the joint last was, not where it is
            if (not (ja.isVisible() and jb.isVisible()) or not self._show_bones
                    or ja.is_placeholder or jb.is_placeholder):
                line.setVisible(False)
                continue
            line.setVisible(True)
            pa, pb = ja.pos(), jb.pos()
            line.setLine(pa.x(), pa.y(), pb.x(), pb.y())

    def set_show_joints(self, on: bool):
        self._show_joints = on
        # Both sets go by what the last `set_pose` decided: this toggle knows
        # nothing about a dropout or the head conventions and may not overrule
        # either. It may only hide, and un-hide what `set_pose` showed.
        for shown, it in zip(self._joint_shown, self._joints):
            it.setVisible(on and shown)
        for shown, it in zip(self._face_shown, self._face):
            it.setVisible(on and shown)
        self._refresh_bones()

    def set_show_bones(self, on: bool):
        self._show_bones = on
        self._refresh_bones()

    def resizeEvent(self, event):
        if self._pixmap_item is not None and not self._zoomed:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

    # --- toolbar actions ---
    _zoomed = False

    def zoom(self, factor: float):
        self._zoomed = True
        self.scale(factor, factor)

    def fit(self):
        self._zoomed = False
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)


    # --- panning: left-drag on empty area pans; left-drag on a joint moves it ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(
                self.itemAt(event.position().toPoint()), JointItem):
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            pos = event.position().toPoint()
            delta = pos - self._pan_start
            self._pan_start = pos
            h, v = self.horizontalScrollBar(), self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CameraPanel(QWidget):
    """Camera view with a header (title + filename) and a vertical toolbar."""

    def __init__(self, cam: str, title: str):
        super().__init__()
        self.view = CameraView(cam)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        tlab = QLabel(title); tlab.setObjectName("panelTitle")
        self.filename = QLabel(""); self.filename.setObjectName("fileLabel")
        header.addWidget(tlab); header.addStretch(1); header.addWidget(self.filename)
        outer.addLayout(header)

        body = QHBoxLayout(); body.setSpacing(4)
        tools = QVBoxLayout(); tools.setSpacing(4)
        # Left-drag on empty area pans, on a joint moves it — no mode needed.
        specs = [("＋", "Zoom in", lambda: self.view.zoom(1.25)),
                 ("－", "Zoom out", lambda: self.view.zoom(1 / 1.25)),
                 ("⤢", "Fit", self.view.fit)]
        for glyph, tip, fn in specs:
            b = QToolButton(); b.setText(glyph); b.setToolTip(tip)
            b.setObjectName("camTool"); b.clicked.connect(fn)
            tools.addWidget(b)
        tools.addStretch(1)
        body.addLayout(tools)
        body.addWidget(self.view, 1)
        outer.addLayout(body, 1)

    def set_filename(self, name: str):
        self.filename.setText(name)

    def set_accuracy(self, errors, delivered=None, states=None) -> None:
        self.view.set_accuracy(errors, delivered, states)

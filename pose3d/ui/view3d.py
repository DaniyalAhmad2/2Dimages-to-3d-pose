"""3D skeleton preview using pyqtgraph.opengl.

Renders a semi-transparent grey "shadow figure" (a ghost body built from
capsules along the bones + spheres at the joints) with the coloured skeleton
(joints + bones) overlaid on top — matching the mockup's grey mannequin.

The pose stands ON the grid (grid = ground): the world up-axis is detected from
the skeleton (head vs ankles), mapped to view +Z, centred horizontally, and the
sole beneath the lower ankle dropped to z = 0 (see `ground_datum`).
Unit-agnostic (metres or centimetres).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint
from pose3d.geometry.orient import detect_vertical, upright_matrix


def ground_datum(verts, joints, drop):
    """View-space z of the ground plane under a posed character.

    The SOLE beneath the lower ANKLE, not the lowest mesh vertex. Which vertex
    is lowest changes from frame to frame — a foot, a knee, a fingertip — so
    the old rule slid the ground plane about under the figure and it bobbed
    against the grid by up to 11 % of body height. The ankles are tracked
    joints, so this datum moves only when the subject does. `drop` is the rig's
    rest ankle-to-sole height in these same units (`Character.ground_drop`).
    Falls back to the old rule when neither ankle could be posed.

    The sole is deliberately NOT levelled onto the plane: the shin of this
    rigid-footed mannequin genuinely tilts 16-86 deg, and flattening the foot
    would replace a measurement with a convention.
    """
    if joints is not None:
        z = [joints[int(j)][2]
             for j in (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)]
        z = [q for q in z if np.isfinite(q)]
        if z:
            return float(min(z)) - float(drop)
    return float(verts[:, 2].min())


class View3D(gl.GLViewWidget):
    GHOST_COLOR = (0.72, 0.72, 0.77, 0.85)     # low-poly character skin
    JOINT_COLOR = (0.30, 0.85, 1.0, 1.0)
    BONE_COLOR = (0.95, 0.95, 0.98, 1.0)
    CAPTURE_COLOR = (1.0, 0.72, 0.25, 0.85)    # the measured skeleton
    FILLED_COLOR = (1.0, 0.62, 0.10, 1.0)      # interpolated, not measured

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((14, 16, 22))
        self.setCameraPosition(distance=4.0, elevation=14, azimuth=-70)
        self._grid = gl.GLGridItem()          # XY plane, z = 0 (the ground)
        self._grid.setSize(2, 2)
        self._grid.setSpacing(0.2, 0.2)
        self.addItem(self._grid)

        # --- smooth human body surface (metaball skin around the skeleton) ---
        self._body = gl.GLMeshItem(
            vertexes=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]),
            smooth=True, color=self.GHOST_COLOR, shader="shaded",
            glOptions="translucent", drawEdges=False)
        self._body.setVisible(False)
        self.addItem(self._body)

        # --- the CHARACTER's own skeleton (drawn on top of the mesh) ---
        # Read off the posed rig rather than from the triangulated points, so
        # the overlay always sits inside the body it belongs to.
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=11.0, color=self.JOINT_COLOR, pxMode=True)
        self.addItem(self._scatter)
        self._lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), width=2.5, color=self.BONE_COLOR, mode="lines")
        self.addItem(self._lines)

        # --- the CAPTURED skeleton (what the cameras measured), off by default.
        # Kept available so the fit can be judged, in a dimmer colour so it
        # reads as reference rather than as the result.
        self._cap_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=8.0, color=self.CAPTURE_COLOR, pxMode=True)
        self._cap_scatter.setVisible(False)
        self.addItem(self._cap_scatter)
        self._cap_lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), width=1.5, color=self.CAPTURE_COLOR, mode="lines")
        self._cap_lines.setVisible(False)
        self.addItem(self._cap_lines)

        self._show_body = True
        self._show_capture = False
        self._character = None          # lazily-loaded skinned character
        self._framed = False
        self._vaxis = None
        self._vsign = 1.0
        self._R = None                  # world->view rotation (sequence de-tilt)

    # --- orientation / framing ---
    def _detect_vertical(self, pose3d, valid):
        return detect_vertical(pose3d, valid)

    def set_orientation(self, R):
        """Set an explicit world->view rotation (3x3), e.g. a whole-sequence
        de-tilt so the figure stands upright. None -> per-frame auto-detect."""
        self._R = None if R is None else np.asarray(R, float).reshape(3, 3)
        self._vaxis = None
        self._framed = False

    def fit_subject(self, poses):
        """Size the character to the subject once, from the whole take.

        One uniform scale, so the character changes size but never shape.
        `poses` are raw world poses; they get de-tilted here exactly as
        set_pose() does."""
        try:
            if self._character is None:
                from pose3d.geometry.character import Character
                self._character = Character()
            poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
            if self._R is not None:
                poses = poses @ self._R.T
            self._character.fit_to_subject(poses)
        except Exception:
            pass

    def _to_view(self, pts):
        """World -> view rotation. Shape-agnostic: used for the canonical
        joints and for the face keypoints alike."""
        if self._R is not None:
            return pts @ self._R.T
        return pts @ upright_matrix(self._vaxis, self._vsign).T

    # --- public API ---
    def set_show_body(self, on: bool):
        self._show_body = on
        self._body.setVisible(on and self._body.opts.get("_active", False))

    def set_show_capture(self, on: bool):
        """Show the skeleton the cameras measured, alongside the character's."""
        self._show_capture = on
        self._cap_scatter.setVisible(on)
        self._cap_lines.setVisible(on)

    def reframe(self):
        self._framed = False
        self._vaxis = None

    def set_projection(self, mode: str):
        """'Perspective' or 'Orthographic' (approximated via a narrow FOV)."""
        self.opts["fov"] = 1.0 if mode.lower().startswith("ortho") else 60.0
        self.update()

    def set_pose(self, pose3d: np.ndarray, head3d: np.ndarray | None = None,
                 filled: np.ndarray | None = None):
        """`head3d` is the optional (NUM_HEAD_KP, 3) face keypoints; with them
        the character's head is oriented rather than left riding the neck.

        `filled` is the optional (NUM_JOINTS,) flag array from
        `pipeline.fill_gaps`: those joints were interpolated across a
        one-frame dropout, and are drawn in a distinct amber so a fill is
        never read as a measurement."""
        pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
        valid = ~np.isnan(pose3d).any(1)
        if not valid.any():
            self._clear()
            return

        if self._R is None and self._vaxis is None:
            self._vaxis, self._vsign = self._detect_vertical(pose3d, valid)

        v = self._to_view(pose3d)
        vv = v[valid]
        # centre horizontally; ground tentatively on the lowest joint (the ankle,
        # since feet aren't detected)
        cx, cy = vv[:, 0].mean(), vv[:, 1].mean()
        v[:, 0] -= cx; v[:, 1] -= cy; v[:, 2] -= vv[:, 2].min()

        # pose the character, then ground it on the sole under its lower ankle
        # so the feet rest ON the plane instead of the ankle (feet would pierce
        # it) without the ground sliding about frame to frame.
        vpose = np.where(valid[:, None], v, np.nan)
        # Same world->view rotation as the pose; the grounding translation is
        # deliberately NOT applied, because only a direction basis is read off
        # these and directions are translation-invariant.
        vhead = self._to_view(head3d) if head3d is not None else None
        verts, faces, cj, drop = self._skin(vpose, vhead)
        if verts is not None and len(verts):
            dz = ground_datum(verts, cj, drop)
            verts = verts.copy(); verts[:, 2] -= dz
            v[:, 2] -= dz
            if cj is not None:
                cj = cj.copy(); cj[:, 2] -= dz          # keep the overlay with the body

        # The character is the subject of the view, so frame on it; fall back to
        # the captured points when it could not be posed.
        ref = verts if (verts is not None and len(verts)) else v[valid]
        height = float(ref[:, 2].max() - ref[:, 2].min()) or 1.0

        # primary overlay: the character's OWN joints, which by construction lie
        # inside the mesh. Falls back to the captured points if there's no rig.
        if cj is not None:
            self._draw_skeleton(self._scatter, self._lines, cj,
                                ~np.isnan(cj).any(1), self.JOINT_COLOR, filled)
        else:
            self._draw_skeleton(self._scatter, self._lines, v, valid,
                                self.JOINT_COLOR, filled)

        # Reference overlay: what the cameras actually MEASURED — so a joint
        # that was interpolated across a dropout is simply absent from it,
        # rather than sitting there in the same amber as everything else.
        measured = valid if filled is None else valid & ~np.asarray(filled, bool)
        self._draw_skeleton(self._cap_scatter, self._cap_lines, v, measured,
                            self.CAPTURE_COLOR)
        self._set_body(verts, faces)

        if not self._framed:
            span = float(np.linalg.norm(ref.max(0) - ref.min(0))) or 1.0
            self._grid.setSize(span * 1.6, span * 1.6)
            self._grid.setSpacing(span / 8.0, span / 8.0)
            self.setCameraPosition(pos=Vector(0, 0, height * 0.5),
                                   distance=span * 1.9, elevation=12, azimuth=-70)
            self._framed = True

    @classmethod
    def _draw_skeleton(cls, scatter, lines, pts, valid, base_color=None,
                       filled=None):
        idx = np.flatnonzero(valid)
        if idx.size and base_color is not None:
            colors = np.tile(np.asarray(base_color, float), (idx.size, 1))
            if filled is not None:
                colors[np.asarray(filled, bool)[idx]] = cls.FILLED_COLOR
            scatter.setData(pos=pts[idx], color=colors)
        else:
            scatter.setData(pos=pts[idx] if idx.size else np.zeros((1, 3)))
        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(pts[int(a)]); seg.append(pts[int(b)])
        lines.setData(pos=np.array(seg) if seg else np.zeros((2, 3)))

    def _skin(self, vpose, vhead=None):
        """Pose the character -> (verts, faces, canonical joints, ground drop).

        The ground drop is computed HERE, inside the blanket guard, rather than
        at the call site: it is a `Character` call like the skinning itself, so
        a character failure must land in the same place instead of escaping
        `set_pose` into the Qt slot.
        """
        try:
            if self._character is None:
                from pose3d.geometry.character import Character
                self._character = Character()
            valid = ~np.isnan(vpose).any(1)
            verts, faces, cj = self._character.pose_and_joints(
                vpose, valid, vhead)
            return verts, faces, cj, self._character.ground_drop(vpose, valid)
        except Exception:
            return None, None, None, 0.0

    def _set_body(self, verts, faces):
        active = verts is not None and len(verts) > 0
        self._body.opts["_active"] = active
        if active:
            self._body.setMeshData(vertexes=verts, faces=faces)
        self._body.setVisible(self._show_body and active)

    def _clear(self):
        for s, l in ((self._scatter, self._lines),
                     (self._cap_scatter, self._cap_lines)):
            s.setData(pos=np.zeros((1, 3)))
            l.setData(pos=np.zeros((2, 3)))
        self._body.opts["_active"] = False
        self._body.setVisible(False)

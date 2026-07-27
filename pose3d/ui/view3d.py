"""3D skeleton preview using pyqtgraph.opengl.

Renders a semi-transparent grey "shadow figure" (a ghost body built from
capsules along the bones + spheres at the joints) with the coloured skeleton
(joints + bones) overlaid on top — matching the mockup's grey mannequin.

The pose stands ON the grid (grid = ground): the world up-axis is detected from
the skeleton (head vs ankles), mapped to view +Z, centred horizontally, and its
lowest joint dropped to z = 0. Unit-agnostic (metres or centimetres).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint
from pose3d.geometry.orient import detect_vertical, upright_matrix


class View3D(gl.GLViewWidget):
    GHOST_COLOR = (0.72, 0.72, 0.77, 0.85)     # low-poly character skin
    JOINT_COLOR = (0.30, 0.85, 1.0, 1.0)
    BONE_COLOR = (0.95, 0.95, 0.98, 1.0)

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

        # --- coloured skeleton overlay (drawn on top) ---
        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=11.0, color=self.JOINT_COLOR, pxMode=True)
        self.addItem(self._scatter)
        self._lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), width=2.5, color=self.BONE_COLOR, mode="lines")
        self.addItem(self._lines)

        self._show_body = True
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

    def _to_view(self, pose3d):
        if self._R is not None:
            return pose3d @ self._R.T
        return pose3d @ upright_matrix(self._vaxis, self._vsign).T

    # --- public API ---
    def set_show_body(self, on: bool):
        self._show_body = on
        self._body.setVisible(on and self._body.opts.get("_active", False))

    def reframe(self):
        self._framed = False
        self._vaxis = None

    def set_projection(self, mode: str):
        """'Perspective' or 'Orthographic' (approximated via a narrow FOV)."""
        self.opts["fov"] = 1.0 if mode.lower().startswith("ortho") else 60.0
        self.update()

    def set_pose(self, pose3d: np.ndarray):
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

        # skin the character, then ground on ITS lowest vertex (the sole) so the
        # feet rest ON the plane instead of the ankle (feet would pierce it).
        vpose = np.where(valid[:, None], v, np.nan)
        verts, faces = self._skin(vpose)
        if verts is not None and len(verts):
            dz = float(verts[:, 2].min())
            v[:, 2] -= dz
            verts = verts.copy(); verts[:, 2] -= dz

        height = float(v[valid][:, 2].max() - v[valid][:, 2].min()) or 1.0

        # coloured skeleton
        self._scatter.setData(pos=v[valid])
        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(v[int(a)]); seg.append(v[int(b)])
        self._lines.setData(pos=np.array(seg) if seg else np.zeros((2, 3)))
        self._set_body(verts, faces)

        if not self._framed:
            vv = v[valid]
            span = float(np.linalg.norm(vv.max(0) - vv.min(0))) or 1.0
            self._grid.setSize(span * 1.6, span * 1.6)
            self._grid.setSpacing(span / 8.0, span / 8.0)
            self.setCameraPosition(pos=Vector(0, 0, height * 0.5),
                                   distance=span * 1.9, elevation=12, azimuth=-70)
            self._framed = True

    def _skin(self, vpose):
        """Skin the bundled character to the (upright, centred) pose -> verts."""
        try:
            if self._character is None:
                from pose3d.geometry.character import Character
                self._character = Character()
            valid = ~np.isnan(vpose).any(1)
            return self._character.pose(vpose, valid)
        except Exception:
            return None, None

    def _set_body(self, verts, faces):
        active = verts is not None and len(verts) > 0
        self._body.opts["_active"] = active
        if active:
            self._body.setMeshData(vertexes=verts, faces=faces)
        self._body.setVisible(self._show_body and active)

    def _clear(self):
        self._scatter.setData(pos=np.zeros((1, 3)))
        self._lines.setData(pos=np.zeros((2, 3)))
        self._body.opts["_active"] = False
        self._body.setVisible(False)

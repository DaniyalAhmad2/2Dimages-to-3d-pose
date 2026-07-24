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


def upright_matrix(axis, sign):
    """3x3 matrix mapping world coords to upright view coords (up-axis -> +Z).

    Guaranteed to be a proper ROTATION (det=+1): one horizontal axis is flipped
    when the naive axis-permutation would be a reflection, so the figure is
    never left/right mirrored (a raised left hand stays a left hand).
    """
    others = [i for i in range(3) if i != axis]
    perm_parity = -1.0 if axis == 1 else 1.0
    hx = sign * perm_parity
    M = np.zeros((3, 3))
    M[0, others[0]] = hx
    M[1, others[1]] = 1.0
    M[2, axis] = sign
    return M


class View3D(gl.GLViewWidget):
    GHOST_COLOR = (0.66, 0.68, 0.74, 0.45)     # translucent grey ghost body
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
        self._framed = False
        self._vaxis = None
        self._vsign = 1.0

    # --- orientation / framing ---
    def _detect_vertical(self, pose3d, valid):
        """Find the world up-axis from head vs the lowest available body joint.

        Ankles can be dropped (occlusion gating), so fall back through
        knees -> pelvis -> hips to keep the figure upright.
        """
        # "up" reference (head) vs a lower-body reference that is present
        head = pose3d[int(Joint.HEAD)]
        if np.isnan(head).any():
            head = np.nanmean(pose3d[[int(Joint.NECK), int(Joint.HEAD)]], axis=0)
        ref = None
        for idxs in ([Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE],
                     [Joint.LEFT_KNEE, Joint.RIGHT_KNEE],
                     [Joint.PELVIS],
                     [Joint.LEFT_HIP, Joint.RIGHT_HIP]):
            pts = pose3d[[int(i) for i in idxs]]
            if np.isnan(pts).all():          # avoid empty-slice nanmean warning
                continue
            cand = np.nanmean(pts, axis=0)
            if not np.isnan(cand).any():
                ref = cand
                break
        if ref is not None and not np.isnan(head).any():
            diff = head - ref
            axis = int(np.argmax(np.abs(diff)))
            return axis, float(np.sign(diff[axis]) or 1.0)
        vpts = pose3d[valid]
        return int(np.argmax(vpts.max(0) - vpts.min(0))), 1.0

    def _to_view(self, pose3d):
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

        if self._vaxis is None:
            self._vaxis, self._vsign = self._detect_vertical(pose3d, valid)

        v = self._to_view(pose3d)
        vv = v[valid]
        cx, cy, floor = vv[:, 0].mean(), vv[:, 1].mean(), vv[:, 2].min()
        v[:, 0] -= cx; v[:, 1] -= cy; v[:, 2] -= floor

        height = float(vv[:, 2].max() - vv[:, 2].min()) or 1.0

        # coloured skeleton
        self._scatter.setData(pos=v[valid])
        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(v[int(a)]); seg.append(v[int(b)])
        self._lines.setData(pos=np.array(seg) if seg else np.zeros((2, 3)))

        # smooth human body surface (metaball skin) around the skeleton
        vpose = np.where(valid[:, None], v, np.nan)
        self._update_body(vpose)

        if not self._framed:
            span = float(np.linalg.norm(vv.max(0) - vv.min(0))) or 1.0
            self._grid.setSize(span * 1.6, span * 1.6)
            self._grid.setSpacing(span / 8.0, span / 8.0)
            self.setCameraPosition(pos=Vector(0, 0, height * 0.5),
                                   distance=span * 1.9, elevation=12, azimuth=-70)
            self._framed = True

    def _update_body(self, vpose):
        """Rebuild the smooth body surface from the (upright, centred) pose."""
        try:
            from pose3d.geometry.bodymesh import human_body_mesh
            verts, faces = human_body_mesh(vpose, resolution=42)
        except Exception:
            verts = None
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

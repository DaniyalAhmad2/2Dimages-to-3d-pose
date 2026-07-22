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
from pyqtgraph import Transform3D, Vector
from PySide6.QtGui import QMatrix4x4

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint

# thicker body parts (torso / hips / shoulders); everything else is a limb
_THICK_BONES = {
    (int(Joint.PELVIS), int(Joint.NECK)),
    (int(Joint.PELVIS), int(Joint.LEFT_HIP)),
    (int(Joint.PELVIS), int(Joint.RIGHT_HIP)),
    (int(Joint.NECK), int(Joint.LEFT_SHOULDER)),
    (int(Joint.NECK), int(Joint.RIGHT_SHOULDER)),
}


def _bone_transform(a, b, radius):
    """4x4 mapping a unit z-cylinder (z in [0,1], r=1) onto the segment a->b."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return None
    z = d / L
    ref = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c0, c1, c2 = x * radius, y * radius, d          # columns of the 3x3
    return QMatrix4x4(
        c0[0], c1[0], c2[0], a[0],
        c0[1], c1[1], c2[1], a[1],
        c0[2], c1[2], c2[2], a[2],
        0, 0, 0, 1)


def _sphere_transform(p, r):
    m = QMatrix4x4()
    m.translate(float(p[0]), float(p[1]), float(p[2]))
    m.scale(r, r, r)
    return m


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

        # --- ghost body meshes (unit primitives, transformed per frame) ---
        cyl = gl.MeshData.cylinder(rows=1, cols=12, radius=[1.0, 1.0], length=1.0)
        sph = gl.MeshData.sphere(rows=8, cols=12, radius=1.0)
        self._limbs = []
        for _ in BONES:
            m = gl.GLMeshItem(meshdata=cyl, smooth=True, color=self.GHOST_COLOR,
                              shader="shaded", glOptions="translucent")
            m.setVisible(False)
            self.addItem(m)
            self._limbs.append(m)
        self._blobs = []
        for _ in range(NUM_JOINTS):
            m = gl.GLMeshItem(meshdata=sph, smooth=True, color=self.GHOST_COLOR,
                              shader="shaded", glOptions="translucent")
            m.setVisible(False)
            self.addItem(m)
            self._blobs.append(m)

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
        ax, sign = self._vaxis, self._vsign
        others = [i for i in range(3) if i != ax]
        out = np.empty_like(pose3d)
        out[:, 0] = pose3d[:, others[0]]
        out[:, 1] = pose3d[:, others[1]]
        out[:, 2] = sign * pose3d[:, ax]
        return out

    # --- public API ---
    def set_show_body(self, on: bool):
        self._show_body = on
        for m in self._limbs + self._blobs:
            m.setVisible(on and m.opts.get("_active", False))

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
        limb_r = height * 0.055
        torso_r = height * 0.095

        # coloured skeleton
        self._scatter.setData(pos=v[valid])
        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(v[int(a)]); seg.append(v[int(b)])
        self._lines.setData(pos=np.array(seg) if seg else np.zeros((2, 3)))

        # ghost limbs (capsules)
        for (a, b), mesh in zip(BONES, self._limbs):
            ok = valid[int(a)] and valid[int(b)]
            key = (int(a), int(b))
            r = torso_r if key in _THICK_BONES else limb_r
            tr = _bone_transform(v[int(a)], v[int(b)], r) if ok else None
            mesh.opts["_active"] = ok and tr is not None
            if ok and tr is not None:
                mesh.setTransform(Transform3D(tr))
            mesh.setVisible(self._show_body and ok and tr is not None)

        # ghost joint blobs
        for j, mesh in enumerate(self._blobs):
            ok = bool(valid[j])
            r = torso_r if j in (int(Joint.PELVIS), int(Joint.NECK)) else limb_r
            mesh.opts["_active"] = ok
            if ok:
                mesh.setTransform(Transform3D(_sphere_transform(v[j], r * 1.1)))
            mesh.setVisible(self._show_body and ok)

        if not self._framed:
            span = float(np.linalg.norm(vv.max(0) - vv.min(0))) or 1.0
            self._grid.setSize(span * 1.6, span * 1.6)
            self._grid.setSpacing(span / 8.0, span / 8.0)
            self.setCameraPosition(pos=Vector(0, 0, height * 0.5),
                                   distance=span * 1.9, elevation=12, azimuth=-70)
            self._framed = True

    def _clear(self):
        self._scatter.setData(pos=np.zeros((1, 3)))
        self._lines.setData(pos=np.zeros((2, 3)))
        for m in self._limbs + self._blobs:
            m.opts["_active"] = False
            m.setVisible(False)

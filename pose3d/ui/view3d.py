"""3D skeleton preview using pyqtgraph.opengl.

GLViewWidget (a QWidget) with a GLScatterPlotItem for joints and a
GLLinePlotItem for bones. Refreshed via setData when the pose changes.
Built-in mouse rotate/pan/zoom.

The pose is placed STANDING ON the grid (the grid is the ground):
- the world "up" axis is detected from the skeleton (head vs ankles), so the
  view is correct whether up is +Z (a typical ArUco rig) or -Y (Panoptic);
- that axis is mapped to the view's +Z, the pose is centred horizontally, and
  its lowest point is dropped onto z = 0 so the feet rest on the grid.
This is also unit-agnostic (metres or centimetres).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from pose3d.core.skeleton import BONES, NUM_JOINTS, Joint


class View3D(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setCameraPosition(distance=4.0, elevation=14, azimuth=-70)
        self._grid = gl.GLGridItem()          # lies in the XY plane, z = 0
        self._grid.setSize(2, 2)
        self._grid.setSpacing(0.2, 0.2)
        self.addItem(self._grid)

        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), size=12.0,
            color=(0.3, 0.85, 1.0, 1.0), pxMode=True)
        self.addItem(self._scatter)

        self._lines = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), width=2.5,
            color=(0.85, 0.85, 0.85, 1.0), mode="lines")
        self.addItem(self._lines)

        self._framed = False
        self._vaxis = None    # detected world vertical axis (0/1/2)
        self._vsign = 1.0     # sign so that head maps to +Z

    def _detect_vertical(self, pose3d, valid):
        """Return (axis, sign) mapping world-up to view +Z, from head vs feet."""
        head = pose3d[int(Joint.HEAD)]
        ankles = pose3d[[int(Joint.LEFT_ANKLE), int(Joint.RIGHT_ANKLE)]]
        ankle = np.nanmean(ankles, axis=0)
        if not (np.isnan(head).any() or np.isnan(ankle).any()):
            diff = head - ankle
            axis = int(np.argmax(np.abs(diff)))
            sign = float(np.sign(diff[axis]) or 1.0)   # view_z = sign * coord
            return axis, sign
        # fallback: axis of largest extent is vertical, up = increasing
        vpts = pose3d[valid]
        axis = int(np.argmax(vpts.max(0) - vpts.min(0)))
        return axis, 1.0

    def _to_view(self, pose3d):
        """Remap world coords to view coords (vertical axis -> +Z)."""
        ax, sign = self._vaxis, self._vsign
        others = [i for i in range(3) if i != ax]
        out = np.empty_like(pose3d)
        out[:, 0] = pose3d[:, others[0]]
        out[:, 1] = pose3d[:, others[1]]
        out[:, 2] = sign * pose3d[:, ax]
        return out

    def set_pose(self, pose3d: np.ndarray):
        pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
        valid = ~np.isnan(pose3d).any(1)
        if not valid.any():
            self._scatter.setData(pos=np.zeros((1, 3)))
            self._lines.setData(pos=np.zeros((2, 3)))
            return

        if self._vaxis is None:
            self._vaxis, self._vsign = self._detect_vertical(pose3d, valid)

        v = self._to_view(pose3d)                 # vertical -> +Z
        vv = v[valid]
        # centre horizontally on the grid, drop feet (min Z) onto z = 0
        cx, cy = vv[:, 0].mean(), vv[:, 1].mean()
        floor = vv[:, 2].min()
        v[:, 0] -= cx
        v[:, 1] -= cy
        v[:, 2] -= floor

        self._scatter.setData(pos=v[valid])
        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(v[int(a)]); seg.append(v[int(b)])
        self._lines.setData(pos=np.array(seg) if seg else np.zeros((2, 3)))

        if not self._framed:
            height = float(vv[:, 2].max() - vv[:, 2].min()) or 1.0
            span = float(np.linalg.norm(vv.max(0) - vv.min(0))) or 1.0
            self._grid.setSize(span * 1.5, span * 1.5)
            self._grid.setSpacing(span / 8.0, span / 8.0)
            # look at the figure's mid-height, standing on the grid
            self.setCameraPosition(pos=Vector(0, 0, height * 0.5),
                                   distance=span * 1.8,
                                   elevation=12, azimuth=-70)
            self._framed = True

    def reframe(self):
        """Re-detect orientation and re-frame on the next set_pose."""
        self._framed = False
        self._vaxis = None

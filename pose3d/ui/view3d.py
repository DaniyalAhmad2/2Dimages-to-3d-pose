"""3D skeleton preview using pyqtgraph.opengl.

GLViewWidget (a QWidget) with a GLScatterPlotItem for joints and a
GLLinePlotItem for bones. Refreshed via setData when the pose changes.
Built-in mouse rotate/pan/zoom.

The view AUTO-FRAMES the pose: points are recentred on their centroid and the
camera distance + grid size are scaled to the pose extent. This makes the view
unit-agnostic — it looks right whether coordinates are in metres (from the
client's ArUco rig) or centimetres (from the Panoptic validation data).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

from pose3d.core.skeleton import BONES, NUM_JOINTS


class View3D(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setCameraPosition(distance=4.0, elevation=12, azimuth=-70)
        self._grid = gl.GLGridItem()
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

    def set_pose(self, pose3d: np.ndarray):
        """pose3d: (NUM_JOINTS,3), may contain NaN. Auto-centred + auto-scaled."""
        pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
        valid = ~np.isnan(pose3d).any(1)
        if not valid.any():
            self._scatter.setData(pos=np.zeros((1, 3)))
            self._lines.setData(pos=np.zeros((2, 3)))
            return

        vpts = pose3d[valid]
        centroid = vpts.mean(axis=0)
        centred = pose3d - centroid              # bring pose to origin
        extent = float(np.linalg.norm(vpts.max(0) - vpts.min(0))) or 1.0

        # joints
        self._scatter.setData(pos=centred[valid])

        # bones (both endpoints valid)
        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(centred[int(a)])
                seg.append(centred[int(b)])
        self._lines.setData(pos=np.array(seg) if seg else np.zeros((2, 3)))

        # frame the camera + grid to the pose the first time (and keep it stable
        # afterwards so rotating/zooming by hand isn't reset every frame)
        if not self._framed:
            self._grid.setSize(extent, extent)
            self._grid.setSpacing(extent / 10.0, extent / 10.0)
            self.setCameraPosition(pos=Vector(0, 0, 0),
                                   distance=extent * 1.6,
                                   elevation=12, azimuth=-70)
            self._framed = True

    def reframe(self):
        """Force re-framing on the next set_pose (e.g. after loading a project)."""
        self._framed = False

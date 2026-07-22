"""3D skeleton preview using pyqtgraph.opengl.

GLViewWidget (a QWidget) with a GLScatterPlotItem for joints and a
GLLinePlotItem for bones. Refreshed via setData when the pose changes.
Built-in mouse rotate/pan/zoom.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QColor

from pose3d.core.skeleton import BONES, NUM_JOINTS


class View3D(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setCameraPosition(distance=4.0, elevation=12, azimuth=-70)
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        self.addItem(grid)

        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((NUM_JOINTS, 3)), size=10.0,
            color=(0.3, 0.85, 1.0, 1.0), pxMode=True)
        self.addItem(self._scatter)

        self._lines = gl.GLLinePlotItem(
            pos=np.zeros((2 * len(BONES), 3)), width=2.0,
            color=(0.8, 0.8, 0.8, 1.0), mode="lines")
        self.addItem(self._lines)

    def set_pose(self, pose3d: np.ndarray):
        """pose3d: (NUM_JOINTS,3), may contain NaN. Y-up remap for display."""
        pose3d = np.asarray(pose3d, float).reshape(NUM_JOINTS, 3)
        # our world is Z-up (metres); pyqtgraph default is Z-up too -> pass through
        pts = np.nan_to_num(pose3d, nan=0.0)
        valid = ~np.isnan(pose3d).any(1)
        self._scatter.setData(pos=pts[valid] if valid.any() else pts)

        seg = []
        for a, b in BONES:
            if valid[int(a)] and valid[int(b)]:
                seg.append(pose3d[int(a)])
                seg.append(pose3d[int(b)])
        if seg:
            self._lines.setData(pos=np.array(seg))
        else:
            self._lines.setData(pos=np.zeros((2, 3)))

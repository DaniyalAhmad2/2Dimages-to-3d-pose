"""Regression: the 3D view/export up-remap must be a rotation, never a mirror.

A reflection (det = -1) left/right-flips the character (raised left hand shows
as a raised right hand).
"""
import numpy as np

from pose3d.ui.view3d import upright_matrix


def test_upright_matrix_is_rotation_for_all_axes():
    for axis in (0, 1, 2):
        for sign in (1.0, -1.0):
            M = upright_matrix(axis, sign)
            det = float(np.linalg.det(M))
            assert np.isclose(det, 1.0), f"axis={axis} sign={sign} det={det} (mirror!)"
            assert np.allclose(M @ M.T, np.eye(3)), "not orthonormal"


def test_upright_matrix_keeps_up_axis_up():
    # the up reference (head) must end up with a larger +Z than the feet
    for axis in (0, 1, 2):
        for sign in (1.0, -1.0):
            M = upright_matrix(axis, sign)
            head = np.zeros(3); head[axis] = sign * 1.0    # "up" world point
            foot = np.zeros(3); foot[axis] = -sign * 1.0
            assert (M @ head)[2] > (M @ foot)[2]


def test_chirality_preserved():
    # scalar triple product sign (handedness) must be unchanged by the remap
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(4, 3))
    tri0 = np.dot(pts[1] - pts[0], np.cross(pts[2] - pts[0], pts[3] - pts[0]))
    for axis in (0, 1, 2):
        for sign in (1.0, -1.0):
            M = upright_matrix(axis, sign)
            q = pts @ M.T
            tri = np.dot(q[1] - q[0], np.cross(q[2] - q[0], q[3] - q[0]))
            assert np.sign(tri) == np.sign(tri0)

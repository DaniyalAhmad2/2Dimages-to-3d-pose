"""Regression: how the 3D view places the figure in the world.

The up-remap must be a rotation, never a mirror — a reflection (det = -1)
left/right-flips the character, so a raised left hand shows as a raised right
one — and the ground datum must sit still under the figure's ankle instead of
sliding about under whichever mesh vertex is lowest.
"""
import numpy as np
import pytest

from pose3d.core.skeleton import Joint
from pose3d.ui.view3d import ground_datum, upright_matrix
from tests.gates import needs_character
from tests.test_retarget import fixture_poses


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


def _z_extent(pose):
    v = pose[~np.isnan(pose).any(1)]
    return float(v[:, 2].max() - v[:, 2].min())


@needs_character()
def test_grounding_is_ankle_based():
    """The view used to ground on the character's lowest MESH vertex, which is
    a foot vertex whose height above the ankle swings with the shin (16-86 deg
    on this take): the whole scene translated against the fixed grid by 9.2 %
    of body height across 26 frames. Grounding a fixed drop below the posed
    ankle holds the datum still by construction.
    """
    from pose3d.geometry.character import Character
    poses = fixture_poses()
    if poses is None:
        pytest.skip("the client take is gitignored; not checked out here")
    ch = Character()
    ch.fit_to_subject(poses)
    height = float(np.median([_z_extent(p) for p in poses]))

    over_grid = []
    for p in poses:
        valid = ~np.isnan(p).any(1)
        # what View3D.set_pose does before it grounds
        v = p.copy()
        v[:, 0] -= p[valid, 0].mean()
        v[:, 1] -= p[valid, 1].mean()
        v[:, 2] -= p[valid, 2].min()
        v = np.where(valid[:, None], v, np.nan)
        verts, _, joints = ch.pose_and_joints(v, valid)
        dz = ground_datum(verts, joints, ch.ground_drop(v, valid))
        ankle = min(joints[int(Joint.LEFT_ANKLE), 2],
                    joints[int(Joint.RIGHT_ANKLE), 2])
        over_grid.append((ankle - dz) / height * 100.0)

    # measured 0.0000 % (9.2 % grounding on the lowest mesh vertex)
    assert np.ptp(over_grid) <= 0.5, "the ground datum still slides"
    # and it stands at the rig's own rest ankle height, 4.94 % of rig height,
    # which is 5.32 % of this subject's: a drop left in RIG units instead of
    # pose units would put the figure hundreds of percent off the grid.
    assert 3.0 <= np.median(over_grid) <= 8.0


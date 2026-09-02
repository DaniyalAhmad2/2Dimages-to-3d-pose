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

    over_grid, below_grid = [], []
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
        below_grid.append((dz - float(verts[:, 2].min())) / height * 100.0)

    # The cheap guard. Note it is arithmetic, not behaviour: with the ankle
    # branch in place this expression reduces to drop/height, a constant. It
    # still bites the one regression it was written for — delete the branch and
    # the datum falls back to verts.min() and this reads 9.2 % again.
    # measured 0.0000 % (9.2 % grounding on the lowest mesh vertex)
    assert np.ptp(over_grid) <= 0.5, "the ground datum still slides"
    # and it stands at the rig's own rest ankle height, 4.94 % of rig height,
    # which is 5.32 % of this subject's: a drop left in RIG units instead of
    # pose units would put the figure hundreds of percent off the grid.
    assert 3.0 <= np.median(over_grid) <= 8.0

    # What a viewer actually sees, which the two assertions above cannot show.
    # The datum no longer tracks whichever vertex happens to be lowest: under
    # the old rule the mesh minimum sat exactly ON the grid every frame, and
    # under this one a tilted sole dips below it (measured 3.62 % median).
    assert max(below_grid) > 1.0, \
        "the datum is back on the lowest mesh vertex"
    # ...but the dip is the accepted cost of an ankle datum with an unlevelled
    # sole, not a licence to sink: measured 9.57 % of body height worst case.
    # A drop mistakenly left in RIG units, or read off a fingertip on a rig
    # whose arms hang below its feet, buries the figure far deeper than this.
    assert max(below_grid) <= 11.0, \
        f"the figure sinks {max(below_grid):.1f} % of body height below the grid"



# --- Phase 3: one placement rule, shared with the export --------------------

def _travelling_take(n=6, step=0.06):
    """A synthetic take whose subject genuinely walks, so a rule that deletes
    the travel can be told from one that keeps it."""
    from tests.synth import sample_skeleton_3d
    base = sample_skeleton_3d()
    out = []
    for i in range(n):
        p = base.copy()
        p[6, 0] -= 0.02 * i
        p[:, 0] += step * i
        out.append(p)
    return np.stack(out)


def _headless_view(poses):
    """A View3D with no Qt behind it.

    `__new__` rather than `View3D()`: the placement rule is arithmetic over the
    take and a `Character`, and it should be assertable without a GL context.
    """
    from pose3d.geometry.character import Character
    from pose3d.ui.view3d import View3D
    v = View3D.__new__(View3D)
    v._R = np.eye(3)
    v._vaxis, v._vsign = 2, 1.0
    v._character = Character()
    v._character.fit_to_subject(poses)
    v._take = poses
    v._place = None
    return v


@needs_character()
def test_the_view_places_the_whole_take_by_one_rigid_map():
    """The subject's travel must survive into the preview.

    The old rule re-centred on the mean of the valid joints every frame, which
    deleted the translation from the view — by a different rule again from the
    one the export used to delete it with, so the two disagreed by up to 25.5 %
    of rig height. One take-wide offset means what moves on screen is the
    subject and nothing else.
    """
    from pose3d.geometry.character import take_pelvis_ref
    poses = _travelling_take()
    view = _headless_view(poses)

    place, travel = view._take_placement()
    assert place is not None
    ref = take_pelvis_ref(poses)
    assert np.allclose(place[:2], ref[:2])          # the shared horizontal rule
    assert travel > 0.2                             # sized for the walk, not one pose

    pel = np.stack([take_pelvis_ref(p[None]) for p in poses]) - place
    kept = np.ptp(pel[:, 0])
    real = np.ptp(poses[:, int(Joint.PELVIS), 0])
    assert kept == pytest.approx(real, rel=1e-9)

    # ...and what the old rule did with the same take, for contrast: centring
    # on the mean of the valid joints every frame leaves only the wobble
    # between that mean and the pelvis. Measured 2.2 % of the real travel.
    per_frame = np.stack([p[~np.isnan(p).any(1)][:, 0].mean() for p in poses])
    dropped = np.ptp(poses[:, int(Joint.PELVIS), 0] - per_frame)
    assert dropped <= 0.03 * real


@needs_character()
def test_the_view_and_the_export_place_the_figure_the_same_way():
    """The preview-vs-export placement gap, asserted rather than described.

    The view subtracts one take-wide offset; the export offsets the hips by
    `Rz @ (pelvis - pelvis_ref) * scale`. Those are the same rigid map through
    the same uniform scale, so the figure's displacement from frame to frame
    must agree to within numerical noise — 0 by construction, and this is what
    fails if either side goes back to placing per frame.
    """
    from pose3d.geometry.character import take_pelvis_ref
    poses = _travelling_take()
    view = _headless_view(poses)
    ch = view._character
    ref = take_pelvis_ref(poses)
    place, _travel = view._take_placement()

    for k in (0, 2, 5):
        valid = ~np.isnan(poses[k]).any(1)
        # the export's own root offset for this frame, back in capture units
        skin_ref = ch._skin_matrices(poses[k], valid, pelvis_ref=ref)[0]
        skin_own = ch._skin_matrices(poses[k], valid)[0]
        _s, _pelvis, scale, Rz = ch._skin_matrices(poses[k], valid)
        off_rig = skin_ref[ch.hips_idx][:3, 3] - skin_own[ch.hips_idx][:3, 3]
        off_capture = (Rz.T @ off_rig) / scale
        # what the view draws for the same frame: the posed pelvis, placed
        drawn = ch.posed_joints(poses[k], valid)[int(Joint.PELVIS)] - place
        assert np.allclose(drawn[:2], off_capture[:2], atol=1e-9), \
            f"frame {k}: view {drawn[:2]} vs export {off_capture[:2]}"

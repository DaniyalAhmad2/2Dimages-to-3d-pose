"""Regression: how the 3D view places the figure in the world.

The up-remap must be a rotation, never a mirror — a reflection (det = -1)
left/right-flips the character, so a raised left hand shows as a raised right
one — and the ground datum must sit still under the figure's ankle instead of
sliding about under whichever mesh vertex is lowest.
"""
import numpy as np

from pose3d.core.skeleton import Joint
from pose3d.ui.view3d import ground_datum, upright_matrix
from tests.gates import needs_character
from tests.test_retarget import fixture_head_source, fixture_poses


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

    Measured on the COMMITTED client take (tests/fixtures/client_take), so this
    runs everywhere: reading the gitignored workspace copy made Phase 2's
    ground-datum fix a gate that existed on one machine and skipped on every
    other checkout.
    """
    from pose3d.geometry.character import Character
    poses = fixture_poses()
    ch = Character(head_source=fixture_head_source())
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
    # which is 4.91 % of this subject's: a drop left in RIG units instead of
    # pose units would put the figure hundreds of percent off the grid.
    assert 3.0 <= np.median(over_grid) <= 8.0

    # What a viewer actually sees, which the two assertions above cannot show.
    # The datum no longer tracks whichever vertex happens to be lowest: under
    # the old rule the mesh minimum sat exactly ON the grid every frame, and
    # under this one a tilted sole dips below it (measured 2.55 % median).
    assert max(below_grid) > 1.0, \
        "the datum is back on the lowest mesh vertex"
    # ...but the dip is the accepted cost of an ankle datum with an unlevelled
    # sole, not a licence to sink: measured 8.40 % of body height worst case
    # (9.57 % on the COCO-17 workspace copy this used to read).
    # A drop mistakenly left in RIG units, or read off a fingertip on a rig
    # whose arms hang below its feet, buries the figure far deeper than this.
    assert max(below_grid) <= 11.0, \
        f"the figure sinks {max(below_grid):.1f} % of body height below the grid"



# --- Phase 3: one placement rule, shared with the export --------------------

def _travelling_take(n=6, step=0.06, turn_deg=0.0):
    """A synthetic take whose subject genuinely walks, so a rule that deletes
    the travel can be told from one that keeps it.

    `turn_deg` per pose also turns the subject about its own pelvis, which a
    rule that deletes the FACING can be told from one that keeps it. Off by
    default because the "what the old per-frame rule dropped" contrast below
    measures the wobble between the joint mean and the pelvis, and a turning
    subject swings that mean about the pelvis for reasons that have nothing to
    do with the placement rule.
    """
    from tests.synth import rot_about, sample_skeleton_3d
    base = sample_skeleton_3d()
    pelvis = base[int(Joint.PELVIS)].copy()
    out = []
    for i in range(n):
        p = base.copy()
        p[6, 0] -= 0.02 * i
        if turn_deg:
            p = (p - pelvis) @ rot_about((0, 0, 1), turn_deg * i).T + pelvis
        p[:, 0] += step * i
        out.append(p)
    return np.stack(out)


def _headless_view(poses):
    """A View3D with no Qt behind it, and its draw calls captured.

    `__new__` rather than `View3D()`: the placement rule is arithmetic over the
    take and a `Character`, and it should be assertable without a GL context.
    Everything `set_pose` DRAWS with is stubbed and recorded on `v.drawn`, so
    the tests can assert on the values that actually reach the screen rather
    than only on `_take_placement`'s return value — a change that left the
    rule intact but stopped `set_pose` applying it would otherwise pass.
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
    v._framed = True                 # nothing to frame without a GL camera
    v._show_body = True
    v._char_error = ""
    v._char_error_source = ""
    v.drawn = {}

    def _draw_skeleton(scatter, lines, pts, valid, base_color=None,
                       filled=None):
        v.drawn[scatter] = (np.asarray(pts, float).copy(),
                            np.asarray(valid, bool).copy())

    def _set_body(verts, faces):
        v.drawn["body"] = None if verts is None else np.asarray(verts).copy()

    v._draw_skeleton = _draw_skeleton
    v._set_body = _set_body
    v._scatter, v._lines = "char", "char_lines"
    v._cap_scatter, v._cap_lines = "capture", "capture_lines"
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

    # what `set_pose` actually hands the capture overlay, frame by frame: the
    # raw take minus ONE constant offset, elementwise. (`np.ptp` of the pelvis
    # track would be satisfied by any offset whatsoever, constant or not, since
    # a peak-to-peak is translation-invariant — it asserted nothing.)
    for k, pose in enumerate(poses):
        view.set_pose(pose)
        drawn, valid = view.drawn["capture"]
        assert valid.all()
        assert np.allclose(drawn, pose - place, atol=1e-9), f"frame {k}"

    # ...and what the old rule did with the same take, for contrast: centring
    # on the mean of the valid joints every frame leaves only the wobble
    # between that mean and the pelvis. Measured 2.2 % of the real travel.
    real = np.ptp(poses[:, int(Joint.PELVIS), 0])
    per_frame = np.stack([p[~np.isnan(p).any(1)][:, 0].mean() for p in poses])
    dropped = np.ptp(poses[:, int(Joint.PELVIS), 0] - per_frame)
    assert dropped <= 0.03 * real


@needs_character()
def test_the_view_and_the_export_place_the_figure_the_same_way():
    """The preview-vs-export placement gap, asserted rather than described.

    The view subtracts one take-wide offset from the capture-space pose; the
    export maps the rig into that same capture frame
    (`Character.export_transform`) and expresses it in rig units. Those are the
    same rigid map through the same uniform scale, so the character's joints
    must agree — position AND facing — up to the scale and the constant offset
    between the two origins. This is what fails if either side goes back to
    placing (or turning) per frame.
    """
    from pose3d.geometry.character import take_pelvis_ref
    poses = _travelling_take(turn_deg=8.0)   # walks AND turns: both must agree
    view = _headless_view(poses)
    ch = view._character
    ref = take_pelvis_ref(poses)
    place, _travel = view._take_placement()

    seen, expected = [], []
    for k in range(len(poses)):
        valid = ~np.isnan(poses[k]).any(1)
        # the export's own matrices for this frame, and the joints they carry
        _mats, al = ch.pose_bone_matrices(poses[k], valid, None,
                                          keep_root_motion=True,
                                          pelvis_ref=ref, return_alignment=True)
        skin = ch._skin_matrices(poses[k], valid)[0]
        rig = ch._joints_from_skin(skin)
        exported = (al.transform[:3, :3] @ rig.T).T + al.transform[:3, 3]
        # what the view draws for the same frame
        view.set_pose(poses[k])
        drawn = view.drawn["char"][0]
        for j in range(len(drawn)):
            if np.isfinite(rig[j]).all() and np.isfinite(drawn[j]).all():
                seen.append(drawn[j]); expected.append(exported[j])

    seen, expected = np.asarray(seen), np.asarray(expected)
    # the export is the view through one similarity: scale, no rotation, and a
    # constant offset (the view seats on the ankle datum, the export on the
    # pelvis). Solve for the offset and check the residual is numerical noise.
    resid = expected - seen * ch._scale
    assert np.abs(resid - resid.mean(0)).max() <= 1e-6 * ch.rig_h, \
        f"view and export disagree by {np.abs(resid - resid.mean(0)).max():.6f} rig units"


@needs_character()
def test_one_unposable_frame_does_not_cost_the_take_its_placement():
    """A frame with no PELVIS and no hips must be SKIPPED, not fatal.

    `Character.pose_and_joints` raises `PoseUnavailable` for such a frame, and
    a blanket `except Exception` around the whole take turned that into "this
    take has no placement" — which sends `set_pose` back to centring every
    frame on its own valid joints (the F13 preview defect Phase 3 fixed) while
    the export, which skips exactly the same frame and keeps the take-wide
    rule, goes on placing by the take. The two then disagree with no message
    anywhere. One frame in, everything else must be unchanged.
    """
    from pose3d.geometry.character import take_pelvis_ref
    poses = _travelling_take()
    healthy = _headless_view(poses.copy())._take_placement()

    gappy = poses.copy()
    for j in (Joint.PELVIS, Joint.LEFT_HIP, Joint.RIGHT_HIP):
        gappy[2, int(j)] = np.nan            # no root to stand on, this frame
    view = _headless_view(gappy)
    place, travel = view._take_placement()
    assert place is not None, "one unposable frame reverted the whole take"
    # the same rule as the healthy take: the seat is the lowest sole over the
    # frames that COULD be posed, and the missing frame is simply not one of
    # them (its own pelvis is gone, so the horizontal reference moves with it)
    assert np.allclose(place[:2], take_pelvis_ref(gappy)[:2])
    assert np.isclose(place[2], healthy[0][2])
    assert travel > 0.0

    # and every frame that CAN be posed is still drawn through that one offset
    for k, pose in enumerate(gappy):
        if k == 2:
            continue
        view.set_pose(pose)
        drawn, _valid = view.drawn["capture"]
        assert np.allclose(drawn, pose - place, atol=1e-9), f"frame {k}"


@needs_character()
def test_a_take_with_no_placement_is_measured_once():
    """The fallback is cached like the answer is.

    The failure path used to return without storing anything, so a take that
    yields no placement re-posed EVERY frame of the take on every frame
    change — the one thing the cache exists to prevent.
    """
    poses = _travelling_take()
    for j in (Joint.PELVIS, Joint.LEFT_HIP, Joint.RIGHT_HIP):
        poses[:, int(j)] = np.nan            # no frame can be posed at all
    view = _headless_view(poses)
    calls = []
    real = view._character.pose_and_joints

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    view._character.pose_and_joints = counted
    assert view._take_placement() == (None, 0.0)
    n = len(calls)
    assert view._take_placement() == (None, 0.0)
    assert view._take_placement() == (None, 0.0)
    assert len(calls) == n, "the whole take is re-posed on every frame"


@needs_character()
def test_a_real_character_fault_in_the_placement_is_reported():
    """`PoseUnavailable` is normal and quiet; anything else is a fault.

    A missing rig asset or a corrupt .npz used to land in the same blanket
    catch as a hip-less frame and leave the view silently placing per frame.
    """
    poses = _travelling_take()
    view = _headless_view(poses)
    said = []
    view._report = lambda msg, source="": said.append((msg, source))

    def boom(*a, **kw):
        raise RuntimeError("the character asset is a directory")

    view._character.pose_and_joints = boom
    assert view._take_placement() == (None, 0.0)
    assert said and "RuntimeError" in said[0][0]
    assert said[0][1] == "place"           # not withdrawn by a later good skin

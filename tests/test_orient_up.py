"""Which way is up — and why the calibration cannot tell us.

The world frame's axes come from whichever ArUco tag the calibration happened
to pick, and tags taped at different rotations define different "ups" (6, 92
and 89 degrees apart on the client's rig). So sequence_up levels on the
subject's own body line instead, and the UI says that is what it did.
"""
import numpy as np
import pytest

from pose3d.core.skeleton import Joint
from pose3d.geometry.orient import (
    _frame_up, de_tilt_matrix, sequence_up,
)
from tests.gates import needs_character
from tests.synth import rot_about as _rot, sample_skeleton_3d


def _torso_tilt(p):
    t = p[int(Joint.NECK)] - p[int(Joint.PELVIS)]
    return float(np.degrees(np.arccos(np.clip(t[2] / np.linalg.norm(t), -1, 1))))


def test_markers_taped_at_different_rotations_are_not_a_vertical_reference():
    """Why orientation does NOT trust the calibration frame's axes.

    ArUco tags encode their own orientation, and on the client's rig the wall
    tags are taped at different rotations — three markers in one image put
    "up" 6, 92 and 89 degrees from the camera's. resolve_calibration picks the
    lowest-id common tag, so the world frame's vertical is whatever rotation
    that one happens to have. A version that snapped to the
    nearest world axis inherited that and leaned the figure ~26 degrees.
    """
    base = sample_skeleton_3d()
    # a world frame rotated 90 deg about the view axis, as a sideways tag gives
    R = _rot([0, 1, 0], 90.0)
    seq = np.stack([base @ R.T] * 3)
    up = sequence_up(seq)
    # levelling still stands the figure up rather than following the bogus axis
    assert _torso_tilt(seq[0] @ de_tilt_matrix(up).T) < 6.0


def test_levelling_stands_the_subject_up_whatever_the_world_frame():
    """The world frame is arbitrary, so orientation must not depend on it."""
    base = sample_skeleton_3d()
    R = _rot([1, 1, 0], 45.0)           # up lands ~45 deg from every axis
    seq = np.stack([base @ R.T] * 3)
    up = sequence_up(seq)
    out = seq[0] @ de_tilt_matrix(up).T
    assert _torso_tilt(out) < 6.0       # normalised, as before


def test_no_usable_frames_returns_none():
    assert sequence_up(np.full((2, 15, 3), np.nan)) is None


def test_frame_up_ignores_the_nose():
    """HEAD is the nose — forward of the body axis. It tipped the estimated
    up ~10 deg forward on a real take; the NECK must be the top reference."""
    p = sample_skeleton_3d()
    u1 = _frame_up(p, ~np.isnan(p).any(1))
    q = p.copy()
    q[int(Joint.HEAD)] = q[int(Joint.HEAD)] + np.array([0.0, 0.5, 0.0])
    u2 = _frame_up(q, ~np.isnan(q).any(1))
    assert np.degrees(np.arccos(np.clip(np.dot(u1, u2), -1, 1))) < 0.1

    # ...but HEAD still serves when the neck is missing
    q[int(Joint.NECK)] = np.nan
    assert _frame_up(q, ~np.isnan(q).any(1)) is not None


# --------------------------------------------------------------------------
# A vertical recorded at calibration time (geometry/gravity.py). It is a
# property of the ROOM, so unlike sequence_up it keeps a lean held for the
# whole take — which is exactly why it must not be trusted blindly, and why it
# is always carried with the spread between the proxies it averages.
# --------------------------------------------------------------------------
def _rig(Rl, Rr):
    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.calib.intrinsics import Intrinsics
    from pose3d.pipeline import CalibratedRig
    k = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(64, 48))
    return CalibratedRig(k, k, Extrinsics(R=Rl, t=np.zeros(3)),
                         Extrinsics(R=Rr, t=np.zeros(3)))


def _upright_pair(roll_deg=0.0):
    """Two cameras 80 deg apart looking at the origin in a world where +Z is
    up, optionally rolled about their own viewing axes."""
    from tests.synth import default_two_cam
    rig = default_two_cam()
    Rs = []
    for cam in ("left", "right"):
        R = rig[cam][0]
        Rs.append(_rot([0, 0, 1], roll_deg) @ R if roll_deg else R)
    return _rig(*Rs)


def test_the_camera_proxies_find_a_known_vertical():
    from pose3d.geometry.gravity import estimate_world_up

    up, spread, cands = estimate_world_up(_upright_pair())
    assert np.allclose(up, [0, 0, 1], atol=1e-9)
    assert spread < 1e-9
    assert set(cands) == {"camera pair", "camera up"}   # no tags supplied


def test_a_tag_row_is_used_only_when_two_tags_survived_admission():
    """The third proxy is the only one that does not assume the cameras are
    unrolled — and the only one a rejected tag can poison, so it needs two
    admitted tags before it says anything."""
    from pose3d.geometry.gravity import estimate_world_up

    rig = _upright_pair(roll_deg=8.0)          # both cameras rolled: (a),(b) tilt
    # two tags on a wall facing -Y, sitting in a horizontal row
    face = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    layout = {14: (face, np.array([-0.15, 1.0, 1.2])),
              17: (face, np.array([0.15, 1.0, 1.2]))}
    up, spread, cands = estimate_world_up(rig, layout)
    assert set(cands) == {"camera pair", "camera up", "tag row"}
    assert abs(np.degrees(np.arccos(np.dot(cands["tag row"], [0, 0, 1])))) < 1e-6
    assert spread > 5.0                         # the roll shows up as disagreement
    # one tag alone is not a row
    assert "tag row" not in estimate_world_up(rig, {14: layout[14]})[2]


def test_sign_is_stable():
    """The two cross-product proxies have an arbitrary sign; the answer must
    not. With poses in hand the sense is settled by a binary test — is the mean
    neck above the mean ankle — which cannot rotate the axis, only flip it."""
    from pose3d.geometry.gravity import estimate_world_up

    rig = _upright_pair()
    poses = np.stack([sample_skeleton_3d()] * 4)
    up_a = estimate_world_up(rig, poses=poses)[0]
    assert np.allclose(up_a, [0, 0, 1], atol=1e-9)

    # the same rig and a subject standing on its head: only the sign moves
    flipped = poses.copy()
    flipped[..., 2] *= -1
    up_b = estimate_world_up(rig, poses=flipped)[0]
    assert np.allclose(up_b, -up_a, atol=1e-9)
    # ...and with no poses at all the camera-up proxy still fixes the sense
    assert np.allclose(estimate_world_up(rig)[0], up_a, atol=1e-9)


def test_recorded_up_wins():
    """A lean held for the whole take is a real lean, and the recorded
    vertical is the only thing that can keep it: sequence_up averages the
    subject's own body line, so by construction it stands the figure up."""
    from pose3d.geometry.orient import take_up

    lean = _rot([1, 0, 0], 12.0)                    # the whole take leans 12 deg
    seq = np.stack([sample_skeleton_3d() @ lean.T] * 5)
    recorded = (np.array([0.0, 0.0, 1.0]), "camera pair + tag row", 9.0)

    up, source, spread = take_up(seq, recorded)
    assert np.allclose(up, [0, 0, 1])
    assert source == "camera pair + tag row" and spread == 9.0
    tilt = _torso_tilt(seq[0] @ de_tilt_matrix(up).T)
    assert tilt > 10.0                              # the lean survives...
    assert _torso_tilt(seq[0] @ de_tilt_matrix(sequence_up(seq)).T) < 6.0  # ...only here


def test_a_vague_vertical_is_refused():
    """Above ~20 deg of spread the three proxies are not describing the same
    axis, and the subject's body line is the better answer."""
    from pose3d.geometry.orient import take_up

    seq = np.stack([sample_skeleton_3d()] * 3)
    vague = (np.array([0.0, 0.4, 0.9]), "camera pair", 25.0)
    up, source, spread = take_up(seq, vague)
    assert source == "subject" and spread is None
    assert np.allclose(up, sequence_up(seq))


def test_legacy_rig_unchanged():
    """A project calibrated before the vertical was recorded must orient
    exactly as it does today — bit for bit, not approximately."""
    from pose3d.geometry.orient import take_up

    seq = np.stack([sample_skeleton_3d() @ _rot([1, 1, 0], 20.0).T] * 4)
    up, source, spread = take_up(seq, None)
    assert source == "subject" and spread is None
    assert np.array_equal(up, sequence_up(seq))
    assert np.array_equal(de_tilt_matrix(up), de_tilt_matrix(sequence_up(seq)))
    # ...and that is still what an absent calibration file yields
    from pose3d.calib.resolve import load_world_up
    assert load_world_up("/nonexistent/calibration") is None


def test_per_frame_lean_is_preserved():
    """de_tilt_matrix is ONE constant rotation for the whole take. The
    subject's per-frame lean (10.8 deg median, 21.1 deg max on the client's
    take) is the performance; only the take-wide tilt may move."""
    def torso(p):
        v = p[int(Joint.NECK)] - p[int(Joint.PELVIS)]
        return v / np.linalg.norm(v)

    def between(u, v):
        # atan2 form: arccos loses precision to 1e-6 deg for nearly parallel
        # vectors, which is bigger than the effect under test
        return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)),
                                           np.dot(u, v))))

    base = sample_skeleton_3d()
    seq = np.stack([base @ _rot([1, 0, 0], a).T for a in (-8.0, 0.0, 5.0, 17.0)])
    R = de_tilt_matrix(sequence_up(seq))
    before = [torso(p) for p in seq]
    after = [torso(p @ R.T) for p in seq]

    # frame-to-frame lean is untouched: every pairwise angle survives
    for i in range(len(seq)):
        for j in range(len(seq)):
            assert abs(between(before[i], before[j])
                       - between(after[i], after[j])) < 1e-9
    # ...while the take-wide tilt IS removed (that is the whole job)
    assert between(sequence_up(seq), [0, 0, 1]) > 3.0
    assert between(sequence_up(seq @ R.T), [0, 0, 1]) < 1e-9
    # and the levelled take still leans, frame by frame, as it did
    assert max(_torso_tilt(p @ R.T) for p in seq) > 9.0


@needs_character()
def test_the_export_levels_on_the_recorded_vertical_too():
    """The 3D view and the Blender export must stay pose-identical, so the
    export has to take the same vertical — not fall back to the subject while
    the preview uses the room."""
    from pose3d.export.blender_export import _character_bone_frames
    from pose3d.geometry.character import Character
    from pose3d.geometry.orient import take_up

    seq = np.stack([sample_skeleton_3d() @ _rot([1, 0, 0], 12.0).T] * 3)
    recorded = (np.array([0.0, 0.0, 1.0]), "camera pair", 9.0)

    got, _names = _character_bone_frames(seq, 0, None, recorded_up=recorded)

    R = de_tilt_matrix(take_up(seq, recorded)[0]).T   # what the view applies
    ch = Character()
    ch.fit_to_subject(seq @ R)
    pose = seq[0] @ R
    expected = ch.pose_bone_matrices(pose, ~np.isnan(pose).any(1))
    assert set(got[0]) == set(expected)
    for bone, M in expected.items():
        assert np.allclose(got[0][bone], M, atol=1e-12), bone

    # and it really is the recorded vertical doing the work
    legacy, _ = _character_bone_frames(seq, 0, None, recorded_up=None)
    assert any(not np.allclose(got[0][b], legacy[0][b], atol=1e-6)
               for b in expected)


def test_the_sidebar_states_the_vertical_and_its_uncertainty():
    """The number is the deliverable: "±14°" is something the client can
    weigh against what they see, "levelled" is not."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from pose3d.ui.panels import Sidebar

    QApplication.instance() or QApplication([])
    bar = Sidebar()

    bar.show_levelling_note(True, "camera pair + tag row", 9.2)
    assert bar.vertical_ref.text() == (
        "Vertical: recorded at calibration from the camera pair + tag row, "
        "±9° between those estimates. The 3D view and the export both use it, "
        "so a lean held all take stays a lean.")

    bar.show_levelling_note(True)                     # legacy project
    assert "estimated from the subject" in bar.vertical_ref.text()

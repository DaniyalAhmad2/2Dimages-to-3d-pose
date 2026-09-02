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


# --------------------------------------------------------------------------
# Two ways the recorded vertical could be believed when it should not be: a
# sense settled before there were any poses to settle it with, and a lone
# proxy reporting perfect agreement with itself.
# --------------------------------------------------------------------------
def _parallel_pair():
    """The canonical rig: two phones side by side, both pointing the same way.

    Their image-x axes are then PARALLEL, so cross(x_left, x_right) is zero and
    the camera-pair proxy drops out — leaving one candidate.
    """
    from tests.synth import look_at
    R, _t = look_at((-0.3, -3.0, 1.5), (-0.3, 0.0, 1.5))
    return _rig(R, R.copy())


def _upside_down_pair():
    """Both phones mounted rotated 180 deg about their own viewing axis."""
    from tests.synth import default_two_cam
    rig = default_two_cam()
    return _rig(*[_rot([0, 0, 1], 180.0) @ rig[cam][0]
                  for cam in ("left", "right")])


def test_one_candidate_is_unverified_not_certain():
    """A single proxy has nothing disagreeing with it AND nothing confirming
    it. Reporting spread 0.0 there would rank the weakest evidence this module
    can produce above the 8-14 deg that three agreeing proxies earn, and would
    hand `take_up` an unvalidated guess to level the view and the export on."""
    from pose3d.geometry.gravity import estimate_world_up
    from pose3d.geometry.orient import take_up

    up, spread, cands = estimate_world_up(_parallel_pair())
    assert list(cands) == ["camera up"]         # the pair proxy cannot form
    assert up is not None and spread is None    # not 0.0

    # ...so the view refuses it and levels on the subject, as for a project
    # with no recorded vertical at all
    seq = np.stack([sample_skeleton_3d()] * 3)
    got, source, got_spread = take_up(seq, (up, "camera up", spread))
    assert source == "subject" and got_spread is None
    assert np.array_equal(got, sequence_up(seq))

    # two candidates that genuinely agree still earn their zero
    assert estimate_world_up(_upright_pair())[1] == 0.0


def test_a_spread_the_file_does_not_state_is_unknown(tmp_path):
    """An extrinsics.json carrying a vertical but no spread — hand-edited, or
    written by something that is not save_rig — must not read as maximum
    confidence."""
    import json

    from pose3d.calib.resolve import load_world_up

    calib = tmp_path / "calibration"
    calib.mkdir()
    doc = {"left": {"R": np.eye(3).tolist(), "t": [0, 0, 0]},
           "right": {"R": np.eye(3).tolist(), "t": [0, 0, 0]},
           "world_up": [0.0, 0.0, 1.0], "world_up_source": "camera up"}
    (calib / "extrinsics.json").write_text(json.dumps(doc))
    assert load_world_up(calib)[2] is None

    # an explicit null (what save_rig writes for a single candidate) too
    doc["world_up_spread_deg"] = None
    (calib / "extrinsics.json").write_text(json.dumps(doc))
    up, source, spread = load_world_up(calib)
    assert spread is None and source == "camera up"

    # and a stated spread is still believed
    doc["world_up_spread_deg"] = 9.5
    (calib / "extrinsics.json").write_text(json.dumps(doc))
    assert load_world_up(calib)[2] == 9.5


def test_the_sidebar_says_unverified_rather_than_plus_minus_zero():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from pose3d.ui.panels import Sidebar

    QApplication.instance() or QApplication([])
    bar = Sidebar()

    # the view levelled on the subject, but a vertical WAS recorded — say why
    # it went unused rather than implying the project has none
    bar.show_levelling_note(True, "subject", None,
                            (np.array([0.0, 0.0, 1.0]), "camera up", None))
    text = bar.vertical_ref.text()
    assert "unverified" in text and "camera up" in text
    assert "±" not in text


# --------------------------------------------------------------------------
# The sign rule has to run where the poses are (calib.resolve.finalize_world_up)
# --------------------------------------------------------------------------
def _project_with_poses(poses):
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
    p = ProjectData(name="take")
    for i, pose in enumerate(poses):
        f = Frame(frame_id=f"{i:04d}",
                  images={CAM_LEFT: "l.png", CAM_RIGHT: "r.png"})
        f.fitted3d = np.asarray(pose, float)
        p.frames.append(f)
    return p


def _neck_above_ankle(up, pose):
    d = pose[int(Joint.NECK)] - np.nanmean(
        pose[[int(Joint.LEFT_ANKLE), int(Joint.RIGHT_ANKLE)]], axis=0)
    return float(np.dot(np.asarray(up, float), d))


def test_a_rig_mounted_upside_down_records_an_inverted_vertical(tmp_path):
    """The failure this fix exists for, stated as the bug it is.

    resolve_calibration necessarily runs BEFORE detection and triangulation —
    it is what makes triangulation possible — so at import time no frame has a
    pose and the sense of the vertical falls back to "the phones were held
    upright". Mount both phones rotated 180 deg and that proxy is exactly
    inverted, and everything downstream renders and exports upside down.
    """
    from pose3d.geometry.gravity import estimate_world_up

    up, _spread, _cands = estimate_world_up(_upside_down_pair())
    assert np.allclose(up, [0.0, 0.0, -1.0], atol=1e-9)
    assert _neck_above_ankle(up, sample_skeleton_3d()) < 0


def test_the_sign_is_settled_again_once_triangulation_has_produced_poses(tmp_path):
    """finalize_world_up re-runs the NECK-above-ANKLE test on the poses that
    now exist and rewrites both calibration files, so the binary test really
    decides the sense — and says so in the report."""
    from pose3d.calib.resolve import (
        SIGN_FROM_CAMERA_UP, SIGN_FROM_POSES, finalize_world_up, load_world_up,
        save_rig,
    )
    from pose3d.geometry.gravity import estimate_world_up
    import json

    rig = _upside_down_pair()
    up, spread, cands = estimate_world_up(rig)          # no poses yet
    report = {"world_up": up.tolist(), "world_up_source": " + ".join(cands),
              "world_up_spread_deg": spread,
              "world_up_candidates": {k: v.tolist() for k, v in cands.items()},
              "world_up_sign_source": SIGN_FROM_CAMERA_UP}
    calib = tmp_path / "calibration"
    save_rig(rig, calib, report)
    assert np.allclose(load_world_up(calib)[0], [0, 0, -1], atol=1e-9)

    poses = np.stack([sample_skeleton_3d()] * 4)
    got = finalize_world_up(_project_with_poses(poses), calib)

    assert np.allclose(got[0], [0.0, 0.0, 1.0], atol=1e-9)
    assert _neck_above_ankle(got[0], poses[0]) > 0
    assert got[2] == spread                              # only the sense moved
    # ...and it is on disk, for the view and the export to read
    assert np.allclose(load_world_up(calib)[0], got[0], atol=1e-9)
    written = json.loads((calib / "report.json").read_text())
    assert written["world_up_sign_source"] == SIGN_FROM_POSES
    assert np.allclose(written["world_up"], [0, 0, 1], atol=1e-9)
    for v in written["world_up_candidates"].values():
        assert float(np.dot(v, got[0])) > 0              # candidates flipped too

    # an upright rig is left exactly as it was (the sign only ever flips)
    calib2 = tmp_path / "calibration2"
    up2, spread2, cands2 = estimate_world_up(_upright_pair(), poses=poses)
    save_rig(_upright_pair(), calib2,
             {"world_up": up2.tolist(), "world_up_source": " + ".join(cands2),
              "world_up_spread_deg": spread2})
    again = finalize_world_up(_project_with_poses(poses), calib2)
    assert np.array_equal(again[0], load_world_up(calib2)[0])
    assert np.allclose(again[0], up2, atol=1e-12)


def test_a_take_with_no_usable_poses_leaves_the_vertical_alone(tmp_path, recwarn):
    """No NECK and no ankle anywhere is not evidence for either sense — and it
    must not warn about an empty slice on the way to saying so."""
    from pose3d.calib.resolve import finalize_world_up, save_rig
    from pose3d.geometry.gravity import estimate_world_up, sign_from_poses

    rig = _upside_down_pair()
    up, spread, cands = estimate_world_up(rig)
    calib = tmp_path / "calibration"
    save_rig(rig, calib, {"world_up": up.tolist(), "world_up_source": "camera up",
                          "world_up_spread_deg": spread})

    blank = np.full((3, 15, 3), np.nan)
    assert sign_from_poses(up, blank) is None
    assert finalize_world_up(_project_with_poses(blank), calib) is None
    assert not [w for w in recwarn.list
                if "empty slice" in str(w.message).lower()]


def test_the_import_settles_the_sign_after_it_has_poses():
    """Ordering is the whole finding: `resolve_calibration` cannot see poses,
    so a call placed before triangulation always falls back to the camera
    proxy. Pin that the import's call sits after the fit."""
    import inspect

    from pose3d.ui.import_dialog import ImportDialog
    from pose3d.ui.model import ProjectModel

    src = inspect.getsource(ImportDialog._process)
    assert src.index("finalize_world_up") > src.index("fit_project(project")
    # and the same step runs on every recompute, so re-solving 3D cannot leave
    # a stale sense behind
    assert "finalize_world_up" in inspect.getsource(ProjectModel.recompute_all)

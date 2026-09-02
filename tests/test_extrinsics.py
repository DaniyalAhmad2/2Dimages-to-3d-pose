"""Phase 2 verification: ArUco extrinsics recover known camera geometry.

We render ArUco markers into a synthetic camera at a known pose, then check
that estimate_extrinsics recovers that pose (camera center within tolerance).
"""
import inspect
import json
from pathlib import Path

import cv2
import numpy as np

from pose3d.calib.extrinsics import (
    detect_markers, estimate_extrinsics, estimate_extrinsics_for_marker,
    make_detector, marker_object_points,
)
from pose3d.calib.intrinsics import Intrinsics


def _render_marker(K, dist, R, t, marker_id, marker_len, world_center, size=(1280, 720)):
    """Render one ArUco marker (in world at world_center) into an image."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = np.full((size[1], size[0]), 255, np.uint8)
    # marker world corners (TL,TR,BR,BL) around world_center in z=0 plane
    obj = marker_object_points(marker_len) + np.asarray(world_center, np.float32)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(obj, rvec, t.reshape(3, 1), K, dist)
    proj = proj.reshape(-1, 2).astype(np.float32)
    # generate the marker bitmap and warp it onto the projected quad
    S = 400
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, S)
    src = np.array([[0, 0], [S, 0], [S, S], [0, S]], np.float32)
    Hmat = cv2.getPerspectiveTransform(src, proj)
    warped = cv2.warpPerspective(marker_img, Hmat, size, borderValue=255,
                                 flags=cv2.INTER_NEAREST)
    ones = np.full((S, S), 255, np.uint8)
    mask = cv2.warpPerspective(ones, Hmat, size, borderValue=0,
                               flags=cv2.INTER_NEAREST)
    img[mask > 0] = warped[mask > 0]
    return img


def _cam(f=900.0, w=1280, h=720):
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    return Intrinsics(K=K, dist=np.zeros((1, 5)), image_size=(w, h))


def _look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)
    return R, -R @ eye


def test_single_marker_extrinsics_recovers_center():
    intr = _cam()
    eye = np.array([0.3, -1.2, 0.25])
    R, t = _look_at(eye, (0, 0, 0))
    marker_len = 0.30   # large enough to project to ~100+ px and be detectable
    img = _render_marker(intr.K, intr.dist, R, t, marker_id=0,
                         marker_len=marker_len, world_center=(0, 0, 0))
    ext = estimate_extrinsics(img, intr, marker_length=marker_len)
    # single-marker mode: marker frame IS world; recovered center ~ eye
    assert np.linalg.norm(ext.camera_center - eye) < 0.02, ext.camera_center


def test_projection_matrix_shapes():
    intr = _cam()
    R, t = _look_at((0.4, -2.0, 0.3))
    from pose3d.calib.extrinsics import Extrinsics
    ext = Extrinsics(R=R, t=t)
    assert ext.P_normalized.shape == (3, 4)
    assert ext.projection_pixel(intr.K).shape == (3, 4)


# --------------------------------------------------------------------------
# The real capture: tests/fixtures/client_take_aruco_corners.json holds the
# ArUco corners detected in all 26 frames of the client's take (DICT_6X6_250)
# plus the extrinsics that take shipped with. The images are not in the repo,
# so these corners are the only way to test the resolver against the capture
# that motivated it.
# --------------------------------------------------------------------------
FIXTURE = Path(__file__).with_name("fixtures") / "client_take_aruco_corners.json"


def _client_take():
    d = json.loads(FIXTURE.read_text())
    obs = {fid: {cam: {int(t): np.asarray(c, float) for t, c in per.items()}
                 for cam, per in cams.items()}
           for fid, cams in d["observations"].items()}
    intr = {}
    for cam, (w, h) in d["image_size"].items():
        f = float(max(w, h))
        intr[cam] = Intrinsics(
            K=np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]]),
            dist=np.zeros((1, 5)), image_size=(w, h), source=d["focal_source"])
    return d, obs, intr


def _rot_deg(A, B):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(np.asarray(A).T @ np.asarray(B)) - 1) / 2, -1, 1))))


def test_the_ambiguous_tag_is_rejected():
    """BLOCKING: scoring every tag must not re-calibrate the client's take.

    This is the only change in the phase that could pick a different
    calibration than the shipped first-hit loop did, so it is pinned: the same
    tag (14), the same frame (0001), the same IPPE branch (0) in both cameras,
    within 0.01 deg and 0.1 mm of the extrinsics the take shipped with
    (measured: 1.3e-5 deg, 1.4e-4 mm — the residue of solvePnPGeneric vs
    solvePnP). What is new is that tag 13 — stuck over a curved sweep, 2.41 px
    of IPPE reprojection rms against 0.23-0.82 for the others — is now
    rejected instead of merely never being reached.
    """
    from pose3d.calib.resolve import (
        TAG_RMS_MAX_PX, solve_rig_from_observations,
    )
    d, obs, intr = _client_take()

    sol = solve_rig_from_observations(obs, d["frame_order"], intr,
                                      d["marker_length_m"], d["dictionary"])
    assert sol.ok, sol.message
    assert sol.report["world_tag_id"] == 14
    assert sol.report["world_frame_id"] == "0001"
    assert sol.report["branch_index"] == {"left": 0, "right": 0}
    for cam in ("left", "right"):
        R = np.array(d["shipped_extrinsics"][cam]["R"], float)
        t = np.array(d["shipped_extrinsics"][cam]["t"], float)
        assert _rot_deg(R, sol.ext[cam].R) < 0.01
        centre = -R.T @ t
        assert np.linalg.norm(centre - sol.ext[cam].camera_center) * 1000 < 0.1

    assert 13 in sol.report["tags_rejected_with_reason"]
    assert sol.report["tags_admitted"] == [14, 15, 17]

    # ...and rejected on its own evidence in every single frame, not on a
    # pooled average that one bad frame could have dragged down
    bad = [fid for fid in d["frame_order"]
           if estimate_extrinsics_for_marker(
               [obs[fid]["left"][13]], [13], 13, intr["left"],
               d["marker_length_m"])[0].err > TAG_RMS_MAX_PX]
    assert len(bad) == 26, len(bad)


def test_joint_branch_scoring():
    """Both cameras choosing the wrong IPPE branch does not cancel out.

    L1R1 on the client's take returns a 0.554 m baseline — the same as the
    right answer to within 0.2 % — and is MORE consistent across frames
    (1.01 deg of relative-pose spread against 1.13). So neither a baseline
    sanity check nor cross-frame agreement can reject it, and its relative pose
    is ~104 deg wrong. What rejects it is each camera's own branch ratio:
    IPPE prefers branch 0 by 9.3x (left) and 13.4x (right), far above the 2x
    the scorer demands, so branch 1 is not admissible in either camera.
    """
    from pose3d.calib.resolve import BRANCH_RATIO_MIN, score_branch_pairs

    d, obs, intr = _client_take()
    frames = [f for f in d["frame_order"]
              if 14 in obs[f]["left"] and 14 in obs[f]["right"]]
    scored = score_branch_pairs(obs, frames, 14, intr, d["marker_length_m"])
    pairs = scored["pairs"]

    assert pairs[(0, 0)]["admissible"]
    assert not pairs[(1, 1)]["admissible"]
    assert not pairs[(0, 1)]["admissible"] and not pairs[(1, 0)]["admissible"]
    # the two checks that CANNOT tell them apart
    b00, b11 = pairs[(0, 0)]["baseline_m"], pairs[(1, 1)]["baseline_m"]
    assert abs(b00 - b11) / b00 < 0.01
    assert pairs[(1, 1)]["relpose_spread_deg"] < pairs[(0, 0)]["relpose_spread_deg"]
    # the one that can
    for cam in ("left", "right"):
        assert scored["branch_ratio"][cam] > BRANCH_RATIO_MIN * 4


def test_both_ippe_branches_come_back_with_their_errors():
    intr = _cam()
    R, t = _look_at(np.array([0.3, -1.2, 0.25]), (0, 0, 0))
    img = _render_marker(intr.K, intr.dist, R, t, marker_id=0,
                         marker_len=0.30, world_center=(0, 0, 0))
    corners, ids = detect_markers(img, make_detector(cv2.aruco.DICT_4X4_50))
    solves = estimate_extrinsics_for_marker(corners, ids, ids[0], intr, 0.30)
    assert len(solves) == 2
    assert solves[0].err <= solves[1].err          # best branch first
    assert estimate_extrinsics_for_marker(corners, ids, 42, intr, 0.30) == []


def test_multi_tag_branch_is_gone():
    """One rotation shared by every tag is wrong by up to 179 deg on a
    hand-taped backdrop, and its one-known-tag path fed a non-centred square to
    IPPE_SQUARE (camera-centre errors of 527-1020 mm). It had no callers."""
    import pose3d.calib.extrinsics as ex

    assert not hasattr(ex, "tag_world_positions")
    assert "tag_world_positions" not in inspect.signature(
        ex.estimate_extrinsics).parameters


def test_the_failure_message_names_which_camera_saw_what():
    """"No markers were detected" is not a diagnosis. The client's right camera
    saw one tag in 11 of 26 frames and nothing in the other 15; the message has
    to say so, because the fix is to move THAT camera."""
    from pose3d.calib.resolve import solve_rig_from_observations

    d, obs, intr = _client_take()
    blind = {fid: {"left": cams["left"], "right": {}}
             for fid, cams in obs.items()}
    sol = solve_rig_from_observations(blind, d["frame_order"], intr,
                                      d["marker_length_m"], d["dictionary"])
    assert not sol.ok
    msg = sol.message
    assert "left camera saw tag 13 in 26/26 frames" in msg
    assert "tag 14 in 26/26 frames" in msg and "tag 17 in 26/26 frames" in msg
    assert "right camera saw no tags in 26/26 frames" in msg
    assert "move the cameras" in msg


def test_a_take_carried_only_by_a_rejected_tag_fails_loudly():
    """If the one tag both cameras share is the bent one, there is no
    calibration to be had — and the reason must not be silence.

    The right camera never actually saw tag 13, so its corners here are the
    left camera's; only the admission verdict is under test.
    """
    from pose3d.calib.resolve import solve_rig_from_observations

    d, obs, intr = _client_take()
    only13 = {fid: {"left": cams["left"],
                    "right": {13: cams["left"][13]} if 13 in cams["left"] else {}}
              for fid, cams in obs.items()}
    sol = solve_rig_from_observations(only13, d["frame_order"], intr,
                                      d["marker_length_m"], d["dictionary"])
    assert not sol.ok
    assert "13" in sol.message and "not flat" in sol.message


def test_a_common_tag_that_would_not_solve_still_gets_a_reason():
    """A tag can be seen by both cameras and be in NEITHER list: its solve
    failed, so it was never scored, so it was never admitted or rejected. The
    failure message then read "... were rejected — ." and named no reason at
    all — the same silence about tag 13 that took an audit to diagnose."""
    from pose3d.calib.resolve import solve_rig_from_observations

    d, obs, intr = _client_take()
    # tag 99 in both cameras with degenerate corners: IPPE returns no pose
    dead = np.full((4, 2), 100.0)
    broken = {fid: {cam: {99: dead.copy()} for cam in ("left", "right")}
              for fid in d["frame_order"]}
    sol = solve_rig_from_observations(broken, d["frame_order"], intr,
                                      d["marker_length_m"], d["dictionary"])
    assert not sol.ok
    assert "99" in sol.message
    assert "no usable pose could be solved" in sol.message
    assert not sol.message.endswith("— .")


def test_the_branch_tie_break_wants_evidence_before_agreement():
    """`relpose_spread_deg` is a MAXIMUM over pairwise angles, so a pair seen
    in one frame has no pairs and scores 0.0 — the best possible value, on no
    evidence. Comparing that against a pair scored over 11 frames hands the
    calibration to whichever wrong branch was seen least."""
    from pose3d.calib.resolve import pick_branch_pair

    pairs = {
        (0, 0): {"admissible": True, "n_frames": 11, "relpose_spread_deg": 1.13},
        (1, 1): {"admissible": True, "n_frames": 1, "relpose_spread_deg": 0.0},
    }
    assert pick_branch_pair(pairs) == (0, 0)

    # with the evidence equal, the more consistent pair wins as before
    pairs[(1, 1)]["n_frames"] = 11
    assert pick_branch_pair(pairs) == (1, 1)

    # inadmissible pairs are still out of the running whatever they score
    pairs[(1, 1)]["admissible"] = False
    assert pick_branch_pair(pairs) == (0, 0)


def test_the_report_says_what_settled_the_sense_of_the_vertical():
    """At import time there are no triangulated poses, so the sign of the
    recorded vertical rests on "the phones were held upright". That is a
    materially weaker claim than the NECK-above-ANKLE test and the report has
    to distinguish them — calib.resolve.finalize_world_up rewrites it once the
    poses exist."""
    from pose3d.calib.resolve import (
        SIGN_FROM_CAMERA_UP, solve_rig_from_observations,
    )

    d, obs, intr = _client_take()
    sol = solve_rig_from_observations(obs, d["frame_order"], intr,
                                      d["marker_length_m"], d["dictionary"])
    assert sol.ok
    assert sol.report["world_up_sign_source"] == SIGN_FROM_CAMERA_UP


# --- one validation for the app loader and the headless one ----------------

def _calibration_folder(folder, mangle=None):
    """A minimal, well-formed calibration folder, optionally mangled."""
    from pose3d.calib.intrinsics import Intrinsics

    calib = Path(folder) / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
    for name in ("left_intrinsics.json", "right_intrinsics.json"):
        Intrinsics(K=K, dist=np.zeros((1, 5)),
                   image_size=(1280, 720)).save(calib / name)
    doc = {"left": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0]},
           "right": {"R": np.eye(3).tolist(), "t": [0.12, 0.0, 0.0]}}
    if mangle is not None:
        mangle(doc)
    (calib / "extrinsics.json").write_text(json.dumps(doc))
    return calib


def test_the_headless_loader_refuses_a_malformed_rig_by_name(tmp_path):
    """`app.load_rig_with_reason` gained this validation because a 2x2 `R`
    used to explode inside triangulation as an unreadable numpy broadcast
    error. The headless loader every non-UI caller uses (`quality.load_rig`,
    tools/measure_take.py, the tests) had the old behaviour, so the same file
    failed in a sentence in the app and unreadably in the CLI. One
    implementation, `calib.rigio.check_extrinsics`, for both.
    """
    import pytest

    from pose3d.app import _check_extrinsics
    from pose3d.calib.rigio import check_extrinsics, load_rig, load_rig_or_none

    # the app's name for it IS the headless one: same verdict, same sentence
    d = {"R": [[1.0, 0.0], [0.0, 1.0]], "t": [0.0, 0.0, 0.0]}
    with pytest.raises(ValueError) as ui:
        _check_extrinsics("left", d)
    with pytest.raises(ValueError) as headless:
        check_extrinsics("left", d)
    assert str(ui.value) == str(headless.value)

    good = _calibration_folder(tmp_path / "good")
    assert load_rig(good) is not None

    def flatten(doc):
        doc["right"]["R"] = [[1.0, 0.0], [0.0, 1.0]]

    bad = _calibration_folder(tmp_path / "bad_shape", flatten)
    with pytest.raises(ValueError) as e:
        load_rig(bad)
    assert "right" in str(e.value) and "3x3" in str(e.value)
    # ...and the forgiving wrapper degrades to "no calibration", not to a rig
    # that explodes on the first triangulation
    assert load_rig_or_none(bad) is None

    def shear(doc):
        R = np.eye(3)
        R[0] *= 1.4
        doc["left"]["R"] = R.tolist()

    skew = _calibration_folder(tmp_path / "bad_rotation", shear)
    with pytest.raises(ValueError) as e:
        load_rig(skew)
    assert "not a rotation" in str(e.value)

    # the app's own loader turns the SAME exception into the SAME sentence
    from pose3d.app import load_rig_with_reason
    rig, reason = load_rig_with_reason(skew)
    assert rig is None and "not a rotation" in reason

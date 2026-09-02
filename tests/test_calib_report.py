"""A calibration nobody can reconstruct is a calibration nobody can dispute.

The client's take ships two 3x3 rotations and two translations and no record of
where they came from — which tag, which frame, which of IPPE's two poses, what
the tag was assumed to measure, whether the tags that were NOT used had been
rejected or merely not reached. Every question the audit asked of that rig took
re-detecting the tags to answer. `calibration/report.json` answers them, and
this file holds it to the standard that makes it worth writing: the rig can be
rebuilt from it bit for bit.
"""
import json
from pathlib import Path

import numpy as np

from pose3d.calib.intrinsics import Intrinsics
from pose3d.calib.resolve import (
    rebuild_extrinsics_from_report, save_rig, solve_rig_from_observations,
)
from pose3d.core.io_project import load_project, save_project
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, ProjectData
from pose3d.pipeline import CalibratedRig

FIXTURE = Path(__file__).with_name("fixtures") / "client_take_aruco_corners.json"


def _client_take():
    """The client take's real ArUco corners (see test_extrinsics.py)."""
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


def _solve():
    d, obs, intr = _client_take()
    sol = solve_rig_from_observations(obs, d["frame_order"], intr,
                                      d["marker_length_m"], d["dictionary"])
    assert sol.ok, sol.message
    return d, obs, intr, sol


def test_report_json_rebuilds_the_rig(tmp_path):
    """The report is a recipe, not a summary: replaying it reproduces the
    extrinsics EXACTLY (not approximately — same corners, same solver, same
    branch, so bit-identical)."""
    d, obs, intr, sol = _solve()
    rig = CalibratedRig(intr[CAM_LEFT], intr[CAM_RIGHT],
                        sol.ext[CAM_LEFT], sol.ext[CAM_RIGHT])
    save_rig(rig, tmp_path / "calibration", sol.report)

    report = json.loads((tmp_path / "calibration" / "report.json").read_text())
    rebuilt = rebuild_extrinsics_from_report(report, obs, intr)

    saved = json.loads(
        (tmp_path / "calibration" / "extrinsics.json").read_text())
    for cam in (CAM_LEFT, CAM_RIGHT):
        assert np.array_equal(rebuilt[cam].R, np.array(saved[cam]["R"]))
        assert np.array_equal(rebuilt[cam].t, np.array(saved[cam]["t"]))


def test_the_report_says_what_was_rejected_and_why():
    """A tag that was thrown out has to be distinguishable from a tag that was
    never looked at — that ambiguity is what made tag 13 take an audit to
    diagnose."""
    _d, _obs, _intr, sol = _solve()
    r = sol.report

    for key in ("world_tag_id", "world_frame_id", "dictionary",
                "marker_length_m", "branch_index", "per_camera_ippe_ratio",
                "per_tag_rms_px", "tags_admitted", "tags_rejected_with_reason",
                "n_observations_used", "n_frames_with_a_common_tag",
                "baseline_m", "convergence_deg", "focal_source_per_camera",
                "solver", "residual_rms_px", "camera_motion_check",
                "observation_counts"):
        assert key in r, key

    assert "not flat" in r["tags_rejected_with_reason"][13]
    assert set(r["per_tag_rms_px"]) == {13, 14, 15, 17}
    # the right camera saw one tag in 11 of 26 frames: the single most useful
    # fact about this capture, and the app never said it
    assert r["observation_counts"][CAM_RIGHT]["tags"][14] == 11
    assert r["observation_counts"][CAM_RIGHT]["empty_frames"] == 15
    assert r["n_frames_with_a_common_tag"] == 11
    assert r["focal_source_per_camera"] == {CAM_LEFT: "assumed",
                                            CAM_RIGHT: "assumed"}


def test_the_report_notices_a_camera_that_held_still():
    """Nothing in the app detects a camera that moved mid-take, so the check
    goes in the report while the tags are being solved anyway. On this take
    both cameras are still (0.76 deg / 8.4 mm of solve noise)."""
    _d, _obs, _intr, sol = _solve()
    motion = sol.report["camera_motion_check"]
    for cam in (CAM_LEFT, CAM_RIGHT):
        assert motion[cam]["moved"] is False
        assert motion[cam]["max_rotation_deg"] < 1.2       # measured 1.02
        assert motion[cam]["max_centre_mm"] < 10.0         # measured 8.42
    assert motion[CAM_LEFT]["n_frames"] == 26


def test_a_project_records_the_tag_size_and_the_detector(tmp_path):
    """Scale and keypoint provenance survive save/load — and a project written
    before they existed still loads, with both simply unknown."""
    p = ProjectData(name="take", marker_length=0.05,
                    keypoint_model="rtmpose-balanced")
    save_project(p, tmp_path)
    got = load_project(tmp_path)
    assert got.marker_length == 0.05
    assert got.keypoint_model == "rtmpose-balanced"

    doc = json.loads((tmp_path / "project.json").read_text())
    del doc["marker_length"], doc["keypoint_model"]
    (tmp_path / "project.json").write_text(json.dumps(doc))
    legacy = load_project(tmp_path)
    assert legacy.marker_length is None and legacy.keypoint_model is None


def test_a_rig_without_a_report_is_still_written(tmp_path):
    """save_rig's report is optional: the uploaded-calibration path has no
    provenance to record and must not start failing."""
    intr = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(64, 48))
    from pose3d.calib.extrinsics import Extrinsics
    ext = Extrinsics(R=np.eye(3), t=np.zeros(3))
    save_rig(CalibratedRig(intr, intr, ext, ext), tmp_path / "calibration")
    assert (tmp_path / "calibration" / "extrinsics.json").exists()
    assert not (tmp_path / "calibration" / "report.json").exists()

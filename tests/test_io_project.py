"""Phase 1 verification: project save/reload round-trip (NaN-aware) + log."""
import numpy as np

from pose3d.core.io_project import (
    append_correction, load_project, save_project,
)
from pose3d.core.project import (
    CAM_LEFT, CAM_RIGHT, Correction, Frame, ProjectData,
)
from pose3d.core.skeleton import Joint, NUM_JOINTS


def _make_project():
    p = ProjectData(name="Test_Shoot_01", fps=30, calibration_ref="calibration")
    f = Frame(frame_id="0001",
              images={CAM_LEFT: "images/left_0001.jpg",
                      CAM_RIGHT: "images/right_0001.jpg"})
    # populate some joints, leave others NaN to test NaN round-trip
    f.set_kp(CAM_LEFT, Joint.HEAD, 100.5, 200.5, score=0.9)
    f.set_kp(CAM_RIGHT, Joint.HEAD, 110.5, 200.5, score=0.8)
    f.set_kp(CAM_LEFT, Joint.RIGHT_ANKLE, 50.0, 700.0, score=0.3, corrected=True)
    f.pose3d[Joint.HEAD] = (0.1, 1.7, 0.0)
    f.fitted3d[Joint.HEAD] = (0.1, 1.7, 0.0)
    p.frames.append(f)
    return p


def test_roundtrip(tmp_path):
    p = _make_project()
    save_project(p, tmp_path)
    q = load_project(tmp_path)

    assert q.name == p.name
    assert q.fps == p.fps
    assert q.calibration_ref == p.calibration_ref
    assert q.frame_ids() == p.frame_ids()

    fp, fq = p.frames[0], q.frames[0]
    # images are stored relative to the project folder (so it can be moved or
    # mounted elsewhere) and resolved against it on load
    import json
    from pathlib import Path
    stored = json.loads((tmp_path / "project.json").read_text())["frames"][0]["images"]
    assert stored == fp.images                      # still relative on disk
    for cam, rel in fp.images.items():
        assert Path(fq.images[cam]) == tmp_path / rel   # absolute after load
    # exact values preserved
    assert np.allclose(fq.kp2d[CAM_LEFT][Joint.HEAD], [100.5, 200.5])
    assert fq.scores[CAM_RIGHT][Joint.HEAD] == 0.8
    assert fq.corrected[CAM_LEFT][Joint.RIGHT_ANKLE]
    # NaN preserved where unset
    assert np.isnan(fq.kp2d[CAM_LEFT][Joint.LEFT_WRIST]).all()
    assert np.isnan(fq.pose3d[Joint.LEFT_KNEE]).all()
    assert np.allclose(fq.pose3d[Joint.HEAD], [0.1, 1.7, 0.0])


def test_shapes_preserved(tmp_path):
    p = _make_project()
    save_project(p, tmp_path)
    q = load_project(tmp_path)
    f = q.frames[0]
    assert f.kp2d[CAM_LEFT].shape == (NUM_JOINTS, 2)
    assert f.pose3d.shape == (NUM_JOINTS, 3)


def test_correction_log(tmp_path):
    p = _make_project()
    save_project(p, tmp_path)
    append_correction(tmp_path, Correction(
        "0001", CAM_LEFT, int(Joint.RIGHT_ANKLE),
        (50.0, 700.0), (55.0, 690.0), "2026-07-21T23:00:00"))
    q = load_project(tmp_path)
    assert len(q.corrections) == 1
    c = q.corrections[0]
    assert c.frame_id == "0001"
    assert c.new_xy == (55.0, 690.0)

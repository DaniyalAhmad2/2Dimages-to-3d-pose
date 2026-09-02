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


# --- pipeline version stamp (how a fix reaches an existing take) ----------

def test_version_stamp_round_trips(tmp_path):
    """A project saved by this build reopens with no recompute."""
    from pose3d.core.project import PIPELINE_VERSION

    p = _make_project()
    assert p.pipeline_version == PIPELINE_VERSION     # built in memory = current
    save_project(p, tmp_path)

    import json
    doc = json.loads((tmp_path / "project.json").read_text())
    assert doc["pipeline_version"] == PIPELINE_VERSION
    assert doc["smoothing"] == "none"

    q = load_project(tmp_path)
    assert q.pipeline_version == PIPELINE_VERSION
    assert q.smoothing == "none"
    assert q.keypoint_model == "coco17"


def test_legacy_project_without_version_loads_and_recomputes(tmp_path):
    """A file written before the key existed came out of the causal-EMA build
    and must be recomputed exactly once — so it loads as version 0, not as
    current."""
    import json
    from pose3d.core.project import PIPELINE_VERSION

    save_project(_make_project(), tmp_path)
    doc = json.loads((tmp_path / "project.json").read_text())
    for key in ("pipeline_version", "smoothing", "keypoint_model"):
        doc.pop(key)
    for fd in doc["frames"]:
        fd.pop("filled")
    (tmp_path / "project.json").write_text(json.dumps(doc))

    q = load_project(tmp_path)

    assert q.pipeline_version == 0 < PIPELINE_VERSION
    assert q.smoothing == "none"                      # defaults, not a crash
    assert q.keypoint_model == "coco17"
    assert not q.frames[0].filled.any()
    assert np.allclose(q.frames[0].pose3d[Joint.HEAD], [0.1, 1.7, 0.0])


def test_filled_flags_round_trip(tmp_path):
    p = _make_project()
    p.frames[0].filled[Joint.LEFT_KNEE] = True
    save_project(p, tmp_path)
    q = load_project(tmp_path)
    assert q.frames[0].filled[Joint.LEFT_KNEE]
    assert q.frames[0].filled.sum() == 1
    assert q.frames[0].filled.shape == (NUM_JOINTS,)


def test_smoothing_choice_round_trips(tmp_path):
    p = _make_project()
    p.smoothing = "ema0.6"
    save_project(p, tmp_path)
    assert load_project(tmp_path).smoothing == "ema0.6"


# --- Halpe-26 / head_source ------------------------------------------------

def test_head_source_round_trips(tmp_path):
    """What the canonical HEAD point IS has to survive save/load: the retarget
    corrects a nose HEAD for the nose's ~45 deg forward offset and must not
    correct a skull one, so guessing it would double-correct the head."""
    import json

    p = _make_project()
    assert p.head_source == "nose"            # the safe default in memory
    p.keypoint_model = "halpe26"
    p.head_source = "skull"
    save_project(p, tmp_path)

    doc = json.loads((tmp_path / "project.json").read_text())
    assert doc["head_source"] == "skull"

    q = load_project(tmp_path)
    assert q.head_source == "skull"
    assert q.keypoint_model == "halpe26"

    # a file written before the key existed is a COCO-17 project, whose HEAD
    # is the nose — that is a fact about the data, not a default to be clever
    # about, so it must load as "nose" rather than as anything else
    doc.pop("head_source")
    (tmp_path / "project.json").write_text(json.dumps(doc))
    assert load_project(tmp_path).head_source == "nose"


def test_halpe_frame_loses_nothing(tmp_path):
    """Halpe-26 detections must survive storage unchanged.

    `_json_to_arr` truncates rows past NUM_JOINTS, so a 26-row array would be
    silently cut. It cannot happen — `map_halpe26` reduces the 26 keypoints to
    the 15 canonical joints before anything is stored, and head2d is 5 rows by
    construction — and this is the test that keeps that true.
    """
    from pose3d.core.skeleton import (
        NUM_HEAD_KP, extract_head, map_halpe26,
    )

    rng = np.random.default_rng(4)
    kp = rng.uniform(0, 3000, (26, 2))
    scores = rng.uniform(0.2, 1.0, 26)
    xy, sc = map_halpe26(kp, scores)
    hxy, hsc = extract_head(kp, scores)
    assert xy.shape == (NUM_JOINTS, 2) and hxy.shape == (NUM_HEAD_KP, 2)

    p = ProjectData(name="halpe", keypoint_model="halpe26", head_source="skull")
    f = Frame(frame_id="0001")
    for cam in (CAM_LEFT, CAM_RIGHT):
        f.kp2d[cam] = xy.copy()
        f.scores[cam] = sc.copy()
        f.head2d[cam] = hxy.copy()
        f.head_scores[cam] = hsc.copy()
    p.frames.append(f)

    save_project(p, tmp_path)
    g = load_project(tmp_path).frames[0]

    for cam in (CAM_LEFT, CAM_RIGHT):
        assert np.allclose(g.kp2d[cam], xy, equal_nan=True)
        assert np.allclose(g.scores[cam], sc, equal_nan=True)
        assert np.allclose(g.head2d[cam], hxy, equal_nan=True)
        assert np.allclose(g.head_scores[cam], hsc, equal_nan=True)
        # nothing was dropped off the end of either array
        assert g.kp2d[cam].shape == (NUM_JOINTS, 2)
        assert g.head2d[cam].shape == (NUM_HEAD_KP, 2)


def test_the_raw_2d_and_the_gate_mask_survive_a_round_trip(tmp_path):
    """`kp2d_raw`/`scores_raw`/`rejected` are the difference between a
    rejection you can undo by recalibrating and one you cannot."""
    p = _make_project()
    f = p.frames[0]
    f.kp2d_raw[CAM_LEFT][Joint.HEAD] = (101.5, 201.5)
    f.scores_raw[CAM_LEFT][Joint.HEAD] = 0.95
    f.rejected[CAM_RIGHT][Joint.HEAD] = True
    save_project(p, tmp_path)
    q = load_project(tmp_path)

    g = q.frames[0]
    assert np.allclose(g.kp2d_raw[CAM_LEFT][Joint.HEAD], (101.5, 201.5))
    assert g.scores_raw[CAM_LEFT][Joint.HEAD] == 0.95
    assert g.rejected[CAM_RIGHT][Joint.HEAD]
    assert not g.rejected[CAM_LEFT][Joint.HEAD]
    # the raw arrays are NaN-aware like every other per-joint array
    assert np.isnan(g.kp2d_raw[CAM_RIGHT][Joint.LEFT_KNEE]).all()
    # and the working 2D is NOT overwritten by them
    assert np.allclose(g.kp2d[CAM_LEFT][Joint.HEAD], (100.5, 200.5))


def test_a_project_written_before_the_raw_arrays_still_loads(tmp_path):
    """Back-filled from `kp2d`, with an empty mask: that is the truthful
    reading of a file that has one copy of its 2D and no record of the gate."""
    import json

    p = _make_project()
    save_project(p, tmp_path)
    doc = json.loads((tmp_path / "project.json").read_text())
    for fd in doc["frames"]:
        for key in ("kp2d_raw", "scores_raw", "rejected"):
            del fd[key]
    (tmp_path / "project.json").write_text(json.dumps(doc))

    q = load_project(tmp_path)
    g = q.frames[0]
    for c in (CAM_LEFT, CAM_RIGHT):
        assert np.array_equal(g.kp2d_raw[c], g.kp2d[c], equal_nan=True)
        assert np.array_equal(g.scores_raw[c], g.scores[c], equal_nan=True)
        assert not g.rejected[c].any()
    # back-filled, not aliased: writing one must not write the other
    g.kp2d_raw[CAM_LEFT][Joint.HEAD] = (0.0, 0.0)
    assert np.allclose(g.kp2d[CAM_LEFT][Joint.HEAD], (100.5, 200.5))

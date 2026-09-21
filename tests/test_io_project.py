"""Phase 1 verification: project save/reload round-trip (NaN-aware) + log."""
import sqlite3

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


def test_head_mode_round_trips(tmp_path):
    """Which mode ORIENTS the neck+head chain is a per-project choice the user
    makes in the app, so it has to survive save/load: reopening a take set to
    Face mode and getting Nose back would silently re-pose every frame's head.

    Unlike `head_source` this is a preference, not a fact about the data —
    which is why a file written before the key existed loads as the safe
    default ("nose", the mannequin-safe mode), the same value a fresh project
    starts with.
    """
    import json

    p = _make_project()
    assert p.head_mode == "nose"              # the safe default in memory
    p.head_mode = "face"
    save_project(p, tmp_path)

    doc = json.loads((tmp_path / "project.json").read_text())
    assert doc["head_mode"] == "face"

    q = load_project(tmp_path)
    assert q.head_mode == "face"

    doc.pop("head_mode")
    (tmp_path / "project.json").write_text(json.dumps(doc))
    assert load_project(tmp_path).head_mode == "nose"


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


def _legacy_log(folder, rows):
    """A corrections.sqlite exactly as every build before the toes wrote it:
    no user_version, face rows carrying joint = 15 + k."""
    conn = sqlite3.connect(folder / "corrections.sqlite")
    conn.execute(
        """CREATE TABLE corrections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               frame_id TEXT, cam TEXT, joint INTEGER,
               old_x REAL, old_y REAL, new_x REAL, new_y REAL, ts TEXT)""")
    conn.executemany(
        "INSERT INTO corrections (frame_id,cam,joint,old_x,old_y,new_x,new_y,ts) "
        "VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_the_face_ids_have_a_fixed_base_that_is_not_the_joint_count():
    from pose3d.core.io_project import _LEGACY_NUM_JOINTS
    from pose3d.core.skeleton import (
        FACE_KP_BASE, NUM_HEAD_KP, NUM_JOINTS, face_kp_id, face_kp_index,
        is_face_kp)
    assert FACE_KP_BASE == 100 and NUM_JOINTS < FACE_KP_BASE
    # the count on the day the old convention was retired, frozen: the
    # migration reads files written back then, and NUM_JOINTS has moved since
    assert _LEGACY_NUM_JOINTS == 15
    for k in range(NUM_HEAD_KP):
        assert is_face_kp(face_kp_id(k)) and face_kp_index(face_kp_id(k)) == k
    for j in range(NUM_JOINTS):
        assert not is_face_kp(j)


def test_an_old_log_has_its_face_rows_moved_to_the_fixed_base_once(tmp_path):
    from pose3d.core.io_project import (
        CORRECTIONS_SCHEMA, LEGACY_BACKUP, _read_corrections)
    from pose3d.core.skeleton import face_kp_id
    _legacy_log(tmp_path, [("0001", "left", 6, 1, 2, 3, 4, ""),
                           ("0001", "left", 15, 1, 2, 3, 4, ""),
                           ("0002", "right", 19, 1, 2, 3, 4, "")])

    got = _read_corrections(tmp_path)

    assert [c.joint for c in got] == [6, face_kp_id(0), face_kp_id(4)]
    assert (tmp_path / LEGACY_BACKUP).exists(), "no backup of the old log"
    with sqlite3.connect(tmp_path / "corrections.sqlite") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == CORRECTIONS_SCHEMA
    # reading again rewrites nothing: the migration ran exactly once
    assert [c.joint for c in _read_corrections(tmp_path)] == [c.joint for c in got]


def test_a_new_log_is_stamped_and_never_backed_up(tmp_path):
    from pose3d.core.io_project import (
        CORRECTIONS_SCHEMA, LEGACY_BACKUP, _read_corrections, append_correction)
    from pose3d.core.project import Correction
    from pose3d.core.skeleton import face_kp_id
    append_correction(tmp_path, Correction("0001", "left", face_kp_id(1),
                                           (0.0, 0.0), (1.0, 1.0), ""))
    append_correction(tmp_path, Correction("0001", "left", 16,
                                           (0.0, 0.0), (1.0, 1.0), ""))

    assert not (tmp_path / LEGACY_BACKUP).exists()
    assert [c.joint for c in _read_corrections(tmp_path)] == [face_kp_id(1), 16]
    with sqlite3.connect(tmp_path / "corrections.sqlite") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == CORRECTIONS_SCHEMA


def test_a_log_whose_rows_already_sit_at_the_fixed_base_is_only_stamped(tmp_path):
    """The half-migrated state a killed build can leave behind: rows already
    moved, `user_version` still 0. Moving them a SECOND time — to 185-189,
    where nothing lives — is what the bounded predicate makes impossible: at
    or above the base is already migrated, whatever the version says."""
    from pose3d.core.io_project import (
        CORRECTIONS_SCHEMA, LEGACY_BACKUP, _read_corrections)
    from pose3d.core.skeleton import face_kp_id
    _legacy_log(tmp_path, [("0001", "left", 6, 1, 2, 3, 4, ""),
                           ("0001", "left", face_kp_id(0), 1, 2, 3, 4, ""),
                           ("0002", "right", face_kp_id(4), 1, 2, 3, 4, "")])

    got = _read_corrections(tmp_path)

    assert [c.joint for c in got] == [6, face_kp_id(0), face_kp_id(4)]
    assert not (tmp_path / LEGACY_BACKUP).exists(), \
        "nothing was rewritten, so nothing needed a backup"
    with sqlite3.connect(tmp_path / "corrections.sqlite") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == CORRECTIONS_SCHEMA


def test_a_migration_killed_before_its_stamp_still_migrates_exactly_once(tmp_path):
    """The rows and the version stamp are ONE transaction.

    As two, a process killed between them left the face rows at the fixed
    base with the version still 0, and the next open moved them again — the
    client's eye and ear corrections landing 85 ids past anything that
    exists. The kill is simulated by making the stamp itself fail.
    """
    from pose3d.core.io_project import (
        CORRECTIONS_DB, _migrate_corrections, _read_corrections)
    from pose3d.core.skeleton import face_kp_id

    class _Killed(Exception):
        pass

    class _DiesOnTheStamp(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.strip().upper().startswith("PRAGMA USER_VERSION ="):
                raise _Killed(sql)
            return super().execute(sql, *args)

    _legacy_log(tmp_path, [("0001", "left", 15, 1, 2, 3, 4, ""),
                           ("0002", "right", 19, 1, 2, 3, 4, "")])
    path = tmp_path / CORRECTIONS_DB
    conn = sqlite3.connect(path, factory=_DiesOnTheStamp)
    try:
        _migrate_corrections(path, conn)
    except _Killed:
        pass
    else:                                    # pragma: no cover - the guard
        raise AssertionError("the stamp did not run")
    conn.close()

    assert [c.joint for c in _read_corrections(tmp_path)] == [face_kp_id(0),
                                                              face_kp_id(4)]

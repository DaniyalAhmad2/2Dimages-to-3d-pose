# Toe Joints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two new skeleton joints — the big toes — so the character's feet point where the figure's toes point, in the 3D view and the export alike, without changing any number, colour or file the client already has.

**Architecture:** Face-point ids first get a fixed base (`FACE_KP_BASE = 100`) with a one-time, backed-up migration of `corrections.sqlite`, because today they are `NUM_JOINTS + k`. Then `LEFT_TOE`/`RIGHT_TOE` are appended to `Joint`, mapped from Halpe-26 indices 20/21, and carried through the generic `NUM_JOINTS`/`BONES` loops; every take-wide aggregate is restricted to `CORE_JOINTS` (the original 15); the rig's existing `foot.L/R` bones aim at the toe when it is seen and inherit the shin when it is not.

**Tech Stack:** Python 3.12, numpy, PySide6 (offscreen tests), sqlite3, headless Blender 5.1.1 for the export gate.

**Spec:** `docs/superpowers/specs/2026-09-21-toe-joints-design.md` — read it first; this plan argues from it. One deviation, ruled during planning: `PIPELINE_VERSION` stays 2 (`ui/model.py:178` recomputes any older project on open; a recompute cannot add toes, so a bump would only cost the client a wait). The foot's roll uses the leg's bend plane through the existing `_BEND_REF` machinery (the same reference the shin uses), which is what "keeps the shin's twist" means in this codebase.

## Global Constraints

- Worktree root: `/media/athena/hd3/Projects/pose3d-tool` (branch `fix/accuracy-audit`, base `16c1595`) or a per-task worktree the controller names. Never commit anything under `data/` (stale demo-project modifications live there).
- Tests: `cd <worktree> && PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen /media/athena/hd3/Projects/pose3d-tool/.venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Blender-gated files add `POSE3D_BLENDER=/home/athena/Downloads/blender-5.1.1-linux-x64/blender POSE3D_REQUIRE_BLENDER=1 POSE3D_REQUIRE_ASSETS=1`. Never set `DISPLAY`. No subagents from implementers.
- `Joint` members are APPENDED, never inserted; existing indices are the storage format of every `project.json` and `corrections.sqlite`.
- `FACE_KP_BASE = 100` and `_LEGACY_NUM_JOINTS = 15` are literals that never change again.
- Toes are `EXTREMITY_JOINTS`; every take-wide number and colour (`frame_stat`, `figure_height_px`, the timeline band, the joint-count messages) is computed over `CORE_JOINTS`.
- The floor datum (`placement.ground_datum`) and the rig asset (`assets/character.blend`/`.npz`) are not touched. The BVH/FBX bone list stays the rig's 19 bones.
- Every product change starts with a failing test. Commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: Face-point ids get a fixed base, and old logs are migrated once

**Files:**
- Modify: `pose3d/core/skeleton.py:55-58` (after `HEAD_KP_INDEX`)
- Modify: `pose3d/core/corrections.py:14,36-54,67-96`
- Modify: `pose3d/ui/model.py:704-739` (`_resolve_joint` docstring + branch)
- Modify: `pose3d/ui/camera_view.py:95-108`
- Modify: `pose3d/core/io_project.py:24,238-247,316-326`
- Test: `tests/test_io_project.py`, `tests/test_corrections.py`, `tests/test_head_keypoints.py`, `tests/test_integration_seams.py:170`

**Interfaces:**
- Produces: `skeleton.FACE_KP_BASE: int`, `skeleton.face_kp_id(k) -> int`, `skeleton.is_face_kp(joint) -> bool`, `skeleton.face_kp_index(joint) -> int`; `io_project.CORRECTIONS_SCHEMA: int`, `io_project.LEGACY_BACKUP: str`, `io_project._migrate_corrections(path, conn) -> int`.
- Consumed by every later task: nothing may compare a joint id with `NUM_JOINTS` again.

- [ ] **Step 1: Write the failing tests for the helpers and the migration**

Append to `tests/test_io_project.py`:

```python
import sqlite3


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
    from pose3d.core.skeleton import (
        FACE_KP_BASE, NUM_HEAD_KP, NUM_JOINTS, face_kp_id, face_kp_index,
        is_face_kp)
    assert FACE_KP_BASE == 100 and NUM_JOINTS < FACE_KP_BASE
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
```

Append to `tests/test_corrections.py` (use the file's existing frame/stack fixture style; if it builds frames with `Frame(frame_id=...)`, do the same):

```python
def test_a_face_id_edits_the_face_point_and_a_body_id_the_body(tmp_path):
    from pose3d.core.corrections import CorrectionStack
    from pose3d.core.project import CAM_LEFT, Frame
    from pose3d.core.skeleton import NUM_JOINTS, face_kp_id
    f = Frame(frame_id="0001")
    stack = CorrectionStack({"0001": f})

    stack.apply("0001", CAM_LEFT, face_kp_id(2), 10.0, 20.0)
    assert tuple(f.head2d[CAM_LEFT][2]) == (10.0, 20.0)
    assert f.head_corrected[CAM_LEFT][2]
    assert np.isnan(f.kp2d[CAM_LEFT]).all(), "a face id must not touch a body joint"

    stack.apply("0001", CAM_LEFT, NUM_JOINTS - 1, 30.0, 40.0)
    assert tuple(f.kp2d[CAM_LEFT][NUM_JOINTS - 1]) == (30.0, 40.0)
    assert f.corrected[CAM_LEFT][NUM_JOINTS - 1]

    stack.undo(); stack.undo()
    assert np.isnan(f.head2d[CAM_LEFT][2]).all() and not f.head_corrected[CAM_LEFT][2]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_io_project.py tests/test_corrections.py -q -p no:cacheprovider -k "face_ids or old_log or new_log or face_id_edits"`
Expected: FAIL — `ImportError: cannot import name 'face_kp_id'`.

- [ ] **Step 3: Add the constants and helpers to `pose3d/core/skeleton.py`**

Insert after line 58 (`HEAD_KP_INDEX = ...`):

```python
#: Face keypoints share the body joints' integer id space — the correction
#: stack, the camera views and the `joint` column of corrections.sqlite all
#: address them with one int — at a FIXED offset. It used to be NUM_JOINTS
#: itself, which made every stored eye and ear correction change meaning the
#: day a body joint was appended (the toes, 2026-09-22). Body joints will
#: never reach 100; this constant never changes again, and
#: `io_project._migrate_corrections` moves the old rows here once.
FACE_KP_BASE = 100


def face_kp_id(k: int) -> int:
    """The shared integer id of face keypoint `k` (0..NUM_HEAD_KP-1)."""
    return FACE_KP_BASE + int(k)


def is_face_kp(joint: int) -> bool:
    """True for an id `face_kp_id` produced; False for any body joint."""
    return int(joint) >= FACE_KP_BASE


def face_kp_index(joint: int) -> int:
    """The `k` behind a face id — the inverse of `face_kp_id`."""
    return int(joint) - FACE_KP_BASE
```

- [ ] **Step 4: Route `CorrectionStack` through the helpers**

In `pose3d/core/corrections.py` replace the import on line 14 with:

```python
from pose3d.core.skeleton import face_kp_index, is_face_kp
```

Rewrite `apply`'s docstring and branches (lines 36-54):

```python
    def apply(self, frame_id: str, cam: str, joint: int,
              x: float, y: float, ts: str = "") -> Edit:
        """A face id (`skeleton.face_kp_id(k)`, at the fixed FACE_KP_BASE)
        addresses face keypoint k (nose/eyes/ears); anything below it is a
        body joint. The one convention shared with the camera views and the
        SQLite log, whose integer column simply extends."""
        f = self.frames_by_id[frame_id]
        if is_face_kp(joint):
            k = face_kp_index(joint)
            old = tuple(f.head2d[cam][k])
            edit = Edit(frame_id, cam, joint, (float(old[0]), float(old[1])),
                        (float(x), float(y)), float(f.head_scores[cam][k]),
                        bool(f.head_corrected[cam][k]))
            f.set_head_kp(cam, k, x, y, score=1.0, corrected=True)
        else:
            old = tuple(f.kp2d[cam][joint])
            edit = Edit(frame_id, cam, joint, (float(old[0]), float(old[1])),
                        (float(x), float(y)), float(f.scores[cam][joint]),
                        bool(f.corrected[cam][joint]))
            f.set_kp(cam, joint, x, y, score=1.0, corrected=True)
```

In `undo` replace `if e.joint >= NUM_JOINTS:` / `k = e.joint - NUM_JOINTS` with `if is_face_kp(e.joint):` / `k = face_kp_index(e.joint)`; in `redo` replace `if e.joint >= NUM_JOINTS:` with `if is_face_kp(e.joint):` and `e.joint - NUM_JOINTS` with `face_kp_index(e.joint)`.

- [ ] **Step 5: Same in the model and the camera view**

`pose3d/ui/model.py`: add `face_kp_index, is_face_kp` to the `pose3d.core.skeleton` import at the top of the file (find the existing `from pose3d.core.skeleton import (...)` block). In `_resolve_joint` change the docstring line 707-708 to "A face id (`skeleton.face_kp_id`) addresses face keypoint `face_kp_index(joint)`; the camera views and the correction stack share this convention." and line 726 `if joint >= NUM_JOINTS:` to `if is_face_kp(joint):`. Grep the file for any other `>= NUM_JOINTS` / `- NUM_JOINTS` and convert them the same way (there is one more in the drag path near line 826 where the HEAD/nose sync compares ids — convert it).

`pose3d/ui/camera_view.py`: change line 108 to

```python
FACE_KP_IDS = tuple(face_kp_id(k) for k in range(NUM_HEAD_KP))
```

import `face_kp_id` from `pose3d.core.skeleton`, and reword the comment at 98-99 to "Their item ids are `face_kp_id(k)` — the fixed base the model and the correction stack share."

- [ ] **Step 6: The migration in `pose3d/core/io_project.py`**

Add `import shutil` and `from pose3d.core.skeleton import FACE_KP_BASE, NUM_HEAD_KP, NUM_JOINTS` at the top. After `CORRECTIONS_DB = "corrections.sqlite"` add:

```python
#: `PRAGMA user_version` of corrections.sqlite. 0 is every log written before
#: face ids had a fixed base (face rows carry joint = 15 + k); at
#: CORRECTIONS_SCHEMA they carry `skeleton.face_kp_id(k)`.
CORRECTIONS_SCHEMA = 2
#: NUM_JOINTS on the day the old convention was retired — a literal on
#: purpose: the migration reads OLD files, and NUM_JOINTS has since grown.
_LEGACY_NUM_JOINTS = 15
#: The untouched copy the migration leaves beside the log, once.
LEGACY_BACKUP = "corrections.sqlite.pre-toes"


def _migrate_corrections(path: Path, conn: sqlite3.Connection) -> int:
    """Bring a log up to CORRECTIONS_SCHEMA; returns the rows rewritten.

    Runs on every open and costs one PRAGMA when there is nothing to do. A
    log at version 0 that holds rows at or above the legacy joint count is
    an old log whose face rows are about to be misread as body joints: it is
    copied to LEGACY_BACKUP first (never overwritten if that exists), then
    those rows move to the fixed base. A fresh, empty log is just stamped.
    """
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version >= CORRECTIONS_SCHEMA:
        return 0
    n_old = int(conn.execute(
        "SELECT COUNT(*) FROM corrections WHERE joint >= ?",
        (_LEGACY_NUM_JOINTS,)).fetchone()[0])
    if n_old:
        backup = path.with_name(LEGACY_BACKUP)
        if not backup.exists():
            conn.commit()
            shutil.copy2(path, backup)
        with conn:
            conn.execute(
                "UPDATE corrections SET joint = ? + (joint - ?) WHERE joint >= ?",
                (FACE_KP_BASE, _LEGACY_NUM_JOINTS, _LEGACY_NUM_JOINTS))
    with conn:
        conn.execute(f"PRAGMA user_version = {CORRECTIONS_SCHEMA}")
    return n_old
```

Change `_connect` to run it after the table exists:

```python
def _connect(folder: Path) -> sqlite3.Connection:
    path = folder / CORRECTIONS_DB
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS corrections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               frame_id TEXT, cam TEXT, joint INTEGER,
               old_x REAL, old_y REAL, new_x REAL, new_y REAL, ts TEXT)""")
    _migrate_corrections(path, conn)
    return conn
```

- [ ] **Step 7: Move the tests off the old convention**

In `tests/test_head_keypoints.py` replace every `NUM_JOINTS + 0`, `NUM_JOINTS + 1`, `NUM_JOINTS + 2`, `NUM_JOINTS + 3`, `NUM_JOINTS + 4`, `NUM_JOINTS + HEAD_KP_INDEX["nose"]` and `NUM_JOINTS + k` with `face_kp_id(0)` … `face_kp_id(k)` (lines 181, 210, 252, 258, 290, 442, 507, 509, 510, 538, 539), importing `face_kp_id` from `pose3d.core.skeleton`. In `tests/test_integration_seams.py:170` change `eye = NUM_JOINTS + 1` to `eye = face_kp_id(1)` with the import. Grep `tests/` and `pose3d/` for `NUM_JOINTS *[+-]` afterwards: it must return nothing.

- [ ] **Step 8: Run the tests**

Run: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_io_project.py tests/test_corrections.py tests/test_head_keypoints.py tests/test_integration_seams.py tests/test_camera_view_handles.py tests/test_ui_smoke.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add pose3d/core/skeleton.py pose3d/core/corrections.py pose3d/ui/model.py pose3d/ui/camera_view.py pose3d/core/io_project.py tests/test_io_project.py tests/test_corrections.py tests/test_head_keypoints.py tests/test_integration_seams.py
git commit -m "core: face-point ids get a fixed base, and old correction logs move to it once

Face keypoints were addressed as NUM_JOINTS + k in the stack, the views and
the joint column of corrections.sqlite, so appending a body joint would have
re-read every saved eye and ear correction as a body joint. FACE_KP_BASE is
100 and never changes; a version-0 log is backed up beside itself and its
face rows moved, exactly once.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Two toe joints in the skeleton, the detector mapping and the loader

**Files:**
- Modify: `pose3d/core/skeleton.py:73-116,138-154,215-218,276-278`
- Modify: `pose3d/geometry/bonefit.py:47-62`
- Modify: `pose3d/core/io_project.py:204`
- Modify: `pose3d/detect/base.py:25` (comment only)
- Modify: `tests/synth.py:109-133`
- Test: `tests/test_skeleton.py`, `tests/test_io_project.py`

**Interfaces:**
- Consumes: Task 1 (no `NUM_JOINTS` id comparisons remain).
- Produces: `Joint.LEFT_TOE = 15`, `Joint.RIGHT_TOE = 16`, `NUM_JOINTS == 17`, `BONES` with 16 edges, `skeleton.EXTREMITY_JOINTS`, `skeleton.CORE_JOINTS`, `skeleton.CORE_INDEX: list[int]`, `skeleton.EXTREMITY_PARENT: dict[Joint, Joint]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_skeleton.py`, replace the "six foot keypoints have nowhere to go" block (lines 111-115) of `test_map_halpe26` with:

```python
    # the two big toes ARE mapped — one point per foot, the foot's direction
    assert np.allclose(cxy[Joint.LEFT_TOE], xy[HALPE26_INDEX["left_big_toe"]])
    assert np.allclose(cxy[Joint.RIGHT_TOE], xy[HALPE26_INDEX["right_big_toe"]])
    assert csc[Joint.LEFT_TOE] == sc[HALPE26_INDEX["left_big_toe"]]
    # the small toes and heels still have nowhere to go, and must not leak
    for foot in ("left_small_toe", "right_small_toe", "left_heel", "right_heel"):
        assert not (cxy == xy[HALPE26_INDEX[foot]]).all(axis=1).any()
```

and append:

```python
def test_the_toes_are_appended_extremities_and_the_core_set_is_the_old_fifteen():
    from pose3d.core.skeleton import (
        CORE_INDEX, CORE_JOINTS, EXTREMITY_JOINTS, EXTREMITY_PARENT, BONES)
    assert Joint.LEFT_TOE == 15 and Joint.RIGHT_TOE == 16 and NUM_JOINTS == 17
    assert Joint.RIGHT_ANKLE == 14, "existing indices are a file format"
    assert EXTREMITY_JOINTS == (Joint.LEFT_TOE, Joint.RIGHT_TOE)
    assert CORE_JOINTS == tuple(Joint)[:15] and CORE_INDEX == list(range(15))
    assert (Joint.LEFT_ANKLE, Joint.LEFT_TOE) in BONES
    assert (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE) in BONES
    assert EXTREMITY_PARENT == {Joint.LEFT_TOE: Joint.LEFT_ANKLE,
                                Joint.RIGHT_TOE: Joint.RIGHT_ANKLE}


def test_coco17_leaves_the_toes_undetected():
    xy, sc = _fake_coco()
    cxy, csc = derive_joints(xy, sc)
    assert np.isnan(cxy[Joint.LEFT_TOE]).all() and np.isnan(cxy[Joint.RIGHT_TOE]).all()
    assert csc[Joint.LEFT_TOE] == 0.0


def test_every_bone_has_a_fallback_length():
    from pose3d.geometry.bonefit import fallback_bone_lengths
    lengths = fallback_bone_lengths()
    assert set(lengths) == {(int(a), int(b)) for a, b in BONES}
```

In `tests/test_io_project.py` append:

```python
def test_a_fifteen_joint_project_loads_with_undetected_toes(tmp_path):
    """Every project saved before the toes: 15 rows in every per-joint array.
    They load with NaN toes, and — the one array that used to be truncated
    instead of padded — a `corrected` flag vector of the full length."""
    import json
    from pose3d.core.skeleton import Joint
    p = ProjectData(name="old", keypoint_model="halpe26", head_source="skull")
    f = Frame(frame_id="0001")
    for cam in (CAM_LEFT, CAM_RIGHT):
        f.kp2d[cam][:] = 1.0
        f.corrected[cam][3] = True
    p.frames.append(f)
    save_project(p, tmp_path)
    doc = json.loads((tmp_path / "project.json").read_text(encoding="utf-8"))
    fd = doc["frames"][0]
    for cam in (CAM_LEFT, CAM_RIGHT):
        for key in ("kp2d", "scores", "kp2d_raw", "scores_raw", "corrected", "rejected"):
            fd[key][cam] = fd[key][cam][:15]
    for key in ("pose3d", "fitted3d", "filled"):
        fd[key] = fd[key][:15]
    (tmp_path / "project.json").write_text(json.dumps(doc), encoding="utf-8")

    g = load_project(tmp_path).frames[0]

    for cam in (CAM_LEFT, CAM_RIGHT):
        assert g.kp2d[cam].shape == (NUM_JOINTS, 2)
        assert np.isnan(g.kp2d[cam][Joint.LEFT_TOE]).all()
        assert g.corrected[cam].shape == (NUM_JOINTS,) and g.corrected[cam][3]
        assert not g.corrected[cam][Joint.LEFT_TOE]
    assert g.pose3d.shape == (NUM_JOINTS, 3) and g.filled.shape == (NUM_JOINTS,)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_skeleton.py tests/test_io_project.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: LEFT_TOE`, and the fifteen-joint test fails on `corrected` shape `(15,)`.

- [ ] **Step 3: The skeleton**

In `pose3d/core/skeleton.py` change the `Joint` docstring and members (lines 73-93):

```python
class Joint(IntEnum):
    """Canonical joint set used everywhere downstream of detection.

    COCO-17 based, plus the two big toes Halpe-26 detects — one point per
    foot, so the foot has a direction (the client's build-17 question,
    2026-09-21). Small toes and heels are still excluded. Members are
    APPENDED, never inserted: the indices are the storage format of every
    project.json and corrections.sqlite.
    """
    HEAD = 0          # Halpe-26: the skull vertex; COCO-17: the nose
    NECK = 1          # derived: midpoint(shoulders)
    LEFT_SHOULDER = 2
    RIGHT_SHOULDER = 3
    LEFT_ELBOW = 4
    RIGHT_ELBOW = 5
    LEFT_WRIST = 6
    RIGHT_WRIST = 7
    PELVIS = 8        # derived: midpoint(hips)
    LEFT_HIP = 9
    RIGHT_HIP = 10
    LEFT_KNEE = 11
    RIGHT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_ANKLE = 14
    LEFT_TOE = 15     # Halpe-26: the left big toe; COCO-17: never detected
    RIGHT_TOE = 16    # Halpe-26: the right big toe
```

Append two edges to `BONES` (after `(Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),`):

```python
    (Joint.LEFT_ANKLE, Joint.LEFT_TOE),
    (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE),
```

Directly after `BONES` add:

```python
#: The toes are EXTREMITIES: often cropped or hidden, one point each. They
#: colour their own handle and dot and nothing else — every take-wide number
#: and colour (`ui.model.frame_stat`, `quality.figure_height_px`, the
#: timeline band, the joint-count messages) is computed over CORE_JOINTS, the
#: set the app shipped with, so a take with its feet out of frame reads
#: exactly as it did before the toes existed.
EXTREMITY_JOINTS: tuple[Joint, ...] = (Joint.LEFT_TOE, Joint.RIGHT_TOE)
CORE_JOINTS: tuple[Joint, ...] = tuple(j for j in Joint if j not in EXTREMITY_JOINTS)
CORE_INDEX: list[int] = [int(j) for j in CORE_JOINTS]
#: An extremity's parent (the ankle for a toe): where a camera view parks the
#: placeholder for a toe it has never seen.
EXTREMITY_PARENT: dict[Joint, Joint] = {
    child: parent for parent, child in BONES if child in EXTREMITY_JOINTS}
```

Add to `MIXAMO_BONE`:

```python
    Joint.LEFT_TOE: "mixamorig:LeftToeBase",
    Joint.RIGHT_TOE: "mixamorig:RightToeBase",
```

Replace `_DIRECT_FROM_HALPE26` (lines 213-218) with:

```python
# The joints Halpe-26 and COCO-17 agree about, index for index, plus the two
# big toes only Halpe has. HEAD, NECK and PELVIS are deliberately absent:
# those three are the policy (see map_halpe26).
_DIRECT_FROM_HALPE26: dict[Joint, int] = {
    **{joint: idx for joint, idx in _DIRECT_FROM_COCO.items()
       if joint is not Joint.HEAD},
    Joint.LEFT_TOE: HALPE26_INDEX["left_big_toe"],
    Joint.RIGHT_TOE: HALPE26_INDEX["right_big_toe"],
}
```

Replace the `map_halpe26` docstring lines 276-277 with: "The big toes (Halpe 20/21) map to `LEFT_TOE`/`RIGHT_TOE`; the small toes and heels (22-25) are not mapped — one point per foot gives the foot a direction, which is all the rig's foot bone can take." Update the `derive_joints` comment at 180-181 to "(feet — COCO-17 has none, so the toes stay NaN)".

- [ ] **Step 4: Fallback lengths, the loader, the comment**

`pose3d/geometry/bonefit.py`: append to `_FALLBACK_LENGTHS`:

```python
    (Joint.LEFT_ANKLE, Joint.LEFT_TOE): 0.20,      # ankle to big toe, 1.75 m adult
    (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE): 0.20,
```

`pose3d/core/io_project.py:204`: replace `fr.corrected[c] = np.array(fd["corrected"][c][:NUM_JOINTS], dtype=bool)` with `fr.corrected[c] = _json_to_flags(fd["corrected"][c])` (padded like every other array; a 15-row file gets False toes).

`pose3d/detect/base.py:25`: change the comment "so NUM_JOINTS stays 15" to "so NUM_JOINTS is the skeleton's own count (17 with the toes)".

`tests/synth.py`: change the docstring to "(NUM_JOINTS joints)" and append two rows before the closing `]`:

```python
        [-0.12, 0.20, 0.02],  # LEFT_TOE  (forward of the ankle, on the ground)
        [0.12, 0.20, 0.02],   # RIGHT_TOE
```

- [ ] **Step 5: Run the two files, then the whole suite (no Blender needed yet)**

Run: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_skeleton.py tests/test_io_project.py -q -p no:cacheprovider` → PASS.
Then: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests -q -p no:cacheprovider -x --timeout=600`.
Expected: everything that pinned "15" in prose only still passes; fix any test that hard-codes a 15-row literal by using `NUM_JOINTS`/`sample_skeleton_3d`. Tests that assert bone-length CV / symmetry over the client fixture (`tests/test_client_regression.py`) must pass unchanged — the fixture's toes are NaN and NaN pairs are skipped. If `test_client_regression` moves, STOP and report: it means an aggregate already counts the toes, which Task 3 handles — do not loosen a threshold.

- [ ] **Step 6: Commit**

```bash
git add pose3d/core/skeleton.py pose3d/geometry/bonefit.py pose3d/core/io_project.py pose3d/detect/base.py tests/synth.py tests/test_skeleton.py tests/test_io_project.py
git commit -m "skeleton: the two big toes, appended — detected by Halpe-26, NaN under COCO-17, padded on load

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Every take-wide number stays on the core set; toes get a placeholder under the ankle and a hint

**Files:**
- Modify: `pose3d/ui/model.py:925-941,1111-1120` (+ a new `predates_toes`)
- Modify: `pose3d/quality.py:153-167`
- Modify: `pose3d/ui/main_window.py:169-190,984-1018`
- Modify: `pose3d/ui/camera_view.py:338-352`
- Test: `tests/test_ui_smoke.py`, `tests/test_camera_view_handles.py`, `tests/test_client_regression.py` (unchanged, must stay green)

**Interfaces:**
- Consumes: `skeleton.CORE_INDEX`, `CORE_JOINTS`, `EXTREMITY_JOINTS`, `EXTREMITY_PARENT` (Task 2).
- Produces: `ProjectModel.predates_toes() -> bool`; `main_window.TOES_HINT: str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ui_smoke.py` (the file's `_keyboard_window(n)` helper builds a window over `_long_project`, whose frames come from `sample_skeleton_3d` and now carry toes):

```python
def test_missing_toes_do_not_change_the_frame_band_or_the_dial(qapp):
    """The toes are extremities: a frame whose every core joint is good and
    whose toes the detector never saw reads exactly as it did before the
    toes existed — its dot, its dial, its figure height."""
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT
    from pose3d.core.skeleton import Joint
    from pose3d.ui.model import frame_stat, worst_per_joint
    win = _keyboard_window(n=4)
    m = win.model
    before_h = dict(m.figure_h_px())
    errs_before = m._accuracy(1)
    stat_before = frame_stat(worst_per_joint(errs_before, "measured"))

    for f in m.project.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.kp2d[cam][[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = np.nan
            f.scores[cam][[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = 0.0
        f.pose3d[[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = np.nan
        f.fitted3d[[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = np.nan
    m.invalidate_readouts()

    assert m.figure_h_px() == pytest.approx(before_h, rel=1e-9)
    stat_after = frame_stat(worst_per_joint(m._accuracy(1), "measured"))
    assert stat_after == pytest.approx(stat_before, rel=1e-9)
    win._refresh_timeline_status()
    assert win.timeline.status(1) != "red" or not np.isfinite(stat_before)


def test_a_project_detected_before_the_toes_gets_one_hint(qapp):
    from PySide6.QtWidgets import QApplication
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT
    from pose3d.core.skeleton import Joint
    from pose3d.ui.main_window import MainWindow, TOES_HINT
    from pose3d.ui.model import ProjectModel
    data, rig, _ = _long_project(3)
    data.keypoint_model = "halpe26"
    for f in data.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.kp2d[cam][[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = np.nan
    model = ProjectModel(data, rig)
    assert model.predates_toes()
    win = MainWindow(model)
    win.show(); QApplication.processEvents()
    assert win.statusBar().currentMessage() == TOES_HINT

    data2, rig2, _ = _long_project(3)
    data2.keypoint_model = "halpe26"
    assert not ProjectModel(data2, rig2).predates_toes(), "toes present: no hint"
    data3, rig3, _ = _long_project(3)
    data3.keypoint_model = "coco17"
    for f in data3.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.kp2d[cam][[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = np.nan
    assert not ProjectModel(data3, rig3).predates_toes(), "COCO-17 cannot add toes"
```

Append to `tests/test_camera_view_handles.py` (use its existing fixture that builds a `CameraView` with an image and a pose — copy the pattern of `test_an_undetected_joint_gets_a_placeholder_where_it_was_last_seen`):

```python
def test_a_never_seen_toe_is_parked_under_its_ankle(qapp, tmp_path):
    """A toe the view has never had a position for parks just below the
    ankle it belongs to, not in the middle of the picture: that is where the
    user's cursor already is when the foot needs fixing."""
    from pose3d.core.skeleton import Joint
    view, pose = _view_with_pose(tmp_path)          # the file's helper
    pose[Joint.LEFT_TOE] = np.nan
    view.set_pose(pose)
    toe = view._joints[int(Joint.LEFT_TOE)]
    ankle = view._joints[int(Joint.LEFT_ANKLE)]
    assert toe.is_placeholder
    assert abs(toe.pos().x() - ankle.pos().x()) < 1e-6
    assert toe.pos().y() > ankle.pos().y(), "below the ankle, in image coordinates"
    assert toe.pos().y() - ankle.pos().y() <= 0.1 * view._pixmap_item.boundingRect().height()
```

(If the file has no such helper, build the view the way `test_an_undetected_joint_gets_a_placeholder_where_it_was_last_seen` does and name the helper `_view_with_pose`.)

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_ui_smoke.py tests/test_camera_view_handles.py -q -p no:cacheprovider -k "toes or toe_is_parked"`
Expected: FAIL — `figure_h_px` changes when the toes go NaN; `TOES_HINT` import error; the toe parks at the image centre.

- [ ] **Step 3: The core-set rule**

`pose3d/ui/model.py` — `frame_stat` (lines 1111-1120):

```python
def frame_stat(per_joint):
    """One number for a frame: the MEDIAN CORE joint, not the worst.

    The worst of 15 joints is a max over 15 samples; on a good take it is red
    almost every frame, which is how the old timeline managed to be red 21
    times out of 26 (and green never) and tell the user nothing.

    Over CORE_JOINTS only: the toes are extremities that are often out of
    frame, and a take whose feet are cropped must read exactly as it did
    before they existed. They keep their own handle and dot colour.
    """
    a = np.asarray(per_joint, float)
    if a.size == NUM_JOINTS:
        a = a[CORE_INDEX]
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")
```

(import `CORE_INDEX` from `pose3d.core.skeleton` at the top.) Add to `ProjectModel` beside `figure_h_px`:

```python
    def predates_toes(self) -> bool:
        """True for a Halpe-26 project detected before the toe joints
        existed: every toe of every frame is NaN in both views. Such a take
        can gain its toes from Run Detection; a COCO-17 take cannot, and a
        take that has any toe was detected by this build."""
        from pose3d.core.skeleton import EXTREMITY_JOINTS
        idx = [int(j) for j in EXTREMITY_JOINTS]
        if self.project.keypoint_model != "halpe26" or not self.project.frames:
            return False
        return all(np.isnan(f.kp2d[c][idx]).all()
                   for f in self.project.frames for c in CAMERAS)
```

`pose3d/quality.py` `figure_height_px` (lines 153-167):

```python
def figure_height_px(kp2d: np.ndarray) -> float:
    """Median over frames of the 2D bounding-box height of one camera's finite
    CORE keypoints, in pixels.

    The two cameras are 2:1 apart in resolution, so a pixel error means twice
    as much in the right view as in the left. Every px figure is reported
    against this per-camera denominator as well as raw. The toes are left
    out so the denominator — and every percentage built on it — does not move
    when a foot comes into frame.
    """
    kp2d = np.asarray(kp2d, float).reshape(-1, NUM_JOINTS, 2)[:, CORE_INDEX]
    heights = []
    for f in kp2d:
        ys = f[np.isfinite(f).all(1), 1]
        if ys.size >= 2:
            heights.append(float(ys.max() - ys.min()))
    return float(np.median(heights)) if heights else float("nan")
```

(import `CORE_INDEX` in the file's `pose3d.core.skeleton` import.)

`pose3d/ui/main_window.py` `_on_export` (lines 986-1011): replace `from pose3d.core.skeleton import NUM_JOINTS` with `from pose3d.core.skeleton import CORE_INDEX, CORE_JOINTS`, count over the core set:

```python
        per_frame = [int((~np.isnan(f.fitted3d[CORE_INDEX]).any(1)
                          & ~np.asarray(f.filled, bool)[CORE_INDEX]).sum())
                     for f in frames]
```

and the message `f"Only about {avg:.0f} of {len(CORE_JOINTS)} joints were reconstructed "`.

- [ ] **Step 4: The hint and the placeholder**

`pose3d/ui/main_window.py`: module-level constant after `HEAD_MODE_ITEMS`:

```python
#: Shown once on opening a Halpe-26 project whose toes were never detected —
#: every project saved before the toe joints existed. Nothing is re-detected
#: uninvited; Run Detection adds them.
TOES_HINT = ("This project was detected before toe points existed — "
             "Run Detection adds them.")
```

At the end of `MainWindow.__init__`, replace `self.statusBar().showMessage("Ready")` (line 190) with:

```python
        self.statusBar().showMessage(
            TOES_HINT if self.model.predates_toes() else "Ready", 15000)
```

`pose3d/ui/camera_view.py` `_placeholder_pos` (lines 338-352):

```python
    def _placeholder_pos(self, j: int) -> QPointF:
        """Where to park the handle for a joint this view did not detect.

        The last position this view had for the joint — for a dropout mid-take
        that is the previous frame's, a few pixels from where the limb really
        is. A toe never seen in this view parks just below its ankle (the
        cursor is already there when a foot needs fixing). Else the middle of
        the image, the one point always on screen.
        """
        prev = self._last_seen.get(j)
        if prev is not None:
            return QPointF(prev)
        parent = EXTREMITY_PARENT.get(Joint(j))
        if parent is not None:
            anchor = self._joints[int(parent)]
            if anchor.isVisible() and not anchor.is_placeholder:
                drop = 0.05 * (self._pixmap_item.boundingRect().height()
                               if self._pixmap_item is not None else 0.0)
                return QPointF(anchor.pos().x(), anchor.pos().y() + drop)
        if self._pixmap_item is not None:
            r = self._pixmap_item.boundingRect()
            if r.width() and r.height():
                return r.center()
        return self._scene.sceneRect().center()
```

(import `EXTREMITY_PARENT, Joint` from `pose3d.core.skeleton`.) `set_pose` positions joints in `Joint` order, so the ankle (13/14) is placed before its toe (15/16) is asked for.

- [ ] **Step 5: Run the UI files and the client regression**

Run: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_ui_smoke.py tests/test_camera_view_handles.py tests/test_camera_view_accuracy.py tests/test_client_regression.py tests/test_quality.py tests/test_theme.py -q -p no:cacheprovider --timeout=600`.
Expected: PASS; `test_client_regression` numbers unchanged.

- [ ] **Step 6: Commit**

```bash
git add pose3d/ui/model.py pose3d/quality.py pose3d/ui/main_window.py pose3d/ui/camera_view.py tests/test_ui_smoke.py tests/test_camera_view_handles.py
git commit -m "ui: the toes colour their own handle and nothing else; a never-seen toe parks under its ankle; one hint for pre-toe projects

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The character's foot aims at the toe; the export gate covers it

**Files:**
- Modify: `pose3d/geometry/character.py:220-239,252-299,342-364`
- Modify: `tests/test_export_smoke.py:404-443`
- Test: `tests/test_character.py`, `tests/test_retarget.py` (existing must stay green), `tests/test_export_smoke.py`

**Interfaces:**
- Consumes: `Joint.LEFT_TOE/RIGHT_TOE`, `sample_skeleton_3d` with 17 rows (Task 2).
- Produces: `Character._joint_src` has entries for the toes (`("foot.L", "tail")`); `posed_joints()[LEFT_TOE]` is the foot's tail.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_character.py`:

```python
@needs_character()
def test_the_foot_aims_at_the_toe_when_it_is_seen():
    """One point per foot: the posed foot bone points from the ankle toward
    the captured big toe (pitch and yaw measured; roll from the leg's bend
    plane), rotation only, rest length kept."""
    from pose3d.geometry.character import Character
    ch = Character()
    pose = sample_skeleton_3d()
    ch.fit_to_subject(pose[None])
    valid = ~np.isnan(pose).any(1)
    j = ch.posed_joints(pose, valid)
    for ankle, toe in ((Joint.LEFT_ANKLE, Joint.LEFT_TOE),
                       (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE)):
        want = pose[toe] - j[ankle]                 # from the posed ankle
        got = j[toe] - j[ankle]
        cos = np.dot(want, got) / (np.linalg.norm(want) * np.linalg.norm(got))
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 1.0, (ankle, toe)
    # the foot keeps the rig's own length: the toe is a bone tail, not the point
    rest = ch.rest_joints()
    for ankle, toe in ((Joint.LEFT_ANKLE, Joint.LEFT_TOE),
                       (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE)):
        assert np.linalg.norm(j[toe] - j[ankle]) == pytest.approx(
            np.linalg.norm(rest[toe] - rest[ankle]) * ch._scale, rel=1e-6)


@needs_character()
def test_a_missing_toe_leaves_the_foot_exactly_as_before():
    """No toe: the foot rides the shin's matrix verbatim, which is what every
    take without feet got before the toes existed."""
    from pose3d.geometry.character import Character
    ch = Character()
    pose = sample_skeleton_3d()
    pose[[Joint.LEFT_TOE, Joint.RIGHT_TOE]] = np.nan
    valid = ~np.isnan(pose).any(1)
    skin = ch._skin_matrices(pose, valid)[0]
    for foot, shin in (("foot.L", "shin.L"), ("foot.R", "shin.R")):
        assert np.allclose(skin[ch.role[foot]], skin[ch.role[shin]])
    j = ch.posed_joints(pose, valid)
    assert np.isfinite(j[Joint.LEFT_TOE]).all(), "the toe is read off the foot's tail"
```

In `tests/test_export_smoke.py::test_bvh_keyframes_match_the_view` extend the mapping so tail-sourced joints (the toes) are compared too. Replace lines 425-429 with:

```python
    names = [b.name for b in bvh.joints]

    def _row(b, which):
        """The FK row holding this rig point: the bone for its head, the
        bone's End Site child for its tail (a leaf bone's tail is an End
        Site in the file; `parse` keeps End Sites as channel-less joints)."""
        if ch.bone_names[b] not in names:
            return None
        bi = bvh.index(ch.bone_names[b])
        if which == "head":
            return bi
        ends = [c for c in bvh.joints[bi].children if not bvh.joints[c].channels]
        return ends[0] if len(ends) == 1 else None

    mapping = {j: r for j, (b, which) in ch._joint_src.items()
               if which in ("head", "tail") and (r := _row(b, which)) is not None}
    assert len(mapping) >= 12
    assert int(Joint.LEFT_TOE) in mapping and int(Joint.RIGHT_TOE) in mapping, \
        "the toes must be inside the export-matches-view gate"
```

If `Bvh.forward_kinematics` turns out not to return rows for End Sites (check `pose3d/export/bvh.py:165-190`), compute the tail as `fk[bi] + bvh.world_rotations(row)[bi] @ bvh.joints[end].offset` inside the loop instead — do not drop the toes from the gate.

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=$PWD POSE3D_REQUIRE_ASSETS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_character.py -q -p no:cacheprovider -k "foot_aims or missing_toe"`
Expected: FAIL — `posed_joints()[LEFT_TOE]` is NaN (no `_joint_src` entry), the aim assertion fails.

- [ ] **Step 3: Give the foot a target, a roll reference and a read-back point**

`pose3d/geometry/character.py` — append to `_DIRECT` (after `"shin.R"`):

```python
    # one point per foot: the foot aims from the ankle at the big toe when
    # it is seen, and rides the shin's matrix — today's behaviour — when not
    "foot.L": (Joint.LEFT_ANKLE, Joint.LEFT_TOE),
    "foot.R": (Joint.RIGHT_ANKLE, Joint.RIGHT_TOE),
```

Append to `_BEND_REF` (after `"shin.R"`), so the foot's roll follows the same leg bend plane the shin's does:

```python
    "foot.L":      (Joint.LEFT_HIP, Joint.LEFT_KNEE,
                    Joint.LEFT_ANKLE, _HIP_LINE),
    "foot.R":      (Joint.RIGHT_HIP, Joint.RIGHT_KNEE,
                    Joint.RIGHT_ANKLE, _HIP_LINE),
```

Append to `_JOINT_FROM_RIG` (after `Joint.RIGHT_ANKLE`):

```python
    Joint.LEFT_TOE: (("foot.L", "tail"),),
    Joint.RIGHT_TOE: (("foot.R", "tail"),),
```

Update the prose: line 261 "there are no hand or foot keypoints" → "there are no hand keypoints, and the foot's one point fixes its aim, not its spin"; lines 296-298 → "`clavicle.*` and `hand.*` get no reference — they have no keypoints of their own, so once the parent's roll is right theirs is inherited right; `foot.*` follows the leg's bend plane like the shin."

- [ ] **Step 4: Run the rig and export suites**

Run: `PYTHONPATH=$PWD POSE3D_REQUIRE_ASSETS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_character.py tests/test_retarget.py tests/test_view_orientation.py -q -p no:cacheprovider` → PASS (`test_no_bone_is_ever_scaled`, `test_bones_stay_connected_when_posed`, `test_the_rest_face_reference_points_at_the_face` and the ankle-based grounding tests all unchanged).
Then: `PYTHONPATH=$PWD POSE3D_BLENDER=/home/athena/Downloads/blender-5.1.1-linux-x64/blender POSE3D_REQUIRE_BLENDER=1 POSE3D_REQUIRE_ASSETS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_export_smoke.py tests/test_client_regression.py -q -p no:cacheprovider --timeout=900` → PASS, including the gate now covering 17 joints and `test_the_recorded_rig_height_is_still_the_bundled_rigs` (the rig is untouched).

- [ ] **Step 5: Commit**

```bash
git add pose3d/geometry/character.py tests/test_character.py tests/test_export_smoke.py
git commit -m "character: the foot aims at the big toe when it is seen, rides the shin when not; the export gate covers the toes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Docs, release note, decisions

**Files:**
- Modify: `README.md:161`, `docs/client/RELEASE_NOTES_v1.md` (heading stays until the build is known; add one line under "Working with the frames"), `docs/DECISIONS.md`, `tests/test_diagnostics.py:477-495` (`CLIENT_COMPLAINTS`)

- [ ] **Step 1: Write the failing test**

Add to `CLIENT_COMPLAINTS` in `tests/test_diagnostics.py`:

```python
    "P17 does the program not detect toe position?": "toe",
```

Run `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_diagnostics.py -q -p no:cacheprovider -k release_note` → FAIL (no "toe" in the note).

- [ ] **Step 2: Write the docs**

`docs/client/RELEASE_NOTES_v1.md`, under "## Working with the frames", after the missing-joints bullet:

```markdown
- **"Does the program not detect toe position?"** It does now: one point per
  foot, the big toe, so the character's foot points where the figure's toes
  point. A toe the detector could not see is a red dashed handle under the
  ankle that you can drag into place. Projects made before this build show
  those handles until you press Run Detection once. *Check: a side-on frame —
  the foot follows the toe.*
```

`README.md:161`: "- The skeleton is 17 joints — the 15 body joints plus one big-toe point per foot; no fingers or facial detail."

`docs/DECISIONS.md`, one dated entry after the build-19 bullets:

```markdown
- **The big toes are joints 15 and 16 — appended, one point per foot** (client, 2026-09-21: "does
  the program not detect toe position?"). Halpe-26 detects them already; the skeleton now keeps
  them so the rig's foot bone can aim. Small toes and heels stay out. Cost if wrong: two NaN
  columns on a take without feet.
- **The toes are extremities: no take-wide number or colour counts them.** Frame band, dial,
  figure height and the joint-count messages run over the original 15 (`CORE_JOINTS`); a cropped
  foot cannot turn a good take red or move the accuracy percentages. Cost if wrong: a wrong toe is
  visible only on its own handle.
- **Face-point ids have a fixed base (100), and old logs are migrated once.** They were
  `NUM_JOINTS + k`, which appending a joint would have silently re-read as body joints in every
  saved corrections.sqlite. The migration backs the file up beside itself first.
- **The floor stays ankle-based** for this cut; a foot pointing straight down can dip below the
  grid as it can today. The heel/toe floor is the next cut.
```

- [ ] **Step 3: Run the doc tests and commit**

Run: `PYTHONPATH=$PWD QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_diagnostics.py -q -p no:cacheprovider` → PASS.

```bash
git add README.md docs/client/RELEASE_NOTES_v1.md docs/DECISIONS.md tests/test_diagnostics.py
git commit -m "docs: the toe joints, the core-set rule and the face-id base, in the note and the decisions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Verification (controller, after the last merge)

```
PYTHONPATH=$PWD POSE3D_BLENDER=/home/athena/Downloads/blender-5.1.1-linux-x64/blender POSE3D_REQUIRE_BLENDER=1 POSE3D_REQUIRE_ASSETS=1 POSE3D_REQUIRE_FONTS=1 QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests -q -p no:cacheprovider
```

Then, offscreen through the real `MainWindow` on a COPY of `~/pose3d_projects/V1_050542` (a 15-row pre-toe project): it opens with the hint, 17 handles per view with the toes as placeholders under the ankles, `corrections.sqlite.pre-toes` beside a migrated log if the copy had face rows, Run Detection fills the toes (the take's feet are on a table, in frame), the 3D foot follows the toe, Export writes a 19-bone Y-up BVH whose FK passes the gate. Finally the Windows workflows for the next build.

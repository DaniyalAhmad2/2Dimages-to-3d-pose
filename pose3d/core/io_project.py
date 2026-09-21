"""Project persistence: a project is a folder on disk.

Layout:
    <project>/
        project.json        # metadata + per-frame poses (NaN-aware)
        images/             # (referenced by relative path; not copied here)
        calibration/        # K/dist/extrinsics (written by calib layer)
        corrections.sqlite  # append-only correction log (seeds v2 learning)

JSON stores NaN as null so the file stays valid JSON; load restores NaN.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path, PurePosixPath

import numpy as np

from pose3d.core.project import (
    CAMERAS, Correction, Frame, ProjectData,
)
from pose3d.core.skeleton import FACE_KP_BASE, NUM_HEAD_KP, NUM_JOINTS

PROJECT_JSON = "project.json"
CORRECTIONS_DB = "corrections.sqlite"
#: `PRAGMA user_version` of corrections.sqlite. 0 is every log written before
#: face ids had a fixed base (face rows carry joint = 15 + k); at
#: CORRECTIONS_SCHEMA they carry `skeleton.face_kp_id(k)`.
CORRECTIONS_SCHEMA = 2
#: NUM_JOINTS on the day the old convention was retired — a literal on
#: purpose: the migration reads OLD files, and NUM_JOINTS has since grown.
_LEGACY_NUM_JOINTS = 15
#: The untouched copy the migration leaves beside the log, once.
LEGACY_BACKUP = "corrections.sqlite.pre-toes"


def _arr_to_json(a: np.ndarray) -> list:
    """NaN-aware: np.nan -> None so the JSON is valid."""
    return [[None if np.isnan(v) else float(v) for v in row]
            for row in np.atleast_2d(a)]


def _json_to_arr(data: list, cols: int, rows: int = NUM_JOINTS) -> np.ndarray:
    out = np.full((rows, cols), np.nan, dtype=float)
    for i, row in enumerate(data or []):
        if i >= rows:            # tolerate files saved with more joints (e.g. feet)
            break
        for j, v in enumerate(row):
            out[i, j] = np.nan if v is None else float(v)
    return out


def _vec_to_json(a: np.ndarray) -> list:
    return [None if np.isnan(v) else float(v) for v in np.asarray(a).ravel()]


def _json_to_vec(data: list, rows: int = NUM_JOINTS) -> np.ndarray:
    out = np.full((rows,), np.nan, dtype=float)
    for i, v in enumerate(data or []):
        if i >= rows:
            break
        out[i] = np.nan if v is None else float(v)
    return out


def _json_to_flags(data, rows: int = NUM_JOINTS) -> np.ndarray:
    """Bool vector from JSON; a file written before the key existed gives all
    False, which is the truthful answer (nothing was flagged)."""
    out = np.zeros((rows,), dtype=bool)
    for i, v in enumerate(data or []):
        if i >= rows:
            break
        out[i] = bool(v)
    return out


def _store_image(path: str, folder: Path) -> str:
    """Path to write into project.json.

    Images living inside the project folder are stored RELATIVE to it, so the
    project can be moved, zipped, or mounted at a different path (e.g. inside a
    container) and still find them. Anything outside stays absolute.
    """
    if not path:
        return path
    try:
        return Path(path).resolve().relative_to(folder.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def _is_absolute(path: str) -> bool:
    """Absolute on the machine that WROTE it, not just on this one.

    Neither OS recognises the other's absolute paths: Windows sees no drive in
    '/data/a.jpg', POSIX sees no leading slash in 'D:\\data\\a.jpg'. Either way
    the path would be treated as relative, joined to the project folder, and
    silently resolved to nonsense — so recognise both forms on both platforms.
    """
    if Path(path).is_absolute() or path.startswith(("/", "\\")):
        return True
    return len(path) > 2 and path[1] == ":" and path[2] in "\\/"


def _resolve_image(path: str, folder: Path) -> str:
    """Path to hand back to the app when loading."""
    if not path:
        return path
    p = Path(path)
    if not _is_absolute(path):
        return str(folder / p)
    if p.exists():
        return str(p)
    # An absolute path written on another machine (or another OS): recover it
    # if the same filename is present in this project's images/ folder.
    cand = folder / "images" / PurePosixPath(path.replace("\\", "/")).name
    return str(cand) if cand.exists() else str(p)


def save_project(project: ProjectData, folder: str | Path) -> Path:
    """Write the project to <folder>, creating it if needed."""
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "calibration").mkdir(parents=True, exist_ok=True)

    doc = {
        "name": project.name,
        "fps": project.fps,
        "calibration_ref": project.calibration_ref,
        # what produced the stored 3D, so a later build knows whether the file
        # needs recomputing (pose3d.core.project.PIPELINE_VERSION)
        "pipeline_version": int(project.pipeline_version),
        "smoothing": project.smoothing,
        "marker_length": project.marker_length,
        "detector": project.detector,
        "keypoint_model": project.keypoint_model,
        # what the stored HEAD point is: see ProjectData.head_source
        "head_source": project.head_source,
        # which mode orients the head: see ProjectData.head_mode
        "head_mode": project.head_mode,
        "frames": [],
    }
    for f in project.frames:
        doc["frames"].append({
            "frame_id": f.frame_id,
            "images": {c: _store_image(p, folder) for c, p in f.images.items()},
            "kp2d": {c: _arr_to_json(f.kp2d[c]) for c in CAMERAS},
            "scores": {c: _vec_to_json(f.scores[c]) for c in CAMERAS},
            # the detector's own output, and the cross-view gate's verdict on
            # it. Both here so that reopening a project can rebuild exactly
            # what the app showed, and so no step downstream is ever holding
            # the only copy of what the cameras saw. Costs roughly a doubling
            # of project.json (200 KB -> ~400 KB on the client take) and buys
            # never destroying detector output again. NOTHING IN THE APP READS
            # kp2d_raw/scores_raw YET — they are the record, not a code path,
            # so whoever finds the file doubled in size and greps for a reader
            # is not looking at dead weight by mistake.
            "kp2d_raw": {c: _arr_to_json(f.kp2d_raw[c]) for c in CAMERAS},
            "scores_raw": {c: _vec_to_json(f.scores_raw[c]) for c in CAMERAS},
            "rejected": {c: [bool(v) for v in f.rejected[c]] for c in CAMERAS},
            "pose3d": _arr_to_json(f.pose3d),
            "fitted3d": _arr_to_json(f.fitted3d),
            "corrected": {c: [bool(v) for v in f.corrected[c]] for c in CAMERAS},
            "filled": [bool(v) for v in f.filled],
            "head2d": {c: _arr_to_json(f.head2d[c]) for c in CAMERAS},
            "head_scores": {c: _vec_to_json(f.head_scores[c]) for c in CAMERAS},
            # the face points' own hand-placed flags: a correction that is not
            # persisted is a correction the next session overwrites
            "head_corrected": {c: [bool(v) for v in f.head_corrected[c]]
                               for c in CAMERAS},
            "head3d": _arr_to_json(f.head3d),
        })

    (folder / PROJECT_JSON).write_text(json.dumps(doc, indent=2),
                                       encoding="utf-8")
    _write_corrections(folder, project.corrections)
    return folder


def load_project(folder: str | Path) -> ProjectData:
    folder = Path(folder)
    doc = json.loads((folder / PROJECT_JSON).read_text(encoding="utf-8"))

    frames: list[Frame] = []
    for fd in doc["frames"]:
        fr = Frame(frame_id=fd["frame_id"],
                   images={c: _resolve_image(p, folder)
                           for c, p in fd.get("images", {}).items()})
        # .get for the head keys: projects written before head keypoints
        # existed have none, and must still load.
        head2d, head_sc = fd.get("head2d") or {}, fd.get("head_scores") or {}
        head_corr = fd.get("head_corrected") or {}
        # A project written before the raw arrays existed has one copy of its
        # 2D and no record of which observations the gate rejected. Back-fill
        # the raw arrays FROM kp2d and leave the mask empty: that is the
        # truthful reading of such a file (nothing is flagged, and what is in
        # kp2d is all that survived), and the next recompute re-derives the
        # mask. It cannot recover what the old destructive gate already NaN'd
        # — nothing can, short of re-running the detector — but it can stop it
        # happening again.
        raw2d, raw_sc = fd.get("kp2d_raw") or {}, fd.get("scores_raw") or {}
        rejected = fd.get("rejected") or {}
        for c in CAMERAS:
            fr.kp2d[c] = _json_to_arr(fd["kp2d"][c], 2)
            fr.scores[c] = _json_to_vec(fd["scores"][c])
            fr.kp2d_raw[c] = (_json_to_arr(raw2d[c], 2) if raw2d.get(c)
                              else fr.kp2d[c].copy())
            fr.scores_raw[c] = (_json_to_vec(raw_sc[c]) if raw_sc.get(c)
                                else fr.scores[c].copy())
            fr.rejected[c] = _json_to_flags(rejected.get(c))
            fr.corrected[c] = np.array(fd["corrected"][c][:NUM_JOINTS], dtype=bool)
            fr.head2d[c] = _json_to_arr(head2d.get(c), 2, NUM_HEAD_KP)
            fr.head_scores[c] = _json_to_vec(head_sc.get(c), NUM_HEAD_KP)
            # absent before the key existed -> nothing was hand-placed, which
            # is what such a file holds
            fr.head_corrected[c] = _json_to_flags(head_corr.get(c), NUM_HEAD_KP)
        fr.filled = _json_to_flags(fd.get("filled"))
        fr.pose3d = _json_to_arr(fd["pose3d"], 3)
        fr.fitted3d = _json_to_arr(fd["fitted3d"], 3)
        fr.head3d = _json_to_arr(fd.get("head3d"), 3, NUM_HEAD_KP)
        frames.append(fr)

    project = ProjectData(
        name=doc["name"], fps=doc["fps"],
        calibration_ref=doc.get("calibration_ref"), frames=frames,
        smoothing=doc.get("smoothing") or "none",
        keypoint_model=doc.get("keypoint_model") or "coco17",
        # absent -> "nose": every project written before the key existed was
        # detected with COCO-17, whose HEAD is the nose
        head_source=doc.get("head_source") or "nose",
        # absent -> "nose": the mannequin-safe mode, which is how every take
        # written before the choice existed was posed
        head_mode=doc.get("head_mode") or "nose",
        # 0, not PIPELINE_VERSION: a file written before the key existed was
        # produced by the causal-EMA build and must be recomputed on open.
        pipeline_version=int(doc.get("pipeline_version", 0)),
        # absent in projects written before it was recorded
        marker_length=doc.get("marker_length"),
        detector=doc.get("detector"),
    )
    project.corrections = _read_corrections(folder)
    return project


# --- correction log (SQLite; append-only, seeds the v2 learning loop) -------

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


def _write_corrections(folder: Path, corrections: list[Correction]) -> int:
    """Store these corrections, keeping every row already in the log.

    APPEND-OR-REPLACE BY ROW, never a rewrite. This used to run
    `DELETE FROM corrections` first and re-insert whatever the caller held, so
    saving a project destroyed every correction made in an earlier session —
    on the button labelled "Save Corrections", silently, and the log the
    docstring above calls append-only was neither append-only nor cumulative.
    A correction carries the row it occupies (`Correction.id`, set when it was
    read back); one that has never been stored is appended and told its row, so
    saving twice stores it once.

    Rows this caller never loaded are left alone: `append_correction` and a
    second window on the same folder both write here, and their work is the
    client's too.

    Returns the number of corrections stored in the log afterwards — the TOTAL,
    which is what the user has to be told, not the size of this session's share
    of it.
    """
    with closing(_connect(folder)) as conn, conn:
        for c in corrections:
            row = (c.frame_id, c.cam, c.joint, c.old_xy[0], c.old_xy[1],
                   c.new_xy[0], c.new_xy[1], c.ts)
            if c.id is None:
                cur = conn.execute(
                    "INSERT INTO corrections "
                    "(frame_id,cam,joint,old_x,old_y,new_x,new_y,ts) "
                    "VALUES (?,?,?,?,?,?,?,?)", row)
                c.id = int(cur.lastrowid)
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO corrections "
                    "(id,frame_id,cam,joint,old_x,old_y,new_x,new_y,ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", (c.id, *row))
        return int(conn.execute(
            "SELECT COUNT(*) FROM corrections").fetchone()[0])


def count_corrections(folder: str | Path) -> int:
    """How many corrections this project's log holds, in total.

    The number the user is told after a save, and it is a question about the
    FILE rather than about the session: the log accumulates across sessions
    (and across a second window on the same folder), so counting what this
    session happens to be holding would under-report the client's own work.
    """
    folder = Path(folder)
    if not (folder / CORRECTIONS_DB).exists():
        return 0
    with closing(_connect(folder)) as conn:
        return int(conn.execute(
            "SELECT COUNT(*) FROM corrections").fetchone()[0])


def append_correction(folder: str | Path, c: Correction) -> None:
    """Append a single correction (used live during editing)."""
    with closing(_connect(Path(folder))) as conn, conn:
        conn.execute(
            "INSERT INTO corrections "
            "(frame_id,cam,joint,old_x,old_y,new_x,new_y,ts) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (c.frame_id, c.cam, c.joint, c.old_xy[0], c.old_xy[1],
             c.new_xy[0], c.new_xy[1], c.ts))


def _read_corrections(folder: Path) -> list[Correction]:
    if not (folder / CORRECTIONS_DB).exists():
        return []
    with closing(_connect(folder)) as conn:
        rows = conn.execute(
            "SELECT id,frame_id,cam,joint,old_x,old_y,new_x,new_y,ts "
            "FROM corrections ORDER BY id").fetchall()
    # the row id travels with the correction: it is what lets the next save
    # recognise a correction it has already stored (see _write_corrections).
    return [Correction(fr, cam, j, (ox, oy), (nx, ny), ts, id=int(i))
            for (i, fr, cam, j, ox, oy, nx, ny, ts) in rows]

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
import sqlite3
from pathlib import Path

import numpy as np

from pose3d.core.project import (
    CAMERAS, Correction, Frame, ProjectData,
)
from pose3d.core.skeleton import NUM_JOINTS

PROJECT_JSON = "project.json"
CORRECTIONS_DB = "corrections.sqlite"


def _arr_to_json(a: np.ndarray) -> list:
    """NaN-aware: np.nan -> None so the JSON is valid."""
    return [[None if np.isnan(v) else float(v) for v in row]
            for row in np.atleast_2d(a)]


def _json_to_arr(data: list, cols: int) -> np.ndarray:
    out = np.full((NUM_JOINTS, cols), np.nan, dtype=float)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            out[i, j] = np.nan if v is None else float(v)
    return out


def _vec_to_json(a: np.ndarray) -> list:
    return [None if np.isnan(v) else float(v) for v in np.asarray(a).ravel()]


def _json_to_vec(data: list) -> np.ndarray:
    out = np.full((NUM_JOINTS,), np.nan, dtype=float)
    for i, v in enumerate(data):
        out[i] = np.nan if v is None else float(v)
    return out


def save_project(project: ProjectData, folder: str | Path) -> Path:
    """Write the project to <folder>, creating it if needed."""
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "calibration").mkdir(parents=True, exist_ok=True)

    doc = {
        "name": project.name,
        "fps": project.fps,
        "calibration_ref": project.calibration_ref,
        "frames": [],
    }
    for f in project.frames:
        doc["frames"].append({
            "frame_id": f.frame_id,
            "images": f.images,
            "kp2d": {c: _arr_to_json(f.kp2d[c]) for c in CAMERAS},
            "scores": {c: _vec_to_json(f.scores[c]) for c in CAMERAS},
            "pose3d": _arr_to_json(f.pose3d),
            "fitted3d": _arr_to_json(f.fitted3d),
            "corrected": {c: [bool(v) for v in f.corrected[c]] for c in CAMERAS},
        })

    (folder / PROJECT_JSON).write_text(json.dumps(doc, indent=2))
    _write_corrections(folder, project.corrections)
    return folder


def load_project(folder: str | Path) -> ProjectData:
    folder = Path(folder)
    doc = json.loads((folder / PROJECT_JSON).read_text())

    frames: list[Frame] = []
    for fd in doc["frames"]:
        fr = Frame(frame_id=fd["frame_id"], images=fd.get("images", {}))
        for c in CAMERAS:
            fr.kp2d[c] = _json_to_arr(fd["kp2d"][c], 2)
            fr.scores[c] = _json_to_vec(fd["scores"][c])
            fr.corrected[c] = np.array(fd["corrected"][c], dtype=bool)
        fr.pose3d = _json_to_arr(fd["pose3d"], 3)
        fr.fitted3d = _json_to_arr(fd["fitted3d"], 3)
        frames.append(fr)

    project = ProjectData(
        name=doc["name"], fps=doc["fps"],
        calibration_ref=doc.get("calibration_ref"), frames=frames,
    )
    project.corrections = _read_corrections(folder)
    return project


# --- correction log (SQLite; append-only, seeds the v2 learning loop) -------

def _connect(folder: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(folder / CORRECTIONS_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS corrections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               frame_id TEXT, cam TEXT, joint INTEGER,
               old_x REAL, old_y REAL, new_x REAL, new_y REAL, ts TEXT)""")
    return conn


def _write_corrections(folder: Path, corrections: list[Correction]) -> None:
    conn = _connect(folder)
    with conn:
        conn.execute("DELETE FROM corrections")
        conn.executemany(
            "INSERT INTO corrections "
            "(frame_id,cam,joint,old_x,old_y,new_x,new_y,ts) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(c.frame_id, c.cam, c.joint, c.old_xy[0], c.old_xy[1],
              c.new_xy[0], c.new_xy[1], c.ts) for c in corrections])
    conn.close()


def append_correction(folder: str | Path, c: Correction) -> None:
    """Append a single correction (used live during editing)."""
    conn = _connect(Path(folder))
    with conn:
        conn.execute(
            "INSERT INTO corrections "
            "(frame_id,cam,joint,old_x,old_y,new_x,new_y,ts) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (c.frame_id, c.cam, c.joint, c.old_xy[0], c.old_xy[1],
             c.new_xy[0], c.new_xy[1], c.ts))
    conn.close()


def _read_corrections(folder: Path) -> list[Correction]:
    if not (folder / CORRECTIONS_DB).exists():
        return []
    conn = _connect(folder)
    rows = conn.execute(
        "SELECT frame_id,cam,joint,old_x,old_y,new_x,new_y,ts "
        "FROM corrections ORDER BY id").fetchall()
    conn.close()
    return [Correction(fr, cam, j, (ox, oy), (nx, ny), ts)
            for (fr, cam, j, ox, oy, nx, ny, ts) in rows]

#!/usr/bin/env python3
"""Rebuild tests/fixtures/client_take/ from the real client take.

The take itself lives at `workspace/pose3d_projects/Imported_Session`, which is
gitignored (it is the client's data, and it carries 52 phone photographs). The
committed fixture is therefore the only copy the test suite can see, and this
script is the record of how it was made:

    .venv/bin/python tests/fixtures/regen_client_take.py [SOURCE_PROJECT]

It writes

    client_take/project.json          the 26 frames' kp2d / scores / corrected
                                      / pose3d / fitted3d, no images
    client_take/calibration/*.json    the shipped calibration, byte for byte
    client_take/aruco_corners.json    the DICT_6X6_250 tags detected in each
                                      real image, so calibration work needs
                                      neither the photographs nor a detector

Nothing here runs at test time: the ArUco detection happens once, now.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from pose3d.calib.extrinsics import detect_markers, make_detector  # noqa: E402
from pose3d.core.project import CAMERAS                            # noqa: E402

DEFAULT_SOURCE = REPO / "workspace" / "pose3d_projects" / "Imported_Session"
FIXTURE = Path(__file__).resolve().parent / "client_take"

# The tags taped to the client's sweep. 6x6 because that is what they printed;
# the app still defaults to 4x4, which is why this has to be recorded.
ARUCO_DICT = "DICT_6X6_250"

# kp2d is in pixels on a 3072x4080 sensor and scores are in [0, 1], so this
# much precision is far below anything measurable. pose3d/fitted3d are metres
# on a 0.118 m subject and are kept at full precision — 1e-3 there would be
# 1 % of the whole body.
KP_DECIMALS = 3
SCORE_DECIMALS = 4


def _round(v, nd):
    return None if v is None else round(float(v), nd)


def trim_project(doc: dict) -> dict:
    """The project the tests need: poses and flags, no image data."""
    out = {"name": doc["name"], "fps": doc["fps"],
           "calibration_ref": doc.get("calibration_ref"), "frames": []}
    for f in doc["frames"]:
        out["frames"].append({
            "frame_id": f["frame_id"],
            # kept as relative placeholders: they say which photographs the
            # fixture came from, and nothing in the tests opens them
            "images": dict(f.get("images", {})),
            "kp2d": {c: [[_round(v, KP_DECIMALS) for v in row]
                         for row in f["kp2d"][c]] for c in CAMERAS},
            "scores": {c: [_round(v, SCORE_DECIMALS) for v in f["scores"][c]]
                       for c in CAMERAS},
            "pose3d": f["pose3d"],
            "fitted3d": f["fitted3d"],
            "corrected": {c: [bool(v) for v in f["corrected"][c]]
                          for c in CAMERAS},
        })
    return out


def detect_aruco(source: Path, doc: dict) -> dict:
    """Every DICT_6X6_250 tag in every image, as 4x2 corner pixels."""
    detector = make_detector(getattr(cv2.aruco, ARUCO_DICT))
    frames = {}
    for f in doc["frames"]:
        per_cam = {}
        for cam in CAMERAS:
            rel = f.get("images", {}).get(cam)
            if not rel:
                continue
            path = source / rel
            img = cv2.imread(str(path))
            if img is None:
                raise SystemExit(f"cannot read {path}")
            corners, ids = detect_markers(img, detector)
            per_cam[cam] = {
                str(int(i)): [[round(float(x), 2), round(float(y), 2)]
                              for x, y in c.reshape(4, 2)]
                for c, i in zip(corners, ids)}
        frames[f["frame_id"]] = per_cam
    return {"dictionary": ARUCO_DICT,
            "corner_order": "TL, TR, BR, BL (cv2.detectMarkers)",
            "frames": frames}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default=str(DEFAULT_SOURCE),
                    help=f"project folder to copy (default: {DEFAULT_SOURCE})")
    args = ap.parse_args(argv)

    source = Path(args.source)
    doc = json.loads((source / "project.json").read_text())

    FIXTURE.mkdir(parents=True, exist_ok=True)
    (FIXTURE / "calibration").mkdir(exist_ok=True)
    for name in ("left_intrinsics.json", "right_intrinsics.json",
                 "extrinsics.json"):
        shutil.copyfile(source / "calibration" / name,
                        FIXTURE / "calibration" / name)

    def write(path: Path, obj):
        path.write_text(json.dumps(obj, separators=(",", ":")))
        print(f"  {path.relative_to(REPO)}  {path.stat().st_size / 1024:.0f} KB")

    print(f"regenerating {FIXTURE.relative_to(REPO)} from {source}")
    write(FIXTURE / "project.json", trim_project(doc))
    write(FIXTURE / "aruco_corners.json", detect_aruco(source, doc))
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    print(f"  total {total / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

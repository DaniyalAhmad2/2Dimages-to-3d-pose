#!/usr/bin/env python3
"""Rebuild tests/fixtures/client_take/ from the real client take.

The take itself lives at `workspace/pose3d_projects/Imported_Session`, which is
gitignored (it is the client's data, and it carries 52 phone photographs). The
committed fixture is therefore the only copy the test suite can see, and this
script is the record of how it was made:

    .venv/bin/python tests/fixtures/regen_client_take.py [SOURCE_PROJECT]

It writes

    client_take/project.json          the 26 frames' kp2d / scores / corrected
                                      / head2d / head_scores / pose3d /
                                      head3d / fitted3d / filled, no images
    client_take/calibration/*.json    the shipped calibration, byte for byte
    client_take/aruco_corners.json    the DICT_6X6_250 tags detected in each
                                      real image, so calibration work needs
                                      neither the photographs nor a detector
    client_take/as_delivered.json     the SOURCE project's own `fitted3d` —
                                      the pose the client was actually sent by
                                      the August build. Copied from the source
                                      untouched whether or not --redetect ran,
                                      because a re-detected project.json holds
                                      today's pose and the lag defect would
                                      otherwise stop being pinned by anything

Nothing here runs at test time: the ArUco detection happens once, now.

With `--redetect` the take is re-detected and re-reconstructed with the
CURRENT detector and pipeline before being trimmed. That is how the fixture is
re-baselined when either changes, and it is not optional when the DETECTOR
changes: switching to the Halpe-26 layout (detect.rtmpose.USE_HALPE26) moves
the reconstructed body height, and every "% of body height" threshold in
tests/test_client_regression.py with it. The source project is only read; the
new poses exist in memory and land in the fixture.
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
    # what the stored 2D IS, so the fixture cannot be measured under the wrong
    # convention: "halpe26"/"skull" says HEAD is the skull vertex rather than
    # the nose, which is what the retarget branches on
    for key in ("keypoint_model", "head_source", "pipeline_version",
                "smoothing", "detector"):
        if doc.get(key) is not None:
            out[key] = doc[key]
    for f in doc["frames"]:
        head2d = f.get("head2d") or {}
        head_scores = f.get("head_scores") or {}
        trimmed = {
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
        }
        # the face keypoints orient the head; without them the fixture cannot
        # measure the path the app actually runs
        if head2d:
            trimmed["head2d"] = {
                c: [[_round(v, KP_DECIMALS) for v in row]
                    for row in head2d.get(c, [])] for c in CAMERAS}
            trimmed["head_scores"] = {
                c: [_round(v, SCORE_DECIMALS)
                    for v in head_scores.get(c, [])] for c in CAMERAS}
        if f.get("head3d") is not None:
            trimmed["head3d"] = f["head3d"]
        if f.get("filled") is not None:
            trimmed["filled"] = [bool(v) for v in f["filled"]]
        out["frames"].append(trimmed)
    return out


def as_delivered(doc: dict) -> dict:
    """The pose the client was SENT, lifted out of the source project.

    `project.json` in the fixture holds whatever this build reconstructs (see
    `redetect`), so once the fixture has been regenerated on a fixed build,
    nothing in it remembers the smoothed, lagging pose that was delivered in
    August. This file does, and it is what
    `test_the_fixture_still_carries_the_pose_the_client_was_sent` measures
    against — the defect stays pinned across every re-baseline.
    """
    return {
        "note": "fitted3d exactly as the source project.json carries it: the "
                "pose the client received, NOT what this build computes",
        "source_pipeline_version": doc.get("pipeline_version"),
        "source_head_source": doc.get("head_source", "nose"),
        "frames": {f["frame_id"]: f["fitted3d"] for f in doc["frames"]},
    }


def redetect(source: Path, doc: dict) -> dict:
    """Re-run detection and reconstruction on the take, in memory.

    Everything the shipped import does — the current detector, the current
    pipeline — with the source project only READ: what comes back is a
    project.json document, not a folder. Slow (CPU-only ONNX over 52
    photographs), which is why it is opt-in.
    """
    import tempfile

    from pose3d.core.io_project import load_project, save_project
    from pose3d.core.project import PIPELINE_VERSION
    from pose3d.detect.rtmpose import RTMPoseDetector
    from pose3d.pipeline import run_full
    from pose3d.quality import load_rig

    project = load_project(source)
    # no feet=: which pose model the app runs is RTMPoseDetector's own default
    # (detect.rtmpose.USE_HALPE26). Pinning it here would re-baseline the
    # regression net onto a layout the app does not detect with.
    det = RTMPoseDetector(mode="balanced", device="cpu")
    project.detector = f"rtmpose-{det.mode}" + ("-feet" if det.feet else "")
    project.head_source = det.head_source
    print(f"  detecting {len(project.frames)} frames x {len(CAMERAS)} cameras "
          f"with {project.detector} ({project.head_source} HEAD)…", flush=True)
    report = run_full(project, det, load_rig(source / "calibration"),
                      lambda p: cv2.imread(str(p)))
    print(f"  {report.note() or 'fit clean'}")
    # these poses ARE the current pipeline's, whatever the source file said:
    # leaving the loaded 0 behind would have the app recompute a fixture it
    # had just computed, and would call the fresh poses the old build's.
    project.pipeline_version = PIPELINE_VERSION

    with tempfile.TemporaryDirectory() as tmp:
        new_doc = json.loads((save_project(project, Path(tmp))
                              / "project.json").read_text())
    # save_project stored absolute image paths (the images live in the source,
    # not in the temp folder); the fixture wants the source's own relative
    # placeholders back
    images = {f["frame_id"]: f.get("images", {}) for f in doc["frames"]}
    for f in new_doc["frames"]:
        f["images"] = dict(images.get(f["frame_id"], {}))
    return new_doc


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
    ap.add_argument("--redetect", action="store_true",
                    help="re-detect and re-reconstruct with the current "
                         "detector and pipeline first (slow; re-baselines "
                         "tests/test_client_regression.py)")
    args = ap.parse_args(argv)

    source = Path(args.source)
    doc = json.loads((source / "project.json").read_text())
    delivered = as_delivered(doc)            # BEFORE any re-detection
    if args.redetect:
        doc = redetect(source, doc)

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
    write(FIXTURE / "as_delivered.json", delivered)
    write(FIXTURE / "aruco_corners.json", detect_aruco(source, doc))
    total = sum(p.stat().st_size for p in FIXTURE.rglob("*") if p.is_file())
    print(f"  total {total / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure the Phase 5 gate table: is the native skull HEAD worth switching on?

`detect.rtmpose.USE_HALPE26` decides which pose model the app runs. Halpe-26
carries a head keypoint on the skull axis where COCO-17 has only the nose, and
the nose is the worst thing in the baseline. Whether that trade is taken is not
a judgement call: it was fixed in advance as the gate table below, and this
script is what measures it, so the number that decided the switch can be
re-derived rather than believed.

    tools/measure_head_gates.py --project <take> --cache <dir> [--out gates.json]

Both pose models are run over the take through the IDENTICAL current pipeline
(detect -> triangulate -> fit), and every metric comes from `pose3d.quality`,
which is the audit harness ported verbatim — so the numbers stay comparable
with docs/audit-2026-09/wf_baseline.md. Detection is CPU-only ONNX and takes
minutes, so the raw 2D/3D of each run is cached in `--cache` as an .npz and
reused; delete the .npz (or pass --redetect) to measure again from the images.
The project folder is only ever READ.

Cross-checks the harness itself, too: `--variants` also measures the naive
whole-model swap (native NECK/PELVIS) and the head read-back point (head-bone
mid vs tail), which are the two choices the gate table does not cover.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# The gate table, fixed BEFORE the run (PLAN.md Phase 5), as
# {metric key: the rule it must satisfy}. The COCO-17 "today" column is
# re-measured here rather than assumed.
GATES: dict[str, str] = {
    "head_retarget_no_face_pct": "<= 4.0 % of body height",
    "head_retarget_with_face_pct": "<= 3.0 % of body height",
    "neck_head_bone_cv_pct": "<= 4.0 %",
    "neck_lshoulder_bone_cv_pct": "<= 5.15 % (the COCO-17 baseline)",
    "worst_body_joint_regression_pct": "<= 0.5 % of body height",
    "head_aim_error_deg": "< 5 deg with the nose path disabled",
    "epipolar_median_px": "no regression on the COCO-17 baseline",
}

# HEAD is the change itself, so it cannot also be a no-regression gate. NECK
# and PELVIS are NOT excluded even though the plan's wording ("any directly-
# mapped body joint") would allow it: both are the same 2D midpoint under both
# layouts, so they are comparable, and holding them to the gate is strictly
# harder than what was asked for.
_POLICY_JOINTS = ("HEAD",)


def _detect(project_dir: Path, feet: bool, npz: Path,
            policy: dict[str, str] | None = None) -> None:
    """Run one pose model over the take and cache what the pipeline produced.

    `policy` overrides `skeleton.map_halpe26`'s HEAD/NECK/PELVIS choice, which
    is how the naive whole-model swap (everything native) is measured without
    shipping it.
    """
    import cv2

    from pose3d.core.io_project import load_project
    from pose3d.detect.rtmpose import RTMPoseDetector
    from pose3d.pipeline import run_full
    from pose3d.quality import load_rig

    project = load_project(project_dir)
    det = RTMPoseDetector(mode="balanced", device="cpu", feet=feet)
    if policy is not None:
        from functools import partial

        from pose3d.core.skeleton import map_halpe26
        det._map = partial(map_halpe26, **policy)
    project.detector = f"rtmpose-{det.mode}" + ("-feet" if det.feet else "")
    project.head_source = det.head_source
    t0 = time.time()
    report = run_full(project, det, load_rig(project_dir / "calibration"),
                      lambda p: cv2.imread(str(p)))
    print(f"  {len(project.frames)} frames in {time.time() - t0:.0f} s  "
          f"layout={project.keypoint_model} head={project.head_source}  "
          f"{report.note() or 'fit clean'}", flush=True)

    stack = lambda attr, cam=None: np.stack(
        [getattr(f, attr) if cam is None else getattr(f, attr)[cam]
         for f in project.frames])
    from pose3d.core.project import CAMERAS
    out = {a: stack(a) for a in ("pose3d", "fitted3d", "head3d", "filled")}
    for cam in CAMERAS:
        for attr in ("kp2d", "scores", "head2d", "head_scores"):
            out[f"{attr}__{cam}"] = stack(attr, cam)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **out)


def _project_from(npz: Path, head_source: str):
    """Rebuild a ProjectData from a cached run (no images, no detector)."""
    from pose3d.core.project import CAMERAS, Frame, ProjectData

    d = np.load(npz)
    p = ProjectData(
        name=npz.stem, head_source=head_source,
        keypoint_model="halpe26" if head_source == "skull" else "coco17")
    for t in range(d["pose3d"].shape[0]):
        f = Frame(frame_id=f"{t:04d}")
        for cam in CAMERAS:
            f.kp2d[cam] = d[f"kp2d__{cam}"][t]
            f.scores[cam] = d[f"scores__{cam}"][t]
            f.head2d[cam] = d[f"head2d__{cam}"][t]
            f.head_scores[cam] = d[f"head_scores__{cam}"][t]
        f.pose3d = d["pose3d"][t]
        f.fitted3d = d["fitted3d"][t]
        f.head3d = d["head3d"][t]
        f.filled = d["filled"][t]
        p.frames.append(f)
    return p


def _head_aim_error(character, up, head3d=None) -> dict:
    """Angle between the captured NECK->HEAD direction and the posed rig's.

    This is the gate "head aim error with the nose path disabled": it asks
    whether the character's head points where the capture says, which is what
    the 45 deg nose correction exists to fake and a skull HEAD makes real.
    """
    from pose3d import quality as Q
    from pose3d.core.skeleton import Joint

    out = []
    for t, pose in enumerate(up):
        valid = ~np.isnan(pose).any(1)
        if not (valid[int(Joint.HEAD)] and valid[int(Joint.NECK)]):
            continue
        hp = None
        if head3d is not None:
            h = np.asarray(head3d[t], float)
            hp = None if np.isnan(h).all() else h
        J = character.posed_joints(pose, valid, hp)
        if J is None or np.isnan(J[[int(Joint.HEAD), int(Joint.NECK)]]).any():
            continue
        out.append(Q._angle_deg(pose[int(Joint.HEAD)] - pose[int(Joint.NECK)],
                                J[int(Joint.HEAD)] - J[int(Joint.NECK)]))
    out = [v for v in out if np.isfinite(v)]
    return {"n": len(out),
            "median_deg": float(np.median(out)) if out else float("nan"),
            "max_deg": float(np.max(out)) if out else float("nan")}


def face_to_mesh(npz: Path, project_dir: Path, head_source: str,
                 readback: str) -> dict:
    """How far the reconstructed NOSE and EAR land from the posed mesh.

    The head read-back point may NOT be chosen on the head-position metric
    alone: F03's refuter improved exactly that metric while tripling this one
    (1.56 -> 4.34 mm), which is the visible error — a head aimed or read back
    wrongly pushes the face away from where the capture says it is. So the
    ruling is taken on both.
    """
    from pose3d import quality as Q
    from pose3d.geometry import character as chmod
    from pose3d.geometry.character import Character

    p = _project_from(npz, head_source)
    delivered = np.stack([f.fitted3d for f in p.frames])
    height = Q.subject_height(delivered)
    R = Q.de_tilt_rotation(delivered)
    up = delivered @ R.T
    head3d = np.stack([f.head3d for f in p.frames]) @ R.T

    saved = dict(chmod._HEAD_FROM_RIG)
    chmod._HEAD_FROM_RIG[head_source] = (("head", readback),)
    try:
        ch = Character(head_source=head_source)
        ch.fit_to_subject(up)
        nose, ear = [], []
        for t, pose in enumerate(up):
            h = head3d[t]
            hp = None if np.isnan(h).all() else h
            verts, _, _ = ch.pose_and_joints(pose, ~np.isnan(pose).any(1), hp)
            if verts is None:
                continue
            for k, acc in ((0, nose), (3, ear)):    # skeleton.HEAD_KP: nose, L ear
                if not np.isnan(h[k]).any():
                    acc.append(float(np.linalg.norm(verts - h[k], axis=1).min()))
    finally:
        chmod._HEAD_FROM_RIG.clear()
        chmod._HEAD_FROM_RIG.update(saved)

    stat = lambda a: {"median_mm": float(np.median(a)) * 1000,
                      "max_mm": float(np.max(a)) * 1000, "n": len(a)}
    return {"readback": readback, "height_m": float(height),
            "nose_to_mesh": stat(nose), "ear_to_mesh": stat(ear)}


def measure(label: str, npz: Path, project_dir: Path, head_source: str,
            readback: str | None = None, neck: str | None = None) -> dict:
    """Every number the gate table needs, for one (model, policy) run."""
    from pose3d import quality as Q
    from pose3d.core.project import CAMERAS
    from pose3d.geometry import character as chmod
    from pose3d.geometry.character import Character

    p = _project_from(npz, head_source)
    rig = Q.load_rig(project_dir / "calibration")
    delivered = np.stack([f.fitted3d for f in p.frames])
    measured = np.stack([f.pose3d for f in p.frames])
    kp2d = {c: np.stack([f.kp2d[c] for f in p.frames]) for c in CAMERAS}

    height = Q.subject_height(delivered)
    R = Q.de_tilt_rotation(delivered)
    up = delivered @ R.T
    head3d = np.stack([f.head3d for f in p.frames]) @ R.T

    saved = dict(chmod._HEAD_FROM_RIG)
    if readback is not None:                    # a read-back variant, not the shipped one
        chmod._HEAD_FROM_RIG[head_source] = (("head", readback),)
    try:
        ch = Character(head_source=head_source)
        scale = ch.fit_to_subject(up)
        r_no = Q.retarget_error(ch, up, height, scale, None)
        r_face = Q.retarget_error(ch, up, height, scale, head3d)
        aim_no = _head_aim_error(ch, up, None)
        aim_face = _head_aim_error(ch, up, head3d)
        used_readback = chmod._HEAD_FROM_RIG[head_source][0][1]
    finally:
        chmod._HEAD_FROM_RIG.clear()
        chmod._HEAD_FROM_RIG.update(saved)

    bl = Q.bone_length_stats(measured)
    epi = Q.body_epipolar(kp2d, rig)
    return {
        "label": label, "head_source": head_source, "readback": used_readback,
        "n_frames": len(p.frames),
        "neck_policy": neck or "derived",
        "height_m": float(height), "scale": float(scale),
        "retarget_no_face_pct": {k: v["median_pct_height"]
                                 for k, v in r_no["per_joint"].items()},
        "retarget_with_face_pct": {k: v["median_pct_height"]
                                   for k, v in r_face["per_joint"].items()},
        "retarget_median_pct": {"no_face": r_no["median_pct_height"],
                                "with_face": r_face["median_pct_height"]},
        "head_aim_no_face": aim_no, "head_aim_with_face": aim_face,
        "bone_cv_pct": {bl["bones"][k]["name"]: bl["bones"][k]["cv_pct"]
                        for k in bl["bones"]},
        "bone_cv_median_pct": bl["median_cv_pct"],
        "bone_cv_max_pct": bl["max_cv_pct"],
        "epipolar_median_px": epi["median_px"],
        "epipolar_p90_px": epi["p90_px"],
        "gaps": Q.gap_stats(p),
    }


def gate_table(base: dict, cand: dict) -> list[dict]:
    """Score the candidate against the COCO-17 baseline, gate by gate."""
    worst_joint, worst_delta = None, 0.0
    for name, after in cand["retarget_no_face_pct"].items():
        if name in _POLICY_JOINTS:
            continue                     # the policy joints are the change itself
        delta = after - base["retarget_no_face_pct"][name]
        if delta > worst_delta:
            worst_joint, worst_delta = name, delta

    rows = [
        ("HEAD retarget, no face points", "head_retarget_no_face_pct",
         base["retarget_no_face_pct"]["HEAD"],
         cand["retarget_no_face_pct"]["HEAD"],
         cand["retarget_no_face_pct"]["HEAD"] <= 4.0, "%"),
        ("HEAD retarget, with face points", "head_retarget_with_face_pct",
         base["retarget_with_face_pct"]["HEAD"],
         cand["retarget_with_face_pct"]["HEAD"],
         cand["retarget_with_face_pct"]["HEAD"] <= 3.0, "%"),
        ("neck-head bone CV", "neck_head_bone_cv_pct",
         base["bone_cv_pct"]["neck-head"], cand["bone_cv_pct"]["neck-head"],
         cand["bone_cv_pct"]["neck-head"] <= 4.0, "%"),
        ("neck-Lshoulder bone CV", "neck_lshoulder_bone_cv_pct",
         base["bone_cv_pct"]["neck-Lshoulder"],
         cand["bone_cv_pct"]["neck-Lshoulder"],
         cand["bone_cv_pct"]["neck-Lshoulder"] <= 5.15, "%"),
        (f"worst body-joint regression ({worst_joint or 'none'})",
         "worst_body_joint_regression_pct", 0.0, worst_delta,
         worst_delta <= 0.5, "% of height"),
        ("head aim error, nose path off", "head_aim_error_deg",
         base["head_aim_no_face"]["median_deg"],
         cand["head_aim_no_face"]["median_deg"],
         cand["head_aim_no_face"]["median_deg"] < 5.0, "deg"),
        ("body epipolar median", "epipolar_median_px",
         base["epipolar_median_px"], cand["epipolar_median_px"],
         cand["epipolar_median_px"] <= base["epipolar_median_px"], "px"),
    ]
    return [{"gate": name, "key": key, "rule": GATES[key],
             "baseline": float(b), "measured": float(m), "pass": bool(ok),
             "unit": unit}
            for name, key, b, m, ok, unit in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, type=Path,
                    help="the take to measure (read only)")
    ap.add_argument("--cache", required=True, type=Path,
                    help="where the per-model detections are cached (.npz)")
    ap.add_argument("--out", type=Path, help="write the full numbers here")
    ap.add_argument("--redetect", action="store_true",
                    help="re-run detection even if the cache has it")
    ap.add_argument("--variants", action="store_true",
                    help="also measure the read-back point and native NECK")
    args = ap.parse_args(argv)

    runs = {"coco": False, "halpe": True}
    for name, feet in runs.items():
        npz = args.cache / f"{name}.npz"
        if args.redetect or not npz.exists():
            print(f"detecting {name} (CPU ONNX, minutes)…", flush=True)
            _detect(args.project, feet, npz)
        else:
            print(f"reusing {npz}")

    base = measure("COCO-17 (today)", args.cache / "coco.npz",
                   args.project, "nose")
    cand = measure("Halpe-26, skull HEAD, derived NECK/PELVIS",
                   args.cache / "halpe.npz", args.project, "skull")

    rows = gate_table(base, cand)
    width = max(len(r["gate"]) for r in rows)
    print(f"\nbody height {base['height_m']*1000:.2f} -> "
          f"{cand['height_m']*1000:.2f} mm   "
          f"scale {base['scale']:.2f} -> {cand['scale']:.2f}\n")
    for r in rows:
        print(f"  {r['gate']:<{width}}  {r['baseline']:7.2f} -> "
              f"{r['measured']:7.2f} {r['unit']:<12} {r['rule']:<34} "
              f"{'PASS' if r['pass'] else 'FAIL'}")
    passed = sum(r["pass"] for r in rows)
    print(f"\n{passed}/{len(rows)} gates pass — the switch "
          f"({'SHIPS' if passed == len(rows) else 'STAYS OFF'}: "
          "detect.rtmpose.USE_HALPE26)")

    out = {
        "generated_by": "tools/measure_head_gates.py",
        "reproduce": ("tools/measure_head_gates.py --project <the client take> "
                      "--cache <dir> --variants --out <this file>"),
        "take": args.project.name, "frames": base["n_frames"],
        "switch": {"constant": "pose3d.detect.rtmpose.USE_HALPE26",
                   "ships_on": passed == len(rows)},
        "gates": rows, "passed": passed, "of": len(rows),
        "runs": {"coco": base, "halpe": cand},
    }

    if args.variants:
        # The two choices the gate table does not cover, measured on the same
        # cached detections: the head read-back point, and the naive swap.
        for readback in ("mid", "tail"):
            v = measure(f"skull HEAD, read-back {readback}",
                        args.cache / "halpe.npz", args.project, "skull",
                        readback=readback)
            out.setdefault("readback", {})[readback] = v
            print(f"  read-back {readback:4s}: HEAD retarget "
                  f"{v['retarget_no_face_pct']['HEAD']:.2f} % (no face) / "
                  f"{v['retarget_with_face_pct']['HEAD']:.2f} % (with), "
                  f"aim {v['head_aim_no_face']['median_deg']:.2f} deg median / "
                  f"{v['head_aim_no_face']['max_deg']:.2f} max, "
                  f"whole body {v['retarget_median_pct']['no_face']:.2f} %")
        for readback in ("mid", "tail"):
            f = face_to_mesh(args.cache / "halpe.npz", args.project,
                             "skull", readback)
            out.setdefault("face_to_mesh", {})[readback] = f
            print(f"  read-back {readback:4s}: reconstructed nose -> mesh "
                  f"{f['nose_to_mesh']['median_mm']:.2f} mm median, L ear "
                  f"{f['ear_to_mesh']['median_mm']:.2f} mm "
                  "(F03's refuter metric: must not grow)")
        # the naive whole-model swap: everything native, which is what the
        # policy dict exists to avoid. The audit measured neck-Lshoulder CV
        # 8.13 % and neck-head 3.85 % for it; reproducing those is what says
        # this harness and the audit's are measuring the same thing.
        native = args.cache / "halpe_native.npz"
        if args.redetect or not native.exists():
            print("detecting the naive whole-model swap (native "
                  "HEAD/NECK/PELVIS)…", flush=True)
            _detect(args.project, True, native,
                    policy={"head": "native", "neck": "native",
                            "pelvis": "native"})
        v = measure("Halpe-26, everything native (the naive swap)",
                    native, args.project, "skull")
        out["naive_swap"] = v
        print(f"  naive swap (native NECK/PELVIS): neck-Lshoulder CV "
              f"{v['bone_cv_pct']['neck-Lshoulder']:.2f} %, neck-head "
              f"{v['bone_cv_pct']['neck-head']:.2f} % "
              "(audit: 8.13 / 3.85 — why NECK stays derived)")

        v = measure("Halpe-26 with the nose path left live",
                    args.cache / "halpe.npz", args.project, "nose")
        out["nose_path_left_live"] = v
        print(f"  nose path left live on skull data: HEAD retarget "
              f"{v['retarget_no_face_pct']['HEAD']:.2f} %, aim "
              f"{v['head_aim_no_face']['median_deg']:.2f} deg "
              "(this is the double correction head_source prevents)")

    if args.out:
        args.out.write_text(json.dumps(out, indent=1, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

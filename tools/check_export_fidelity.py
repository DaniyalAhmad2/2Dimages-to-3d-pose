#!/usr/bin/env python3
"""Does the delivered BVH contain the pose the app shows? A dev script.

The audit could not answer that: Blender was never run against the client take,
and the export's only tests were "exit code 0, file not empty". This runs the
real export, reads the file back, evaluates forward kinematics in numpy and
reports five things:

  (a) deviation from `Character.posed_joints` at each captured keyframe, as a
      % of body height, after ONE global similarity fit (the file is in the
      .blend armature's space, the app's joints are in metres);
  (b) per in-between frame: how far each bone leaves the slerp arc between its
      bracketing keys, how far it overshoots past either of them, and how far
      each Euler channel overshoots the interval its two keys span;
  (c) consecutive keyframe quaternion pairs with dot < 0 (hemisphere flips);
  (d) the largest non-hips translation channel, as a fraction of rig height;
  (e) what the fallback path ships when the character asset is missing.

    POSE3D_BLENDER=/path/to/blender \\
        .venv/bin/python tools/check_export_fidelity.py

Not part of CI: it needs Blender and takes minutes. The facts it measured on
2026-09-01 are asserted every run, without Blender, in tests/test_export_smoke.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pose3d.config import blender_binary, character_blend          # noqa: E402
from pose3d.core.io_project import load_project                    # noqa: E402
from pose3d.core.skeleton import JOINT_NAMES                       # noqa: E402
from pose3d.export.blender_export import export_animation          # noqa: E402
from pose3d.geometry.character import Character                    # noqa: E402
from pose3d.quality import de_tilt_rotation, subject_height        # noqa: E402
from tests import bvh_util                                         # noqa: E402

DEFAULT_PROJECT = REPO / "tests" / "fixtures" / "client_take"
# Character().rig_h for the bundled rig; the audit's denominator for (d).
RIG_HEIGHT = 14.4228


def similarity(src: np.ndarray, dst: np.ndarray):
    """Umeyama: the single (scale, R, t) that best maps src onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(a.T @ b / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = (U @ D @ Vt).T
    scale = float(S @ np.diag([1, 1, d]).diagonal() / (a ** 2).sum() * len(src))
    return scale, R, dst.mean(0) - scale * R @ mu_s


def report_keyframe_fidelity(bvh, character, up, valid, n_poses) -> None:
    """(a) the FK skeleton in the file vs the joints the app draws."""
    # Only the canonical joints the rig reports at a bone HEAD: BVH forward
    # kinematics gives head positions, and the one joint read off a bone's
    # midpoint (HEAD) has no counterpart there.
    mapping, skipped = {}, []
    for j, (bone, which) in character._joint_src.items():
        name = character.bone_names[bone]
        if which == "head" and name in [b.name for b in bvh.joints]:
            mapping[j] = bvh.index(name)
        else:
            skipped.append(JOINT_NAMES[j])

    holds = bvh_util.stepped_holds(n_poses)
    src, dst, index, per_frame = [], [], [], []
    for k, (row, _) in enumerate(holds):
        fk = bvh.forward_kinematics(row)
        app = character.posed_joints(up[k], valid[k], None)
        if app is None:
            continue
        a, b = [], []
        for j, bi in mapping.items():
            if valid[k][j] and np.isfinite(app[j]).all():
                a.append(fk[bi])
                b.append(app[j])
                index.append((k, j))
        src.extend(a)
        dst.extend(b)
        # the same fit per keyframe: this is pure SHAPE, with the frame's own
        # placement allowed to differ
        a, b = np.asarray(a), np.asarray(b)
        s, R, t = similarity(a, b)
        per_frame.append(np.linalg.norm((s * (R @ a.T).T + t) - b, axis=1))
    src, dst = np.asarray(src), np.asarray(dst)
    scale, R, t = similarity(src, dst)
    err = np.linalg.norm((scale * (R @ src.T).T + t) - dst, axis=1)
    per_frame = np.concatenate(per_frame)
    height = float(np.median([p[v, 2].max() - p[v, 2].min()
                              for p, v in zip(up, valid) if v.sum() >= 2]))

    print(f"(a) BVH forward kinematics vs Character.posed_joints, "
          f"{len(mapping)} joints x {n_poses} keyframes, "
          f"body height {height:.4f} m")
    print(f"    ONE global similarity fit (scale {scale:.6f}): median "
          f"{100 * np.median(err) / height:.3f} % of height, p90 "
          f"{100 * np.percentile(err, 90) / height:.3f} %, max "
          f"{100 * err.max() / height:.3f} %")
    worst = int(np.argmax(err))
    print(f"    worst: pose {index[worst][0]} {JOINT_NAMES[index[worst][1]]}")
    print(f"    a similarity fit PER KEYFRAME instead: median "
          f"{100 * np.median(per_frame) / height:.6f} % of height, max "
          f"{100 * per_frame.max() / height:.6f} %")
    print("    -> the gap between those two lines is placement, not pose: the "
          "file holds the shape the view shows and drops where it stood.")
    if skipped:
        print(f"    not comparable (not a bone head in the file): "
              f"{', '.join(skipped)}")


def report_interpolation(bvh, n_poses) -> None:
    """(b) and (c): what the in-betweens do, and hemisphere flips."""
    holds = bvh_util.stepped_holds(n_poses)
    arc_dev, endpoint_over, flips = [], [], 0
    for j in bvh.joints:
        order = [c for c in j.channels if c.endswith("rotation")]
        if not order:
            continue
        r = bvh.rotations(j.name)
        keys = [bvh_util.euler_to_quat(r[a], order) for a, _ in holds]
        flips += sum(1 for q0, q1 in zip(keys, keys[1:])
                     if float(np.dot(q0, q1)) < 0)
        for (_, k0), (k1, _) in zip(holds, holds[1:]):
            q0 = bvh_util.euler_to_quat(r[k0], order)
            q1 = bvh_util.euler_to_quat(r[k1], order)
            theta = bvh_util.quat_angle_deg(q0, q1)
            for f in range(k0 + 1, k1):
                qf = bvh_util.euler_to_quat(r[f], order)
                a = bvh_util.quat_angle_deg(q0, qf)
                c = bvh_util.quat_angle_deg(qf, q1)
                arc_dev.append(max(0.0, (a + c - theta) / 2.0))
                endpoint_over.append(max(0.0, a - theta, c - theta))
    arc_dev = np.asarray(arc_dev)
    endpoint_over = np.asarray(endpoint_over)
    chan = bvh_util.channel_overshoot(bvh, n_poses).ravel()

    print(f"(b) in-betweens: {len(arc_dev)} (bone, frame) samples")
    print(f"    off the slerp arc:  max {arc_dev.max():.4f} deg, "
          f"p99 {np.percentile(arc_dev, 99):.4f}")
    print(f"    past an endpoint:   max {endpoint_over.max():.4f} deg")
    print(f"    Euler-channel overshoot: max {chan.max():.4f} deg, "
          f"{int((chan > 1e-9).sum())} of {chan.size} samples nonzero")
    print(f"(c) consecutive keyframe quaternion pairs with dot < 0: {flips}")


def report_translation(bvh) -> None:
    """(d) which bone carries the motion the hips should have."""
    ranges = {}
    for j in bvh.joints:
        p = bvh.positions(j.name)
        if p.shape[1] == 3:
            ranges[j.name] = float(np.max(p.max(0) - p.min(0)))
    hips = ranges.get("hips", float("nan"))
    others = {k: v for k, v in ranges.items() if k != "hips"}
    name, worst = max(others.items(), key=lambda kv: kv[1])
    print(f"(d) hips translation range {hips:.6f} rig units "
          f"({100 * hips / RIG_HEIGHT:.2f} % of the {RIG_HEIGHT} rig height)")
    print(f"    largest non-hips: {name} {worst:.6f} = "
          f"{100 * worst / RIG_HEIGHT:.4f} % of rig height")
    for k, v in sorted(others.items(), key=lambda kv: -kv[1])[:4]:
        print(f"      {k:<14} {v:.6f}")


def report_fallback(poses, out_dir, blender, timeout) -> None:
    """(e) what ships when character_blend() points at nothing."""
    missing = out_dir / "no_such_character.blend"
    res = export_animation(poses, out_dir, name="fallback", fps=30,
                           render_video=False, blender=blender,
                           timeout=timeout, character=str(missing))
    print(f"(e) missing character asset -> ExportResult.ok = {res.ok}, "
          f"returncode {res.returncode}")
    if not res.bvh:
        print("    no BVH written")
        return
    bvh = bvh_util.parse(res.bvh)
    rot = sum(1 for j in bvh.joints for c in j.channels
              if c.endswith("rotation"))
    pos = sum(1 for j in bvh.joints for c in j.channels
              if c.endswith("position"))
    print(f"    {len(bvh.joints)} joints, {bvh.n_frames} frames, "
          f"{rot} rotation channels, {pos} position channels")
    print(f"    root '{bvh.joints[0].name}' children: "
          f"{[bvh.joints[c].name for c in bvh.joints[0].children]}")
    print("    -> the client receives this instead of the character, and the "
          "host reports a successful export" if res.ok else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=str(DEFAULT_PROJECT))
    ap.add_argument("--out", default=None,
                    help="where to export (default: a temp dir)")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args(argv)

    project = load_project(Path(args.project))
    poses = np.stack([f.fitted3d for f in project.frames])
    n_poses = len(poses)
    blender = blender_binary()
    character = character_blend()
    if not character:
        raise SystemExit("the bundled character asset is missing")

    import tempfile
    tmp = None
    if args.out:
        out_dir = Path(args.out)
    else:
        tmp = tempfile.TemporaryDirectory()
        out_dir = Path(tmp.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"project {args.project}: {n_poses} poses")
    print(f"blender {blender}\ncharacter {character}\nout {out_dir}\n")

    res = export_animation(poses, out_dir, name="fidelity", fps=30,
                           render_video=False, blender=blender,
                           timeout=args.timeout, character=character)
    if not res.ok or not res.bvh:
        print(res.stdout[-3000:])
        raise SystemExit(f"export failed (rc={res.returncode}):\n"
                         f"{res.stderr[-3000:]}")
    bvh = bvh_util.parse(res.bvh)
    print(f"exported {res.bvh.name}: {bvh.n_frames} frames at "
          f"{bvh.fps:.1f} fps, {len(bvh.joints)} joints, expected "
          f"{bvh_util.expected_frames(n_poses)} frames\n")

    R = de_tilt_rotation(poses)
    up = poses @ R.T
    valid = ~np.isnan(up).any(2)
    ch = Character()
    ch.fit_to_subject(up)
    print(f"subject height {subject_height(poses):.4f} m, "
          f"character scale {ch._scale:.4f}\n")

    report_keyframe_fidelity(bvh, ch, up, valid, n_poses)
    print()
    report_interpolation(bvh, n_poses)
    print()
    report_translation(bvh)
    print()
    report_fallback(poses, out_dir, blender, args.timeout)
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

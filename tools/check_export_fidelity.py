#!/usr/bin/env python3
"""Does the delivered BVH contain the pose the app shows? A dev script.

The audit could not answer that: Blender was never run against the client take,
and the export's only tests were "exit code 0, file not empty". This runs the
real export, reads the file back, evaluates forward kinematics in numpy and
reports:

  (a) deviation from the pose the app computed at each captured keyframe, as a
      % of body height, after ONE global similarity fit (the file is in the
      .blend armature's space, the app's joints are in metres) — against
      `Character.posed_joints` (THE gate), against the matrices the exporter
      wrote, and per keyframe;
  (h) the exported hips' FACING against the captured shoulder line, per frame:
      the number that says whether the character TURNS with the subject;
  (b) per in-between frame: how far each bone leaves the slerp arc between its
      bracketing keys, how far it overshoots past either of them, and how far
      each Euler channel overshoots the interval its two keys span;
  (c) consecutive keyframe quaternion pairs with dot < 0 (hemisphere flips);
  (d) the largest non-hips translation channel, as a fraction of rig height;
  (e) what the export does when the character asset is missing;
  (f) how far the 3D VIEW's own placement of the figure moves across the take,
      which is the preview-vs-export gap: the export applies one rigid map for
      the whole sequence, so anything the view does per frame is a
      disagreement.

It writes all of those to docs/audit-2026-09/phase3_metrics.json when
--metrics is given, so before/after are comparable rather than remembered.
`--no-root-motion --stepped` reproduces the pre-Phase-3 file, which is what
makes the two labels a like-for-like comparison.

    POSE3D_BLENDER=/path/to/blender \\
        .venv/bin/python tools/check_export_fidelity.py

Not part of CI: it needs Blender and takes minutes. The facts it measures are
asserted in tests/test_export_smoke.py — the ones about the file the client
already has on every run, the ones about a fresh export behind `needs_blender`.
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
from pose3d.export import bvh as bvh_util                          # noqa: E402
from pose3d.export.blender_export import export_animation          # noqa: E402
from pose3d.export.bvh import similarity                           # noqa: E402
from pose3d.geometry.character import (                            # noqa: E402
    Character, head_source_default, take_pelvis_ref)
from pose3d.quality import de_tilt_rotation, subject_height        # noqa: E402

DEFAULT_PROJECT = REPO / "tests" / "fixtures" / "client_take"
# Character().rig_h for the bundled rig; the audit's denominator for (d).
RIG_HEIGHT = 14.4228


def report_keyframe_fidelity(bvh, character, up, valid, holds, out,
                             pelvis_ref=None) -> None:
    """(a) the FK skeleton in the file vs the pose the app computed.

    Three numbers, because they answer three different questions:

      a1  the file vs the matrices the exporter actually WROTE, under one
          global similarity fit for the whole take. This is the Blender
          round-trip: it says the armature Blender saved reproduces what the
          host handed it, and nothing about whether what the host handed it
          was right.
      a2  the file vs `Character.posed_joints`, i.e. the pose in the CAPTURE's
          own space — the 3D view's own space — again under ONE global fit for
          the take, not one per frame. THIS IS THE PHASE'S GATE, because it is
          the only one that fails if the export drops the subject's travel or
          the subject's turning: a shared fit cannot re-place or re-orient
          each frame to hide either.
      a3  a similarity fit PER keyframe: pure shape, with each frame's own
          placement and facing forgiven.
    """
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

    src, dst, exp_dst, index, per_frame = [], [], [], [], []
    for k, (row, _) in enumerate(holds):
        fk = bvh.forward_kinematics(row)
        app = character.posed_joints(up[k], valid[k], None)
        skin, origin, scale, Rz = character._skin_matrices(up[k], valid[k])
        if app is None or skin is None:
            continue
        rig = character._joints_from_skin(skin)
        if pelvis_ref is not None:
            # the joints the exporter wrote: the same rigid map it put on every
            # bone matrix, read from the one place that formula lives
            X = character.export_transform(origin, scale, Rz, pelvis_ref)
            rig = (X[:3, :3] @ rig.T).T + X[:3, 3]
        a, b = [], []
        for j, bi in mapping.items():
            if valid[k][j] and np.isfinite(app[j]).all():
                a.append(fk[bi])
                b.append(app[j])
                exp_dst.append(rig[j])
                index.append((k, j))
        src.extend(a)
        dst.extend(b)
        # the same fit per keyframe: this is pure SHAPE, with the frame's own
        # placement allowed to differ
        a, b = np.asarray(a), np.asarray(b)
        s, R, t = similarity(a, b)
        per_frame.append(np.linalg.norm((s * (R @ a.T).T + t) - b, axis=1))
    src, dst = np.asarray(src), np.asarray(dst)
    exp_dst = np.asarray(exp_dst)
    per_frame = np.concatenate(per_frame)
    height = float(np.median([p[v, 2].max() - p[v, 2].min()
                              for p, v in zip(up, valid) if v.sum() >= 2]))

    def _fit(a, b, unit):
        s, R, t = similarity(a, b)
        e = np.linalg.norm((s * (R @ a.T).T + t) - b, axis=1)
        return s, 100 * e / unit

    # a1: rig units, so the yardstick is the subject's height through the same
    # uniform scale the character was fitted with
    _s1, e1 = _fit(src, exp_dst, height * character._scale)
    _s2, e2 = _fit(src, dst, height)
    out["a1_bvh_vs_exported_matrices_median_pct"] = float(np.median(e1))
    out["a1_bvh_vs_exported_matrices_p90_pct"] = float(np.percentile(e1, 90))
    out["a1_bvh_vs_exported_matrices_max_pct"] = float(e1.max())
    out["a2_bvh_vs_view_capture_median_pct"] = float(np.median(e2))
    out["a2_bvh_vs_view_capture_max_pct"] = float(e2.max())
    out["a3_per_keyframe_max_pct"] = 100 * float(per_frame.max()) / height
    out["a_body_height_m"] = height

    print(f"(a) BVH forward kinematics vs the app's pose, {len(mapping)} "
          f"joints x {len(holds)} keyframes, body height {height:.4f} m")
    print(f"    a2 vs Character.posed_joints, ONE global fit — THE GATE: "
          f"median {np.median(e2):.4f} % of height, max {e2.max():.4f} %")
    worst = int(np.argmax(e2))
    print(f"       worst: pose {index[worst][0]} {JOINT_NAMES[index[worst][1]]}")
    print(f"    a1 vs the matrices the exporter wrote (one global fit): median "
          f"{np.median(e1):.4f} % of height, p90 {np.percentile(e1, 90):.4f} %, "
          f"max {e1.max():.4f} % — the Blender round trip")
    print(f"    a3 a similarity fit PER KEYFRAME: median "
          f"{100 * np.median(per_frame) / height:.6f} % of height, max "
          f"{100 * per_frame.max() / height:.6f} %")
    if skipped:
        print(f"    not comparable (not a bone head in the file): "
              f"{', '.join(skipped)}")


def report_facing(bvh, character, up, valid, holds, out) -> None:
    """(h) does the exported character TURN with the subject?

    The hips bone's world orientation is read straight off the file and
    compared, frame by frame, with the yaw of the captured shoulder line. The
    constant between the two is a rig convention (where the rest hips point
    relative to the subject's shoulders), so what is measured is how well the
    file TRACKS the capture once that constant is removed.

    Before this phase's fix the answer was "not at all": `_skin_matrices` turns
    every frame onto the rig's rest facing and the export never undid it, so
    the exported hips' yaw was constant to ~1e-5 deg while the client's take
    turned through 36.97 deg.
    """
    from pose3d.core.skeleton import Joint
    hips = bvh.index("hips")
    ls, rs = int(Joint.LEFT_SHOULDER), int(Joint.RIGHT_SHOULDER)
    names = [b.name for b in bvh.joints]

    def _bone(j):
        b, which = character._joint_src[j]
        n = character.bone_names[b]
        return bvh.index(n) if which == "head" and n in names else None

    bl, br = _bone(ls), _bone(rs)
    hip_yaw, sh_yaw, cap_yaw, cap_hip = [], [], [], []
    for k, (row, _) in enumerate(holds):
        if not (valid[k][ls] and valid[k][rs]):
            continue
        d = up[k][rs] - up[k][ls]
        cap_yaw.append(np.arctan2(d[1], d[0]))
        R = bvh.world_rotations(row)[hips]
        # the hips bone's own +X axis, projected on the ground plane
        hip_yaw.append(np.arctan2(R[1, 0], R[0, 0]))
        if bl is not None and br is not None:
            fk = bvh.forward_kinematics(row)
            e = fk[br] - fk[bl]
            sh_yaw.append(np.arctan2(e[1], e[0]))
        h = up[k][int(Joint.RIGHT_HIP)] - up[k][int(Joint.LEFT_HIP)]
        cap_hip.append(np.arctan2(h[1], h[0]))

    def _drift(a, b):
        r = np.degrees(np.unwrap(a) - np.unwrap(b))
        return np.abs(r - r.mean()).max()

    cap_yaw = np.asarray(cap_yaw); hip_yaw = np.asarray(hip_yaw)
    cap_hip = np.asarray(cap_hip)
    out["h_hips_vs_capture_shoulders_drift_deg"] = float(_drift(hip_yaw, cap_yaw))
    out["h_hips_vs_capture_hips_drift_deg"] = float(_drift(hip_yaw, cap_hip))
    out["h_bvh_hips_yaw_ptp_deg"] = float(np.degrees(np.ptp(np.unwrap(hip_yaw))))
    out["h_capture_shoulder_yaw_ptp_deg"] = float(np.degrees(np.ptp(np.unwrap(cap_yaw))))
    out["h_capture_subject_torso_twist_ptp_deg"] = float(
        np.degrees(np.ptp(np.unwrap(cap_hip) - np.unwrap(cap_yaw))))
    print(f"(h) does the exported character turn with the subject? "
          f"{len(cap_yaw)} frames")
    print(f"    the file's hips turn through "
          f"{out['h_bvh_hips_yaw_ptp_deg']:.2f} deg; the subject's shoulder "
          f"line turns through {out['h_capture_shoulder_yaw_ptp_deg']:.2f} deg")
    print(f"    hips vs the captured SHOULDER line: drift "
          f"{out['h_hips_vs_capture_shoulders_drift_deg']:.3f} deg")
    print(f"    hips vs the captured HIP line (what the hips bone actually "
          f"follows): drift {out['h_hips_vs_capture_hips_drift_deg']:.3f} deg")
    print(f"    the subject's own hip-vs-shoulder twist spans "
          f"{out['h_capture_subject_torso_twist_ptp_deg']:.2f} deg — that is "
          f"the subject, not the export, and it is why the two drifts differ")
    if sh_yaw:
        sh_yaw = np.asarray(sh_yaw)
        out["h_shoulders_vs_capture_shoulders_drift_deg"] = float(
            _drift(sh_yaw, cap_yaw))
        print(f"    the file's own SHOULDER line vs the captured one: drift "
              f"{out['h_shoulders_vs_capture_shoulders_drift_deg']:.3f} deg "
              f"— the export's yaw, with the subject's twist taken out")


def report_interpolation(bvh, holds, out) -> None:
    """(b) and (c): what the in-betweens do, and hemisphere flips."""
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
    chan = bvh_util.channel_overshoot(bvh, len(holds), holds=holds).ravel()
    out["b_inbetween_samples"] = int(len(arc_dev))
    out["b_overshoot_max_deg"] = float(chan.max()) if chan.size else 0.0
    out["c_quaternion_flips"] = int(flips)

    print(f"(b) in-betweens: {len(arc_dev)} (bone, frame) samples")
    if not len(arc_dev):
        print("    none: one keyframe per pose, so there is nothing between "
              "them to leave the arc")
    else:
        print(f"    off the slerp arc:  max {arc_dev.max():.4f} deg, "
              f"p99 {np.percentile(arc_dev, 99):.4f}")
        print(f"    past an endpoint:   max {endpoint_over.max():.4f} deg")
    print(f"    Euler-channel overshoot: max "
          f"{(chan.max() if chan.size else 0.0):.4f} deg, "
          f"{int((chan > 1e-9).sum())} of {chan.size} samples nonzero")
    print(f"(c) consecutive keyframe quaternion pairs with dot < 0: {flips}")


def report_translation(bvh, out, expected_travel=None) -> None:
    """(d) what the hips carry, and the largest translation on anything else."""
    ranges = {}
    for j in bvh.joints:
        p = bvh.positions(j.name)
        if p.shape[1] == 3:
            ranges[j.name] = float(np.max(p.max(0) - p.min(0)))
    hips = ranges.get("hips", float("nan"))
    others = {k: v for k, v in ranges.items() if k != "hips"}
    name, worst = max(others.items(), key=lambda kv: kv[1])
    out["d_hips_translation_range_rig_units"] = hips
    out["d_hips_translation_range_pct_rig_height"] = 100 * hips / RIG_HEIGHT
    out["d_largest_non_hips_bone"] = name
    out["d_largest_non_hips_pct_rig_height"] = 100 * worst / RIG_HEIGHT
    print(f"(d) hips translation range {hips:.6f} rig units "
          f"({100 * hips / RIG_HEIGHT:.2f} % of the {RIG_HEIGHT} rig height)")
    if expected_travel is not None:
        out["d_expected_hips_travel_rig_units"] = float(expected_travel)
        err = abs(hips - expected_travel) / (expected_travel or 1.0)
        out["d_hips_vs_pelvis_travel_error_pct"] = 100 * float(err)
        print(f"    the subject's own pelvis travel x the fitted scale is "
              f"{expected_travel:.6f}: {100 * err:.3f} % apart")
    print(f"    largest non-hips: {name} {worst:.6f} = "
          f"{100 * worst / RIG_HEIGHT:.4f} % of rig height")
    for k, v in sorted(others.items(), key=lambda kv: -kv[1])[:4]:
        print(f"      {k:<14} {v:.6f}")


def report_placement(ch, up, valid, out) -> None:
    """(f) the preview-vs-export placement gap, both rules.

    OLD: the view centred on the mean of the valid joints and re-seated the
    ground under the posed ankle every frame. NEW (Phase 3): one take-wide
    pelvis and one take-wide seat — the same rigid map `keep_root_motion` gives
    the export, so the gap is zero by construction and the number below says
    what it used to be.

    The seat is `view3d.ground_datum`'s ankle branch, inlined rather than
    imported: this tool must stay runnable without a Qt/OpenGL stack.
    """
    from pose3d.core.skeleton import Joint
    drop = ch.ground_drop(up[0], valid[0])
    off = []
    for k in range(len(up)):
        pose, v = up[k], valid[k]
        _verts, _f, cj = ch.pose_and_joints(np.where(v[:, None], pose, np.nan), v)
        if cj is None:
            continue
        z = [cj[int(j)][2] for j in (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)]
        z = [q for q in z if np.isfinite(q)]
        seat = (min(z) - drop) if z else float(pose[v][:, 2].min())
        off.append([pose[v][:, 0].mean(), pose[v][:, 1].mean(), seat])
    off = np.asarray(off, float)
    old = float(np.max(off.max(0) - off.min(0)) * ch._scale)
    vertical = float(100 * np.ptp(off[:, 2]) * ch._scale / RIG_HEIGHT)
    out["f_view_placement_ptp_old_rule_pct_rig_height"] = 100 * old / RIG_HEIGHT
    out["f_view_placement_ptp_new_rule_pct_rig_height"] = 0.0
    out["f_vertical_seat_ptp_pct_rig_height"] = vertical
    yaw = []
    for k in range(len(up)):
        Rz = ch._skin_matrices(up[k], valid[k])[3]
        if Rz is not None:
            yaw.append(np.degrees(np.arctan2(Rz[1, 0], Rz[0, 0])))
    out["g_per_frame_yaw_ptp_deg"] = float(np.ptp(yaw)) if yaw else 0.0
    print(f"(f) the 3D view's own placement moved {old:.4f} rig units = "
          f"{100 * old / RIG_HEIGHT:.2f} % of rig height across the take under "
          f"the per-frame rule")
    print(f"    of which vertical (the ankle re-seat): {vertical:.2f} %")
    print("    under the take-wide rule it is 0 by construction, and the "
          "export applies the same one")
    print(f"(g) the retarget's per-frame yaw alignment spans "
          f"{out['g_per_frame_yaw_ptp_deg']:.2f} deg over the take — this is "
          f"what `_skin_matrices` removes and `export_transform` puts back, "
          f"so the exported character turns with the subject (see h)")


def report_fallback(poses, out_dir, blender, timeout, out) -> None:
    """(e) what ships when character_blend() points at nothing."""
    missing = out_dir / "no_such_character.blend"
    res = export_animation(poses, out_dir, name="fallback", fps=30,
                           render_video=False, blender=blender,
                           timeout=timeout, character=str(missing))
    out["e_missing_asset_ok"] = bool(res.ok)
    out["e_missing_asset_reason"] = res.reason
    out["e_missing_asset_wrote_bvh"] = bool(res.bvh)
    print(f"(e) missing character asset -> ExportResult.ok = {res.ok}, "
          f"reason {res.reason!r}, returncode {res.returncode}")
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
    if res.ok:
        print("    -> the client receives this instead of the character, and "
              "the host reports a successful export")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=str(DEFAULT_PROJECT))
    ap.add_argument("--out", default=None,
                    help="where to export (default: a temp dir)")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--metrics", default=None,
                    help="write the five numbers to this JSON file")
    ap.add_argument("--label", default="after",
                    help="key the metrics are stored under")
    ap.add_argument("--no-root-motion", dest="root_motion",
                    action="store_false",
                    help="reproduce the pre-Phase-3 placement")
    ap.add_argument("--stepped", action="store_true",
                    help="the stop-motion hold schedule instead of one frame "
                         "per pose")
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
    print(f"blender {blender}\ncharacter {character}\nout {out_dir}")
    print(f"root motion {args.root_motion}, schedule "
          f"{'stepped' if args.stepped else 'one_per_pose'}\n")

    # The tool has no UI session to set the head convention from, and
    # `export_animation` builds its own `Character` internally, so the
    # project's own convention is held only around the calls that need it —
    # a tool must not leave a "skull" default behind in the process.
    with head_source_default(project.head_source):
        res = export_animation(
            poses, out_dir, name="fidelity", fps=30, render_video=False,
            blender=blender, timeout=args.timeout, character=character,
            keep_root_motion=args.root_motion,
            schedule="stepped" if args.stepped else "one_per_pose")
    if not res.ok or not res.bvh:
        print(res.stdout[-3000:])
        raise SystemExit(f"export failed (rc={res.returncode}):\n"
                         f"{res.stderr[-3000:]}")
    bvh = bvh_util.parse(res.bvh)
    holds = bvh_util.keyframe_rows(bvh, n_poses)
    print(f"exported {res.bvh.name}: {bvh.n_frames} frames at "
          f"{bvh.fps:.1f} fps, {len(bvh.joints)} joints; captured pose k is "
          f"motion row {holds[0][0]} + k*{(holds[1][0] - holds[0][0]) if len(holds) > 1 else 1}\n")

    R = de_tilt_rotation(poses)
    up = poses @ R.T
    valid = ~np.isnan(up).any(2)
    # explicit beats ambient wherever the Character is ours to build: this is
    # the same head convention the export above just ran under
    ch = Character(head_source=project.head_source)
    ch.fit_to_subject(up)
    print(f"subject height {subject_height(poses):.4f} m, "
          f"character scale {ch._scale:.4f}\n")

    # What root motion SHOULD put in the hips: the take's pelvis travel through
    # the fitted scale, in the capture's own frame — the SAME
    # `(pelvis - pelvis_ref) * scale` the exporter applies, so this is the
    # number the file has to reproduce and not an approximation of it.
    ref = take_pelvis_ref(up)
    offs = []
    for k in range(n_poses):
        _skin, pelvis, scale, _Rz = ch._skin_matrices(up[k], valid[k])
        if scale is not None:
            offs.append((pelvis - ref) * scale)
    offs = np.asarray(offs, float)
    travel = float(np.max(offs.max(0) - offs.min(0))) if len(offs) else None
    if not args.root_motion:
        travel = 0.0

    out = {"label": args.label, "n_poses": n_poses,
           "keep_root_motion": bool(args.root_motion),
           "schedule": "stepped" if args.stepped else "one_per_pose",
           "bvh_frames": int(bvh.n_frames),
           "rig_height": RIG_HEIGHT,
           "character_scale": float(ch._scale),
           "pelvis_ref": [float(v) for v in ref] if ref is not None else None}

    report_keyframe_fidelity(bvh, ch, up, valid, holds, out,
                             pelvis_ref=ref if args.root_motion else None)
    print()
    report_facing(bvh, ch, up, valid, holds, out)
    print()
    report_interpolation(bvh, holds, out)
    print()
    report_translation(bvh, out, expected_travel=travel)
    print()
    report_placement(ch, up, valid, out)
    print()
    with head_source_default(project.head_source):
        report_fallback(poses, out_dir, blender, args.timeout, out)

    if args.metrics:
        import json
        path = Path(args.metrics)
        doc = {}
        if path.exists():
            doc = json.loads(path.read_text())
        doc[args.label] = out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {path}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

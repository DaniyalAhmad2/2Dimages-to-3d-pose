#!/usr/bin/env python3
"""Measure one take: the numbers every phase of the accuracy work is judged on.

    .venv/bin/python tools/measure_take.py <project_dir> [--calib DIR]
                                           [--json OUT] [--no-retarget]
    .venv/bin/python tools/measure_take.py --diff before.json after.json

The first form prints a markdown report and optionally writes the whole
`pose3d.quality.TakeQuality` as JSON; the second prints the per-metric delta
between two such JSON files, which is how a change is shown to have helped.

All the arithmetic lives in `pose3d.quality` — this file only formats.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS  # noqa: E402
from pose3d.core.io_project import load_project               # noqa: E402
from pose3d.core.skeleton import BONES, JOINT_NAMES           # noqa: E402
from pose3d.quality import (                                  # noqa: E402
    SYMMETRY_PAIRS, TakeQuality, load_rig, take_quality,
)


def _f(x, nd=3) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if v != v or v in (float("inf"), float("-inf")) else f"{v:.{nd}f}"


def markdown(q: TakeQuality, title: str) -> str:
    L: list[str] = []
    A = L.append
    A(f"### {title} — {q.n_frames} frames")
    A("")
    A(f"Body height (median z-extent of the de-tilted pose): "
      f"**{_f(q.subject_height_m, 4)} m**. Figure height in view: "
      f"{_f(q.figure_h_px[CAM_LEFT], 0)} px left / "
      f"{_f(q.figure_h_px[CAM_RIGHT], 0)} px right.")
    A("")

    A("**Reprojection (px)** — `measured` is the raw triangulation, "
      "`delivered` is the pose the app shows and exports")
    A("")
    A("| camera | measured median | delivered median | delivered/measured | "
      "delivered % of figure |")
    A("|---|---:|---:|---:|---:|")
    for cam in CAMERAS:
        m, d = q.reproj[cam]["measured"], q.reproj[cam]["delivered"]
        A(f"| {cam} | {_f(m['median_px'], 2)} | {_f(d['median_px'], 2)} | "
          f"{_f(q.reproj_ratio(cam), 2)}x | "
          f"{_f(d['median_pct_figure'], 2)} % |")
    A("")
    A("| joint | measured L | delivered L | measured R | delivered R |")
    A("|---|---:|---:|---:|---:|")
    for name in JOINT_NAMES:
        ml = q.reproj[CAM_LEFT]["measured"]["per_joint"][name]["median_px"]
        dl = q.reproj[CAM_LEFT]["delivered"]["per_joint"][name]["median_px"]
        mr = q.reproj[CAM_RIGHT]["measured"]["per_joint"][name]["median_px"]
        dr = q.reproj[CAM_RIGHT]["delivered"]["per_joint"][name]["median_px"]
        A(f"| {name} | {_f(ml, 2)} | {_f(dl, 2)} | {_f(mr, 2)} | {_f(dr, 2)} |")
    A("")

    A("**Bone length across frames, RAW triangulation** "
      "(a rigid mannequin: a perfect pipeline gives std = 0)")
    A("")
    A("| bone | n | median (m) | std (m) | CV % |")
    A("|---|---:|---:|---:|---:|")
    for a, b in BONES:
        v = q.bone_cv["bones"][(int(a), int(b))]
        A(f"| {v['name']} | {v['n']} | {_f(v['median_m'], 4)} | "
          f"{_f(v['std_m'], 4)} | {_f(v['cv_pct'], 1)} |")
    A(f"| **median / max over bones** |  |  |  | "
      f"**{_f(q.bone_cv['median_cv_pct'], 1)} / "
      f"{_f(q.bone_cv['max_cv_pct'], 1)}** |")
    A("")

    A("**Left/right symmetry** (the same mannequin limb on both sides)")
    A("")
    A("| limb | left (m) | right (m) | L/R | asym % |")
    A("|---|---:|---:|---:|---:|")
    for label, *_ in SYMMETRY_PAIRS:
        s = q.symmetry[label]
        A(f"| {label} | {_f(s['left_m'], 4)} | {_f(s['right_m'], 4)} | "
          f"{_f(s['ratio_L_over_R'], 3)} | {_f(s['asym_pct'], 1)} |")
    A("")

    e = q.epipolar
    A(f"**Body-keypoint epipolar (Sampson, px)** — {e['n_pairs']} pairs, "
      f"pipeline threshold {_f(e['threshold_px'], 1)} px, "
      f"{_f(100 * (e['frac_over_threshold'] or 0), 1)} % over it")
    A("")
    A("| median | p90 | p99 | max | % of left diagonal |")
    A("|---:|---:|---:|---:|---:|")
    A(f"| {_f(e['median_px'], 2)} | {_f(e['p90_px'], 2)} | "
      f"{_f(e['p99_px'], 2)} | {_f(e['max_px'], 2)} | "
      f"{_f(e['median_pct_diag'], 3)} % |")
    A("")
    A("| image | point-to-line median px | p90 px | % of own diagonal |")
    A("|---|---:|---:|---:|")
    for cam in CAMERAS:
        v = e["per_image"][cam]
        A(f"| {cam} ({_f(v['image_diag_px'], 0)} px diag) | "
          f"{_f(v['median_px'], 2)} | {_f(v['p90_px'], 2)} | "
          f"{_f(v['median_pct_diag'], 3)} % |")
    A("")

    g = q.gaps
    A(f"**Gaps** — {g['n_missing']} joints with no 3D "
      f"({_f(g['missing_pct'], 1)} %), {g['rejected']} 2D observations "
      f"gated out, {g['undetected']} never detected, {g['filled']} filled")
    if g["missing"]:
        A("")
        A(", ".join(f"{fid} {jname}" for fid, jname in g["missing"][:20]))
    A("")

    if q.retarget_pct_height is None:
        A("_No character supplied: retarget, roll, sole tilt and ground datum "
          "not measured._")
        return "\n".join(L)

    r = q.retarget_pct_height
    A(f"**Retargeting onto the bundled character** — uniform scale "
      f"{_f(r['character_scale'], 4)}, {r['n_frames_posed']}/{q.n_frames} "
      f"frames posed")
    A("")
    A("| joint | n | median (m) | median % height | max % height |")
    A("|---|---:|---:|---:|---:|")
    for name in JOINT_NAMES:
        v = r["per_joint"][name]
        A(f"| {name} | {v['n']} | {_f(v['median_m'], 4)} | "
          f"{_f(v['median_pct_height'], 1)} | {_f(v['max_pct_height'], 1)} |")
    A(f"| **ALL** |  | {_f(r['median_m'], 4)} | "
      f"**{_f(r['median_pct_height'], 1)}** | p90 "
      f"{_f(r['p90_pct_height'], 1)} |")
    A("")
    A("**Bend-plane normal, captured vs character (deg)**")
    A("")
    A("| chain | n | median deg | max deg |")
    A("|---|---:|---:|---:|")
    for label, v in r["bend_plane"].items():
        A(f"| {label} | {v['n']} | {_f(v['median_deg'], 1)} | "
          f"{_f(v['max_deg'], 1)} |")
    A("")

    A("**Limb roll vs the captured bend plane (deg)** — the angle about each "
      "bone's own aim axis; no positional metric can see it")
    A("")
    A("| bone | n | median deg | max deg |")
    A("|---|---:|---:|---:|")
    for role, v in q.roll_error_deg.items():
        A(f"| {role} | {v.get('n', '')} | {_f(v['median_deg'], 1)} | "
          f"{_f(v['max_deg'], 1)} |")
    A("")
    A("| foot | sole tilt off horizontal, median deg | max deg |")
    A("|---|---:|---:|")
    for role, v in q.sole_tilt_deg.items():
        A(f"| {role} | {_f(v['median_deg'], 1)} | {_f(v['max_deg'], 1)} |")
    A("")
    gd = q.ground_datum_pct
    A(f"**Ground datum** — captured ankle above the character's lowest vertex: "
      f"median {_f(gd['median'], 1)} % of height, peak-to-peak "
      f"{_f(gd['peak_to_peak'], 1)} % (the figure bobs against a fixed grid by "
      f"the peak-to-peak).")
    return "\n".join(L)


def _flatten(o, prefix="", out=None) -> dict[str, float]:
    """Every numeric leaf of a TakeQuality dict, keyed by its dotted path."""
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(o, bool):
        pass
    elif isinstance(o, (int, float)):
        out[prefix] = float(o)
    return out


def diff(a_path: Path, b_path: Path) -> str:
    a = _flatten(json.loads(Path(a_path).read_text()))
    b = _flatten(json.loads(Path(b_path).read_text()))
    L = [f"### {a_path} -> {b_path}", "",
         "| metric | before | after | delta |", "|---|---:|---:|---:|"]
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            L.append(f"| {key} | {_f(va, 4)} | {_f(vb, 4)} | — |")
        elif va != vb:
            L.append(f"| {key} | {_f(va, 4)} | {_f(vb, 4)} | "
                     f"{vb - va:+.4f} |")
    only = [k for k in sorted(set(a) ^ set(b))]
    if only:
        L += ["", f"_{len(only)} metric(s) present in only one file._"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project_dir", nargs="?",
                    help="a pose3d project folder (holds project.json)")
    ap.add_argument("--calib", default=None,
                    help="calibration folder (default: <project>/calibration)")
    ap.add_argument("--json", default=None, help="write the full metrics here")
    ap.add_argument("--no-retarget", action="store_true",
                    help="skip everything that needs the bundled character")
    ap.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"),
                    help="print the per-metric delta between two --json files")
    args = ap.parse_args(argv)

    if args.diff:
        print(diff(Path(args.diff[0]), Path(args.diff[1])))
        return 0
    if not args.project_dir:
        ap.error("a project_dir is required (or use --diff A.json B.json)")

    project_dir = Path(args.project_dir)
    calib_dir = Path(args.calib) if args.calib else project_dir / "calibration"
    project = load_project(project_dir)
    rig = load_rig(calib_dir)

    character = None
    if not args.no_retarget:
        from pose3d.geometry.character import Character
        # the take's OWN head convention, not the process default: measuring a
        # skull-HEAD project under the nose convention applies the ~45 deg nose
        # correction to a point that needs none, and the retarget number that
        # comes back is of a pose the app never shows.
        #
        # `head_mode` is DECLARED rather than read from the project: it is the
        # shipped default, and these numbers are the Nose-mode ones. A take the
        # user has put in Face mode wants "face" passed here — the chain is
        # rigid either way, so only what ORIENTS it changes.
        character = Character(head_source=project.head_source,
                              head_mode="nose")

    q = take_quality(project, rig, character)
    print(markdown(q, project_dir.name))
    if args.json:
        Path(args.json).write_text(json.dumps(q.to_dict(), indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

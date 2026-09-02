#!/usr/bin/env python3
"""Stage the ONNX pose weights into a folder the bundle can ship.

Used by the Windows release workflow and by docker/build.sh, so both deliveries
carry the same files and neither downloads anything at run time.

    python tools/fetch_weights.py --out docker/vendor/rtmlib-cache

Stages the checkpoints for BOTH pose models — COCO-17 and Halpe-26, either of
which the app can be set to detect with — plus the shared YOLOX person
detector.

Files already present are left alone, so this is cheap to re-run and works
offline once a machine has them. With no network and nothing cached it exits
non-zero rather than producing a bundle that would download on the client's
machine.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose3d.detect import models  # noqa: E402


def stage(out: Path, mode: str, feet: bool, allow_download: bool) -> list[Path]:
    from rtmlib.tools.file import download_checkpoint

    entry = models._mode_table(feet)[mode]
    out.mkdir(parents=True, exist_ok=True)
    staged = []
    for url in (entry["det"], entry["pose"]):
        name = models.checkpoint_name(url)
        dst = out / name
        if dst.is_file():
            print(f"  have    {name}")
            staged.append(dst)
            continue
        src = models._find(name)          # e.g. this machine's rtmlib cache
        if src is not None:
            print(f"  copy    {name}  <- {src.parent}")
            shutil.copy2(src, dst)
        elif allow_download:
            print(f"  fetch   {name}")
            got = download_checkpoint(url)
            shutil.copy2(got, dst)
        else:
            raise SystemExit(
                f"missing {name} and --no-download was given.\n"
                f"Run once with a network, or copy it into {out}.")
        staged.append(dst)
    return staged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path,
                    help="folder to stage the .onnx files into")
    ap.add_argument("--mode", default="balanced",
                    choices=("lightweight", "balanced", "performance"))
    ap.add_argument("--no-download", action="store_true",
                    help="fail rather than reach the network")
    a = ap.parse_args()

    print(f"staging into {a.out}")
    # BOTH pose models, always. The Halpe-26 checkpoint used to sit behind a
    # --feet flag that no delivery passed, so every bundle ever built shipped
    # without it and would have fallen through to rtmlib's downloader on the
    # client's machine the moment the app asked for that model — the exact
    # failure pose3d/detect/models.py exists to prevent. Which model the app
    # runs is one constant (detect.rtmpose.USE_HALPE26); what the bundle
    # CONTAINS must not be a second decision. The YOLOX person detector is
    # shared, so this is one extra file (55.7 MB).
    staged = stage(a.out, a.mode, feet=False, allow_download=not a.no_download)
    staged += stage(a.out, a.mode, feet=True,
                    allow_download=not a.no_download)

    total = sum(p.stat().st_size for p in set(staged))
    print(f"{len(set(staged))} file(s), {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

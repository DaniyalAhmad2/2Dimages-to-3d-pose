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

Everything staged is then verified against packaging/windows/inputs.json — the
one place a checkpoint's checksum is written. A truncated download, a half
finished earlier run, or an upstream file that changed under its own name all
produce a bundle that builds, ships, and then detects nothing on the client's
machine; only a hash notices. "Already present" files are verified too, because
that is exactly what an interrupted earlier run leaves behind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose3d.detect import models  # noqa: E402

#: The mode a bare invocation stages — what make_windows_bundle.ps1 uses.
DEFAULT_MODE = "balanced"

INPUTS = (Path(__file__).resolve().parent.parent
          / "packaging" / "windows" / "inputs.json")


#: What every weights entry has to say. A file with no `sha256` is a file
#: nothing verifies, which is the state this script exists to make impossible.
ENTRY_KEYS = ("name", "sha256", "size")


def expected(mode: str, inputs: Path = INPUTS) -> dict[str, dict]:
    """{filename: entry} for one mode, from the single source of truth.

    Every way this file can be unusable is a `SystemExit` naming it. It used
    to be an unhandled `FileNotFoundError`, `JSONDecodeError` or `KeyError` —
    a traceback out of a build script, on the one file whose whole job is to
    be the place someone corrects.
    """
    try:
        data = json.loads(inputs.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"{inputs} is missing. It is the only place a checkpoint's "
            "checksum is written, so nothing staged could be verified.") from None
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"{inputs} could not be read ({type(e).__name__}: {e}).") from None

    weights = data.get("weights") if isinstance(data, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise SystemExit(f"{inputs} has no 'weights' section.")
    if mode not in weights:
        raise SystemExit(
            f"{inputs} has no weights entry for --mode {mode} "
            f"(it lists {', '.join(sorted(weights))}).\n"
            "Add one — with the real checksums — or the bundle would ship "
            "files nothing has ever verified.")
    for entry in weights[mode]:
        missing = [k for k in ENTRY_KEYS if k not in entry]
        if missing:
            raise SystemExit(
                f"{inputs}: a weights entry for --mode {mode} is missing "
                f"{', '.join(missing)} ({entry}).")
    return {w["name"]: w for w in weights[mode]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(paths, want: dict[str, dict]) -> None:
    """SystemExit naming the file and BOTH hashes on the first mismatch.

    The actual value is in the message on purpose: a checksum typed wrong into
    inputs.json fails every build until someone corrects it, and correcting it
    should not need a Windows machine to read the hash off."""
    for path in sorted(set(paths)):
        if path.name not in want:
            raise SystemExit(
                f"{path.name} is not listed in {INPUTS} for this mode, so "
                "nothing can verify it.")
        if not path.is_file():
            raise SystemExit(f"{path} is missing.")
        entry, got = want[path.name], sha256(path)
        if got != entry["sha256"]:
            raise SystemExit(
                f"checksum mismatch for {path.name}\n"
                f"  expected {entry['sha256']}  {entry['size']} bytes "
                f"({INPUTS})\n"
                f"  actual   {got}  {path.stat().st_size} bytes ({path})\n"
                "Delete it and re-run, or correct inputs.json if the "
                "upstream checkpoint really did change.")
        print(f"  verified {path.name}")


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
    # Only the modes inputs.json describes. A third choice, "performance", was
    # offered here with no entry behind it, so the one thing it could do was
    # stage 150 MB that nothing would ever verify.
    ap.add_argument("--mode", default=DEFAULT_MODE,
                    choices=("lightweight", "balanced"))
    ap.add_argument("--no-download", action="store_true",
                    help="fail rather than reach the network")
    ap.add_argument("--verify-only", action="store_true",
                    help="hash what is already in --out; stage nothing")
    a = ap.parse_args()

    # Before anything is fetched: a mode inputs.json says nothing about could
    # be staged but never verified, which is the state this file exists to
    # make impossible.
    want = expected(a.mode)

    if a.verify_only:
        print(f"verifying {a.out}")
        verify([a.out / name for name in want], want)
        return 0

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

    # Exactly what this invocation put there — including the ones it found
    # already present and left alone. Verifying the expected LIST instead
    # would make the two modes the test workflow stages into one folder
    # each fail on the other's files.
    verify(staged, want)

    total = sum(p.stat().st_size for p in set(staged))
    print(f"{len(set(staged))} file(s), {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

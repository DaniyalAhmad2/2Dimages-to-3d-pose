#!/usr/bin/env python3
"""Read one Blender fact out of packaging/windows/inputs.json.

    python tools/blender_input.py --url
    python tools/blender_input.py --cache-key

`inputs.json` is the only place a Blender version or checksum is written down.
It used to be three: the bundle script's `-BlenderVersion` default and a
hard-coded URL in each of the two workflows, none of them checked against the
others and none of them verified against a hash. Bumping Blender meant editing
three files correctly, and downloading 414 MB over HTTP with no way to notice a
truncated or substituted archive.

So every consumer asks this, one value at a time — a single line on stdout,
which is all a PowerShell `$(...)` or a workflow `$GITHUB_OUTPUT` needs — and
gets nothing at all when the file does not say what it should:

    --version    5.1.1
    --url        https://download.blender.org/release/Blender5.1/blender-...zip
    --sha256     the published checksum for that file
    --zip-name   blender-<version>-windows-x64.zip
    --cache-key  blender-<version>-windows-x64-<first 12 of sha256>

The cache key carries the hash on purpose. Keyed on the version alone, a
corrected checksum would keep restoring the old archive from the runner's cache
and every build would fail on a file nobody can see.

The checksum is Blender's own, published beside the release
(https://download.blender.org/release/Blender5.1/blender-5.1.1.sha256), not one
we computed from a download of our own — a hash taken from the same fetch it is
meant to police proves nothing.

Where a checksum genuinely cannot be produced yet, `inputs.json` carries the
literal string `FILL-FROM-CI`; `--sha256` and `--cache-key` then refuse rather
than hand a build a value it would compare a real file against.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: The value that means "not hashed yet"; see the module docstring.
UNFILLED = "FILL-FROM-CI"

DEFAULT_INPUTS = (Path(__file__).resolve().parent.parent
                  / "packaging" / "windows" / "inputs.json")

FIELDS = ("version", "url", "sha256", "zip-name", "cache-key")


class Malformed(Exception):
    """inputs.json cannot be trusted to describe the Blender download."""


def load(path: Path = DEFAULT_INPUTS) -> dict:
    """The `blender` section, checked for the shape every caller assumes."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Malformed(f"{path}: no such file")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Malformed(f"{path}: not readable as JSON ({exc})")

    blender = data.get("blender") if isinstance(data, dict) else None
    if not isinstance(blender, dict):
        raise Malformed(f'{path}: no "blender" section')
    missing = [k for k in ("version", "url", "sha256", "size")
               if k not in blender]
    if missing:
        raise Malformed(f"{path}: blender has no {', '.join(missing)}")

    sha = blender["sha256"]
    if sha != UNFILLED and not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
        raise Malformed(
            f"{path}: blender.sha256 is neither 64 hex characters nor "
            f"{UNFILLED} ({sha!r})")

    expected = f"blender-{blender['version']}-windows-x64.zip"
    if blender["url"].rsplit("/", 1)[-1] != expected:
        raise Malformed(
            f"{path}: blender.url does not end in {expected} — the version "
            "and the URL disagree, so the build would verify one file against "
            "another file's checksum")
    return blender


def value(field: str, blender: dict, path: Path = DEFAULT_INPUTS) -> str:
    name = f"blender-{blender['version']}-windows-x64"
    if field in ("sha256", "cache-key") and blender["sha256"] == UNFILLED:
        raise Malformed(
            f"{path}: blender.sha256 is still {UNFILLED}. Nobody has hashed "
            f"{name}.zip yet, so the download cannot be verified. Take the "
            "hash the first CI run prints and write it into inputs.json.")
    return {
        "version": blender["version"],
        "url": blender["url"],
        "sha256": blender["sha256"],
        "zip-name": name + ".zip",
        "cache-key": f"{name}-{blender['sha256'][:12]}",
    }[field]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS,
                    help="where inputs.json is (tests use this)")
    which = ap.add_mutually_exclusive_group(required=True)
    for field in FIELDS:
        which.add_argument(f"--{field}", dest="field", action="store_const",
                           const=field)
    a = ap.parse_args(argv)

    try:
        print(value(a.field, load(a.inputs), a.inputs))
    except Malformed as exc:
        print(f"blender_input.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check an assembled Windows bundle against packaging/windows/manifest.json.

    python tools/check_bundle_layout.py build/Pose3D-Windows
    python tools/check_bundle_layout.py C:\\Pose3D\\Pose3D-Windows --no-hash

`tools/check_bundle_deps.py` asks whether every DLL in the bundle can resolve
its imports. This asks the other question — is everything there at all? — which
nothing asked before, and which has a different answer for every file:

    platforms\\qwindows.dll   QApplication aborts before the app runs a line
    imageformats\\qjpeg.dll   every photo loads as an empty grey frame
    onnxruntime\\*.dll        detection raises inside a worker thread
    models\\*.onnx            detection reaches the network, or detects noise
    blender\\blender.exe      export fails at the end of a long job

The rules live in `manifest.json` rather than here so that the client's own
copy of the app can check itself against the same list at startup
(`pose3d/integrity.py`, name and size only — this tool is the one that hashes).

Everything is `pathlib.glob`: no Windows API, no PE parsing, no `os.access`,
which lied about writability in Program Files once already. So this runs on the
build machine, on a Linux CI runner, and against a folder extracted from the
delivered zip — the copy that actually broke last time.

Every problem is reported, not the first: a bundle that fails one missing file
per CI round trip costs a day to assemble.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_MANIFEST = (Path(__file__).resolve().parent.parent
                    / "packaging" / "windows" / "manifest.json")
DEFAULT_INPUTS = DEFAULT_MANIFEST.with_name("inputs.json")

#: What inputs.json carries where a checksum could not be produced yet; see
#: tools/blender_input.py. Never a pass.
UNFILLED = "FILL-FROM-CI"

KINDS = ("required", "forbidden")

#: Who can act on a rule. See the manifest's own `_comment`.
STAGES = ("bootloader", "pre-qt", "later")


class Malformed(Exception):
    """The manifest (or inputs.json) cannot be trusted to describe a bundle."""


@dataclass(frozen=True)
class Rule:
    """One line of the inventory.

    `pattern` is a `pathlib` glob relative to the bundle root; a trailing
    slash means it must match a directory. `min_size` is per matched file and
    exists to catch the file that is present and empty — a OneDrive
    placeholder, an interrupted copy, an antivirus that took the contents and
    left the name.
    """

    pattern: str
    kind: str
    why: str
    stage: str
    min_count: int = 1
    min_size: int = 0
    sha256_from_inputs: str | None = None


def load(manifest: Path = DEFAULT_MANIFEST) -> list[Rule]:
    """The rules, or `Malformed` naming what is wrong with them.

    Strict on purpose: a rule with a misspelled key, or with no `why`, is a
    rule that quietly checks nothing and passes every build from then on.
    """
    manifest = Path(manifest)
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Malformed(f"{manifest}: no such file")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Malformed(f"{manifest}: not readable as JSON ({exc})")

    rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(rules, list) or not rules:
        raise Malformed(f'{manifest}: no "rules" list')

    known = {f.name for f in fields(Rule)}
    out: list[Rule] = []
    for i, raw in enumerate(rules):
        where = f"{manifest}: rule {i}"
        if not isinstance(raw, dict):
            raise Malformed(f"{where} is not an object")
        unknown = sorted(set(raw) - known)
        if unknown:
            raise Malformed(
                f"{where} ({raw.get('pattern', '?')}) has unknown key(s): "
                f"{', '.join(unknown)}. A rule nothing reads is a rule that "
                "passes every bundle.")
        try:
            rule = Rule(**raw)
        except TypeError as exc:
            raise Malformed(f"{where}: {exc}")
        if rule.kind not in KINDS:
            raise Malformed(f"{where} ({rule.pattern}): kind must be one of "
                            f"{', '.join(KINDS)}, not {rule.kind!r}")
        if rule.stage not in STAGES:
            raise Malformed(f"{where} ({rule.pattern}): stage must be one of "
                            f"{', '.join(STAGES)}, not {rule.stage!r}")
        if not rule.why.strip():
            raise Malformed(
                f"{where} ({rule.pattern}) has no why. The why is what a "
                "failed build and a failed launch print; a rule without one "
                "reports a filename to somebody who cannot act on it.")
        out.append(rule)
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(root: Path, rule: Rule) -> list[Path]:
    """What the rule's pattern finds under `root`."""
    pattern = rule.pattern
    if pattern.endswith("/"):
        return [p for p in sorted(root.glob(pattern.rstrip("/")))
                if p.is_dir()]
    hits = sorted(root.glob(pattern))
    if rule.kind == "required":
        # a directory that happens to match the pattern is not the file
        return [p for p in hits if p.is_file()]
    return hits                      # forbidden: a folder counts, see `locale`


def _expected(key: str, inputs: Path) -> dict[str, str]:
    """{filename: sha256} for a dotted path into inputs.json."""
    try:
        node = json.loads(Path(inputs).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise Malformed(f"{inputs}: no such file")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Malformed(f"{inputs}: not readable as JSON ({exc})")
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise Malformed(f"{inputs}: has no {key}")
        node = node[part]
    if not isinstance(node, list):
        raise Malformed(f"{inputs}: {key} is not a list of files")
    try:
        return {entry["name"]: entry["sha256"] for entry in node}
    except (TypeError, KeyError):
        raise Malformed(f"{inputs}: {key} has an entry without name/sha256")


def check(root: Path, rules, hash_files: bool = False,
          inputs: Path | None = None) -> list[str]:
    """Every way `root` fails `rules`, as lines a human can act on.

    `hash_files` is off by default because the caller that runs on the
    client's machine cannot afford it: hashing 211 MB of checkpoints on every
    launch is seconds of nothing happening, for a fault that a build gate
    catches once.
    """
    root = Path(root)
    if not root.is_dir():
        return [f"{root} is not a directory: there is no bundle here to check"]

    inputs = Path(inputs) if inputs is not None else DEFAULT_INPUTS
    wanted: dict[str, dict[str, str]] = {}
    problems: list[str] = []

    for rule in rules:
        found = _matches(root, rule)

        if rule.kind == "forbidden":
            problems += [f"{_rel(root, path)} must not be in the bundle. "
                         f"{rule.why}" for path in found]
            continue

        if len(found) < rule.min_count:
            problems.append(
                f"{rule.pattern}: found {len(found)}, need {rule.min_count}. "
                f"{rule.why}")

        for path in found:
            if not path.is_file():
                continue                       # a directory rule, e.g. workspace/
            size = path.stat().st_size
            if size < rule.min_size:
                problems.append(
                    f"{_rel(root, path)}: {size} bytes, expected at least "
                    f"{rule.min_size}. {rule.why}")
                continue                       # a truncated file's hash says nothing new
            if not (hash_files and rule.sha256_from_inputs):
                continue
            key = rule.sha256_from_inputs
            if key not in wanted:
                wanted[key] = _expected(key, inputs)
            problems += _checksum_problems(root, path, wanted[key], key, inputs)

    return problems


def _checksum_problems(root: Path, path: Path, wanted: dict[str, str],
                       key: str, inputs: Path) -> list[str]:
    want = wanted.get(path.name)
    rel = _rel(root, path)
    if want is None:
        return [f"{rel}: {inputs} does not list it under {key}, so nothing "
                "can verify it. Either it does not belong in the bundle or "
                "inputs.json has not been told about it."]
    if want == UNFILLED:
        return [f"{rel}: {inputs} still says {UNFILLED} for it, so this file "
                "cannot be verified. Take the hash the first CI run prints "
                "and write it into inputs.json."]
    got = sha256(path)
    if got != want:
        return [f"{rel}: sha256 {got} does not match {want} ({inputs}). "
                "Same name, different file: delete it and stage it again."]
    return []


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:                        # pragma: no cover - defensive
        return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path, help="the extracted bundle folder")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="the inventory to check against")
    ap.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS,
                    help="where the checkpoint checksums are written down")
    ap.add_argument("--no-hash", action="store_true",
                    help="name and size only; skip the model checksums")
    a = ap.parse_args(argv)

    try:
        rules = load(a.manifest)
        problems = check(a.root, rules, hash_files=not a.no_hash,
                         inputs=a.inputs)
    except Malformed as exc:
        print(f"check_bundle_layout.py: {exc}", file=sys.stderr)
        return 1

    if problems:
        print(f"{a.root}: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"{a.root}: {len(rules)} rule(s) satisfied"
          f"{' (checksums skipped)' if a.no_hash else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

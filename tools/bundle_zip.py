#!/usr/bin/env python3
"""Zip the Windows bundle, and refuse one the client could not extract.

    python tools/bundle_zip.py build/Pose3D-Windows

The first bundle the client received would not extract: "path too long".
Windows' MAX_PATH is 260 characters for the whole path, this bundle's deepest
entry — Blender's USD/MaterialX tree — is already ~148 characters inside it,
and every character left over is the folder they extract into.

Two rules follow, and both used to live only as PowerShell: one block pasted
into the release workflow, another in make_windows_bundle.ps1 that no CI job
ever ran, with no depth gate in it at all.

1. Archive from the PARENT of the bundle, so entries begin at
   `Pose3D-Windows\\` and not `build\\Pose3D-Windows\\`. Those six characters
   come straight off the client's extraction path.
2. Nothing deeper than 180 characters ships. That leaves ~80 for the
   extraction root, which `C:\\Users\\<name>\\Downloads\\` mostly spends.

Checked on the NAMES, before compressing: a 1.6 GB tree should not have to be
zipped for five minutes to learn that it cannot be unzipped.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Characters an archive entry may use before the extraction root runs out.
DEFAULT_LIMIT = 180


def deepest_path(names) -> int:
    """Longest entry name, in characters. 0 for an empty archive."""
    return max((len(name) for name in names), default=0)


def check_depth(names, limit: int = DEFAULT_LIMIT) -> list[str]:
    """One problem line per entry too deep to extract, longest first."""
    over = sorted((name for name in names if len(name) > limit),
                  key=len, reverse=True)
    return [f"{len(name)} chars (limit {limit}): {name}" for name in over]


def _entry_names(folder: Path) -> list[str]:
    """What the archive will call each entry, rooted at the leaf.

    Directories included: a zip stores an entry for each one, and an EMPTY
    directory is the only entry that can be too deep to extract without any
    file under it being too deep — so checking files alone would pass a bundle
    that still fails on the client.
    """
    root = folder.parent
    return sorted(p.relative_to(root).as_posix() for p in folder.rglob("*"))


def zip_bundle(folder: Path, out: Path,
               limit: int = DEFAULT_LIMIT) -> Path:
    """Archive `folder` into `out`; SystemExit if anything is too deep."""
    folder = Path(folder)
    out = Path(out)
    if not folder.is_dir():
        raise SystemExit(f"bundle_zip: {folder} is not a directory")

    names = _entry_names(folder)
    problems = check_depth(names, limit)
    if problems:
        raise SystemExit(
            f"bundle_zip: {len(problems)} path(s) in {folder.name} are too "
            "deep to extract on Windows; the client sees 'path too long'.\n  "
            + "\n  ".join(problems))

    print(f"{len(names)} entries, deepest {deepest_path(names)} chars "
          f"(limit {limit})")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    # 7-Zip where it exists: shutil's ZipFile takes many minutes on a 1.6 GB
    # tree. Both are run from the bundle's parent, which is what keeps the
    # entries rooted at the leaf.
    if shutil.which("7z"):
        # 7-Zip names the file itself just as make_archive does below: given a
        # name with no extension it appends the archive type's, so `--out
        # delivery-2026-09` produced delivery-2026-09.zip while this function
        # returned — and main() printed, and the release step would publish —
        # a path with nothing at it.
        #
        # So it is given a name that already ends in .zip, and the result is
        # moved onto `out`. A temporary name of our own rather than
        # `<out>.zip`, because that name can belong to somebody else — a
        # previous release's archive, sitting exactly where 7-Zip would have
        # written by default — and `7z a` ADDS, which would publish its
        # contents as this bundle's, while deleting it first would destroy a
        # file this tool did not create.
        tmp = out.parent / f".{out.name}.{os.getpid()}.tmp.zip"
        # this name is ours; only an earlier run of this tool that died
        # between the two lines below can have left one, and `a` would add to
        # it. Nothing else in the directory is touched.
        tmp.unlink(missing_ok=True)
        try:
            # captured so that a failure carries 7z's own words into the
            # CalledProcessError; -bso0/-bsp0 mean there is nothing else to
            # see.
            subprocess.run(["7z", "a", "-tzip", "-mx=5", "-bso0", "-bsp0",
                            str(tmp.resolve()), folder.name],
                           cwd=folder.parent, check=True,
                           capture_output=True, text=True)
            tmp.replace(out)
        finally:
            # a 7z that died half-way leaves a partial archive; the release
            # directory is not the place to find one later and wonder.
            tmp.unlink(missing_ok=True)
    else:
        # make_archive names the file itself, appending .zip to the base it is
        # given; move it if that is not what was asked for, so the path this
        # returns is always the file that exists.
        made = Path(shutil.make_archive(str(out.parent / out.stem), "zip",
                                        root_dir=folder.parent,
                                        base_dir=folder.name))
        if made != out:
            made.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path, help="the assembled bundle folder")
    ap.add_argument("--out", type=Path,
                    help="archive to write (default: <folder>.zip)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"longest permitted entry (default {DEFAULT_LIMIT})")
    a = ap.parse_args(argv)

    out = a.out or a.folder.parent / (a.folder.name + ".zip")
    try:
        written = zip_bundle(a.folder, out, a.limit)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        # A full disk or a locked output file is a normal way for the release
        # step to end, and a traceback out of a build script says nothing the
        # person reading the log can act on. 7z's own last line does.
        said = (exc.stderr or exc.stdout or "").strip().splitlines()
        print(f"bundle_zip: 7z failed (exit {exc.returncode})"
              + (f": {said[-1]}" if said else ""), file=sys.stderr)
        return 1
    print(f"{written}, {written.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

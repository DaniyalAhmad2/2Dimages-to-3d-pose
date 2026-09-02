#!/usr/bin/env python3
"""Fail the Windows build when the bundle imports a DLL it does not ship.

The client's copy died at launch with

    Failed to load Python DLL '...\\_internal\\python312.dll'.
    LoadLibrary: The specified module could not be found.

That message is Windows saying python312.dll needs something it cannot find —
a *dependency*, not python312.dll itself. The bundle self-test cannot catch
that class of fault, because the GitHub runner is a developer machine: the
Visual C++ runtime and much of what Qt links against are installed system-wide
there, so a DLL missing from `_internal` still loads on the runner and only
fails on a clean machine.

So this checks the bundle against itself rather than against the runner. Read
the import table of every binary the app ships and require each imported name
to be either a file in the bundle or one of the DLLs Windows itself always
provides. Nothing else counts, and in particular nothing on the build
machine's PATH does.

    python tools/check_bundle_deps.py build/Pose3D-Windows

Only the app is examined — `Pose3D.exe` and everything under `_internal`.
`blender\\` is a separate application that ships its own C runtime in
`blender.crt`, and auditing someone else's release is not this script's job.

Reading import tables needs pefile, which PyInstaller already depends on for
Windows builds; the resolver below is pure Python and is unit-tested on Linux
(tests/test_bundle_deps.py).
"""
from __future__ import annotations

import argparse
import posixpath
from collections.abc import Iterable
from pathlib import Path

# DLLs that are part of Windows and are therefore never bundled. Anything not
# on this list has to be in the bundle: assuming a client machine has it is how
# the python312.dll failure happened in the first place.
SYSTEM_DLLS = frozenset({
    "advapi32.dll", "authz.dll", "avicap32.dll", "bcrypt.dll", "cfgmgr32.dll",
    "comctl32.dll", "comdlg32.dll", "crypt32.dll", "d2d1.dll", "d3d9.dll",
    "d3d11.dll", "d3d12.dll", "dbghelp.dll", "dnsapi.dll", "dwmapi.dll",
    "dwrite.dll", "dxgi.dll", "gdi32.dll", "glu32.dll", "imagehlp.dll",
    "imm32.dll", "iphlpapi.dll", "kernel32.dll", "mpr.dll", "msvcrt.dll",
    "ncrypt.dll", "netapi32.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll",
    "opengl32.dll", "propsys.dll", "psapi.dll", "rpcrt4.dll", "secur32.dll",
    "setupapi.dll", "shell32.dll", "shlwapi.dll",
    # ucrtbase.dll is the universal CRT, shipped with Windows 10 and serviced
    # by Windows Update. PyInstaller bundles a copy as well; either resolves.
    "ucrtbase.dll",
    "uiautomationcore.dll", "user32.dll", "userenv.dll", "uxtheme.dll",
    "version.dll", "winhttp.dll", "wintrust.dll", "winmm.dll", "ws2_32.dll",
    "wsock32.dll", "wtsapi32.dll",
    # Qt6Core links the ICU that Windows itself ships in System32. That is a
    # Windows 10 1903 feature, so it is also the bundle's real floor: on 1809
    # or older Qt6Core.dll cannot load at all.
    "icuuc.dll",
})

# api-ms-win-* / ext-ms-* are API sets: virtual names the loader resolves
# through the schema baked into the OS, so a file of that name need not exist
# anywhere. d3dcompiler_NN.dll ships with Windows and with the graphics
# drivers.
SYSTEM_PREFIXES = ("api-ms-win-", "ext-ms-", "d3dcompiler_")

BINARY_SUFFIXES = (".dll", ".pyd", ".exe")

# The only directories on the DLL search path when the bootloader starts. What
# it loads first — the exe, python312.dll, the C runtime under it — has to be
# satisfied from here, before any package has run os.add_dll_directory and
# before anything is loaded that Windows could match by name alone.
SEARCH_ROOTS = ("", "_internal")


def is_system_dll(name: str) -> bool:
    """Is `name` provided by Windows itself?"""
    low = name.lower()
    return low in SYSTEM_DLLS or low.startswith(SYSTEM_PREFIXES)


def unresolved(imports: Iterable[tuple[str, str]],
               bundled: Iterable[str]) -> list[tuple[str, list[str]]]:
    """The imported DLLs that are neither bundled nor provided by Windows.

    `imports` is (importer, imported DLL name) pairs and `bundled` is the paths
    the bundle ships, both relative to the bundle root. Matching is by file
    name, case-insensitively.

    For a binary in a subdirectory, any copy anywhere in the bundle counts.
    Windows' real search order is narrower, but the gap is closed at run time
    by things a static reader cannot see: a DLL already loaded under that name
    satisfies the import whatever its path, and numpy, scipy and PySide6 all
    call os.add_dll_directory for their own folders. Demanding more would
    condemn ten loads this bundle performs correctly today.

    For a binary in SEARCH_ROOTS, only a copy in SEARCH_ROOTS counts. That set
    is loaded first, when none of the above is true yet — and it is the set the
    client's failure was in: _internal/PySide6/VCRUNTIME140.dll is no use to
    _internal/python312.dll.

    Returns (missing DLL, importers that wanted it) pairs, sorted.
    """
    def at_root(rel):
        return posixpath.dirname(rel.replace("\\", "/")) in SEARCH_ROOTS

    anywhere, in_roots = set(), set()
    for rel in bundled:
        name = posixpath.basename(rel.replace("\\", "/")).lower()
        anywhere.add(name)
        if at_root(rel):
            in_roots.add(name)

    missing: dict[str, set[str]] = {}
    for importer, dll in imports:
        low = dll.lower()
        have = in_roots if at_root(importer) else anywhere
        if low in have or is_system_dll(low):
            continue
        missing.setdefault(low, set()).add(importer)
    return [(dll, sorted(who)) for dll, who in sorted(missing.items())]


def read_imports(path: Path) -> list[str]:
    """The DLL names in one binary's import and delay-import tables."""
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    try:
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ])
        names = []
        for table in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT"):
            for entry in getattr(pe, table, None) or []:
                if entry.dll:
                    names.append(entry.dll.decode("ascii", "replace"))
        return names
    finally:
        pe.close()


def app_files(bundle: Path) -> list[Path]:
    """Everything the app itself ships: the bundle root plus _internal.

    Deliberately not blender/ — a separate application, with its own C runtime
    in blender.crt, that is not on this app's DLL search path either.
    """
    internal = bundle / "_internal"
    if not internal.is_dir():
        raise SystemExit(f"{bundle} has no _internal/ — is that a onedir build?")
    return ([p for p in sorted(bundle.glob("*")) if p.is_file()]
            + [p for p in sorted(internal.rglob("*")) if p.is_file()])


def audit(bundle: Path) -> list[tuple[str, list[str]]]:
    """Read every app binary and report what it imports but does not ship."""
    files = app_files(bundle)
    binaries = [p for p in files if p.suffix.lower() in BINARY_SUFFIXES]
    if not binaries:
        raise SystemExit(f"no .dll/.pyd/.exe found under {bundle}")
    print(f"{len(binaries)} binaries under {bundle}")
    imports: list[tuple[str, str]] = []
    for path in binaries:
        importer = path.relative_to(bundle).as_posix()
        for dll in read_imports(path):
            imports.append((importer, dll))
    print(f"{len(imports)} import entries")
    # resolved against every file, not just the audited ones: an import name
    # need not end in .dll (winspool.drv and friends).
    return unresolved(imports,
                      (p.relative_to(bundle).as_posix() for p in files))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path,
                    help="the assembled bundle, or dist/Pose3D straight from "
                         "PyInstaller")
    a = ap.parse_args()

    if not a.bundle.is_dir():
        raise SystemExit(f"{a.bundle} is not a directory")
    try:
        import pefile  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pefile is not installed, so the bundle's imports cannot be read.\n"
            "It comes with PyInstaller on Windows: run this under the "
            "environment the build used — uv run python "
            "tools/check_bundle_deps.py ...") from None

    missing = audit(a.bundle)
    if not missing:
        print("every imported DLL is either bundled or provided by Windows")
        return 0

    print(f"\n{len(missing)} imported DLL(s) are neither bundled nor part of "
          "Windows.\nOn this machine they resolve from PATH or a system-wide "
          "install; on the client's they will not:\n")
    for dll, importers in missing:
        shown = ", ".join(importers[:4])
        if len(importers) > 4:
            shown += f", and {len(importers) - 4} more"
        print(f"  {dll:40s} imported by {shown}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

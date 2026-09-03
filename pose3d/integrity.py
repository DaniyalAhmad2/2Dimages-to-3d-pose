"""Is the folder this app is running from still the bundle we shipped?

The client's copy has been wrong twice, and neither time was the build's
fault: once a file named in a Windows dialog they had never heard of
(`python312.dll`), once an archive that would not extract at all. What both
have in common is that the app cannot say anything about it — by the time
something is missing, the thing that would have reported it is the thing that
is missing.

So the app checks its own folder against the inventory it ships with
(`packaging/windows/manifest.json`) before it asks Qt for anything, and says
what is wrong in words the client can act on. Four constraints shape all of it:

* **Before QApplication.** `platforms\\qwindows.dll` is loaded by the
  QApplication constructor, which aborts the process when it is missing. A Qt
  dialog can therefore never be the messenger; this runs earlier, from
  `pose3d.app._pre_qt_checks`, and reports through stderr (which the frozen
  build has already pointed at pose3d-log.txt) and a native
  `MessageBoxW` — no Qt, no imports beyond the standard library.
* **Frozen only.** A source checkout has no `_internal/`, no `blender/` and
  often no `models/`, and none of that is a fault.
* **Name and size only.** Hashing 211 MB of checkpoints on every launch would
  be seconds of nothing happening, to re-answer what the release gate answers
  once. `tools/check_bundle_layout.py` is the caller that hashes.
* **Fail open.** A bug in this module must cost the check, never the
  application: `run_startup_check` logs anything that goes wrong inside it and
  lets the app start.

Only the manifest's `pre-qt` rules are this module's business. `bootloader`
rules (python312.dll, the Visual C++ runtime) are loaded by Windows before any
Python runs, so if one were missing this code would never execute — the build
and Diagnose.cmd are where those are caught. `later` rules are worth a build
failure and not worth refusing to start.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from pose3d import runtime

#: The inventory. Frozen, `pose3d/` lives in `_internal/`, so this resolves to
#: `_internal/packaging/windows/manifest.json`, which is where pose3d.spec's
#: `datas` entry puts it; from a checkout it is the file in the repository.
MANIFEST = (Path(__file__).resolve().parent.parent
            / "packaging" / "windows" / "manifest.json")

#: The one stage a check running inside the application can both see and act
#: on. See the module docstring.
STAGE = "pre-qt"

_TITLE = "Pose3D cannot start"


def problems(root: Path | None = None,
             manifest: Path | None = None) -> list[str]:
    """What is missing or empty in `root`, in manifest order, one entry each.

    With no `root`: the folder the frozen executable lives in, and nothing at
    all when the app is not frozen. `pathlib` only — `os.access` lied about
    Program Files once already, and a bundle under OneDrive answers questions
    about its files by downloading them.
    """
    if root is None:
        if not runtime.IS_FROZEN:
            return []
        root = runtime.app_dir()
    root = Path(root)
    manifest = Path(manifest) if manifest is not None else MANIFEST

    rules = json.loads(manifest.read_text(encoding="utf-8"))["rules"]
    found: list[str] = []
    for rule in rules:
        if rule.get("stage") != STAGE or rule.get("kind") != "required":
            continue
        pattern = rule["pattern"]
        hits = [p for p in sorted(root.glob(pattern)) if p.is_file()]
        if len(hits) < rule.get("min_count", 1):
            found.append(f"MISSING  {pattern}\n           {rule['why']}")
            continue
        for path in hits:
            size = path.stat().st_size
            if size < rule.get("min_size", 0):
                rel = path.relative_to(root).as_posix()
                found.append(f"EMPTY    {rel}  ({size} bytes)\n"
                             f"           {rule['why']}")
    return found


def explain(problems: list[str]) -> str:
    """The whole message: what is wrong, why it probably happened, what to do.

    The causes are not guesses — they are the three ways a correct download
    becomes an incorrect folder on a Windows desktop, and the remedy for all
    three is the same one sentence.

    Plain ASCII, deliberately, and the manifest's `why` texts with it: this
    goes to a console whose code page is cp1252 in Europe and cp932 in Japan,
    where a single em dash turns the whole message into a UnicodeEncodeError —
    which this module would then swallow, and start a bundle it had just found
    to be broken. dark.qss already cost us that lesson once.
    """
    if not problems:
        return ""
    return "\n".join([
        f"{_TITLE}: files are missing from its folder.",
        "",
        *(f"  {problem}" for problem in problems),
        "",
        "The download is almost certainly fine; the extracted copy is not.",
        "Any of these leaves exactly this state:",
        "",
        "  * an antivirus quarantined a file after extraction: it was there,",
        "    and now it is not, and nothing was said;",
        "  * the folder is inside OneDrive and those files are placeholders",
        "    that were never downloaded to this machine;",
        "  * the extraction stopped part-way. Windows' built-in extractor",
        "    gives up silently when a path gets too long or the disk fills.",
        "",
        "To fix it: delete this folder and extract Pose3D-Windows.zip again,",
        "to a short path on a local disk (C:\\Pose3D, not a OneDrive or",
        "Documents folder), and exclude that folder from your antivirus.",
        "Then start Pose3D.exe from there.",
    ])


def run_startup_check() -> None:
    """Check the bundle; report and exit(1) if it is broken. Never raises.

    Called from `pose3d.app._pre_qt_checks`, which runs after the crash
    handler (so a failure in here is still reported) and after the `--selftest`
    exit (so the self-test's own report is never pre-empted by a dialog), and
    before `QApplication`.
    """
    try:
        found = problems()
        if not found:
            return
        message = explain(found)
        print(message, file=sys.stderr, flush=True)
        _message_box(message)
    except Exception:                # noqa: BLE001 - see the module docstring
        try:
            print("the bundle integrity check could not run; starting anyway:",
                  file=sys.stderr)
            traceback.print_exc()
            sys.stderr.flush()
        except Exception:                # noqa: BLE001 - stderr itself is gone
            pass
        return
    sys.exit(1)


def _message_box(text: str) -> None:
    """A Windows message box drawn by the OS, not by Qt.

    Qt is exactly what may be unable to start, and a frozen windowed build has
    no console for the client to read stderr in, so this is the only channel
    that is certain to reach them. Anywhere else it does nothing.
    """
    if not (runtime.IS_FROZEN and runtime.IS_WINDOWS):
        return
    try:
        import ctypes
        MB_ICONERROR, MB_SETFOREGROUND = 0x10, 0x10000
        ctypes.windll.user32.MessageBoxW(          # type: ignore[attr-defined]
            None, text, _TITLE, MB_ICONERROR | MB_SETFOREGROUND)
    except Exception:                    # noqa: BLE001 - a dialog is a bonus
        pass

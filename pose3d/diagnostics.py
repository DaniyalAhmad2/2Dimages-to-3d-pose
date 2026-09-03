"""One file the client can send us, written by the app about itself.

Every failure this bundle has had reached us as a sentence in an email —
"it says python312.dll is missing", "the 3D box is black", "it will not
extract" — and every one of them took days of questions to turn into a fact.
The questions are always the same: which Windows, where did you extract it,
is that folder inside OneDrive, what does the log say, does Blender start,
are the models there, does anything render. So the app answers them itself,
in one plain-text report the client can copy or attach:

    Pose3D-diagnose.exe --diagnose      # a console, and a dialog
    Help > Diagnostics                  # the same report from the running app

Two rules shape the whole module.

**Nothing in here may raise.** A diagnostics report that dies on the machine
it was meant to diagnose is worse than none: it removes the last channel we
have. Every fact is gathered behind `_safe`, so a missing Qt, an absent
Blender, a drive that will not answer or a locale API that is not there costs
its own line and nothing else.

**Its own wording is ASCII, deliberately.** It gets printed to a console
whose code page is cp1252 in Europe and cp932 in Japan, where a single em dash
is a UnicodeEncodeError. Not everything it quotes is ours — the self-test
transcript and Blender's own output are not — so `_print` re-encodes rather
than lose the whole report to protect one character.
"""
from __future__ import annotations

import io
import locale
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pose3d import runtime

#: Beside the log, which is the file the client is already asked for.
FILENAME = "pose3d-diagnostics.txt"

# Kept alive for the offscreen GL probe: a QApplication that is garbage
# collected takes Qt's whole platform integration with it.
_QAPP = None


def default_path() -> Path:
    """Where the report is written when nobody says otherwise."""
    log = runtime.log_path()
    folder = log.parent if log is not None else runtime.app_dir()
    return folder / FILENAME


# --- the facts -------------------------------------------------------------

def _system() -> str:
    return f"{platform.platform()}  ({sys.platform})"


def _python() -> str:
    return f"{platform.python_version()}  ({sys.executable})"


def _frozen() -> str:
    if runtime.IS_FROZEN:
        return "yes (PyInstaller bundle)"
    return "no (running from source)"


def _app_folder() -> str:
    return str(runtime.app_dir())


def _log_file() -> str:
    log = runtime.log_path()
    if log is None:
        return ("none: nothing could be written anywhere, so the log went "
                "to the null device")
    return f"{log}" + ("" if log.exists() else "   (nothing written to it yet)")


def _under(path: Path, other) -> bool:
    try:
        Path(path).resolve().relative_to(Path(other).resolve())
    except (ValueError, OSError):
        return False
    return True


def _in_onedrive(root: Path) -> bool:
    """OneDrive hands out placeholder files that are not on this machine, and
    a bundle whose DLLs are placeholders fails in ways nothing can explain."""
    if any(part.lower().startswith("onedrive") for part in Path(root).parts):
        return True
    return any(_under(root, value) for name, value in os.environ.items()
               if name.lower().startswith("onedrive") and value)


def _in_program_files(root: Path) -> bool:
    """Read-only for a normal account, and os.access lied about it once."""
    roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)")]
    return any(_under(root, value) for value in roots if value)


def _install() -> str:
    root = runtime.app_dir()
    downloads = any(part.lower() == "downloads" for part in Path(root).parts)
    return (f"OneDrive: {'yes' if _in_onedrive(root) else 'no'} | "
            f"Downloads: {'yes' if downloads else 'no'} | "
            f"Program Files: {'yes' if _in_program_files(root) else 'no'}")


def _free_disk() -> str:
    usage = shutil.disk_usage(runtime.app_dir())
    return (f"{usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB "
            f"on {runtime.app_dir().anchor or '/'}")


def _code_page() -> str:
    """What decides whether reading a file without an explicit encoding works.

    dark.qss cost us this once: one UTF-8 em dash, and on cp932 the app died
    before its first window.
    """
    parts = [f"preferred {locale.getpreferredencoding(False)}",
             f"filesystem {sys.getfilesystemencoding()}",
             f"stdout {getattr(sys.stdout, 'encoding', None) or 'none'}"]
    if runtime.IS_WINDOWS:
        import ctypes
        kernel32 = ctypes.windll.kernel32          # type: ignore[attr-defined]
        parts.insert(0, f"ANSI {kernel32.GetACP()}, OEM {kernel32.GetOEMCP()}")
    return ", ".join(parts)


def _opengl() -> str:
    """Who is drawing the 3D view — the driver, a software rasteriser, or
    nobody. Asked through an offscreen surface so it costs no window."""
    global _QAPP
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:
        _QAPP = QApplication([])
    surface = QOffscreenSurface()
    surface.create()
    context = QOpenGLContext()
    if not context.create() or not context.makeCurrent(surface):
        return "no OpenGL context could be created on this machine"
    try:
        from OpenGL import GL
        strings = [GL.glGetString(name)
                   for name in (GL.GL_VENDOR, GL.GL_RENDERER, GL.GL_VERSION)]
    finally:
        context.doneCurrent()
    return " / ".join(
        s.decode("ascii", "replace") if isinstance(s, bytes) else str(s)
        for s in strings)


def _blender() -> str:
    from pose3d.config import blender_binary
    exe = blender_binary()
    if not Path(exe).is_file() and shutil.which(exe) is None:
        return f"not found (looked for {exe})"
    out = subprocess.run([exe, "--version"], capture_output=True, timeout=120,
                         **runtime.subprocess_kwargs()).stdout or ""
    first = out.strip().splitlines()[0] if out.strip() else "(printed nothing)"
    return f"{exe}   ({first})"


# --- the blocks ------------------------------------------------------------

def _models() -> str:
    """Both configurations, because the app can be switched between them and
    a bundle carrying only one downloads the other on first use."""
    from pose3d.detect import models
    lines = []
    for feet in (True, False):
        label = "Halpe-26" if feet else "COCO-17"
        w = models.resolve(feet=feet)
        if w is None:
            want = ", ".join(models.required_files(feet=feet))
            lines.append(f"{label}: NOT FOUND ({want})")
        else:
            lines.append(f"{label}: {Path(w.pose).name} + {Path(w.det).name}")
    lines.append("searched: " + " ; ".join(str(d) for d in models.search_dirs()))
    return "\n".join(lines)


def _bundle_check() -> str:
    from pose3d import integrity
    if not runtime.IS_FROZEN:
        return "not a frozen bundle, so there is no inventory to check"
    found = integrity.problems()
    return integrity.explain(found) if found else "no problems found"


def _selftest() -> str:
    """The self-test transcript, without the preview render.

    Rendering a video drives EEVEE through whatever GL the machine has and can
    take minutes; everything else here answers in seconds. The release gate is
    where the video check belongs.
    """
    from pose3d import selftest
    buffer = io.StringIO()
    status = selftest.run(video=False, out=buffer)
    return f"{buffer.getvalue().rstrip()}\n(exit status {status})"


# --- assembling it ---------------------------------------------------------

def _safe(fn) -> str:
    try:
        value = fn()
    except Exception as e:                # noqa: BLE001 - see the docstring
        return f"(could not be determined: {type(e).__name__}: {e})"
    return "(unknown)" if value is None else str(value)


def report() -> str:
    """The whole report, as text. Never raises."""
    # Looked up here rather than in a module-level table so that a caller (and
    # a test) can replace one of them.
    facts = [("system", _system), ("python", _python), ("frozen", _frozen),
             ("app folder", _app_folder), ("log file", _log_file),
             ("install", _install), ("free disk", _free_disk),
             ("code page", _code_page), ("OpenGL", _opengl),
             ("Blender", _blender)]
    blocks = [("models", _models), ("bundle check", _bundle_check),
              ("self-test", _selftest)]

    lines = ["Pose3D diagnostics", "==================",
             f"{'generated':<12}{datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    lines += [f"{label:<12}{_safe(fn)}" for label, fn in facts]
    for label, fn in blocks:
        lines += ["", label, "-" * len(label), _safe(fn)]
    return "\n".join(lines) + "\n"


def write_report(path=None, text: str | None = None) -> Path:
    """Write the report to `path` (default: beside the log) and return it.

    `text` is for the caller that already has one: producing it twice would
    run the self-test twice, Blender and all.
    """
    path = Path(path) if path is not None else default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report() if text is None else text, encoding="utf-8")
    return path


# --- showing it ------------------------------------------------------------
#
# The dialog lives in this module rather than under pose3d/ui/ because it is
# the same two callers' second half: `--diagnose` from a console and Help >
# Diagnostics from the window both produce a report and then have to put it in
# front of someone. Qt is imported inside the functions, so a machine where Qt
# is the broken thing still gets the console report.

def build_dialog(parent, text: str, path=None):
    """The report, in a box the client can read and copy out of."""
    from PySide6.QtGui import QFont, QGuiApplication
    from PySide6.QtWidgets import (
        QDialog, QDialogButtonBox, QLabel, QPlainTextEdit, QPushButton,
        QVBoxLayout)

    dialog = QDialog(parent)
    dialog.setWindowTitle("Pose3D diagnostics")
    dialog.resize(820, 620)
    layout = QVBoxLayout(dialog)

    where = QLabel(f"Saved to:  {path}" if path is not None else
                   "This report could not be saved to a file. Copy it instead.")
    where.setWordWrap(True)
    layout.addWidget(where)

    view = QPlainTextEdit(text)
    view.setReadOnly(True)
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    view.setFont(font)
    layout.addWidget(view, 1)

    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    copy = QPushButton("Copy")
    copy.setToolTip("Copy the whole report so you can paste it into an email")
    copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(text))
    buttons.addButton(copy, QDialogButtonBox.ButtonRole.ActionRole)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    return dialog


def show_report(parent, text: str, path=None) -> None:
    """Build the dialog and run it modally."""
    build_dialog(parent, text, path).exec()


def _print(text: str) -> None:
    """Print the report to a console that may not be able to spell it."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding, "replace"),
              flush=True)


def _show(text: str, path) -> None:
    """The dialog, from the command line. A failure here costs the dialog and
    not the report: the console output is the deliverable."""
    global _QAPP
    import traceback
    try:
        from PySide6.QtWidgets import QApplication

        from pose3d.app import apply_dark_theme
        app = QApplication.instance()
        if app is None:
            _QAPP = app = QApplication(sys.argv)
        apply_dark_theme(app)
        show_report(None, text, path)
    except Exception:                     # noqa: BLE001 - see the docstring
        traceback.print_exc()


def main() -> int:
    """`--diagnose`: write the report, print it, show it. Always exits 0.

    The report is not a verdict — `--selftest` is, and its exit status is what
    the release gate reads. This one is a description, and a description that
    exits non-zero would be read as a second failure.
    """
    text = report()
    path = None
    try:
        path = write_report(text=text)
    except OSError as e:
        text += (f"\n(this report could not be saved: "
                 f"{type(e).__name__}: {e})\n")
    _print(text)
    _show(text, path)
    return 0

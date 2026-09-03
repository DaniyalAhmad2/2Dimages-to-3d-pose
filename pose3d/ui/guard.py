"""The one place a failure inside the GUI becomes something the user sees.

Qt swallows an exception raised inside a slot. The button appears to do
nothing, the traceback reaches only a log file nobody has been told about, and
the app carries on in whatever half-finished state the failure left behind.
Before this module exactly one slot — `_on_import` — guarded itself, with its
own bespoke dialog and its own wording.

Two things follow from routing every one of them through `report_error`.
The user gets the same dialog every time, and it names the log file they have
to send us. And the test suite can replace this single function (see the
autouse `recorded_errors` fixture in tests/conftest.py), so an unexpected
exception in a UI test can never park a modal dialog in front of a CI job with
nobody there to click it.
"""
from __future__ import annotations

import functools
import traceback

TITLE = "Pose3D hit a problem"


def log_hint() -> str:
    """Where to find the details, phrased for the person reporting the bug."""
    from pose3d.runtime import log_path

    log = log_path()
    if log is None:
        return ("No log file could be written on this machine, so there are "
                "no details to send. Run Pose3D-diagnose.exe from the same "
                "folder and send what it prints instead.")
    return f"Details were written to:\n{log}"


def report_error(parent, title: str, text: str) -> None:
    """Show one error to the user. THE only slot/job failure dialog.

    Deliberately thin: it composes nothing, so what a test records is exactly
    what the user would have read.
    """
    from PySide6.QtWidgets import QMessageBox

    QMessageBox.critical(parent, title, text)


def guarded(fn):
    """Decorate a slot so a failure inside it is reported, not swallowed.

    The decorated call returns None when it fails. `parent` for the dialog is
    the bound instance when it is a widget, so the box belongs to the window
    the user clicked in.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:                # not BaseException: Ctrl-C, exit
            traceback.print_exc()
            report_error(_parent(self), TITLE,
                         f"{fn.__name__} could not be completed.\n\n"
                         f"{type(e).__name__}: {e}\n\n{log_hint()}")
            return None
    return wrapper


def _parent(obj):
    try:
        from PySide6.QtWidgets import QWidget
    except ImportError:                       # pragma: no cover - no Qt build
        return None
    return obj if isinstance(obj, QWidget) else None

"""The one place a slot or a job failure becomes a message box.

Qt swallows exceptions raised inside a slot, so a failed handler used to look
like a button that does nothing and the traceback went only to the log, where
nobody is looking. Everything that can fail on the GUI side reports HERE, so
there is exactly one place to monkeypatch in a test and exactly one wording
for the user.
"""
from __future__ import annotations

import functools
import traceback


def report_error(parent, title: str, text: str) -> None:
    """Show `text` under `title`. The ONLY place a failure becomes a dialog."""
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.critical(parent, title, text)


def guarded(fn):
    """Wrap a slot so an exception is reported instead of being swallowed."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            # a module-level lookup, so a test that replaces report_error is
            # seen by every already-decorated slot
            report_error(self, "Something went wrong",
                         f"{type(e).__name__}: {e}")
    return wrapper

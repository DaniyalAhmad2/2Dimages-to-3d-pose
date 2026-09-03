"""One mechanism for every long job: off the GUI thread, with progress and cancel.

Detection, the face re-detect, the recompute, the import dialog's copy and
detection loops and the Blender export each used to have their own answer to
"how do I not freeze the window" — three of them had none at all, and Windows
painted "Not Responding" over a five-minute run with no way to stop it. They
now all go through `run_job`.

The job function is written as `fn(report, cancelled)`:

    report(done, total, label)   -> drives the progress dialog
    cancelled()                  -> True once the user pressed Cancel

and raises `Cancelled` when it sees that flag. `Cancelled` is
`pipeline.Cancelled` — one class, so `except Cancelled` on either side of the
UI/pipeline line catches the same thing — and it is not an error: `run_job`
hands it back to the caller without a message box.
"""
from __future__ import annotations

from PySide6.QtCore import QEventLoop, QObject, Qt, QThread, Signal
from PySide6.QtWidgets import QProgressDialog

from pose3d.pipeline import Cancelled  # noqa: F401  (re-exported by contract)


class Job(QThread):
    """Runs `fn(report, cancelled)` on a worker thread."""

    progress = Signal(int, int, str)       # done, total, label
    finished_res = Signal(object)          # the result, or the Exception raised

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the job to stop. Safe from either thread: it sets a flag."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            res = self._fn(self.progress.emit, self.is_cancelled)
        except Exception as e:             # including Cancelled
            res = e
        self.finished_res.emit(res)


class _Sink(QObject):
    """The GUI-thread end of a Job's signals.

    A QObject rather than plain callables on purpose: the job's signals are
    emitted from the worker thread, and Qt only queues them into this thread
    when the RECEIVER is an object living here. Connecting a bare function
    would run `setValue` on the worker thread, which is the class of bug this
    module exists to remove.
    """

    def __init__(self, dialog, loop):
        super().__init__()
        self._dialog = dialog
        self._loop = loop
        self.result = None

    def on_progress(self, done: int, total: int, label: str) -> None:
        self._dialog.setMaximum(int(total))
        self._dialog.setValue(int(done))
        if label:
            self._dialog.setLabelText(label)

    def on_finished(self, res) -> None:
        self.result = res
        self._loop.quit()


def run_job(parent, title: str, fn, cancellable: bool = True):
    """Run `fn(report, cancelled)` off the GUI thread and return its result.

    Shows a window-modal progress dialog for as long as the job runs, wires
    its Cancel button to `Job.cancel`, and spins a local event loop so the
    caller reads like a blocking call while the window stays alive. The return
    value is whatever `fn` returned, or the exception it raised — a real
    failure has already been reported through `guard.report_error` by then; a
    `Cancelled` has not, because the user asked for it.
    """
    from pose3d.ui import guard

    dialog = QProgressDialog(title, "Cancel" if cancellable else None,
                             0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumWidth(420)
    dialog.setMinimumDuration(0)
    # the dialog closes when the JOB ends, not when the bar happens to reach
    # its maximum — a job that reports 10/10 and then commits is not finished
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    if not cancellable:
        dialog.setCancelButton(None)

    loop = QEventLoop()
    job = Job(fn, parent)
    sink = _Sink(dialog, loop)
    job.progress.connect(sink.on_progress)
    job.finished_res.connect(sink.on_finished)
    # belt and braces: a job that ends without a result (a SystemExit, say)
    # would otherwise leave this loop spinning with the window frozen behind a
    # modal dialog — the exact failure mode this module removes.
    job.finished.connect(loop.quit)
    if cancellable:
        dialog.canceled.connect(job.cancel)

    dialog.show()
    job.start()
    loop.exec()
    job.wait()
    dialog.close()
    # This runs once per detection, re-detect, recompute, export and import
    # phase. Both objects are parented to the window, so without this the
    # session accumulates a dead QProgressDialog and a finished QThread per
    # job for as long as the window is open.
    dialog.deleteLater()
    job.deleteLater()

    res = sink.result
    if isinstance(res, Exception) and not isinstance(res, Cancelled):
        guard.report_error(parent, title, f"{type(res).__name__}: {res}")
    return res

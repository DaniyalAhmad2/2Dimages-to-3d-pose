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

from PySide6.QtCore import (
    QCoreApplication, QEventLoop, QObject, Qt, QThread, Signal,
)
from PySide6.QtWidgets import QProgressDialog

from pose3d.pipeline import Cancelled  # noqa: F401  (re-exported by contract)


# Finished jobs and their dialogs, held by a Python reference until it is safe
# to delete them. `deleteLater()` at the end of `run_job` is NOT safe on its
# own: the moment `run_job` returns, Python drops the last reference to the
# `job`/`sink` wrappers, and shiboken deletes the C++ objects immediately on
# garbage collection. A cross-thread `QMetaCallEvent` from the worker (a
# progress or finished signal, which PySide may route through a global receiver
# that `removePostedEvents(sink)` never reaches) can still be sitting in the
# queue; the top-level event loop then dispatches it into freed memory. That is
# the segfault (and the "pure virtual method called" abort) seen in
# `sendPostedEvents` right after an import ran detection.
#
# So a finished job is not deleted inside its own `run_job`. Its objects are
# reparented off the window (so the window's own teardown never races their
# deletion) and parked here, keeping the Python wrapper alive. They are deleted
# at the START of the NEXT `run_job`, by which point the application event loop
# has run and drained every stale metacall to the now-detached sink. A session
# that never starts another job keeps one parked set until it exits — a few
# small objects, freed by the OS at exit; the alternative is a crash.
_PARKED: list[QObject] = []


def _drain_parked() -> None:
    for obj in _PARKED:
        try:
            obj.deleteLater()
        except RuntimeError:
            pass                        # already gone with its window
    _PARKED.clear()


def _park(*objects: QObject) -> None:
    for obj in objects:
        try:
            obj.setParent(None)         # ours to delete now, not the window's
        except RuntimeError:
            continue
        _PARKED.append(obj)


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

    def __init__(self, dialog, loop, parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self._loop = loop
        self._painting = False
        self._done = False
        self.result = None

    def detach(self) -> None:
        """Stop listening, before the objects behind this sink go away.

        `run_job` calls this the moment the job is reaped, so a signal that
        was already queued when the loop quit is delivered into a slot that
        does nothing rather than into a dialog that has been closed and a
        loop that no longer exists.
        """
        self._done = True
        self._dialog = None
        self._loop = None

    def on_progress(self, done: int, total: int, label: str) -> None:
        # `QProgressDialog.setValue()` calls `QCoreApplication::processEvents()`
        # while the dialog is modal, so delivering one progress signal can
        # deliver the next one from inside this very call. Nothing needs that
        # recursion, and the C++ stack it builds is what a teardown further
        # down would otherwise unwind into.
        if self._done or self._painting or self._dialog is None:
            return
        self._painting = True
        try:
            self._dialog.setMaximum(int(total))
            self._dialog.setValue(int(done))
            if label:
                self._dialog.setLabelText(label)
        finally:
            self._painting = False

    def on_finished(self, res) -> None:
        if self._done:
            return
        self.result = res
        self._done = True
        if self._loop is not None:
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

    # delete the previous job's objects now that the event loop has drained
    # every stale cross-thread call still queued for them (see _PARKED).
    _drain_parked()

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
    # Parented to the job, so the sink is owned by Qt and dies with it on the
    # event loop's own terms. Left to Python it would be destroyed the instant
    # this function returns — with the job's queued signals still connected to
    # it, and a `QMetaCallEvent` for one of them possibly still in the queue.
    # That is a use-after-free, and it is what killed the app with a
    # segmentation fault inside `sendPostedEvents` after a failed import.
    sink = _Sink(dialog, loop, job)
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
    job.wait()                         # the thread is finished AND reaped

    # --- teardown, in this order, and before anything may re-enter ---------
    # `loop.exec()` returns on the FIRST of `finished_res` and `finished`, so
    # the other one — and any progress signal emitted just before them — can
    # still be sitting in this thread's posted-event queue. Take the wiring
    # apart and drop those events while the objects they name are all still
    # alive; every event loop that runs from here on (the error box below, the
    # dialogs the caller shows next, the main loop) would otherwise deliver
    # them into `sink` and `loop`, which this function is about to drop.
    sink.detach()
    job.progress.disconnect()
    job.finished_res.disconnect()
    job.finished.disconnect()
    if cancellable:
        dialog.canceled.disconnect()
    QCoreApplication.removePostedEvents(sink)
    QCoreApplication.removePostedEvents(job)
    QCoreApplication.removePostedEvents(loop)
    dialog.close()

    res = sink.result
    # The modal box comes after the job is reaped and unwired, and before
    # `deleteLater`: it runs a nested event loop, and a DeferredDelete posted
    # across one is a deletion racing whatever that loop runs.
    if isinstance(res, Exception) and not isinstance(res, Cancelled):
        guard.report_error(parent, title, f"{type(res).__name__}: {res}")
    # Do NOT delete here — a stale cross-thread metacall may still be queued
    # for `sink`, and deleting it now (Python GC on return, or deleteLater
    # racing that GC) is the use-after-free this parks around. `job` owns
    # `sink` as its child, so parking the two of them retires all three.
    _park(dialog, job)
    return res

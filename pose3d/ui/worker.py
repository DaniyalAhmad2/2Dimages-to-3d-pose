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
    QCoreApplication, QEvent, QEventLoop, QObject, Qt, QThread, Signal,
)
from PySide6.QtWidgets import QProgressDialog, QPushButton

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


class _JobDialog(QProgressDialog):
    """The progress dialog, with every way of dismissing it taken away.

    A job holds the window until it has really stopped — that is the whole
    point of the modal dialog — and Qt offers three ways out of a
    QProgressDialog, each of which used to hand the window back while the
    worker thread was still running:

    * the **Cancel button**, whose `clicked` Qt wires to `canceled`, which Qt
      wires to `QProgressDialog::cancel()` — `forceHide` plus `reset()`;
    * **Esc**, which arrives as a ShortcutOverride and becomes
      `QDialog::reject()` -> `done()` -> hide. With no Cancel button that
      happens without `canceled` being emitted at all, so the job is not even
      asked to stop — and Recompute 3D and Set Scale are exactly the jobs
      with no Cancel button;
    * the **title-bar close box**, whose `closeEvent` does emit `canceled`,
      but whose hide runs AFTER the handler returns, so re-showing from
      inside it is immediately undone.

    All three end in `on_stop` here, and nothing in this class ever hides the
    dialog: `run_job`'s teardown does, once, after `loop.exec()` has
    returned. Accepting the ShortcutOverride is what brings Esc back as an
    ordinary key press this class can swallow — Qt's documented way to take a
    shortcut over.

    `_job_running` in MainWindow is not a substitute: it keeps another JOB
    out, but not a joint edit, a frame change, a save or an Import, and those
    reach the same ProjectData the worker thread is rewriting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: What a dismissal means. `run_job` sets it and clears it in the
        #: teardown, so one queued behind that teardown reaches nothing.
        self.on_stop = None
        #: Only `run_job`'s teardown may close this dialog.
        self.allow_close = False

    def _stop(self) -> None:
        if self.on_stop is not None:
            self.on_stop()

    def event(self, e):
        if (e.type() == QEvent.Type.ShortcutOverride
                and e.key() == Qt.Key.Key_Escape):
            e.accept()              # comes back as a plain KeyPress, below
            return True
        return super().event(e)

    def keyPressEvent(self, event):
        # NOT super() for Escape: QDialog would `reject()` -> `done()` and
        # hide, which is the release this class exists to prevent.
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            self._stop()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.allow_close:
            super().closeEvent(event)
            return
        event.ignore()
        self._stop()


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
        self._label_frozen = False
        self.result = None

    def freeze_label(self) -> None:
        """Stop letting the job rewrite the dialog's label.

        Once the user has pressed Cancel the label says so, and a progress
        report still in flight — Blender goes on printing render lines for
        seconds after it is asked to stop — would put "Rendering the
        animation…" back over it.
        """
        self._label_frozen = True

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
            if label and not self._label_frozen:
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

    # The Cancel button is ours, not the one the constructor would make, so
    # that `_keep_blocking` below can take it away without deleting a widget
    # from inside its own `clicked` emission.
    dialog = _JobDialog(title, None, 0, 0, parent)
    cancel_button = QPushButton("Cancel") if cancellable else None
    dialog.setCancelButton(cancel_button)      # None: no button at all
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    dialog.setMinimumWidth(420)
    dialog.setMinimumDuration(0)
    # the dialog closes when the JOB ends, not when the bar happens to reach
    # its maximum — a job that reports 10/10 and then commits is not finished
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)

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

    def _keep_blocking():
        """Every way the user has of dismissing this dialog: the button, Esc,
        the close box.

        A job that has been asked to stop does not stop at once — an Export
        polls `cancelled`, then kills Blender, and that is seconds wide — so
        the dialog stays up and modal until `loop.exec()` returns, saying it
        is cancelling. `_JobDialog` is what makes that possible: none of the
        three paths hides it any more.

        The button goes away rather than staying live over a job that has
        already been asked to stop — hidden and disabled, not deleted: this
        runs inside the button's own `clicked` emission, and
        `setCancelButton(None)` would free the widget Qt is in the middle of.

        With no Cancel button there is nothing to ask: a recompute is one
        answer about the whole take. The window stays blocked all the same —
        that is the point — and the label says to wait rather than pretending
        the job was stopped.
        """
        if cancel_button is not None:
            cancel_button.setEnabled(False)
            cancel_button.hide()
            job.cancel()
        dialog.setLabelText(f"{title} — cancelling…" if cancellable
                            else f"{title} — please wait…")
        sink.freeze_label()              # a report in flight must not undo it

    # Esc and the close box, through `_JobDialog`; the button, here. NOT via
    # `canceled`: Qt wires that signal to `QProgressDialog::cancel()`, whose
    # whole job is the hide this module exists to prevent, so the wire from
    # the button to it is taken out and replaced with our own.
    dialog.on_stop = _keep_blocking
    if cancel_button is not None:
        cancel_button.clicked.disconnect()
        cancel_button.clicked.connect(_keep_blocking)

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
    # The dialog is PARKED, not deleted, so a press or a close queued behind
    # this teardown is delivered after it. Unwired here, it reaches nothing —
    # rather than asking a reaped `Job` to stop, or putting the dialog back
    # on screen over the window this function has just released.
    dialog.on_stop = None
    if cancel_button is not None:
        try:
            cancel_button.clicked.disconnect()
        except RuntimeError:
            pass                       # nothing left connected
    QCoreApplication.removePostedEvents(sink)
    QCoreApplication.removePostedEvents(job)
    QCoreApplication.removePostedEvents(loop)
    dialog.allow_close = True          # the one hide, and the only one
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

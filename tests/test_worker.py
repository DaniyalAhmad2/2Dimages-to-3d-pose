"""The one worker mechanism: progress, cancel, and where a failure surfaces.

Detection, face re-detect, recompute and the import dialog's detection loop
all used to run on the GUI thread, so Windows painted "Not Responding" over a
five-minute job and there was no way to stop it. They now go through
`pose3d.ui.worker`, and these are its three obligations: report every step,
stop when asked, and never swallow a failure.
"""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _run_to_completion(job, seconds=10):
    """Start `job` and spin an event loop until its thread finishes.

    `QThread.finished` is delivered to the thread the Job OBJECT lives in
    (here, the test's), so the loop quits from the right side; the watchdog
    keeps a broken worker from hanging the suite instead of failing it.
    """
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    job.finished.connect(loop.quit)
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    job.start()
    loop.exec()
    assert job.wait(1000), "the job's thread did not finish"


def test_a_job_reports_every_step(qapp):
    from pose3d.ui.worker import Job

    seen, results = [], []

    def count(report, cancelled):
        for i in range(10):
            if cancelled():
                raise AssertionError("nobody cancelled this job")
            report(i + 1, 10, f"item {i}")
        return "done"

    job = Job(count)
    job.progress.connect(lambda *a: seen.append(a))
    job.finished_res.connect(results.append)
    _run_to_completion(job)

    assert seen == [(i + 1, 10, f"item {i}") for i in range(10)]
    assert results == ["done"]


def test_cancelling_a_job_stops_it_where_it_was(qapp):
    """The job polls `cancelled()` and raises; the mechanism hands the
    exception back rather than a half-finished result."""
    from pose3d.ui.worker import Cancelled, Job

    at_three = threading.Event()
    seen, results = [], []

    def count(report, cancelled):
        for i in range(10):
            if cancelled():
                raise Cancelled()
            report(i + 1, 10, f"item {i}")
            if i == 2:                     # let the test press Cancel
                at_three.set()
                deadline = time.monotonic() + 10
                while not cancelled() and time.monotonic() < deadline:
                    time.sleep(0.01)
        return "done"

    job = Job(count)
    job.progress.connect(lambda *a: seen.append(a))
    job.finished_res.connect(results.append)

    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    job.finished.connect(loop.quit)
    QTimer.singleShot(10000, loop.quit)
    job.start()
    assert at_three.wait(10), "the job never reached its third step"
    job.cancel()
    loop.exec()
    assert job.wait(1000)

    assert len(seen) == 3, seen
    assert len(results) == 1 and isinstance(results[0], Cancelled)


def test_a_failing_job_is_reported_once_through_the_error_sink(
        qapp, recorded_errors):
    from pose3d.ui.worker import run_job

    def boom(report, cancelled):
        raise RuntimeError("the detector fell over")

    res = run_job(None, "Running detection", boom)

    assert isinstance(res, RuntimeError)
    assert len(recorded_errors) == 1
    title, text = recorded_errors[0]
    assert title == "Running detection"
    assert "the detector fell over" in text


def test_a_cancelled_job_is_not_an_error(qapp, recorded_errors):
    """Cancel is the user's own decision, not a fault to apologise for."""
    from pose3d.ui.worker import Cancelled, run_job

    def stopped(report, cancelled):
        raise Cancelled()

    res = run_job(None, "Running detection", stopped)

    assert isinstance(res, Cancelled)
    assert recorded_errors == []


def test_the_progress_dialogs_cancel_reaches_the_job(qapp, monkeypatch,
                                                     recorded_errors):
    """The wiring the Cancel button rides on: the dialog's `canceled` signal
    is the only thing that ever calls `Job.cancel`, and until this test
    nothing exercised it — every cancel in the suite called `job.cancel()`
    directly, so a dropped connection would have gone unnoticed."""
    from PySide6.QtCore import QTimer

    from pose3d.ui import worker

    dialogs = []
    real = worker.QProgressDialog
    monkeypatch.setattr(worker, "QProgressDialog",
                        lambda *a, **kw: dialogs.append(real(*a, **kw))
                        or dialogs[-1])

    def until_cancelled(report, cancelled):
        deadline = time.monotonic() + 10
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not cancelled():
            return "nobody pressed Cancel"
        raise worker.Cancelled()

    def press():
        if dialogs:
            dialogs[0].canceled.emit()
        else:
            QTimer.singleShot(10, press)

    QTimer.singleShot(0, press)
    res = worker.run_job(None, "Detection", until_cancelled)

    assert isinstance(res, worker.Cancelled)
    assert recorded_errors == []


def test_a_finished_job_is_retired_by_the_next_one_not_in_its_own_teardown(
        qapp, monkeypatch):
    """`run_job` is called once per detection, re-detect, recompute, export and
    import phase, and a QProgressDialog and a QThread kept alive per call is a
    leak the user pays for over a long session — so they are deleted.

    But NOT inside their own `run_job`: deleting them there (Python GC on
    return, or `deleteLater` racing that GC) frees the sink while a stale
    cross-thread metacall from the worker may still be queued for it, which is
    the segfault this whole module now parks around. Instead a finished job is
    parked, and the NEXT `run_job` deletes the previous one — by which point
    the application event loop has drained every stale call. So: one call
    leaves its objects alive but parked; a second call retires the first."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import Shiboken

    from pose3d.ui import worker

    # start from a clean parking lot regardless of test order
    worker._drain_parked()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    made = []
    real_dialog, real_job = worker.QProgressDialog, worker.Job
    monkeypatch.setattr(worker, "QProgressDialog",
                        lambda *a, **kw: made.append(real_dialog(*a, **kw))
                        or made[-1])
    monkeypatch.setattr(worker, "Job",
                        lambda *a, **kw: made.append(real_job(*a, **kw))
                        or made[-1])

    assert worker.run_job(None, "Detection",
                          lambda report, cancelled: "done") == "done"

    # after ONE call its objects are parked, not deleted: still alive.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    dialog1, job1 = made
    assert [Shiboken.isValid(o) for o in (dialog1, job1)] == [True, True], \
        "a job deleted in its own teardown is the use-after-free"

    # a SECOND call retires the first, and parks itself.
    assert worker.run_job(None, "Detection",
                          lambda report, cancelled: "done") == "done"
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert [Shiboken.isValid(o) for o in (dialog1, job1)] == [False, False], \
        "the previous job's objects must be gone once the next one has run"
    dialog2, job2 = made[2:]
    assert [Shiboken.isValid(o) for o in (dialog2, job2)] == [True, True]

    # and draining explicitly clears the last parked set (no session-end leak).
    worker._drain_parked()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert [Shiboken.isValid(o) for o in (dialog2, job2)] == [False, False]


def test_a_job_that_cannot_be_cancelled_still_runs(qapp):
    """Some phases cannot honour a Cancel — a whole-take recompute is one
    answer, not n. Those are shown without the button rather than with a dead
    one, and must otherwise behave identically."""
    from pose3d.ui.worker import run_job

    def work(report, cancelled):
        report(1, 2, "half way")
        return "done"

    assert run_job(None, "Recompute 3D", work, cancellable=False) == "done"


# --- the teardown that a segmentation fault was hiding in -------------------
#
# A user re-imported a project from its own images folder, the copy phase
# failed with `SameFileError`, and the app died with `Segmentation fault (core
# dumped)`. The core's backtrace is `sendPostedEvents -> notifyInternal2 ->
# QObject::event -> PySide6 -> 0xa1`: a queued signal delivered into a slot
# whose Python object had already been freed. `run_job` left the job wired to
# a `_Sink` that its own return destroys, opened a modal error box in the
# middle of that teardown, and `QProgressDialog.setValue()` pumps the event
# loop from inside the very slot that is being torn down.


class _FakeDialog:
    """Enough QProgressDialog for `_Sink`, with a call log."""

    def __init__(self, on_set_value=None):
        self.calls = []
        self._on_set_value = on_set_value

    def setMaximum(self, v):
        self.calls.append(("setMaximum", v))

    def setValue(self, v):
        self.calls.append(("setValue", v))
        if self._on_set_value is not None:
            self._on_set_value()           # what a modal setValue() really does

    def setLabelText(self, t):
        self.calls.append(("setLabelText", t))


class _FakeLoop:
    def __init__(self):
        self.quits = 0

    def quit(self):
        self.quits += 1


def test_a_report_arriving_after_the_job_was_reaped_touches_nothing(qapp):
    """The dangling delivery, in one call.

    Between `loop.exec()` returning and `run_job` returning, the dialog is
    closed and the sink is on its way out. A progress signal already in the
    queue is delivered after that — and used to drive a dialog that had been
    closed and deleted.
    """
    from pose3d.ui.worker import _Sink

    dialog, loop = _FakeDialog(), _FakeLoop()
    sink = _Sink(dialog, loop)
    sink.on_progress(1, 10, "half way")
    assert dialog.calls, "a live sink must drive the dialog"

    sink.detach()
    dialog.calls.clear()
    sink.on_progress(2, 10, "later")
    sink.on_finished("a late result")

    assert dialog.calls == []
    assert loop.quits == 0


def test_a_progress_report_cannot_re_enter_itself(qapp):
    """`QProgressDialog.setValue()` calls `processEvents()` while the dialog
    is modal, so delivering one progress signal can deliver the next one from
    inside the first. Nothing about the dialog needs that recursion, and the
    stack it builds is what a later teardown unwinds into."""
    from pose3d.ui.worker import _Sink

    depth, seen = [0], []

    def pump():
        # what processEvents() does from inside setValue(): deliver the next
        # queued progress signal into the same slot
        depth[0] += 1
        seen.append(depth[0])
        if depth[0] < 5:
            sink.on_progress(depth[0], 10, "deeper")
        depth[0] -= 1

    dialog, loop = _FakeDialog(on_set_value=pump), _FakeLoop()
    sink = _Sink(dialog, loop)
    sink.on_progress(0, 10, "start")

    assert seen == [1], f"on_progress re-entered itself: {seen}"


def test_the_error_box_opens_only_once_the_job_is_finished_and_unwired(
        qapp, monkeypatch):
    """The modal box runs an event loop of its own. Every posted event in the
    queue is delivered inside it, so by the time it opens the job must be
    finished and reaped, the dialog closed, and nothing left connected that
    could reach objects this function is about to drop."""
    from pose3d.ui import guard, worker

    made = {}
    real_dialog, real_job = worker.QProgressDialog, worker.Job
    monkeypatch.setattr(worker, "QProgressDialog",
                        lambda *a, **kw: made.setdefault(
                            "dialog", real_dialog(*a, **kw)))
    monkeypatch.setattr(worker, "Job",
                        lambda *a, **kw: made.setdefault(
                            "job", real_job(*a, **kw)))

    state = {}

    def record(parent, title, text):
        job, dialog = made["job"], made["dialog"]
        state["finished"] = job.isFinished()
        state["running"] = job.isRunning()
        state["visible"] = dialog.isVisible()
        state["connections"] = (job.receivers("2progress(int,int,QString)"),
                                job.receivers("2finished_res(PyObject)"))

    monkeypatch.setattr(guard, "report_error", record)

    def boom(report, cancelled):
        report(1, 2, "half way")
        raise RuntimeError("the copy fell over")

    res = worker.run_job(None, "Importing images", boom)

    assert isinstance(res, RuntimeError)
    assert state["finished"] is True and state["running"] is False
    assert state["visible"] is False
    assert state["connections"] == (0, 0), \
        "the job was still wired to the sink when the modal box opened"

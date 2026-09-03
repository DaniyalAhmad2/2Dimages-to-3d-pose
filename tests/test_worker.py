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


def test_a_finished_job_leaves_neither_a_dialog_nor_a_thread_behind(
        qapp, monkeypatch):
    """`run_job` is called once per detection, re-detect, recompute, export
    and import phase. A QProgressDialog and a QThread kept alive per call is a
    leak the user pays for over a long session."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import Shiboken

    from pose3d.ui import worker

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

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(made) == 2
    assert [Shiboken.isValid(obj) for obj in made] == [False, False]


def test_a_job_that_cannot_be_cancelled_still_runs(qapp):
    """Some phases cannot honour a Cancel — a whole-take recompute is one
    answer, not n. Those are shown without the button rather than with a dead
    one, and must otherwise behave identically."""
    from pose3d.ui.worker import run_job

    def work(report, cancelled):
        report(1, 2, "half way")
        return "done"

    assert run_job(None, "Recompute 3D", work, cancellable=False) == "done"

"""The window while a job runs, and the door back into a saved project.

Three things the client meets on the way through a session, none of which the
dashboard used to get right:

* a long job holds the window, and pressing Cancel does not hand it back
  early (see tests/test_worker.py for the mechanism; here it is the slots
  that must refuse to be re-entered);
* a cancelled export says what it removed, rather than asserting that nothing
  was written over files Blender had already put on disk;
* a project saved yesterday can be opened again today, from inside the app.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from tests.test_ui_smoke import _project_with_rig, _write_calibration


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class _Detector:
    head_source = "nose"


def _window(model=None):
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    if model is None:
        data, rig, _ = _project_with_rig()
        model = ProjectModel(data, rig)
    return MainWindow(model)


# --- one job at a time ------------------------------------------------------

def test_a_second_job_while_one_is_running_does_nothing_and_says_so(
        qapp, monkeypatch):
    """A progress dialog pumps the event loop from inside
    `QProgressDialog.setValue()`, and the cancel path now keeps the window
    blocked but still runs a local loop — so a click queued behind either can
    arrive while the first job is alive. It used to start a second detection
    against the same ProjectData, or a second Blender child writing the same
    output folder.
    """
    import pose3d.ui.main_window as main_window

    win = _window()
    win.detector = _Detector()
    win.model.redetect_all = lambda *a, **kw: None

    calls = []

    def reentering_run_job(parent, title, fn, cancellable=True):
        calls.append(title)
        if len(calls) == 1:
            win._on_run_detection()          # what the pumped loop delivers
        return None

    monkeypatch.setattr(main_window, "run_job", reentering_run_job)

    win._on_run_detection()

    assert calls == ["Detection"], calls
    assert "still running" in win.statusBar().currentMessage().lower()


@pytest.mark.parametrize("slot", ["_on_run_detection", "_on_redetect_head",
                                  "_on_recalibrate", "_on_diagnostics",
                                  "_on_export"])
def test_every_job_slot_is_closed_while_a_job_is_running(qapp, monkeypatch,
                                                         slot):
    """All five, not just the one with a test: the flag is what makes the
    window single-threaded about the project, and a slot that forgot to check
    it is the hole reopened."""
    import pose3d.ui.main_window as main_window
    from pose3d import diagnostics
    from pose3d.ui import filedialog

    win = _window()
    win.detector = _Detector()
    started = []
    monkeypatch.setattr(main_window, "run_job",
                        lambda *a, **kw: started.append(a[1]))
    # nothing here may reach a real picker or a real child process: a slot
    # that ignored the flag has to FAIL this test, not hang it
    monkeypatch.setattr(filedialog, "existing_directory",
                        lambda *a, **k: pytest.fail("the export ran"))
    monkeypatch.setattr(diagnostics, "run_in_child",
                        lambda **k: pytest.fail("the diagnostics ran"))
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda *a, **k: pytest.fail("a report was shown"))

    win._job_running = True
    getattr(win, slot)()

    assert started == [], f"{slot} started a job with one already running"
    assert "still running" in win.statusBar().currentMessage().lower()


def test_the_flag_is_cleared_even_when_the_job_raises(qapp, monkeypatch):
    """`run_job` hands a failure back rather than raising, but the model
    callbacks after it can raise — and a flag left set would lock every job
    slot for the rest of the session with no way to clear it."""
    import pose3d.ui.main_window as main_window

    win = _window()
    win.detector = _Detector()

    def boom(parent, title, fn, cancellable=True):
        raise RuntimeError("the dialog fell over")

    monkeypatch.setattr(main_window, "run_job", boom)

    win._on_run_detection()          # @guarded turns it into a report

    assert win._job_running is False


# --- the set-scale recompute is a job like the others -----------------------

def test_setting_the_scale_runs_through_the_job_mechanism(qapp, monkeypatch,
                                                          tmp_path):
    """`set_scale_from_height` rewrites the calibration folder and then
    re-triangulates and re-fits every frame. On the GUI thread, under nothing
    but a wait cursor, that is the "Not Responding" the worker module exists
    to remove — and `_on_recalibrate` already wraps the identical recompute.

    No Cancel: like the recompute, the rescale is one answer about the whole
    take, and a button that could only abort before the work started would be
    the dead control the export used to have.
    """
    import pose3d.ui.main_window as main_window
    from pose3d.ui.model import ProjectModel

    data, rig, _ = _project_with_rig()
    _write_calibration(tmp_path, rig)
    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    win = _window(model)
    target = model.measured_subject_height() * 2.0

    seen = {}

    def capturing_run_job(parent, title, fn, cancellable=True):
        seen["title"] = title
        seen["cancellable"] = cancellable
        # the model has to be quiet WHILE the job runs: it re-poses every
        # frame, and repainting the window once per frame from a worker
        # thread is both wasted and unsafe
        seen["quiet"] = model._quiet
        return fn(lambda *a: None, lambda: False)

    monkeypatch.setattr(main_window, "run_job", capturing_run_job)

    win._on_set_scale(target)

    assert seen["cancellable"] is False
    assert seen["quiet"] is True
    assert model._quiet is False, "the model was left quiet after the job"
    # and the answer the job computed is on screen, not just in the model
    assert model.measured_subject_height() == pytest.approx(target, rel=1e-3)
    assert win.saved_label.text().startswith("●")


def test_a_scale_that_cannot_be_applied_changes_nothing(qapp, monkeypatch):
    """`set_scale_from_height` returns None when there is nothing to measure
    or the typed distance is nonsense. That is not an edit, so the window must
    not mark the project unsaved over it."""
    import pose3d.ui.main_window as main_window

    win = _window()
    win.model.set_scale_from_height = lambda h: None
    monkeypatch.setattr(main_window, "run_job",
                        lambda parent, title, fn, cancellable=True:
                        fn(lambda *a: None, lambda: False))

    win._on_set_scale(-1.0)

    assert win.saved_label.text() == "✓ Project Saved"

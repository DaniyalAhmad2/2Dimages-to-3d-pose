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

from tests.test_ui_smoke import _project_with_rig


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

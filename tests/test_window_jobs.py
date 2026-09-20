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


# --- a stopped export tells the truth about the folder ----------------------

def _export_into(monkeypatch, out, writes, reason="blender_cancelled"):
    """Point the export at `out` and make Blender write `writes` then stop.

    `writes` is {suffix: text}; the real `blender_job` writes the BVH and the
    FBX before the long video render a Cancel usually interrupts, so this is
    what is genuinely on disk when `_failure` says nothing was written.
    """
    from pathlib import Path

    from pose3d.export import blender_export
    from pose3d.ui import filedialog

    monkeypatch.setattr(filedialog, "existing_directory",
                        lambda *a, **k: str(out))
    monkeypatch.setattr(filedialog, "is_writable", lambda p: True)

    def fake_export(poses, out_dir, name="pose3d", **kw):
        for suffix, text in writes.items():
            (Path(out_dir) / f"{name}{suffix}").write_text(text)
        return blender_export._failure(reason, "Blender was stopped.", 125)

    monkeypatch.setattr(blender_export, "export_animation", fake_export)


def test_a_cancelled_export_removes_the_files_it_wrote_and_names_them(
        qapp, tmp_path, monkeypatch, recorded_errors):
    """Blender writes the BVH and the FBX before the video render, and the
    Cancel lands during that render — minutes wide on the client's laptop. The
    app used to delete only `<name>_poses.json` and say "nothing was changed"
    over a complete .bvh and .fbx it had just put on disk, on top of whatever
    was there from the previous export.
    """
    out = tmp_path / "out"
    out.mkdir()
    _export_into(monkeypatch, out, {".bvh": "BVH", ".fbx": "FBX",
                                    "_poses.json": "{}"})

    win = _window()
    name = win.model.project.name
    win._on_export()

    assert sorted(p.name for p in out.iterdir()) == []
    message = win.statusBar().currentMessage()
    assert "Export cancelled" in message
    for expected in (f"{name}.bvh", f"{name}.fbx"):
        assert expected in message, message
    assert recorded_errors == [], "a cancel is not a failure to report"


def test_a_cancelled_export_leaves_an_earlier_delivery_alone(
        qapp, tmp_path, monkeypatch, recorded_errors):
    """Only what THIS run wrote. A file of the same name that the stopped
    export never touched is the user's previous delivery, and deleting it
    would make Cancel more destructive than the bug it fixes."""
    out = tmp_path / "out"
    out.mkdir()
    win = _window()
    name = win.model.project.name
    (out / f"{name}.bvh").write_text("YESTERDAY")
    _export_into(monkeypatch, out, {".fbx": "FBX"})

    win._on_export()

    assert (out / f"{name}.bvh").read_text() == "YESTERDAY"
    assert not (out / f"{name}.fbx").exists()
    message = win.statusBar().currentMessage()
    assert f"{name}.fbx" in message and f"{name}.bvh" not in message, message
    assert recorded_errors == []


def test_an_export_that_wrote_nothing_still_says_nothing_was_written(
        qapp, tmp_path, monkeypatch, recorded_errors):
    out = tmp_path / "out"
    out.mkdir()
    _export_into(monkeypatch, out, {})

    win = _window()
    win._on_export()

    assert "nothing was written" in win.statusBar().currentMessage()
    assert recorded_errors == []


def test_a_file_that_could_not_be_removed_is_reported_as_written(
        qapp, tmp_path, monkeypatch, recorded_errors):
    """Windows keeps a file open long enough that a delete fails, and a
    truncated FBX under the good file's name is the worst thing the folder
    can hold — so it is named, not silently counted as removed."""
    from pathlib import Path

    out = tmp_path / "out"
    out.mkdir()
    _export_into(monkeypatch, out, {".bvh": "BVH", ".fbx": "FBX"})

    real_unlink = Path.unlink

    def refuse(self, *a, **kw):
        if self.suffix == ".fbx":
            raise PermissionError("the file is open in another program")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", refuse)

    win = _window()
    name = win.model.project.name
    win._on_export()

    assert (out / f"{name}.fbx").exists(), "the test did not block the delete"
    assert len(recorded_errors) == 1, recorded_errors
    title, text = recorded_errors[0]
    assert f"{name}.fbx" in text and "could not be removed" in text
    assert f"{name}.bvh" in text, "what WAS removed has to be named too"


def test_a_timed_out_export_is_cleaned_up_and_still_reported_as_a_failure(
        qapp, tmp_path, monkeypatch, recorded_errors):
    """The idle deadline leaves the same half-written folder as a Cancel, but
    it is not something the user asked for: it keeps its dialog."""
    out = tmp_path / "out"
    out.mkdir()
    _export_into(monkeypatch, out, {".bvh": "BVH"}, reason="blender_timeout")

    win = _window()
    name = win.model.project.name
    win._on_export()

    assert not (out / f"{name}.bvh").exists()
    assert len(recorded_errors) == 1, recorded_errors
    assert f"{name}.bvh" in recorded_errors[0][1]


# --- the window swap after an import ----------------------------------------

class _FinishedImport:
    """An ImportDialog that has just created a project."""

    def __init__(self, folder):
        self.result_folder = str(folder)

    def exec(self):
        return 1


def test_an_import_hands_the_window_over_instead_of_leaking_it(
        qapp, tmp_path, monkeypatch):
    """Finishing an import opens the new project in a window of its own and
    closes this one. `app._WINDOWS` holds a reference for the life of the
    process and nothing ever removed it, so every import left a whole
    MainWindow + ProjectModel behind — and the replacement opened at the
    designed size, losing whatever the client had resized or maximised to.
    """
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import Shiboken

    from pose3d import app as app_module

    old = _window()
    old.resize(1200, 800)
    opened = []

    restored = []

    def open_callback(folder):
        new = _window()
        # what the geometry is restored FROM, rather than the size Qt ends up
        # choosing: `restoreGeometry` clamps to the screen it is replayed on,
        # and the offscreen one is smaller than any window under test
        new.restoreGeometry = lambda data: restored.append(bytes(data))
        opened.append((new, folder))
        app_module._WINDOWS.append(new)
        return new

    old.open_callback = open_callback
    app_module._WINDOWS.append(old)
    try:
        old._run_import_dialog(_FinishedImport(tmp_path))

        assert opened, "the import never opened the new project"
        new, folder = opened[0]
        assert folder == str(tmp_path)
        assert old not in app_module._WINDOWS, "the replaced window is held"
        assert new in app_module._WINDOWS
        # the layout the client was working at comes across
        assert restored == [bytes(old.saveGeometry())]

        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not Shiboken.isValid(old), (
            "the replaced window was closed but never deleted")
    finally:
        for win in (old, *(w for w, _ in opened)):
            if win in app_module._WINDOWS:
                app_module._WINDOWS.remove(win)


def test_an_import_with_nowhere_to_open_it_says_where_it_went(qapp, tmp_path):
    """A window built without the app's open route (the tests, and any
    embedding) must not delete itself over an import it cannot show."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import Shiboken

    win = _window()
    win.open_callback = None

    win._run_import_dialog(_FinishedImport(tmp_path))

    assert str(tmp_path) in win.statusBar().currentMessage()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert Shiboken.isValid(win), "it handed over to nothing"

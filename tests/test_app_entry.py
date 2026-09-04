"""The entry point, as the client's shortcut invokes it.

A double-clicked exe has no console, so every failure here is invisible unless
the app makes it visible itself.
"""
import os
import sys

import pytest

from pose3d import app


def test_selftest_flag_runs_the_checks_and_never_opens_a_window(monkeypatch):
    """The release workflow runs `Pose3D.exe --selftest` against the assembled
    bundle; if that fell through to the GUI it would hang the CI job."""
    called = {}

    def fake_selftest(argv):
        called["argv"] = argv
        return 3

    monkeypatch.setattr("pose3d.selftest.main", fake_selftest)
    opened = []
    monkeypatch.setattr(app, "open_project_window", opened.append)
    monkeypatch.setattr(sys, "argv", ["pose3d", "--selftest", "--no-video"])

    with pytest.raises(SystemExit) as e:
        app.main()
    assert e.value.code == 3
    assert called["argv"] == ["--no-video"]
    assert opened == [], "the GUI must not start"


def test_a_flag_is_not_mistaken_for_a_project_folder(monkeypatch):
    """sys.argv[1] used to be taken as the project path unconditionally, so
    any option would be looked up as a folder."""
    monkeypatch.setattr(sys, "argv", ["pose3d", "--no-video", "/tmp/proj"])
    seen = {}
    monkeypatch.setattr(app, "open_project_window",
                        lambda f: seen.setdefault("folder", f))
    monkeypatch.setattr(app, "apply_dark_theme", lambda a: None)

    class FakeQApp:                      # never start a real event loop here
        def __init__(self, argv):
            pass

        def exec(self):
            return 0

        def setStyle(self, _s):
            pass

        def setPalette(self, _p):
            pass

        def setStyleSheet(self, _s):
            pass

    monkeypatch.setattr(app, "QApplication", FakeQApp)
    with pytest.raises(SystemExit):
        app.main()
    assert seen["folder"] == "/tmp/proj"


def test_crash_handler_logs_and_says_where(monkeypatch, capsys):
    app.install_crash_handler()
    try:
        try:
            raise ValueError("rig went missing")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        err = capsys.readouterr().err
        assert "ValueError: rig went missing" in err
        assert "Traceback" in err
    finally:
        sys.excepthook = sys.__excepthook__


def test_crash_handler_leaves_keyboard_interrupt_alone(monkeypatch):
    app.install_crash_handler()
    try:
        seen = {}
        monkeypatch.setattr(sys, "__excepthook__",
                            lambda *a: seen.setdefault("n", 1))
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            sys.excepthook(*sys.exc_info())
        assert seen.get("n") == 1
    finally:
        sys.excepthook = sys.__excepthook__


def _crash(monkeypatch):
    """Fire the installed excepthook with a QApplication in the process."""
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: object()))
    app.install_crash_handler()
    try:
        try:
            raise ValueError("rig went missing")
        except ValueError:
            sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = sys.__excepthook__


def test_the_crash_dialog_goes_through_the_error_sink(monkeypatch,
                                                      recorded_errors):
    """The crash dialog is a modal like any other, so it has to be the same
    replaceable function — otherwise a crash in any UI test parks a message box
    in front of a CI job with nobody there to click it."""
    _crash(monkeypatch)

    assert [t for t, _ in recorded_errors] == ["Pose3D stopped"]
    assert "ValueError: rig went missing" in recorded_errors[0][1]


def test_the_crash_dialog_says_when_no_log_could_be_written(monkeypatch,
                                                            recorded_errors):
    """After the devnull fallback there is no file to send us, and a dialog
    naming one is worse than a dialog admitting there is none."""
    monkeypatch.setattr("pose3d.runtime.log_path", lambda: None)

    _crash(monkeypatch)

    text = recorded_errors[0][1]
    assert "None" not in text, f"the dialog points at a file called None: {text}"
    assert "no log" in text.lower()


# --- Qt's own warnings ------------------------------------------------------
#
# Qt writes "QOpenGLWidget: Failed to create context", "Could not load the Qt
# platform plugin" and every other diagnosis of the failures this build is
# most likely to hit to its own message handler, which by default goes to a
# console the client's windowed exe does not have.


def test_a_qt_warning_reaches_the_log(capsys):
    from PySide6.QtCore import qInstallMessageHandler, qWarning

    app.install_qt_message_handler()
    try:
        qWarning("no platform plugin")
    finally:
        qInstallMessageHandler(None)
    err = capsys.readouterr().err
    assert "[qt.warning] no platform plugin" in err


def test_the_handler_labels_the_severity(capsys):
    from PySide6.QtCore import (
        qCritical, qInstallMessageHandler, qWarning)

    app.install_qt_message_handler()
    try:
        qWarning("a warning")
        qCritical("a critical")
    finally:
        qInstallMessageHandler(None)
    err = capsys.readouterr().err
    assert "[qt.warning] a warning" in err
    assert "[qt.critical] a critical" in err


# --- software OpenGL --------------------------------------------------------
#
# Qt can only be told to use software OpenGL BEFORE the QApplication exists,
# so the decision is taken from the command line, an environment variable and
# a marker file next to the exe — never from anything the app learns later.


@pytest.fixture
def clean_gl_env(monkeypatch, tmp_path):
    """The three inputs `_configure_gl` reads, all neutral — and the process
    attribute it sets put back afterwards.

    `AA_UseSoftwareOpenGL` belongs to the process, not to the test: Qt keeps it
    for the rest of the run, so without this every GL-dependent test that
    happens to sort after this file would inherit a decision it never made.
    The assertion on the way in is what makes the restoration on the way out
    testable — remove the teardown and the second test in this section fails.
    """
    from PySide6.QtCore import QCoreApplication, Qt

    monkeypatch.delenv("POSE3D_GL", raising=False)
    monkeypatch.delenv("QT_OPENGL", raising=False)
    monkeypatch.setattr("pose3d.runtime.app_dir", lambda: tmp_path)
    attr = Qt.ApplicationAttribute.AA_UseSoftwareOpenGL
    assert not QCoreApplication.testAttribute(attr), \
        "software OpenGL was left on by an earlier test"
    yield tmp_path
    QCoreApplication.setAttribute(attr, False)   # False is the process default


def test_hardware_gl_is_the_default(monkeypatch, clean_gl_env):
    assert app._configure_gl([]) == "hardware"
    assert "QT_OPENGL" not in os.environ


def test_the_flag_selects_software_gl(monkeypatch, clean_gl_env):
    assert app._configure_gl(["--software-gl", "/tmp/proj"]) == "software"
    assert os.environ["QT_OPENGL"] == "software"


def test_the_environment_variable_selects_software_gl(monkeypatch, clean_gl_env):
    monkeypatch.setenv("POSE3D_GL", "software")
    assert app._configure_gl([]) == "software"


def test_the_marker_file_selects_software_gl(monkeypatch, clean_gl_env):
    """What the "Restart with software 3D" button leaves behind, so the choice
    survives the restart it triggers."""
    (clean_gl_env / "use-software-gl").write_text("", encoding="utf-8")
    assert app._configure_gl([]) == "software"


def _record_main_order(monkeypatch, name):
    """Run main() with the application object faked out, recording the order in
    which `name` and the QApplication construction happen.

    Asking `QApplication.instance()` instead would prove nothing: the fake never
    registers an instance, so that question answers None wherever the call sits.
    """
    order = []
    real = getattr(app, name)

    def spy(*args, **kwargs):
        order.append(name)
        return real(*args, **kwargs)

    monkeypatch.setattr(app, name, spy)

    class FakeQApp:
        def __init__(self, argv):
            order.append("QApplication")

        def exec(self):
            return 0

    monkeypatch.setattr(app, "QApplication", FakeQApp)
    monkeypatch.setattr(app, "open_project_window", lambda f: None)
    monkeypatch.setattr(app, "apply_dark_theme", lambda a: None)
    with pytest.raises(SystemExit):
        app.main()
    return order


def test_the_gl_decision_is_taken_before_any_qapplication(monkeypatch, clean_gl_env):
    """Qt ignores AA_UseSoftwareOpenGL once a QApplication exists, so this
    ordering IS the feature."""
    monkeypatch.setattr(sys, "argv", ["pose3d", "--software-gl"])

    assert _record_main_order(monkeypatch, "_configure_gl") == [
        "_configure_gl", "QApplication"]


def test_the_selftest_inherits_the_gl_decision(monkeypatch, clean_gl_env):
    """`Pose3D.exe --selftest --software-gl` is how the client's 3D fault gets
    diagnosed, so the self-test has to run under the same GL as the app."""
    seen = {}
    monkeypatch.setattr("pose3d.selftest.main",
                        lambda argv: seen.setdefault("gl",
                                                     os.environ.get("QT_OPENGL")) or 0)
    monkeypatch.setattr(sys, "argv", ["pose3d", "--selftest", "--software-gl"])
    with pytest.raises(SystemExit):
        app.main()
    assert seen["gl"] == "software"


# --- the seam Task C fills --------------------------------------------------


def test_pre_qt_checks_runs_before_the_qapplication(monkeypatch, clean_gl_env):
    """The bundle-integrity check has to be able to say "files are missing"
    and exit before Qt is asked to load a plugin that is not there."""
    monkeypatch.setattr(sys, "argv", ["pose3d"])

    assert _record_main_order(monkeypatch, "_pre_qt_checks") == [
        "_pre_qt_checks", "QApplication"]


def test_pre_qt_checks_is_not_reached_by_the_selftest(monkeypatch):
    """--selftest has its own report; it must not be stopped by the startup
    dialog that exists for the double-clicking client."""
    monkeypatch.setattr("pose3d.selftest.main", lambda argv: 0)
    monkeypatch.setattr(sys, "argv", ["pose3d", "--selftest"])
    called = []
    monkeypatch.setattr(app, "_pre_qt_checks", lambda: called.append(1))
    with pytest.raises(SystemExit):
        app.main()
    assert called == []


# --- the diagnose executable, as the client is told to run it ---------------

class _FakeQApp:
    """A QApplication that never runs an event loop, so a test that reaches
    the GUI branch fails instead of hanging the suite."""

    def __init__(self, argv):
        pass

    def exec(self):
        return 0

    def setStyle(self, _s):
        pass

    def setPalette(self, _p):
        pass

    def setStyleSheet(self, _s):
        pass


def _no_gui(monkeypatch):
    opened = []
    monkeypatch.setattr(app, "open_project_window", opened.append)
    monkeypatch.setattr(app, "_pre_qt_checks", lambda: None)
    monkeypatch.setattr(app, "apply_dark_theme", lambda a: None)
    monkeypatch.setattr(app, "QApplication", _FakeQApp)
    return opened


def test_the_diagnose_exe_with_no_arguments_writes_the_report(monkeypatch,
                                                              tmp_path):
    """README.txt tells the client to double-click `Pose3D-diagnose.exe`, and
    the whole support workflow is the file it writes. Built from the same
    script as `Pose3D.exe`, with no arguments it used to fall through to the
    GUI — in a console window, having diagnosed nothing."""
    ran = []
    monkeypatch.setattr("pose3d.diagnostics.main",
                        lambda: ran.append("diagnose") or 0)
    monkeypatch.setattr(sys, "argv", ["Pose3D-diagnose"])
    monkeypatch.setattr(sys, "executable",
                        str(tmp_path / "Pose3D-diagnose.exe"))
    opened = _no_gui(monkeypatch)

    with pytest.raises(SystemExit) as e:
        app.main()

    assert e.value.code == 0
    assert ran == ["diagnose"]
    assert opened == [], "the GUI must not start"


def test_the_double_clicked_diagnose_exe_keeps_its_window_open(monkeypatch,
                                                              tmp_path):
    """A double-clicked console exe closes its window the moment it returns,
    so the report the client was told to read scrolls past and vanishes."""
    monkeypatch.setattr("pose3d.diagnostics.main", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["Pose3D-diagnose"])
    monkeypatch.setattr(sys, "executable",
                        str(tmp_path / "Pose3D-diagnose.exe"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    _no_gui(monkeypatch)
    waited = []
    monkeypatch.setattr("builtins.input", lambda *a: waited.append(a) or "")

    with pytest.raises(SystemExit):
        app.main()

    assert len(waited) == 1


def test_the_ordinary_exe_with_no_arguments_still_starts_the_app(monkeypatch,
                                                                 tmp_path):
    """The dispatch is on the executable's own name, and only that."""
    ran = []
    monkeypatch.setattr("pose3d.diagnostics.main", lambda: ran.append(1) or 0)
    monkeypatch.setattr(sys, "argv", ["Pose3D"])
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Pose3D.exe"))
    opened = _no_gui(monkeypatch)

    with pytest.raises(SystemExit):
        app.main()

    assert ran == []
    assert opened == [None]

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


def test_the_crash_dialog_says_when_no_log_could_be_written(monkeypatch):
    """After the devnull fallback there is no file to send us, and a dialog
    naming one is worse than a dialog admitting there is none."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    monkeypatch.setattr("pose3d.runtime.log_path", lambda: None)
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: object()))
    seen = {}
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a: seen.setdefault("args", a)))

    app.install_crash_handler()
    try:
        try:
            raise ValueError("rig went missing")
        except ValueError:
            sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = sys.__excepthook__

    text = seen["args"][2]
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


def _clean_gl_env(monkeypatch, tmp_path):
    monkeypatch.delenv("POSE3D_GL", raising=False)
    monkeypatch.delenv("QT_OPENGL", raising=False)
    monkeypatch.setattr("pose3d.runtime.app_dir", lambda: tmp_path)


def test_hardware_gl_is_the_default(monkeypatch, tmp_path):
    _clean_gl_env(monkeypatch, tmp_path)
    assert app._configure_gl([]) == "hardware"
    assert "QT_OPENGL" not in os.environ


def test_the_flag_selects_software_gl(monkeypatch, tmp_path):
    _clean_gl_env(monkeypatch, tmp_path)
    assert app._configure_gl(["--software-gl", "/tmp/proj"]) == "software"
    assert os.environ["QT_OPENGL"] == "software"


def test_the_environment_variable_selects_software_gl(monkeypatch, tmp_path):
    _clean_gl_env(monkeypatch, tmp_path)
    monkeypatch.setenv("POSE3D_GL", "software")
    assert app._configure_gl([]) == "software"


def test_the_marker_file_selects_software_gl(monkeypatch, tmp_path):
    """What the "Restart with software 3D" button leaves behind, so the choice
    survives the restart it triggers."""
    _clean_gl_env(monkeypatch, tmp_path)
    (tmp_path / "use-software-gl").write_text("", encoding="utf-8")
    assert app._configure_gl([]) == "software"


def test_the_gl_decision_is_taken_before_any_qapplication(monkeypatch, tmp_path):
    """Qt ignores AA_UseSoftwareOpenGL once a QApplication exists, so this
    ordering IS the feature."""
    from PySide6.QtWidgets import QApplication

    _clean_gl_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["pose3d", "--software-gl"])
    monkeypatch.setattr(app, "open_project_window", lambda f: None)
    monkeypatch.setattr(app, "apply_dark_theme", lambda a: None)
    when = {}

    real = app._configure_gl

    def spy(argv):
        when["instance"] = QApplication.instance()
        return real(argv)

    monkeypatch.setattr(app, "_configure_gl", spy)

    class FakeQApp:
        def __init__(self, argv):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr(app, "QApplication", FakeQApp)
    with pytest.raises(SystemExit):
        app.main()
    assert "instance" in when, "_configure_gl was never called"
    assert when["instance"] is None, "a QApplication already existed"


def test_the_selftest_inherits_the_gl_decision(monkeypatch, tmp_path):
    """`Pose3D.exe --selftest --software-gl` is how the client's 3D fault gets
    diagnosed, so the self-test has to run under the same GL as the app."""
    _clean_gl_env(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr("pose3d.selftest.main",
                        lambda argv: seen.setdefault("gl",
                                                     os.environ.get("QT_OPENGL")) or 0)
    monkeypatch.setattr(sys, "argv", ["pose3d", "--selftest", "--software-gl"])
    with pytest.raises(SystemExit):
        app.main()
    assert seen["gl"] == "software"


# --- the seam Task C fills --------------------------------------------------


def test_pre_qt_checks_runs_before_the_qapplication(monkeypatch, tmp_path):
    """The bundle-integrity check has to be able to say "files are missing"
    and exit before Qt is asked to load a plugin that is not there."""
    from PySide6.QtWidgets import QApplication

    _clean_gl_env(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["pose3d"])
    monkeypatch.setattr(app, "open_project_window", lambda f: None)
    monkeypatch.setattr(app, "apply_dark_theme", lambda a: None)
    when = {}
    monkeypatch.setattr(app, "_pre_qt_checks",
                        lambda: when.setdefault("instance", QApplication.instance()))

    class FakeQApp:
        def __init__(self, argv):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr(app, "QApplication", FakeQApp)
    with pytest.raises(SystemExit):
        app.main()
    assert "instance" in when, "_pre_qt_checks was never called"
    assert when["instance"] is None


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

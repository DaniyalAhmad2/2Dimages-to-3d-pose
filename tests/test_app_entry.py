"""The entry point, as the client's shortcut invokes it.

A double-clicked exe has no console, so every failure here is invisible unless
the app makes it visible itself.
"""
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

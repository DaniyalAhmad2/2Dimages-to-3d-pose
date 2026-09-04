"""One file the client can send us, written by the app about itself.

Every question we have had to ask over email — which Windows, where did you
extract it, is it in OneDrive, what does the log say, does the 3D work — is a
line in this report, and it is produced by the same code on their machine and
in CI. The tests below are about the two properties that make it worth
anything: it answers those questions, and it never fails to be produced.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pose3d import diagnostics

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def fast_selftest(monkeypatch):
    """The transcript is part of the report; running Blender is not part of
    these tests."""
    monkeypatch.setattr("pose3d.selftest.run",
                        lambda video=True, out=None, **kw:
                        (out.write("  PASS  everything\n"), 0)[1])


def test_the_report_answers_the_questions_we_would_have_asked():
    text = diagnostics.report()
    for label in ("system", "python", "frozen", "app folder", "log file",
                  "install", "free disk", "code page", "OpenGL", "Blender",
                  "models", "bundle check", "self-test"):
        assert label in text, label
    assert "PASS  everything" in text, "the self-test transcript is the point"


def test_a_section_that_explodes_does_not_cost_the_report(monkeypatch):
    """A report that raises is a report nobody ever sees. Qt missing, Blender
    missing, a drive that will not answer: each costs its own line."""
    monkeypatch.setattr(diagnostics, "_opengl",
                        lambda: (_ for _ in ()).throw(ImportError("no PySide6")))
    text = diagnostics.report()
    assert "could not be determined" in text
    assert "no PySide6" in text
    assert "self-test" in text, "the rest of the report still has to be there"


def test_no_blender_is_a_sentence_not_a_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr("pose3d.config.blender_binary",
                        lambda: str(tmp_path / "no-such-blender"))
    assert "not found" in diagnostics.report()


def test_the_report_names_the_folders_that_break_the_app(monkeypatch, tmp_path):
    """OneDrive turns bundled files into placeholders that are not there,
    Program Files makes the install read-only (and os.access lie about it),
    and running from Downloads is what antivirus quarantines out from under."""
    one = tmp_path / "OneDrive - Contoso" / "Pose3D"
    one.mkdir(parents=True)
    monkeypatch.setenv("OneDriveCommercial", str(tmp_path / "OneDrive - Contoso"))
    monkeypatch.setattr("pose3d.runtime.app_dir", lambda: one)
    assert "OneDrive: yes" in diagnostics.report()

    plain = tmp_path / "Downloads" / "Pose3D"
    plain.mkdir(parents=True)
    monkeypatch.setattr("pose3d.runtime.app_dir", lambda: plain)
    text = diagnostics.report()
    assert "OneDrive: no" in text and "Downloads: yes" in text


def test_the_report_says_which_models_are_actually_there(monkeypatch):
    from pose3d.detect import models
    monkeypatch.setattr(models, "resolve", lambda *a, **k: None)
    text = diagnostics.report()
    assert "NOT FOUND" in text
    assert "searched" in text, "where it looked is half the answer"


# --- writing it out --------------------------------------------------------

def test_the_report_lands_beside_the_log_the_client_is_already_sent(
        monkeypatch, tmp_path):
    monkeypatch.setattr("pose3d.runtime.log_path",
                        lambda: tmp_path / "pose3d-log.txt")
    assert diagnostics.default_path() == tmp_path / "pose3d-diagnostics.txt"


def test_writing_the_report_does_not_run_it_twice(monkeypatch, tmp_path):
    """The dialog shows the text and the file holds it; producing it twice
    would run the self-test twice, Blender and all."""
    runs = []
    monkeypatch.setattr(diagnostics, "report",
                        lambda: runs.append(1) or "the report")
    path = diagnostics.write_report(tmp_path / "sub" / "out.txt",
                                    text="the report")
    assert path.read_text(encoding="utf-8") == "the report"
    assert runs == [], "the caller already had the text"


def test_writing_the_report_produces_one_when_it_was_not_given_one(tmp_path):
    path = diagnostics.write_report(tmp_path / "out.txt")
    assert "Pose3D diagnostics" in path.read_text(encoding="utf-8")


# --- the dialog and the command line ---------------------------------------

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_dialog_hands_over_the_whole_report_not_a_summary(qapp, tmp_path):
    """The client has to be able to send us this. A read-only text box they
    cannot copy out of would leave them retyping it."""
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QLabel, QPlainTextEdit, QPushButton

    dlg = diagnostics.build_dialog(None, "line one\nline two",
                                   tmp_path / "pose3d-diagnostics.txt")
    assert dlg.findChild(QPlainTextEdit).toPlainText() == "line one\nline two"
    assert any("pose3d-diagnostics.txt" in label.text()
               for label in dlg.findChildren(QLabel)), "where it was saved"

    copy = [b for b in dlg.findChildren(QPushButton) if b.text() == "Copy"]
    assert copy, [b.text() for b in dlg.findChildren(QPushButton)]
    copy[0].click()
    assert QGuiApplication.clipboard().text() == "line one\nline two"


def test_diagnose_writes_the_file_and_shows_it(monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr(diagnostics, "report", lambda emit=None: emit("REPORT"))
    monkeypatch.setattr(diagnostics, "_show",
                        lambda text, path: shown.append((text, path)))
    monkeypatch.setattr("pose3d.runtime.log_path",
                        lambda: tmp_path / "pose3d-log.txt")
    assert diagnostics.main() == 0
    written = tmp_path / "pose3d-diagnostics.txt"
    assert written.read_text(encoding="utf-8") == "REPORT"
    assert shown == [("REPORT", written)]


def test_a_report_that_cannot_be_saved_is_still_shown(monkeypatch, tmp_path):
    """Every reason the file cannot be written — a read-only install under
    Program Files, a full disk — is itself a thing worth reading."""
    shown = []
    monkeypatch.setattr(diagnostics, "report", lambda emit=None: emit("REPORT"))
    monkeypatch.setattr(diagnostics, "_show",
                        lambda text, path: shown.append((text, path)))
    monkeypatch.setattr(diagnostics, "_open_report_file",
                        lambda: (_ for _ in ()).throw(
                            OSError("read-only file system")))
    assert diagnostics.main() == 0
    text, path = shown[0]
    assert path is None and "read-only file system" in text


def test_the_console_report_survives_a_code_page_that_cannot_spell_it(
        monkeypatch):
    """The diagnose exe prints to whatever console the client has — cp1252 in
    Europe, cp932 in Japan. A UnicodeEncodeError here would lose the report to
    protect a dash."""
    monkeypatch.setattr(diagnostics, "report",
                        lambda emit=None: emit("café — 3D"))
    monkeypatch.setattr(diagnostics, "_open_report_file",
                        lambda: (_ for _ in ()).throw(OSError("nope")))
    monkeypatch.setattr(diagnostics, "_show", lambda text, path: None)

    class Narrow:
        encoding = "ascii"

        def write(self, text):
            text.encode("ascii")          # what the real console does
            return len(text)

        def flush(self):
            pass

    monkeypatch.setattr("sys.stdout", Narrow())
    assert diagnostics.main() == 0


def test_diagnose_runs_before_the_bundle_check_that_might_stop_the_app(
        monkeypatch):
    """A bundle with files missing is exactly when diagnostics are wanted, so
    the startup check must not get to exit first."""
    import sys

    from pose3d import app
    called = []
    monkeypatch.setattr("pose3d.diagnostics.main", lambda: 0)
    monkeypatch.setattr(app, "_pre_qt_checks", lambda: called.append("checks"))
    monkeypatch.setattr(app, "open_project_window", lambda f: called.append("gui"))
    monkeypatch.setattr(sys, "argv", ["pose3d", "--diagnose"])
    with pytest.raises(SystemExit) as e:
        app.main()
    assert e.value.code == 0
    assert called == [], "no startup check, no window"


# --- the fallback that needs neither the GUI nor a working exe -------------

DIAGNOSE_CMD = ROOT / "packaging" / "windows" / "Diagnose.cmd"


def test_the_cmd_fallback_runs_the_diagnostics_and_shows_the_log():
    """`--selftest` is a verdict and writes no file. What the client is asked
    to send is `pose3d-diagnostics.txt`, and `--diagnose` is the only thing
    that writes it."""
    text = DIAGNOSE_CMD.read_bytes().decode("ascii")
    assert "--diagnose" in text
    # the prose may explain the difference; no command line may run the other
    assert '.exe" --selftest' not in text
    assert "pose3d-diagnostics.txt" in text, "name the file to send"
    assert "pose3d-log.txt" in text, "the windowed exe prints only in there"
    assert "pause" in text, "double-clicked, it must not vanish"
    assert "\r\n" in text, "cmd.exe needs CRLF"


def test_the_cmd_fallback_looks_where_the_log_really_is():
    """`runtime._open_log()` falls back to %LOCALAPPDATA%\\Pose3D whenever the
    install folder is not writable — which is every install under Program
    Files, the case the fallback was written for. Printing only the local copy
    says "no log file" while the log exists."""
    text = DIAGNOSE_CMD.read_bytes().decode("ascii")
    assert "%LOCALAPPDATA%\\Pose3D" in text


def test_the_cmd_fallback_works_without_the_second_exe():
    """It is the fallback FOR the two-EXE build misbehaving, so it cannot
    require it."""
    text = DIAGNOSE_CMD.read_bytes().decode("ascii")
    assert "Pose3D-diagnose.exe" in text and "Pose3D.exe" in text


CLIENT_DOCS = (ROOT / "packaging" / "windows" / "README.txt",
               ROOT / "README.md")


def test_the_client_is_told_to_run_the_thing_that_writes_the_report():
    """The support workflow is only as good as its first instruction. It used
    to be "double-click Pose3D-diagnose.exe", which with no arguments started
    the GUI in a console window and wrote nothing."""
    for doc in CLIENT_DOCS:
        text = doc.read_text(encoding="utf-8")
        assert "Diagnose.cmd" in text, doc
        assert "pose3d-diagnostics.txt" in text, doc


def test_the_client_is_told_both_places_the_log_can_be():
    """`runtime._open_log()` falls back to %LOCALAPPDATA%\\Pose3D when the
    install folder is not writable, which is every install under Program
    Files. A doc that names only the folder next to the exe sends the client
    looking for a file that is somewhere else."""
    for doc in CLIENT_DOCS:
        text = doc.read_text(encoding="utf-8")
        assert "pose3d-log.txt" in text, doc
        assert "%LOCALAPPDATA%\\Pose3D" in text, doc


def test_the_bundle_script_ships_the_fallback():
    """A file only in the repository is no use to the client."""
    assert "Diagnose.cmd" in (
        ROOT / "tools" / "make_windows_bundle.ps1").read_text(encoding="utf-8")


# --- the route through the app --------------------------------------------

def _window(qapp):
    from pose3d.app import build_model
    from pose3d.ui.main_window import MainWindow
    return MainWindow(build_model(None))


def _diagnostics_action(win):
    """The Help > Diagnostics action, found through the window rather than by
    walking the menus: holding a QMenu wrapper here takes the menu (and its
    actions) with it when the wrapper is collected."""
    from PySide6.QtGui import QAction
    found = [a for a in win.findChildren(QAction) if "Diagnostics" in a.text()]
    return found[0] if found else None


def test_help_diagnostics_is_in_the_menu(qapp):
    """The 3D placeholder tells the user to go to Help > Diagnostics, so it
    has to be there."""
    win = _window(qapp)
    assert _diagnostics_action(win) is not None


def test_help_diagnostics_writes_the_report_and_shows_it(qapp, monkeypatch,
                                                         tmp_path):
    shown = []
    monkeypatch.setattr(diagnostics, "report", lambda: "REPORT")
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda parent, text, path: shown.append((text, path)))
    monkeypatch.setattr("pose3d.runtime.log_path",
                        lambda: tmp_path / "pose3d-log.txt")
    win = _window(qapp)
    _diagnostics_action(win).trigger()
    assert shown == [("REPORT", tmp_path / "pose3d-diagnostics.txt")]
    assert (tmp_path / "pose3d-diagnostics.txt").exists()


def test_a_report_that_cannot_be_saved_still_reaches_the_user(
        qapp, monkeypatch, recorded_errors):
    shown = []
    monkeypatch.setattr(diagnostics, "report", lambda: "REPORT")
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda parent, text, path: shown.append((text, path)))
    monkeypatch.setattr(diagnostics, "write_report",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError("Access is denied")))
    win = _window(qapp)
    _diagnostics_action(win).trigger()
    assert shown == [("REPORT", None)]
    assert recorded_errors, "the save failure is worth saying out loud"
    assert "Access is denied" in recorded_errors[0][1]


# --- a Qt that cannot start ------------------------------------------------
#
# The machine that needs this report is, more often than not, the machine
# where Qt itself is broken: the bundle is missing platforms/qwindows.dll, or
# that DLL is there but a dependency of it is not. Qt does not raise then. It
# calls qFatal(), which calls abort(): no exception, no `finally`, no file.

def test_the_gl_probe_is_not_run_when_starting_qt_would_abort(monkeypatch):
    """Asking about OpenGL means building a QApplication, and building one
    where the platform plugin will not load ends the process. The question is
    asked before the constructor, because there is no asking it after."""
    from pose3d import selftest
    monkeypatch.setattr(selftest, "qt_would_abort",
                        lambda: "there is no Qt platform plugin in /nowhere")
    line = diagnostics._opengl()
    assert "/nowhere" in line
    assert "abort" in line, "the report has to say why it did not look"


def test_the_sections_that_can_kill_the_process_come_last():
    """Nothing but Qt in here can end the process instead of raising, so the
    two sections that start Qt are ordered behind every fact that does not."""
    text = diagnostics.report()
    assert text.index("models") < text.index("OpenGL") < text.index("self-test")


def test_every_fact_is_handed_over_before_the_next_one_is_gathered(monkeypatch):
    """`report(emit)` is not a convenience: a report assembled in memory and
    saved at the end is lost in exactly the case it exists for."""
    emitted = []
    already = []
    real = diagnostics._models

    def models():
        already.extend(emitted)          # everything found before this one
        return real()

    monkeypatch.setattr(diagnostics, "_models", models)
    text = diagnostics.report(emitted.append)
    assert "".join(emitted) == text, "emitted piecemeal, whole at the end"
    early = "".join(already)
    assert "system" in early and "Blender" in early


def test_the_file_holds_what_was_found_before_the_process_died(
        monkeypatch, tmp_path):
    class Aborted(BaseException):
        """qFatal() in miniature: nothing catches it, nothing runs after."""

    def die(out):
        raise Aborted()

    monkeypatch.setattr(diagnostics, "_selftest", die)
    with pytest.raises(Aborted):
        diagnostics.write_report(tmp_path / "out.txt")
    text = (tmp_path / "out.txt").read_text(encoding="utf-8")
    for label in ("system", "install", "models", "OpenGL"):
        assert label in text, label


CHILD = """
import pathlib
import sys

from pose3d import diagnostics, selftest

selftest.run = lambda video=True, out=None, **kw: (
    out.write("  PASS  stubbed for the test\\n"), 0)[1]
diagnostics.default_path = lambda: pathlib.Path(sys.argv[1])
raise SystemExit(diagnostics.main())
"""


def test_a_qt_that_cannot_start_does_not_take_the_report_with_it(tmp_path):
    """The whole feature, against a real Qt that really cannot start.

    `QT_QPA_PLATFORM=nosuchplatform` is the client's broken bundle in
    miniature. This used to abort the interpreter (exit 134, core dumped)
    while gathering the OpenGL line, and every fact already found died with
    it: the Windows version, where the app was extracted, the free disk, the
    missing files and the self-test transcript that names the plugin.
    """
    out = tmp_path / "pose3d-diagnostics.txt"
    env = dict(os.environ, QT_QPA_PLATFORM="nosuchplatform",
               PYTHONPATH=str(ROOT))
    res = subprocess.run([sys.executable, "-c", CHILD, str(out)],
                         capture_output=True, text=True, timeout=300,
                         cwd=str(tmp_path), env=env)
    assert res.returncode == 0, f"{res.returncode}\n{res.stderr[-2000:]}"
    written = out.read_text(encoding="utf-8")
    assert "Pose3D diagnostics" in written
    assert "nosuchplatform" in written, "it says why there is no OpenGL line"
    assert "stubbed for the test" in written, "the transcript still got there"
    assert written in res.stdout, "and the console has it too"

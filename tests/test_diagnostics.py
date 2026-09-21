"""One file the client can send us, written by the app about itself.

Every question we have had to ask over email — which Windows, where did you
extract it, is it in OneDrive, what does the log say, does the 3D work — is a
line in this report, and it is produced by the same code on their machine and
in CI. The tests below are about the two properties that make it worth
anything: it answers those questions, and it never fails to be produced.
"""
import os
import re
import subprocess
import sys
import time
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


# --- the install instructions the client actually followed ----------------
#
# The client's launch failure is the one the milestone was rejected over, and
# the two documents below are what steered him into it: README.md sent him to
# "Desktop or Documents" while the README.txt inside the same zip said the
# opposite, and neither mentioned unblocking the download in its install steps
# — Unblock appeared only inside the write-up of the error it prevents, which
# is a page nobody reads until it is too late. On 2026-09-19 he wrote that he
# "followed the exact instructions" and still could not open the app.

#: Where each document stops instructing and starts troubleshooting. Anything
#: after this is read only once something has already gone wrong.
TROUBLESHOOTING_HEADING = {"README.txt": "If something goes wrong",
                           "README.md": "## Troubleshooting"}


def _install_steps(doc: Path) -> str:
    """The part of a client document that is followed BEFORE anything fails."""
    text = doc.read_text(encoding="utf-8")
    heading = TROUBLESHOOTING_HEADING[doc.name]
    cut = text.find(heading)
    assert cut > 0, f"{doc.name}: no {heading!r} section to cut at"
    return text[:cut]


def test_the_install_steps_say_to_unblock_the_downloaded_zip():
    """Windows tags a downloaded zip, the tag survives extraction, and a
    tagged `_internal\\` is one of the ways the "Failed to load Python DLL"
    dialog happens. It has to be step one, not a troubleshooting footnote."""
    for doc in CLIENT_DOCS:
        assert "Unblock" in _install_steps(doc), doc


def test_both_install_instructions_name_the_same_short_path():
    """They used to disagree: the zip's README.txt said `C:\\Pose3D`, the
    repository's README.md said Desktop or Documents. The client can only
    follow one of them."""
    for doc in CLIENT_DOCS:
        assert "C:\\Pose3D" in _install_steps(doc), doc


_ONEDRIVE_FOLDER = re.compile(r"(?<!Docker )\bDesktop\b|\bDocuments\b")
#: What makes naming the folder a warning rather than an instruction.
_NEGATION = re.compile(r"avoid|\bnot\b|never|\bout of\b|instead of", re.I)
#: What stops an earlier negation from governing the folder name: a contrast
#: ("not Program Files, BUT the Desktop") or the end of the sentence it was in.
_NEGATION_ENDS = re.compile(
    r"\bbut\b|\bhowever\b|\bthough\b|\bexcept\b|\brather than\b"
    r"|\binstead\b(?! of)|[.!?](\s|$)", re.I)


def _recommends_a_onedrive_folder(line: str) -> bool:
    """True if this line points the client AT Desktop or Documents.

    Naming them is fine — as the folders to keep out of. So each mention is
    read against the negation nearest in front of it, and only while that
    negation still governs: "somewhere writable — Desktop or Documents, not
    Program Files" has its "not" after the folder names, and "not Program
    Files, but the Desktop" has a contrast in between. Both send the client to
    the folder that breaks the app, and both are recommendations here.

    Deliberately strict in one direction: a negation that a line wrap has
    carried onto the previous line reads as a recommendation and fails the
    test, which costs a rewording. The other way round would ship the sentence
    the client followed into a launch failure.
    """
    for hit in _ONEDRIVE_FOLDER.finditer(line):
        before = line[:hit.start()]
        negations = list(_NEGATION.finditer(before))
        if not negations:
            return True
        if _NEGATION_ENDS.search(before[negations[-1].end():]):
            return True
    return False


#: The heuristic's own fixtures. The first line is what README.md really said
#: while the client could not open the app; the second is the "not X, but Y"
#: shape a whole-line or fixed-window check waves through.
READS_AS_A_RECOMMENDATION = (
    "extract it somewhere writable — Desktop or Documents, **not** Program Files —",
    "Do not put it in Program Files, but the Desktop is fine.",
    "Do not extract it to C:\\Pose3D. Put it on the Desktop.",
    "Extract it to Documents.",
)
READS_AS_A_WARNING = (
    "   we test. Do **not** extract to Desktop, Documents or any other folder",
    "   Do NOT run it from inside the .zip. Avoid Desktop, Documents and anything",
    "Avoid folders OneDrive syncs (often Desktop and Documents):",
    "do not put it on the Desktop or in Documents — those are the folders "
    "OneDrive syncs",
    "a short path on C: instead of the Desktop",
    "You need [Docker Desktop](https://www.docker.com/products/docker-desktop/).",
)


def test_the_rule_about_those_folders_reads_the_sentence_the_way_a_client_does():
    """The install steps are only as good as the check that holds them, so the
    check is tested on the sentences it exists to tell apart."""
    for line in READS_AS_A_RECOMMENDATION:
        assert _recommends_a_onedrive_folder(line), f"let through: {line!r}"
    for line in READS_AS_A_WARNING:
        assert not _recommends_a_onedrive_folder(line), f"flagged: {line!r}"


def test_neither_document_recommends_the_folders_that_break_it():
    """Desktop and Documents are the two folders OneDrive syncs by default,
    where "files on-demand" leaves placeholder stubs instead of the real
    DLLs."""
    for doc in CLIENT_DOCS:
        for line in doc.read_text(encoding="utf-8").splitlines():
            assert not _recommends_a_onedrive_folder(line), \
                f"{doc.name}: reads as a recommendation — {line.strip()!r}"


def test_both_install_instructions_are_the_same_steps_in_the_same_order():
    """One order, in both documents: unblock the download, extract it to the
    short path, then run the exe. Unblocking after extraction does nothing for
    the files already extracted."""
    for doc in CLIENT_DOCS:
        steps = _install_steps(doc)
        order = [steps.find(s) for s in ("Unblock", "C:\\Pose3D", "Pose3D.exe")]
        assert -1 not in order, f"{doc.name}: a step is missing {order}"
        assert order == sorted(order), f"{doc.name}: steps out of order {order}"


def test_the_zip_readme_does_not_ask_for_a_calibration_the_client_has_not_got():
    """Step 2 used to be "Pick the calibration for that camera setup". The
    client has never been given a calibration file and the bundle has no tool
    that makes one: the dialog's three Browse buttons are optional, and the
    one number he must set — the marker size — went unmentioned."""
    steps = _install_steps(ROOT / "packaging" / "windows" / "README.txt").lower()
    assert "pick the calibration" not in steps
    assert "marker size" in steps, "the one calibration input he does have"


# --- the documents we hand the client, and their source -------------------
#
# The getting-started PDF sent on 2026-09-04 existed nowhere in this
# repository: it was written before the UCRT was bundled and before
# Diagnose.cmd learned to check the bootloader files, so it describes neither
# the failure the client is seeing nor the diagnostic we now ask him for — and
# there was no source to correct. `docs/client/` is that source.

CLIENT_DIR = ROOT / "docs" / "client"
GUIDE_SOURCE = CLIENT_DIR / "content.json"
RELEASE_NOTES = CLIENT_DIR / "RELEASE_NOTES_v1.md"


def _guide_text() -> str:
    """Every line of prose in the guide, flattened — what the client reads."""
    import json

    content = json.loads(GUIDE_SOURCE.read_text(encoding="utf-8"))
    out = [content["title"], content["subtitle"], content.get("intro", "")]
    for section in content["sections"]:
        out.append(section["heading"])
        for block in section["blocks"]:
            out += [v for v in (block.get("p"), block.get("sub")) if v]
            out += block.get("bullets", []) + block.get("steps", [])
    return "\n".join(out)


def test_the_getting_started_guide_has_a_source_in_the_repository():
    """A document only the client has is a document we cannot correct."""
    assert GUIDE_SOURCE.is_file(), "the guide's content"
    assert (CLIENT_DIR / "build.js").is_file(), "what renders it"
    assert (CLIENT_DIR / "README.md").is_file(), "how to render it"
    assert "node build.js" in (CLIENT_DIR / "README.md").read_text(
        encoding="utf-8")
    assert _guide_text().strip(), "the guide parses and has prose in it"


def test_the_guide_asks_for_the_marker_size_in_centimetres():
    """It is the one number the client must type, and the box he types it into
    is centimetres."""
    marker_lines = [line for line in _guide_text().splitlines()
                    if "marker size" in line.lower()
                    or "Bigger is better" in line]
    assert marker_lines, "the guide never mentions the marker size"
    for line in marker_lines:
        assert "centimetre" in line.lower(), line
    assert "for example 8" in _guide_text(), "a worked number, not a unit note"


def test_the_guide_does_not_pin_the_download_to_one_build():
    """It names the build the client is to download, and a guide that says
    "build-17" is wrong the moment the resubmission build is published."""
    text = _guide_text()
    assert "newest release" in text
    assert "build-17" not in text


def test_the_guide_gives_the_same_install_steps_as_the_two_readmes():
    """Three documents reach the client — this one, `README.md` and the
    `README.txt` in the zip — and the launch failure came out of two of them
    disagreeing. Unblock, then `C:\\Pose3D`, then the exe, in all three."""
    text = _guide_text()
    order = [text.find(s) for s in ("Unblock", "C:\\Pose3D", "Pose3D.exe")]
    assert -1 not in order, f"a step is missing {order}"
    assert order == sorted(order), f"steps out of order {order}"
    for line in text.splitlines():
        assert not _recommends_a_onedrive_folder(line), line


#: Every problem the client raised, and the phrase the release note answers it
#: with. One line each, in his words rather than ours — this is the document
#: that goes back with the resubmission.
CLIENT_COMPLAINTS = {
    "P1 left/right inverted": "left and right",
    "P3 forward lean": "lean",
    "P3 clipping below the floor": "floor",
    "P5 export did not match the images": "export",
    "P6 distorted preview": "distort",
    "P8/P10 head and neck": "head and neck",
    "P11 joint handles too small": "handle",
    "P11 arrow keys": "arrow key",
    "P4 missing joints cannot be placed": "missing joint",
    "save and reopen a project": "reopen",
    "P12/P13 python312.dll launch error": "python312.dll",
    "P13 path too long": "path",
    "corrections kept across sessions": "correction",
    "marker size in centimetres": "centimetre",
    # build-17 feedback, 2026-09-21
    "P14 BVH imports lying flat in Blender": "upright",
    "P15 the accuracy dial's numbers are cut off": "accuracy dial",
    "P16 arrow keys even in a number box": "number box",
    "P17 does the program not detect toe position?": "toe",
}


def test_the_release_note_answers_every_complaint_the_client_made():
    text = RELEASE_NOTES.read_text(encoding="utf-8").lower()
    for complaint, phrase in CLIENT_COMPLAINTS.items():
        assert phrase in text, complaint


#: The GitHub release the note describes. Bump it with every resubmission, so
#: a note that still names the previous build fails here before it is sent.
RELEASE_BUILD = "build-21"


def test_the_release_note_names_the_build_that_carries_it():
    """A note that names the wrong build sends the client to the wrong zip."""
    text = RELEASE_NOTES.read_text(encoding="utf-8")
    assert "[build NN]" not in text
    assert text.splitlines()[0].endswith(RELEASE_BUILD)


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
    monkeypatch.setattr(diagnostics, "run_in_child",
                        lambda on_line=None, cancelled=None: ("REPORT", ""))
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda parent, text, path: shown.append((text, path)))
    monkeypatch.setattr("pose3d.runtime.log_path",
                        lambda: tmp_path / "pose3d-log.txt")
    # the child writes the file, as `--diagnose` does; this stands in for it
    (tmp_path / "pose3d-diagnostics.txt").write_text("REPORT", encoding="utf-8")
    win = _window(qapp)
    _diagnostics_action(win).trigger()
    assert shown == [("REPORT", tmp_path / "pose3d-diagnostics.txt")]


def test_a_report_that_could_not_be_saved_still_reaches_the_user(
        qapp, monkeypatch, tmp_path, recorded_errors):
    """A read-only install is itself half the diagnosis, and the report says
    so in its own text (the child writes "this report could not be saved").
    What must not happen is a dialog pointing at a file that is not there."""
    shown = []
    monkeypatch.setattr(
        diagnostics, "run_in_child",
        lambda on_line=None, cancelled=None: (
            "REPORT\n(this report could not be saved: OSError: Access is "
            "denied)\n", ""))
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda parent, text, path: shown.append((text, path)))
    monkeypatch.setattr("pose3d.runtime.log_path",
                        lambda: tmp_path / "pose3d-log.txt")
    win = _window(qapp)
    _diagnostics_action(win).trigger()
    assert len(shown) == 1
    assert "Access is denied" in shown[0][0]
    assert shown[0][1] is None, "no path is better than one that is not there"


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


# --- Help > Diagnostics runs somewhere else --------------------------------
#
# `diagnostics.report()` starts a real Blender export (timeout 600) and, on a
# machine with dead GL, a software-GL child (timeout 300) — up to a quarter of
# an hour of "Not Responding" on the one code path a client reaches when
# something is already wrong. And `check_qt_opengl` builds a View3D, shows it
# and calls processEvents twice, which from inside the live app is a stray
# window over the dashboard and every slot re-entered from the middle of this
# one. It belongs in a process of its own.

def test_the_child_command_is_the_diagnose_exe_when_frozen(monkeypatch,
                                                           tmp_path):
    exe = tmp_path / "Pose3D-diagnose.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Pose3D.exe"))

    assert diagnostics.child_command() == [str(exe), "--diagnose"]


def test_the_child_command_is_the_module_from_source(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert diagnostics.child_command() == [sys.executable, "-m",
                                           "pose3d.diagnostics"]


def _fake_child(tmp_path, body: str) -> list[str]:
    script = tmp_path / "child.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_the_report_comes_back_from_the_child_line_by_line(monkeypatch,
                                                           tmp_path):
    monkeypatch.setattr(
        diagnostics, "child_command",
        lambda: _fake_child(tmp_path,
                            "print('Pose3D diagnostics')\nprint('OpenGL: ok')\n"))
    lines = []

    text, stopped = diagnostics.run_in_child(on_line=lines.append)

    assert stopped == ""
    assert "Pose3D diagnostics" in text and "OpenGL: ok" in text
    assert lines == ["Pose3D diagnostics", "OpenGL: ok"]


def test_cancelling_the_diagnostics_kills_the_child_and_reaps_it(monkeypatch,
                                                                 tmp_path):
    """The Cancel button has to end the process, not orphan it: this is the
    child that launches Blender."""
    monkeypatch.setattr(
        diagnostics, "child_command",
        lambda: _fake_child(tmp_path,
                            "import time\nprint('starting', flush=True)\n"
                            "time.sleep(120)\n"))
    procs = []
    real_popen = diagnostics.subprocess.Popen
    monkeypatch.setattr(diagnostics.subprocess, "Popen",
                        lambda *a, **k: procs.append(real_popen(*a, **k))
                        or procs[-1])
    started = time.monotonic()

    text, stopped = diagnostics.run_in_child(cancelled=lambda: True)

    assert stopped == "cancelled"
    assert time.monotonic() - started < 20
    assert procs and procs[0].returncode is not None, "the child was orphaned"


def test_the_child_is_told_not_to_open_a_dialog_of_its_own(monkeypatch,
                                                           tmp_path):
    """`--diagnose` ends by showing the report in a Qt dialog. From a child of
    the running app that is a second window nobody asked for, in a process
    whose only job is to write the text back."""
    monkeypatch.setattr(
        diagnostics, "child_command",
        lambda: _fake_child(tmp_path,
                            "import os\nprint(os.environ.get('POSE3D_NO_DIALOG'))\n"))

    text, stopped = diagnostics.run_in_child()

    assert text.strip() == "1"


def test_the_dialog_is_skipped_when_the_parent_says_so(monkeypatch):
    shown = []
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda *a, **k: shown.append(a))
    monkeypatch.setenv("POSE3D_NO_DIALOG", "1")

    diagnostics._show("REPORT", None)

    assert shown == []


def test_help_diagnostics_runs_a_child_and_never_the_checks_here(
        qapp, monkeypatch, tmp_path):
    """Nothing GL-related, and no Blender, in the process holding the window."""
    win = _window(qapp)

    def refuse():
        raise AssertionError("the checks must not run in the GUI process")

    monkeypatch.setattr(diagnostics, "report", refuse)
    monkeypatch.setattr(diagnostics, "run_in_child",
                        lambda on_line=None, cancelled=None: ("REPORT\n", ""))
    shown = []
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda parent, text, path=None: shown.append((text, path)))

    win._on_diagnostics()

    assert shown and shown[0][0] == "REPORT\n"


def test_a_cancelled_diagnostics_run_says_nothing_and_shows_nothing(
        qapp, monkeypatch, recorded_errors):
    win = _window(qapp)
    monkeypatch.setattr(diagnostics, "run_in_child",
                        lambda on_line=None, cancelled=None: ("half a\n",
                                                              "cancelled"))
    shown = []
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda *a, **k: shown.append(a))

    win._on_diagnostics()

    assert shown == []
    assert recorded_errors == []


def test_a_timed_out_diagnostics_report_is_shown_as_partial(
        qapp, monkeypatch, recorded_errors):
    """`run_in_child` has three outcomes — finished, cancelled, timeout — and
    only the cancel was read, so a child killed at its 20-minute deadline had
    its half-written transcript shown and saved exactly like a complete one.

    That is the one fact that would redirect the whole investigation: a
    machine where the self-test genuinely stops at a section reads identically
    to one merely too slow to reach the rest.
    """
    win = _window(qapp)
    monkeypatch.setattr(diagnostics, "run_in_child",
                        lambda on_line=None, cancelled=None: ("half a\n",
                                                              "timeout"))
    shown = []
    monkeypatch.setattr(diagnostics, "show_report",
                        lambda parent, text, path=None: shown.append(text))

    win._on_diagnostics()

    assert shown, "the partial report is still worth reading"
    assert "half a" in shown[0], "the partial transcript was thrown away"
    assert "INCOMPLETE" in shown[0]
    assert "deadline" in shown[0]
    assert recorded_errors == []


def test_diagnose_cmd_checks_the_bootloader_files_without_any_python():
    """`Pose3D-diagnose.exe` is a Python program: when python312.dll will not
    load it dies with the very dialog it was asked to explain. So the batch
    file checks the files Windows loads BEFORE any Python — the interpreter,
    both halves of the C runtime — by itself, in pure cmd, and says which one
    is missing or empty. That has to come before it tries the exe."""
    text = DIAGNOSE_CMD.read_text(encoding="utf-8")
    before_exe = text.split('"Pose3D-diagnose.exe" --diagnose', 1)[0]
    for name in ("_internal\\python312.dll", "_internal\\vcruntime140.dll",
                 "_internal\\vcruntime140_1.dll", "_internal\\msvcp140.dll",
                 "_internal\\ucrtbase.dll", "api-ms-win-crt-"):
        assert name in before_exe, f"{name} is not checked before the exe runs"
    assert ":need" in text and "MISSING" in text and "EMPTY" in text
    # the two causes that are the machine's, not the folder's
    assert "Zone.Identifier" in before_exe, "Mark of the Web is not detected"
    assert "AMD64" in before_exe, "a 32-bit Windows is not detected"

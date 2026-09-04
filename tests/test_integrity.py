"""The bundle checking itself, on the client's machine, before Qt exists.

Every fault this catches has already happened to somebody: a file that was in
the zip and is not in the extracted folder, because an antivirus took it after
the fact; a folder inside OneDrive whose files are placeholders that were never
downloaded; an extraction that stopped part-way and said nothing. All three
look identical from inside the app — a DLL that will not load — and Windows
reports them with a dialog naming a file the client has never heard of.

So the app looks first, before it asks Qt for anything, using nothing but
`pathlib`: a missing `platforms\\qwindows.dll` kills the QApplication
constructor itself, so a Qt message box could never be the messenger. It
reports through stderr (which is the log) and a native MessageBoxW, and exits.

Two properties matter as much as the check: it may never fire on a healthy
bundle (frozen only, name and size only, `min_count` where the contents drift),
and a bug in it may never stop one — the whole thing is wrapped in a
try/except that logs and lets the app start.
"""
import json
import sys
import types
from pathlib import Path

import pytest

from pose3d import app, integrity, runtime
from tests.test_bundle_layout import MODEL_BYTES, fake_bundle

ROOT = Path(__file__).resolve().parent.parent

#: The real `_message_box`, bound before tests/conftest.py's autouse
#: `shown_message_boxes` fixture rebinds the module attribute. The two tests
#: that check the guard INSIDE the function have to call the unpatched one —
#: the same trick, for the same reason, as `report_error` in test_guard.py.
_real_message_box = integrity._message_box


class _Reached(BaseException):
    """Raised where the real function would reach into ctypes.

    Deliberately not an `Exception`: `_message_box` swallows those on purpose
    ("a dialog is a bonus"), so an AssertionError raised inside it would be
    caught and a test asserting nothing would pass.
    """


def _exploding_ctypes():
    """A ctypes stand-in whose `windll` — the door to the dialog — cannot be
    opened quietly.

    A real module object, and only that one name explodes. `import ctypes`
    reads `__spec__` off whatever is in `sys.modules`, and pytest's traceback
    machinery reads `__file__` off every module it walks, so a stand-in that
    answered everything with an exception would take the run down instead of
    failing the test.
    """
    module = types.ModuleType("ctypes")

    def missing(name):
        if name == "windll":
            raise _Reached("ctypes.windll was reached")
        raise AttributeError(name)

    module.__getattr__ = missing
    return module


@pytest.fixture
def bundle(tmp_path):
    """A complete extracted bundle, as the manifest describes one."""
    return fake_bundle(tmp_path / "Pose3D-Windows")


def frozen(monkeypatch, root):
    """Make the app believe it is the frozen bundle in `root`."""
    monkeypatch.setattr(runtime, "IS_FROZEN", True)
    monkeypatch.setattr(runtime, "app_dir", lambda: root)


# --- what it looks at -------------------------------------------------------

def test_a_source_checkout_has_nothing_to_check():
    """The manifest describes a Windows bundle. A developer running from a
    checkout has no _internal/, no blender/ and possibly no models/, and none
    of that is a fault — so the check has to be silent here, or nobody could
    run the app at all."""
    assert integrity.problems() == []


def test_a_complete_bundle_reports_nothing(monkeypatch, bundle):
    frozen(monkeypatch, bundle)
    assert integrity.problems() == []


def test_a_missing_qt_plugin_is_named_and_explained(monkeypatch, bundle):
    frozen(monkeypatch, bundle)
    (bundle / "_internal/PySide6/plugins/platforms/qwindows.dll").unlink()

    problems = integrity.problems()
    assert len(problems) == 1, problems
    assert "qwindows.dll" in problems[0]
    assert "QApplication" in problems[0], "the line has to say what breaks"

    text = integrity.explain(problems)
    assert "qwindows.dll" in text
    for cause in ("antivirus", "OneDrive", "extract"):
        assert cause.lower() in text.lower(), f"{cause} is not in:\n{text}"


def test_an_empty_model_file_is_a_problem_a_present_one_is_not(monkeypatch,
                                                               bundle):
    """The OneDrive placeholder: the name is there, the bytes are not."""
    frozen(monkeypatch, bundle)
    model = sorted((bundle / "models").glob("*.onnx"))[0]
    model.write_bytes(b"")

    problems = integrity.problems()
    assert len(problems) == 1, problems
    assert model.name in problems[0]


def test_startup_never_hashes_the_checkpoints(monkeypatch, bundle):
    """211 MB of SHA-256 on every launch would be seconds of nothing
    happening, to re-answer a question the release gate already answered. A
    file of the right name and size passes here by design."""
    frozen(monkeypatch, bundle)
    for model in (bundle / "models").glob("*.onnx"):
        model.write_bytes(b"?" * MODEL_BYTES)

    assert integrity.problems() == []


def test_only_the_rules_this_check_can_act_on_can_stop_a_launch(monkeypatch,
                                                                bundle):
    """python312.dll is loaded before any Python runs, so if it were missing
    this code would never execute — reporting it here is theatre. README.txt
    and the diagnose exe are worth a build failure and not worth refusing to
    start. Both kinds are in the manifest; neither is this module's business.
    """
    frozen(monkeypatch, bundle)
    (bundle / "_internal/python312.dll").unlink()
    (bundle / "Pose3D-diagnose.exe").unlink()
    (bundle / "README.txt").unlink()

    assert integrity.problems() == []


def test_the_bundle_and_the_manifest_can_both_be_pointed_elsewhere(tmp_path):
    """`root` and `manifest` are what the tests and the diagnose build use;
    neither needs the app to be frozen."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rules": [
        {"pattern": "there-is-no-such-file", "kind": "required",
         "stage": "pre-qt", "why": "a rule invented by this test"}]}),
        encoding="utf-8")

    problems = integrity.problems(root=tmp_path, manifest=manifest)
    assert len(problems) == 1
    assert "there-is-no-such-file" in problems[0]
    assert "a rule invented by this test" in problems[0]


def test_the_manifest_the_frozen_app_reads_is_the_one_the_spec_ships():
    """Frozen, `pose3d/` sits in `_internal/`, so this same expression
    resolves to `_internal/packaging/windows/manifest.json` — which is where
    the spec's datas entry puts it. Without that entry the check would raise
    on every launch of the real bundle (and be swallowed, and check nothing).
    """
    assert integrity.MANIFEST == ROOT / "packaging" / "windows" / "manifest.json"
    spec = (ROOT / "pose3d.spec").read_text(encoding="utf-8")
    assert '("packaging/windows/manifest.json", "packaging/windows")' in spec


# --- what it does about it --------------------------------------------------

def test_a_healthy_bundle_is_not_interrupted(monkeypatch, bundle, capsys):
    frozen(monkeypatch, bundle)
    assert integrity.run_startup_check() is None
    assert capsys.readouterr().err == ""


def test_a_broken_bundle_is_explained_on_stderr_and_stops(monkeypatch, bundle,
                                                          capsys):
    """stderr is the log the client is asked to send us, and the exit code is
    what stops a half-broken app from opening a window that will fail later,
    somewhere less legible."""
    frozen(monkeypatch, bundle)
    (bundle / "blender/blender.exe").unlink()

    with pytest.raises(SystemExit) as exc:
        integrity.run_startup_check()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "blender.exe" in err and "antivirus" in err.lower()


# --- the native box, which no test may ever actually draw -------------------

def test_no_test_can_draw_the_native_message_box():
    """The Windows suite stopped here for 42 minutes and was killed by the
    job's 45-minute cap. The test above fakes a frozen bundle, and on a real
    Windows host `runtime.IS_WINDOWS` is True as well — so the product drew
    its native MessageBoxW on a runner where nobody can press OK. Every test
    that reaches `run_startup_check()` with problems has that hazard, so the
    replacement is autouse in tests/conftest.py rather than opt-in here: the
    same protection `recorded_errors` already gives the Qt dialog.
    """
    assert integrity._message_box is not _real_message_box, (
        "tests/conftest.py's autouse shown_message_boxes fixture is not in "
        "place; a Windows run of this suite would block on a modal dialog")


def test_a_broken_bundle_says_the_same_thing_in_the_box_as_in_the_log(
        monkeypatch, bundle, capsys, shown_message_boxes):
    """Two channels, one text. The windowed build has no console, so the box
    is the only thing the client who double-clicked the exe will ever see;
    stderr is the copy they can send us. A box that said less than the log
    would send them back to us for what the log already answered."""
    frozen(monkeypatch, bundle)
    (bundle / "blender/blender.exe").unlink()

    with pytest.raises(SystemExit) as exc:
        integrity.run_startup_check()

    assert exc.value.code == 1, "a broken bundle still stops the launch"
    err = capsys.readouterr().err
    assert shown_message_boxes == [err.rstrip("\n")]


def test_the_native_box_is_inert_outside_a_frozen_app(monkeypatch):
    """The real function, with the fixture's recorder out of the way: from a
    checkout it must not so much as import ctypes, whatever the platform. This
    is what makes a developer's `python -m pose3d.app` on Windows safe, and
    what the autouse fixture above stands in for on a frozen fake."""
    monkeypatch.setitem(sys.modules, "ctypes", _exploding_ctypes())
    monkeypatch.setattr(runtime, "IS_WINDOWS", True)
    monkeypatch.setattr(runtime, "IS_FROZEN", False)

    assert _real_message_box("anything at all") is None


def test_the_native_box_is_what_the_frozen_windows_app_draws(monkeypatch):
    """...and the guard above is a guard, not dead code: with both flags set,
    the same call does reach ctypes. Without this the inertness test would
    pass just as well on a function that had stopped drawing anything."""
    monkeypatch.setitem(sys.modules, "ctypes", _exploding_ctypes())
    monkeypatch.setattr(runtime, "IS_WINDOWS", True)
    monkeypatch.setattr(runtime, "IS_FROZEN", True)

    with pytest.raises(_Reached, match="windll"):
        _real_message_box("files are missing")


def test_a_broken_checker_can_never_brick_a_healthy_bundle(monkeypatch,
                                                           capsys):
    """The whole point of the try/except: this check is new code running
    before anything else, on machines we cannot reach. A bug in it must cost
    the check, never the application."""
    def boom(*a, **kw):
        raise ValueError("a bug in the checker")

    monkeypatch.setattr(integrity, "problems", boom)
    assert integrity.run_startup_check() is None
    assert "a bug in the checker" in capsys.readouterr().err


def test_a_manifest_that_did_not_ship_does_not_stop_the_app(monkeypatch,
                                                            bundle, capsys):
    """The same fail-open rule, for the one file the check itself needs."""
    frozen(monkeypatch, bundle)
    monkeypatch.setattr(integrity, "MANIFEST", bundle / "no-manifest.json")

    assert integrity.run_startup_check() is None
    assert "no-manifest.json" in capsys.readouterr().err


def test_the_whole_message_survives_a_cp932_console(monkeypatch, bundle):
    """`Pose3D.exe` started from cmd.exe writes to a console encoded cp1252 in
    Europe and cp932 in Japan. One em dash there is a UnicodeEncodeError, which
    this module catches and treats as "the check could not run" — so the
    bundle it just found to be broken would start anyway. dark.qss:57 cost us
    this lesson once already."""
    frozen(monkeypatch, bundle)
    (bundle / "_internal/PySide6/plugins/platforms/qwindows.dll").unlink()

    integrity.explain(integrity.problems()).encode("ascii")


def test_the_check_runs_before_any_qapplication_exists(monkeypatch):
    """This ordering IS the feature: `platforms\\qwindows.dll` is loaded by the
    QApplication constructor, which aborts the process when it is missing, so
    a check placed after that line would never run on the bundle it exists for.

    Asking `QApplication.instance()` instead would prove nothing here: the fake
    never registers an instance, and a real one another test left behind would
    answer for the whole session.
    """
    order = []
    monkeypatch.setattr(integrity, "run_startup_check",
                        lambda: order.append("integrity"))

    class FakeQApp:
        def __init__(self, argv):
            order.append("QApplication")

        def exec(self):
            return 0

    monkeypatch.setattr(app, "QApplication", FakeQApp)
    monkeypatch.setattr(app, "apply_dark_theme", lambda a: None)
    monkeypatch.setattr(app, "open_project_window", lambda f: None)
    monkeypatch.setattr(sys, "argv", ["pose3d"])

    with pytest.raises(SystemExit):
        app.main()
    assert order == ["integrity", "QApplication"]

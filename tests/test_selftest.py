"""The self-test is what stands between a broken bundle and the client.

It ships in both deliveries, so its own accounting has to be right: a required
check that fails must make the exit status non-zero, and an optional one must
not.
"""
import io
from pathlib import Path

import pytest

from pose3d import selftest


def _run(monkeypatch, checks, strict=False):
    monkeypatch.setattr(selftest, "checks", lambda video=True: checks)
    out = io.StringIO()
    return selftest.run(out=out, strict=strict), out.getvalue()


def test_all_passing_exits_zero(monkeypatch):
    rc, text = _run(monkeypatch, [("a", lambda: "fine"), ("b", lambda: "")])
    assert rc == 0
    assert "All required checks passed." in text
    assert "fine" in text


def test_a_failed_check_is_reported_and_fails(monkeypatch):
    def boom():
        raise AssertionError("Blender 4.2 is too old")

    rc, text = _run(monkeypatch, [("blender", boom), ("other", lambda: "ok")])
    assert rc == 1
    assert "FAILED: blender" in text
    assert "too old" in text
    assert "  PASS  other" in text, "later checks must still run"


def test_a_degraded_check_warns_without_failing(monkeypatch):
    """The GPU-less CI runner cannot render the preview video. That must not
    block a release whose BVH and FBX — the data the client imports — are
    fine, but it must be visible rather than silent."""
    def no_video():
        raise selftest.Degraded("bvh 33KB, fbx 294KB; no preview video")

    rc, text = _run(monkeypatch, [("export", no_video)])
    assert rc == 0
    assert "WARN" in text
    assert "Degraded (not fatal): export" in text


def test_strict_mode_promotes_a_degraded_check_to_a_failure(monkeypatch):
    """The container ships Mesa so headless Blender can render; there a missing
    video means a broken image, and docker/build.sh must not go green on it."""
    def no_video():
        raise selftest.Degraded("Blender rendered no mp4")

    rc, text = _run(monkeypatch, [("video", no_video)], strict=True)
    assert rc == 1
    assert "FAIL" in text and "FAILED: video" in text


def test_require_video_env_var_turns_strictness_on(monkeypatch):
    monkeypatch.setenv("POSE3D_REQUIRE_VIDEO", "1")
    monkeypatch.setattr(
        selftest, "checks",
        lambda video=True: [("video", lambda: (_ for _ in ()).throw(
            selftest.Degraded("no mp4")))])
    assert selftest.run(out=io.StringIO()) == 1


def test_a_skipped_check_does_not_pass_silently(monkeypatch):
    def nope():
        raise selftest.Skip("no DISPLAY")

    rc, text = _run(monkeypatch, [("gl", nope)])
    assert rc == 0
    assert "SKIP" in text and "Skipped: gl" in text
    assert "  PASS  gl" not in text


def test_a_missing_3d_view_fails_even_where_gl_is_unavailable(monkeypatch):
    """The two GL failures must stay distinguishable. A GPU-less runner cannot
    provide a context — that degrades. The 3D view being absent from the bundle
    is a packaging defect and must fail even there, or the exact thing this
    check exists to catch would be waved through on every CI build."""
    import builtins
    monkeypatch.setenv("POSE3D_NO_GL", "1")
    monkeypatch.setenv("DISPLAY", ":0")
    real_import = builtins.__import__

    def no_view3d(name, *a, **k):
        if name == "pose3d.ui.view3d":
            raise ImportError("No module named 'pose3d.ui.view3d'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_view3d)
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_opengl()
    assert "missing from this build" in str(e.value)


def test_no_gpu_degrades_the_gl_check_instead_of_failing(monkeypatch):
    monkeypatch.setenv("POSE3D_NO_GL", "1")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(selftest, "_software_gl_child",
                        lambda: (False, "no frame either"))
    monkeypatch.setattr(
        "pose3d.ui.view3d.View3D",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("GLError 1282")))
    with pytest.raises(selftest.Degraded) as e:
        selftest.check_qt_opengl()
    assert "no usable OpenGL" in str(e.value)


def test_without_the_flag_a_dead_gl_context_is_still_fatal(monkeypatch):
    """On the client's machine `--selftest` is what diagnoses a black 3D view.

    It fails as an AssertionError rather than as the driver's own exception:
    by the time it is raised the check has also tried software OpenGL, and the
    message has to carry both answers."""
    monkeypatch.delenv("POSE3D_NO_GL", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(selftest, "_software_gl_child",
                        lambda: (False, "no frame either"))
    monkeypatch.setattr(
        "pose3d.ui.view3d.View3D",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("GLError 1282")))
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_opengl()
    assert "GLError 1282" in str(e.value)


def test_missing_weights_names_where_it_looked(monkeypatch):
    """A bundle assembled with the models folder in the wrong place is the
    likeliest packaging mistake; the message has to be actionable."""
    from pose3d.detect import models
    monkeypatch.setattr(models, "resolve", lambda *a, **k: None)
    with pytest.raises(AssertionError) as e:
        selftest.check_pose_weights_offline()
    assert "download" in str(e.value)
    assert "Looked in" in str(e.value)


def test_export_check_requires_bvh_and_fbx_not_just_a_zero_returncode(monkeypatch):
    """A green returncode with no files on disk is the failure mode that would
    otherwise ship."""
    class Res:
        ok = True
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr(
        "pose3d.export.blender_export.export_animation",
        lambda *a, **k: Res())
    with pytest.raises(AssertionError) as e:
        selftest.check_export()
    assert "bvh" in str(e.value) and "fbx" in str(e.value)


def test_a_failed_render_degrades_but_a_failed_export_does_not(monkeypatch):
    """These are one Blender invocation each on purpose. Writing BVH/FBX is
    arithmetic; rendering drives EEVEE through whatever GL exists, and on a
    loaded machine software GL can blow any timeout. A slow renderer must not
    be able to mask the fact that the animation data was fine."""
    class Timeout:
        ok = False
        returncode = 124
        stdout = ""
        stderr = "Blender did not finish within 900 s and was stopped."

    monkeypatch.setattr(
        "pose3d.export.blender_export.export_animation",
        lambda *a, **k: Timeout())

    with pytest.raises(selftest.Degraded):
        selftest.check_video()
    with pytest.raises(AssertionError):
        selftest.check_export()


def test_video_is_a_separate_check_that_can_be_skipped_entirely():
    names = [n for n, _ in selftest.checks(video=True)]
    assert "export BVH + FBX" in names and "render preview video" in names
    assert "render preview video" not in [n for n, _ in selftest.checks(video=False)]


def test_both_pose_models_must_be_in_the_build(monkeypatch):
    """The bundle has to carry the model the app RUNS, not just the one it
    used to run. Checking only one configuration is how a delivery could ship
    without the other and fall through to rtmlib's downloader on the client's
    machine — the exact failure detect/models.py exists to prevent."""
    from pose3d.detect import models

    asked = []

    def only_coco(mode="balanced", feet=False):
        asked.append(feet)
        return None if feet else models.Weights(
            det="yolox.onnx", det_input_size=(640, 640),
            pose="rtmpose.onnx", pose_input_size=(192, 256))

    monkeypatch.setattr(models, "resolve", only_coco)
    with pytest.raises(AssertionError) as e:
        selftest.check_pose_weights_offline()
    assert "Halpe-26" in str(e.value)
    assert True in asked


def test_a_failed_render_names_why_when_blender_merged_its_output(monkeypatch):
    """The export runs Blender through one launch path with stderr folded into
    stdout, so a failed render's diagnostic arrives on `stdout` and `stderr` is
    empty. Reading only `stderr` degrades the report to a bare returncode —
    "the client sees nothing", which is the failure class this check exists to
    remove. `check_export` already falls back; so must this one."""
    class Merged:
        ok = False
        returncode = 1
        stdout = "Error: EEVEE requires an OpenGL 3.3 context\nAborting\n"
        stderr = ""

    monkeypatch.setattr(
        "pose3d.export.blender_export.export_animation",
        lambda *a, **k: Merged())

    with pytest.raises(selftest.Degraded) as e:
        selftest.check_video()
    assert "EEVEE requires an OpenGL 3.3 context" in str(e.value)


# --- Task D: the checks a broken bundle cannot pass -------------------------

def test_check_blender_reads_the_version_from_anywhere_in_the_output(monkeypatch):
    """`ver.split()[1]` assumed the version was the second word of the first
    line. A Blender that prints a warning first — or a `blender` that is not
    Blender at all — made the check die with IndexError, which reads as a bug
    in the self-test rather than as a wrong Blender."""
    class Res:
        stdout = ("Warning: could not open display\n"
                  "Blender 5.1.1\n\tbuild date: 2026-01-01\n")
        stderr = ""

    monkeypatch.setattr(selftest.subprocess, "run", lambda *a, **k: Res())
    assert "5.1.1" in selftest.check_blender()


def test_check_blender_quotes_what_it_could_not_parse(monkeypatch):
    class Res:
        stdout = "'blender' is not recognized as an internal or external command\n"
        stderr = ""

    monkeypatch.setattr(selftest.subprocess, "run", lambda *a, **k: Res())
    with pytest.raises(AssertionError) as e:
        selftest.check_blender()
    assert "not recognized" in str(e.value), "the output has to be quoted back"


def test_check_blender_still_rejects_an_old_blender(monkeypatch):
    class Res:
        stdout = "Blender 4.2.1\n"
        stderr = ""

    monkeypatch.setattr(selftest.subprocess, "run", lambda *a, **k: Res())
    with pytest.raises(AssertionError, match="4.2.1"):
        selftest.check_blender()


# --- check_bundle_layout ---------------------------------------------------

def test_the_layout_check_is_skipped_in_a_source_checkout(monkeypatch):
    """A checkout has no _internal/, no blender/ and often no models/, and
    none of that is a fault."""
    monkeypatch.setattr(selftest, "IS_FROZEN", False)
    with pytest.raises(selftest.Skip):
        selftest.check_bundle_layout()


def test_a_frozen_build_that_cannot_find_its_manifest_FAILS(monkeypatch, tmp_path):
    """Not Skip. A frozen build whose manifest path is wrong checks nothing at
    startup, and failing open there is exactly the silence this whole change
    exists to remove — so the release gate has to catch it."""
    from pose3d import integrity
    monkeypatch.setattr(selftest, "IS_FROZEN", True)
    monkeypatch.setattr(integrity, "MANIFEST", tmp_path / "no-such-manifest.json")
    with pytest.raises(AssertionError) as e:
        selftest.check_bundle_layout()
    assert "no-such-manifest.json" in str(e.value)


def test_the_layout_check_reports_what_the_integrity_check_found(monkeypatch):
    from pose3d import integrity
    monkeypatch.setattr(selftest, "IS_FROZEN", True)
    monkeypatch.setattr(integrity, "problems",
                        lambda *a, **k: ["MISSING  models/*.onnx\n  why"])
    with pytest.raises(AssertionError) as e:
        selftest.check_bundle_layout()
    assert "models/*.onnx" in str(e.value)


def test_a_frozen_build_that_matches_its_manifest_passes(monkeypatch):
    from pose3d import integrity
    monkeypatch.setattr(selftest, "IS_FROZEN", True)
    monkeypatch.setattr(integrity, "problems", lambda *a, **k: [])
    assert "manifest.json" in selftest.check_bundle_layout()


# --- check_qt_plugins ------------------------------------------------------

def test_a_missing_qt_platform_plugin_is_named(monkeypatch, tmp_path):
    """platforms/qwindows.dll is loaded by name at run time, so no import scan
    can see it missing; the QApplication constructor aborts the process when it
    is, which is why nothing inside the app could ever report it."""
    monkeypatch.setattr(selftest, "IS_WINDOWS", True)
    monkeypatch.setattr(selftest, "_qt_plugin_dir", lambda: tmp_path)
    (tmp_path / "platforms").mkdir()
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_plugins()
    assert "qwindows.dll" in str(e.value)


def test_qt_must_be_able_to_read_the_photographs_the_client_imports(monkeypatch):
    """imageformats/qjpeg.dll is a plugin too. Without it every photograph in
    the app is a grey rectangle and nothing says why."""
    monkeypatch.setattr(selftest, "_qt_image_formats", lambda: {"png", "bmp"})
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_plugins()
    assert "jpg" in str(e.value)
    assert "png" not in str(e.value).split("Found")[0], "png is present"


def test_the_qt_plugin_check_passes_on_a_complete_build(monkeypatch, tmp_path):
    monkeypatch.setattr(selftest, "IS_WINDOWS", True)
    monkeypatch.setattr(selftest, "_qt_plugin_dir", lambda: tmp_path)
    (tmp_path / "platforms").mkdir()
    (tmp_path / "platforms" / "qwindows.dll").write_bytes(b"MZ")
    monkeypatch.setattr(selftest, "_qt_image_formats",
                        lambda: {"png", "jpg", "jpeg", "bmp"})
    assert "qwindows.dll" in selftest.check_qt_plugins()


# --- check_inference -------------------------------------------------------

class _FakeDetector:
    """RTMPoseDetector without onnxruntime, for the accounting tests."""
    def __init__(self, bundled=True, shape=None):
        self.bundled = bundled
        self._shape = shape

    def detect(self, image):
        import numpy as np

        from pose3d.core.skeleton import NUM_JOINTS
        n = NUM_JOINTS if self._shape is None else self._shape
        return type("Det", (), {"xy": np.zeros((n, 2)),
                                "scores": np.zeros(n)})()


def _weights_present(monkeypatch, present=True):
    from pose3d.detect import models
    monkeypatch.setattr(
        models, "resolve",
        lambda *a, **k: (models.Weights(det="d.onnx", det_input_size=(640, 640),
                                        pose="p.onnx", pose_input_size=(192, 256))
                         if present else None))


def test_the_inference_check_skips_in_a_checkout_without_weights(monkeypatch):
    """A developer checkout with no weights staged is not a broken build."""
    _weights_present(monkeypatch, False)
    monkeypatch.setattr(selftest, "IS_FROZEN", False)
    with pytest.raises(selftest.Skip):
        selftest.check_inference()


def test_a_frozen_bundle_without_weights_fails_without_downloading(monkeypatch):
    """Constructing the detector here would fall straight through to rtmlib's
    downloader — a network call, from the check that exists to prove the
    bundle needs none."""
    _weights_present(monkeypatch, False)
    monkeypatch.setattr(selftest, "IS_FROZEN", True)
    built = []
    monkeypatch.setattr("pose3d.detect.rtmpose.RTMPoseDetector",
                        lambda *a, **k: built.append(1))
    with pytest.raises(AssertionError):
        selftest.check_inference()
    assert built == [], "the downloader must not be reached"


def test_a_detector_that_fell_back_to_downloading_fails_the_check(monkeypatch):
    _weights_present(monkeypatch, True)
    monkeypatch.setattr("pose3d.detect.rtmpose.RTMPoseDetector",
                        lambda *a, **k: _FakeDetector(bundled=False))
    with pytest.raises(AssertionError) as e:
        selftest.check_inference()
    assert "bundled" in str(e.value)


def test_the_inference_check_reports_the_onnxruntime_it_ran_on(monkeypatch):
    """A bundle without onnxruntime detects nothing, and today that is found
    after delivery. The version and the providers are what tell us which."""
    _weights_present(monkeypatch, True)
    monkeypatch.setattr("pose3d.detect.rtmpose.RTMPoseDetector",
                        lambda *a, **k: _FakeDetector())
    import onnxruntime
    assert onnxruntime.__version__ in selftest.check_inference()


def test_a_detection_of_the_wrong_shape_fails(monkeypatch):
    _weights_present(monkeypatch, True)
    monkeypatch.setattr("pose3d.detect.rtmpose.RTMPoseDetector",
                        lambda *a, **k: _FakeDetector(shape=3))
    with pytest.raises(AssertionError) as e:
        selftest.check_inference()
    assert "shape" in str(e.value)


# --- check_qt_opengl: a cleared framebuffer is not a rendered frame ---------

def _frame(drawn=0, background=(14, 16, 22)):
    """A 100x100 readback with `drawn` pixels that are not the background."""
    from PySide6.QtGui import QImage, qRgb
    img = QImage(100, 100, QImage.Format.Format_RGB32)
    img.fill(qRgb(*background))
    for i in range(drawn):
        img.setPixel(i % 100, i // 100, qRgb(255, 255, 255))
    return img


class _FakeView:
    """A View3D whose framebuffer is whatever the test says it is."""
    def __init__(self, image):
        self._image = image
        self.opts = {"bgcolor": (14 / 255, 16 / 255, 22 / 255, 1.0)}

    def resize(self, *a): pass
    def show(self): pass
    def set_pose(self, pose): pass
    def isValid(self): return True
    def context(self): return self
    def grabFramebuffer(self): return self._image


@pytest.fixture
def gl_env(monkeypatch):
    """A machine that looks capable of 3D, with the child process stubbed."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("POSE3D_NO_GL", raising=False)
    monkeypatch.delenv("QT_OPENGL", raising=False)
    monkeypatch.delenv("POSE3D_GL", raising=False)
    calls = []

    def child():
        calls.append(1)
        return None, "stubbed"

    monkeypatch.setattr(selftest, "_software_gl_child", child)
    return calls


def test_a_frame_with_a_figure_in_it_passes(monkeypatch, gl_env):
    monkeypatch.setattr("pose3d.ui.view3d.View3D",
                        lambda *a, **k: _FakeView(_frame(drawn=400)))
    assert "100x100" in selftest.check_qt_opengl()
    assert gl_env == [], "a working view must not spawn anything"


def test_a_cleared_framebuffer_is_not_a_rendered_frame(monkeypatch, gl_env):
    """The check used to go green on any readback at all, so a driver that
    hands back a cleared buffer — the black 3D card the client reported —
    passed it. Fewer than 0.1 % of pixels differing from the background is not
    a drawing."""
    monkeypatch.setattr("pose3d.ui.view3d.View3D",
                        lambda *a, **k: _FakeView(_frame(drawn=5)))
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_opengl()
    assert "nothing drawn" in str(e.value)
    assert gl_env == [1], "the software path has to be tried, once"


def test_when_the_software_child_renders_the_check_says_so(monkeypatch):
    """Not a pass: hardware 3D is broken. Not a failure either: there is a way
    to run, and the client needs to be told which one."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("pose3d.ui.view3d.View3D",
                        lambda *a, **k: _FakeView(_frame(drawn=0)))
    monkeypatch.setattr(selftest, "_software_gl_child",
                        lambda: (True, "All required checks passed."))
    with pytest.raises(selftest.Degraded) as e:
        selftest.check_qt_opengl()
    assert "--software-gl" in str(e.value)


def test_when_both_paths_fail_it_is_fatal_unless_the_runner_says_otherwise(
        monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("pose3d.ui.view3d.View3D",
                        lambda *a, **k: _FakeView(_frame(drawn=0)))
    monkeypatch.setattr(selftest, "_software_gl_child",
                        lambda: (False, "FAILED: Qt + OpenGL 3D view"))
    monkeypatch.delenv("POSE3D_NO_GL", raising=False)
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_opengl()
    assert "software OpenGL also" in str(e.value)

    monkeypatch.setenv("POSE3D_NO_GL", "1")
    with pytest.raises(selftest.Degraded):
        selftest.check_qt_opengl()


def test_a_child_that_could_not_be_launched_is_untested_not_failed(monkeypatch):
    """"Software GL also failed" would be a diagnosis we did not make, and it
    is the one that sends the client to buy a graphics card."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("POSE3D_NO_GL", raising=False)
    monkeypatch.setattr("pose3d.ui.view3d.View3D",
                        lambda *a, **k: _FakeView(_frame(drawn=0)))
    monkeypatch.setattr(selftest, "_software_gl_child",
                        lambda: (None, "could not run Pose3D-diagnose.exe"))
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_opengl()
    assert "untested" in str(e.value)
    assert "also" not in str(e.value)


def test_the_software_retry_runs_the_diagnose_exe_when_frozen(monkeypatch):
    """One child, with the GL forced to software in its environment, and it
    has to be a build that can print — the windowed exe prints into the log."""
    seen = {}

    class Res:
        returncode = 0
        stdout = "  PASS  Qt + OpenGL 3D view\n"
        stderr = ""

    def run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw["env"]
        return Res()

    monkeypatch.setattr(selftest.subprocess, "run", run)
    monkeypatch.setattr(selftest, "IS_FROZEN", True)
    monkeypatch.setattr(selftest, "IS_WINDOWS", True)
    monkeypatch.setattr(selftest, "app_dir", lambda: Path("C:/Pose3D"))
    for var in ("QT_OPENGL", "POSE3D_GL"):
        monkeypatch.delenv(var, raising=False)

    ok, _ = selftest._software_gl_child()
    assert ok is True
    assert seen["cmd"][0].endswith("Pose3D-diagnose.exe")
    assert seen["cmd"][1:] == ["--selftest", "--gl-only", "--no-video"]
    assert seen["env"]["QT_OPENGL"] == "software"
    assert seen["env"]["POSE3D_GL"] == "software"


def test_the_software_retry_runs_this_module_from_a_checkout(monkeypatch):
    seen = {}

    class Res:
        returncode = 0
        stdout = "  PASS  Qt + OpenGL 3D view — a frame\n"
        stderr = ""

    def run(cmd, **kw):
        seen["cmd"] = cmd
        return Res()

    monkeypatch.setattr(selftest.subprocess, "run", run)
    monkeypatch.setattr(selftest, "IS_FROZEN", False)
    for var in ("QT_OPENGL", "POSE3D_GL"):
        monkeypatch.delenv(var, raising=False)
    assert selftest._software_gl_child()[0] is True
    assert seen["cmd"][1:] == ["-m", "pose3d.selftest", "--gl-only"]


@pytest.mark.parametrize("printed, verdict", [
    ("  PASS  Qt + OpenGL 3D view — rendered 320x240\n", True),
    ("  FAIL  Qt + OpenGL 3D view — no context\nFAILED: Qt + OpenGL\n", False),
    ("  WARN  Qt + OpenGL 3D view — no usable OpenGL\n", False),
    ("  SKIP  Qt + OpenGL 3D view — no DISPLAY\n", None),
    ("Traceback (most recent call last):\n", None),
])
def test_the_child_is_believed_only_when_it_says_PASS(monkeypatch, printed,
                                                      verdict):
    """Its exit status is not the answer: a check that SKIPs and one that
    degrades to a WARN both exit 0, and reading that as "software OpenGL
    works" would send the client to a --software-gl that renders nothing."""
    class Res:
        returncode = 0
        stdout = printed
        stderr = ""

    monkeypatch.setattr(selftest.subprocess, "run", lambda *a, **k: Res())
    monkeypatch.setattr(selftest, "IS_FROZEN", False)
    for var in ("QT_OPENGL", "POSE3D_GL"):
        monkeypatch.delenv(var, raising=False)
    assert selftest._software_gl_child()[0] is verdict


def test_the_child_does_not_spawn_a_child_of_its_own(monkeypatch):
    """The retry runs with QT_OPENGL=software; if it spawned another retry the
    self-test would fork until something ran out."""
    monkeypatch.setenv("QT_OPENGL", "software")
    ran = []
    monkeypatch.setattr(selftest.subprocess, "run",
                        lambda *a, **k: ran.append(1))
    ok, detail = selftest._software_gl_child()
    assert ok is None and ran == []
    assert "software" in detail


def test_gl_only_runs_the_gl_check_and_nothing_else(monkeypatch):
    """What the software-GL child is asked to do: one check, no Blender."""
    monkeypatch.setattr(selftest, "check_qt_opengl", lambda: "a frame")
    out = io.StringIO()
    assert selftest.run(out=out, gl_only=True) == 0
    text = out.getvalue()
    assert "a frame" in text
    assert "Blender" not in text and "export" not in text


def test_the_gl_only_flag_reaches_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(selftest, "run",
                        lambda **kw: seen.update(kw) or 0)
    selftest.main(["--gl-only", "--no-video"])
    assert seen["gl_only"] is True and seen["video"] is False


# --- a Qt that cannot start ------------------------------------------------
#
# Qt does not raise when it cannot load a platform plugin: it calls qFatal(),
# and qFatal() calls abort(). No exception is raised, no `finally` runs, no
# report is written — the process is simply gone (exit 134). Anything about to
# build the first QApplication in a process that must survive asks first.

def _platforms(tmp_path, *names):
    folder = tmp_path / "platforms"
    folder.mkdir()
    for name in names:
        (folder / name).write_bytes(b"MZ")
    return folder


def test_a_platform_qt_cannot_load_is_named_rather_than_walked_into(
        monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "nosuchplatform")
    monkeypatch.setattr(selftest, "_qt_plugin_dir", lambda: tmp_path)
    _platforms(tmp_path, "libqxcb.so", "libqoffscreen.so")
    problem = selftest.qt_platform_problem()
    assert "nosuchplatform" in problem
    assert "qxcb" in problem, "what it does have is half the answer"


def test_the_platform_this_process_asked_for_is_not_a_problem(
        monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(selftest, "_qt_plugin_dir", lambda: tmp_path)
    _platforms(tmp_path, "libqoffscreen.so")
    assert selftest.qt_platform_problem() is None


def test_windows_needs_its_own_plugin_even_when_nothing_asked_for_one(
        monkeypatch, tmp_path):
    """The client's failure: nothing sets QT_QPA_PLATFORM, Qt looks for
    qwindows.dll, and a bundle without it aborts before the first window."""
    monkeypatch.setattr(selftest, "IS_WINDOWS", True)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(selftest, "_qt_plugin_dir", lambda: tmp_path)
    _platforms(tmp_path, "qminimal.dll")
    assert "windows" in selftest.qt_platform_problem()


def test_no_platform_plugin_at_all_is_the_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(selftest, "_qt_plugin_dir", lambda: tmp_path)
    _platforms(tmp_path)
    assert "no Qt platform plugin" in selftest.qt_platform_problem()


def test_a_qt_that_is_already_running_found_its_plugin(monkeypatch):
    """Whatever the environment says now, this process has a QApplication, so
    its constructor is not going to abort anything."""
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(selftest, "qt_platform_problem", lambda: "nonsense")
    assert selftest.qt_would_abort() is None


def test_the_gl_check_does_not_build_a_qapplication_that_would_abort(
        monkeypatch, gl_env):
    """A platform plugin that will not load is a packaging fault, not a GL
    one. The software-GL child would abort in exactly the same way, so it is
    not spawned, and the check says what is actually wrong."""
    monkeypatch.setattr(selftest, "qt_would_abort",
                        lambda: "there is no Qt platform plugin in /nowhere")
    with pytest.raises(AssertionError) as e:
        selftest.check_qt_opengl()
    assert "/nowhere" in str(e.value)
    assert gl_env == [], "no child either: it would abort too"

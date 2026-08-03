"""The self-test is what stands between a broken bundle and the client.

It ships in both deliveries, so its own accounting has to be right: a required
check that fails must make the exit status non-zero, and an optional one must
not.
"""
import io

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

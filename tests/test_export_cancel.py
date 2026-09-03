"""A hung Blender export can be stopped — and the child really dies.

The timeout used to be inoperative on the path the GUI actually took: it
reached only the `subprocess.run` branch, while `_on_export` always streamed,
where `for line in p.stdout` blocks forever and nothing killed the child. The
progress dialog had `setCancelButton(None)` because there was nothing behind
it. There is now ONE launch path, with an idle deadline, an overall deadline
and a `cancelled` poll, and each of the three kills and reaps the process.

Runs on Linux without Blender: the export's own `Popen` is redirected to a
Python script that behaves like a Blender that hangs, or one that answers.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from pose3d.core.skeleton import NUM_JOINTS
from pose3d.export import blender_export

# a Blender that says one thing and then goes quiet forever — the OneDrive /
# antivirus / broken-GL stall the client hit
HANGS = """import time
print("Blender 5.1.1 starting", flush=True)
time.sleep(300)
"""

# ... and one that gets on with it
ANSWERS = """for i in range(3):
    print(f"Fra:{i + 1} Mem:12M", flush=True)
print("POSE3D_EXPORT_OK", flush=True)
"""


def _script(tmp_path, body, name="fake_blender.py"):
    path = tmp_path / name
    path.write_text(body)
    return path


def _spy(monkeypatch, script):
    """Redirect the export's Popen to `python <script>`, recording every real
    process object it starts — so the test can ask what became of the child
    without ever calling poll() or wait() itself."""
    procs = []
    real = subprocess.Popen

    def fake(cmd, **kw):
        p = real([sys.executable, str(script)], **kw)
        procs.append(p)
        return p

    monkeypatch.setattr(blender_export.subprocess, "Popen", fake)
    return procs


def _poses():
    poses = np.zeros((1, NUM_JOINTS, 3))
    poses[:, :, 2] = np.linspace(0, 1.7, NUM_JOINTS)
    return poses


def test_an_export_that_goes_quiet_is_timed_out_and_the_child_killed(
        tmp_path, monkeypatch):
    procs = _spy(monkeypatch, _script(tmp_path, HANGS))

    started = time.monotonic()
    res = blender_export.export_animation(
        _poses(), tmp_path / "out", name="t", render_video=False,
        character=None, blender="blender", idle_timeout=2, timeout=600)
    elapsed = time.monotonic() - started

    assert res.reason == "blender_timeout", res.reason
    assert elapsed < 20, f"took {elapsed:.1f}s to give up on a 2s idle deadline"
    assert len(procs) == 1
    assert procs[0].returncode is not None, "the child was never reaped"
    if os.name == "posix":
        assert procs[0].returncode < 0, "the child exited on its own, unkilled"
    # what it did manage to say is kept: the reader is drained before we leave
    assert "Blender 5.1.1 starting" in res.stdout


def test_cancelling_an_export_stops_it(tmp_path, monkeypatch):
    procs = _spy(monkeypatch, _script(tmp_path, HANGS))

    started = time.monotonic()
    res = blender_export.export_animation(
        _poses(), tmp_path / "out", name="c", render_video=False,
        character=None, blender="blender", idle_timeout=600, timeout=600,
        cancelled=lambda: time.monotonic() - started > 1)
    elapsed = time.monotonic() - started

    assert res.reason == "blender_cancelled", res.reason
    assert elapsed < 20, f"the cancel took {elapsed:.1f}s to be noticed"
    assert procs[0].returncode is not None, "the child was never reaped"
    if os.name == "posix":
        assert procs[0].returncode < 0


def test_an_export_that_answers_is_streamed_and_succeeds(tmp_path, monkeypatch):
    procs = _spy(monkeypatch, _script(tmp_path, ANSWERS))

    lines = []
    res = blender_export.export_animation(
        _poses(), tmp_path / "out", name="a", render_video=False,
        character=None, blender="blender", idle_timeout=30, timeout=60,
        on_line=lines.append)

    assert res.ok, f"rc={res.returncode}\n{res.stdout}"
    assert procs[0].returncode == 0
    assert lines == ["Fra:1 Mem:12M", "Fra:2 Mem:12M", "Fra:3 Mem:12M",
                     "POSE3D_EXPORT_OK"]


def test_the_launch_defaults_are_unchanged_for_todays_callers(tmp_path,
                                                              monkeypatch):
    """`idle_timeout` and `cancelled` are new keyword parameters: a caller
    that passes neither — the self-test, the demo builder, the smoke tests —
    must go on getting exactly what it got before."""
    import inspect

    sig = inspect.signature(blender_export.export_animation)
    assert sig.parameters["idle_timeout"].default == 300.0
    assert sig.parameters["cancelled"].default is None
    assert sig.parameters["timeout"].default == 600

    procs = _spy(monkeypatch, _script(tmp_path, ANSWERS))
    res = blender_export.export_animation(
        _poses(), tmp_path / "out", name="d", render_video=False,
        character=None, blender="blender")
    assert res.ok
    assert procs[0].returncode == 0


@pytest.mark.parametrize("reason", ["blender_timeout", "blender_cancelled"])
def test_a_stopped_export_says_which_way_it_was_stopped(reason):
    """Both reasons carry a sentence of their own; neither is a bare errno."""
    assert reason in blender_export.FAILURE_MESSAGES

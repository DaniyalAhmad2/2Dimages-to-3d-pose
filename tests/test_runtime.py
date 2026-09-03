"""The platform seam: packaging and OS differences resolved in one place."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pose3d import runtime


@pytest.fixture(autouse=True)
def _fresh_log_decision(monkeypatch):
    """Where the log went is decided once per process; give each test its own
    decision so the order they run in cannot change what they see."""
    monkeypatch.setattr(runtime, "_LOG_FILE", None)
    monkeypatch.setattr(runtime, "_LOG_DECIDED", False)


def test_app_dir_from_source_is_the_repo_root():
    assert (runtime.app_dir() / "pose3d" / "runtime.py").is_file()


def test_app_dir_follows_the_executable_when_frozen(monkeypatch, tmp_path):
    """Frozen, things shipped beside the exe must be found relative to it."""
    exe = tmp_path / "bin" / "Pose3D.exe"
    exe.parent.mkdir()
    exe.write_text("")
    monkeypatch.setattr(runtime, "IS_FROZEN", True)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert runtime.app_dir() == exe.parent


def test_exe_name_matches_the_platform():
    assert runtime.exe_name("blender") == (
        "blender.exe" if runtime.IS_WINDOWS else "blender")


def test_in_container_only_when_flagged(monkeypatch):
    monkeypatch.delenv("POSE3D_CONTAINER", raising=False)
    assert not runtime.in_container()
    monkeypatch.setenv("POSE3D_CONTAINER", "1")
    assert runtime.in_container()


def test_subprocess_kwargs_survive_a_windowed_build():
    """A frozen GUI has no usable stdin, and a child's output is UTF-8 whatever
    the machine's code page says."""
    kw = runtime.subprocess_kwargs()
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["encoding"] == "utf-8"
    assert kw["errors"] == "replace"
    if runtime.IS_WINDOWS:
        assert kw["creationflags"] == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in kw


def test_ensure_std_streams_replaces_missing_ones(monkeypatch, tmp_path):
    """Regression guard: a windowed build has sys.stderr = None, and rtmlib
    writes to it while fetching weights, which crashed the first detection."""
    monkeypatch.setattr(runtime, "IS_FROZEN", True)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Pose3D.exe"))
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    runtime.ensure_std_streams()

    assert sys.stdout is not None and sys.stderr is not None
    sys.stderr.write("hello from a windowed build\n")   # must not raise
    sys.stderr.flush()
    assert (tmp_path / "pose3d-log.txt").exists()


def test_ensure_std_streams_leaves_working_streams_alone():
    before_out, before_err = sys.stdout, sys.stderr
    runtime.ensure_std_streams()
    assert sys.stdout is before_out and sys.stderr is before_err


def test_ensure_std_streams_falls_back_when_the_folder_is_read_only(
        monkeypatch, tmp_path):
    """An installation under Program Files may not be writable; the app must
    still start rather than die trying to open a log."""
    monkeypatch.setattr(runtime, "app_dir", lambda: tmp_path / "nope" / "deeper")
    monkeypatch.setattr(runtime, "user_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(sys, "stderr", None)
    try:
        runtime.ensure_std_streams()
        assert sys.stderr is not None
        sys.stderr.write("still writable\n")
    finally:
        monkeypatch.undo()


# --- where the log actually goes -------------------------------------------
#
# `log_path()` is printed in the crash dialog and in the support instructions,
# so it has to name the file that was really opened. It used to name
# app_dir()/pose3d-log.txt unconditionally — including after the fallback had
# sent everything to the null device, i.e. exactly when the client was asked
# to send us a file that did not exist.


def test_log_path_names_the_file_that_was_opened(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "IS_FROZEN", True)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "Pose3D.exe"))
    monkeypatch.setattr(sys, "stderr", None)
    runtime.ensure_std_streams()
    assert runtime.log_path() == tmp_path / "pose3d-log.txt"


def test_an_unwritable_install_folder_falls_back_to_local_appdata(
        monkeypatch, tmp_path):
    """A bundle extracted into Program Files cannot write beside the exe."""
    state = tmp_path / "state" / "Pose3D"
    monkeypatch.setattr(runtime, "app_dir", lambda: tmp_path / "ro" / "deeper")
    monkeypatch.setattr(runtime, "user_state_dir", lambda: state)
    monkeypatch.setattr(sys, "stderr", None)
    runtime.ensure_std_streams()

    assert runtime.log_path() == state / "pose3d-log.txt"
    sys.stderr.write("from a read-only install\n")
    sys.stderr.flush()
    assert (state / "pose3d-log.txt").read_text(encoding="utf-8")


def test_when_nothing_can_be_written_the_log_path_is_none(
        monkeypatch, tmp_path):
    """The devnull fallback keeps the app alive, and log_path() must stop
    naming a file: telling the client to send a log that was never written is
    worse than telling them there is none."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(runtime, "app_dir", lambda: blocked)
    monkeypatch.setattr(runtime, "user_state_dir", lambda: blocked / "x")
    monkeypatch.setattr(sys, "stderr", None)
    runtime.ensure_std_streams()

    assert runtime.log_path() is None
    sys.stderr.write("still alive\n")        # must not raise


def test_the_log_is_rotated_once_it_reaches_five_megabytes(
        monkeypatch, tmp_path):
    """A GUI that logs every Qt warning fills a disk otherwise, and nobody
    will send us a 900 MB file."""
    monkeypatch.setattr(runtime, "app_dir", lambda: tmp_path)
    log = tmp_path / "pose3d-log.txt"
    log.write_bytes(b"x" * runtime.LOG_ROTATE_BYTES)
    monkeypatch.setattr(sys, "stderr", None)

    runtime.ensure_std_streams()

    assert (tmp_path / "pose3d-log.1.txt").stat().st_size == \
        runtime.LOG_ROTATE_BYTES
    sys.stderr.write("new run\n")
    sys.stderr.flush()
    assert log.stat().st_size < runtime.LOG_ROTATE_BYTES


def test_rotation_replaces_the_previous_rolled_file(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "app_dir", lambda: tmp_path)
    (tmp_path / "pose3d-log.1.txt").write_text("older", encoding="utf-8")
    (tmp_path / "pose3d-log.txt").write_bytes(b"y" * runtime.LOG_ROTATE_BYTES)
    monkeypatch.setattr(sys, "stderr", None)

    runtime.ensure_std_streams()

    assert (tmp_path / "pose3d-log.1.txt").read_bytes()[:1] == b"y"


def test_a_small_log_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "app_dir", lambda: tmp_path)
    (tmp_path / "pose3d-log.txt").write_text("previous run\n", encoding="utf-8")
    monkeypatch.setattr(sys, "stderr", None)

    runtime.ensure_std_streams()

    assert not (tmp_path / "pose3d-log.1.txt").exists()
    assert (tmp_path / "pose3d-log.txt").read_text(
        encoding="utf-8").startswith("previous run")


def test_the_state_directory_is_the_platform_one(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    if runtime.IS_WINDOWS:
        assert runtime.user_state_dir() == tmp_path / "AppData" / "Local" / "Pose3D"
    else:
        assert runtime.user_state_dir() == Path.home() / ".local" / "state" / "pose3d"

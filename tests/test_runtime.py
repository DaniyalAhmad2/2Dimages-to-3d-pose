"""The platform seam: packaging and OS differences resolved in one place."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pose3d import runtime


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
    monkeypatch.setattr(sys, "stderr", None)
    try:
        runtime.ensure_std_streams()
        assert sys.stderr is not None
        sys.stderr.write("still writable\n")
    finally:
        monkeypatch.undo()

"""Finding Blender, and failing usefully when it isn't there.

Export is the product's output, and every one of these paths was broken on
Windows: the lookup never tried `blender.exe`, and a missing or unusable
binary surfaced as a bare OSError.
"""
import os
import subprocess
import sys

import pytest

from pose3d import config, runtime


def _fake_blender(dirpath, name=None):
    exe = dirpath / (name or runtime.exe_name("blender"))
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("")
    exe.chmod(0o755)
    return exe


def test_env_override_wins(tmp_path, monkeypatch):
    exe = _fake_blender(tmp_path)
    monkeypatch.setenv("POSE3D_BLENDER", str(exe))
    assert config.blender_binary() == str(exe)


def test_env_pointing_at_a_folder_is_ignored(tmp_path, monkeypatch):
    """A folder is not a binary; falling through beats failing obscurely."""
    monkeypatch.setenv("POSE3D_BLENDER", str(tmp_path))
    assert config.blender_binary() != str(tmp_path)


def test_blender_shipped_beside_the_app_is_found(tmp_path, monkeypatch):
    """Regression guard: the bundled lookup asked for a file named exactly
    'blender', so on Windows — where it is blender.exe — it could never match
    and export was dead on arrival."""
    monkeypatch.delenv("POSE3D_BLENDER", raising=False)
    exe = _fake_blender(tmp_path / "blender")
    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    assert config.blender_binary() == str(exe)
    assert config.blender_available()


def test_missing_blender_reports_the_name_not_a_stale_path(monkeypatch):
    monkeypatch.delenv("POSE3D_BLENDER", raising=False)
    monkeypatch.setattr(config, "app_dir", lambda: __import__("pathlib").Path("/nonexistent"))
    monkeypatch.setattr(config.shutil, "which", lambda *_: None)
    got = config.blender_binary()
    assert got == runtime.exe_name("blender")
    assert "/home/" not in got and "\\Users\\" not in got


def test_export_says_what_to_do_when_blender_is_missing(tmp_path):
    import numpy as np
    from pose3d.core.skeleton import NUM_JOINTS
    from pose3d.export.blender_export import export_animation

    poses = np.zeros((1, NUM_JOINTS, 3))
    poses[:, :, 2] = np.linspace(0, 1.7, NUM_JOINTS)
    res = export_animation(poses, tmp_path, name="x", render_video=False,
                           blender=str(tmp_path / "definitely-not-here"))
    assert not res.ok
    assert res.returncode == 127
    assert "not found" in res.stderr.lower()
    assert "POSE3D_BLENDER" in res.stderr


def test_export_explains_an_unrunnable_binary(tmp_path):
    """A directory, or a file without the executable bit, must not surface as
    a raw errno."""
    import numpy as np
    from pose3d.core.skeleton import NUM_JOINTS
    from pose3d.export.blender_export import export_animation

    bad = tmp_path / "notexec"
    bad.mkdir()
    poses = np.zeros((1, NUM_JOINTS, 3))
    res = export_animation(poses, tmp_path / "out", name="x",
                           render_video=False, blender=str(bad))
    assert not res.ok
    assert res.returncode in (126, 127)
    assert "blender" in res.stderr.lower()


def test_subprocess_kwargs_are_applied(tmp_path, monkeypatch):
    """The export must launch Blender the way a windowed build requires."""
    import numpy as np
    from pose3d.core.skeleton import NUM_JOINTS
    from pose3d.export import blender_export

    seen = {}

    def spy(cmd, **kw):
        # ... and the ONE launch path is Popen: the export used to have a
        # second, `subprocess.run` branch, which is what left the timeout
        # inoperative on the path the GUI actually took.
        seen.update(kw)
        raise FileNotFoundError("stop here")

    monkeypatch.setattr(blender_export.subprocess, "Popen", spy)
    poses = np.zeros((1, NUM_JOINTS, 3))
    blender_export.export_animation(poses, tmp_path, name="x",
                                    render_video=False, blender="blender")
    assert seen.get("stdin") is subprocess.DEVNULL
    assert seen.get("encoding") == "utf-8"
    if sys.platform.startswith("win"):
        assert seen.get("creationflags") == subprocess.CREATE_NO_WINDOW

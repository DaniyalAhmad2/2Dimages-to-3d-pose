"""The container's mount points must not be believed on a desktop.

`/workspace` and `/host` are where the Docker image mounts the user's files.
Written as `Path("/workspace")` they are absolute POSIX paths — and on Windows
an absolute POSIX path is a *drive-relative* one: `Path("/workspace")` is
`C:\\workspace`. Any client with a folder of that name — and `C:\\host` is not
an exotic thing for a machine to have — got it pinned into the file dialog's
sidebar and used as the starting directory, with a hint underneath explaining
that the app can only see files under a folder that has nothing to do with it.

The mounts exist only where something says so: inside the container, or
because the user named them.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from pose3d.ui import filedialog                       # noqa: E402


@pytest.fixture
def fake_windows_workspace(monkeypatch, tmp_path):
    """A C:\\workspace and a C:\\host that exist but mean nothing."""
    ws, host = tmp_path / "workspace", tmp_path / "host"
    ws.mkdir()
    host.mkdir()
    monkeypatch.setattr(filedialog, "WORKSPACE", ws)
    monkeypatch.setattr(filedialog, "HOST", host)
    monkeypatch.delenv("POSE3D_CONTAINER", raising=False)
    monkeypatch.delenv("POSE3D_WORKSPACE", raising=False)
    monkeypatch.delenv("POSE3D_HOST", raising=False)
    return ws, host


def test_a_stray_c_workspace_is_not_treated_as_a_mount(fake_windows_workspace):
    ws, host = fake_windows_workspace
    folders = filedialog.shared_folders()
    assert ws not in folders and host not in folders


def test_and_the_hint_says_nothing_about_it(fake_windows_workspace):
    assert filedialog.location_hint() == ""


def test_a_read_only_folder_is_not_blamed_on_a_mount_that_is_not_there(
        fake_windows_workspace):
    ws, host = fake_windows_workspace
    msg = filedialog.not_writable_message(host / "images")
    assert "shared with the app" not in msg, msg
    assert "read-only" in msg


def test_inside_the_container_the_mounts_are_used(
        fake_windows_workspace, monkeypatch):
    ws, host = fake_windows_workspace
    monkeypatch.setenv("POSE3D_CONTAINER", "1")
    assert filedialog.shared_folders() == [ws, host]
    assert filedialog.default_dir() == str(ws)
    assert "workspace" in filedialog.location_hint()


def test_naming_them_explicitly_is_enough(fake_windows_workspace, monkeypatch):
    """Running the image's layout outside the image — the docker-compose
    mounts bind-mounted somewhere else — is a real thing people do."""
    ws, host = fake_windows_workspace
    monkeypatch.setenv("POSE3D_WORKSPACE", str(ws))
    assert filedialog.shared_folders() == [ws, host]


def test_without_a_mount_the_start_folder_is_the_users_own(
        fake_windows_workspace, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr("pose3d.runtime.app_dir", lambda: Path("/nonexistent"))
    assert filedialog.shared_folders() == [Path.home()]

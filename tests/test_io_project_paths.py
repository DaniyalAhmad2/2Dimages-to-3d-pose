"""Projects must survive moving between machines and operating systems."""
from pathlib import Path

import numpy as np

from pose3d.core.io_project import (
    _is_absolute, _resolve_image, load_project, save_project,
)
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData


def test_posix_paths_count_as_absolute_everywhere():
    """Regression guard: Windows does not consider '/data/a.jpg' absolute, so a
    project written on Linux or in the container had its external images
    treated as relative and silently resolved to the wrong place."""
    assert _is_absolute("/data/shot.jpg")
    assert _is_absolute("\\\\server\\share\\shot.jpg")
    assert _is_absolute("C:\\data\\shot.jpg") or not Path("C:\\").is_absolute()
    assert not _is_absolute("images/shot.jpg")
    assert not _is_absolute("shot.jpg")


def test_foreign_absolute_path_recovers_from_the_images_folder(tmp_path):
    """A project moved from another machine finds its images again by name."""
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "left_0001.jpg").write_bytes(b"x")
    got = _resolve_image("/somewhere/else/left_0001.jpg", tmp_path)
    assert Path(got) == tmp_path / "images" / "left_0001.jpg"


def test_foreign_windows_path_recovers_too(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "right_0002.jpg").write_bytes(b"x")
    got = _resolve_image(r"D:\captures\right_0002.jpg", tmp_path)
    assert Path(got) == tmp_path / "images" / "right_0002.jpg"


def test_unrecoverable_path_is_returned_unchanged(tmp_path):
    got = _resolve_image("/gone/missing.jpg", tmp_path)
    assert got == str(Path("/gone/missing.jpg"))


def test_project_round_trips_after_being_moved(tmp_path):
    src = tmp_path / "a"
    (src / "images").mkdir(parents=True)
    for n in ("left_0001.jpg", "right_0001.jpg"):
        (src / "images" / n).write_bytes(b"x")
    p = ProjectData(name="move")
    p.frames.append(Frame(frame_id="0001", images={
        CAM_LEFT: str(src / "images" / "left_0001.jpg"),
        CAM_RIGHT: str(src / "images" / "right_0001.jpg")}))
    save_project(p, src)

    import shutil
    dst = tmp_path / "b"
    shutil.copytree(src, dst)
    q = load_project(dst)
    for cam in (CAM_LEFT, CAM_RIGHT):
        assert Path(q.frames[0].images[cam]).exists()
        assert Path(q.frames[0].images[cam]).is_relative_to(dst)


def test_saving_twice_does_not_leave_the_database_locked(tmp_path):
    """Regression guard: sqlite handles were closed outside try/finally, so an
    error mid-write left corrections.sqlite locked (WinError 32 on Windows)."""
    from pose3d.core.io_project import append_correction
    from pose3d.core.project import Correction

    p = ProjectData(name="lock")
    p.frames.append(Frame(frame_id="0001"))
    save_project(p, tmp_path)
    for i in range(3):
        append_correction(tmp_path, Correction(
            "0001", CAM_LEFT, i, (1.0, 2.0), (3.0, 4.0), "now"))
    save_project(p, tmp_path)          # rewrites the table

    q = load_project(tmp_path)
    assert isinstance(q.corrections, list)
    # the file must be replaceable, which a leaked handle would prevent
    (tmp_path / "corrections.sqlite").replace(tmp_path / "moved.sqlite")

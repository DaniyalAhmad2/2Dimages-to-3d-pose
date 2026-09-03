"""Reading an image from a path Windows can produce.

`cv2.imread` takes a *bytes* path through OpenCV's own C++ file handling, which
on Windows encodes it with the ANSI code page: a user folder like
`C:\\Users\\Müller` or a Japanese file name simply does not survive, and the
function answers `None` rather than raising. The photo still appeared in the UI
(Qt reads the path as UTF-16) and detection died three frames later in an
unguarded slot with `'NoneType' object has no attribute 'shape'`.

A OneDrive "online-only" placeholder is the same failure with a different
cause: the file exists and is 0 bytes until the sync client materialises it.
"""
import numpy as np
import pytest

from pose3d.imageio import ImageReadError, read_image


def _write_jpeg(path, w=64, h=48):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (20, 180, 220)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    path.write_bytes(buf.tobytes())
    return img


def test_reads_a_jpeg_under_a_non_ascii_path(tmp_path):
    path = tmp_path / "Müller" / "照片.jpg"
    _write_jpeg(path)
    img = read_image(path)
    assert isinstance(img, np.ndarray)
    assert img.shape == (48, 64, 3)


def test_a_missing_file_is_named_not_returned_as_none(tmp_path):
    with pytest.raises(ImageReadError) as e:
        read_image(tmp_path / "nope.jpg")
    assert "nope.jpg" in str(e.value)


def test_a_zero_byte_placeholder_says_so(tmp_path):
    """OneDrive's "files on demand" leaves a 0-byte stub until it syncs."""
    p = tmp_path / "placeholder.jpg"
    p.write_bytes(b"")
    with pytest.raises(ImageReadError) as e:
        read_image(p)
    assert "0 bytes" in str(e.value)


def test_a_truncated_file_is_a_read_error(tmp_path):
    p = tmp_path / "half.jpg"
    _write_jpeg(p)
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 3])
    with pytest.raises(ImageReadError):
        read_image(p)


def test_the_error_is_an_oserror_so_existing_handlers_catch_it():
    assert issubclass(ImageReadError, OSError)


def test_a_str_path_works_too(tmp_path):
    p = tmp_path / "plain.jpg"
    _write_jpeg(p)
    assert read_image(str(p)).shape == (48, 64, 3)


def test_detect_project_refuses_a_none_image_by_name():
    """The old `cv2.imread` loader answers None on a path it cannot encode.
    `detect_project` used to pass that straight to the detector, where it
    surfaced as `'NoneType' object has no attribute 'shape'` with no mention
    of the file that was unreadable."""
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
    from pose3d.pipeline import detect_project

    data = ProjectData(name="t")
    f = Frame(frame_id="0000")
    f.images = {CAM_LEFT: "C:/Users/Müller/left.jpg",
                CAM_RIGHT: "C:/Users/Müller/right.jpg"}
    data.frames.append(f)

    with pytest.raises(ImageReadError) as e:
        detect_project(data, object(), lambda p: None)
    assert "Müller" in str(e.value)

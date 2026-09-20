"""One unreadable photo must cost one pair, never the whole calibration.

`pose3d.imageio.read_image` RAISES where `cv2.imread` used to return None —
0-byte OneDrive placeholders, truncated JPEGs, a path that is not an image —
and the import dialog now hands it to the calibration. The `is None` skip the
resolver still carried was dead code from the old reader, so a single bad file
in a 26-pair take aborted the import and produced nothing.
"""
import shutil

import cv2
import numpy as np
import pytest

from pose3d.calib.resolve import detect_all_tags, resolve_calibration
from pose3d.core.importer import build_project
from pose3d.core.project import CAM_LEFT, CAM_RIGHT
from pose3d.imageio import read_image
from tests.test_extrinsics import _cam, _look_at, _render_marker

PAIRS = 26
BAD = 7                     # the pair whose left photo is a 0-byte placeholder


@pytest.fixture(scope="module")
def _rendered(tmp_path_factory):
    """One left/right image showing a shared tag, rendered once."""
    d = tmp_path_factory.mktemp("render")
    intr = _cam()
    eye_l = np.array([-0.6, -1.3, 0.2]); eye_r = np.array([0.6, -1.3, 0.2])
    Rl, tl = _look_at(eye_l); Rr, tr = _look_at(eye_r)
    left, right = d / "l.png", d / "r.png"
    cv2.imwrite(str(left),
                _render_marker(intr.K, intr.dist, Rl, tl, 0, 0.30, (0, 0, 0)))
    cv2.imwrite(str(right),
                _render_marker(intr.K, intr.dist, Rr, tr, 0, 0.30, (0, 0, 0)))
    return left, right, intr, (eye_l, eye_r)


def _take(tmp_path, _rendered, break_pair=BAD, break_cam="left"):
    """A PAIRS-long take, with one photo replaced by a 0-byte file."""
    src_l, src_r, intr, eyes = _rendered
    lefts, rights = [], []
    for i in range(PAIRS):
        lp = tmp_path / f"left_{i:04d}.png"
        rp = tmp_path / f"right_{i:04d}.png"
        shutil.copyfile(src_l, lp)
        shutil.copyfile(src_r, rp)
        lefts.append(lp); rights.append(rp)
    if break_pair is not None:
        broken = (lefts if break_cam == "left" else rights)[break_pair]
        broken.write_bytes(b"")            # the OneDrive placeholder case
    return build_project(lefts, rights, name="cal"), intr, eyes


def test_one_zero_byte_photo_costs_its_pair_and_nothing_else(tmp_path,
                                                             _rendered):
    proj, intr, (eye_l, eye_r) = _take(tmp_path, _rendered)
    res = resolve_calibration(proj, read_image, marker_length=0.30,
                              intr_left=intr, intr_right=intr)

    assert res.ok, res.message
    assert res.status == "aruco"
    # solved from the other 25, to the same accuracy as a clean take
    assert np.linalg.norm(res.rig.ext[CAM_LEFT].camera_center - eye_l) < 0.05
    assert np.linalg.norm(res.rig.ext[CAM_RIGHT].camera_center - eye_r) < 0.05


def test_the_skipped_pair_is_named_in_the_summary_and_the_report(tmp_path,
                                                                 _rendered):
    proj, intr, _ = _take(tmp_path, _rendered)
    frame_id = proj.frames[BAD].frame_id
    res = resolve_calibration(proj, read_image, marker_length=0.30,
                              intr_left=intr, intr_right=intr)

    assert frame_id in res.message
    assert "0 bytes" in res.message        # the reason, with its own advice
    assert [s["frame"] for s in res.skipped] == [frame_id]
    assert res.skipped[0]["camera"] == CAM_LEFT
    assert [s["frame"] for s in res.report["skipped_pairs"]] == [frame_id]


def test_the_first_pair_being_unreadable_is_not_fatal_either(tmp_path,
                                                             _rendered):
    """The image size is read from the first pair when no intrinsics were
    uploaded — from the first READABLE pair, or a 0-byte first photo would
    abort a take whose other 25 pairs are fine."""
    proj, _intr, _ = _take(tmp_path, _rendered, break_pair=0)
    res = resolve_calibration(proj, read_image, marker_length=0.30)

    assert res.ok and res.approximate, res.message
    assert proj.frames[0].frame_id in res.message


def test_a_take_with_no_readable_photo_at_all_fails_softly(tmp_path,
                                                           _rendered):
    proj, _intr, _ = _take(tmp_path, _rendered, break_pair=None)
    for f in proj.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            open(f.images[cam], "wb").close()
    res = resolve_calibration(proj, read_image, marker_length=0.30)

    assert not res.ok and res.status == "failed"
    assert "no image pair could be read" in res.message.lower()
    assert "online-only" in res.message        # the likely cause, named
    # every reason is one this run MEASURED, on a pair it actually tried —
    # the list used to be fabricated for every frame and camera, with a
    # reason ("could not be read") that nothing had produced
    assert len(res.skipped) == 2 * PAIRS
    assert all("0 bytes" in s["reason"] for s in res.skipped)
    assert {s["frame"] for s in res.skipped} == {f.frame_id
                                                 for f in proj.frames}


def test_the_skipped_pairs_are_listed_in_frame_order(tmp_path, _rendered):
    """Zero-padded ids sort the same either way; "9" and "10" do not, and the
    sentence exists to let the user find the photo."""
    from pose3d.calib.resolve import summarise_skipped
    skipped = [{"frame": f, "camera": "left", "reason": "0 bytes"}
               for f in ("10", "9", "2")]
    assert "2, 9, 10" in summarise_skipped(skipped)


def test_a_frame_with_no_image_for_a_camera_is_skipped(tmp_path, _rendered):
    """`read_image(None)` raises TypeError out of `Path(None)` — not even an
    ImageReadError — so a frame whose dict simply lacks a camera used to take
    the import down with a traceback nobody could act on."""
    proj, intr, _ = _take(tmp_path, _rendered, break_pair=None)
    proj.frames[3].images.pop(CAM_RIGHT)

    skipped = []
    dict_id, obs = detect_all_tags(proj, read_image, skipped=skipped)
    assert dict_id is not None
    assert len(obs) == PAIRS - 1
    assert skipped and skipped[0]["frame"] == proj.frames[3].frame_id
    assert skipped[0]["camera"] == CAM_RIGHT


def test_a_loader_that_returns_none_is_still_skipped(tmp_path, _rendered):
    """The old contract (cv2.imread returning None) has not been withdrawn —
    `pose3d.quality` and the tools still pass a plain `cv2.imread`."""
    proj, intr, _ = _take(tmp_path, _rendered, break_pair=None)
    bad = proj.frames[5].images[CAM_LEFT]

    def loader(path):
        return None if str(path) == bad else cv2.imread(str(path))

    skipped = []
    _dict_id, obs = detect_all_tags(proj, loader, skipped=skipped)
    assert len(obs) == PAIRS - 1
    assert skipped and skipped[0]["frame"] == proj.frames[5].frame_id

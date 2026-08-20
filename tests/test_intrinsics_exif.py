"""Camera focal length from the photos, not from a guess.

With no uploaded calibration the app used f = max(width, height). On a phone
that is wildly wrong — a Pixel 10 Pro shooting 3072x4080 is ~2830 px, not
4080 — and an over-long focal warps triangulated depth, which shows up as the
figure leaning by an amount that changes with where it stands.
"""
import numpy as np
import pytest

from pose3d.calib.intrinsics import Intrinsics, focal_from_exif
from pose3d.calib.quality import check_rig, looks_assumed


def _jpeg_with_f35(path, size, f35, ifd0=False):
    """A JPEG carrying FocalLengthIn35mmFilm where a real camera puts it.

    Phones write it to the Exif sub-IFD (0x8769); only the top-level IFD0 is
    what `Image.getexif()` returns directly. Writing the fixture the easy way
    — into IFD0 — hid exactly the bug this guards, so default to the sub-IFD.
    """
    from PIL import Image
    im = Image.new("RGB", size, (30, 120, 30))
    exif = im.getexif()
    if f35 is not None:
        if ifd0:
            exif[41989] = f35
        else:
            exif.get_ifd(0x8769)[41989] = f35
    im.save(path, "JPEG", exif=exif)
    return path


def test_focal_comes_from_the_35mm_equivalent(tmp_path):
    p = _jpeg_with_f35(tmp_path / "shot.jpg", (3072, 4080), 24)
    f = focal_from_exif(p)
    # 24mm-equivalent over this diagonal
    assert f == pytest.approx(24 * np.hypot(3072, 4080) / 43.266615, rel=1e-6)
    # and crucially: far from the old guess
    assert f < 0.75 * max(3072, 4080)


def test_focal_is_found_in_either_ifd(tmp_path):
    """Tolerate both layouts: cameras use the sub-IFD, re-encoders often
    flatten it into IFD0."""
    a = focal_from_exif(_jpeg_with_f35(tmp_path / "sub.jpg", (4000, 3000), 26))
    b = focal_from_exif(_jpeg_with_f35(tmp_path / "top.jpg", (4000, 3000), 26,
                                       ifd0=True))
    assert a == pytest.approx(b)


def test_missing_or_unreadable_exif_returns_none(tmp_path):
    assert focal_from_exif(_jpeg_with_f35(tmp_path / "no.jpg", (640, 480), None)) is None
    assert focal_from_exif(tmp_path / "does-not-exist.jpg") is None
    (tmp_path / "notanimage.txt").write_text("hello")
    assert focal_from_exif(tmp_path / "notanimage.txt") is None


def test_zero_focal_is_rejected(tmp_path):
    """Some cameras write 0 rather than omitting the tag; a zero focal would
    produce a degenerate camera matrix."""
    assert focal_from_exif(_jpeg_with_f35(tmp_path / "z.jpg", (640, 480), 0)) is None


def test_the_default_focal_is_the_size_guess_not_exif(tmp_path):
    """EXIF is physically the right focal and empirically the wrong one here.

    On the client's captures f=2833 (EXIF) made the two views disagree —
    median epipolar 26.6 px, 38% of keypoints rejected by validate_cross_view,
    which emptied the 3D view — while the baseless f=max(w,h)=4080 gave 4.9 px
    and rejected nothing. With extrinsics solved from a single planar marker
    using the same K, the guess is self-consistent and the true focal is not.
    focal_from_exif stays tested as the input to a future focal *selection*
    step scored on tags not used for the extrinsics.
    """
    from pose3d.calib.resolve import _approx_intrinsics

    img = np.zeros((4080, 3072, 3), np.uint8)
    p = _jpeg_with_f35(tmp_path / "shot.jpg", (3072, 4080), 24)

    for got in (_approx_intrinsics(img), _approx_intrinsics(img, p)):
        assert got.source == "assumed"
        assert got.K[0, 0] == 4080.0
        assert looks_assumed(got)


def test_source_survives_a_save_load_round_trip(tmp_path):
    k = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(640, 480),
                   source="exif")
    k.save(tmp_path / "i.json")
    assert Intrinsics.load(tmp_path / "i.json").source == "exif"

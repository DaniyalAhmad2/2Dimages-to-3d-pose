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


def test_approx_intrinsics_prefer_exif(tmp_path):
    from pose3d.calib.resolve import _approx_intrinsics

    img = np.zeros((4080, 3072, 3), np.uint8)
    p = _jpeg_with_f35(tmp_path / "shot.jpg", (3072, 4080), 24)

    got = _approx_intrinsics(img, p)
    assert got.source == "exif"
    assert got.K[0, 0] == pytest.approx(focal_from_exif(p))
    assert not looks_assumed(got), "EXIF intrinsics must not read as guessed"

    # no path / no EXIF: the old behaviour, still flagged
    fallback = _approx_intrinsics(img)
    assert fallback.source == "assumed"
    assert fallback.K[0, 0] == 4080.0
    assert looks_assumed(fallback)


def test_source_survives_a_save_load_round_trip(tmp_path):
    k = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(640, 480),
                   source="exif")
    k.save(tmp_path / "i.json")
    assert Intrinsics.load(tmp_path / "i.json").source == "exif"


def test_exif_intrinsics_get_their_own_warning():
    """Better than a guess, still not a calibration — the user should be told
    which of the two they have."""
    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.pipeline import CalibratedRig

    def intr(source):
        return Intrinsics(K=np.array([[2830.0, 0, 1536.0],
                                      [0, 2830.0, 2040.0], [0, 0, 1.0]]),
                          dist=np.zeros((1, 5)), image_size=(3072, 4080),
                          source=source)

    e = Extrinsics(R=np.eye(3), t=np.zeros(3))
    msgs = " ".join(check_rig(CalibratedRig(intr("exif"), intr("exif"), e, e)))
    assert "EXIF" in msgs
    assert "assumed from the image size" not in msgs

    msgs2 = " ".join(check_rig(CalibratedRig(intr("measured"), intr("measured"), e, e)))
    assert "EXIF" not in msgs2

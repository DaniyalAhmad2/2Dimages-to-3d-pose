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


def _jpeg_with_f35(path, size, f35, ifd0=False, zoom=None):
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
    if zoom is not None:
        exif.get_ifd(0x8769)[41988] = zoom
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


def test_digital_zoom_is_applied(tmp_path):
    """A phone's 35mm-equivalent describes its LENS, not the frame it wrote.

    The client's left camera records 24 mm with DigitalZoomRatio 1.26, so the
    frame's real focal is ~3570 px. Ignoring the zoom gave 2833 px — 21 %
    short, and THAT is the number the EXIF focal was judged and reverted on.
    """
    p = _jpeg_with_f35(tmp_path / "zoom.jpg", (3072, 4080), 24, zoom=1.26)
    assert focal_from_exif(p) == pytest.approx(3569.5, abs=5.0)

    # no zoom tag, or a zoom of 1: the plain 35mm-equivalent
    q = _jpeg_with_f35(tmp_path / "plain.jpg", (3072, 4080), 24)
    assert focal_from_exif(q) == pytest.approx(2833.0, abs=5.0)
    r = _jpeg_with_f35(tmp_path / "one.jpg", (3072, 4080), 24, zoom=1)
    assert focal_from_exif(r) == pytest.approx(focal_from_exif(q))


def test_the_default_focal_is_the_size_guess_not_exif(tmp_path):
    """Inverted on purpose: EXIF is now computed RIGHT and still not used.

    The original version of this test justified the size guess with "EXIF says
    2833 px and that emptied the 3D view". That number was wrong — it dropped
    DigitalZoomRatio — so the justification would not have survived anyone
    recomputing it, and the revert of the EXIF focal (f4c92f9) would get
    re-litigated as "the approach was wrong" when what was wrong was the
    arithmetic.

    So: EXIF here is 3570 px, 12 % from the guess rather than 31 %, and the
    guess is STILL what the app uses, because it is what measures best on the
    client's rig — 4.91 px of body epipolar error for f=max(w,h) against 5.37
    for the best focal pair a sweep could find and 10.9 for the EXIF pair.
    Body keypoints play no part in the calibration, so that comparison is
    independent of the tags that produced the extrinsics.
    """
    from pose3d.calib.resolve import _approx_intrinsics

    img = np.zeros((4080, 3072, 3), np.uint8)
    p = _jpeg_with_f35(tmp_path / "shot.jpg", (3072, 4080), 24, zoom=1.26)

    f_exif = focal_from_exif(p)
    assert f_exif == pytest.approx(3569.5, abs=5.0)   # readable and correct...
    got = _approx_intrinsics(img)                     # ...and deliberately unused
    assert got.source == "assumed"
    assert got.K[0, 0] == 4080.0
    assert looks_assumed(got)
    assert abs(f_exif - 4080.0) / 4080.0 < 0.15


def test_source_survives_a_save_load_round_trip(tmp_path):
    k = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(640, 480),
                   source="exif")
    k.save(tmp_path / "i.json")
    assert Intrinsics.load(tmp_path / "i.json").source == "exif"


def test_keyless_file_does_not_become_measured(tmp_path):
    """A file with no `source` has no provenance, and "measured" is a claim.

    The client's own intrinsics were guessed from the image size and written
    before the field existed; loading and re-saving them stamped them
    "measured" and laundered the guess into every project that touched them —
    including any checkerboard override gated on this field, which would then
    refuse to run.
    """
    import json

    def written(K, dist=((0.0,) * 5,), size=(3072, 4080)):
        p = tmp_path / f"k{K[0][0]}.json"
        p.write_text(json.dumps({"K": K, "dist": [list(d) for d in dist],
                                 "image_size": list(size)}))
        return Intrinsics.load(p)

    guess = [[4080.0, 0, 1536.0], [0, 4080.0, 2040.0], [0, 0, 1.0]]
    assert written(guess).source == "assumed"          # K still shows the guess

    real = [[3120.0, 0, 1520.0], [0, 3118.0, 2050.0], [0, 0, 1.0]]
    got = written(real, dist=((0.03, -0.01, 0, 0, 0),))
    assert got.source == "unknown"                     # could be anything
    assert not looks_assumed(got)

    # and a file that DOES say stays believed
    k = Intrinsics(K=np.array(real), dist=np.zeros((1, 5)),
                   image_size=(3072, 4080), source="measured")
    k.save(tmp_path / "m.json")
    assert Intrinsics.load(tmp_path / "m.json").source == "measured"


def test_a_checkerboard_never_overwrites_a_measured_calibration():
    """The one-time override is gated on provenance: a guess is replaced, a
    real calibration is not, so re-running the import cannot quietly demote a
    camera that was properly calibrated."""
    from pose3d.calib.intrinsics import checkerboard_override

    measured = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)),
                          image_size=(640, 480), rms=0.4, source="measured")
    # no images are even looked at: the gate short-circuits
    assert checkerboard_override(measured, [], (9, 6)) is measured

    guessed = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)),
                         image_size=(640, 480), source="assumed")
    with pytest.raises(ValueError):        # ...and a guess does go to calibrate
        checkerboard_override(guessed, [], (9, 6))

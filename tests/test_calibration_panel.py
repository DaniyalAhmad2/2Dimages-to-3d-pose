"""The calibration line in the sidebar must say what actually happened.

On the client's rig — two phones, no intrinsics files, tags on a wall — every
correct import painted "⚠ Calibrated (with problems)" over three amber
warnings, two of which prescribed something the app cannot do and our own
client guide says is unnecessary. A non-technical user reads three warnings on
a successful run as "this output cannot be trusted".
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from pose3d.calib.quality import CalibrationNote                  # noqa: E402


@pytest.fixture
def bar():
    from PySide6.QtWidgets import QApplication
    from pose3d.ui.panels import Sidebar
    QApplication.instance() or QApplication([])
    return Sidebar()


def test_notes_alone_read_as_calibrated(bar):
    bar.set_calibrated(True, [
        CalibrationNote("The cameras' lenses were estimated from the photo "
                        "size."),
        CalibrationNote("The two cameras shot at different sizes."),
    ])
    assert bar.calib_status.text() == "✓ Calibrated"
    assert "#4ed67a" in bar.calib_status.styleSheet()     # the green, not amber
    assert "lenses were estimated" in bar.calib_warn.text()
    assert bar.calib_warn.isVisibleTo(bar)
    # and the notes are not painted as warnings either
    assert "#e0a33a" not in bar.calib_warn.styleSheet()


def test_a_real_problem_still_says_so(bar):
    bar.set_calibrated(True, [
        "No tag was seen by both cameras; the rig was rebuilt from the poses.",
        CalibrationNote("The two cameras shot at different sizes."),
    ])
    assert bar.calib_status.text() == "⚠ Calibrated (with problems)"
    assert "#e0a33a" in bar.calib_status.styleSheet()
    # both lines are still shown: the note is not hidden by the problem
    assert "No tag was seen" in bar.calib_warn.text()
    assert "different sizes" in bar.calib_warn.text()


def test_no_calibration_at_all_is_not_dressed_up(bar):
    bar.set_calibrated(False, [CalibrationNote("a note")])
    assert bar.calib_status.text() == "Not calibrated"
    assert "#e65c5c" in bar.calib_status.styleSheet()


def test_a_clean_calibration_says_nothing(bar):
    bar.set_calibrated(True, [])
    assert bar.calib_status.text() == "✓ Calibrated"
    assert not bar.calib_warn.isVisibleTo(bar)

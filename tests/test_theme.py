"""The UI must look the same whatever the host's theme is set to.

This is a dark-themed application, but Qt draws anything the stylesheet does
not name from the *system* palette. On a Windows machine in light mode that
punched light chrome through the dark UI: the splitter handles disappeared
against the panels, and the accuracy gauge was filled with the native window
colour behind its near-white readout.

These render against a deliberately LIGHT palette — the failing condition —
and assert on actual pixels, because the bug is invisible to any test that
only checks widgets were constructed.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPalette          # noqa: E402
from PySide6.QtWidgets import QApplication          # noqa: E402


@pytest.fixture
def light_host(qapp):
    """A host desktop set to light mode, which is what broke."""
    before = qapp.palette()
    light = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base,
                 QPalette.ColorRole.Button):
        light.setColor(role, QColor("#ffffff"))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        light.setColor(role, QColor("#000000"))
    qapp.setPalette(light)
    yield qapp
    qapp.setPalette(before)


@pytest.fixture
def qapp():
    return QApplication.instance() or QApplication([])


def _render(w, size=(200, 200)):
    w.resize(*size)
    w.show()
    QApplication.instance().processEvents()
    return w.grab().toImage()


def _luma(img, x, y):
    c = img.pixelColor(x, y)
    return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()


def test_the_accuracy_gauge_stays_readable_on_a_light_host(light_host):
    """Regression guard: the gauge overrides paintEvent, so the stylesheet
    background is never drawn for it. It used to inherit the host's window
    colour, leaving near-white text on white."""
    from pose3d.ui.panels import PoseAccuracyGauge

    from pose3d.core.project import CAM_LEFT, CAM_RIGHT

    g = PoseAccuracyGauge()
    g.set_cameras({CAM_LEFT: 92.0, CAM_RIGHT: 88.0})
    img = _render(g)

    # sample the corners, which are background whatever the arc is doing
    corners = [_luma(img, 2, 2), _luma(img, img.width() - 3, 2),
               _luma(img, 2, img.height() - 3)]
    assert max(corners) < 90, (
        f"gauge background is light ({corners}) — the near-white readout "
        "would be invisible on it")


def test_the_gauge_readout_contrasts_with_its_own_background(light_host):
    from pose3d.ui.panels import COL_PANEL, COL_TEXT

    def luma(c):
        return 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()

    assert luma(COL_TEXT) - luma(COL_PANEL) > 120, "readout has too little contrast"


def test_the_dark_theme_survives_a_light_system_palette(light_host):
    """A real widget, rendered — not the palette we just set.

    Asserting on the style object is a dead end: a stylesheet replaces it with
    QStyleSheetStyle whose baseStyle() is None. What matters anyway is whether
    ordinary chrome comes out dark on a light-mode host, so render some.
    """
    from PySide6.QtWidgets import QPushButton

    from pose3d.app import apply_dark_theme

    apply_dark_theme(light_host)
    assert light_host.palette().color(
        QPalette.ColorRole.Window).lightness() < 60

    img = _render(QPushButton("Export"), (140, 34))
    corners = [_luma(img, 3, 3), _luma(img, img.width() - 4, img.height() - 4)]
    assert max(corners) < 110, (
        f"button chrome is light ({corners}) — the host theme is leaking in")


def test_the_splitter_handle_is_visible_against_the_panels(light_host):
    """The handle must differ from the panels either side, or the panes look
    fixed and there is no sign they can be dragged at all."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter, QWidget

    from pose3d.app import apply_dark_theme

    apply_dark_theme(light_host)
    sp = QSplitter(Qt.Orientation.Horizontal)
    for _ in range(2):
        w = QWidget()
        w.setObjectName("cardPanel")
        sp.addWidget(w)
    img = _render(sp, (240, 80))

    h = sp.handle(1).geometry()
    assert h.width() >= 4, f"handle is only {h.width()}px — hard to grab"
    mid_y = img.height() // 2
    handle = _luma(img, h.center().x(), mid_y)
    panel = _luma(img, max(0, h.left() - 12), mid_y)
    assert abs(handle - panel) > 6, (
        f"handle ({handle:.0f}) is indistinguishable from the panel "
        f"({panel:.0f}) — the split looks unmovable")


def test_splitter_handles_are_styled():
    """Belt and braces on the stylesheet itself: the hover cue is what makes
    the affordance discoverable, and a render test will not catch its absence."""
    from pathlib import Path

    qss = (Path(__file__).resolve().parent.parent
           / "pose3d" / "ui" / "dark.qss").read_text()
    assert "QSplitter::handle" in qss, "splitter handles are unstyled"
    assert "QSplitter::handle:hover" in qss, "no hover cue on the drag handle"


def test_the_gauge_shows_one_number_per_camera(light_host):
    """Phase 4: the gauge stopped averaging the two cameras into one figure.

    Averaging hid the case that matters — one view agreeing and the other not
    — and the two views of this rig are 2:1 apart in resolution, so they were
    never commensurable in the first place. Two arcs, two numbers, and the
    word underneath is the WORSE band.
    """
    from pose3d.ui.panels import PoseAccuracyGauge, acc_label

    g = PoseAccuracyGauge()
    g.set_cameras({"left": 92.0, "right": 55.0})
    img = _render(g)

    assert g._worst() == 55.0
    assert acc_label(g._worst()) == "Low"
    # still opaque and dark whatever the host theme is doing
    corners = [_luma(img, 2, 2), _luma(img, img.width() - 3, 2),
               _luma(img, 2, img.height() - 3)]
    assert max(corners) < 90, f"gauge background is light ({corners})"
    # and both arcs are painted: the green one and the red one are both there
    seen = {(img.pixelColor(x, y).red(), img.pixelColor(x, y).green(),
             img.pixelColor(x, y).blue())
            for x in range(0, img.width(), 2)
            for y in range(0, img.height(), 2)}
    from pose3d.ui.panels import COL_GREEN, COL_RED
    for col in (COL_GREEN, COL_RED):
        assert any(abs(r - col.red()) < 30 and abs(gr - col.green()) < 30
                   and abs(b - col.blue()) < 30 for r, gr, b in seen), (
            f"{col.name()} arc was not painted")


def test_a_missing_camera_leaves_its_arc_empty_rather_than_zero(light_host):
    """A camera with no number must not be drawn as 0 % — that is a red arc
    claiming a measurement nobody made."""
    import numpy as np

    from pose3d.ui.panels import PoseAccuracyGauge

    g = PoseAccuracyGauge()
    g.set_cameras({"left": 90.0, "right": float("nan")})
    _render(g)
    assert np.isnan(g._pct["right"])
    assert g._worst() == 90.0


def test_the_stylesheet_is_read_as_utf8_whatever_the_code_page_is(light_host):
    """dark.qss holds an em dash, so the file is not ASCII and the encoding
    the app reads it with is a real decision, not a formality.

    Read with the machine's locale encoding — which is what `read_text()` with
    no argument does — the same bytes give three different answers: UTF-8 on
    the developer's Linux box, silent mojibake on a cp1252 Windows machine
    (`â€"` inside a CSS comment, which is harmless right up until the next
    non-ASCII character lands in a selector), and `UnicodeDecodeError` before
    any window on a Japanese, Chinese or Korean one.
    """
    from pathlib import Path

    from pose3d.app import apply_dark_theme

    qss = Path(__file__).resolve().parent.parent / "pose3d" / "ui" / "dark.qss"
    raw = qss.read_bytes()
    assert not raw.isascii(), (
        "dark.qss is pure ASCII, so this test proves nothing any more — put "
        "the em dash back or delete the test")

    apply_dark_theme(light_host)
    sheet = light_host.styleSheet()
    assert "—" in sheet, "the em dash did not survive the read"
    assert not sheet.isascii()
    assert "â€" not in sheet, "cp1252 mojibake: read without encoding="


def test_a_cjk_code_page_would_refuse_the_stylesheet_outright():
    """The Linux companion to the test above: proof that the locale default is
    load-bearing, not merely untidy. On a cp932 machine this is the exception
    that kills the app before it shows a window."""
    from pathlib import Path

    qss = Path(__file__).resolve().parent.parent / "pose3d" / "ui" / "dark.qss"
    with pytest.raises(UnicodeDecodeError):
        qss.read_text(encoding="cp932")


# --- room on a small, scaled screen -----------------------------------------
#
# The client's laptop is 1366x768 at 150 % scaling, i.e. 910x512 in the
# logical pixels Qt lays out in. Every size below is logical.

SMALL_SCREEN = (910, 512)          # 1366x768 @ 150 %
TOPBAR_H = 46
VIEW3D_MIN_H = 160

#: The physical screen the client actually has. The window is laid out in
#: logical pixels, and Qt's `availableGeometry` reports them, so this is what
#: a test faking a screen has to use.
CLIENT_SCREEN = (1366, 768)


def test_the_sidebar_is_no_longer_pinned_to_one_width(qapp):
    """setFixedWidth pins the maximum as well as the minimum, so the layout
    can never give the column another pixel however much the content needs —
    at 150 % scaling that is how a measurement ends up elided to "5…"."""
    from pose3d.ui.panels import Sidebar

    s = Sidebar()
    assert s.minimumWidth() == 232
    assert s.maximumWidth() > s.minimumWidth(), "still a fixed width"
    assert s.sizeHint().width() == 232, "the column changed width"
    assert s.minimumWidth() < SMALL_SCREEN[0]


def test_the_timeline_is_no_longer_pinned_to_one_height(qapp):
    from pose3d.ui.timeline import Timeline

    t = Timeline()
    assert t.minimumHeight() == 120
    assert t.maximumHeight() > t.minimumHeight(), "still a fixed height"
    assert t.sizeHint().height() == 120, "the filmstrip changed height"


def test_the_fixed_parts_still_fit_a_1366x768_screen_at_150_percent(qapp):
    """What has to be true for the window to be usable at all: the parts that
    cannot shrink, plus the 3D view's own minimum, fit the screen.

    Measured off the widgets, not off two literals repeated here. The literals
    version went on passing while the real minimum grew to 949 px, because it
    was only ever asserting about itself."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.panels import Sidebar
    from pose3d.ui.timeline import Timeline
    from pose3d.ui.view3d import View3D

    used = (MainWindow.TOPBAR_H + View3D().minimumHeight()
            + Timeline().minimumHeight())
    assert MainWindow.TOPBAR_H == TOPBAR_H
    assert View3D().minimumHeight() == VIEW3D_MIN_H
    assert used <= SMALL_SCREEN[1], f"{used}px of fixed chrome on a 512px screen"
    assert Sidebar().minimumWidth() * 2 < SMALL_SCREEN[0]


def test_the_window_opens_no_larger_than_the_screen_it_is_on(qapp, monkeypatch):
    """`resize(1540, 920)` is bigger than the client's 1366x768 laptop, so the
    window opened with its timeline and its right column off the bottom and
    right of the desktop — and Windows will not let a title bar be dragged
    above the top of the screen to get them back.
    """
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QScreen

    from pose3d.core.project import ProjectData
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    monkeypatch.setattr(QScreen, "availableGeometry",
                        lambda self: QRect(0, 0, *CLIENT_SCREEN))
    win = MainWindow(ProjectModel(ProjectData(name="Small_Screen"), None))

    assert win.size().width() == CLIENT_SCREEN[0] - 40
    assert win.size().height() == CLIENT_SCREEN[1] - 80

    # AFTER show(), which is the only measurement that means anything: resize()
    # records a size, and the layout's own minimum overrides it the moment the
    # window is mapped. It was 1025x949 here — 181 px of window below the
    # bottom of a 768 px desktop, with the timeline in it.
    #
    # WA_DontShowOnScreen: show() everything the LAYOUT does — polish, activate,
    # apply the minimum — without asking the offscreen platform plugin for a
    # real window and a GL context it cannot give, whose teardown at
    # interpreter exit is where this suite crashes.
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win.show()
    qapp.processEvents()
    hint = win.minimumSizeHint()
    assert hint.width() <= CLIENT_SCREEN[0], hint
    assert hint.height() <= CLIENT_SCREEN[1], hint
    assert win.size().width() <= CLIENT_SCREEN[0], win.size()
    assert win.size().height() <= CLIENT_SCREEN[1], win.size()
    # Take the window down here rather than leaving a shown one (with a live
    # GL context) for the interpreter to collect at exit: offscreen, that
    # teardown is where Qt is least happy, and a suite that dumps core after
    # its last green line is a suite nobody trusts.
    win.close()

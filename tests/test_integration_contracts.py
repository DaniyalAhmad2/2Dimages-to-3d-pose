"""What the on-display end-to-end driver reaches for, asserted headlessly.

The controller drives the finished app once on a real display and clicks its
way through the five tasks' work. That run cannot happen in this suite — it
needs a display, and no subagent may open a window on the user's screen — so
what CAN be pinned offscreen is pinned here: every name, widget and label
that driver addresses. A rename or a quiet removal then fails in CI, where
the cost is a red line, instead of on the machine where the cost is a
confused half-hour with a live window.

These are contracts, deliberately shallow. The BEHAVIOUR behind each one is
tested where it belongs — this file only guarantees the driver can still find
it.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pose3d.core.project import CAM_LEFT, ProjectData                # noqa: E402
from pose3d.core.skeleton import NUM_JOINTS                          # noqa: E402
from tests.test_ui_smoke import _project_with_rig                    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _window(tmp_path=None):
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    data, rig, _gt = _project_with_rig()
    model = ProjectModel(data, rig,
                         project_dir=None if tmp_path is None else str(tmp_path))
    return MainWindow(model)


def _empty_window():
    """The window the app opens with nothing loaded — `app.main`'s own."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    return MainWindow(ProjectModel(ProjectData(name="empty"), None))


# --- opening a project (T3) ------------------------------------------------

def test_open_project_is_reachable_by_menu_and_by_button(qapp, tmp_path):
    from PySide6.QtWidgets import QMenu

    win = _window(tmp_path)
    assert callable(win._on_open_project)
    assert isinstance(win._recent_menu, QMenu)
    assert win._recent_menu.title() == "Recent Projects"
    assert win._recent_menu.toolTipsVisible(), \
        "the tooltip is the only thing that tells two takes of one name apart"
    # rebuilt from disk on every show — a project moved or deleted since the
    # last session is dropped rather than offered — and it says so when there
    # is nothing to offer, instead of dropping an empty menu on the user
    win._fill_recent_menu()
    actions = win._recent_menu.actions()
    assert len(actions) == 1 and not actions[0].isEnabled()
    assert actions[0].text() == "Nothing opened yet"


def test_the_open_button_is_on_the_empty_window_and_only_there(
        qapp, tmp_path, monkeypatch):
    """The button exists for the user who has just launched the app with
    nothing loaded; a window that already holds a project has the menu."""
    from pose3d.ui import filedialog

    empty = _empty_window()
    loaded = _window(tmp_path)
    assert empty.btn_open.isVisibleTo(empty)
    assert not loaded.btn_open.isVisibleTo(loaded)

    # …and it is wired to the same handler the menu uses
    asked = []
    monkeypatch.setattr(filedialog, "existing_directory",
                        lambda *a, **k: asked.append(a) or "")
    empty.btn_open.click()
    assert asked, "the Open Project button is connected to nothing"


# --- stepping frames from anywhere (T2) ------------------------------------

def test_the_frame_keys_step_the_frame(qapp):
    from PySide6.QtCore import Qt

    win = _window()
    assert set(win.FRAME_KEYS) == {Qt.Key.Key_Left, Qt.Key.Key_Right,
                                   Qt.Key.Key_Home, Qt.Key.Key_End}
    win.model.set_frame(0)
    assert win._step_frame(Qt.Key.Key_Right) is True
    assert win.model.current == 1
    assert win._step_frame(Qt.Key.Key_End) is True
    assert win.model.current == len(win.model.project.frames) - 1
    assert win._step_frame(Qt.Key.Key_Home) is True
    assert win.model.current == 0
    assert win._step_frame(Qt.Key.Key_A) is False, "an unrelated key moved it"


# --- the placeholder handles (T2) ------------------------------------------

def test_a_missing_joint_gets_a_placeholder_that_names_itself(qapp):
    """The driver finds the placeholder by these two attributes and drags it,
    so both have to stay on the item and mean what they mean."""
    from pose3d.ui.camera_view import CameraPanel

    panel = CameraPanel("LEFT VIEW", CAM_LEFT)
    xy = np.tile(np.array([50.0, 60.0]), (NUM_JOINTS, 1))
    xy[3] = np.nan                       # nothing detected for this joint
    panel.view.set_pose(xy, np.ones(NUM_JOINTS))

    items = panel.view._joints
    assert [it.joint_id for it in items] == list(range(NUM_JOINTS))
    assert items[3].is_placeholder
    assert not any(it.is_placeholder for i, it in enumerate(items) if i != 3)


# --- the Show filter (T2) --------------------------------------------------

def test_the_show_filter_offers_missing(qapp):
    win = _window()
    combo = win.timeline_header.show_combo
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert any("Missing" in t for t in labels), labels
    # the option carries the STATUS it keeps, so the label can be reworded
    assert combo.itemData(labels.index(
        next(t for t in labels if "Missing" in t))) == "red"


# --- the marker size box (T5) ----------------------------------------------

def test_the_marker_box_is_centimetres(qapp, tmp_path):
    from pose3d.ui.import_dialog import ImportDialog

    dlg = ImportDialog(projects_root=str(tmp_path))
    assert dlg.marker.suffix() == " cm"
    assert dlg.marker.value() == pytest.approx(5.00)


# --- the job dialog that will not let go (T3) ------------------------------

def test_the_job_dialog_swallows_escape(qapp):
    """Esc used to reach `QDialog::reject()` and hand the window back while
    the worker thread was still rewriting the project — and Recompute 3D and
    Set Scale are precisely the jobs with no Cancel button to ask."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QProgressDialog
    from pose3d.ui import worker

    assert issubclass(worker._JobDialog, QProgressDialog)

    dlg = worker._JobDialog("Recompute", None, 0, 0, None)
    stopped = []
    dlg.on_stop = lambda: stopped.append(True)
    dlg.show()

    esc = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                    Qt.KeyboardModifier.NoModifier)
    dlg.keyPressEvent(esc)
    assert stopped == [True], "Esc did not reach the job"
    assert dlg.isVisible(), "Esc released the window mid-job"

    # …and the close box is refused the same way
    dlg.close()
    assert dlg.isVisible()
    assert len(stopped) == 2
    dlg.allow_close = True               # only run_job's teardown may
    dlg.close()
    assert not dlg.isVisible()

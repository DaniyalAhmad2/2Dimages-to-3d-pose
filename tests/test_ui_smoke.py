"""Phase 6 verification: UI builds headlessly and the signal chain works.

Runs under the offscreen Qt platform (no display needed). Verifies:
- ProjectModel emits pose3dChanged on frame change and on joint edit.
- A simulated joint drag re-triangulates and changes the fitted 3D pose.
- MainWindow assembles without error.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
from pose3d.pipeline import CalibratedRig
from tests.synth import default_two_cam, project, sample_skeleton_3d

pytest.importorskip("PySide6")


def _project_with_rig():
    rig_geo = default_two_cam()
    intr = Intrinsics(K=rig_geo["K"], dist=rig_geo["dist"], image_size=rig_geo["size"])
    ext_l = Extrinsics(*rig_geo["left"]); ext_r = Extrinsics(*rig_geo["right"])
    rig = CalibratedRig(intr, intr, ext_l, ext_r)
    gt = sample_skeleton_3d()

    project_data = ProjectData(name="UI_Test")
    for i in range(3):
        f = Frame(frame_id=f"{i:04d}")
        pl = project(gt, rig_geo["K"], rig_geo["dist"], *rig_geo["left"])
        pr = project(gt, rig_geo["K"], rig_geo["dist"], *rig_geo["right"])
        f.kp2d[CAM_LEFT] = pl; f.kp2d[CAM_RIGHT] = pr
        f.scores[CAM_LEFT] = np.ones(len(pl)); f.scores[CAM_RIGHT] = np.ones(len(pr))
        f.pose3d = np.tile(gt, 1).reshape(gt.shape)
        f.fitted3d = gt.copy()
        f.images = {CAM_LEFT: "nonexistent_l.jpg", CAM_RIGHT: "nonexistent_r.jpg"}
        project_data.frames.append(f)
    return project_data, rig, gt


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_model_emits_on_frame_change(qapp):
    from pose3d.ui.model import ProjectModel
    data, rig, gt = _project_with_rig()
    model = ProjectModel(data, rig)
    received = {}
    model.pose3dChanged.connect(lambda p: received.update(pose=p))
    model.set_frame(1)
    assert "pose" in received
    assert received["pose"].shape == gt.shape


def test_joint_edit_resolves_3d(qapp):
    from pose3d.ui.model import ProjectModel
    data, rig, gt = _project_with_rig()
    model = ProjectModel(data, rig)
    model.set_frame(0)
    before = model.frame().fitted3d.copy()
    # move a wrist observation in the left view by 30 px and re-solve
    pl = model.frame().kp2d[CAM_LEFT][6]
    model.set_joint_2d(CAM_LEFT, 6, float(pl[0] + 30), float(pl[1]))
    after = model.frame().fitted3d
    assert not np.allclose(before[6], after[6]), "3D should change after edit"
    assert model.stack.can_undo()
    model.undo()
    assert not model.frame().corrected[CAM_LEFT][6]


def test_joint_edit_does_not_reload_image(qapp):
    """Regression: a joint edit must NOT reload the image / fitInView.

    The freeze was itemChange -> model edit -> joint2dChanged -> _refresh_views
    -> set_image -> fitInView -> itemChange (infinite recursion). A committed
    edit must go through the overlay-only path.
    """
    from PySide6.QtCore import QPointF
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    data, rig, gt = _project_with_rig()
    model = ProjectModel(data, rig)
    win = MainWindow(model)
    model.set_frame(0)

    calls = {"set_image": 0}
    orig = win.cam_left.view.set_image
    win.cam_left.view.set_image = lambda p: (calls.__setitem__(
        "set_image", calls["set_image"] + 1), orig(p))

    before = model.frame().fitted3d.copy()
    # simulate a committed drag (mouse-release) on the left wrist
    pl = model.frame().kp2d[CAM_LEFT][6]
    win.cam_left.view._on_released(6, QPointF(float(pl[0] + 30), float(pl[1])))

    assert calls["set_image"] == 0, "edit must not reload the image"
    assert model.frame().corrected[CAM_LEFT][6]
    assert not np.allclose(before[6], model.frame().fitted3d[6])


def test_3d_fullscreen_toggle_is_in_app(qapp):
    """The 3D fullscreen must expand in-app (hide the rest), not open a popup."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))
    assert not win._fs_active
    win._toggle_fullscreen()
    assert win._fs_active
    assert win._mid.isHidden()
    # in-app, not a popup: still inside the main window, not a top-level window
    assert not win._view3d_card.isWindow()
    assert win._view3d_card.window() is win

    # the readouts and the timeline stay available, so frames can be stepped
    # through and judged without leaving the large view
    for w in (win.pose_acc, win.accuracy, win.selected):
        assert w.window() is win, "accuracy panel left the fullscreen view"
        assert not w.isHidden()
    assert not win._tl_area.isHidden(), "timeline hidden in fullscreen"

    win._toggle_fullscreen()
    assert not win._fs_active
    assert not win._mid.isHidden()
    # panels returned to the right column, in their original order
    order = [win._rightcol.widget(i) for i in range(win._rightcol.count())]
    assert order[:4] == [win._view3d_card, win.pose_acc, win.accuracy, win.selected]


def test_main_window_builds(qapp):
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    data, rig, gt = _project_with_rig()
    model = ProjectModel(data, rig)
    win = MainWindow(model)
    assert win.model is model
    # timeline populated with all frames
    assert win.timeline._model.rowCount() == 3


def test_buttons_are_wired(qapp, tmp_path):
    """Every dashboard button must do something, not silently no-op."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    data, rig, gt = _project_with_rig()
    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    msgs = []
    model.statusMessage.connect(lambda m: msgs.append(m))
    win = MainWindow(model)

    # Recalibrate 3D recomputes (was previously unconnected)
    win.sidebar.recalibrate.emit()
    assert any("Recalculated" in m for m in msgs)

    # Save writes the project folder (was previously print-only)
    win.btn_save.click()
    assert (tmp_path / "project.json").exists()

    # Auto Recalculate toggles the model flag
    win.btn_auto.setChecked(False)
    assert model.auto_recalc is False

    # Show Joints toggles overlay visibility
    before = win.cam_left.view._show_joints
    win.sidebar.cb_joints.setChecked(not before)
    assert win.cam_left.view._show_joints != before

    # Undo starts disabled, enables after an edit, and reverts it
    assert not win.btn_undo.isEnabled()
    model.set_frame(0)
    pl = model.frame().kp2d[CAM_LEFT][6]
    model.set_joint_2d(CAM_LEFT, 6, float(pl[0] + 25), float(pl[1]))
    assert win.btn_undo.isEnabled()
    win.btn_undo.click()
    assert not model.frame().corrected[CAM_LEFT][6]

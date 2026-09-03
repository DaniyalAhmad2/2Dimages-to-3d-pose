"""Phase 6 verification: UI builds headlessly and the signal chain works.

Runs under the offscreen Qt platform (no display needed). Verifies:
- ProjectModel emits pose3dChanged on frame change and on joint edit.
- A simulated joint drag re-triangulates and changes the fitted 3D pose.
- MainWindow assembles without error.
"""
import os
import sys
from pathlib import Path

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
    for w in (win.pose_acc, win.accuracy):
        assert w.window() is win, "accuracy panel left the fullscreen view"
        assert not w.isHidden()
    assert not win._tl_area.isHidden(), "timeline hidden in fullscreen"

    win._toggle_fullscreen()
    assert not win._fs_active
    assert not win._mid.isHidden()
    # panels returned to the right column, in their original order
    order = [win._rightcol.widget(i) for i in range(win._rightcol.count())]
    assert order[:3] == [win._view3d_card, win.pose_acc, win.accuracy]


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


def test_import_dialog_opens(qapp):
    """The Import Images button must actually open its dialog.

    Regression guard: a broken import inside import_dialog.py made the button
    silently do nothing — Qt swallows exceptions raised in a slot, so the only
    trace was a stack dump in the container log. Nothing constructed this
    dialog, so the whole suite stayed green.
    """
    from pose3d.ui.import_dialog import ImportDialog
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    dlg = ImportDialog()                     # imports + builds cleanly
    assert dlg.left_pick is not None and dlg.right_pick is not None

    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))
    opened = {}
    win._run_import_dialog = lambda d: opened.setdefault("dlg", d)
    win.btn_import.click()                   # must reach the dialog, not raise
    assert isinstance(opened.get("dlg"), ImportDialog)


def test_file_pickers_offer_the_shared_folders():
    """The picker must start somewhere the app can actually read, or it lists
    an empty folder and looks broken."""
    from pose3d.ui import filedialog
    folders = filedialog.shared_folders()
    assert folders, "no readable folder offered to the file picker"
    assert all(p.is_dir() for p in folders)
    assert filedialog.default_dir() == str(folders[0])


@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="chmod(0o500) does not make a directory read-only "
                           "on Windows, so this cannot set up its own fixture")
def test_a_read_only_directory_is_reported_as_unwritable(tmp_path):
    """The probe must be a real write.

    os.access(W_OK) — what this used to call — reads the read-only *attribute*
    on Windows and ignores ACLs, so it answers True for C:\\Program Files and
    the export dies inside Blender long after the guard let it through.
    """
    from pose3d.ui import filedialog

    ro = tmp_path / "readonly"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        assert not filedialog.is_writable(ro)
        assert filedialog.is_writable(tmp_path)
    finally:
        ro.chmod(0o700)


def test_is_writable_holds_on_every_platform(tmp_path):
    """Platform-neutral half of the above: a normal folder is writable, a
    missing one is not, and the probe leaves nothing behind."""
    from pose3d.ui import filedialog

    assert filedialog.is_writable(tmp_path)
    assert not filedialog.is_writable(tmp_path / "does-not-exist")
    assert not filedialog.is_writable(tmp_path / "a-file.txt")
    assert list(tmp_path.iterdir()) == [], "the write probe left a file behind"


def test_export_refuses_a_destination_it_cannot_write(qapp, tmp_path, monkeypatch):
    """Picking an unwritable folder must be refused up front.

    /host is shared read-only so source images can be browsed, and choosing it
    used to fail minutes later inside Blender with a bare
    "[Errno 30] Read-only file system".

    The refusal is driven through is_writable rather than through real
    permissions, so this covers Windows — where no chmod can produce the
    fixture, but ACLs produce the situation constantly.
    """
    from pose3d.ui import filedialog
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    dest = tmp_path / "protected"
    dest.mkdir()
    monkeypatch.setattr(filedialog, "is_writable",
                        lambda p: Path(p) != dest)

    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))
    monkeypatch.setattr(filedialog, "existing_directory", lambda *a, **k: str(dest))
    warned = {}
    import PySide6.QtWidgets as W
    monkeypatch.setattr(W.QMessageBox, "warning",
                        lambda *a, **k: warned.setdefault("msg", a[2]))
    started = {"n": 0}
    # the real guard: nothing may reach the worker mechanism at all. (It used
    # to patch a `_start_export_worker` that has never existed, so the
    # assertion below could not have caught an export that did run.)
    import pose3d.ui.main_window as main_window
    monkeypatch.setattr(main_window, "run_job",
                        lambda *a, **k: started.__setitem__("n", 1))

    win._on_export()
    assert "read-only" in warned.get("msg", "").lower(), warned
    assert started["n"] == 0, "export ran despite an unwritable destination"


# --- Phase 4: no diagnostic may be swallowed -------------------------------

def _write_calibration(folder, rig, mangle=None):
    """Write a calibration folder in the format `app._load_rig` reads."""
    import json
    calib = Path(folder) / "calibration"
    calib.mkdir(parents=True, exist_ok=True)
    rig.intr[CAM_LEFT].save(calib / "left_intrinsics.json")
    rig.intr[CAM_RIGHT].save(calib / "right_intrinsics.json")
    doc = {c: {"R": rig.ext[c].R.tolist(), "t": rig.ext[c].t.tolist()}
           for c in (CAM_LEFT, CAM_RIGHT)}
    if mangle is not None:
        mangle(doc)
    (calib / "extrinsics.json").write_text(json.dumps(doc))
    return calib


def test_missing_rig_reports_a_reason(qapp, tmp_path):
    """A calibration that is present but unusable must not be reported the
    same way as no calibration at all.

    One blanket `except Exception: return None` covered both, so a 2x2 R — a
    hand-edited or externally produced file, which is a thing clients send —
    built a perfectly happy CalibratedRig and only exploded inside
    triangulation frames later. The 3D view was empty and nothing anywhere in
    the app said why.
    """
    from pose3d.app import load_rig_with_reason
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()

    # 1. no folder at all: quiet. Being uncalibrated is a normal state.
    assert load_rig_with_reason(tmp_path / "nothing") == (None, "")

    # 2. a good folder loads
    good = tmp_path / "good"
    loaded, reason = load_rig_with_reason(_write_calibration(good, rig))
    assert loaded is not None and reason == ""

    # 3. a 2x2 R is refused, by name, before anything downstream sees it
    bad = tmp_path / "bad_shape"

    def flatten(doc):
        doc["right"]["R"] = [[1.0, 0.0], [0.0, 1.0]]

    loaded, reason = load_rig_with_reason(
        _write_calibration(bad, rig, flatten))
    assert loaded is None
    assert "right" in reason and "3x3" in reason

    # 4. so is an R that is not a rotation (a skew nothing else would catch)
    skew = tmp_path / "bad_rotation"

    def shear(doc):
        R = np.asarray(doc["left"]["R"], float)
        R[0] *= 1.4
        doc["left"]["R"] = R.tolist()

    loaded, reason = load_rig_with_reason(
        _write_calibration(skew, rig, shear))
    assert loaded is None
    assert "not a rotation" in reason

    # 5. and unreadable JSON says so rather than looking uncalibrated
    broken = tmp_path / "broken"
    calib = _write_calibration(broken, rig)
    (calib / "extrinsics.json").write_text("{ this is not json")
    loaded, reason = load_rig_with_reason(calib)
    assert loaded is None and "not valid JSON" in reason

    # 6. and the reason reaches the sidebar, not just a return value
    model = ProjectModel(data, None)
    model.rig_error = "This project's calibration is not usable — left camera."
    win = MainWindow(model)
    assert model.rig_error in win.sidebar.calib_warn.text()
    assert win.sidebar.calib_warn.isVisibleTo(win.sidebar)


def test_a_character_failure_reaches_the_3d_card(qapp):
    """`View3D._skin` used to swallow everything. A frame with no hips is
    normal and stays quiet; anything else is a fault and must be said."""
    from pose3d.geometry.character import PoseUnavailable
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))

    class NoPelvis:
        def pose_and_joints(self, *a, **k):
            raise PoseUnavailable("no usable pelvis")

        def ground_drop(self, *a, **k):
            return 0.0

    win.view3d._character = NoPelvis()
    win.view3d._char_error = ""
    win.view3d._skin(np.zeros((15, 3)))
    assert not win.view3d_error.isVisibleTo(win._view3d_card)

    class Broken(NoPelvis):
        def pose_and_joints(self, *a, **k):
            raise RuntimeError("character.npz is truncated")

    win.view3d._character = Broken()
    win.view3d._skin(np.zeros((15, 3)))
    assert win.view3d_error.isVisibleTo(win._view3d_card)
    assert "truncated" in win.view3d_error.text()

    # ...and it goes away again when the next frame poses. A per-frame fault
    # (a degenerate pose, a transient LinAlgError) used to pin the banner for
    # the rest of the session: `_report` de-duplicates on the last message and
    # only `fit_subject` ever cleared it, so the 3D card went on saying it was
    # "showing the captured skeleton only" over a perfectly good character.
    class Fine(NoPelvis):
        def pose_and_joints(self, *a, **k):
            return np.zeros((4, 3)), np.zeros((1, 3), int), np.zeros((15, 3))

    win.view3d._character = Fine()
    verts, faces, cj, drop = win.view3d._skin(np.zeros((15, 3)))
    assert verts is not None
    assert not win.view3d_error.isVisibleTo(win._view3d_card)
    assert win.view3d_error.text() == ""


def test_a_good_frame_does_not_withdraw_the_take_wide_warning(qapp):
    """A frame that skins perfectly withdraws only the message about THIS
    frame not skinning.

    `fit_subject`'s warning is about the whole take — the rig was never sized,
    so every frame is scaled by its own height ratio and the figure pulses —
    and every one of those frames skins fine. A per-frame "all clear" that
    cleared it would delete the warning on the very next repaint, on exactly
    the take it is about."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))

    class Unsizable:
        def fit_to_subject(self, poses):
            return None                      # the case that pulses

        def pose_and_joints(self, *a, **k):  # ...but posing works fine
            return np.zeros((4, 3)), np.zeros((1, 3), int), np.zeros((15, 3))

        def ground_drop(self, *a, **k):
            return 0.0

    win.view3d._character = Unsizable()
    win.view3d._char_error = ""
    win.view3d.fit_subject(np.stack([gt, gt]))
    assert "pulse" in win.view3d_error.text()

    win.view3d._skin(np.zeros((15, 3)))
    assert "pulse" in win.view3d_error.text(), (
        "a good frame deleted the take-wide sizing warning")
    assert win.view3d_error.isVisibleTo(win._view3d_card)


def test_an_unfittable_character_says_so_instead_of_pulsing(qapp):
    """`fit_to_subject` returning None drops the rig back on a PER-FRAME
    height ratio, which pulses the figure over a 37.9 % range on the client's
    take. It used to return silently."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))

    class Unfittable:
        def fit_to_subject(self, poses):
            return None

    win.view3d._character = Unfittable()
    win.view3d._char_error = ""
    win.view3d.fit_subject(np.stack([gt, gt]))
    assert win.view3d_error.isVisibleTo(win._view3d_card)
    assert "pulse" in win.view3d_error.text()


# --- Phase 4: set the scale from a measured distance ------------------------

def test_setting_the_scale_from_a_measured_height_rescales_the_rig(qapp,
                                                                   tmp_path):
    """The reconstruction is only as correctly SIZED as the marker length
    typed in at import, and nothing in the pipeline can detect that being
    wrong — a similarity leaves every reprojection, every epipolar distance
    and every bone-length spread identical. The one fix is a distance the
    client measures.
    """
    import json

    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    calib = _write_calibration(tmp_path, rig)
    (calib / "report.json").write_text(json.dumps(
        {"marker_length_m": 0.05, "world_tag_id": 15,
         "world_frame_id": "0007"}))

    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    before_t = {c: np.asarray(rig.ext[c].t, float).copy()
                for c in (CAM_LEFT, CAM_RIGHT)}
    before_h = model.measured_subject_height()
    assert np.isfinite(before_h) and before_h > 0

    target = before_h * 1.75
    factor = model.set_scale_from_height(target)
    assert factor == pytest.approx(1.75, rel=1e-6)

    # 1. the subject now reconstructs at the height that was measured
    assert model.measured_subject_height() == pytest.approx(target, rel=1e-3)

    # 2. it was done by scaling the translations, so no IMAGE measurement moved
    for cam in (CAM_LEFT, CAM_RIGHT):
        assert np.allclose(rig.ext[cam].t, before_t[cam] * factor)
    errs = model._accuracy(0)
    for cam in (CAM_LEFT, CAM_RIGHT):
        assert np.nanmax(errs[cam]["measured"]) < 1e-6, (
            "rescaling moved the reprojection: it is not a similarity")

    # 3. the calibration on disk was rewritten, marker length and all, so the
    #    next open and the next recalibration agree with this session
    saved = json.loads((calib / "extrinsics.json").read_text())
    assert np.allclose(saved["left"]["t"], before_t[CAM_LEFT] * factor)
    report = json.loads((calib / "report.json").read_text())
    assert report["marker_length_m"] == pytest.approx(0.05 * factor)
    assert report["world_tag_id"] == 15, "the provenance was thrown away"


def test_setting_the_scale_keeps_a_recorded_vertical(qapp, tmp_path):
    """The recorded vertical is a DIRECTION; a scale cannot touch it, and
    re-saving the rig must not quietly drop it — the 3D view and the export
    both level on it."""
    import json

    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    calib = _write_calibration(tmp_path, rig)
    doc = json.loads((calib / "extrinsics.json").read_text())
    doc |= {"world_up": [0.0, 0.0, 1.0], "world_up_source": "tag layout",
            "world_up_spread_deg": 3.0}
    (calib / "extrinsics.json").write_text(json.dumps(doc))

    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    model.set_scale_from_height(model.measured_subject_height() * 2.0)

    after = json.loads((calib / "extrinsics.json").read_text())
    assert after["world_up"] == [0.0, 0.0, 1.0]
    assert after["world_up_source"] == "tag layout"
    assert after["world_up_spread_deg"] == 3.0


def test_the_scale_button_is_wired_to_the_model(qapp, tmp_path):
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    _write_calibration(tmp_path, rig)
    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    win = MainWindow(model)

    win.sidebar.scale_value.setValue(50.0)          # 50 cm
    win.sidebar.btn_scale.click()
    assert model.measured_subject_height() == pytest.approx(0.50, rel=1e-3)


# --- Phase 4: the sidebar states what it measured ---------------------------

def test_the_sidebar_states_the_facts_a_ruler_can_check(qapp, tmp_path):
    import json

    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    calib = _write_calibration(tmp_path, rig)
    (calib / "report.json").write_text(json.dumps(
        {"marker_length_m": 0.05, "world_tag_id": 15,
         "world_frame_id": "0007"}))

    win = MainWindow(ProjectModel(data, rig, project_dir=str(tmp_path)))
    centres = [rig.ext[c].camera_center for c in (CAM_LEFT, CAM_RIGHT)]
    baseline_cm = 100.0 * float(np.linalg.norm(centres[0] - centres[1]))

    assert win.sidebar.row_baseline._v.text() == f"{baseline_cm:.1f} cm"
    assert "cm" in win.sidebar.row_height._v.text()
    marker = win.sidebar.row_marker._v.text()
    assert "50 mm" in marker and "tag 15" in marker and "0007" in marker

    # and the three rows reprojection cannot see
    for row in (win.sidebar.row_bone_cv, win.sidebar.row_epipolar,
                win.sidebar.row_symmetry):
        assert row._v.text() != "—", "a quality row was left blank"
    assert "%" in win.sidebar.row_bone_cv._v.text()
    assert "px" in win.sidebar.row_epipolar._v.text()


def test_a_project_calibrated_before_the_report_says_so_rather_than_guessing(
        qapp, tmp_path):
    """No report.json — every take calibrated before Phase 6 — must show "—"
    for the marker, not invent a tag id."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    _write_calibration(tmp_path, rig)
    win = MainWindow(ProjectModel(data, rig, project_dir=str(tmp_path)))
    assert win.sidebar.row_marker._v.text() == "—"
    assert "cm" in win.sidebar.row_baseline._v.text()


def test_a_typed_scale_survives_a_recompute(qapp, tmp_path):
    """The "Set scale" box is pre-filled with the height the app measured, but
    once the user has typed the height they MEASURED, that number is theirs.

    Every sidebar refresh — project load, Recalculate 3D, any qualityChanged —
    ran the pre-fill, so a user who typed 178.0 and then pressed "Recalculate
    3D" before "Set scale" watched their measurement silently revert to the
    app's own guess and rescaled the calibration to a number it had invented.
    """
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    _write_calibration(tmp_path, rig)
    win = MainWindow(ProjectModel(data, rig, project_dir=str(tmp_path)))

    prefilled = win.sidebar.scale_value.value()
    assert prefilled > 0, "the box should start at the measured height"

    # a refresh with nothing typed may still track the measurement
    win._refresh_quality()
    assert win.sidebar.scale_value.value() == prefilled

    win.sidebar.scale_value.setValue(178.0)     # the tape measure says so
    win._refresh_quality()
    assert win.sidebar.scale_value.value() == pytest.approx(178.0), (
        "a refresh overwrote a height the user typed")


def test_an_unreadable_calibration_report_says_so(qapp, tmp_path):
    """"No report.json" (a project older than Phase 6) and "report.json is
    corrupt" are different facts. Both used to show "—", so a take whose
    provenance WAS recorded but is unreadable looked like a take that never
    had any."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    calib = _write_calibration(tmp_path, rig)
    (calib / "report.json").write_text("{not json at all")

    win = MainWindow(ProjectModel(data, rig, project_dir=str(tmp_path)))
    text = win.sidebar.row_marker._v.text()
    assert text != "—", "an unreadable report read as 'nothing was recorded'"
    assert "report.json" in text and "could not be read" in text


def test_a_take_that_cannot_be_measured_is_measured_once(qapp, tmp_path):
    """`take_quality` walks every frame twice. When it raises, the failure is
    cached like a success: otherwise the one take the measurement cannot
    handle re-runs it — and re-emits the same status message — on every
    sidebar refresh, which is every frame change."""
    import pose3d.quality as quality
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    calls = {"n": 0}
    messages = []
    model.statusMessage.connect(messages.append)

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("this take cannot be measured")

    original = quality.take_quality
    quality.take_quality = boom
    try:
        assert model.quality() is None
        assert model.quality() is None
        assert model.quality() is None
    finally:
        quality.take_quality = original

    assert calls["n"] == 1, "the failed measurement re-ran on every refresh"
    assert len(messages) == 1 and "cannot be measured" in messages[0]

    # ...and invalidating the readouts lets it be tried again
    model.invalidate_readouts()
    assert model.quality() is not None


def test_the_fixed_camera_is_silent_for_a_project_with_no_folder(qapp, capsys):
    """A project that has not been saved anywhere has no calibration folder to
    look in, and that is a normal state — the `FileNotFoundError` branch above
    exists to keep it quiet. `Path(None)` raises TypeError instead, which the
    generic handler printed as "the project's calibration could not be read":
    a fault message for a project that is simply new.
    """
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))          # no project_dir
    assert win.model.project_dir is None
    capsys.readouterr()
    assert win._left_camera() is None
    assert capsys.readouterr().out == ""


# --- Phase G: the long jobs leave the GUI thread ---------------------------
#
# Detection, the face re-detect and the recompute used to run in the slot that
# started them, so Windows painted "Not Responding" over a five-minute job and
# there was no way to stop it. They now go through `ui.worker.run_job` with
# the model's redraw signals held back — which is only safe if the window
# refreshes ONCE afterwards, and only bearable if the status line keeps
# talking while it runs.

class _NoseDetector:
    head_source = "nose"


class _SkullDetector:
    head_source = "skull"


def test_a_detection_refreshes_once_and_keeps_talking(qapp, tmp_path):
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, gt = _project_with_rig()
    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    win = MainWindow(model)
    win.detector = _NoseDetector()

    refreshes = []
    win._refresh_views = lambda: refreshes.append(1)

    def fake_redetect_all(det, load_image, on_progress=None, cancelled=None):
        model.statusMessage.emit("Running detection…")
        for i in range(3):
            model.set_frame(i)          # would repaint the window, per frame
            on_progress(i + 1, 3, f"frame {i + 1} of 3")

    model.redetect_all = fake_redetect_all
    win._on_run_detection()

    assert refreshes == [1], "the window redrew once per frame, not once"
    assert "Running detection" in win.statusBar().currentMessage()


def test_a_cancelled_detection_puts_the_head_convention_back(
        qapp, recorded_errors):
    """The convention is adopted BEFORE the run, because the 2D is about to be
    replaced. A run that does not finish replaces nothing, so the project, the
    process-wide default and the view's cached character all go back."""
    from pose3d.geometry import character as ch
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel
    from pose3d.ui.worker import Cancelled

    class _Cached:
        head_source = "nose"

    data, rig, gt = _project_with_rig()
    data.head_source = "nose"
    try:
        model = ProjectModel(data, rig)
        win = MainWindow(model)
        win.detector = _SkullDetector()
        cached = _Cached()
        win.view3d._character = cached

        def fake_redetect_all(det, load_image, on_progress=None, cancelled=None):
            raise Cancelled()

        model.redetect_all = fake_redetect_all
        win._on_run_detection()

        assert data.head_source == "nose"
        assert ch.default_head_source() == "nose"
        assert win.view3d._character is cached
        assert "cancel" in win.statusBar().currentMessage().lower()
        assert recorded_errors == [], "a cancel is not a failure to report"
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_a_cancelled_export_says_so_and_removes_the_half_written_file(
        qapp, tmp_path, monkeypatch, recorded_errors):
    from pose3d.export import blender_export
    from pose3d.ui import filedialog
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(filedialog, "existing_directory", lambda *a, **k: str(out))
    monkeypatch.setattr(filedialog, "is_writable", lambda p: True)

    def fake_export(poses, out_dir, name="pose3d", **kw):
        # what the real export has already written by the time Blender is
        # launched, and what a cancel therefore leaves behind
        (Path(out_dir) / f"{name}_poses.json").write_text("{}")
        return blender_export._failure(
            "blender_cancelled", "Export cancelled.", 125)

    monkeypatch.setattr(blender_export, "export_animation", fake_export)

    data, rig, gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))
    win._on_export()

    assert not (out / f"{data.name}_poses.json").exists()
    assert "Export cancelled" in win.statusBar().currentMessage()
    assert recorded_errors == []


def test_the_import_copy_loop_reports_every_pair(tmp_path):
    """Where the import stalls is inside one `copy2` — a virus scanner or a
    OneDrive placeholder — so the copy loop reports per pair."""
    from pose3d.core.importer import build_project

    src = tmp_path / "src"
    src.mkdir()
    left, right = [], []
    for i in range(3):
        lp = src / f"left_{i}.jpg"; lp.write_bytes(b"x"); left.append(lp)
        rp = src / f"right_{i}.jpg"; rp.write_bytes(b"x"); right.append(rp)

    seen = []
    project = build_project(left, right, name="p", copy_into=tmp_path / "proj",
                            on_progress=lambda d, t, label: seen.append((d, t)))
    assert len(project.frames) == 3
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_the_import_dialog_no_longer_pumps_the_event_loop_by_hand():
    """`processEvents` re-enters every slot in the app from the middle of the
    import; the long phases run on a worker thread instead."""
    import inspect

    from pose3d.ui import import_dialog

    src = inspect.getsource(import_dialog)
    assert "processEvents" not in src
    assert "run_job(" in src

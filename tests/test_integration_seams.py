"""The seams five parallel tasks left between each other, closed and pinned.

Each task was built, reviewed and merged on its own branch, so every fact that
crosses two of them — a signal one task emits and another must answer, a rule
two tasks each stated, a value one returns and another must show — had nobody
to hold it. These tests are that holder: they assert the JOIN, end to end
where the join is only observable end to end, so a future edit to either side
cannot quietly take it apart again.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pose3d.core.project import CAM_LEFT, CAM_RIGHT                # noqa: E402
from pose3d.core.skeleton import NUM_JOINTS, Joint, face_kp_id     # noqa: E402
from tests.test_ui_smoke import _project_with_rig                  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------
# Seam 1 — the 3D joint colouring T4 built had no hook to reach it
# --------------------------------------------------------------------------

def _window_with_three_kinds_of_joint(tmp_path):
    """A window whose current frame holds a corrected, a missing and an
    ordinary joint — the three cases the client's complaint names."""
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, _gt = _project_with_rig()
    f = data.frames[0]
    # missing in BOTH views: no 3D was ever reconstructed for it
    missing = int(Joint.LEFT_WRIST)
    for cam in (CAM_LEFT, CAM_RIGHT):
        f.kp2d[cam][missing] = np.nan
        f.scores[cam][missing] = np.nan
    f.pose3d[missing] = np.nan
    f.fitted3d[missing] = np.nan

    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    win = MainWindow(model)
    model.set_frame(0)
    # a hand correction, made the way the user makes one
    corrected = int(Joint.RIGHT_KNEE)
    xy = f.kp2d[CAM_LEFT][corrected]
    model.set_joint_2d(CAM_LEFT, corrected, float(xy[0]) + 3.0, float(xy[1]))
    xy = f.kp2d[CAM_RIGHT][corrected]
    model.set_joint_2d(CAM_RIGHT, corrected, float(xy[0]) + 3.0, float(xy[1]))
    return win, model, {"missing": missing, "corrected": corrected}


def test_the_3d_view_is_banded_by_the_same_rule_as_the_camera_panels(
        qapp, tmp_path):
    """The whole point of T4's banding, and it reached nothing.

    `View3D.set_joint_status` was written, tested and merged, and no line in
    `MainWindow` ever called it — so on the running app every 3D joint stayed
    one cyan, which is the client's 2026-07-26 complaint verbatim. Asserted
    against the CAMERA PANEL's own answer for the same joint, not against a
    copy of the rule.
    """
    win, model, j = _window_with_three_kinds_of_joint(tmp_path)

    statuses = win.view3d._joint_statuses(model.frame().filled)
    assert statuses is not None, "the 3D view was never told the frame's status"

    for joint in range(NUM_JOINTS):
        left = win.cam_left.view._joint_status(joint)[0]
        right = win.cam_right.view._joint_status(joint)[0]
        if left == right:          # the merge has nothing to choose between
            assert statuses[joint] == left, (
                f"joint {joint}: 3D says {statuses[joint]!r}, "
                f"the camera panels say {left!r}")

    assert statuses[j["corrected"]] == "corrected"
    assert statuses[j["missing"]] == "unmeasured"


def test_the_3d_joints_are_actually_drawn_in_those_colours(qapp, tmp_path):
    """…and the colours reach the scatter, not just the status cache.

    The pose and the banding arrive on two different signals and nothing
    orders them, so a hook that ran before the pose would leave the drawing
    uncoloured.
    """
    from pose3d.ui.camera_view import RAG_COLORS
    from pose3d.ui.view3d import _status_rgba

    win, model, j = _window_with_three_kinds_of_joint(tmp_path)
    colors = np.asarray(win.view3d._scatter.color, float)
    assert colors.ndim == 2, "the 3D joints were drawn in one flat colour"

    # the overlay draws the joints its own mask keeps, in that order
    _pts, mask, _filled = win.view3d._last_draw
    drawn = np.flatnonzero(mask)
    row = int(np.flatnonzero(drawn == j["corrected"])[0])
    assert np.allclose(colors[row], _status_rgba("corrected"))
    assert RAG_COLORS["corrected"].name() != RAG_COLORS["green"].name()


# --------------------------------------------------------------------------
# Seam 2 — T2 and T4 each wrote the joint-status rule
# --------------------------------------------------------------------------

def test_the_camera_view_asks_the_shared_joint_status_rule(qapp, monkeypatch):
    """Two copies of the rule is the defect, not two copies that agree today.

    `camera_view._joint_status` (T2) and `panels.joint_status` (T4) stated the
    same five-clause order independently. They agree on every input as merged,
    which is exactly why nothing would catch the next edit to one of them —
    and "the two panels disagree about which joints are flagged" is the
    client's own complaint. So this pins the CALL: the camera view must get
    its band from the shared function, not from a copy that matches it.
    """
    from pose3d.ui import camera_view as cv

    monkeypatch.setattr(cv, "joint_status", lambda *a, **k: "corrected")
    panel = cv.CameraPanel("LEFT VIEW", CAM_LEFT)
    panel.view.set_pose(np.zeros((NUM_JOINTS, 2)), np.ones(NUM_JOINTS))
    panel.set_accuracy(np.full(NUM_JOINTS, 0.001), None, None)

    assert panel.view._joint_status(0)[0] == "corrected"


# --------------------------------------------------------------------------
# Seam 3 — T1 gave face edits their own flag; T2's dot and filter never asked
# --------------------------------------------------------------------------

def test_a_face_only_edit_turns_the_frames_dot_purple(qapp, tmp_path):
    """An eye placed by hand IS a correction to that frame.

    T1 split the flags in two — `Frame.corrected` for the body joints,
    `Frame.head_corrected` for the face points — and gave the question one
    answer, `Frame.has_corrections()`. T2's timeline had already shipped
    reading `Frame.corrected` alone, so a frame whose only hand work was on
    the face reported itself uncorrected: its dot stayed green and it was
    missing from Show = Corrected, which is the one view that exists to find
    the frames the user has worked on.

    An EYE, in Face mode, because that is a dot the user can actually reach:
    under the nose convention the camera panels draw no separate nose handle
    (the canonical HEAD dot IS it — see `ui.model._resolve_joint`), so a test
    that edited face point 0 would prove the flag without ever touching a
    point this window offers.
    """
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data, rig, _gt = _project_with_rig()
    data.head_mode = "face"           # the mode the eyes and ears are drawn in
    for f in data.frames:
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.head2d[cam][:] = 100.0
            f.head_scores[cam][:] = 1.0
    model = ProjectModel(data, rig, project_dir=str(tmp_path))
    win = MainWindow(model)
    model.set_frame(1)
    assert win.timeline.status(1) == "green"

    eye = face_kp_id(1)               # left eye: face point 1
    handle = win.cam_left.view._face[1]
    assert handle.isVisible() and handle.joint_id == eye, \
        "this face point is not one the user can drag"
    model.set_joint_2d(CAM_LEFT, eye, 140.0, 160.0)

    assert data.frames[1].has_corrections()
    assert data.frames[1].head_corrected[CAM_LEFT][1]
    assert not data.frames[1].corrected[CAM_LEFT].any(), "a FACE edit only"
    assert win.timeline.status(1) == "corrected"

    win.timeline_header.show_combo.setCurrentIndex(
        win.timeline_header.show_combo.findData("corrected"))
    assert not win.timeline.isRowHidden(1)
    assert win.timeline.isRowHidden(0) and win.timeline.isRowHidden(2)


def test_a_reopened_project_shows_its_face_corrections_too(qapp):
    """`populate` asks the same question — a project carries its corrections
    back from disk, and the filter was empty in every reopened one."""
    from pose3d.ui.timeline import Timeline

    data, _rig, _gt = _project_with_rig()
    data.frames[2].set_head_kp(CAM_RIGHT, 4, 10.0, 20.0, corrected=True)

    strip = Timeline()
    strip.populate(data.frames, load_thumb=None)

    assert strip.status(2) == "corrected"
    assert strip.status(0) == "green"


# --------------------------------------------------------------------------
# Seam 4 — T4's `ExportResult.fit_note` was returned and never shown
# --------------------------------------------------------------------------

def _export_dialog_text(qapp, tmp_path, monkeypatch, fit_note):
    """Run `_on_export` against a fake Blender and return what it told the
    user."""
    from PySide6.QtWidgets import QMessageBox
    from pose3d.export import blender_export
    from pose3d.ui import filedialog
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(filedialog, "existing_directory",
                        lambda *a, **k: str(out))
    monkeypatch.setattr(filedialog, "is_writable", lambda p: True)

    def fake_export(poses, out_dir, name="pose3d", **kw):
        d = out / f"{name}.bvh"
        d.write_text("")
        return blender_export.ExportResult(
            bvh=d, fbx=None, mp4=None, returncode=0,
            stdout="POSE3D_EXPORT_OK", stderr="", fit_note=fit_note)

    monkeypatch.setattr(blender_export, "export_animation", fake_export)

    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[2])))

    data, rig, _gt = _project_with_rig()
    win = MainWindow(ProjectModel(data, rig))
    win._on_export()
    assert shown, "the export reported nothing at all"
    return shown[-1]


def test_the_export_says_when_the_character_was_sized_the_rough_way(
        qapp, tmp_path, monkeypatch):
    """The figure in the file is the right character posed by the right rule
    and sized by a ROUGHER measurement — a fact about the delivery, not a
    failure of it. T4 put it on `ExportResult.fit_note`, the dialog T3 owns
    never read it, and the 3D view said so while the exported file did not.
    """
    from pose3d.geometry.placement import SCALE_FROM_HEIGHT_NOTE

    body = _export_dialog_text(qapp, tmp_path, monkeypatch,
                               SCALE_FROM_HEIGHT_NOTE)

    assert body.startswith("Wrote:")
    assert SCALE_FROM_HEIGHT_NOTE in body


def test_a_normally_sized_export_adds_no_note(qapp, tmp_path, monkeypatch):
    """The note is for the exception. An empty one must not leave a dangling
    blank paragraph on every ordinary export."""
    from pose3d.geometry.placement import SCALE_FROM_HEIGHT_NOTE

    body = _export_dialog_text(qapp, tmp_path, monkeypatch, "")
    assert body.rstrip() == body
    assert SCALE_FROM_HEIGHT_NOTE not in body
    assert "could not be sized" not in body


# --------------------------------------------------------------------------
# Seam 5 — one unreadable photo still took the whole import down
# --------------------------------------------------------------------------

class _FlatDetector:
    """A detector that answers the same pose for any image it is given."""
    head_source = "nose"
    keypoint_model = "coco17"
    provenance = "flat-stub"

    def detect(self, image_bgr):
        from pose3d.detect.base import Detection
        from pose3d.core.skeleton import NUM_HEAD_KP
        xy = np.tile(np.array([10.0, 20.0]), (NUM_JOINTS, 1))
        return Detection(xy=xy, scores=np.ones(NUM_JOINTS),
                         head_xy=np.tile(np.array([11.0, 21.0]),
                                         (NUM_HEAD_KP, 1)),
                         head_scores=np.ones(NUM_HEAD_KP))


def _photo_pairs(tmp_path, n=4, blank_pair=2, blank_cam="left"):
    """`n` real image pairs on disk, one photo of which is a 0-byte file.

    The OneDrive online-only placeholder: the file exists, the copy succeeds,
    and every reader gets nothing.
    """
    import cv2
    tmp_path.mkdir(parents=True, exist_ok=True)
    lefts, rights = [], []
    img = np.full((48, 64, 3), 120, np.uint8)
    for i in range(n):
        lp, rp = tmp_path / f"l{i}.png", tmp_path / f"r{i}.png"
        cv2.imwrite(str(lp), img)
        cv2.imwrite(str(rp), img)
        lefts.append(lp); rights.append(rp)
    (lefts if blank_cam == "left" else rights)[blank_pair].write_bytes(b"")
    return [str(p) for p in lefts], [str(p) for p in rights]


def test_an_unreadable_photo_does_not_abort_the_whole_import(
        qapp, tmp_path, monkeypatch, recorded_errors):
    """CONFIRMED CRITICAL by T4's reviewer, and only visible end to end.

    T4 taught the CALIBRATION to skip a pair it cannot read. Detection still
    raised `ImageReadError` on the same file, the dialog's outer handler
    caught it, and the import ended with "Import failed" and nothing written —
    after the copy, after the calibration, on a take whose other pairs were
    all fine. A 0-byte OneDrive placeholder is the client's own case.
    """
    from PySide6.QtWidgets import QMessageBox
    from pose3d.ui import import_dialog

    lefts, rights = _photo_pairs(tmp_path / "src")
    monkeypatch.setattr(import_dialog, "run_job",
                        lambda parent, title, fn, cancellable=True:
                        fn(lambda *a, **k: None, lambda: False))
    # the "open for 2D review anyway?" question goes through the one yes/no
    # seam; answer Yes (the autouse fixture would say No)
    from pose3d.ui import guard
    monkeypatch.setattr(guard, "ask_yes_no", lambda parent, title, text: True)
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a[2])))

    out = tmp_path / "projects"
    dlg = import_dialog.ImportDialog(projects_root=str(out))
    dlg._detector = _FlatDetector()
    dlg.left_pick.paths = lefts
    dlg.right_pick.paths = rights
    dlg.out_pick.paths = [str(out)]
    dlg.name.setText("Placeholder_Take")
    dlg._process()

    assert recorded_errors == [], recorded_errors
    assert dlg.result_folder, "the import wrote nothing"
    from pose3d.core.io_project import load_project
    project = load_project(dlg.result_folder)
    assert len(project.frames) == 4

    # the three readable pairs were detected in both views...
    for i in (0, 1, 3):
        for cam in (CAM_LEFT, CAM_RIGHT):
            assert np.isfinite(project.frames[i].kp2d[cam]).all()
    # ...and the one that could not be read is MISSING, not invented
    assert np.isnan(project.frames[2].kp2d[CAM_LEFT]).all()
    assert np.isfinite(project.frames[2].kp2d[CAM_RIGHT]).all()

    # and the user is told which pair it was
    assert shown, "the import summary was never shown"
    assert project.frames[2].frame_id in shown[-1]


def test_detect_project_records_the_pairs_it_could_not_read(tmp_path):
    """The pipeline half of the same fact, asked directly: a skip is recorded
    in the shape the calibration records one, so the caller has a reason to
    show and not just a hole in the data."""
    from pose3d.core.importer import build_project
    from pose3d.imageio import read_image
    from pose3d.pipeline import detect_project

    lefts, rights = _photo_pairs(tmp_path / "src")
    project = build_project(lefts, rights, name="t",
                            copy_into=tmp_path / "proj")
    skipped = []
    detect_project(project, _FlatDetector(), read_image, skipped=skipped)

    assert [s["camera"] for s in skipped] == [CAM_LEFT]
    assert skipped[0]["frame"] == project.frames[2].frame_id
    assert "0 bytes" in skipped[0]["reason"]


def test_a_re_detect_that_could_not_read_a_photo_says_so(tmp_path):
    """A skip must never be SILENT.

    Before the fix an unreadable photo raised, and Run Detection at least
    failed loudly. Now it is skipped — and the re-detect would otherwise
    report "Detection complete (4 frames); hand-corrected points were kept"
    over a frame whose view it never looked at, which is the failure mode the
    raise was protecting against.
    """
    from pose3d.core.importer import build_project
    from pose3d.imageio import read_image
    from pose3d.ui.model import ProjectModel

    lefts, rights = _photo_pairs(tmp_path / "src")
    project = build_project(lefts, rights, name="t",
                            copy_into=tmp_path / "proj")
    model = ProjectModel(project, None)
    said = []
    model.statusMessage.connect(said.append)

    model.redetect_all(_FlatDetector(), read_image)

    assert any("could not be read" in m for m in said), said
    assert any(project.frames[2].frame_id in m for m in said), said


def _take_of_placeholders(tmp_path):
    """A calibrated take every one of whose photos is a 0-byte file.

    OneDrive evicts a FOLDER, not a file, so this — not one bad photo — is
    the shape the client is most likely to hit.
    """
    data, rig, _gt = _project_with_rig()
    for i, f in enumerate(data.frames):
        for cam, stem in ((CAM_LEFT, "l"), (CAM_RIGHT, "r")):
            p = tmp_path / f"{stem}{i}.png"
            p.write_bytes(b"")
            f.images[cam] = str(p)
    return data, rig


def test_a_face_re_detect_that_read_nothing_blames_the_files(tmp_path):
    """Not the build. `detect_project` stages nothing when no photo opens, so
    `wrote` is 0 — the same 0 a detector with no face model returns — and the
    early return told the user "This build's detector does not produce face
    points", which is a sentence about the wrong thing entirely. The photos
    were never opened, so the detector was never even asked.
    """
    from pose3d.imageio import read_image
    from pose3d.ui.model import ProjectModel

    data, rig = _take_of_placeholders(tmp_path)
    model = ProjectModel(data, rig)
    said = []
    model.statusMessage.connect(said.append)

    model.redetect_head(_FlatDetector(), read_image)

    joined = " ".join(said)
    assert "could not be read" in joined, said
    assert data.frames[0].frame_id in joined, said
    assert "0 bytes" in joined, said
    assert "does not produce face points" not in joined, said


class _NoFaceDetector(_FlatDetector):
    """A build whose detector has no face model — the real `wrote == 0`."""

    def detect(self, image_bgr):
        from pose3d.detect.base import Detection
        return Detection(xy=np.zeros((NUM_JOINTS, 2)),
                         scores=np.ones(NUM_JOINTS))


def test_a_build_with_no_face_model_is_still_named_as_the_reason(tmp_path):
    """…and when photos DID open and the detector still gave nothing, the
    build is the reason and must still be named — including when some other
    photo was unreadable, where both facts are true at once."""
    from pose3d.imageio import read_image
    from pose3d.ui.model import ProjectModel

    data, rig, _gt = _project_with_rig()
    lefts, rights = _photo_pairs(tmp_path / "src", n=len(data.frames))
    for f, lp, rp in zip(data.frames, lefts, rights):
        f.images = {CAM_LEFT: lp, CAM_RIGHT: rp}

    model = ProjectModel(data, rig)
    said = []
    model.statusMessage.connect(said.append)

    model.redetect_head(_NoFaceDetector(), read_image)

    joined = " ".join(said)
    assert "does not produce face points" in joined, said
    # ...and the one photo that could not be read is not swept under it
    assert data.frames[2].frame_id in joined, said


def test_the_skip_sentence_does_not_claim_the_whole_pair_was_lost(tmp_path):
    """A skip costs one (frame, camera) — the readable half of the pair IS
    detected — so "N image pair(s) were skipped" overstated the loss to the
    one person who has to decide whether to re-import.

    The sentence says what is true of every caller: those frames had a photo
    that could not be read. What it COST is the caller's own clause, because
    the calibration does lose the whole frame while the detection keeps the
    view it could read.
    """
    from pose3d.pipeline import summarise_unreadable

    note = summarise_unreadable(
        [{"frame": "0007", "camera": CAM_LEFT, "reason": "0 bytes"}],
        "the import kept the rest")

    assert "pair(s) were skipped" not in note
    assert "1 frame(s) had a photo that could not be read (0007)" in note
    assert "the import kept the rest" in note
    assert "The first was: 0 bytes" in note
    assert summarise_unreadable([], "anything") == ""


def test_a_loader_that_returns_none_is_skipped_by_detection_too(tmp_path):
    """`cv2.imread` answers None where `read_image` raises, and the tools and
    `pose3d.quality` still pass a plain `cv2.imread`."""
    import cv2
    from pose3d.core.importer import build_project
    from pose3d.pipeline import detect_project

    lefts, rights = _photo_pairs(tmp_path / "src", blank_pair=0)
    project = build_project(lefts, rights, name="t",
                            copy_into=tmp_path / "proj")
    bad = project.frames[1].images[CAM_RIGHT]

    def loader(path):
        return None if str(path) == str(bad) else cv2.imread(str(path))

    skipped = []
    detect_project(project, _FlatDetector(), loader, skipped=skipped)

    assert {(s["frame"], s["camera"]) for s in skipped} == {
        (project.frames[0].frame_id, CAM_LEFT),
        (project.frames[1].frame_id, CAM_RIGHT)}


# --------------------------------------------------------------------------
# Seam 6 — the sidebar read the marker back in a unit the box does not use
# --------------------------------------------------------------------------

def test_the_sidebar_reads_the_marker_back_in_the_unit_it_was_typed_in(qapp):
    """T5 made the import box centimetres because the client measures with a
    ruler ("5cm x 5cm Aruco markers"). The sidebar went on printing the
    stored metres as "50 mm", so the one number the user typed came back in a
    third unit — and the two rows beside it are already centimetres."""
    from pose3d.ui.panels import _marker_text

    assert _marker_text({"marker_length_m": 0.05}) == "5.0 cm"
    assert _marker_text({"marker_length_m": 0.05, "world_tag_id": 15,
                         "world_frame_id": "0007"}) == \
        "5.0 cm · tag 15 · frame 0007"
    assert _marker_text({}) == "—"
    assert _marker_text({"world_tag_id": 15}) == "tag 15"


def test_the_marker_row_uses_the_same_formatting_as_the_rows_beside_it(qapp):
    """One formatter, so a change of precision cannot leave the three
    ruler-checkable rows disagreeing."""
    from pose3d.ui.panels import _cm, _marker_text

    assert _marker_text({"marker_length_m": 0.083}).startswith(_cm(0.083))
    assert _cm(0.083) == "8.3 cm"


# --------------------------------------------------------------------------
# Seam 7 — "the cameras moved" was decided in two places
# --------------------------------------------------------------------------

def test_one_camera_moved_rule(monkeypatch):
    """`camera_motion_check` (T4's file) and `model._rescale_report` (T1's)
    each compared a rotation and a displacement against the two thresholds.
    A rescale re-takes the verdict because it changes the displacement, so
    the two had to agree — and nothing made them.
    """
    from pose3d.calib import resolve
    from pose3d.ui import model as uimodel

    assert not resolve.camera_moved(0.0, 0.0)
    assert resolve.camera_moved(resolve.MOTION_WARN_DEG + 0.1, 0.0)
    assert resolve.camera_moved(0.0, resolve.MOTION_WARN_MM + 0.1)
    assert not resolve.camera_moved(None, 0.0), "a missing angle is not motion"

    # both callers go through it: swap the rule and both verdicts follow
    monkeypatch.setattr(resolve, "camera_moved", lambda rot, cen: True)
    report = {"max_rotation_deg": 0.0, "max_centre_mm": 0.0}
    uimodel._rescale_report(report, 2.0)
    assert report["moved"] is True


def test_a_rescale_re_takes_the_verdict_from_the_scaled_displacement():
    """The number the rule judges is a LENGTH, so rescaling the calibration
    can carry it over the threshold (or back under it) — and the provenance
    record must not go on claiming the cameras held still."""
    from pose3d.calib.resolve import MOTION_WARN_MM
    from pose3d.ui.model import _rescale_report

    report = {"max_rotation_deg": 0.0, "max_centre_mm": 0.6 * MOTION_WARN_MM,
              "moved": False}
    _rescale_report(report, 2.0)
    assert report["max_centre_mm"] == pytest.approx(1.2 * MOTION_WARN_MM)
    assert report["moved"] is True

    _rescale_report(report, 0.25)
    assert report["moved"] is False


# --------------------------------------------------------------------------
# Seam 11 — the total-failure message contradicted itself
# --------------------------------------------------------------------------

def test_a_calibration_that_read_nothing_does_not_claim_it_used_the_rest(
        tmp_path):
    """The clause "…the calibration used the rest" is the sentence for a take
    that lost a pair. On the branch where NO pair could be read there is no
    rest, and the message said both things one after the other."""
    from pose3d.calib.resolve import resolve_calibration
    from pose3d.core.importer import build_project
    from pose3d.imageio import read_image

    lefts, rights = _photo_pairs(tmp_path / "src", n=3, blank_pair=0)
    for p in lefts + rights:
        open(p, "wb").close()
    project = build_project(lefts, rights, name="t",
                            copy_into=tmp_path / "proj")

    res = resolve_calibration(project, read_image, marker_length=0.05)

    assert not res.ok
    assert "no image pair could be read" in res.message.lower()
    assert "used the rest" not in res.message
    assert "none could be used" in res.message
    assert project.frames[0].frame_id in res.message


def test_a_calibration_that_lost_one_pair_still_says_it_used_the_rest():
    """…and the ordinary sentence is unchanged."""
    from pose3d.calib.resolve import summarise_skipped

    note = summarise_skipped([{"frame": "0007", "camera": CAM_LEFT,
                               "reason": "0 bytes"}])
    assert note.startswith(" ")          # appended to a finished sentence
    assert "the calibration used the rest" in note
    assert "0007" in note


# --------------------------------------------------------------------------
# Seam 12 — the size fallback depended on the caller's frame with no body axis
# --------------------------------------------------------------------------

def test_the_size_fallback_is_the_same_whichever_way_the_take_is_turned():
    """The view hands `take_scale` RAW world poses until the window has given
    it an orientation; the export always de-tilts first. `take_scale` de-tilts
    for both so they agree by construction — but when `sequence_up` cannot
    find a body axis the de-tilt was SKIPPED and the height read straight off
    the z extent, which is the caller's frame again. Two callers, two sizes,
    one figure, and the invariant this whole pass rests on is that the export
    poses and sizes the character exactly as the live view does.
    """
    from scipy.spatial.transform import Rotation

    from pose3d.geometry.placement import take_scale

    class _Rig:
        rig_h = 1.7

        def fit_to_subject(self, poses):
            return None                   # no bone could be measured

    # a pose with no body axis at all: `sequence_up` needs a spine
    rng = np.random.default_rng(7)
    poses = np.full((3, NUM_JOINTS, 3), np.nan)
    for i in range(3):
        poses[i, :4] = rng.normal(size=(4, 3))

    from pose3d.geometry.orient import sequence_up
    assert sequence_up(poses) is None, "this fixture must have no body axis"

    upright, _ = take_scale(_Rig(), poses)
    R = Rotation.from_euler("xyz", [40.0, 25.0, 70.0], degrees=True).as_matrix()
    turned, _ = take_scale(_Rig(), poses @ R.T)

    assert upright is not None
    assert turned == pytest.approx(upright, rel=1e-9)


# --------------------------------------------------------------------------
# Seam 10 — `canceled` no longer means what a future connector would assume
# --------------------------------------------------------------------------

def test_the_job_dialog_says_what_its_canceled_signal_now_does(qapp):
    """`run_job` takes Qt's `clicked -> canceled` wire out, because Qt wires
    `canceled` to `QProgressDialog::cancel()` — the hide this class exists to
    prevent. The signal is therefore NOT the way to hear about a Cancel any
    more: it fires zero times per click, and once at teardown. That is
    exactly the kind of fact the next person connects to and gets wrong, so
    the class has to say it where they will look.
    """
    from pose3d.ui.worker import _JobDialog

    doc = _JobDialog.__doc__ or ""
    assert "canceled" in doc
    assert "on_stop" in doc
    low = doc.lower()
    assert "teardown" in low and ("zero" in low or "never" in low)


def test_a_real_cancel_click_emits_canceled_zero_times(qapp, monkeypatch,
                                                       recorded_errors):
    """…and the documented count is the real one.

    A connector that took `canceled` to mean "the user pressed Cancel" would
    hear nothing at the moment it happened and then hear it once, later, on a
    job that had already stopped — which is the worst of both. Pressed here
    through the REAL button, on a job that keeps running until it is asked to
    stop.
    """
    import time

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QPushButton
    from pose3d.ui import worker

    dialogs, fired = [], []
    real = worker._JobDialog

    def make(*a, **kw):
        dlg = real(*a, **kw)
        dlg.canceled.connect(lambda: fired.append(time.monotonic()))
        dialogs.append(dlg)
        return dlg

    monkeypatch.setattr(worker, "_JobDialog", make)

    def until_cancelled(report, cancelled):
        deadline = time.monotonic() + 10
        while not cancelled() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise worker.Cancelled()

    def press():
        if dialogs:
            dialogs[0].findChild(QPushButton).click()
            clicked.append(time.monotonic())
        else:
            QTimer.singleShot(10, press)

    clicked: list[float] = []
    QTimer.singleShot(0, press)
    res = worker.run_job(None, "Detection", until_cancelled)

    assert isinstance(res, worker.Cancelled)
    assert clicked, "the Cancel button was never pressed"
    # zero per click, and at most the one at teardown — which is after the
    # job has already stopped, so it can never be the cancel notification
    assert len(fired) <= 1
    assert all(t >= clicked[0] for t in fired)
    assert recorded_errors == []


@pytest.mark.parametrize("state", ["ok", "rejected", "not_measured"])
@pytest.mark.parametrize("err", [0.0005, 0.007, 0.05, float("nan")])
@pytest.mark.parametrize("filled,corrected", [(False, False), (True, False),
                                              (False, True), (True, True)])
def test_both_panels_band_a_joint_the_same_way(state, err, filled, corrected):
    """…and the answer itself is unchanged for every combination."""
    from pose3d.ui.panels import joint_status
    from pose3d.ui.model import (
        STATE_NOT_MEASURED, STATE_OK, STATE_REJECTED)

    states = {"ok": STATE_OK, "rejected": STATE_REJECTED,
              "not_measured": STATE_NOT_MEASURED}
    got = joint_status(states[state], err, filled, corrected)
    if corrected:
        assert got == "corrected"
    elif filled:
        assert got == "filled"
    elif state == "rejected":
        assert got == "rejected"
    elif state == "not_measured" or not np.isfinite(err):
        assert got == "unmeasured"
    else:
        assert got in ("green", "amber", "red")

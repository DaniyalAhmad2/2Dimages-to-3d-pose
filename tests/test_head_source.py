"""What the canonical HEAD point IS, and the one correction that depends on it.

A COCO-17 project's HEAD is the NOSE, which sits ~45 deg forward of the torso
line, so the retarget rotates the neck's aim back by that anatomical offset
before using it. A Halpe-26 project's HEAD is the SKULL VERTEX, already on the
head's axis, and applying the same offset to it would be a second correction
on data that needs none. `head_source` is what tells the two apart; these are
the tests that keep it honest in both directions.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from pose3d.core.skeleton import Joint
from tests.gates import needs_character
from tests.synth import sample_skeleton_3d

#: The process-wide head convention as `pose3d.geometry.character` ships it,
#: read once at import so the autouse fixture below restores THAT rather than
#: whatever a test left set.
_IMPORTED_DEFAULT = __import__(
    "pose3d.geometry.character", fromlist=["x"]).default_head_source()

#: The process-wide head MODE as the module ships it, read the same way and
#: for the same reason. It has to be restored as carefully as the convention
#: above: a "face" default left behind by a UI test reaches
#: `measure_gate_table_on_fixture()` later in this very file and re-poses
#: every frame the gate table is measured on.
_IMPORTED_DEFAULT_MODE = __import__(
    "pose3d.geometry.character", fromlist=["x"]).default_head_mode()


def _pitch_head(pose, deg):
    """Pitch the HEAD point forward about the NECK by `deg`."""
    pose = pose.copy()
    a = np.radians(deg)
    R = np.array([[1, 0, 0],
                  [0, np.cos(a), -np.sin(a)],
                  [0, np.sin(a), np.cos(a)]])
    neck = pose[int(Joint.NECK)]
    pose[int(Joint.HEAD)] = neck + R @ (pose[int(Joint.HEAD)] - neck)
    return pose


def _angle(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _aim_error(character, pose):
    """Angle between the captured NECK->HEAD direction and the posed rig's."""
    valid = ~np.isnan(pose).any(1)
    J = character.posed_joints(pose, valid, None)
    assert J is not None
    return _angle(pose[int(Joint.HEAD)] - pose[int(Joint.NECK)],
                  J[int(Joint.HEAD)] - J[int(Joint.NECK)])


# --- the policy plumbing (no rig asset needed) -----------------------------

@pytest.fixture(autouse=True)
def _no_leaked_head_default():
    """BOTH process-wide head defaults are module state, and every test here
    that touches either restores it in a `finally`. This is the belt as well:
    no test in this file may inherit a convention or a mode from another one
    (or from the order pytest happened to run them in), and none may leave one
    behind for the rest of the suite.

    It restores the values the module was IMPORTED with, so
    `test_the_default_head_source_is_the_legacy_convention` still measures the
    real default rather than one this fixture chose.
    """
    from pose3d.geometry import character as ch

    ch.set_default_head_source(_IMPORTED_DEFAULT)
    ch.set_default_head_mode(_IMPORTED_DEFAULT_MODE)
    try:
        yield
    finally:
        ch.set_default_head_source(_IMPORTED_DEFAULT)
        ch.set_default_head_mode(_IMPORTED_DEFAULT_MODE)


def test_the_default_head_source_is_the_legacy_convention():
    """Every project written before `head_source` existed was COCO-17, whose
    HEAD is the nose. Guessing anything else would silently re-pose them."""
    from pose3d.geometry import character as ch

    assert _IMPORTED_DEFAULT == "nose"      # what the module ships with
    assert ch.default_head_source() == "nose"
    try:
        ch.set_default_head_source("skull")
        assert ch.default_head_source() == "skull"
        ch.set_default_head_source(None)            # None -> the safe default
        assert ch.default_head_source() == "nose"
        with pytest.raises(ValueError):
            ch.set_default_head_source("halpe")     # a typo must not pass
    finally:
        ch.set_default_head_source("nose")


def test_a_tool_cannot_leak_a_head_convention_into_the_process():
    """`head_source_default` is the only way anything outside the UI is allowed
    to touch the process-wide default, and it always puts it back.

    A batch tool has no session to own that default, but it still has to reach
    the `Character` that `export.blender_export.export_animation` builds for
    itself. Holding the default around that one call is fine; leaving it set is
    not — the next thing in the process would pose a nose project as a skull
    one, silently, with no error anywhere.
    """
    from pose3d.geometry import character as ch

    assert ch.default_head_source() == "nose"      # the fixture above, not luck
    with ch.head_source_default("skull"):
        assert ch.default_head_source() == "skull"
        assert ch.Character().head_source == "skull"   # what a bare build gets
    assert ch.default_head_source() == "nose"

    # nesting composes, and the inner block restores the OUTER value, not the
    # module's initial one
    with ch.head_source_default("skull"):
        with ch.head_source_default("nose"):
            assert ch.default_head_source() == "nose"
        assert ch.default_head_source() == "skull"
    assert ch.default_head_source() == "nose"

    # and an exception inside the block is not an excuse to keep it
    with pytest.raises(RuntimeError):
        with ch.head_source_default("skull"):
            raise RuntimeError("the export died")
    assert ch.default_head_source() == "nose"

    with pytest.raises(ValueError):                    # a typo must not pass
        with ch.head_source_default("halpe"):
            pass
    assert ch.default_head_source() == "nose"


def test_the_head_mode_default_does_not_leak():
    """`head_mode` is the same kind of process-wide default as `head_source`,
    and it leaks the same way if nothing puts it back.

    It decides whether the neck's roll follows the NOSE alone or the whole
    chain takes the ear-driven face basis, so a stray "face" left behind by
    one tool silently re-poses the next take — on a mannequin, from ears that
    are noise. The trio must therefore behave exactly as `head_source`'s does:
    restore the OUTER value on nesting, restore it when the block raises,
    reject a typo, and reach a bare `Character()` (what the 3D view and the
    Blender export build, never having seen a `ProjectData`). Restored here by
    hand rather than by the autouse fixture above, so this test states the
    guarantee instead of relying on it.
    """
    from pose3d.geometry import character as ch

    assert ch.HEAD_MODES == ("nose", "face")
    previous = ch.default_head_mode()
    try:
        assert previous == "nose"                  # the shipped default
        with ch.head_mode_default("face"):
            assert ch.default_head_mode() == "face"
            assert ch.Character().head_mode == "face"    # a bare build
        assert ch.default_head_mode() == "nose"
        assert ch.Character().head_mode == "nose"
        # explicit still beats ambient
        with ch.head_mode_default("face"):
            assert ch.Character(head_mode="nose").head_mode == "nose"

        # nesting composes, and the inner block restores the OUTER value
        with ch.head_mode_default("face"):
            with ch.head_mode_default("nose"):
                assert ch.default_head_mode() == "nose"
            assert ch.default_head_mode() == "face"
        assert ch.default_head_mode() == "nose"

        # an exception inside the block is not an excuse to keep it
        with pytest.raises(RuntimeError):
            with ch.head_mode_default("face"):
                raise RuntimeError("the export died")
        assert ch.default_head_mode() == "nose"

        with pytest.raises(ValueError):            # a typo must not pass
            with ch.head_mode_default("ears"):
                pass
        assert ch.default_head_mode() == "nose"
        with pytest.raises(ValueError):
            ch.Character(head_mode="ears")

        ch.set_default_head_mode("face")
        assert ch.default_head_mode() == "face"
        ch.set_default_head_mode(None)             # None -> the safe default
        assert ch.default_head_mode() == "nose"
    finally:
        ch.set_default_head_mode(previous)


def test_the_detector_default_is_the_one_switch():
    """Which pose model the app runs is `RTMPoseDetector`'s own default, so
    the import wizard and the re-detect action cannot drift apart from it (or
    from the tools). Pinning `feet=` at a call site is what that would look
    like."""
    from pose3d.detect import rtmpose
    from pose3d.ui import import_dialog, main_window

    sig = inspect.signature(rtmpose.RTMPoseDetector.__init__)
    assert sig.parameters["feet"].default is rtmpose.USE_HALPE26

    for mod in (import_dialog, main_window):
        for line in inspect.getsource(mod).splitlines():
            if "RTMPoseDetector(" in line and "import" not in line:
                assert "feet" not in line, f"{mod.__name__}: {line.strip()}"


# --- the correction itself (needs the bundled rig) -------------------------

@needs_character()
def test_a_nose_head_keeps_the_anatomical_offset():
    """The legacy path, unchanged: a nose 25 deg off the torso line is a head
    held level, not a head nodded 25 deg, so the rig must NOT follow it."""
    from pose3d.geometry.character import Character

    pose = _pitch_head(sample_skeleton_3d(), 25.0)
    assert _aim_error(Character(head_source="nose"), pose) > 15.0


@needs_character()
def test_a_skull_head_is_aimed_at_directly():
    """A skull-vertex HEAD is already on the head's axis, so the neck aims
    straight at it — no 45 deg offset, no clamp.

    The residual is the rig's own geometry, not a correction: the head bone
    sits a few degrees off the neck bone's axis at rest, which is why even a
    0 deg nod reads ~3 deg. What matters is that it stays put as the nod grows,
    while the nose convention's error tracks the nod almost one for one. On the
    client take the same measurement is 1.19 deg median, 3.8 deg max.
    """
    from pose3d.geometry.character import Character

    skull = Character(head_source="skull")
    nose = Character(head_source="nose")
    for deg in (0.0, 10.0, 15.0, 25.0, 40.0, 60.0):
        pose = _pitch_head(sample_skeleton_3d(), deg)
        err = _aim_error(skull, pose)
        # today 3.02 / 4.96 / 5.76 / 6.74 / 6.31 / 2.39 deg
        assert err < 7.8, f"{deg} deg nod: aim error {err:.1f} deg"
        if deg >= 15.0:
            # today 12.83 / 22.86 / 37.10 / 37.45 deg for the nose convention
            assert _aim_error(nose, pose) > err + 5.0


@needs_character()
def test_the_clamp_does_not_silently_swallow_a_skull_nod():
    """The legacy correction saturates at the torso line, so on skull data it
    throws the measured pitch away entirely — which is why it is gated and not
    left as dead-but-lucky code. The two conventions must actually differ."""
    from pose3d.geometry.character import Character

    pose = _pitch_head(sample_skeleton_3d(), 25.0)
    nose = Character(head_source="nose").pose_bone_matrices(
        pose, ~np.isnan(pose).any(1))
    skull = Character(head_source="skull").pose_bone_matrices(
        pose, ~np.isnan(pose).any(1))
    assert not np.allclose(np.asarray(nose["neck"], float),
                           np.asarray(skull["neck"], float))


@needs_character()
def test_the_head_is_read_back_from_the_point_the_capture_names():
    """A nose HEAD has no counterpart on the rig, so it is read back from the
    middle of the head bone; a skull HEAD is the top of the skull, which is
    the head bone's tail."""
    from pose3d.geometry.character import Character

    from pose3d.geometry import character as chmod

    # a head convention is valid exactly when the rig knows where to read it
    # back from; the two lists cannot drift apart
    assert set(chmod.HEAD_SOURCES) == set(chmod._HEAD_FROM_RIG)

    nose, skull = Character(head_source="nose"), Character(head_source="skull")
    b_nose, which_nose = nose._joint_src[int(Joint.HEAD)]
    b_skull, which_skull = skull._joint_src[int(Joint.HEAD)]
    assert (which_nose, which_skull) == ("mid", "tail")
    assert b_nose == b_skull == nose.role["head"]

    # and the difference is real: the skull read-back sits further up the head
    # bone, by half its length
    axis = nose.tail[b_nose] - nose.head[b_nose]
    rest_nose = nose.rest_joints()[int(Joint.HEAD)]
    rest_skull = skull.rest_joints()[int(Joint.HEAD)]
    assert np.allclose(rest_skull - rest_nose, axis / 2.0)


@needs_character()
def test_a_legacy_project_is_posed_exactly_as_before():
    """No head_source means "nose", and a nose project must be bit-for-bit
    what the last build produced — this change may not move a single legacy
    frame."""
    from pose3d.geometry.character import Character

    poses = [sample_skeleton_3d(), _pitch_head(sample_skeleton_3d(), 30.0)]
    default, explicit = Character(), Character(head_source="nose")
    for pose in poses:
        valid = ~np.isnan(pose).any(1)
        a = default.pose_bone_matrices(pose, valid)
        b = explicit.pose_bone_matrices(pose, valid)
        assert a.keys() == b.keys()
        for name in a:
            assert np.array_equal(np.asarray(a[name], float),
                                  np.asarray(b[name], float)), name


# --- the app publishes what the open project was detected under ------------

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_window_publishes_the_open_projects_head_convention(qapp):
    """The 3D view and the Blender export each build their own Character and
    never see a ProjectData, so the window publishes the convention once, up
    front. Both then read the same value — which is what keeps the preview and
    the export pose-identical."""
    from pose3d.core.project import ProjectData
    from pose3d.geometry import character as ch
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    try:
        MainWindow(ProjectModel(ProjectData(name="skull"), None))
        assert ch.default_head_source() == "nose"      # a plain project

        data = ProjectData(name="skull", keypoint_model="halpe26",
                           head_source="skull")
        MainWindow(ProjectModel(data, None))
        assert ch.default_head_source() == "skull"
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_re_detecting_moves_the_head_convention_with_the_2d(qapp):
    """`Re-run detection` replaces every 2D point, so a project detected as
    COCO-17 becomes whatever the detector emits now. Leaving `head_source`
    behind would have the retarget correct a skull HEAD for the nose's forward
    offset — the double correction this whole key exists to prevent."""
    from pose3d.core.project import ProjectData
    from pose3d.geometry import character as ch
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    class _Det:
        head_source = "skull"
        feet = True

    try:
        data = ProjectData(name="legacy")
        win = MainWindow(ProjectModel(data, None))
        assert data.head_source == "nose"

        win._adopt_head_source(_Det())

        assert data.head_source == "skull"
        assert ch.default_head_source() == "skull"
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_every_detector_declares_what_its_head_point_is():
    """`head_source` belongs to the detector INTERFACE, not to RTMPose. A
    detector that simply forgot it used to fall through a
    `getattr(det, "head_source", "nose")` to the nose convention with no
    signal — i.e. straight into the double correction the key exists to
    prevent. The base class answers instead, and answers "nose", which is what
    every detector written before the key emitted."""
    from pose3d.detect.base import KeypointDetector
    from pose3d.geometry.character import HEAD_SOURCES

    assert KeypointDetector.head_source == "nose"
    assert KeypointDetector.head_source in HEAD_SOURCES

    class _Forgetful(KeypointDetector):
        def detect(self, image_bgr):        # pragma: no cover - never called
            raise NotImplementedError

    assert _Forgetful().head_source == "nose"


def _stub_rtmpose(monkeypatch):
    """Build RTMPoseDetector without loading (or downloading) any ONNX."""
    from pose3d.detect import models, rtmpose

    monkeypatch.setattr(models, "resolve",
                        lambda mode, feet: models.Weights(
                            det="det.onnx", det_input_size=(640, 640),
                            pose="pose.onnx", pose_input_size=(192, 256)))
    monkeypatch.setattr(models, "TwoStageDetector",
                        lambda w, backend="onnxruntime", device="cpu": object())
    return rtmpose.RTMPoseDetector


def test_a_built_detector_reports_the_head_point_its_model_emits(monkeypatch):
    """The one-switch invariant, asserted on a CONSTRUCTED detector rather
    than on the source text: what `USE_HALPE26` selects and what the detector
    says its HEAD is must be the same fact, or a project would record a
    convention its 2D does not hold."""
    from pose3d.core.skeleton import HALPE26_HEAD_SOURCE
    from pose3d.detect import rtmpose

    RTMPoseDetector = _stub_rtmpose(monkeypatch)

    default = RTMPoseDetector()
    assert default.feet is rtmpose.USE_HALPE26
    assert default.head_source == (HALPE26_HEAD_SOURCE
                                   if rtmpose.USE_HALPE26 else "nose")

    assert RTMPoseDetector(feet=True).head_source == "skull"
    assert RTMPoseDetector(feet=False).head_source == "nose"


def test_the_re_baseline_route_uses_the_shipped_detector():
    """`regen_client_take.py --redetect` is what re-baselines the regression
    fixture. If it pinned `feet=`, it would re-baseline the whole regression
    net onto a layout the app does not detect with — silently, since every
    threshold moves with it."""
    from tests.fixtures import regen_client_take

    src = inspect.getsource(regen_client_take)
    for line in src.splitlines():
        if "RTMPoseDetector(" in line and "import" not in line:
            assert "feet" not in line, line.strip()


class _FakeCharacter:
    def __init__(self, head_source):
        self.head_source = head_source


def test_re_detecting_drops_the_views_cached_character(qapp):
    """The 3D view builds its Character once and keeps it; the Blender export
    builds a fresh one per export. If a re-detection moved the convention and
    the cached one stayed, the preview and the export would pose the same
    frame under different conventions — the one thing the two may never do."""
    from pose3d.core.project import ProjectData
    from pose3d.geometry import character as ch
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    class _Skull:
        head_source = "skull"

    class _Nose:
        head_source = "nose"

    try:
        win = MainWindow(ProjectModel(ProjectData(name="legacy"), None))

        win.view3d._character = _FakeCharacter("nose")
        win._adopt_head_source(_Skull())
        assert win.view3d._character is None      # convention moved: rebuild

        kept = _FakeCharacter("nose")
        win.view3d._character = kept
        win._adopt_head_source(_Nose())
        assert win.view3d._character is kept      # unchanged: no churn
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_the_convention_moves_before_anything_is_re_posed(qapp, monkeypatch):
    """`redetect_all` re-poses and redraws as it goes, so the head convention
    has to be in place BEFORE it runs, not after: anything posed in between
    would be posed under the convention the 2D no longer holds."""
    from pose3d.core.project import Frame, ProjectData
    from pose3d.geometry import character as ch
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    class _Skull:
        head_source = "skull"

    seen = {}

    try:
        data = ProjectData(name="legacy")
        data.frames.append(Frame(frame_id="0000"))   # the window draws one
        win = MainWindow(ProjectModel(data, None))
        win.detector = _Skull()                   # no model loading
        win.view3d._character = _FakeCharacter("nose")

        def fake_redetect_all(det, load_image):
            seen["published"] = ch.default_head_source()
            seen["project"] = win.model.project.head_source
            seen["cached"] = win.view3d._character

        monkeypatch.setattr(win.model, "redetect_all", fake_redetect_all)
        win._on_run_detection()

        assert seen == {"published": "skull", "project": "skull",
                        "cached": None}
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


# --- the head MODE the user picks, per project -----------------------------
#
# `head_source` is a fact about the stored 2D and the app adopts it; the MODE
# is a choice the user makes about THIS figure — a mannequin's ears are noise
# and must not steer its head, a person's are real features. So it lives in
# the project file, is published process-wide the same way, and is switched
# from one combo in the 3D PREVIEW header.

def _mode_window(**kw):
    """A window on a one-frame project, plus the project it was built on."""
    from pose3d.core.project import Frame, ProjectData
    from pose3d.ui.main_window import MainWindow
    from pose3d.ui.model import ProjectModel

    data = ProjectData(name="modes", **kw)
    data.frames.append(Frame(frame_id="0000"))     # the window draws one
    model = ProjectModel(data, None)
    return MainWindow(model), data


def test_the_head_mode_combo_publishes_the_mode_and_rebuilds_the_character(qapp):
    """The 3D view and the Blender export each build their own Character and
    never see a ProjectData, so switching the mode has to reach them the way
    the convention does: through the process-wide default, with the view's
    cached Character dropped so it is rebuilt under the new one. A kept cache
    would leave the preview posing the head one way and the export the other."""
    from pose3d.geometry import character as ch

    try:
        win, data = _mode_window()
        assert data.head_mode == "nose"
        assert ch.default_head_mode() == "nose"
        assert win.head_combo.count() == 2
        assert [win.head_combo.itemText(i) for i in range(2)] == [
            "Head: nose", "Head: face (nose + ears)"]

        win.view3d._character = _FakeCharacter("nose")
        win.head_combo.setCurrentIndex(1)

        assert data.head_mode == "face"
        assert ch.default_head_mode() == "face"
        assert win.view3d._character is None      # mode moved: rebuild

        win.view3d._character = _FakeCharacter("nose")
        win.head_combo.setCurrentIndex(0)
        assert data.head_mode == "nose"
        assert ch.default_head_mode() == "nose"
        assert win.view3d._character is None
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_re_selecting_the_current_head_mode_changes_nothing(qapp):
    """Re-selecting the mode already in force must not churn: dropping the
    view's Character would rebuild and re-fit the whole rig for no reason, and
    marking the project unsaved would invent an edit the user never made."""
    from pose3d.geometry import character as ch

    try:
        win, data = _mode_window()
        kept = _FakeCharacter("nose")
        win.view3d._character = kept

        win.head_combo.setCurrentIndex(0)         # already "nose"

        assert win.view3d._character is kept      # unchanged: no churn
        assert win.saved_label.text() == "✓ Project Saved"
        assert data.head_mode == "nose"
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_switching_the_head_mode_is_an_unsaved_edit_that_is_saved(qapp, tmp_path):
    """The mode is stored in the project file, so switching it is an edit like
    any other: the header says so, and "Save Corrections" writes it. Reopening
    the folder and getting the old mode back would silently re-pose the head on
    every frame of the take."""
    from pose3d.core.io_project import load_project
    from pose3d.geometry import character as ch

    try:
        win, _ = _mode_window()
        win.model.project_dir = str(tmp_path)
        assert win.saved_label.text() == "✓ Project Saved"

        win.head_combo.setCurrentIndex(1)
        assert win.saved_label.text() == "● Unsaved changes"

        win.model.save()
        assert load_project(tmp_path).head_mode == "face"
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_opening_a_face_mode_project_publishes_it_without_an_edit(qapp):
    """Opening a project set to Face mode has to publish that mode before
    anything is drawn — and must not fire the combo's handler doing it, which
    would mark a freshly opened, unmodified project as having unsaved
    changes."""
    from pose3d.geometry import character as ch

    try:
        win, data = _mode_window(head_mode="face")

        assert ch.default_head_mode() == "face"
        assert win.head_combo.currentIndex() == 1
        assert win.saved_label.text() == "✓ Project Saved"
        assert data.head_mode == "face"
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


def test_the_camera_views_are_told_both_head_conventions(qapp):
    """Which face dots are draggable depends on both: the nose dot only under
    the skull convention (under "nose" the HEAD dot IS the nose), the eyes and
    ears only in Face mode. `_refresh_overlays` is the one place that draws
    them, so it is the one place that has to pass both."""
    from pose3d.geometry import character as ch

    try:
        win, _ = _mode_window(keypoint_model="halpe26",
                              head_source="skull", head_mode="face")
        seen = {}

        def fake_set_pose(*a, **kw):
            seen.update(kw)

        win.cam_left.view.set_pose = fake_set_pose
        win.cam_right.view.set_pose = fake_set_pose
        win._refresh_overlays()

        assert seen["head_source"] == "skull"
        assert seen["head_mode"] == "face"
    finally:
        ch.set_default_head_source("nose")
        ch.set_default_head_mode("nose")


# --- the gate table that decided the switch --------------------------------
#
# `docs/audit-2026-09/phase5_metrics.json` is the PRE-REGISTERED run: the table
# fixed before anything was measured, and what `tools/measure_head_gates.py`
# measured against it on the client take. Its `gates`, its 6-of-7 verdict and
# every number in it are evidence and stay exactly as measured. It also carries
# the restatement — `restated_gate`, and `switch.ships_on_note` — because a
# reader who opens only the evidence must not be left thinking the switch is
# off; that is a record ALONGSIDE the pre-registered table, never a re-scoring
# of it, and the first test below is what holds those two apart.
#
# `docs/audit-2026-09/phase5_gates.json` is the SHIPPING table: the same seven
# gates with the seventh restated on review, re-derived from the committed
# fixture rather than from the images, which is why the three tests below can
# check it in CI in about a second.
#
# Both files quote `tools.measure_head_gates.RESTATED_GATE` verbatim, so the
# restatement has ONE set of words and the tests check every copy against it.

_METRICS = "phase5_metrics.json"
_GATES = "phase5_gates.json"
#: The Phase 7 head-chain table (the rigid neck+head chain and its two modes),
#: re-derived from the same fixture by the last test in this file.
_HEAD_CHAIN = "phase7_head_chain.json"
#: The one gate that was restated after measurement, and the only bar in the
#: table that is not the one fixed in advance. The reasoning is
#: `tools.measure_head_gates.RESTATED_GATE`, quoted into both evidence files;
#: `pose3d.detect.rtmpose.USE_HALPE26` carries it too.
RESTATED = ("neck_lshoulder_bone_cv_pct", 6.5)


def _audit(name: str):
    from pathlib import Path as _P
    path = _P(__file__).resolve().parents[1] / "docs" / "audit-2026-09" / name
    if not path.exists():                    # a checkout without the evidence
        pytest.skip(f"{path} not present")
    import json
    return json.loads(path.read_text())


def measure_gate_table_on_fixture() -> dict:
    """Everything `measure_head_gates.gate_table` needs, from the FIXTURE.

    The committed client take is the Halpe-26 run of the Phase 5 measurement —
    same 26 frames, same calibration, same pipeline — so every gate except the
    COCO-17 comparison column can be re-derived here with no images and no
    detector. It reproduces the audit's numbers to the 3 decimal places of a
    pixel the fixture stores its 2D at.

    RECONSTRUCTED, like everything in `test_client_regression.py`: the fixture
    supplies the 2D and the calibration and the pipeline is run over it, so a
    regression in the triangulation or the bone fit moves these numbers too,
    not only one in the retarget.

    The measuring is `measure_head_gates.measure_project` — the same function
    the tool's own `measure` calls once it has loaded its npz cache, so the
    scoring AND the measurement are the shipped ones and neither can drift
    from what the evidence was written with. All this adds is the take.
    """
    from tools.measure_head_gates import measure_project

    p, rig = _fixture_reconstructed()
    return measure_project(
        "Halpe-26, skull HEAD, derived NECK/PELVIS (the fixture)", p, rig)


def _fixture_reconstructed():
    """The committed take, run through the shipped pipeline: (project, rig).

    Every evidence table in this file starts here, so they all measure the
    SAME reconstruction: the fixture's 2D and calibration, triangulated (face
    points and cross-view gate included) and bone-fitted with the shipped
    defaults — no smoothing, no images, no detector.
    """
    from pathlib import Path as _P

    from pose3d import pipeline
    from pose3d import quality as Q
    from pose3d.core.io_project import load_project

    fixture = _P(__file__).resolve().parent / "fixtures" / "client_take"
    p = load_project(fixture)
    rig = Q.load_rig(fixture / "calibration")
    pipeline.triangulate_project(p, rig)
    pipeline.fit_project(p)                  # shipped defaults: no smoothing
    return p, rig


def test_the_pre_registered_gate_table_is_still_what_it_was():
    """The table was fixed BEFORE the run so it could not be argued away
    afterwards, and `phase5_metrics.json` is the run. Nobody may loosen a
    threshold inside the evidence: the rules in it must still be the ones
    `tools/measure_head_gates.py` carries, its verdict must be its own
    arithmetic, and the gate it failed must still be the one it failed.
    """
    import tools.measure_head_gates as gates

    doc = _audit(_METRICS)
    assert {g["key"] for g in doc["gates"]} == set(gates.GATES)
    for g in doc["gates"]:
        assert g["rule"] == gates.GATES[g["key"]], g["key"]
    assert doc["passed"] == sum(bool(g["pass"]) for g in doc["gates"])
    assert doc["of"] == len(doc["gates"]) == 7
    assert doc["passed"] == 6
    failed = [g["key"] for g in doc["gates"] if not g["pass"]]
    assert failed == [RESTATED[0]]
    assert doc["switch"]["constant"] == "pose3d.detect.rtmpose.USE_HALPE26"
    # ...and `ships_on` is this table's own arithmetic, so it stays False even
    # though the app ships the switch ON. That is exactly the trap the file
    # has to defuse in its own words, next to the flag.
    assert doc["switch"]["ships_on"] is False
    assert "USE_HALPE26 = True" in doc["switch"]["ships_on_note"]
    assert "restated" in doc["switch"]["ships_on_note"]


def test_the_evidence_file_records_the_restatement_beside_the_table():
    """The restated gate is written into the pre-registered file too.

    The controller's ruling has to be findable from the evidence a reader
    actually opens, and `phase5_metrics.json` is that file — it is the run
    whose seventh gate failed. So it carries `restated_gate` (the rule, who
    decided it, the reasoning, the cost if wrong, what would reopen it) and a
    note on `switch.ships_on`, and BOTH sit beside the pre-registered table
    rather than inside it: the gate row for that key still reads
    "<= 5.15 % (the COCO-17 baseline)" and still says `pass: false`.

    That is the whole difference between recording a decision and rewriting
    the measurement it was taken against, and this test is where it is held.
    """
    import tools.measure_head_gates as gates

    doc = _audit(_METRICS)
    key, bar = RESTATED
    restated = doc["restated_gate"]

    # the words are the module's, not a paraphrase that can drift from it
    for field, value in gates.RESTATED_GATE.items():
        assert restated[field] == value, field
    assert restated["key"] == key
    assert restated["rule"] == f"<= {bar} %"
    assert restated["reason"].strip() and restated["cost_if_wrong"].strip()
    assert restated["what_would_reopen_it"].strip()

    # the pre-registered row it restates is untouched, and still a failure
    row = next(g for g in doc["gates"] if g["key"] == key)
    assert row["rule"] == restated["pre_registered_rule"] == gates.GATES[key]
    assert row["pass"] is False
    assert restated["measured"] == pytest.approx(row["measured"])
    assert restated["measured"] > 5.15 and restated["measured"] <= bar


def test_the_switch_ships_on_the_restated_gate_table():
    """What the app actually does, and why it is allowed to.

    Six gates passed outright. The seventh — the neck-Lshoulder bone CV —
    was restated from "<= 5.15 %" to "<= 6.5 %" on review, because 5.15 was
    the COCO-17 measurement itself rather than a tolerance anybody had
    derived: it made "no worse than today, at all, on this bone" the rule, and
    0.83 pp of it is ~0.13 mm on a 16 mm bone and about one standard error of
    a CV at n=26. The restatement is recorded, with its reasoning and its
    cost, in `phase5_gates.json` and — beside the table it restates, never
    inside it — in `phase5_metrics.json`; it is the ONLY bar in the table
    that is not the pre-registered one, and moving the switch means facing it.
    """
    from pose3d.detect import rtmpose
    from tools.measure_head_gates import RESTATED_GATE as RESTATED_GATE_WORDS

    doc = _audit(_GATES)
    pre = _audit(_METRICS)

    key, bar = RESTATED
    restated = doc["restated_gate"]
    assert restated["key"] == key
    assert restated["rule"] == f"<= {bar} %"
    assert restated["pre_registered_rule"] == \
        next(g["rule"] for g in pre["gates"] if g["key"] == key)
    assert restated["reason"].strip(), "a restated gate needs its reasoning"
    assert restated["cost_if_wrong"].strip()
    # the shipping table and the evidence quote the SAME restatement, word for
    # word, from `tools.measure_head_gates.RESTATED_GATE`
    for field, value in RESTATED_GATE_WORDS.items():
        assert restated[field] == value == pre["restated_gate"][field], field

    # exactly one bar was moved, and every other rule is still verbatim the
    # pre-registered one
    moved = [g["key"] for g in doc["gates"]
             if g["rule"] != next(x["rule"] for x in pre["gates"]
                                  if x["key"] == g["key"])]
    assert moved == [key]

    assert doc["passed"] == sum(bool(g["pass"]) for g in doc["gates"])
    assert doc["of"] == len(doc["gates"]) == 7
    assert doc["passed"] == doc["of"]
    assert doc["switch"]["constant"] == "pose3d.detect.rtmpose.USE_HALPE26"
    assert doc["switch"]["ships_on"] is True
    assert rtmpose.USE_HALPE26 is doc["switch"]["ships_on"], (
        "USE_HALPE26 and the shipping gate table disagree: face "
        "docs/audit-2026-09/phase5_gates.json before moving the switch")


@needs_character()
def test_the_gate_table_still_reads_the_same_on_the_committed_fixture():
    """The gate table is not a story about a run nobody can repeat.

    The committed fixture IS the Halpe-26 run, so every gate but one is
    re-measured here — through `measure_head_gates.gate_table`, the same
    scoring the evidence was written with — and must still say what
    `phase5_gates.json` says. A change anywhere in the retarget, the bone fit
    or the triangulation that quietly gives back the head win fails here, on a
    machine with no images and no detector.

    The COCO-17 column cannot be re-derived (the fixture is Halpe-26 now), so
    the one gate that IS a comparison — the worst body-joint regression — is
    scored against the baseline the pre-registered run recorded.
    """
    import tools.measure_head_gates as gates

    doc = _audit(_GATES)
    rows = gates.gate_table(_audit(_METRICS)["runs"]["coco"],
                            measure_gate_table_on_fixture())
    recorded = {g["key"]: g for g in doc["gates"]}
    assert {r["key"] for r in rows} == set(recorded)
    for r in rows:
        want = recorded[r["key"]]
        assert r["measured"] == pytest.approx(want["measured"], rel=2e-3), \
            f"{r['key']}: {r['measured']:.4f} vs recorded {want['measured']:.4f}"
        assert r["baseline"] == pytest.approx(want["baseline"], rel=2e-3)
        # and each still satisfies the bar it ships under
        bar = RESTATED[1] if r["key"] == RESTATED[0] else None
        assert (r["measured"] <= bar if bar is not None else r["pass"]), r["key"]


# --- the correction path: what a HEAD drag may touch ------------------------

def _fixture_model(head_source: str):
    """The committed take, reconstructed, held by a `ProjectModel`.

    `head_source` is the convention the project claims. The fixture IS a skull
    project; asking for "nose" relabels that same data, which is exactly the
    comparison wanted here — the two branches then differ ONLY in the
    convention, and the fixture's 86 px gap between the canonical HEAD (the
    skull vertex) and the nose face keypoint makes a wrong copy impossible to
    miss. A genuine COCO-17 project has that gap at 0 px by construction,
    which is why the nose branch is a no-op there and a corruption here.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from pathlib import Path as _P

    from pose3d import pipeline
    from pose3d.core.io_project import load_project
    from pose3d.quality import load_rig
    from pose3d.ui.model import ProjectModel

    fixture = _P(__file__).resolve().parent / "fixtures" / "client_take"
    p = load_project(fixture)
    assert p.head_source == "skull" and p.keypoint_model == "halpe26"
    p.head_source = head_source
    if head_source == "nose":
        p.keypoint_model = "coco17"
    rig = load_rig(fixture / "calibration")
    pipeline.triangulate_project(p, rig)
    pipeline.fit_project(p)
    model = ProjectModel(p, rig)
    model.set_frame(0)
    return model


def _drag_the_head(model, dx=5.0):
    """Drag the canonical HEAD `dx` px right in the LEFT view."""
    from pose3d.core.project import CAM_LEFT

    f = model.frame()
    xy = f.kp2d[CAM_LEFT][int(Joint.HEAD)]
    model.set_joint_2d(CAM_LEFT, int(Joint.HEAD),
                       float(xy[0]) + dx, float(xy[1]))
    return f


def test_dragging_a_skull_head_leaves_the_nose_where_it_was():
    """The canonical HEAD and the nose are two different detections here.

    `_resolve_joint` copies a dragged HEAD into `head2d[cam][0]` — the nose,
    `skeleton.HEAD_KP_NAMES[0]` — because under COCO-17 they ARE the same
    point. Under the layout the app now ships they are 86.0 px apart on frame
    0 of this fixture, so that copy teleported the nose onto the skull vertex:
    a 5 px drag moved head2d['left'][0] from [1800.689, 1763.494] to
    [1805.689, 1682.491], head3d[0] was re-triangulated from it, the head
    basis (built from nose + ears) turned with it, and `save_project` wrote
    all of it to disk. Silent wrong data on an ordinary user action.
    """
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT

    model = _fixture_model("skull")
    f = model.frame()
    gap = float(np.linalg.norm(f.kp2d[CAM_LEFT][int(Joint.HEAD)]
                               - f.head2d[CAM_LEFT][0]))
    assert gap == pytest.approx(86.0, abs=0.5), gap     # today 86.00 px

    before2d = {c: f.head2d[c].copy() for c in (CAM_LEFT, CAM_RIGHT)}
    before3d = f.head3d.copy()
    head2d_before = f.kp2d[CAM_LEFT][int(Joint.HEAD)].copy()

    _drag_the_head(model)

    # the drag did its job on the joint the user actually dragged...
    assert f.kp2d[CAM_LEFT][int(Joint.HEAD)][0] == \
        pytest.approx(head2d_before[0] + 5.0)
    # ...and touched no face keypoint, in either view, in 2D or in 3D
    for c in (CAM_LEFT, CAM_RIGHT):
        assert np.array_equal(f.head2d[c], before2d[c], equal_nan=True), c
    assert np.array_equal(f.head3d, before3d, equal_nan=True)


def test_dragging_a_nose_head_still_moves_the_nose_face_point():
    """...and the sync is not removed, only gated.

    Under the nose convention the canonical HEAD IS `head2d[cam][0]`, so a
    drag that did not move it would leave the head orientation built from a
    nose the user has just contradicted — the defect the sync was added for.
    One Ctrl+Z still reverses both, because the sync is re-derived in
    `_resolve_joint` rather than pushed as a second correction.
    """
    from pose3d.core.project import CAM_LEFT

    model = _fixture_model("nose")
    f = _drag_the_head(model)

    assert np.array_equal(f.head2d[CAM_LEFT][0],
                          f.kp2d[CAM_LEFT][int(Joint.HEAD)])

    before = f.head2d[CAM_LEFT][0].copy()
    model.undo()
    assert np.array_equal(f.head2d[CAM_LEFT][0],
                          f.kp2d[CAM_LEFT][int(Joint.HEAD)])
    assert not np.array_equal(f.head2d[CAM_LEFT][0], before)


def test_the_neck_follows_a_shoulder_drag_under_the_shipped_layout():
    """The other half of the same defect, on the client's own take.

    `_sync_derived` used to early-return for any project whose
    `keypoint_model` is not "coco17", on the belief that Halpe-26 detects
    NECK/PELVIS natively. It does — but `skeleton.HALPE26_POLICY` does not
    take those points (`map_halpe26` says why), so NECK is the shoulder
    midpoint under both layouts and the early return simply stopped
    maintaining it: a 40 px shoulder drag left NECK 28 px from the midpoint it
    is defined to be, and the bone fit was then solved against that. The
    synthetic cover is tests/test_pipeline_fit.py, parametrised over both
    layouts; this is the same thing on the real take.
    """
    from pose3d.core.project import CAM_LEFT

    model = _fixture_model("skull")
    f = model.frame()
    for parent, derived, other in (
            (Joint.LEFT_SHOULDER, Joint.NECK, Joint.RIGHT_SHOULDER),
            (Joint.LEFT_HIP, Joint.PELVIS, Joint.RIGHT_HIP)):
        xy = f.kp2d[CAM_LEFT][int(parent)]
        model.set_joint_2d(CAM_LEFT, int(parent),
                           float(xy[0]) - 40.0, float(xy[1]))
        want = 0.5 * (f.kp2d[CAM_LEFT][int(parent)]
                      + f.kp2d[CAM_LEFT][int(other)])
        assert np.allclose(f.kp2d[CAM_LEFT][int(derived)], want), (
            f"{derived.name} left "
            f"{np.linalg.norm(f.kp2d[CAM_LEFT][int(derived)] - want):.1f} px "
            f"from the midpoint it is defined to be")


# --- the head chain: the gates the rigid neck+head chain ships under --------

#: Two of the four head-chain rows are angles an `arccos` returns on matrices
#: the rigid chain makes IDENTICAL, so what the file records for them is
#: floating-point noise: `head_turn_error_deg` is 0.0 and
#: `head_neck_relative_rotation` is 1.6e-4 deg on top of a matrix difference
#: the same file records as exactly 0. Pinning those at `rel=2e-3` would pin
#: the noise — one ulp in the dot product moves an arccos near 1.0 by about
#: 1e-6 deg, and by O(1) RELATIVE — and the test would go red on another
#: BLAS/CPU/numpy build with nothing wrong. They are held to an ABSOLUTE
#: epsilon instead: three orders of magnitude under the 2 deg bar the turn
#: gate ships with, and three orders over the noise. Nothing is lost, because
#: the rigid-chain verdict comes from the exact matrix comparison at the
#: bottom of the test, never from this angle.
_NOISE_FLOOR_DEG = 1e-3
_ANGLE_ROWS_AT_THE_NOISE_FLOOR = ("head_neck_relative_rotation",
                                  "head_turn_error_deg")


@needs_character()
def test_the_head_chain_gates_still_read_the_same_on_the_committed_fixture():
    """The Phase 7 head-chain table, re-derived on the fixture in a second.

    `docs/audit-2026-09/phase7_head_chain.json` is the evidence for Decision 4
    (the dimensions invariant) and for the rigid chain that restored it: the
    skull may not shear, the head bone must carry the neck's own matrix, the
    nose must actually turn the chain in Nose mode, and the head must aim
    where the capture says even with the face points in play. Every one of
    those is measured HERE, on the committed take, through
    `measure_head_gates.measure_head_chain` and `head_chain_table` — the same
    functions the file was written with — so a change that gives the shear or
    the aim back fails on a machine with no images and no detector.

    Both modes are measured because the chain is rigid in both: Face mode
    follows the mannequin's bad ears (which is why Nose is the default) but it
    may not deform the skull either.

    The shear gate was NARROWED after the measurement — off the throat seam,
    which Decision 4 excludes — so the file carries that restatement the way
    `phase5_gates.json` carries its own, and this test holds the file, the
    module and the re-measurement to one set of words and one set of numbers.
    """
    import tools.measure_head_gates as gates
    from pose3d.geometry.character import HEAD_MODES

    p, rig = _fixture_reconstructed()
    m = {mode: gates.measure_head_chain(p, rig, mode) for mode in HEAD_MODES}
    rows = gates.head_chain_table(m)

    doc = _audit(_HEAD_CHAIN)
    recorded = {g["key"]: g for g in doc["gates"]}
    assert {r["key"] for r in rows} == set(recorded) == set(gates.HEAD_CHAIN_GATES)
    for r in rows:
        want = recorded[r["key"]]
        assert r["rule"] == want["rule"] == gates.HEAD_CHAIN_GATES[r["key"]]
        if r["key"] in _ANGLE_ROWS_AT_THE_NOISE_FLOOR:
            assert r["measured"] == pytest.approx(want["measured"],
                                                  abs=_NOISE_FLOOR_DEG), \
                (f"{r['key']}: {r['measured']:.6f} vs recorded "
                 f"{want['measured']:.6f} deg")
        else:
            assert r["measured"] == pytest.approx(want["measured"], rel=2e-3), \
                (f"{r['key']}: {r['measured']:.4f} vs recorded "
                 f"{want['measured']:.4f}")
        assert r["pass"] is True and want["pass"] is True, r["key"]
    assert doc["passed"] == doc["of"] == len(rows) == len(gates.HEAD_CHAIN_GATES)

    # the one gate that was narrowed after the measurement says so, in the
    # module's own words, and the file records what it reads under the rule as
    # it was first written rather than leaving the reader to find out
    restated = doc["restated_gate"]
    for field, value in gates.HEAD_CHAIN_RESTATED_GATE.items():
        assert restated[field] == value, field
    key = restated["key"]
    assert key in gates.HEAD_CHAIN_GATES
    assert restated["rule"] == gates.HEAD_CHAIN_GATES[key] \
        == recorded[key]["rule"], "the row does not ship under the rule it quotes"
    assert restated["pre_registered_rule"] != restated["rule"]
    assert restated["reason"].strip() and restated["cost_if_wrong"].strip()
    assert restated["what_would_reopen_it"].strip()
    assert restated["verdict_under_the_pre_registered_rule"] == "fail"

    # ...and both modes are in the file, with the shear each was measured at
    assert set(doc["modes"]) == set(HEAD_MODES)
    for mode, rec in doc["modes"].items():
        assert m[mode]["skull_shear"]["max_ratio"] == pytest.approx(
            rec["skull_shear"]["max_ratio"], rel=2e-3)
        assert m[mode]["head_vs_neck"]["max_abs_matrix_diff"] == 0.0, mode
        # the edges the narrowed gate drops are measured, not forgotten: the
        # throat seam is pinned here too — and against the restatement's own
        # column — so a regression that stretches it turns this test red even
        # though it is not a gate
        wider = m[mode]["skull_shear"]["with_the_throat_blend"]["max_ratio"]
        assert wider == pytest.approx(
            rec["skull_shear"]["with_the_throat_blend"]["max_ratio"], rel=2e-3)
        assert wider == pytest.approx(
            restated["measured_under_the_pre_registered_rule"][mode], rel=2e-3)
        assert wider > 1.05, (
            f"{mode}: the throat seam no longer stretches — the narrowed gate "
            "has nothing left to exclude, so restate it back to every skull "
            "edge (docs/audit-2026-09/phase7_head_chain.json:restated_gate)")

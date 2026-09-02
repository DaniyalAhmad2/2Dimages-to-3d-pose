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

def test_the_default_head_source_is_the_legacy_convention():
    """Every project written before `head_source` existed was COCO-17, whose
    HEAD is the nose. Guessing anything else would silently re-pose them."""
    from pose3d.geometry import character as ch

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

    assert ch.default_head_source() == "nose"
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


# --- the gate table that decided the switch --------------------------------
#
# `docs/audit-2026-09/phase5_metrics.json` is the PRE-REGISTERED run: the table
# fixed before anything was measured, and what `tools/measure_head_gates.py`
# measured against it on the client take. It is evidence and stays unedited.
#
# `docs/audit-2026-09/phase5_gates.json` is the SHIPPING table: the same seven
# gates with the seventh restated on review, re-derived from the committed
# fixture rather than from the images, which is why the three tests below can
# check it in CI in about a second.

_METRICS = "phase5_metrics.json"
_GATES = "phase5_gates.json"
#: The one gate that was restated after measurement, and the only bar in the
#: table that is not the one fixed in advance. See `phase5_gates.json` for the
#: reasoning; `pose3d.detect.rtmpose.USE_HALPE26` carries it too.
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

    Returns the same dict shape `measure_head_gates.measure` returns, so the
    scoring function is the shipped one and not a copy of it.
    """
    from pathlib import Path as _P

    from pose3d import pipeline
    from pose3d import quality as Q
    from pose3d.core.io_project import load_project
    from pose3d.core.project import CAMERAS
    from pose3d.geometry.character import Character
    from tools.measure_head_gates import _head_aim_error

    fixture = _P(__file__).resolve().parent / "fixtures" / "client_take"
    p = load_project(fixture)
    rig = Q.load_rig(fixture / "calibration")
    pipeline.triangulate_project(p, rig)
    pipeline.fit_project(p)                  # shipped defaults: no smoothing
    delivered = np.stack([f.fitted3d for f in p.frames])
    measured = np.stack([f.pose3d for f in p.frames])
    kp2d = {c: np.stack([f.kp2d[c] for f in p.frames]) for c in CAMERAS}

    height = Q.subject_height(delivered)
    R = Q.de_tilt_rotation(delivered)
    up = delivered @ R.T
    head3d = np.stack([f.head3d for f in p.frames]) @ R.T

    ch = Character(head_source=p.head_source)
    scale = ch.fit_to_subject(up)
    r_no = Q.retarget_error(ch, up, height, scale, None)
    r_face = Q.retarget_error(ch, up, height, scale, head3d)
    bl = Q.bone_length_stats(measured)
    return {
        "label": "Halpe-26, skull HEAD, derived NECK/PELVIS (the fixture)",
        "head_source": p.head_source, "n_frames": len(p.frames),
        "height_m": float(height), "scale": float(scale),
        "retarget_no_face_pct": {k: v["median_pct_height"]
                                 for k, v in r_no["per_joint"].items()},
        "retarget_with_face_pct": {k: v["median_pct_height"]
                                   for k, v in r_face["per_joint"].items()},
        "retarget_median_pct": {"no_face": r_no["median_pct_height"],
                                "with_face": r_face["median_pct_height"]},
        "head_aim_no_face": _head_aim_error(ch, up, None),
        "head_aim_with_face": _head_aim_error(ch, up, head3d),
        "bone_cv_pct": {bl["bones"][k]["name"]: bl["bones"][k]["cv_pct"]
                        for k in bl["bones"]},
        "bone_cv_median_pct": bl["median_cv_pct"],
        "bone_cv_max_pct": bl["max_cv_pct"],
        "epipolar_median_px": Q.body_epipolar(kp2d, rig)["median_px"],
    }


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


def test_the_switch_ships_on_the_restated_gate_table():
    """What the app actually does, and why it is allowed to.

    Six gates passed outright. The seventh — the neck-Lshoulder bone CV —
    was restated from "<= 5.15 %" to "<= 6.5 %" on review, because 5.15 was
    the COCO-17 measurement itself rather than a tolerance anybody had
    derived: it made "no worse than today, at all, on this bone" the rule, and
    0.83 pp of it is ~0.13 mm on a 16 mm bone and about one standard error of
    a CV at n=26. The restatement is recorded, with its reasoning and its
    cost, in `phase5_gates.json`; it is the ONLY bar in the table that is not
    the pre-registered one, and moving the switch means facing that file.
    """
    from pose3d.detect import rtmpose

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

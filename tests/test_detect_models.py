"""Shipped weights, so the client's first detection needs no network.

rtmlib downloads ~150 MB on first use and narrates it to `sys.stderr`, which is
None in a windowed build — that combination is a guaranteed crash on the very
first detection of the Windows bundle.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.detect import models
from tests.gates import needs_weights


def _stage(dirpath, mode="balanced", feet=False):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name in models.required_files(mode=mode, feet=feet):
        (dirpath / name).write_bytes(b"not a real onnx")
    return dirpath


def test_checkpoint_name_matches_rtmlib_naming():
    """rtmlib splits at the FIRST dot, not the last. A name derived any other
    way points at a file that is never there, so every run downloads."""
    got = models.checkpoint_name(
        "https://x/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip")
    assert got == "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"


def test_required_files_are_read_from_rtmlib_not_hardcoded():
    """If rtmlib changes a checkpoint, the bundle must follow it rather than
    ship a stale file the app then ignores."""
    from rtmlib.tools.solution.body import Body
    names = models.required_files("balanced", feet=False)
    assert len(names) == 2
    for url in (Body.MODE["balanced"]["det"], Body.MODE["balanced"]["pose"]):
        assert models.checkpoint_name(url) in names


def test_feet_needs_the_halpe26_pose_model():
    body = set(models.required_files("balanced", feet=False))
    feet = set(models.required_files("balanced", feet=True))
    assert body != feet
    assert any("halpe26" in n for n in feet)


def test_env_override_is_searched_first(tmp_path, monkeypatch):
    _stage(tmp_path / "staged")
    monkeypatch.setenv("POSE3D_MODELS", str(tmp_path / "staged"))
    w = models.resolve()
    assert w is not None
    assert Path(w.det).parent == tmp_path / "staged"
    assert Path(w.pose).parent == tmp_path / "staged"
    assert w.det_input_size == (640, 640)


def test_half_a_bundle_falls_back_rather_than_downloading_one_file(
        tmp_path, monkeypatch):
    d = _stage(tmp_path / "staged")
    next(d.glob("yolox*")).unlink()
    monkeypatch.setattr(models, "search_dirs", lambda: [d])
    assert models.resolve() is None


def test_no_weights_anywhere_leaves_rtmlib_in_charge(tmp_path, monkeypatch):
    """Linux checkouts have always downloaded; that must keep working."""
    monkeypatch.setattr(models, "search_dirs", lambda: [tmp_path / "empty"])
    assert models.resolve() is None


def test_detector_reports_which_path_it_took(tmp_path, monkeypatch):
    """Regression guard for the rtmlib 'rtmo' substring landmine: we must not
    reach Body/BodyWithFeet at all once weights are staged, because Body picks
    its architecture with `if 'rtmo' in pose` — a test against a filesystem
    path, so an install under C:\\Users\\rtmorgan\\ would load another model."""
    from pose3d.detect.rtmpose import RTMPoseDetector

    built = {}

    class FakeTwoStage:
        def __init__(self, w, backend="onnxruntime", device="cpu"):
            built["weights"] = w

        def __call__(self, image):
            return np.zeros((0, 17, 2)), np.zeros((0, 17))

    # staged for the layout the app actually detects with, or the pose model
    # resolves out of the developer's rtmlib cache and the assertion below is
    # about the wrong file
    from pose3d.detect.rtmpose import USE_HALPE26

    monkeypatch.setenv(
        "POSE3D_MODELS", str(_stage(tmp_path / "staged", feet=USE_HALPE26)))
    monkeypatch.setattr(models, "TwoStageDetector", FakeTwoStage)
    det = RTMPoseDetector(mode="balanced", device="cpu")
    assert det.bundled
    assert Path(built["weights"].pose).parent == tmp_path / "staged"


@needs_weights("lightweight")
def test_bundled_path_gives_the_same_numbers_as_rtmlib():
    """Composing YOLOX + RTMPose ourselves must be a pure refactor of what
    rtmlib's Body does — not a different model, nor a different input size,
    which would silently degrade every pose the client ever sees.

    Both stages are driven with a fixed bbox rather than a detection, so the
    comparison is over real keypoint numbers and cannot pass vacuously on an
    image where the detector happens to find nobody.
    """
    from rtmlib import Body

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = [[160.0, 60.0, 480.0, 440.0]]

    ref = Body(mode="lightweight", backend="onnxruntime", device="cpu")
    ours = models.TwoStageDetector(models.resolve(mode="lightweight"))

    assert ours.pose_model.model_input_size == ref.pose_model.model_input_size
    assert ours.det_model.model_input_size == ref.det_model.model_input_size

    kp_a, sc_a = ref.pose_model(img, bboxes=bbox)
    kp_b, sc_b = ours.pose_model(img, bboxes=bbox)
    assert np.asarray(kp_a).shape == (1, 17, 2)
    assert np.allclose(np.asarray(kp_a), np.asarray(kp_b))
    assert np.allclose(np.asarray(sc_a), np.asarray(sc_b))


def test_staging_carries_both_pose_models(monkeypatch, tmp_path):
    """The Halpe-26 checkpoint used to be behind a --feet flag no delivery
    passed, so every bundle shipped without it. Whether the app runs that model
    is one constant (detect.rtmpose.USE_HALPE26); whether the bundle CONTAINS
    it must not be a second decision, or flipping the constant ships a build
    that downloads on first use."""
    import sys

    import tools.fetch_weights as fw

    staged = []
    monkeypatch.setattr(
        fw, "stage",
        lambda out, mode, feet, allow_download: staged.append((mode, feet)) or [])
    monkeypatch.setattr(sys, "argv",
                        ["fetch_weights.py", "--out", str(tmp_path),
                         "--no-download"])
    fw.main()
    assert staged == [("balanced", False), ("balanced", True)]

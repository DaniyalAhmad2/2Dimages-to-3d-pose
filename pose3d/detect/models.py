"""Where the ONNX pose weights live.

rtmlib's default is to download ~150 MB from openmmlab the first time a
detector is built. That is wrong for a shipped app in three ways: it needs a
network the client may not have, it writes to a cache directory outside the
install, and it prints its progress to ``sys.stderr`` — which is ``None`` in a
windowed PyInstaller build, so the very first detection dies with an
``AttributeError`` before any pose is produced.

So we ship the weights and hand rtmlib explicit paths. Nothing here changes
behaviour when the files are absent: ``resolve()`` returns ``None`` and the
caller falls back to rtmlib's own downloading path, which is what a developer
checkout on Linux has always done.

The expected filenames are read from rtmlib's own MODE tables rather than
hard-coded, so a rtmlib upgrade that changes a checkpoint cannot leave us
silently pointing at the wrong model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pose3d.runtime import app_dir

#: Folder name used both inside the shipped bundle and by tools/fetch_weights.py
MODELS_DIRNAME = "models"


@dataclass(frozen=True)
class Weights:
    """Everything rtmlib's ``Body``/``BodyWithFeet`` need to skip downloading."""
    det: str
    det_input_size: tuple[int, int]
    pose: str
    pose_input_size: tuple[int, int]


def _mode_table(feet: bool) -> dict:
    if feet:
        from rtmlib.tools.solution.body_with_feet import BodyWithFeet
        return BodyWithFeet.MODE
    from rtmlib.tools.solution.body import Body
    return Body.MODE


def checkpoint_name(url: str) -> str:
    """Local filename rtmlib gives a checkpoint URL.

    Mirrors ``rtmlib.tools.file.download_checkpoint``, which strips at the
    FIRST dot — not the last — and appends ``.onnx``.
    """
    stem = os.path.basename(urlparse(url).path).split(".")[0]
    return stem + ".onnx"


def required_files(mode: str = "balanced", feet: bool = False) -> list[str]:
    """Filenames the bundle must contain for this configuration."""
    entry = _mode_table(feet)[mode]
    return [checkpoint_name(entry["det"]), checkpoint_name(entry["pose"])]


def search_dirs() -> list[Path]:
    """Where to look, most specific first.

    ``POSE3D_MODELS`` is the escape hatch; then the ``models/`` folder shipped
    beside the executable; then the source tree, so a checkout can stage
    weights without being frozen; then rtmlib's own cache, which is where a
    developer machine already has them.
    """
    out: list[Path] = []
    env = os.environ.get("POSE3D_MODELS")
    if env:
        out.append(Path(env))
    out.append(app_dir() / MODELS_DIRNAME)
    out.append(Path(__file__).resolve().parent.parent / "assets" / MODELS_DIRNAME)
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    out.append(Path(cache) / "rtmlib" / "hub" / "checkpoints")
    return out


def _find(name: str) -> Path | None:
    for d in search_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


def resolve(mode: str = "balanced", feet: bool = False) -> Weights | None:
    """Bundled weights for this configuration, or None to let rtmlib download.

    Both files must be present: a half-staged bundle would otherwise download
    only the missing one, which defeats the point and still crashes a windowed
    build.
    """
    try:
        entry = _mode_table(feet)[mode]
    except (ImportError, KeyError):
        return None
    det = _find(checkpoint_name(entry["det"]))
    pose = _find(checkpoint_name(entry["pose"]))
    if det is None or pose is None:
        return None
    return Weights(
        det=str(det), det_input_size=tuple(entry["det_input_size"]),
        pose=str(pose), pose_input_size=tuple(entry["pose_input_size"]))


class TwoStageDetector:
    """Detect people, then estimate keypoints — rtmlib's own pipeline.

    This is what ``Body``/``BodyWithFeet`` do internally, assembled here so we
    can hand them explicit file paths. Going through ``Body`` instead would be
    unsafe: it decides which architecture to load with ``if 'rtmo' in pose``,
    a substring test against what is now a filesystem path. An install under
    ``C:\\Users\\rtmorgan\\`` would quietly load a different model and download
    it — the exact behaviour bundling the weights exists to prevent.
    """

    def __init__(self, w: Weights, backend: str = "onnxruntime",
                 device: str = "cpu"):
        from rtmlib import RTMPose, YOLOX
        self.det_model = YOLOX(w.det, model_input_size=w.det_input_size,
                               backend=backend, device=device)
        self.pose_model = RTMPose(w.pose, model_input_size=w.pose_input_size,
                                  to_openpose=False, backend=backend,
                                  device=device)

    def __call__(self, image):
        bboxes = self.det_model(image)
        return self.pose_model(image, bboxes=bboxes)

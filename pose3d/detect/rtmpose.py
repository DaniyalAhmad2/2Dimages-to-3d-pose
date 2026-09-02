"""Markerless human pose detection via rtmlib (RTMPose, ONNX).

Pure onnxruntime; no mmpose/mmcv/torch. Two layouts are available:

* ``feet=False`` is the COCO-17 Body model. Today's shipped choice.
* ``feet=True`` runs the Halpe-26 model (BodyWithFeet). What that would buy
  is not the feet but its native HEAD — a point on the skull rather than the
  nose, worth 12.73 -> 1.54 % of body height at the head on the client take.
  NECK and PELVIS would stay 2D midpoints; see `skeleton.map_halpe26` for why
  that split is not a compromise.

Which one ran decides how the canonical HEAD may be used downstream, so the
detector reports it as `head_source` for the project to persist.

`USE_HALPE26` below is the app-wide switch, and it is OFF: the Halpe-26
change was measured against a fixed gate table on the client take and came
back 6 gates passed, 1 failed (see the constant).

Picks the highest-confidence person when several are detected.

CAVEAT (client-known): RTMPose is trained on real people and is unreliable on
the grey mannequin test rig and on extreme inverted poses. That is exactly why
the manual-correction path is first-class. Detector is swappable via base.py.
"""
from __future__ import annotations

import numpy as np

from pose3d.core.skeleton import (
    HALPE26_HEAD_SOURCE, NUM_HEAD_KP, NUM_JOINTS, derive_joints, extract_head,
    map_halpe26)
from pose3d.detect import models
from pose3d.detect.base import Detection, KeypointDetector

#: Whether the app detects with Halpe-26 rather than COCO-17. The whole switch
#: is this one line: it is `RTMPoseDetector`'s default, so the import wizard,
#: the re-detect action and every tool move together, and the project records
#: what it was detected under (`head_source`) either way.
#:
#: OFF, on the evidence. Measured on the client's 26-frame take (both models
#: through the identical pipeline; the harness reproduces the audit's COCO-17
#: and native-swap numbers exactly), against a gate table fixed before the run:
#:
#:   HEAD retarget, no face points   12.73 -> 1.54 % of height   (gate <= 4.0)  PASS
#:   HEAD retarget, with face points  9.03 -> 1.67 %             (gate <= 3.0)  PASS
#:   neck-head bone CV                9.53 -> 3.13 %             (gate <= 4.0)  PASS
#:   neck-Lshoulder bone CV           5.15 -> 5.98 %          (gate <= 5.15)  FAIL
#:   worst body-joint regression             +0.32 % of height  (gate <= 0.5)  PASS
#:   head aim error, nose path off            1.19 deg median   (gate <  5)    PASS
#:   body epipolar median             4.90 -> 4.68 px      (gate: no regress)  PASS
#:
#: The head win is large and real, and keeping NECK derived does contain most
#: of the neck regression the native swap causes (8.13 %), but not all of it:
#: Halpe's own shoulder points are slightly less consistent frame to frame than
#: COCO's on this take, so the shoulder half-width the derived NECK is built
#: from spreads more. That is the one gate, and it was fixed in advance
#: precisely so it could not be argued away afterwards.
#:
#: Flipping this to True turns the switch on; `tests/test_client_regression.py`
#: must then be re-baselined in the same commit (body height moves
#: 0.1195 -> 0.1302 m, +8.9 %, so every "% of height" threshold shifts), which
#: `tests/fixtures/regen_client_take.py --redetect` does.
USE_HALPE26 = False


class RTMPoseDetector(KeypointDetector):
    def __init__(self, mode: str = "balanced", device: str = "cpu",
                 backend: str = "onnxruntime", feet: bool = USE_HALPE26,
                 kpt_thr: float = 0.2):
        self.feet = feet
        self.kpt_thr = kpt_thr           # below this -> treated as not detected
        self._map = map_halpe26 if feet else derive_joints  # Halpe26 / COCO-17
        # What the canonical HEAD means in this detector's output. Persisted
        # per project (project.head_source) because the retarget corrects a
        # nose HEAD for its ~45 deg forward offset and must NOT correct a
        # skull one — see pose3d.geometry.character.
        self.head_source = HALPE26_HEAD_SOURCE if feet else "nose"

        # Prefer weights shipped with the app: rtmlib otherwise downloads
        # ~150 MB on first use, which needs a network and writes its progress
        # to a stderr that does not exist in a windowed build. See detect.models.
        w = models.resolve(mode=mode, feet=feet)
        self.bundled = w is not None
        if w is not None:
            self._model = models.TwoStageDetector(w, backend=backend, device=device)
        elif feet:
            from rtmlib import BodyWithFeet
            self._model = BodyWithFeet(mode=mode, backend=backend, device=device)
        else:
            from rtmlib import Body
            self._model = Body(mode=mode, backend=backend, device=device)
        self.mode = mode
        self.device = device

    def detect(self, image_bgr: np.ndarray) -> Detection:
        keypoints, scores = self._model(image_bgr)   # (N,K,2), (N,K)
        keypoints = np.asarray(keypoints, float)
        scores = np.asarray(scores, float)
        if keypoints.ndim != 3 or keypoints.shape[0] == 0:
            return _empty_detection()
        best = int(np.argmax(scores.mean(axis=1)))   # most confident person
        cxy, cscore = self._map(keypoints[best], scores[best])
        hxy, hscore = extract_head(keypoints[best], scores[best])

        # Reject unreliable keypoints so they are not drawn as phantom points
        # or fed into triangulation: (a) below the confidence threshold, or
        # (b) extrapolated OUTSIDE the image (e.g. feet below a cropped frame).
        h, w = image_bgr.shape[:2]

        def reject(xy, sc):
            bad = (
                (sc < self.kpt_thr)
                | (xy[:, 0] < 0) | (xy[:, 0] >= w)
                | (xy[:, 1] < 0) | (xy[:, 1] >= h)
            )
            xy[bad] = np.nan             # missing -> not drawn, not triangulated

        reject(cxy, cscore)
        # the face points get the identical treatment, or an out-of-frame ear
        # would reach triangulation and skew the head basis
        reject(hxy, hscore)
        return Detection(xy=cxy, scores=cscore, head_xy=hxy, head_scores=hscore)


def _empty_detection() -> Detection:
    return Detection(
        xy=np.full((NUM_JOINTS, 2), np.nan),
        scores=np.zeros(NUM_JOINTS),
        head_xy=np.full((NUM_HEAD_KP, 2), np.nan),
        head_scores=np.zeros(NUM_HEAD_KP))

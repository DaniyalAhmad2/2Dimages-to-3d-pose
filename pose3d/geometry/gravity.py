"""Which way is up, recorded at calibration time from the rig itself.

`orient.sequence_up` levels the view on the SUBJECT's own body line, which is
robust but throws away any lean held for the whole take (10.8 deg median /
21.1 deg maximum of per-frame lean survives; only the mean is removed). It has
to, because the calibration world frame is one hand-taped ArUco tag whose
rotation is arbitrary.

The rig itself carries better evidence than that one tag. Three independent
proxies for gravity are available once the extrinsics are solved:

  (a) `cross(x_left, x_right)` — the two cameras' image-x axes are both
      horizontal if neither camera is rolled, so their cross product is
      vertical. Independent of the tags entirely.
  (b) `-mean(camera up)` — each camera's own up-axis; the same assumption
      stated per camera rather than per pair, and it is the one candidate with
      an unambiguous SIGN.
  (c) `cross(tag row, mean tag normal)` — the wall tags sit in a horizontal
      row (0.298 m long, straightness singular-value ratio 0.093 on the
      client's take), so the row direction crossed with the wall normal is
      vertical. This is the only candidate that does not assume camera roll is
      zero. It must only be built from tags the calibration ADMITTED: a tag
      whose IPPE branches are ambiguous (tag 13, bent over a curved sweep,
      returns the wrong normal in 96 % of solves) poisons it.

None of them is gravity. Their pairwise spread is the honest uncertainty
(8-14 deg on this rig) and is reported alongside the answer, so a caller can
refuse to use it — `main_window` and the Blender export ignore a vertical
whose spread is above ~20 deg and fall back to `orient.sequence_up`. When only
ONE candidate survives there is no spread to report and the answer is
UNVERIFIED (spread None), not certain: a lone proxy has nothing checking it.

The sign is settled separately, by `sign_from_poses` (is the mean NECK above
the mean ANKLE), and only the camera-up proxy can stand in for that before the
poses exist — see `calib.resolve.finalize_world_up`, which re-runs the test
once triangulation has produced poses.
"""
from __future__ import annotations

import numpy as np

from pose3d.core.skeleton import Joint, NUM_JOINTS

# Candidate names, in the order they are reported.
CAMERA_PAIR = "camera pair"
CAMERA_UP = "camera up"
TAG_ROW = "tag row"


def _unit(v):
    v = np.asarray(v, float).ravel()
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def _angle_deg(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def _camera_pair_up(rots) -> np.ndarray | None:
    """cross() of the two cameras' image-x axes expressed in world coords."""
    if len(rots) != 2:
        return None
    xs = [R.T @ np.array([1.0, 0.0, 0.0]) for R in rots]
    return _unit(np.cross(xs[0], xs[1]))


def _camera_up(rots) -> np.ndarray | None:
    """Mean of the cameras' own up-axes. Image +y points DOWN, hence the sign.

    This is the only candidate whose sign is meaningful, so the others are
    aligned to it.
    """
    if not rots:
        return None
    ups = [-(R.T @ np.array([0.0, 1.0, 0.0])) for R in rots]
    return _unit(np.mean(ups, axis=0))


def _tag_row_up(tag_layout) -> np.ndarray | None:
    """cross(row direction, mean tag normal) from >=2 admitted tags."""
    if not tag_layout or len(tag_layout) < 2:
        return None
    centres, normals = [], []
    for R, t in tag_layout.values():
        centres.append(np.asarray(t, float).ravel())
        normals.append(np.asarray(R, float) @ np.array([0.0, 0.0, 1.0]))
    centres = np.asarray(centres, float)
    normal = _unit(np.mean(normals, axis=0))
    if normal is None:
        return None
    centred = centres - centres.mean(axis=0)
    # principal direction of the tag centres = the row
    row = _unit(np.linalg.svd(centred, full_matrices=False)[2][0])
    if row is None:
        return None
    return _unit(np.cross(row, normal))


def _mean_point(points):
    """Mean of the rows that are fully finite, or None if there are none.

    A masked mean rather than `np.nanmean`: on an all-NaN take nanmean warns
    ("Mean of empty slice") on exactly the degenerate input this function
    exists to handle, and a per-axis nanmean would average the x of one frame
    with the z of another.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    ok = np.isfinite(points).all(axis=1)
    if not ok.any():
        return None
    return points[ok].mean(axis=0)


def sign_from_poses(up: np.ndarray, poses):
    """+1/-1 so that the take's mean NECK sits above its mean ANKLE.

    A binary test on two averaged points: it decides the SENSE of an axis that
    was already estimated, and cannot rotate it, so no amount of body lean
    leaks into the vertical.

    Returns None when the take cannot decide — no frames, or no frame with
    both a NECK and an ankle — so a caller can tell "the poses say keep it"
    from "the poses say nothing" and not record a decision it did not make.
    """
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    if poses.size == 0:
        return None
    top = _mean_point(poses[:, int(Joint.NECK)])
    bottom = _mean_point(
        poses[:, [int(Joint.LEFT_ANKLE), int(Joint.RIGHT_ANKLE)]])
    if top is None or bottom is None:
        return None
    d = float(np.dot(np.asarray(up, float).ravel(), top - bottom))
    if d == 0.0:                        # exactly perpendicular: no evidence
        return None
    return -1.0 if d < 0 else 1.0


def estimate_world_up(rig, tag_layout=None, poses=None):
    """Gravity in world (calibration) coordinates, with its uncertainty.

    Parameters
    ----------
    rig : CalibratedRig — only the extrinsics are used.
    tag_layout : {tag_id: (R_tag_to_world, t_tag_in_world)} for the tags the
        calibration ADMITTED, or None. Fewer than two tags disables (c).
    poses : (T, NUM_JOINTS, 3) reconstructed poses, or None. Used ONLY to
        settle the sign; when absent the camera-up candidate settles it.

    Returns
    -------
    (up, spread_deg, candidates) — `up` is a unit vector or None if nothing
    could be estimated; `spread_deg` is the largest pairwise angle between the
    candidates, or None when fewer than two candidates survived; `candidates`
    maps name -> unit vector, sign-aligned with the returned `up`.

    A SINGLE candidate reports spread None, not 0.0. Zero would mean "three
    independent proxies agree perfectly", which is the strongest evidence this
    module can produce, whereas one proxy is the weakest: nothing checked it.
    On a parallel (non-converging) two-phone rig `_camera_pair_up` returns None
    because the image-x axes are parallel, so this is the canonical case, and
    the caller (`orient.take_up`) must fall back to the subject's body line
    rather than be sold an unvalidated proxy as certainty.
    """
    rots = [np.asarray(e.R, float) for e in rig.ext.values()] if rig else []
    raw = {
        CAMERA_PAIR: _camera_pair_up(rots),
        CAMERA_UP: _camera_up(rots),
        TAG_ROW: _tag_row_up(tag_layout),
    }
    cands = {k: v for k, v in raw.items() if v is not None}
    if not cands:
        return None, None, {}

    # (a) and (c) are cross products: their sign is arbitrary. Align every
    # candidate with the one candidate that has a physical sign.
    ref = cands[CAMERA_UP] if CAMERA_UP in cands else next(iter(cands.values()))
    cands = {k: (v if float(np.dot(v, ref)) >= 0 else -v) for k, v in cands.items()}

    up = _unit(np.mean(list(cands.values()), axis=0))
    if up is None:                      # candidates cancelled: no answer
        return None, None, cands
    names = list(cands)
    pairwise = [_angle_deg(cands[a], cands[b])
                for i, a in enumerate(names) for b in names[i + 1:]]
    spread = float(max(pairwise)) if pairwise else None

    if poses is not None:
        s = sign_from_poses(up, poses)
        if s is not None and s < 0:
            up = -up
            cands = {k: -v for k, v in cands.items()}
    return up, spread, cands

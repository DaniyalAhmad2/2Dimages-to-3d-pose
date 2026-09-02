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
whose spread is above ~20 deg and fall back to `orient.sequence_up`.
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


def _sign_from_poses(up: np.ndarray, poses) -> float:
    """+1/-1 so that the take's mean NECK sits above its mean ANKLE.

    A binary test on two averaged points: it decides the SENSE of an axis that
    was already estimated, and cannot rotate it, so no amount of body lean
    leaks into the vertical.
    """
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    if poses.size == 0:
        return 1.0
    top = np.nanmean(poses[:, int(Joint.NECK)], axis=0)
    ankles = poses[:, [int(Joint.LEFT_ANKLE), int(Joint.RIGHT_ANKLE)]]
    with np.errstate(invalid="ignore"):
        bottom = np.nanmean(ankles.reshape(-1, 3), axis=0)
    if np.isnan(top).any() or np.isnan(bottom).any():
        return 1.0
    d = float(np.dot(up, top - bottom))
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
    candidates (0.0 when there is only one); `candidates` maps name -> unit
    vector, sign-aligned with the returned `up`.
    """
    rots = [np.asarray(e.R, float) for e in rig.ext.values()] if rig else []
    raw = {
        CAMERA_PAIR: _camera_pair_up(rots),
        CAMERA_UP: _camera_up(rots),
        TAG_ROW: _tag_row_up(tag_layout),
    }
    cands = {k: v for k, v in raw.items() if v is not None}
    if not cands:
        return None, 0.0, {}

    # (a) and (c) are cross products: their sign is arbitrary. Align every
    # candidate with the one candidate that has a physical sign.
    ref = cands[CAMERA_UP] if CAMERA_UP in cands else next(iter(cands.values()))
    cands = {k: (v if float(np.dot(v, ref)) >= 0 else -v) for k, v in cands.items()}

    up = _unit(np.mean(list(cands.values()), axis=0))
    if up is None:                      # candidates cancelled: no answer
        return None, 0.0, cands
    names = list(cands)
    spread = max((_angle_deg(cands[a], cands[b])
                  for i, a in enumerate(names) for b in names[i + 1:]),
                 default=0.0)

    if poses is not None:
        s = _sign_from_poses(up, poses)
        if s < 0:
            up = -up
            cands = {k: -v for k, v in cands.items()}
    return up, float(spread), cands

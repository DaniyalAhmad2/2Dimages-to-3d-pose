"""Phase 4 verification: two-view triangulation accuracy.

Every test runs on BOTH rigs: the wide symmetric 720p rig the suite grew up on,
and the client's close-range asymmetric one (2:1 in sensor size and focal
length). Anything that quietly assumes a single K or a single pixel scale
passes the first and fails the second.
"""
import numpy as np
import pytest

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.geometry.triangulate import (
    reprojection_error, triangulate_one, triangulate_points,
)
from tests.synth import cameras, close_range_two_cam, default_two_cam, project


@pytest.fixture(params=["wide symmetric", "close asymmetric"])
def rig(request):
    """(intr_left, intr_right, ext_left, ext_right, gt, px_left, px_right)."""
    geo = (default_two_cam() if request.param == "wide symmetric"
           else close_range_two_cam())
    cams = cameras(geo)
    intr = {c: Intrinsics(K=cams[c][0], dist=cams[c][1], image_size=cams[c][2])
            for c in ("left", "right")}
    ext = {c: Extrinsics(R=geo[c][0], t=geo[c][1]) for c in ("left", "right")}
    gt = geo["subject"]
    px = {c: project(gt, cams[c][0], cams[c][1], *geo[c])
          for c in ("left", "right")}
    return (intr["left"], intr["right"], ext["left"], ext["right"],
            gt, px["left"], px["right"])


def _height(gt) -> float:
    return float(gt[:, 2].max() - gt[:, 2].min())


def test_triangulation_perfect_data(rig):
    intr_l, intr_r, ext_l, ext_r, gt, pl, pr = rig
    xyz = triangulate_points(pl, pr, intr_l, intr_r, ext_l, ext_r)
    err = np.linalg.norm(xyz - gt, axis=1)
    # 6e-7, not 1e-6: the bound went relative when the close-range rig was
    # added, and 1e-6 x the 1.62 m symmetric figure would have LOOSENED the
    # exactness assertion that rig already had by 1.6x. Both rigs land at
    # ~1e-15 m — this is float noise, not geometry.
    assert np.nanmax(err) < 6e-7 * _height(gt), f"max err {np.nanmax(err)}"


def test_triangulation_with_pixel_noise(rig):
    intr_l, intr_r, ext_l, ext_r, gt, pl, pr = rig
    rng = np.random.default_rng(0)
    pl_n = pl + rng.normal(0, 0.5, pl.shape)   # 0.5 px noise
    pr_n = pr + rng.normal(0, 0.5, pr.shape)
    xyz = triangulate_points(pl_n, pr_n, intr_l, intr_r, ext_l, ext_r)
    err = np.linalg.norm(xyz - gt, axis=1)
    # under 0.6 % of the subject's height with half-pixel noise at either
    # geometry (0.01 m on the 1.7 m figure, as this test has always asserted)
    assert np.nanmean(err) < 0.006 * _height(gt), f"mean err {np.nanmean(err)} m"


def test_nan_propagation(rig):
    intr_l, intr_r, ext_l, ext_r, gt, pl, pr = rig
    pl[3] = np.nan   # joint 3 unseen in left view
    xyz = triangulate_points(pl, pr, intr_l, intr_r, ext_l, ext_r)
    assert np.isnan(xyz[3]).all()
    assert not np.isnan(xyz[0]).any()


def test_reprojection_error_small(rig):
    intr_l, intr_r, ext_l, ext_r, gt, pl, pr = rig
    xyz = triangulate_points(pl, pr, intr_l, intr_r, ext_l, ext_r)
    for intr, ext, obs in ((intr_l, ext_l, pl), (intr_r, ext_r, pr)):
        err = reprojection_error(xyz, obs, intr, ext)
        assert np.nanmax(err) < 1e-3


def test_triangulate_one_matches_batch(rig):
    intr_l, intr_r, ext_l, ext_r, gt, pl, pr = rig
    one = triangulate_one(pl[5], pr[5], intr_l, intr_r, ext_l, ext_r)
    assert np.linalg.norm(one - gt[5]) < 6e-7 * _height(gt)   # as above


def test_epipolar_distance_consistent_vs_hallucinated(rig):
    from pose3d.geometry.triangulate import epipolar_distance
    intr_l, intr_r, ext_l, ext_r, gt, pl, pr = rig
    # a correct correspondence has ~zero epipolar distance
    good = epipolar_distance(pl[13], pr[13], intr_l, intr_r, ext_l, ext_r)
    assert good < 2.0, good
    # move the LEFT ankle onto the LEFT knee (occlusion hallucination):
    # it can no longer correspond to the right view's ankle -> large distance
    bad = epipolar_distance(pl[11], pr[13], intr_l, intr_r, ext_l, ext_r)
    assert bad > 20.0, bad


def test_validate_cross_view_drops_bad_observation(rig):
    from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
    from pose3d.pipeline import CalibratedRig, triangulate_project, validate_cross_view
    intr_l, intr_r, ext_l, ext_r, gt, pl, pr = rig
    calibrated = CalibratedRig(intr_l, intr_r, ext_l, ext_r)
    f = Frame(frame_id="0")
    f.kp2d[CAM_LEFT] = pl.copy(); f.kp2d[CAM_RIGHT] = pr.copy()
    f.scores[CAM_LEFT] = np.ones(len(pl)); f.scores[CAM_RIGHT] = np.ones(len(pr))
    # hallucinate the LEFT ankle onto the knee, with lower confidence
    f.kp2d[CAM_LEFT][13] = pl[11]
    f.scores[CAM_LEFT][13] = 0.4
    proj = ProjectData(frames=[f])
    dropped = validate_cross_view(proj, calibrated)
    assert dropped >= 1
    assert np.isnan(f.kp2d[CAM_LEFT][13]).all()      # bad view dropped
    assert not np.isnan(f.kp2d[CAM_RIGHT][13]).any()  # good view kept
    triangulate_project(proj, calibrated)
    assert np.isnan(f.pose3d[13]).all()               # 3D point dropped too

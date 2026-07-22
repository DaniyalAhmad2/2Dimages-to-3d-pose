"""Phase 4 verification: two-view triangulation accuracy."""
import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.geometry.triangulate import (
    reprojection_error, triangulate_one, triangulate_points,
)
from tests.synth import default_two_cam, project, sample_skeleton_3d


def _setup():
    rig = default_two_cam()
    intr = Intrinsics(K=rig["K"], dist=rig["dist"], image_size=rig["size"])
    Rl, tl = rig["left"]; Rr, tr = rig["right"]
    ext_l = Extrinsics(R=Rl, t=tl); ext_r = Extrinsics(R=Rr, t=tr)
    gt = sample_skeleton_3d()
    pl = project(gt, rig["K"], rig["dist"], Rl, tl)
    pr = project(gt, rig["K"], rig["dist"], Rr, tr)
    return intr, ext_l, ext_r, gt, pl, pr


def test_triangulation_perfect_data():
    intr, ext_l, ext_r, gt, pl, pr = _setup()
    xyz = triangulate_points(pl, pr, intr, intr, ext_l, ext_r)
    err = np.linalg.norm(xyz - gt, axis=1)
    assert np.nanmax(err) < 1e-6, f"max err {np.nanmax(err)}"


def test_triangulation_with_pixel_noise():
    intr, ext_l, ext_r, gt, pl, pr = _setup()
    rng = np.random.default_rng(0)
    pl_n = pl + rng.normal(0, 0.5, pl.shape)   # 0.5 px noise
    pr_n = pr + rng.normal(0, 0.5, pr.shape)
    xyz = triangulate_points(pl_n, pr_n, intr, intr, ext_l, ext_r)
    err = np.linalg.norm(xyz - gt, axis=1)
    # sub-cm with half-pixel noise at this geometry
    assert np.nanmean(err) < 0.01, f"mean err {np.nanmean(err)} m"


def test_nan_propagation():
    intr, ext_l, ext_r, gt, pl, pr = _setup()
    pl[3] = np.nan   # joint 3 unseen in left view
    xyz = triangulate_points(pl, pr, intr, intr, ext_l, ext_r)
    assert np.isnan(xyz[3]).all()
    assert not np.isnan(xyz[0]).any()


def test_reprojection_error_small():
    intr, ext_l, ext_r, gt, pl, pr = _setup()
    xyz = triangulate_points(pl, pr, intr, intr, ext_l, ext_r)
    err = reprojection_error(xyz, pl, intr, ext_l)
    assert np.nanmax(err) < 1e-3


def test_triangulate_one_matches_batch():
    intr, ext_l, ext_r, gt, pl, pr = _setup()
    one = triangulate_one(pl[5], pr[5], intr, intr, ext_l, ext_r)
    assert np.linalg.norm(one - gt[5]) < 1e-6

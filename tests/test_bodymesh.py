"""The human body surface builds a plausible closed mesh around the skeleton."""
import numpy as np
import pytest

pytest.importorskip("mcubes")

from pose3d.geometry.bodymesh import human_body_mesh
from pose3d.core.skeleton import Joint
from tests.synth import sample_skeleton_3d


def test_body_mesh_wraps_the_pose():
    pose = sample_skeleton_3d()
    verts, faces = human_body_mesh(pose, resolution=40)
    assert verts is not None and len(verts) > 200
    assert faces is not None and len(faces) > 200
    # the surface should extend a bit beyond the joints (body thickness) and
    # cover the full height
    vmin, vmax = verts.min(0), verts.max(0)
    pmin, pmax = pose.min(0), pose.max(0)
    assert (vmin <= pmin + 1e-6).all() and (vmax >= pmax - 1e-6).all()


def test_body_mesh_handles_sparse_pose():
    pose = sample_skeleton_3d()
    pose[[Joint.LEFT_WRIST, Joint.RIGHT_WRIST,
          Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE]] = np.nan
    verts, faces = human_body_mesh(pose)
    assert verts is not None and len(verts) > 100


def test_body_mesh_none_when_too_few_joints():
    pose = np.full((15, 3), np.nan)
    pose[0] = (0, 0, 1.7)
    verts, faces = human_body_mesh(pose)
    assert verts is None and faces is None

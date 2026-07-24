"""The bundled character skins to a pose (live LBS) and stays aligned to it."""
import numpy as np
import pytest

from pose3d.config import character_blend

pytest.importorskip("numpy")

_HAVE = character_blend() is not None
from pose3d.core.skeleton import Joint
from tests.synth import sample_skeleton_3d


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_character_skins_to_pose():
    from pose3d.geometry.character import Character
    pose = sample_skeleton_3d()
    valid = ~np.isnan(pose).any(1)
    verts, faces = Character().pose(pose, valid)
    assert verts is not None and len(verts) > 500
    assert not np.isnan(verts).any()
    # character occupies roughly the pose's vertical span, at its position
    assert abs(verts[:, 2].min() - pose[:, 2].min()) < 0.3 * (pose[:, 2].max() - pose[:, 2].min())
    assert abs(verts[:, 2].max() - pose[:, 2].max()) < 0.4 * (pose[:, 2].max() - pose[:, 2].min())


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_character_raises_correct_side():
    """Raising the LEFT wrist must lift the character's LEFT-side vertices."""
    from pose3d.geometry.character import Character
    ch = Character()
    base = sample_skeleton_3d()
    valid = ~np.isnan(base).any(1)
    v0, _ = ch.pose(base, valid)
    raised = base.copy()
    raised[int(Joint.LEFT_ELBOW)] = [-0.30, 0, 1.45]
    raised[int(Joint.LEFT_WRIST)] = [-0.30, 0, 1.75]
    v1, _ = ch.pose(raised, valid)
    # left-side verts (x<0, matching LEFT_SHOULDER at -x) should rise on average
    left = v0[:, 0] < -0.05
    assert (v1[left, 2].mean() - v0[left, 2].mean()) > 0.05


@pytest.mark.skipif(not _HAVE, reason="bundled character asset missing")
def test_character_handles_sparse_pose():
    from pose3d.geometry.character import Character
    pose = sample_skeleton_3d()
    pose[[Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE, Joint.LEFT_WRIST]] = np.nan
    verts, faces = Character().pose(pose, ~np.isnan(pose).any(1))
    assert verts is not None and not np.isnan(verts).any()

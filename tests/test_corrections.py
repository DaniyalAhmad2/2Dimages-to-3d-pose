"""Phase 7 verification: reversible correction stack + live re-solve.

Plus the storage half of "correction data is stored so the tool can improve
with future use": a correction the client made in an earlier session is their
work, and no later save may destroy it.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.core.corrections import CorrectionStack
from pose3d.core.io_project import (
    append_correction, count_corrections, load_project, save_project,
)
from pose3d.core.project import (
    CAM_LEFT, CAM_RIGHT, Correction, Frame, ProjectData,
)
from pose3d.geometry.triangulate import triangulate_one
from tests.synth import default_two_cam, project, sample_skeleton_3d

pytest.importorskip("PySide6")


def _frame_with_pose():
    rig = default_two_cam()
    gt = sample_skeleton_3d()
    pl = project(gt, rig["K"], rig["dist"], *rig["left"])
    pr = project(gt, rig["K"], rig["dist"], *rig["right"])
    f = Frame(frame_id="0001")
    f.kp2d[CAM_LEFT] = pl
    f.kp2d[CAM_RIGHT] = pr
    f.scores[CAM_LEFT] = np.ones(len(pl))
    f.scores[CAM_RIGHT] = np.ones(len(pr))
    return f, rig, gt


def test_apply_undo_redo():
    f, rig, gt = _frame_with_pose()
    stack = CorrectionStack({f.frame_id: f})

    old = tuple(f.kp2d[CAM_LEFT][6])
    stack.apply("0001", CAM_LEFT, 6, 123.0, 456.0, ts="t")
    assert tuple(f.kp2d[CAM_LEFT][6]) == (123.0, 456.0)
    assert f.corrected[CAM_LEFT][6]
    assert len(stack.log) == 1

    stack.undo()
    assert np.allclose(f.kp2d[CAM_LEFT][6], old)
    assert not f.corrected[CAM_LEFT][6]

    stack.redo()
    assert tuple(f.kp2d[CAM_LEFT][6]) == (123.0, 456.0)


def test_live_resolve_moves_only_that_joint():
    f, rig, gt = _frame_with_pose()
    intr = Intrinsics(K=rig["K"], dist=rig["dist"], image_size=rig["size"])
    ext_l = Extrinsics(*rig["left"]); ext_r = Extrinsics(*rig["right"])

    # correct joint 6's left-view point to a new location, re-triangulate it
    stack = CorrectionStack({f.frame_id: f})
    # shift the left observation by 10px and re-solve just that joint
    new_l = f.kp2d[CAM_LEFT][6] + np.array([10.0, 0.0])
    stack.apply("0001", CAM_LEFT, 6, new_l[0], new_l[1])
    new_xyz = triangulate_one(
        f.kp2d[CAM_LEFT][6], f.kp2d[CAM_RIGHT][6], intr, intr, ext_l, ext_r)
    # the 3D point changed from ground truth (because we moved the obs)
    assert np.linalg.norm(new_xyz - gt[6]) > 1e-3


def test_undo_empty_is_safe():
    f, rig, gt = _frame_with_pose()
    stack = CorrectionStack({f.frame_id: f})
    assert stack.undo() is None
    assert not stack.can_undo()


# --- the log outlives the session that made it -----------------------------

def _saved_project(folder, n_frames=2):
    """An empty project on disk, as "Save Corrections" would have left it."""
    data = ProjectData(name="corr")
    for i in range(n_frames):
        data.frames.append(Frame(frame_id=f"{i:04d}"))
    save_project(data, folder)
    return data


def test_corrections_from_an_earlier_session_survive_the_next_save(tmp_path):
    """The client's Monday corrections must still be there on Tuesday.

    `ProjectModel.__init__` built the stack with an empty log while
    `load_project` had just read the stored corrections back, and `save()`
    assigned that empty-plus-today's log over `project.corrections` while
    `_write_corrections` DELETEd the table first — so every earlier session's
    corrections were destroyed by the button labelled "Save Corrections",
    which then reported the number that survived as though it were the total.
    """
    from pose3d.ui.model import ProjectModel

    _saved_project(tmp_path)

    monday = ProjectModel(load_project(tmp_path), None,
                          project_dir=str(tmp_path))
    for j in range(3):
        monday.set_joint_2d(CAM_LEFT, j, 10.0 + j, 20.0 + j)
    monday.save()

    tuesday = ProjectModel(load_project(tmp_path), None,
                           project_dir=str(tmp_path))
    assert len(tuesday.stack.log) == 3, "the stack was not seeded from the file"
    # ...as history, not as pending edits: Monday's work is not undoable today
    assert not tuesday.stack.can_undo()
    said = []
    tuesday.statusMessage.connect(said.append)
    tuesday.set_joint_2d(CAM_RIGHT, 5, 1.0, 2.0)
    tuesday.save()

    stored = load_project(tmp_path).corrections
    assert [c.joint for c in stored] == [0, 1, 2, 5]
    assert [c.cam for c in stored] == [CAM_LEFT] * 3 + [CAM_RIGHT]
    # the exact sentence: "4" alone also matches a tmp_path with a 4 in it
    assert said[-1].startswith("Saved 4 corrections to "), said[-1]


def test_saving_twice_stores_each_correction_once(tmp_path):
    """The log is keyed by the row it already occupies, so re-saving an
    unchanged project is not a way to double it."""
    from pose3d.ui.model import ProjectModel

    _saved_project(tmp_path)
    model = ProjectModel(load_project(tmp_path), None,
                         project_dir=str(tmp_path))
    model.set_joint_2d(CAM_LEFT, 4, 7.0, 8.0)
    model.save()
    model.save()
    model.save()

    assert len(load_project(tmp_path).corrections) == 1
    assert count_corrections(tmp_path) == 1


def test_a_save_keeps_a_correction_it_never_saw(tmp_path):
    """A row this session did not load is still the client's work.

    `append_correction` writes straight to the log, and a second app window on
    the same folder would too. Rewriting the table from one session's memory
    is how that work disappears.
    """
    from pose3d.ui.model import ProjectModel

    _saved_project(tmp_path)
    model = ProjectModel(load_project(tmp_path), None,
                         project_dir=str(tmp_path))
    model.set_joint_2d(CAM_LEFT, 2, 3.0, 4.0)
    append_correction(tmp_path, Correction(
        "0001", CAM_RIGHT, 9, (1.0, 1.0), (2.0, 2.0), "elsewhere"))

    model.save()

    # both are in the log; the order is the order they reached the FILE, and
    # the row written while this session held its edit in memory came first
    stored = load_project(tmp_path).corrections
    assert {(c.cam, c.joint) for c in stored} == {(CAM_LEFT, 2), (CAM_RIGHT, 9)}

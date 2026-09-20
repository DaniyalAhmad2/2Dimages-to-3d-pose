"""A take the bone fit cannot size must still be ONE size, in both panels.

`Character.fit_to_subject` returns None when no bone in the take could be
measured. `_frame_scale` then falls back to THIS frame's height ratio, which
changes the figure's size on every keyframe — 37.9 % over the client's take.
The 3D view reported that and carried on; the export discarded the same None
two lines below a comment promising "the same fit the 3D view applies", and
shipped the pulsing figure with `ok` True, no note, and a fixed camera placed
from whichever frame happened to be posable first.
"""
import numpy as np
import pytest

from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.geometry.placement import (
    NO_SCALE_NOTE, SCALE_FROM_HEIGHT_NOTE, take_scale)
from tests.gates import needs_character
from tests.synth import sample_skeleton_3d


@pytest.fixture
def unfittable(monkeypatch):
    """A Character whose bone fit finds nothing to fit against."""
    from pose3d.geometry.character import Character
    monkeypatch.setattr(Character, "fit_to_subject",
                        lambda self, poses: None)


def _take(n=5, grow=0.0, tilt_deg=0.0):
    """n frames of a standing pose; `grow` scales each frame a little more, so
    a per-frame size ratio can be told from one take-wide size.

    `tilt_deg` leans the whole take, which is what a calibration world frame
    from a tag taped to a wall does to every take shot against it — the
    client's own is 90 deg out.
    """
    from tests.synth import rot_about
    base = sample_skeleton_3d()
    poses = np.stack([base * (1.0 + grow * i) for i in range(n)])
    if tilt_deg:
        poses = poses @ rot_about((1, 0, 0), tilt_deg).T
    return poses


@needs_character()
def test_the_fallback_is_one_size_for_the_take(unfittable):
    from pose3d.geometry.character import Character
    ch = Character()
    poses = _take(grow=0.10)
    scale, note = take_scale(ch, poses)

    assert scale is not None
    assert note == SCALE_FROM_HEIGHT_NOTE
    # pinned, so every frame is posed through it rather than through its own
    # height — which is what stops the figure changing size keyframe to
    # keyframe, and what makes the preview and the export agree
    assert ch._scale == scale
    # the rig's height over the subject's median height, measured on the
    # DE-TILTED take (see `take_scale`: a height is a measurement in a frame
    # and the bone fit it replaces is not)
    from pose3d.geometry.orient import de_tilt_matrix, sequence_up
    level = poses @ de_tilt_matrix(sequence_up(poses)).T
    heights = [float(np.ptp(p[:, 2])) for p in level]
    assert np.isclose(scale, ch.rig_h / np.median(heights))
    for p in poses:
        assert ch._frame_scale(p, np.ones(NUM_JOINTS, bool)) == scale


@needs_character()
def test_a_take_with_no_height_at_all_says_so(unfittable):
    from pose3d.geometry.character import Character
    ch = Character()
    poses = np.full((3, NUM_JOINTS, 3), np.nan)
    poses[:, int(Joint.PELVIS)] = 0.0          # one point: no height
    scale, note = take_scale(ch, poses)
    assert scale is None and note == NO_SCALE_NOTE


@needs_character()
def test_a_fit_that_works_is_left_alone():
    from pose3d.geometry.character import Character
    ch = Character()
    poses = _take()
    scale, note = take_scale(ch, poses)
    assert note == ""
    assert scale == ch.fit_to_subject(poses)


@needs_character()
def test_the_export_takes_the_same_fallback_and_reports_it(unfittable):
    """The export's own document, built without Blender."""
    from pose3d.export.blender_export import _character_document
    from pose3d.geometry.character import Character

    from pose3d.geometry.orient import de_tilt_matrix, take_up

    poses = _take(grow=0.10)
    frag, reason = _character_document(poses, 0, None)
    assert reason is None, reason

    # the same take, de-tilted the way both sides de-tilt it
    up, _source, _spread = take_up(poses, None)
    ch = Character()
    want, note = take_scale(ch, poses @ de_tilt_matrix(up).T)
    assert frag["take_scale"] == pytest.approx(want)
    assert frag["fit_note"] == note == SCALE_FROM_HEIGHT_NOTE


@needs_character()
def test_the_view_says_exactly_what_the_export_says(unfittable):
    """One sentence, from one place: the client reads both panels."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from pose3d.export.blender_export import _character_document
    from pose3d.ui.view3d import View3D

    from pose3d.geometry.orient import de_tilt_matrix, take_up

    poses = _take(grow=0.10)
    view = View3D.__new__(View3D)
    # the orientation main_window hands the view (`_apply_view_orientation`),
    # which is the same one `_character_document` computes for itself — the
    # fit is measured on the de-tilted take, so a test that skipped this would
    # be comparing two different takes and calling the 0.02 % a disagreement
    up, _source, _spread = take_up(poses, None)
    view._R = de_tilt_matrix(up)
    view._vaxis, view._vsign = 2, 1.0
    view._character = None
    view._take = view._place = None
    view._char_error = view._char_error_source = ""
    said = []
    view._report = lambda msg, source="": said.append(msg)

    view.fit_subject(poses)
    frag, _reason = _character_document(poses, 0, None)
    assert said == [SCALE_FROM_HEIGHT_NOTE]
    assert frag["fit_note"] == said[0]
    assert view._character._scale == pytest.approx(frag["take_scale"])


@needs_character()
def test_the_fallback_does_not_depend_on_which_frame_it_is_handed(unfittable):
    """A height is a measurement in a frame; the bone fit it stands in for is
    not. A take leaning 30 deg must be sized the same as the same take upright,
    or the view and the export size the figure differently the moment they
    hold it in different frames."""
    from pose3d.geometry.character import Character

    upright = _take(grow=0.10)
    leaning = _take(grow=0.10, tilt_deg=30.0)
    a, _note = take_scale(Character(), upright)
    b, _note = take_scale(Character(), leaning)
    assert a == pytest.approx(b, rel=1e-9)


@needs_character()
def test_a_view_with_no_orientation_yet_still_matches_the_export(unfittable):
    """`View3D.fit_subject` uses the raw world poses while `_R` is None —
    before `main_window._apply_view_orientation` has run, or after it decided
    there was no vertical to level on — and the export always de-tilts. They
    must still agree: the export IS the preview, and a size that depended on
    which of the two ran first is exactly the class of bug this pass is about.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from pose3d.export.blender_export import _character_document
    from pose3d.ui.view3d import View3D

    poses = _take(grow=0.10, tilt_deg=30.0)
    view = View3D.__new__(View3D)
    view._R = None                      # no orientation has been set
    view._vaxis, view._vsign = None, 1.0
    view._character = None
    view._take = view._place = None
    view._char_error = view._char_error_source = ""
    view._report = lambda msg, source="": None

    view.fit_subject(poses)
    frag, _reason = _character_document(poses, 0, None)
    assert view._character._scale == pytest.approx(frag["take_scale"], rel=1e-9)

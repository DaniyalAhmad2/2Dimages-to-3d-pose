"""Calibration faults must be reported, not silently skew every reconstruction.

Both faults below still produce a 3D pose — a wrong one — and they look like the
character is at fault, so the app has to name them.
"""
import numpy as np

from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics
from pose3d.calib.quality import (
    CalibrationNote, check_rig, is_problem, looks_assumed, world_up_tilt)
from pose3d.pipeline import CalibratedRig
from pose3d.core.project import CAM_LEFT, CAM_RIGHT


def _intr(w=1920, h=1080, f=None, dist=None, measured=False):
    f = float(max(w, h)) if f is None else f
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    d = np.zeros((1, 5)) if dist is None else np.asarray(dist, float).reshape(1, -1)
    return Intrinsics(K=K, dist=d, image_size=(w, h), rms=0.4 if measured else 0.0)


def _upright_cam(pos=(0.0, -3.0, 1.5)):
    """A camera standing upright in a world whose +Z really is up."""
    fwd = np.array([0.0, 1.0, 0.0])            # looking along +Y
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    R = np.stack([right, -up, fwd])            # world -> camera
    t = -R @ np.asarray(pos, float)
    return Extrinsics(R=R, t=t)


def _tipped_cam(pos=(0.0, -3.0, 1.5)):
    """The same camera, but the calibration board stood upright, so the world's
    Z axis points sideways instead of up."""
    e = _upright_cam(pos)
    # rotate the world 90 deg about X: world +Z now lies horizontal
    W = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    return Extrinsics(R=e.R @ W.T, t=e.t)


def _rig(ext, intr_l, intr_r):
    return CalibratedRig(intr_l, intr_r, ext[0], ext[1])


def test_level_calibration_reports_nothing():
    rig = _rig((_upright_cam((-0.3, -3, 1.5)), _upright_cam((0.3, -3, 1.5))),
               _intr(dist=[0.1, -0.02, 0, 0, 0], f=1500, measured=True),
               _intr(dist=[0.1, -0.02, 0, 0, 0], f=1500, measured=True))
    assert world_up_tilt(rig) < 5.0
    assert check_rig(rig) == []


def test_world_frame_not_vertical_is_reported():
    """The world frame's up comes from one arbitrarily-rotated marker tag, so
    the user has to be told upright is being taken from the subject instead —
    what that costs (a lean held all take reads as upright), and the one thing
    that would change it (a tag on the floor next time)."""
    rig = _rig((_tipped_cam((-0.3, -3, 1.5)), _tipped_cam((0.3, -3, 1.5))),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True))
    assert world_up_tilt(rig) > 60.0
    msgs = " ".join(check_rig(rig)).lower()
    assert "away from upright" in msgs             # the number, in plain words
    assert "from the subject's own body" in msgs   # which vertical is in use
    assert "lay one tag flat on the floor" in msgs # the one action it implies
    # ...and none of the jargon it used to be written in
    assert "world frame" not in msgs and "nominal" not in msgs


def test_assumed_intrinsics_are_reported_as_a_note_with_one_action():
    """Focal guessed from the image size, principal point dead centre, no
    distortion — the signature of intrinsics that were never measured.

    It is a NOTE, not a problem: the reconstruction is real, the depth is
    simply looser. The old wording ("shoot a checkerboard with each camera to
    calibrate them") prescribed something the client cannot do — he owns no
    checkerboard, there is no calibration screen in the app to use one with,
    and CLIENT_GUIDE.md tells him neither is needed.
    """
    assert looks_assumed(_intr())                       # f == max(w, h)
    assert not looks_assumed(_intr(f=1500, dist=[0.1, -0.02, 0, 0, 0]))
    rig = _rig((_upright_cam(), _upright_cam()), _intr(), _intr())
    notes = check_rig(rig)
    msgs = " ".join(notes).lower()
    assert "estimated from the photo size" in msgs
    assert "checkerboard" not in msgs
    assert "one-time calibration photo set" in msgs
    assert not any(is_problem(n) for n in notes)


def test_mismatched_resolutions_are_reported_as_a_note():
    rig = _rig((_upright_cam(), _upright_cam()),
               _intr(3072, 4080, f=4000, dist=[0.1, 0, 0, 0, 0], measured=True),
               _intr(1536, 2048, f=2000, dist=[0.1, 0, 0, 0, 0], measured=True))
    notes = check_rig(rig)
    msgs = " ".join(notes).lower()
    assert "different sizes" in msgs and "3072x4080" in msgs
    # ...and it says what it costs and what to do, not "each needs its own
    # calibration", which reads as an instruction and is not one
    assert "not a fault" in msgs
    assert not any(is_problem(n) for n in notes)


def test_the_clients_own_rig_reads_as_calibrated_with_notes():
    """Two different phones, no intrinsics files, tags on a wall: the client's
    actual setup, on which every correct import used to raise three warnings.

    A non-technical user reads three warnings on a successful run as "this
    output cannot be trusted", which is how P2 ("all of the detections seem to
    be a very low accuracy score") restarts.
    """
    rig = _rig((_tipped_cam((-0.3, -3, 1.5)), _tipped_cam((0.3, -3, 1.5))),
               _intr(3072, 4080), _intr(1536, 2048))
    recorded = (np.array([0.0, 0.0, 1.0]), "camera pair + tag row", 9.0)
    notes = check_rig(rig, recorded)

    assert notes, "a rig with guessed lenses should still say so"
    assert not any(is_problem(n) for n in notes)
    # the tag frame's own tilt is not a problem once the view is not using it
    assert "away from upright" not in " ".join(notes)


def test_a_real_problem_is_still_a_problem():
    """Only a note knows it is a note. Anything else — a rig_error the window
    inserts, a message from a failed solve — is a problem by default, so the
    headline cannot go green on a calibration that did not work."""
    assert is_problem("No tag was seen by both cameras.")
    assert not is_problem(CalibrationNote("the lenses were estimated"))
    assert is_problem(CalibrationNote("the solve failed",
                                      CalibrationNote.PROBLEM))


def test_a_note_put_through_a_string_operation_fails_safe():
    """`severity` cannot survive `str` returning plain strings, so the class
    says so and this pins which way it fails: a note that has been sliced,
    stripped or f-stringed reads as a PROBLEM, which at worst calls a good
    calibration doubtful — never the other way round."""
    note = CalibrationNote("the lenses were estimated")
    assert not is_problem(note)
    assert is_problem(note.strip())
    assert is_problem(f"{note}")
    assert is_problem(" ".join([note]))


def test_no_rig_is_not_an_error():
    assert check_rig(None) == []


def test_a_recorded_vertical_in_use_makes_the_tilt_a_non_event():
    """The world frame's tilt is only worth a line because of what the view
    does about it. Once a recorded vertical is in use, the tag frame's own
    rotation costs nothing at all: it is a fact about a marker taped to a
    wall, not a problem with the take, and printing it as one is how a correct
    import came out amber.

    (The sidebar still says which vertical is in use — `show_levelling_note`,
    which has the project's answer rather than a guess from the rig.)
    """
    rig = _rig((_tipped_cam((-0.3, -3, 1.5)), _tipped_cam((0.3, -3, 1.5))),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True))
    recorded = (np.array([0.0, 0.0, 1.0]), "camera pair + tag row", 9.0)

    assert check_rig(rig, recorded) == []
    # ...and without one, the line about where upright comes from stands
    assert "from the subject's own body" in " ".join(check_rig(rig))


def test_a_vertical_too_uncertain_to_use_is_named_as_unused():
    """`orient.take_up` refuses a recorded vertical whose proxies disagree by
    more than 20 deg, and the sidebar used to claim it was in use anyway — so
    the user was told a lean held for the whole take is preserved while it was
    in fact being normalised away, in the preview and in the export both."""
    rig = _rig((_tipped_cam((-0.3, -3, 1.5)), _tipped_cam((0.3, -3, 1.5))),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True))
    too_wide = (np.array([0.0, 0.0, 1.0]), "camera pair + tag row", 25.0)

    msgs = " ".join(check_rig(rig, too_wide))
    assert "from the subject's own body" in msgs
    assert "±25°" in msgs and "20°" in msgs     # why it is not being used
    # ...and just inside the gate it IS used, and the tilt stops being news
    ok = (np.array([0.0, 0.0, 1.0]), "camera pair + tag row", 19.0)
    assert check_rig(rig, ok) == []


def test_an_unverified_vertical_does_not_change_what_the_tilt_costs():
    """The softened warning is only true because the view USES the recorded
    vertical. A vertical with an unknown spread (one proxy, nothing checking
    it) is refused by orient.take_up, so the view is back on the subject and
    the original warning is the true one. A warning that is no longer true is
    worse than none — and so is one that is true but suppressed."""
    rig = _rig((_tipped_cam((-0.3, -3, 1.5)), _tipped_cam((0.3, -3, 1.5))),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True),
               _intr(dist=[0.1, 0, 0, 0, 0], f=1500, measured=True))
    unverified = (np.array([0.0, 0.0, 1.0]), "camera up", None)
    msgs = " ".join(check_rig(rig, unverified))
    assert "from the subject's own body" in msgs
    assert "±" not in msgs                     # nothing to put a number on

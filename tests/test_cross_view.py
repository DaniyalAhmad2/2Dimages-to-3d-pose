"""The cross-view gate must reject hallucinations, not good data.

When it over-fires the result reads as the detector failing: gaps in the 2D
views and a sparse 3D preview. That is exactly what a flat 30 px tolerance did
on 3072x4080 phone captures — 0.7% of image height, which threw away 38% of a
real take.

It is a MASK (`Frame.rejected`), not a deletion: the observations stay in
`frame.kp2d`, drawn and draggable, and the mask is re-derived from them on
every recompute. It used to NaN them in place, which `save_project` then made
permanent — see `test_a_bad_rig_does_not_destroy_kp2d`.

Everything that exercises the gate runs on both rigs: the wide symmetric one
blown up to phone resolution, and the client's genuinely asymmetric close-range
rig, whose two images differ 2:1 in pixel scale.
"""
from pathlib import Path

import numpy as np
import pytest

from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData
from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.pipeline import (
    CalibratedRig, epipolar_threshold, triangulate_project, validate_cross_view,
)
from tests.synth import (
    cameras, close_range_two_cam, default_two_cam, project as project_points,
)
from pose3d.calib.extrinsics import Extrinsics
from pose3d.calib.intrinsics import Intrinsics


def _upscaled_symmetric(scale=4.0):
    """The 720p synthetic rig blown up to phone-sized images."""
    geo = default_two_cam()
    K = geo["K"].copy() * scale
    K[2, 2] = 1.0
    size = (int(geo["size"][0] * scale), int(geo["size"][1] * scale))
    return {**geo, "K": K, "size": size}


@pytest.fixture(params=["wide symmetric at phone resolution",
                        "close asymmetric (client rig)"])
def geo(request):
    return (_upscaled_symmetric() if request.param.startswith("wide")
            else close_range_two_cam())


def _rig_and_project(geo, n_frames=2):
    """A consistent two-view capture of `geo`'s subject."""
    cams = cameras(geo)
    intr = {c: Intrinsics(K=cams[c][0], dist=cams[c][1], image_size=cams[c][2])
            for c in (CAM_LEFT, CAM_RIGHT)}
    rig = CalibratedRig(intr[CAM_LEFT], intr[CAM_RIGHT],
                        Extrinsics(*geo["left"]), Extrinsics(*geo["right"]))

    gt = geo["subject"]
    data = ProjectData(name="cv")
    for i in range(n_frames):
        f = Frame(frame_id=f"{i:04d}")
        for cam in (CAM_LEFT, CAM_RIGHT):
            f.kp2d[cam] = project_points(gt, cams[cam][0], cams[cam][1], *geo[cam])
            f.scores[cam] = np.full(NUM_JOINTS, 0.9)
        data.frames.append(f)
    return rig, data, gt


def test_consistent_observations_survive_at_phone_resolution(geo):
    """Regression guard: with a flat 30 px tolerance this dropped a third of a
    real take, which read to the user as 'half the keypoints not detected'."""
    rig, data, _ = _rig_and_project(geo)
    dropped = validate_cross_view(data, rig)
    assert dropped == 0, f"{dropped} good observations rejected"
    assert not np.isnan(data.frames[0].kp2d[CAM_LEFT]).any()


def test_the_tolerance_scales_with_the_image():
    """With no take to measure, the gate is its CEILING — the widest it may
    ever be — and that still scales with the sensor."""
    small, _, _ = _rig_and_project(default_two_cam())
    big, _, _ = _rig_and_project(_upscaled_symmetric(4.0))
    assert epipolar_threshold(big) > 3.0 * epipolar_threshold(small)
    # and still reproduces roughly the historical 30 px at ~1080p
    hd = Intrinsics(K=np.eye(3), dist=np.zeros((1, 5)), image_size=(1920, 1080))
    assert 25.0 < epipolar_threshold(CalibratedRig(hd, hd, *[Extrinsics(
        R=np.eye(3), t=np.zeros(3))] * 2)) < 40.0


def test_triangulate_reports_what_it_threw_away(geo):
    """The count has to reach the UI, or a calibration problem masquerades as
    a detection problem with nothing to tell them apart."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][int(Joint.LEFT_ANKLE)] = 0.3
    assert triangulate_project(data, rig) == 1


def test_the_rejection_note_only_fires_when_it_matters():
    """One note, owned by the pipeline that owns the policy, so the import
    dialog and the recalculate status line cannot drift apart."""
    from pose3d.pipeline import rejection_note
    assert rejection_note(0, 10) == ""
    assert rejection_note(5, 10) == ""                   # 3%: normal occlusion
    note = rejection_note(60, 10)                        # 40%: something is wrong
    assert "40%" in note and "calibration" in note.lower()


def test_an_uncalibrated_import_still_reports_a_count():
    """Regression guard: `dropped` used to be bound only inside the
    `if rig is not None` branch, so importing without a calibration raised
    NameError into the dialog's blanket handler and showed "Import failed"."""
    from pose3d.pipeline import rejection_note
    assert rejection_note(0, 0) == ""      # no frames, no calibration, no crash


def test_a_hallucinated_joint_is_still_dropped(geo):
    """The gate's actual job: one view putting the ankle on the knee."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    f.kp2d[CAM_LEFT][int(Joint.LEFT_ANKLE)] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][int(Joint.LEFT_ANKLE)] = 0.3      # the weaker view

    dropped = validate_cross_view(data, rig)
    assert dropped == 1
    assert f.rejected[CAM_LEFT][int(Joint.LEFT_ANKLE)]
    assert not f.rejected[CAM_RIGHT][int(Joint.LEFT_ANKLE)]


# --- the gate is a mask, not a deletion (F06) -------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "client_take"


def _damaged_left(rig, deg=12.0, axis=(0, 1, 0)):
    """The same rig with the LEFT camera's orientation wrong by `deg`.

    12 deg is the audit's measured arming point for the destructive gate: it
    dropped 0 observations up to 5 deg, 12 at 8 deg and 185 of 780 at 12 deg
    on the client take. It is not a hypothetical — taking the wrong IPPE
    branch for a single tag is worth 63.7 deg, and this rig is solved from one
    tag.
    """
    from tests.synth import rot_about
    dR = rot_about(axis, deg)
    return CalibratedRig(
        rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
        Extrinsics(R=dR @ rig.ext[CAM_LEFT].R, t=dR @ rig.ext[CAM_LEFT].t),
        rig.ext[CAM_RIGHT])


def test_a_bad_rig_does_not_destroy_kp2d(tmp_path):
    """Damage -> save -> reload -> recompute with the good rig: EVERY
    observation comes back.

    Before this, it recovered 0 of them. The gate wrote NaN into `frame.kp2d`
    and 0.0 into `frame.scores`, `save_project` persisted that, and nothing in
    the app could undo it: "Recalculate 3D" re-ran the same gate over the
    already emptied 2D, and the NaN'd joints were not even drawn, so the user
    could not drag them back or tell the loss from a detection failure. Only a
    full re-detection restored them, and nothing said so.

    On the client's own rig and 2D (the committed fixture), a 12 deg error
    about the left camera's vertical rejects 347 of its 778 observations.
    """
    from pose3d.core.io_project import load_project, save_project
    from pose3d.quality import load_rig

    data = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    before = {(i, c): data.frames[i].kp2d[c].copy()
              for i in range(len(data.frames)) for c in (CAM_LEFT, CAM_RIGHT)}
    n_obs = sum(int(np.isfinite(v).all(1).sum()) for v in before.values())

    dropped = triangulate_project(data, _damaged_left(rig))
    assert dropped > 100, f"only {dropped} rejected; the damage must bite"
    # ...and every one of them is a MASK, not a deletion
    assert sum(int(f.rejected[c].sum()) for f in data.frames
               for c in (CAM_LEFT, CAM_RIGHT)) == dropped
    for (i, c), xy in before.items():
        assert np.array_equal(data.frames[i].kp2d[c], xy, equal_nan=True)

    save_project(data, tmp_path)
    reloaded = load_project(tmp_path)
    triangulate_project(reloaded, rig)          # the GOOD rig, as recompute does

    after = sum(int(np.isfinite(f.kp2d[c]).all(1).sum())
                for f in reloaded.frames for c in (CAM_LEFT, CAM_RIGHT))
    assert after == n_obs, f"{n_obs - after} of {n_obs} observations lost"
    assert not any(f.rejected[c].any() for f in reloaded.frames
                   for c in (CAM_LEFT, CAM_RIGHT)), "stale mask survived"
    # and the 3D the take had before the damage is back, joint for joint
    clean = load_project(FIXTURE)
    triangulate_project(clean, rig)
    for a, b in zip(clean.frames, reloaded.frames):
        assert np.array_equal(a.pose3d, b.pose3d, equal_nan=True)


def test_the_mask_decides_the_3d_and_nothing_else(geo):
    """A rejected observation must not reach the triangulation — and must not
    leave the 2D arrays either."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    ankle = int(Joint.LEFT_ANKLE)
    f.kp2d[CAM_LEFT][ankle] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][ankle] = 0.3

    assert triangulate_project(data, rig) == 1
    assert f.rejected[CAM_LEFT][ankle] and not f.rejected[CAM_RIGHT][ankle]
    assert np.isfinite(f.kp2d[CAM_LEFT][ankle]).all()   # still there, still drawn
    assert f.scores[CAM_LEFT][ankle] == 0.3             # and still its own score
    assert np.isnan(f.pose3d[ankle]).all()              # but no 3D from it


def test_the_mask_is_re_derived_not_accumulated(geo):
    """Put the hallucinated point back where it belongs and the rejection
    goes away — the mask describes the 2D as it is now, every time."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    ankle = int(Joint.LEFT_ANKLE)
    good = f.kp2d[CAM_LEFT][ankle].copy()
    f.kp2d[CAM_LEFT][ankle] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][ankle] = 0.3
    assert triangulate_project(data, rig) == 1

    f.kp2d[CAM_LEFT][ankle] = good
    assert triangulate_project(data, rig) == 0
    assert not any(f.rejected[c].any() for f in data.frames
                   for c in (CAM_LEFT, CAM_RIGHT))
    assert not np.isnan(f.pose3d[ankle]).any()


def test_a_hand_placed_point_is_not_left_flagged(geo):
    """`Frame.set_kp` is the user overruling the gate; the purple dot must not
    outlive the point it was measured on."""
    rig, data, _ = _rig_and_project(geo)
    f = data.frames[0]
    ankle = int(Joint.LEFT_ANKLE)
    f.kp2d[CAM_LEFT][ankle] = f.kp2d[CAM_LEFT][int(Joint.LEFT_KNEE)]
    f.scores[CAM_LEFT][ankle] = 0.3
    triangulate_project(data, rig)
    assert f.rejected[CAM_LEFT][ankle]

    f.set_kp(CAM_LEFT, ankle, 10.0, 20.0, score=1.0, corrected=True)
    assert not f.rejected[CAM_LEFT][ankle]


# --- the gate is sized from the data, per image (F36) -----------------------

def test_the_gate_uses_each_image_s_own_scale(geo):
    """1.4 % of THAT image's diagonal, not 1.4 % of the left one's.

    The old rule read `rig.intr[CAM_LEFT].image_size` and applied the answer
    to a Sampson distance that mixes both cameras' pixel units. On the
    client's genuinely 2:1 rig that made the gate 1.400 % of the left diagonal
    and 2.793 % of the right's — the low-resolution camera, whose pixels are
    worth twice as much, got twice the licence.
    """
    from pose3d.pipeline import per_image_allowances

    rig, data, _ = _rig_and_project(geo)
    allow = per_image_allowances(rig)
    for cam in (CAM_LEFT, CAM_RIGHT):
        assert allow[cam] == pytest.approx(
            0.014 * np.hypot(*rig.intr[cam].image_size))
    ratio = np.hypot(*rig.intr[CAM_LEFT].image_size) / np.hypot(
        *rig.intr[CAM_RIGHT].image_size)
    assert allow[CAM_LEFT] / allow[CAM_RIGHT] == pytest.approx(ratio)
    # and consistent observations still pass both halves of the gate
    assert validate_cross_view(data, rig) == 0


def test_the_per_image_gate_rejects_what_sampson_would_wave_through(geo):
    """"EITHER exceeds", not "both".

    Sampson is (d_L^-2 + d_R^-2)^-1/2 — below BOTH per-image distances and
    dominated by the smaller — so a disagreement can be over an image's own
    allowance while the pair's Sampson distance is not. Requiring both to
    exceed would have needed d_R > 35.8 px on the client rig when its measured
    maximum is 33.8: strictly more permissive than the rule it replaces.
    """
    from pose3d.pipeline import per_image_allowances, point_line_distances
    from pose3d.geometry.triangulate import fundamental_matrix

    rig, data, _ = _rig_and_project(geo)
    allow = per_image_allowances(rig)
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    f = data.frames[0]
    j = int(Joint.LEFT_WRIST)

    # walk the right-view point off its epipolar line until it is just over
    # that image's own allowance
    base = f.kp2d[CAM_RIGHT][j].copy()
    for step in np.arange(1.0, 400.0, 1.0):
        f.kp2d[CAM_RIGHT][j] = base + (0.0, step)
        _, d_r = point_line_distances(f.kp2d[CAM_LEFT][j],
                                      f.kp2d[CAM_RIGHT][j], F)
        if d_r > allow[CAM_RIGHT]:
            break
    else:                                        # pragma: no cover
        pytest.fail("could not push the point past its own image's allowance")

    # a Sampson-only gate wide enough to miss it still leaves the per-image
    # test to catch it
    assert validate_cross_view(data, rig, epi_thr=1e9) == 1
    assert f.rejected[CAM_RIGHT][j] or f.rejected[CAM_LEFT][j]


def test_the_gate_is_sized_from_the_take_not_the_sensor():
    """clip(6 x median, 25 px, 1.4 % of the smaller diagonal)."""
    from pose3d.pipeline import epipolar_gate

    assert epipolar_gate(5.0, 100.0) == pytest.approx(30.0)   # 6 x median
    assert epipolar_gate(0.0, 100.0) == pytest.approx(25.0)   # floored
    assert epipolar_gate(50.0, 35.8) == pytest.approx(35.8)   # capped
    assert epipolar_gate(float("nan"), 35.8) == pytest.approx(35.8)


def test_the_client_take_is_gated_on_its_own_distribution():
    """The acceptance number: 71.5 px -> ~29 px, still 0 of 388 rejected.

    Measured on the committed fixture: Sampson median 4.908, p99 23.97, max
    29.384 px; the gate lands at 29.45, which clears that tail by 0.06 px. The
    left image's own allowance is 71.50 px (max observed 59.33) and the
    right's 35.84 px (max observed 33.82), so neither per-image test fires
    either. If this ever drops a pair, the gate has become tighter than the
    take it is judging and the k in `pipeline._EPI_K` is what to look at.
    """
    from pose3d.core.io_project import load_project
    from pose3d.quality import load_rig

    data = load_project(FIXTURE)
    rig = load_rig(FIXTURE / "calibration")
    thr = epipolar_threshold(rig, data)
    assert 25.0 <= thr <= 30.0, thr                  # today 29.45
    assert epipolar_threshold(rig) == pytest.approx(35.84, abs=0.01)  # ceiling
    assert validate_cross_view(data, rig) == 0


def test_the_gate_and_the_metric_measure_the_same_thing():
    """`pipeline.point_line_distances` and `quality._point_line_px` are one
    formula in two places; a drift between them would let the sidebar report
    a disagreement the gate never saw."""
    from pose3d.geometry.triangulate import fundamental_matrix
    from pose3d.pipeline import point_line_distances
    from pose3d.quality import _point_line_px

    rig, data, _ = _rig_and_project(close_range_two_cam())
    F = fundamental_matrix(rig.intr[CAM_LEFT], rig.intr[CAM_RIGHT],
                           rig.ext[CAM_LEFT], rig.ext[CAM_RIGHT])
    rng = np.random.default_rng(0)
    f = data.frames[0]
    for j in range(NUM_JOINTS):
        pl = f.kp2d[CAM_LEFT][j] + rng.normal(0, 40, 2)
        pr = f.kp2d[CAM_RIGHT][j] + rng.normal(0, 40, 2)
        assert point_line_distances(pl, pr, F) == pytest.approx(
            _point_line_px(pl, pr, F))
    assert all(np.isnan(v) for v in
               point_line_distances([np.nan, 0.0], f.kp2d[CAM_RIGHT][0], F))


def test_the_narrower_gate_catches_the_hallucination_it_names():
    """The gate's own docstring names an ankle collapsed onto the knee; the
    71.5 px rule caught it in 15 of the client take's 25 eligible frames.

    Measured on the committed fixture, injecting the hallucination one frame
    at a time (so the corruption cannot widen the very median that sizes the
    gate): ankle-on-knee 15 -> 21 of 25, wrist-on-elbow 8 -> 25 of 26. The
    four ankle frames still missed sit as low as 9.2 px of disagreement —
    below this take's own tail (max 29.38 px), so no threshold can catch them
    without rejecting good data instead.
    """
    from pose3d.core.io_project import load_project
    from pose3d.quality import load_rig

    rig = load_rig(FIXTURE / "calibration")
    clean = load_project(FIXTURE)
    thr = epipolar_threshold(rig, clean)          # sized on the CLEAN take

    caught = {}
    for name, bad, good in (("ankle", Joint.LEFT_ANKLE, Joint.LEFT_KNEE),
                            ("wrist", Joint.LEFT_WRIST, Joint.LEFT_ELBOW)):
        n = hit = 0
        for i in range(len(clean.frames)):
            data = load_project(FIXTURE)
            f = data.frames[i]
            if np.isnan(f.kp2d[CAM_LEFT][int(good)]).any() \
                    or np.isnan(f.kp2d[CAM_RIGHT][int(bad)]).any():
                continue
            n += 1
            f.kp2d[CAM_LEFT][int(bad)] = f.kp2d[CAM_LEFT][int(good)]
            f.scores[CAM_LEFT][int(bad)] = 0.3
            validate_cross_view(data, rig, epi_thr=thr)
            hit += bool(f.rejected[CAM_LEFT][int(bad)])
        caught[name] = (hit, n)

    # floors with the project's 10-15 % margin under today's measurement, so
    # a change that moves one or two frames reports a number instead of a
    # failure; the OLD rule is the thing they have to stay clear of (15/25).
    assert caught["ankle"][0] >= 18, caught["ankle"]     # today 21 of 25
    assert caught["wrist"][0] >= 22, caught["wrist"]     # today 25 of 26
    assert caught["ankle"][0] > 15, "no better than the 71.5 px rule"


def test_the_sidebar_quotes_the_gate_that_was_applied():
    """`quality.body_epipolar`'s `threshold_px` IS `epipolar_threshold`.

    The sidebar's "Cross-view gate" row and `frac_over_threshold` come from
    the quality block, and it computed the same formula with the wrong
    ceiling: the smaller image's RAW diagonal (2560 px) instead of 1.4 % of it
    (35.84 px). It agreed on a clean take, where the cap does not bind, and
    lied exactly when it mattered — on the 12 deg rig below it reported a
    threshold of 555.77 px and "0 % over threshold" for a take whose every
    pair the gate had just rejected, which is the one diagnosis the row
    exists to give.
    """
    from pose3d.core.io_project import load_project
    from pose3d.quality import load_rig, take_quality

    rig = load_rig(FIXTURE / "calibration")
    for label, r in (("good", rig), ("12 deg out", _damaged_left(rig))):
        data = load_project(FIXTURE)
        dropped = triangulate_project(data, r)
        q = take_quality(data, r)
        assert q.epipolar["threshold_px"] == pytest.approx(
            epipolar_threshold(r, data)), label
        # and the two agree about what they have just done to the take
        n_pairs = q.epipolar["n_pairs"]
        assert (q.epipolar["frac_over_threshold"] > 0.5) == (
            dropped > 0.5 * n_pairs), (label, dropped, q.epipolar)


def test_losing_a_joint_narrows_the_gate_onto_the_takes_own_tail():
    """The gate is sized from the take it is judging, and this take's tail is
    0.06 px inside it.

    `clip(6 x median, 25 px, ceiling)` on the fixture is 29.449 px against a
    measured Sampson maximum of 29.384 — the margin is 0.2 %. Black out one
    joint in one view for the whole take and those 26 pairs leave the
    distribution, the median falls, and the gate closes to 28.439 px, which
    rejects two pairs of the take's own tail that nothing is wrong with.

    Recorded, not gated: k = 6 is the plan's constant and the take still drops
    0 of 388 undamaged. This is the number to look at when a change to the
    detector or the rig moves the median, and it is the reason
    `test_fallback_lengths_scale_to_the_subject` measures the fallback table
    with the gate held fixed.
    """
    from pose3d.core.io_project import load_project
    from pose3d.quality import load_rig

    rig = load_rig(FIXTURE / "calibration")
    clean = load_project(FIXTURE)
    thr_clean = epipolar_threshold(rig, clean)
    assert validate_cross_view(clean, rig) == 0

    damaged = load_project(FIXTURE)
    for f in damaged.frames:
        f.kp2d[CAM_LEFT][int(Joint.RIGHT_HIP)] = np.nan
        f.scores[CAM_LEFT][int(Joint.RIGHT_HIP)] = 0.0
    thr_damaged = epipolar_threshold(rig, damaged)

    assert thr_damaged < thr_clean                  # 28.439 vs 29.449 px
    assert thr_clean - thr_damaged < 2.0, (thr_clean, thr_damaged)   # 1.01 px
    # ...and that alone rejects a few pairs the clean gate is happy with
    assert validate_cross_view(damaged, rig) <= 6   # today 2

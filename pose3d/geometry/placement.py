"""Where a take's figure sits: the ground under one frame, the floor under a take.

Pure geometry, deliberately Qt-free. The live rule used to be stranded inside
`pose3d.ui.view3d`, which imports pyqtgraph and OpenGL, so the two callers that
cannot have a Qt stack — `pose3d.quality`, which every CLI tool and this
suite's metrics go through, and `tools/check_export_fidelity.py` — each carried
their own copy. `quality` kept measuring the lowest-mesh-vertex rule the view
had already replaced and printed the result to the client as "the figure bobs
against a fixed grid by the peak-to-peak", which the take-wide seat means
nobody ever sees. One implementation here, imported by all three, is what makes
changing the seat change what all three say.

It sits beside `Character.ground_drop` and `take_pelvis_ref` because those are
the other two halves of the same question — how tall the rig's foot is, where
the take's pelvis is, and where its floor is.
"""
from __future__ import annotations

import math

import numpy as np

from pose3d.core.skeleton import NUM_JOINTS, Joint
from pose3d.geometry.orient import de_tilt_matrix, sequence_up

#: The share of a take's frames allowed to sit BELOW its floor. The floor is
#: an order statistic over the per-frame seats, not a mean or an interpolated
#: percentile: it is always a height some frame's sole actually reached.
FLOOR_QUANTILE = 0.10

#: Said by the view AND by the export, because it is the same fact about the
#: same take and the client is looking at both.
SCALE_FROM_HEIGHT_NOTE = (
    "The character could not be sized to this subject by its bones (no bone "
    "was reconstructed well enough to fit against), so it is sized from the "
    "take's overall height instead: one size for the whole take, but a "
    "rougher one.")
NO_SCALE_NOTE = (
    "The character could not be sized to this subject at all (no frame has a "
    "measurable height), so it is drawn at the rig's own size.")


def ground_datum(verts, joints, drop):
    """View-space z of the ground plane under a posed character.

    The SOLE beneath the lower ANKLE, not the lowest mesh vertex. Which vertex
    is lowest changes from frame to frame — a foot, a knee, a fingertip — so
    the old rule slid the ground plane about under the figure and it bobbed
    against the grid by up to 11 % of body height. The ankles are tracked
    joints, so this datum moves only when the subject does. `drop` is the rig's
    rest ankle-to-sole height in these same units (`Character.ground_drop`).
    Falls back to the old rule when neither ankle could be posed.

    The sole is deliberately NOT levelled onto the plane: the shin of this
    rigid-footed mannequin genuinely tilts 16-86 deg, and flattening the foot
    would replace a measurement with a convention.
    """
    if joints is not None:
        z = [joints[int(j)][2]
             for j in (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)]
        z = [q for q in z if np.isfinite(q)]
        if z:
            return float(min(z)) - float(drop)
    return float(verts[:, 2].min())


def take_floor(seats, quantile: float = FLOOR_QUANTILE) -> float | None:
    """The one ground plane for a whole take, from its per-frame `seats`.

    The take is seated ONCE (see `View3D._take_placement`) so that what moves
    on screen is the subject and nothing else, and so that the preview and the
    export are the same rigid map. That makes the choice of seat the whole
    ballgame: seating on `min(seats)` hands the decision to the single worst
    frame of the take, and every other frame then floats by however far that
    one dipped — 34.3 % of body height on the client's take, which is how a fix
    for "the model clips through the floor" produced "the model hovers over the
    floor".

    So: the lowest seat EXCEPT the lowest `quantile` of the frames, taken as an
    order statistic (`ceil`, so a single dip is discounted even in a short
    take). A subject who stood still is seated on the ground he stood on; a
    frame that really was lower — a crouch, or a foot the reconstruction put
    through the floor — is drawn below the grid, which is the honest reading of
    it; and a jump still leaves the floor, because the frames that are high are
    the ones the quantile keeps.

    With fewer than three frames there is nothing to be robust about and the
    lowest seat is used: discarding one of two samples as an outlier is not
    robustness, it is a coin toss. Returns None when no frame yielded a seat.
    """
    finite = sorted(float(s) for s in seats if s is not None and np.isfinite(s))
    if not finite:
        return None
    if len(finite) < 3:
        return finite[0]
    k = int(math.ceil(float(quantile) * (len(finite) - 1)))
    return finite[min(k, len(finite) - 1)]


def _widest_span(points) -> float:
    """The largest distance between any two of these points.

    Rotation-invariant, which is the whole reason it is here: it stands in
    for a height on a take with no body axis to measure a height along, and a
    measure that changed with the caller's frame would put the 3D view and
    the export back on two different sizes. On a standing figure the widest
    span IS head-to-foot.
    """
    p = np.asarray(points, float)
    if len(p) < 2:
        return 0.0
    d = p[:, None, :] - p[None, :, :]
    return float(np.sqrt((d * d).sum(-1)).max())


def take_scale(character, poses):
    """(scale, note): ONE uniform size for the whole take, and why.

    `Character.fit_to_subject` is the real answer — least squares against the
    subject's median bone lengths — and it returns None when no bone in the
    take could be measured. What used to happen then was nothing, twice over:
    `_frame_scale` fell back to THIS frame's height ratio, so the figure
    changed size on every keyframe (37.9 % over the client's take), and while
    the 3D view at least said so, the export discarded the same None two lines
    below a comment promising "the same fit the 3D view applies" — shipping a
    pulsing character with `ok` True and no note anywhere, and a fixed camera
    placed from whichever frame happened to be posable first.

    So the fallback is made ONCE, here, for both of them: the rig's height
    over the subject's MEDIAN height, pinned on the character so every frame
    is posed through it. It is a worse measurement than the bone fit — height
    is a single number and a pose can lose it (a crouch measures short) — but
    it is one measurement, so the figure keeps its size, and the preview and
    the exported file keep each other's.

    The take is DE-TILTED here before its height is read. The fit this stands
    in for is rotation-blind — a bone is the same length whichever way the
    room leans — and a height is not, so measuring the poses as they arrive
    would make the answer depend on the caller's frame: the export always
    de-tilts before calling (`take_up` then `de_tilt_matrix`), while the view
    hands over RAW world poses until `main_window` has given it an
    orientation. Two callers, two sizes, one figure. Doing it here means they
    agree by construction rather than by call order; a take that arrives
    already upright is de-tilted by an identity and pays a matrix multiply.

    When there is NO body axis to de-tilt by — too few joints reconstructed
    for `sequence_up` to find a spine — the extent along z is the caller's
    frame all over again, and skipping the de-tilt quietly reinstated the
    very disagreement it removes. So the measure changes instead of being
    dropped: the largest distance between any two reconstructed joints, which
    is the same number whichever way the take is turned and is head-to-foot
    on a standing figure, i.e. the height this is standing in for. A rougher
    measurement again, on a take that already had nothing to measure — but
    one number, and the same one in both callers.
    """
    scale = character.fit_to_subject(poses)
    if scale is not None:
        return float(scale), ""
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    up = sequence_up(poses)
    if up is not None:
        poses = poses @ de_tilt_matrix(up).T
    heights = []
    for pose in poses:
        seen = pose[~np.isnan(pose).any(1)]
        if len(seen) >= 2:
            h = (float(seen[:, 2].max() - seen[:, 2].min()) if up is not None
                 else _widest_span(seen))
            if h > 1e-9:
                heights.append(h)
    if not heights:
        return None, NO_SCALE_NOTE
    scale = float(character.rig_h / np.median(heights))
    # The attribute `fit_to_subject` itself writes. Reached from here because
    # this module is the geometry package's own — the fallback is the same
    # kind of answer as the fit, made where both callers can share it, and
    # `Character` has no other way to be told "this is the take's size".
    character._scale = scale
    return scale, SCALE_FROM_HEIGHT_NOTE

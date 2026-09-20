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

from pose3d.core.skeleton import Joint

#: The share of a take's frames allowed to sit BELOW its floor. The floor is
#: an order statistic over the per-frame seats, not a mean or an interpolated
#: percentile: it is always a height some frame's sole actually reached.
FLOOR_QUANTILE = 0.10


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

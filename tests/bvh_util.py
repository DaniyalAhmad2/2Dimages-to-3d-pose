"""The BVH reader, re-exported for the tests, plus the delivered fixture path.

The parser itself lives in `pose3d.export.bvh`: `tools/check_export_fidelity.py`
is a shipped developer tool and must not import the test package to read an
export. Everything below is that module's public surface, kept here so the
existing `from tests import bvh_util` call sites read unchanged.
"""
from __future__ import annotations

from pathlib import Path

from pose3d.export.bvh import (  # noqa: F401  (re-exported for the tests)
    HOLD, SEGMENT, Bvh, Joint, channel_overshoot, euler_to_quat,
    expected_frames, keyframe_rows, parse, quat_angle_deg, quat_to_matrix,
    similarity, similarity_error, stepped_holds,
)

# The delivered export of the client take (`assets/Imported_Session.bvh`, which
# is gitignored along with the rest of the capture data), stored compressed
# because it is 1.9 MB of text that gzips to 0.2 MB. Refresh with:
#     gzip -9 -c assets/Imported_Session.bvh > tests/fixtures/imported_session.bvh.gz
DELIVERED_BVH = Path(__file__).parent / "fixtures" / "imported_session.bvh.gz"

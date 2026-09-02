"""A minimal BVH reader, so the delivered file can be asserted on directly.

The export path's only tests used to be "Blender exited 0 and wrote a file that
is not empty", which is why nobody noticed that the hips never move and that an
orphan IK helper bone carries 42 % of the rig's height of translation (F13).
Reading the file needs no Blender, so those facts can be asserted on every run.

This parses the subset a Blender BVH export actually emits: one ROOT, nested
JOINTs, End Sites, OFFSET/CHANNELS, then a MOTION block of whitespace-separated
floats. Rotation order is read per joint rather than assumed.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The delivered export of the client take (`assets/Imported_Session.bvh`, which
# is gitignored along with the rest of the capture data), stored compressed
# because it is 1.9 MB of text that gzips to 0.2 MB. Refresh with:
#     gzip -9 -c assets/Imported_Session.bvh > tests/fixtures/imported_session.bvh.gz
DELIVERED_BVH = Path(__file__).parent / "fixtures" / "imported_session.bvh.gz"


@dataclass
class Joint:
    name: str
    offset: np.ndarray
    channels: list[str]
    parent: int
    # index of this joint's first column in the motion matrix
    first_channel: int
    children: list[int] = field(default_factory=list)


@dataclass
class Bvh:
    joints: list[Joint]
    motion: np.ndarray          # (n_frames, n_channels)
    frame_time: float

    @property
    def n_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    def index(self, name: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == name:
                return i
        raise KeyError(name)

    def channel(self, joint: int | str, channel: str) -> int:
        """Column of one channel, e.g. `channel("hips", "Xposition")`."""
        j = self.joints[self.index(joint) if isinstance(joint, str) else joint]
        return j.first_channel + j.channels.index(channel)

    def positions(self, joint: int | str) -> np.ndarray:
        """(n_frames, 3) of a joint's position channels, or an empty array."""
        j = self.joints[self.index(joint) if isinstance(joint, str) else joint]
        names = [f"{a}position" for a in "XYZ"]
        if not all(n in j.channels for n in names):
            return np.zeros((self.n_frames, 0))
        return np.stack([self.motion[:, self.channel(j.name, n)] for n in names], 1)

    def rotations(self, joint: int | str) -> np.ndarray:
        """(n_frames, 3) of a joint's rotation channels IN CHANNEL ORDER."""
        j = self.joints[self.index(joint) if isinstance(joint, str) else joint]
        cols = [j.first_channel + i for i, c in enumerate(j.channels)
                if c.endswith("rotation")]
        if not cols:
            return np.zeros((self.n_frames, 0))
        return self.motion[:, cols]

    def rotation_columns(self) -> list[int]:
        return [j.first_channel + i
                for j in self.joints
                for i, c in enumerate(j.channels) if c.endswith("rotation")]

    def forward_kinematics(self, frame: int) -> np.ndarray:
        """(n_joints, 3) world position of every joint on one motion row.

        BVH semantics: a joint's OFFSET is expressed in its parent's ROTATED
        frame, so the skeleton only takes its real shape once the rotations are
        applied — accumulating offsets alone gives a folded rig, not the rest
        pose.
        """
        pos = np.zeros((len(self.joints), 3))
        rot = [np.eye(3) for _ in self.joints]
        row = self.motion[frame]
        for i, j in enumerate(self.joints):
            base_p = pos[j.parent] if j.parent >= 0 else np.zeros(3)
            base_R = rot[j.parent] if j.parent >= 0 else np.eye(3)
            local = j.offset.copy()
            names = [f"{a}position" for a in "XYZ"]
            if all(n in j.channels for n in names):
                local = np.array([row[self.channel(i, n)] for n in names])
            pos[i] = base_p + base_R @ local
            order = [c for c in j.channels if c.endswith("rotation")]
            if order:
                angles = [row[j.first_channel + j.channels.index(c)] for c in order]
                rot[i] = base_R @ quat_to_matrix(euler_to_quat(angles, order))
            else:
                rot[i] = base_R
        return pos


def _read_text(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return Path(path).read_text()


def parse(path: Path | str) -> Bvh:
    text = _read_text(Path(path))
    head, _, motion_text = text.partition("MOTION")

    joints: list[Joint] = []
    stack: list[int] = []
    n_channels = 0
    pending: str | None = None      # name of the joint whose "{" is next
    for raw in head.splitlines():
        tok = raw.split()
        if not tok:
            continue
        key = tok[0]
        if key in ("ROOT", "JOINT"):
            pending = tok[1]
        elif key == "End":
            pending = None          # End Site: no channels, no motion columns
        elif key == "{":
            if pending is not None:
                joints.append(Joint(name=pending, offset=np.zeros(3),
                                    channels=[],
                                    parent=stack[-1] if stack else -1,
                                    first_channel=0))
                if stack:
                    joints[stack[-1]].children.append(len(joints) - 1)
                stack.append(len(joints) - 1)
            else:
                stack.append(-1)    # End Site block, popped unchanged
        elif key == "}":
            stack.pop()
            pending = None
        elif key == "OFFSET" and stack and stack[-1] >= 0 and pending is not None:
            joints[stack[-1]].offset = np.array([float(v) for v in tok[1:4]])
            pending = None          # consumed by the joint we just opened
        elif key == "CHANNELS" and stack and stack[-1] >= 0:
            j = joints[stack[-1]]
            j.channels = tok[2:2 + int(tok[1])]
            j.first_channel = n_channels
            n_channels += len(j.channels)

    rows, frame_time = [], 0.0
    for raw in motion_text.splitlines():
        tok = raw.split()
        if not tok:
            continue
        if tok[0] == "Frames:":
            continue
        if tok[0] == "Frame":
            frame_time = float(tok[-1])
            continue
        rows.append([float(v) for v in tok])
    return Bvh(joints=joints, motion=np.asarray(rows, float),
               frame_time=frame_time)


# --- the stop-motion schedule ----------------------------------------------
# Mirrors blender_job._stepped_schedule(n, fps, hold_s=0.7, trans_s=0.3) at
# 30 fps: each captured pose is keyed twice, HOLD frames apart, and the next
# pose's first key follows SEGMENT frames after this one's. blender_job cannot
# be imported here — it runs inside Blender and imports bpy at module scope.
SEGMENT = 30
HOLD = 21


def stepped_holds(n_poses: int, segment: int = SEGMENT, hold: int = HOLD):
    """(first_row, last_row) of each captured pose's hold, 0-based."""
    return [(i * segment, i * segment + hold) for i in range(n_poses)]


def expected_frames(n_poses: int, segment: int = SEGMENT,
                    hold: int = HOLD) -> int:
    return 1 + (n_poses - 1) * segment + hold


def channel_overshoot(bvh: Bvh, n_poses: int) -> np.ndarray:
    """How far each eased in-between frame goes past its bracketing keys (deg).

    Per rotation channel, per in-between frame: the distance outside the
    interval the two bracketing keyframe values span. Zero means the ease stays
    between the poses it connects; a large value means the character swings
    past a pose before settling back onto it.
    """
    cols = bvh.rotation_columns()
    holds = stepped_holds(n_poses)
    out = []
    for (_, k0), (k1, _) in zip(holds, holds[1:]):
        lo = np.minimum(bvh.motion[k0, cols], bvh.motion[k1, cols])
        hi = np.maximum(bvh.motion[k0, cols], bvh.motion[k1, cols])
        seg = bvh.motion[k0 + 1:k1][:, cols]
        out.append(np.maximum(0.0, np.maximum(seg - hi, lo - seg)))
    return np.concatenate(out) if out else np.zeros((0, len(cols)))


# --- rotations -------------------------------------------------------------
def _axis_quat(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg) / 2.0
    v = np.zeros(4)
    v[0] = np.cos(a)
    v["XYZ".index(axis) + 1] = np.sin(a)
    return v


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def euler_to_quat(angles_deg, order: list[str]) -> np.ndarray:
    """Quaternion for one joint's rotation channels, in BVH channel order.

    BVH applies its channels left to right as intrinsic rotations, so the
    composed quaternion is q0 * q1 * q2 in channel order.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for ang, chan in zip(angles_deg, order):
        q = _qmul(q, _axis_quat(chan[0], float(ang)))
    return q


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest rotation angle between two unit quaternions, in degrees."""
    d = abs(float(np.dot(a, b)))
    return float(np.degrees(2.0 * np.arccos(min(1.0, d))))

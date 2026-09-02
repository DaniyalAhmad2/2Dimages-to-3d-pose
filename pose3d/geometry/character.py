"""Pose the bundled character to mimic a reconstructed skeleton.

The character's bone lengths are INVIOLABLE. Motion transfers as rotations only,
so the mesh can never be stretched or sheared: the rig is fitted to the capture,
not the other way round. Every bone aims at its own captured joint, so joint
DIRECTIONS follow the capture exactly; two-bone IK is the occlusion fallback,
recovering a limb from its end effector when the elbow or knee is missing.

One uniform scale, fitted once per take, sizes the whole character to the
subject. Being uniform it changes size, never shape.

Rig bone names are reached through ROLES (see `_ROLE_ALIASES`), so swapping in a
differently-named rig is a data change rather than a code change.

Pure numpy, so it runs live in the 3D view; the Blender export drives the same
matrices via `pose_bone_matrices`.
"""
from __future__ import annotations

import contextlib
import json
from collections import namedtuple
from pathlib import Path

import numpy as np

from pose3d.core.skeleton import BONES, Joint, NUM_HEAD_KP, NUM_JOINTS

_ASSET = Path(__file__).parent.parent / "assets" / "character.npz"

# How one frame was placed, read off the solve that posed it.
#   origin       the pelvis this frame was centred on, in capture units
#   scale        the uniform rig-per-capture scale (take-wide once fitted)
#   Rz           the yaw that aligns this frame's shoulder line to the rig
#   root_offset  (origin - pelvis_ref) * scale, in the CAPTURE frame
#   transform    the (4,4) rig-space -> capture-frame map, identity with root
#                motion off (see `Character.export_transform`)
RigAlignment = namedtuple(
    "RigAlignment", "origin scale Rz root_offset transform")


class PoseUnavailable(Exception):
    """This frame's pose cannot drive the rig: it has no usable pelvis.

    Raised (rather than returned as `None`) so a caller can tell "this one
    frame had no hips" apart from "the character asset is broken" — the two
    used to arrive at `View3D._skin` as the same blanket `except Exception`,
    and a missing rig file was reported to the user as an empty 3D view.
    """


_MID = "MID"        # midpoint(pelvis, neck) — the torso split point
_EAR_MID = "EAR_MID"  # midpoint of the ears — a real point on the skull axis

# LEGACY fallback only — used when a frame has no face keypoints (projects
# saved before they existed, or the manual detector) AND the canonical HEAD is
# the NOSE (`head_source == "nose"`, i.e. a COCO-17 project). The nose sits
# ~45 deg forward of the torso line on real captures, so aiming the neck
# straight at it would pitch the head down permanently; _head_aim_target
# rotates the target back by this anatomical offset.
#
# A Halpe-26 project's HEAD is the SKULL VERTEX, which is already on the head's
# axis, so this correction must not be applied to it — that would be a second
# correction on data that needs none. It is not merely redundant there: it only
# ever looked harmless because `max(rest_pitch + theta - _NOSE_PITCH, 0)`
# saturates at 0 for a skull-axis point, which throws the measured pitch away.
_NOSE_PITCH = np.radians(45.0)

# What the canonical HEAD is, when nobody says. "nose" is the legacy answer and
# the truthful default: every project written before `head_source` existed was
# COCO-17. `pose3d.ui.main_window` sets this from the open project, so the
# live 3D view and the Blender export — which both build their own Character
# and never see a ProjectData — stay pose-identical by construction.
HEAD_SOURCES = ("nose", "skull")
_DEFAULT_HEAD_SOURCE = "nose"


def set_default_head_source(source: str | None) -> None:
    """Set the head convention a `Character()` built with no argument uses."""
    global _DEFAULT_HEAD_SOURCE
    source = source or "nose"
    if source not in HEAD_SOURCES:
        raise ValueError(f"head_source must be one of {HEAD_SOURCES}, "
                         f"not {source!r}")
    _DEFAULT_HEAD_SOURCE = source


def default_head_source() -> str:
    """The head convention a `Character()` built with no argument uses."""
    return _DEFAULT_HEAD_SOURCE


@contextlib.contextmanager
def head_source_default(source: str | None):
    """Make `Character()` build with `source` for the duration of the block.

    Setting the process-wide default is the UI's job: `main_window` sets it
    once from the open project, and the 3D view and the Blender export — which
    build their own `Character` and never see a `ProjectData` — follow it. A
    batch tool has no such session, so anything that has to reach a `Character`
    it does not construct itself (`export.blender_export.export_animation`
    builds its own) sets the default only around that call and puts it back:

        with head_source_default(project.head_source):
            export_animation(...)

    Anything that DOES construct the `Character` should pass
    `head_source=project.head_source` instead — explicit beats ambient.
    Restores the previous value even when the block raises, so a tool cannot
    leave a "skull" default behind for whatever runs next in the process.
    """
    previous = _DEFAULT_HEAD_SOURCE
    set_default_head_source(source)
    try:
        yield
    finally:
        set_default_head_source(previous)


# Pipeline role -> candidate bone names, tried in order. Covers the legacy
# Blender meta-rig (what we ship), Rigify, and Mixamo/UE naming so a replacement
# rig usually needs no configuration at all. A rig can also carry an explicit
# {role: bone} map in the npz under "roles", which wins outright.
_ROLE_ALIASES = {
    "hips":        ("hips", "pelvis", "mixamorig:Hips", "Hips"),
    "spine":       ("spine", "spine.001", "mixamorig:Spine", "Spine"),
    "chest":       ("chest", "spine.002", "spine.003", "mixamorig:Spine2", "Spine2"),
    "neck":        ("neck", "spine.004", "mixamorig:Neck", "Neck"),
    "head":        ("head", "spine.006", "mixamorig:Head", "Head"),
    "clavicle.L":  ("shoulder.L", "clavicle.L", "mixamorig:LeftShoulder", "LeftShoulder"),
    "clavicle.R":  ("shoulder.R", "clavicle.R", "mixamorig:RightShoulder", "RightShoulder"),
    "upper_arm.L": ("upper_arm.L", "mixamorig:LeftArm", "LeftArm"),
    "upper_arm.R": ("upper_arm.R", "mixamorig:RightArm", "RightArm"),
    "forearm.L":   ("forearm.L", "mixamorig:LeftForeArm", "LeftForeArm"),
    "forearm.R":   ("forearm.R", "mixamorig:RightForeArm", "RightForeArm"),
    "hand.L":      ("hand.L", "mixamorig:LeftHand", "LeftHand"),
    "hand.R":      ("hand.R", "mixamorig:RightHand", "RightHand"),
    "thigh.L":     ("thigh.L", "mixamorig:LeftUpLeg", "LeftUpLeg"),
    "thigh.R":     ("thigh.R", "mixamorig:RightUpLeg", "RightUpLeg"),
    "shin.L":      ("shin.L", "shin.L", "mixamorig:LeftLeg", "LeftLeg"),
    "shin.R":      ("shin.R", "mixamorig:RightLeg", "RightLeg"),
    "foot.L":      ("foot.L", "mixamorig:LeftFoot", "LeftFoot"),
    "foot.R":      ("foot.R", "mixamorig:RightFoot", "RightFoot"),
}

# Role -> (start, end) canonical joints. Only `end` drives the pose: the head is
# carried by the parent (FK) and the bone rotates to aim at `end`. `start` is
# used when measuring the subject for the uniform scale fit.
_DIRECT = {
    "spine": (Joint.PELVIS, _MID),
    "chest": (_MID, Joint.NECK),
    # aimed at the EAR MIDPOINT, which sits on the skull axis — so unlike the
    # nose it needs no anatomical fudge factor. The head bone on top of it gets
    # a full rotation from the face keypoints (see _head_basis).
    "neck": (Joint.NECK, _EAR_MID),
    # the clavicles carry the arms; drive them or the arm roots miss the
    # shoulders and a noisy shoulder swings the whole arm
    "clavicle.L": (Joint.NECK, Joint.LEFT_SHOULDER),
    "clavicle.R": (Joint.NECK, Joint.RIGHT_SHOULDER),
    "upper_arm.L": (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    "forearm.L": (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    "upper_arm.R": (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    "forearm.R": (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    "thigh.L": (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    "shin.L": (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    "thigh.R": (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    "shin.R": (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
}

# Limb chains: (upper role, lower role, mid joint, end joint). When the mid
# joint is captured, both bones aim at their own joints via _DIRECT so the
# bend follows the capture; two-bone IK runs only when the mid joint is
# occluded, recovering the chain from the end effector alone.
_IK_CHAINS = (
    ("upper_arm.L", "forearm.L", Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    ("upper_arm.R", "forearm.R", Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    ("thigh.L", "shin.L", Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    ("thigh.R", "shin.R", Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
)

# --- roll references -------------------------------------------------------
# A bone's aim fixes two of its three rotational degrees of freedom; the third
# — its spin ABOUT that aim — used to be whatever `_align`'s minimal rotation
# happened to give, which is a numerical accident rather than a measurement.
# These tables say what each bone's roll is measured against instead.
#
# For `upper_arm` and `thigh` the roll IS a measurement: the elbow and the knee
# are hinges, so the plane the limb bends in genuinely fixes the parent bone's
# spin. For `forearm` and `shin` it is a CONVENTION — forearm pronation and
# shin twist are never observed (there are no hand or foot keypoints), so those
# bones simply carry the hinge plane on. A future reader must not read a small
# roll error on a forearm as accuracy.
_SHOULDER_LINE = (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)
_HIP_LINE = (Joint.LEFT_HIP, Joint.RIGHT_HIP)

# role -> (root, mid, end, torso line). cross(mid - root, end - mid) is the
# limb's bend-plane normal; the torso line pins it to a repeatable hemisphere
# (a cross product's sign does not follow from joint order alone).
_BEND_REF = {
    "upper_arm.L": (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW,
                    Joint.LEFT_WRIST, _SHOULDER_LINE),
    "forearm.L":   (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW,
                    Joint.LEFT_WRIST, _SHOULDER_LINE),
    "upper_arm.R": (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW,
                    Joint.RIGHT_WRIST, _SHOULDER_LINE),
    "forearm.R":   (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW,
                    Joint.RIGHT_WRIST, _SHOULDER_LINE),
    "thigh.L":     (Joint.LEFT_HIP, Joint.LEFT_KNEE,
                    Joint.LEFT_ANKLE, _HIP_LINE),
    "shin.L":      (Joint.LEFT_HIP, Joint.LEFT_KNEE,
                    Joint.LEFT_ANKLE, _HIP_LINE),
    "thigh.R":     (Joint.RIGHT_HIP, Joint.RIGHT_KNEE,
                    Joint.RIGHT_ANKLE, _HIP_LINE),
    "shin.R":      (Joint.RIGHT_HIP, Joint.RIGHT_KNEE,
                    Joint.RIGHT_ANKLE, _HIP_LINE),
}

# role -> the captured line the bone's roll follows directly. The pelvis and
# the spine follow the HIP line, the chest the SHOULDER line: that is what
# makes the character's hips face where the subject's do instead of inheriting
# whatever spin the torso aim left behind.
#
# `neck`/`head` get no reference (the face basis already gives the head a full
# measured orientation, and a roll would double-drive it); `clavicle.*`,
# `hand.*` and `foot.*` get none either — they have no keypoints of their own,
# so once the parent's roll is right theirs is inherited right.
_LINE_REF = {"hips": _HIP_LINE, "spine": _HIP_LINE, "chest": _SHOULDER_LINE}

# Straight-limb fallback. Near full extension the bend plane is undefined, so
# the roll fades out CONTINUOUSLY instead of switching off at a threshold: the
# left elbow's minimum bend on the client take is 22.7 deg — above any sane
# guard, so a guard would never fire — while its bend normal still jumps
# 55.7 deg between neighbouring frames there. At 22.7 deg this ramp gives
# w = 0.14, which is what suppresses the pop.
_ROLL_BEND_MIN_DEG = 20.0       # below this the roll term vanishes
_ROLL_BEND_FULL_DEG = 40.0      # at and above this it applies in full
# Rollback switch: every roll angle is multiplied by this, so 0.0 restores the
# pre-roll minimal-rotation matrices bit for bit — EXCEPT at `_align`'s
# antipodal (c < -0.999999) branch, where the rest references still exist and
# still pick the spin axis. That is the F43 fix and it is deliberately kept:
# the old branch flipped a bone's roll by 163.8 deg for a 1 deg wobble, so
# "bit for bit" holds everywhere the rollback is a rollback and the one place
# it differs is the place the old answer was arbitrary. See
# tests/test_retarget.py::test_roll_weight_zero_restores_the_minimal_rotation.
_ROLL_WEIGHT = 1.0


# Where each canonical joint is read off the posed rig. Heads are preferred:
# a bone's head is the exact FK position carried by its parent, so the reported
# skeleton is guaranteed consistent with the mesh. Fallbacks are used when the
# rig lacks the bone (e.g. no hand bone -> the forearm's tail is the wrist).
# HEAD is deliberately absent: what it is read back from depends on what the
# capture's HEAD point IS, so it lives in _HEAD_FROM_RIG and is filled in per
# head_source by _resolve_joint_sources.
_JOINT_FROM_RIG = {
    Joint.PELVIS: (("hips", "head"),),
    Joint.NECK: (("neck", "head"), ("chest", "tail")),
    Joint.LEFT_SHOULDER: (("upper_arm.L", "head"),),
    Joint.RIGHT_SHOULDER: (("upper_arm.R", "head"),),
    Joint.LEFT_ELBOW: (("forearm.L", "head"), ("upper_arm.L", "tail")),
    Joint.RIGHT_ELBOW: (("forearm.R", "head"), ("upper_arm.R", "tail")),
    Joint.LEFT_WRIST: (("hand.L", "head"), ("forearm.L", "tail")),
    Joint.RIGHT_WRIST: (("hand.R", "head"), ("forearm.R", "tail")),
    Joint.LEFT_HIP: (("thigh.L", "head"),),
    Joint.RIGHT_HIP: (("thigh.R", "head"),),
    Joint.LEFT_KNEE: (("shin.L", "head"), ("thigh.L", "tail")),
    Joint.RIGHT_KNEE: (("shin.R", "head"), ("thigh.R", "tail")),
    Joint.LEFT_ANKLE: (("foot.L", "head"), ("shin.L", "tail")),
    Joint.RIGHT_ANKLE: (("foot.R", "head"), ("shin.R", "tail")),
}

# Where HEAD is read back from, per head_source. A nose HEAD has no counterpart
# on the rig at all (the rig has no face), so the head bone's MIDPOINT — inside
# the skull, roughly the centre of the head — is the honest stand-in. A skull
# HEAD is Halpe's head point, at the top of the skull, which is exactly where
# the head bone's TAIL sits; reading the midpoint for it would report the head
# half a bone short of where the capture says it is. This is not only a
# readout: `rest_joints` feeds the uniform scale fit and the rest references,
# so it must name the same anatomy the capture does.
#
# Measured on the client take rather than assumed, and deliberately NOT on the
# head-position metric alone (F03's refuter showed that one misleads — its
# variant improved the head metric while tripling the visible nose-to-mesh
# distance). Skull HEAD, mid -> tail: HEAD retarget 3.42 -> 1.54 % of height
# with no face points and 3.93 -> 1.67 % with them; head aim error 4.95 ->
# 3.83 deg max (1.22 -> 1.19 median); whole-body retarget median 1.57 ->
# 1.49 %; and the refuter's own metric flat — reconstructed nose to nearest
# mesh vertex 1.24 -> 1.20 mm, ear 1.22 -> 1.19 mm.
# `tools/measure_head_gates.py --variants` re-derives all of those.
_HEAD_FROM_RIG = {
    "nose": (("head", "mid"),),
    "skull": (("head", "tail"),),
}

# Legs weigh double when fitting the uniform scale: feet not reaching the floor
# is the mismatch the eye picks up first.
_SCALE_WEIGHTS = {Joint.LEFT_KNEE: 2.0, Joint.RIGHT_KNEE: 2.0,
                  Joint.LEFT_ANKLE: 2.0, Joint.RIGHT_ANKLE: 2.0}


def _align(a, b, ref=None):
    """3x3 rotation taking unit vector a to unit vector b.

    At (near) 180 deg the minimal rotation is undefined — every axis
    perpendicular to `a` maps it onto `b` — and the arbitrary pick used to flip
    the bone's roll by 163.8 deg for a 1 deg wobble. Given `ref`, the bone's
    rest roll reference, spin about the axis that defines instead: it is the
    same choice on either side of 180 deg, so the branch stops deciding
    anything.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.999999:
        axis = _unit(np.cross(a, ref)) if ref is not None else None
        if axis is None:
            axis = _perp(a)
        return 2 * np.outer(axis, axis) - np.eye(3)  # 180° about axis
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _perp(u):
    """Any unit vector perpendicular to u."""
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(u, a)
    return v / (np.linalg.norm(v) + 1e-12)


def _unit(v):
    """v normalised, or None if it is too short (or not finite) to have a
    direction — every roll reference is a direction, so this is the one place
    that decides a reference is unusable."""
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 and np.isfinite(n) else None


def _proj_perp(v, axis):
    """Unit component of v perpendicular to the unit vector `axis`, or None
    when v is (near) parallel to it and carries no roll information."""
    w = v - float(np.dot(v, axis)) * axis
    n = float(np.linalg.norm(w))
    return w / n if n > 1e-6 else None


def _rot(axis, ang):
    """Rotation of `ang` radians about the unit vector `axis` (Rodrigues)."""
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _bend_weight(bend_deg):
    """How much roll authority a limb bent `bend_deg` off straight has."""
    return float(np.clip(
        (bend_deg - _ROLL_BEND_MIN_DEG)
        / (_ROLL_BEND_FULL_DEG - _ROLL_BEND_MIN_DEG), 0.0, 1.0))


def root_offset(origin, scale, pelvis_ref):
    """The subject's travel away from the take's reference, in rig units.

    In the CAPTURE's own frame — deliberately not turned by the retarget's
    per-frame `Rz`. Round 1 wrote `Rz @ (pelvis - pelvis_ref) * scale`, which
    expresses the travel in a frame that rotates with the subject: the
    direction the exported hips moved then depended on which way the subject
    happened to be facing that frame.
    """
    return ((np.asarray(origin, float) - np.asarray(pelvis_ref, float))
            * float(scale))


def take_pelvis_ref(poses):
    """The take's pelvis: one reference point for placing the whole sequence.

    Resolved per frame exactly as `Character._skin_matrices` does — the PELVIS
    joint when it is there, the hip midpoint when it is not — and then averaged
    over the frames that have one. Returns None if no frame does.

    The MEAN, not the first frame: the first frame can be the one whose pelvis
    dropped out, and averaging leaves the subject's travel spread either side
    of the origin, so the 3D view's grid and camera (sized from the same pelvis
    box) frame the whole take rather than its opening pose. It is a property of
    the take, computed once, which is what makes it usable as the SHARED
    placement rule: the view subtracts it, the export offsets the hips by the
    travel away from it, and the two agree frame for frame.
    """
    poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
    out = []
    for pose in poses:
        p = pose[int(Joint.PELVIS)]
        if not np.isfinite(p).all():
            hips = [pose[int(j)] for j in (Joint.LEFT_HIP, Joint.RIGHT_HIP)
                    if np.isfinite(pose[int(j)]).all()]
            if not hips:
                continue
            p = np.mean(hips, axis=0)
        out.append(p)
    return np.mean(out, axis=0) if out else None


class Character:
    """The bundled rig, posed to mimic a reconstructed skeleton.

    `head_source` says what the capture's canonical HEAD point is — "nose"
    (COCO-17) or "skull" (Halpe-26's head point, on the skull axis). It
    changes two things and nothing else: where HEAD is read back from on the
    rig, and whether the legacy no-face-keypoints neck aim applies the nose's
    anatomical offset. Defaults to the process-wide `default_head_source()`,
    which is "nose" unless the application has said otherwise.
    """

    def __init__(self, path=_ASSET, head_source: str | None = None):
        self.head_source = head_source or default_head_source()
        if self.head_source not in HEAD_SOURCES:
            raise ValueError(f"head_source must be one of {HEAD_SOURCES}, "
                             f"not {self.head_source!r}")
        d = np.load(path, allow_pickle=True)
        self.verts0 = d["verts"].astype(float)          # (V,3) rest, world
        self.faces = d["faces"].astype(np.int32)
        self.w_idx = d["w_idx"]; self.w_val = d["w_val"].astype(float)
        self.rest = d["rest_mat"].astype(float)          # (B,4,4) world
        self.head = d["head"].astype(float)
        self.tail = d["tail"].astype(float)
        self.parent = d["parent"].astype(int)
        names = [str(n) for n in d["bone_names"]]
        self.bone_names = names
        self.bidx = {n: i for i, n in enumerate(names)}
        self.source = str(d["source"]) if "source" in d else ""

        self.role = self._resolve_roles(d)
        for r in ("hips", "upper_arm.L", "upper_arm.R"):
            if r not in self.role:
                raise ValueError(f"rig is missing the '{r}' role; got {names[:8]}…")
        self.hips_idx = self.role["hips"]

        # Helper bones with no parent (leg IK targets like "shin.L.001") would
        # otherwise keep their rest transform while the body moves and float
        # away from it. Adopt them onto the bone they are named after, and pose
        # every bone through this EFFECTIVE parent so their children follow too
        # (a child of an orphan used to stay frozen at the origin).
        self._eparent = self.parent.copy()
        for b, name in enumerate(names):
            if b == self.hips_idx or self.parent[b] >= 0:
                continue
            base = name.rsplit(".", 1)[0] if name.rsplit(".", 1)[-1].isdigit() else None
            self._eparent[b] = self.bidx.get(base, self.hips_idx) if base else self.hips_idx
        self.order = self._topo()

        self.vh = np.hstack([self.verts0, np.ones((len(self.verts0), 1))])
        self.rig_h = float(self.head[:, 2].max() - self.head[:, 2].min()) or 1.0
        self.hips_world = self.head[self.hips_idx]
        lh = self.head[self.role["upper_arm.L"]]
        rh = self.head[self.role["upper_arm.R"]]
        self.rig_right = np.array([rh[0] - lh[0], rh[1] - lh[1]])
        self.rig_right /= (np.linalg.norm(self.rig_right) + 1e-12)

        self._direct = {self.role[r]: se for r, se in _DIRECT.items()
                        if r in self.role}
        # {upper bone: (lower bone, mid joint, end joint)}
        self._ik = {}
        for up, lo, mid_j, end_j in _IK_CHAINS:
            if up in self.role and lo in self.role:
                self._ik[self.role[up]] = (self.role[lo], mid_j, end_j)
        self._rest_pole = self._compute_rest_poles()
        self._joint_src = self._resolve_joint_sources()
        # The pelvis is aimed at the torso midpoint. Measure that from the rig's
        # OWN rest torso direction, not from the hips bone's axis: the two do not
        # coincide, so using the bone axis rotates the pelvis (and both legs with
        # it) even when the subject matches the rig exactly.
        rj = self.rest_joints()
        mid = (rj[int(Joint.NECK)] + rj[int(Joint.PELVIS)]) / 2.0
        d = mid - rj[int(Joint.PELVIS)]
        n = np.linalg.norm(d)
        self._rest_torso = (d / n if n > 1e-9 and not np.isnan(d).any()
                            else self.tail[self.hips_idx] - self.head[self.hips_idx])
        self._rest_ref = self._rest_roll_refs(rj)
        # The rig's rest ankle-to-sole height, which is what the 3D view drops
        # the ankle by to stand the character on the ground. 0.71217 rig units
        # on the bundled rig = 4.94 % of its 14.4228 height.
        self.ankle_sole_drop = self._rest_ankle_sole_drop(rj)
        self._rest_head = self._rest_head_basis()
        # The neck BONE's rest angle off the torso line, for the legacy
        # no-face-keypoints fallback: _bone_fk aims the bone axis at the
        # target, so this is the angle that leaves the neck at rest for a
        # subject holding the anatomical neutral (_NOSE_PITCH).
        self._rest_head_pitch = 0.0
        if "neck" in self.role:
            nb = self.role["neck"]
            axis = self.tail[nb] - self.head[nb]
            td = rj[int(Joint.NECK)] - rj[int(Joint.PELVIS)]
            an, tn = np.linalg.norm(axis), np.linalg.norm(td)
            if an > 1e-9 and tn > 1e-9 and np.isfinite(axis).all() \
                    and np.isfinite(td).all():
                cosr = float(np.clip(np.dot(axis / an, td / tn), -1.0, 1.0))
                self._rest_head_pitch = float(np.arccos(cosr))
        self._scale = None          # uniform scale, set by fit_to_subject

    # --- rig introspection -------------------------------------------------
    def _resolve_roles(self, d) -> dict[str, int]:
        """Map pipeline roles to bone indices, preferring an explicit map baked
        into the asset and falling back to the alias table."""
        explicit = {}
        if "roles" in d:
            try:
                explicit = json.loads(str(d["roles"]))
            except (ValueError, TypeError):
                explicit = {}
        out = {}
        for role, aliases in _ROLE_ALIASES.items():
            name = explicit.get(role)
            if name in self.bidx:
                out[role] = self.bidx[name]
                continue
            for cand in aliases:
                if cand in self.bidx:
                    out[role] = self.bidx[cand]
                    break
        return out

    def _topo(self):
        """Bone order with effective parents strictly before their children."""
        order, seen = [], set()

        def visit(b):
            if b in seen or b < 0:
                return
            if self._eparent[b] >= 0:
                visit(int(self._eparent[b]))
            seen.add(b); order.append(b)

        for b in range(len(self.parent)):
            visit(b)
        return order

    def _compute_rest_poles(self):
        """Unit bend direction of each IK limb in the REST pose.

        Used when the captured mid-joint can't say which way the limb bends;
        the rig's own rest bend is anatomically correct by construction.
        """
        poles = {}
        for ub, (lb, _, _) in self._ik.items():
            root, knee, tip = self.head[ub], self.head[lb], self.tail[lb]
            axis = tip - root
            n = np.linalg.norm(axis)
            if n < 1e-9:
                poles[ub] = _perp(np.array([0, 0, 1.0])); continue
            u = axis / n
            v = (knee - root) - np.dot(knee - root, u) * u
            nv = np.linalg.norm(v)
            poles[ub] = v / nv if nv > 1e-9 else _perp(u)
        return poles

    @staticmethod
    def _hemisphere(n, line):
        """`n` as a unit vector on `line`'s side, or None if either is unusable.

        The sign of a bend normal does not follow from joint order alone, so it
        is pinned to the torso's medial-lateral axis — where a limb hinge's
        normal anatomically lies. Doing that from the torso of the SAME frame,
        rather than from the previous frame's answer, is what keeps posing a
        pure function of one frame and so keeps the 3D view and the Blender
        export identical for free.
        """
        n = _unit(n)
        line = _unit(line) if line is not None else None
        if n is None or line is None:
            return None
        return -n if float(np.dot(n, line)) < 0.0 else n

    def _rest_roll_refs(self, rj):
        """{bone: unit roll reference} measured on the rig at REST.

        Built exactly as the captured references in `_roll_targets` are,
        hemisphere fix included, so the two live in the same half-space and the
        angle between them is the bone's roll error. A bone whose reference is
        degenerate at rest (a limb the rig holds straight) simply gets none and
        keeps the minimal rotation.
        """
        out = {}
        for role, (ja, jm, jb, line) in _BEND_REF.items():
            b = self.role.get(role)
            if b is None:
                continue
            n = self._hemisphere(
                np.cross(rj[int(jm)] - rj[int(ja)], rj[int(jb)] - rj[int(jm)]),
                rj[int(line[1])] - rj[int(line[0])])
            if n is not None:
                out[b] = n
        for role, (jl, jr) in _LINE_REF.items():
            b = self.role.get(role)
            if b is None:
                continue
            d = _unit(rj[int(jr)] - rj[int(jl)])
            if d is not None:
                out[b] = d
        return out

    def _rest_ankle_sole_drop(self, rj):
        """How far the rig's rest ankles sit above its soles, in rig units.

        ASSUMES the rest mesh's lowest vertex IS a sole — true of the bundled
        rig (its minimum is a left-foot vertex at x = +2.010 against a left
        ankle at x = +2.196, giving 0.7121720 rig units = 4.94 % of the
        14.4228 rig height) and of any rig modelled standing. On a rig posed
        at rest with its arms hanging below its feet this would silently
        become a fingertip datum, which would put the figure below the grid by
        that error; a replacement rig wants checking here rather than a
        different formula, since restricting the minimum to foot-weighted
        vertices needs a skin-weight lookup this class does not otherwise do.
        """
        z = [rj[int(j)][2] for j in (Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)]
        z = [q for q in z if np.isfinite(q)]
        return float(min(z) - self.verts0[:, 2].min()) if z else 0.0

    def _resolve_joint_sources(self):
        """{canonical joint: (bone index, 'head'|'tail'|'mid')} for this rig."""
        out = {}
        sources = dict(_JOINT_FROM_RIG)
        # the one joint whose rig point depends on what the capture's HEAD is
        sources[Joint.HEAD] = _HEAD_FROM_RIG[self.head_source]
        for j, cands in sources.items():
            for role, which in cands:
                if role in self.role:
                    out[int(j)] = (self.role[role], which)
                    break
        return out

    def _bone_point(self, b, which):
        if which == "head":
            return self.head[b]
        if which == "tail":
            return self.tail[b]
        return (self.head[b] + self.tail[b]) / 2.0

    def _joints_from_skin(self, skin):
        """Canonical joint positions read off a posed rig, in RIG space."""
        out = np.full((NUM_JOINTS, 3), np.nan)
        for j, (b, which) in self._joint_src.items():
            p = self._bone_point(b, which)
            out[j] = skin[b][:3, :3] @ p + skin[b][:3, 3]
        return out

    def rest_joints(self):
        """Canonical joint positions of the UNPOSED rig, in rig space."""
        return self._joints_from_skin(np.tile(np.eye(4), (len(self.rest), 1, 1)))

    def rig_bone_lengths(self) -> dict[tuple[int, int], float]:
        """Length of each canonical skeleton bone as built into the rig."""
        rj = self.rest_joints()
        out = {}
        for a, b in BONES:
            pa, pb = rj[int(a)], rj[int(b)]
            if not (np.isnan(pa).any() or np.isnan(pb).any()):
                out[(int(a), int(b))] = float(np.linalg.norm(pb - pa))
        return out

    # --- fitting -----------------------------------------------------------
    def fit_to_subject(self, poses) -> float | None:
        """Size the character to the subject once, for a whole take.

        A single UNIFORM scale, so the character changes size but never shape.
        It is the least-squares best match between the subject's median bone
        lengths and the rig's own, weighted toward the legs. Fitting once per
        take also stops the character pulsing: the scale used to be recomputed
        per frame from whichever joints were visible, so it jumped whenever the
        ankles dropped out.

        `poses` are upright (de-tilted) poses in the same space as `pose()`.
        """
        from pose3d.geometry.bonefit import measure_bone_lengths

        poses = np.asarray(poses, float).reshape(-1, NUM_JOINTS, 3)
        sub = measure_bone_lengths(poses)
        rig = self.rig_bone_lengths()
        num = den = 0.0
        for key, rig_len in rig.items():
            sub_len = sub.get(key)
            if not sub_len or not np.isfinite(sub_len) or sub_len < 1e-9:
                continue
            w = max(_SCALE_WEIGHTS.get(key[0], 1.0), _SCALE_WEIGHTS.get(key[1], 1.0))
            num += w * sub_len * rig_len
            den += w * sub_len * sub_len
        self._scale = float(num / den) if den > 1e-12 else None
        return self._scale

    def reset_fit(self):
        self._scale = None

    # --- posing ------------------------------------------------------------
    def _head_aim_target(self, J, pelvis):
        """Neck target when a frame has no face keypoints.

        With a SKULL head_source the captured HEAD is already on the head's
        axis, so it IS the target: the neck aims straight at it, with no
        anatomical offset and no clamp. That is the whole of the head fix on
        the no-face-keypoints path.

        With a NOSE head_source (COCO-17, and every project written before
        head_source existed) the target is the nose, whose direction off the
        torso line is shifted so the anatomical neutral (_NOSE_PITCH) lands on
        the rig's own rest head pitch, clamped at the torso line (with one
        head point, "looking up" cannot be told apart from skull-convention
        data, so it saturates at neutral). Computed in pose space — angles
        survive the rigid + uniform-scale to_rig map. This is exactly the
        behaviour of the last build before face keypoints existed, so old
        projects and the manual detector are unchanged, bit for bit.
        """
        h, n = J(Joint.HEAD), J(Joint.NECK)
        if self.head_source == "skull":
            return h                       # already on the skull axis
        if h is None or n is None or pelvis is None \
                or not np.isfinite(np.asarray(pelvis)).all():
            return None                    # nothing to correct against: inherit
        d = h - n
        t = n - pelvis
        dn, tn = float(np.linalg.norm(d)), float(np.linalg.norm(t))
        if dn < 1e-9 or tn < 1e-9:
            return None
        t_hat = t / tn
        cos_th = float(np.clip(np.dot(d / dn, t_hat), -1.0, 1.0))
        theta = float(np.arccos(cos_th))
        want = max(self._rest_head_pitch + theta - _NOSE_PITCH, 0.0)
        perp = d / dn - cos_th * t_hat     # the plane the nod happens in
        pn = float(np.linalg.norm(perp))
        if pn < 1e-9:
            return n + dn * t_hat          # collinear: corrected angle is 0
        dir_c = np.cos(want) * t_hat + np.sin(want) * (perp / pn)
        return n + dn * dir_c

    def _head_basis(self, head_rig):
        """3x3 orientation of the head from the face keypoints, or None.

        Columns are (right, forward, up) as unit vectors. A single nose point
        cannot carry orientation — it only says which way the face is, not how
        the head is turned about that — so the lateral axis comes from the
        EARS (or the eyes when an ear is missing), which are detected at least
        as reliably. `head_rig` is (NUM_HEAD_KP, 3) already in rig space.
        """
        if head_rig is None:
            return None
        pts = np.asarray(head_rig, float).reshape(-1, 3)
        if len(pts) < NUM_HEAD_KP:
            return None
        nose, eye_l, eye_r, ear_l, ear_r = pts[:5]

        # Lateral axis: ears preferred, eyes as the fallback pair. Points to
        # the subject's RIGHT, matching _rest_head_basis's shoulder line — get
        # this backwards and R becomes a 180 deg flip that buries the head in
        # the neck.
        for left, right_kp in ((ear_l, ear_r), (eye_l, eye_r)):
            if not (np.isnan(left).any() or np.isnan(right_kp).any()):
                right = right_kp - left
                lateral_mid = (left + right_kp) / 2.0
                break
        else:
            return None
        n_r = np.linalg.norm(right)
        if n_r < 1e-9 or np.isnan(nose).any():
            return None
        right = right / n_r

        fwd = nose - lateral_mid                 # where the face points
        fwd = fwd - np.dot(fwd, right) * right   # orthogonalise against it
        n_f = np.linalg.norm(fwd)
        if n_f < 1e-9:
            return None                          # nose on the ear line: degenerate
        fwd = fwd / n_f
        up = np.cross(right, fwd)
        n_u = np.linalg.norm(up)
        if n_u < 1e-9:
            return None
        return np.column_stack([right, fwd, up / n_u])

    def _rest_head_basis(self):
        """The same (right, forward, up) basis, measured on the rig at rest.

        The rig has no ears, so its axes come from geometry that means the same
        thing: the shoulder line is `right`, the head bone's own axis is `up`.
        """
        rj = self.rest_joints()
        b = self.role.get("head")
        if b is None:
            return None
        up = self.tail[b] - self.head[b]
        right = rj[int(Joint.RIGHT_SHOULDER)] - rj[int(Joint.LEFT_SHOULDER)]
        if np.isnan(right).any():
            return None
        n_u, n_r = np.linalg.norm(up), np.linalg.norm(right)
        if n_u < 1e-9 or n_r < 1e-9:
            return None
        up = up / n_u
        right = right - np.dot(right / n_r, up) * up
        n_r = np.linalg.norm(right)
        if n_r < 1e-9:
            return None
        right = right / n_r
        return np.column_stack([right, np.cross(up, right), up])

    def _bone_rot(self, b, base, R):
        """Skin matrix giving bone b the world orientation `R`.

        Head pinned to the parent exactly as `_bone_fk` does, so the chain
        cannot separate; only the rotation source differs — measured here,
        aimed there.
        """
        head = base[:3, :3] @ self.head[b] + base[:3, 3]
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = head - R @ self.head[b]
        return M

    def _roll(self, b, R, aim, ref_target, weight):
        """Spin `R` about `aim` so bone b's carried rest reference meets the
        captured one, by `weight` of the way.

        This is a PURE roll: the bone still aims exactly where it did, and
        because each limb child's head sits ON its parent's axis, rolling a
        limb bone moves no canonical joint at all. `weight` fades the term out
        as a limb straightens and its bend plane stops being defined.
        """
        r0 = self._rest_ref.get(b)
        if r0 is None or ref_target is None or weight <= 0.0:
            return R
        cur = _proj_perp(R @ r0, aim)
        tgt = _proj_perp(ref_target, aim)
        if cur is None or tgt is None:
            return R                    # a reference parallel to the aim
        ang = np.arctan2(float(np.dot(np.cross(cur, tgt), aim)),
                         float(np.dot(cur, tgt)))
        return _rot(aim, weight * ang) @ R

    def _roll_targets(self, up_pose, valid, Rz):
        """{bone: (captured roll reference, weight)} for one frame, RIG space.

        References are directions, so `to_rig`'s uniform scale and translation
        are irrelevant and only the yaw alignment `Rz` is applied.

        KNOWN, ACCEPTED DISCONTINUITY: a limb's bend plane needs all three of
        its joints, so a wrist (or ankle) that drops to NaN for a single frame
        removes the parent's roll term for that frame and the bone snaps back
        to the minimal rotation — up to ~50 deg on the client take. It is an
        on/off gate where `_bend_weight` is a ramp, and that is deliberate:
        fading it in over neighbouring frames would need caller-owned temporal
        state, which desyncs the 3D view (scrubbed in any order) from the
        Blender export (iterated once, in order) and is forbidden outright.
        The stateless fallback is weight 0 — today's minimal rotation, no
        exception, view and export agreeing on it — and Phase 1's flagged
        single-frame gap fill makes the gap rare. Pinned by
        tests/test_retarget.py::test_a_missing_wrist_falls_back_to_weight_zero.
        """
        def line(pair):
            jl, jr = pair
            if not (valid[int(jl)] and valid[int(jr)]):
                return None
            return _unit(Rz @ (up_pose[int(jr)] - up_pose[int(jl)]))

        out = {}
        if _ROLL_WEIGHT <= 0.0:
            return out                  # the rollback switch, thrown
        for role, (ja, jm, jb, pair) in _BEND_REF.items():
            b = self.role.get(role)
            if b is None or b not in self._rest_ref:
                continue
            if not all(valid[int(x)] for x in (ja, jm, jb)):
                continue
            v1 = _unit(up_pose[int(jm)] - up_pose[int(ja)])
            v2 = _unit(up_pose[int(jb)] - up_pose[int(jm)])
            if v1 is None or v2 is None:
                continue
            bend = np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)), -1.0, 1.0)))
            w = _ROLL_WEIGHT * _bend_weight(bend)
            if w <= 0.0:
                continue                # straight limb: no bend plane to follow
            n = self._hemisphere(Rz @ np.cross(v1, v2), line(pair))
            if n is not None:
                out[b] = (n, w)
        # The torso lines get full weight. The hip line is only a 16.3 mm
        # segment, but at this take's own 2-3 px noise a Monte-Carlo moves it by
        # just 2.3-3.4 deg against an 11.8 deg error, and its frame-to-frame
        # coherence matches the shoulder line's: signal, not jitter.
        for role, pair in _LINE_REF.items():
            b = self.role.get(role)
            if b is None or b not in self._rest_ref:
                continue
            d = line(pair)
            if d is not None:
                out[b] = (d, _ROLL_WEIGHT)
        return out

    def _bone_fk(self, b, base, end, *, ref_target=None, weight=1.0):
        """Skin matrix for bone b, hanging off its already-posed parent.

        The head is carried by the parent, so joined bones can never separate,
        and the bone then rotates about that head to aim at `end`. Rotation
        only — the bone keeps its rest length, so the mesh cannot deform. The
        aim leaves the bone's spin about its own axis free; `ref_target` (in
        rig space) is what it is set from, instead of the numerical accident
        the minimal rotation happens to give.
        """
        head = base[:3, :3] @ self.head[b] + base[:3, 3]
        rest_d = self.tail[b] - self.head[b]
        n_rest = float(np.linalg.norm(rest_d))
        d_want = end - head
        n_want = float(np.linalg.norm(d_want))
        if n_rest < 1e-9 or n_want < 1e-9:
            return base
        aim = d_want / n_want
        R = _align(rest_d / n_rest, aim, ref=self._rest_ref.get(b))
        R = self._roll(b, R, aim, ref_target, weight)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = head - R @ self.head[b]      # pin the head in place
        return M

    def _pole(self, ub, base, u):
        """Unit vector, perpendicular to u, giving the limb's bend direction.

        The rig's own rest bend is the only source available: IK now runs
        exclusively when the captured mid joint is MISSING (see
        `_skin_matrices`), so there is nothing else to aim the bend at.
        """
        rest = base[:3, :3] @ self._rest_pole[ub]
        rest = rest - np.dot(rest, u) * u
        n_rest = np.linalg.norm(rest)
        return rest / n_rest if n_rest > 1e-9 else _perp(u)

    def _solve_ik(self, ub, lb, base, end):
        """Two-bone IK with FIXED bone lengths — the OCCLUSION path.

        Returns (mid_target, end_target) to feed straight through `_bone_fk`:
        both are exactly one bone length from their respective roots, so the
        FK pass reproduces this solution and keeps the chain connected. When
        the target is out of reach the limb extends fully toward it, which is
        the closest a fixed-length limb can get.
        """
        root = base[:3, :3] @ self.head[ub] + base[:3, 3]
        L1 = float(np.linalg.norm(self.tail[ub] - self.head[ub]))
        L2 = float(np.linalg.norm(self.tail[lb] - self.head[lb]))
        d_vec = end - root
        d = float(np.linalg.norm(d_vec))
        if d < 1e-9 or L1 < 1e-9 or L2 < 1e-9:
            return None
        u = d_vec / d
        dc = float(np.clip(d, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6))
        w = self._pole(ub, base, u)
        cos_a = float(np.clip((dc * dc + L1 * L1 - L2 * L2) / (2.0 * dc * L1), -1.0, 1.0))
        sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))
        knee = root + L1 * (cos_a * u + sin_a * w)
        return knee, root + u * dc

    def _frame_scale(self, up_pose, valid):
        """The uniform scale for one frame: the take-wide fit once
        `fit_to_subject` has run, else this frame's own height ratio."""
        if self._scale is not None:
            return self._scale
        vpts = np.asarray(up_pose, float).reshape(NUM_JOINTS, 3)[valid]
        our_h = float(vpts[:, 2].max() - vpts[:, 2].min()) or 1.0
        return self.rig_h / our_h

    def _skin_matrices(self, up_pose, valid, head_pts=None):
        """Per-bone skin (deform) matrices in RIG space + the alignment used.

        Returns (skin (B,4,4), origin (3,), scale, Rz (3,3)) or (None,)*4 if the
        pose has no usable pelvis. Every skin matrix is a pure rotation plus a
        translation — no bone is ever scaled.

        `head_pts` is the optional (NUM_HEAD_KP, 3) face keypoints in the same
        space as `up_pose`; given them the head bone gets a real orientation
        instead of riding the neck.

        RIG SPACE means two things are removed here and put back by whoever
        needs them: `to_rig` centres on THIS frame's pelvis (so the subject's
        travel is not in the result) and `Rz` turns this frame's shoulder line
        onto the rig's rest facing (so the subject's turning is not either).
        `_from_rig` puts both back for the live view; `export_transform` puts
        both back for the exported file. Neither is a property of the take, so
        this stays a pure function of one frame.
        """
        up_pose = np.asarray(up_pose, float).reshape(NUM_JOINTS, 3)
        j = Joint

        def J(i):
            return up_pose[int(i)] if valid[int(i)] else None

        pelvis = J(j.PELVIS)
        if pelvis is None:
            hips = [h for h in (J(j.LEFT_HIP), J(j.RIGHT_HIP)) if h is not None]
            if not hips:
                return None, None, None, None
            pelvis = np.mean(hips, axis=0)

        origin = pelvis

        scale = self._frame_scale(up_pose, valid)

        ls, rs = J(j.LEFT_SHOULDER), J(j.RIGHT_SHOULDER)
        Rz = np.eye(3)
        if ls is not None and rs is not None:
            our_right = np.array([rs[0] - ls[0], rs[1] - ls[1]])
            n = np.linalg.norm(our_right)
            if n > 1e-9:
                our_right /= n
                dt = (np.arctan2(self.rig_right[1], self.rig_right[0])
                      - np.arctan2(our_right[1], our_right[0]))
                ca, sa = np.cos(dt), np.sin(dt)
                Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])

        roll = self._roll_targets(up_pose, valid, Rz)

        def to_rig(p):
            return self.hips_world + (Rz @ (p - origin)) * scale

        # Face keypoints, once: the ear midpoint aims the neck (it is on the
        # skull axis, so no anatomical offset is needed) and the full basis
        # orients the head bone.
        head_rig = ear_mid = None
        if head_pts is not None:
            hp = np.asarray(head_pts, float).reshape(-1, 3)
            if len(hp) >= NUM_HEAD_KP:
                head_rig = np.array([to_rig(q) if not np.isnan(q).any()
                                     else q for q in hp])
                for a, b in ((3, 4), (1, 2)):          # ears, then eyes
                    if not (np.isnan(hp[a]).any() or np.isnan(hp[b]).any()):
                        ear_mid = (hp[a] + hp[b]) / 2.0
                        break

        def resolve(spec):
            if spec == _MID:
                # use the resolved pelvis, which may have come from the hips:
                # re-reading Joint.PELVIS here used to freeze the whole torso
                # whenever the pelvis itself was missing
                a = J(Joint.NECK)
                return None if a is None else to_rig((a + pelvis) / 2.0)
            if spec == _EAR_MID:
                if ear_mid is not None:
                    return to_rig(ear_mid)
                # no face keypoints this frame: legacy bias-corrected aim at
                # the canonical HEAD, so the neck still follows a nose drag
                p = self._head_aim_target(J, pelvis)
                return None if p is None else to_rig(p)
            p = J(spec)
            return None if p is None else to_rig(p)

        skin = np.tile(np.eye(4), (len(self.rest), 1, 1))

        # The pelvis is the root, so nothing above it can aim it: point it at
        # the same torso midpoint the spine targets, keeping the two collinear
        # so the waist doesn't crease.
        hips_head = self.head[self.hips_idx]
        hips_pos = to_rig(pelvis)
        R_hips = np.eye(3)
        mid = resolve(_MID)
        if mid is not None:
            n_want = float(np.linalg.norm(mid - hips_pos))
            if n_want > 1e-9:
                # the pelvis aims up the torso; the captured HIP line then says
                # which way it faces about that aim (F12)
                aim = (mid - hips_pos) / n_want
                ref0 = self._rest_ref.get(self.hips_idx)
                R_hips = _align(self._rest_torso, aim, ref=ref0)
                tgt, w = roll.get(self.hips_idx, (None, 0.0))
                R_hips = self._roll(self.hips_idx, R_hips, aim, tgt, w)
        skin[self.hips_idx][:3, :3] = R_hips
        skin[self.hips_idx][:3, 3] = hips_pos - R_hips @ hips_head

        # The head is the one bone whose ORIENTATION we can measure rather than
        # infer from an aim: two ears give the lateral axis a single nose point
        # cannot. Without face keypoints it falls through and rides the neck,
        # exactly as before.
        head_bone = self.role.get("head")
        head_R = None
        if head_rig is not None and self._rest_head is not None:
            target = self._head_basis(head_rig)
            if target is not None:
                head_R = target @ self._rest_head.T

        solved: dict[int, np.ndarray] = {}     # bone -> target from an IK solve
        for b in self.order:
            if b == self.hips_idx:
                continue
            p = int(self._eparent[b])
            base = skin[p] if p >= 0 else np.eye(4)

            end = solved.pop(b, None)
            if end is None and b in self._ik:
                # IK is the OCCLUSION fallback, not the primary path. When the
                # mid joint (knee/elbow) is captured, per-bone aiming below
                # tracks it exactly; two-bone IK would instead set the bend
                # angle purely from the root->end distance and use the captured
                # mid only to pick the bend plane — so dragging a knee up to
                # the hip left the rig's knee half bent, which is the defect
                # this ordering fixes. The trade: the end effector can sit off
                # by the (fitted, ~2%) proportion mismatch instead of landing
                # exactly.
                lb, mid_j, end_j = self._ik[b]
                if not valid[int(mid_j)]:              # mid joint occluded
                    target = resolve(end_j)
                    if target is not None:
                        sol = self._solve_ik(b, lb, base, target)
                        if sol is not None:
                            end, solved[lb] = sol
            if end is None and b in self._direct:
                # the primary path: aim this bone at its own captured joint,
                # so every joint DIRECTION follows the capture
                end = resolve(self._direct[b][1])

            if b == head_bone and head_R is not None:
                skin[b] = self._bone_rot(b, base, head_R)
            elif end is None:
                skin[b] = base
            else:
                tgt, w = roll.get(b, (None, 0.0))
                skin[b] = self._bone_fk(b, base, end, ref_target=tgt, weight=w)
        return skin, origin, scale, Rz

    # --- outputs -----------------------------------------------------------
    def _from_rig(self, pts, origin, scale, Rz):
        """Rig space -> the pose space the caller supplied.

        `origin`, `scale` and `Rz` are what `_skin_matrices` returned for this
        frame; feeding them straight back is what makes the round trip exact,
        and undoing `Rz` here is why the live view shows the subject TURN. The
        exported file gets the same two things back through
        `export_transform`, which is the same map with the rig's units and
        origin kept.
        """
        pts = np.atleast_2d(np.asarray(pts, float))
        return origin + (Rz.T @ (pts - self.hips_world).T).T / scale

    def ground_drop(self, up_pose, valid) -> float:
        """The rig's rest ankle-to-sole height in the CALLER's pose units.

        Grounding on the sole beneath the ankle, rather than on whichever mesh
        vertex is lowest this frame, is what stops the figure bobbing against
        the grid; see `pose3d.ui.view3d.ground_datum`.
        """
        return self.ankle_sole_drop / self._frame_scale(up_pose, valid)

    def pose_and_joints(self, up_pose, valid, head_pts=None):
        """(verts, faces, joints) — mesh and canonical joints of the posed rig.

        Both are returned in the SAME space as `up_pose`, so the joints can be
        drawn straight over the mesh. `joints` is (NUM_JOINTS,3); entries the
        rig cannot supply are NaN.

        Raises `PoseUnavailable` when the frame has no usable pelvis.

        No `pelvis_ref` here on purpose: the output is already in the capture's
        space, where the subject's travel is simply present, so there is
        nothing for a root offset to add.
        """
        skin, origin, scale, Rz = self._skin_matrices(up_pose, valid, head_pts)
        if skin is None:
            raise PoseUnavailable(
                "no usable pelvis in this frame: neither PELVIS nor either "
                "hip was reconstructed, so the rig has no root to stand on")

        out = np.zeros((len(self.verts0), 3))
        for k in range(self.w_idx.shape[1]):
            bi = self.w_idx[:, k]; wv = self.w_val[:, k]
            v = np.einsum("vij,vj->vi", skin[bi], self.vh)[:, :3]
            out += wv[:, None] * v

        verts = self._from_rig(out, origin, scale, Rz).astype(np.float32)
        joints = self._from_rig(self._joints_from_skin(skin), origin, scale, Rz)
        return verts, self.faces, joints

    def pose(self, up_pose, valid, head_pts=None):
        """Skinned vertices (V,3) in the same space as up_pose."""
        verts, faces, _ = self.pose_and_joints(up_pose, valid, head_pts)
        return verts, faces

    def posed_joints(self, up_pose, valid, head_pts=None):
        """Canonical joints of the posed rig, in the same space as up_pose."""
        return self.pose_and_joints(up_pose, valid, head_pts)[2]

    def export_transform(self, origin, scale, Rz, pelvis_ref):
        """The rig-space -> CAPTURE-frame map for one frame, as a (4,4).

        THE formula, in one place. It was written out three times (here, the
        export's `root_offsets`, the fidelity tool) and the copies could not
        be checked against each other.

            X = T(root_offset) @ R_about_hips(Rz.T)

        `R_about_hips` rotates about the rig's REST hips position, so the
        pelvis does not move and only the facing changes; `root_offset =
        (pelvis - pelvis_ref) * scale` is the subject's travel in the capture's
        own frame, NOT rotated by `Rz`. Composed with the rig-space pose it
        gives, for every joint,

            X @ j_rig = hips_world + scale * (j_capture - pelvis_ref)

        i.e. the de-tilted capture, uniformly scaled into rig units. One rigid
        transform applied to every bone's world matrix, which is exactly what
        Blender is driven with, so it is exact rather than an approximation.
        """
        R = np.asarray(Rz, float).T
        X = np.eye(4)
        X[:3, :3] = R
        X[:3, 3] = (self.hips_world - R @ self.hips_world
                    + root_offset(origin, scale, pelvis_ref))
        return X

    def pose_bone_matrices(self, up_pose, valid, head_pts=None, *,
                           keep_root_motion: bool = False, pelvis_ref=None,
                           return_alignment: bool = False):
        """Posed bone world matrices: {bone_name: (4,4) list}.

        M_posed[b] = skin[b] @ rest[b]. Blender's deform is
        pose_bone.matrix @ rest[b]^-1, so setting pose_bone.matrix = M_posed[b]
        reproduces this class's skinning exactly — that is what keeps the
        export identical to the 3D view. Returns None for an unusable pose.

        With `keep_root_motion` OFF the matrices are in RIG space: the rig's
        own rest facing, the pelvis pinned at `hips_world`. That is the
        rollback, bit-for-bit, and it is what the turntable and the pose-only
        probes read.

        With `keep_root_motion` ON and a take-wide `pelvis_ref` they are in the
        de-tilted CAPTURE frame expressed in rig units — `export_transform`
        above. This is the fix for BOTH halves of "the character does not
        follow the keypoints": `_skin_matrices` yaw-aligns every frame's
        shoulder line to the rig's rest facing, and the live view undoes that
        with `_from_rig`'s `Rz.T` while the export used not to, so the subject
        turned on screen (36.97 deg over the client take) and never turned in
        the delivered file; and the pelvis was pinned, so the travel was
        absent too. Both are undone here, per frame, with no caller-owned
        state: `Rz` and the pelvis are read off this frame's own solve.

        `return_alignment` also yields the frame's `(origin, scale, Rz,
        root_offset, transform)`, so a caller that needs the placement — the
        export's `root_offsets`, the fixed camera, the fidelity tool — reads it
        off the SAME solve instead of posing the frame a second time.
        """
        skin, origin, scale, Rz = self._skin_matrices(up_pose, valid, head_pts)
        if skin is None:
            return (None, None) if return_alignment else None
        X, off = None, np.zeros(3)
        if keep_root_motion and pelvis_ref is not None:
            X = self.export_transform(origin, scale, Rz, pelvis_ref)
            off = root_offset(origin, scale, pelvis_ref)
        mats = {}
        for b, name in enumerate(self.bone_names):
            M = skin[b] @ self.rest[b]
            mats[name] = (M if X is None else X @ M).tolist()
        if not return_alignment:
            return mats
        return mats, RigAlignment(np.asarray(origin, float), float(scale),
                                  np.asarray(Rz, float), off,
                                  np.eye(4) if X is None else X)

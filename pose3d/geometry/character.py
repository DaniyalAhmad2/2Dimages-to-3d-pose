"""Pose the bundled character to mimic a reconstructed skeleton.

The character's bone lengths are INVIOLABLE. Motion transfers as rotations only,
so the mesh can never be stretched or sheared: the rig is fitted to the capture,
not the other way round. Two-bone IK on the arms and legs puts the wrists and
ankles as close to the captured joints as fixed-length bones allow, using the
captured elbow/knee to choose which way the limb bends.

One uniform scale, fitted once per take, sizes the whole character to the
subject. Being uniform it changes size, never shape.

Rig bone names are reached through ROLES (see `_ROLE_ALIASES`), so swapping in a
differently-named rig is a data change rather than a code change.

Pure numpy, so it runs live in the 3D view; the Blender export drives the same
matrices via `pose_bone_matrices`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose3d.core.skeleton import BONES, Joint, NUM_JOINTS

_ASSET = Path(__file__).parent.parent / "assets" / "character.npz"

_MID = "MID"   # midpoint(pelvis, neck) — the torso split point

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

# Limbs solved with two-bone IK: (upper role, lower role, mid joint, end joint).
# The upper bone's aim comes out of the IK solve rather than _DIRECT.
_IK_CHAINS = (
    ("upper_arm.L", "forearm.L", Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    ("upper_arm.R", "forearm.R", Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    ("thigh.L", "shin.L", Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    ("thigh.R", "shin.R", Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
)

# Where each canonical joint is read off the posed rig. Heads are preferred:
# a bone's head is the exact FK position carried by its parent, so the reported
# skeleton is guaranteed consistent with the mesh. Fallbacks are used when the
# rig lacks the bone (e.g. no hand bone -> the forearm's tail is the wrist).
_JOINT_FROM_RIG = {
    Joint.PELVIS: (("hips", "head"),),
    Joint.NECK: (("neck", "head"), ("chest", "tail")),
    # the head bone's tail sits on the mesh surface; its midpoint sits inside
    # the skull, which is what an overlay needs
    Joint.HEAD: (("head", "mid"),),
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

# Legs weigh double when fitting the uniform scale: feet not reaching the floor
# is the mismatch the eye picks up first.
_SCALE_WEIGHTS = {Joint.LEFT_KNEE: 2.0, Joint.RIGHT_KNEE: 2.0,
                  Joint.LEFT_ANKLE: 2.0, Joint.RIGHT_ANKLE: 2.0}

# Below this fraction of the upper bone's length, the captured mid-joint is too
# close to the root->target axis to say which way the limb bends, so the rig's
# own rest bend takes over (blended, not switched, or the knee snaps sides).
_POLE_EPS = 0.05


def _align(a, b):
    """3x3 rotation taking unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.999999:
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
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


class Character:
    def __init__(self, path=_ASSET):
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

    def _resolve_joint_sources(self):
        """{canonical joint: (bone index, 'head'|'tail'|'mid')} for this rig."""
        out = {}
        for j, cands in _JOINT_FROM_RIG.items():
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
    def _bone_fk(self, b, base, end):
        """Skin matrix for bone b, hanging off its already-posed parent.

        The head is carried by the parent, so joined bones can never separate,
        and the bone then rotates about that head to aim at `end`. Rotation
        only — the bone keeps its rest length, so the mesh cannot deform.
        """
        head = base[:3, :3] @ self.head[b] + base[:3, 3]
        rest_d = self.tail[b] - self.head[b]
        n_rest = float(np.linalg.norm(rest_d))
        d_want = end - head
        n_want = float(np.linalg.norm(d_want))
        if n_rest < 1e-9 or n_want < 1e-9:
            return base
        R = _align(rest_d / n_rest, d_want / n_want)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = head - R @ self.head[b]      # pin the head in place
        return M

    def _pole(self, ub, base, root, u, mid, L1):
        """Unit vector, perpendicular to u, giving the limb's bend direction."""
        rest = base[:3, :3] @ self._rest_pole[ub]
        rest = rest - np.dot(rest, u) * u
        n_rest = np.linalg.norm(rest)
        rest = rest / n_rest if n_rest > 1e-9 else _perp(u)
        if mid is None:
            return rest
        v = (mid - root) - np.dot(mid - root, u) * u
        n = float(np.linalg.norm(v))
        tau = _POLE_EPS * L1
        if n < 1e-12 or tau < 1e-12:
            return rest
        # blend rather than switch: a hard cutoff makes the knee snap sides
        # frame to frame whenever the limb passes near-straight
        t = float(np.clip((n - tau) / tau, 0.0, 1.0))
        w = (1.0 - t) * rest + t * (v / n)
        w = w - np.dot(w, u) * u
        nw = np.linalg.norm(w)
        return w / nw if nw > 1e-9 else rest

    def _solve_ik(self, ub, lb, base, mid, end):
        """Two-bone IK with FIXED bone lengths.

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
        w = self._pole(ub, base, root, u, mid, L1)
        cos_a = float(np.clip((dc * dc + L1 * L1 - L2 * L2) / (2.0 * dc * L1), -1.0, 1.0))
        sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))
        knee = root + L1 * (cos_a * u + sin_a * w)
        return knee, root + u * dc

    def _skin_matrices(self, up_pose, valid):
        """Per-bone skin (deform) matrices in RIG space + the alignment used.

        Returns (skin (B,4,4), pelvis (3,), scale, Rz (3,3)) or (None,)*4 if the
        pose has no usable pelvis. Every skin matrix is a pure rotation plus a
        translation — no bone is ever scaled.
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

        if self._scale is not None:
            scale = self._scale
        else:
            vpts = up_pose[valid]
            our_h = float(vpts[:, 2].max() - vpts[:, 2].min()) or 1.0
            scale = self.rig_h / our_h

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

        def to_rig(p):
            return self.hips_world + (Rz @ (p - pelvis)) * scale

        def resolve(spec):
            if spec == _MID:
                # use the resolved pelvis, which may have come from the hips:
                # re-reading Joint.PELVIS here used to freeze the whole torso
                # whenever the pelvis itself was missing
                a = J(Joint.NECK)
                return None if a is None else to_rig((a + pelvis) / 2.0)
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
            want = mid - hips_pos
            if np.linalg.norm(want) > 1e-9:
                R_hips = _align(self._rest_torso, want)
        skin[self.hips_idx][:3, :3] = R_hips
        skin[self.hips_idx][:3, 3] = hips_pos - R_hips @ hips_head

        solved: dict[int, np.ndarray] = {}     # bone -> target from an IK solve
        for b in self.order:
            if b == self.hips_idx:
                continue
            p = int(self._eparent[b])
            base = skin[p] if p >= 0 else np.eye(4)

            end = solved.pop(b, None)
            if end is None and b in self._ik:
                lb, mid_j, end_j = self._ik[b]
                target = resolve(end_j)
                if target is not None:
                    sol = self._solve_ik(b, lb, base, resolve(mid_j), target)
                    if sol is not None:
                        end, solved[lb] = sol
            if end is None and b in self._direct:
                # no IK (occluded end joint): fall back to aiming at this
                # bone's own joint, which still tracks the capture
                end = resolve(self._direct[b][1])

            skin[b] = base if end is None else self._bone_fk(b, base, end)
        return skin, pelvis, scale, Rz

    # --- outputs -----------------------------------------------------------
    def _from_rig(self, pts, pelvis, scale, Rz):
        """Rig space -> the pose space the caller supplied."""
        pts = np.atleast_2d(np.asarray(pts, float))
        return pelvis + (Rz.T @ (pts - self.hips_world).T).T / scale

    def pose_and_joints(self, up_pose, valid):
        """(verts, faces, joints) — mesh and canonical joints of the posed rig.

        Both are returned in the SAME space as `up_pose`, so the joints can be
        drawn straight over the mesh. `joints` is (NUM_JOINTS,3); entries the
        rig cannot supply are NaN.
        """
        skin, pelvis, scale, Rz = self._skin_matrices(up_pose, valid)
        if skin is None:
            return None, None, None

        out = np.zeros((len(self.verts0), 3))
        for k in range(self.w_idx.shape[1]):
            bi = self.w_idx[:, k]; wv = self.w_val[:, k]
            v = np.einsum("vij,vj->vi", skin[bi], self.vh)[:, :3]
            out += wv[:, None] * v

        verts = self._from_rig(out, pelvis, scale, Rz).astype(np.float32)
        joints = self._from_rig(self._joints_from_skin(skin), pelvis, scale, Rz)
        return verts, self.faces, joints

    def pose(self, up_pose, valid):
        """Skinned vertices (V,3) in the same space as up_pose."""
        verts, faces, _ = self.pose_and_joints(up_pose, valid)
        return verts, faces

    def posed_joints(self, up_pose, valid):
        """Canonical joints of the posed rig, in the same space as up_pose."""
        return self.pose_and_joints(up_pose, valid)[2]

    def pose_bone_matrices(self, up_pose, valid):
        """Posed bone world matrices in RIG space: {bone_name: (4,4) list}.

        M_posed[b] = skin[b] @ rest[b]. Blender's deform is
        pose_bone.matrix @ rest[b]^-1, so setting pose_bone.matrix = M_posed[b]
        reproduces this class's skinning exactly — that is what keeps the
        export identical to the 3D view. Returns None for an unusable pose.
        """
        skin, *_ = self._skin_matrices(up_pose, valid)
        if skin is None:
            return None
        return {name: (skin[b] @ self.rest[b]).tolist()
                for b, name in enumerate(self.bone_names)}

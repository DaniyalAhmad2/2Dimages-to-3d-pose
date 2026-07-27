"""Pose the bundled low-poly character to match a reconstructed skeleton.

Loads the baked character asset (mesh + skin weights + rest bones) and, for a
given upright pose, computes posed bone matrices by aiming each rig bone at the
corresponding joint (the same retarget the Blender export does), then linear-
blend-skins the mesh. Pure numpy, so it runs live in the 3D view.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pose3d.core.skeleton import Joint, NUM_JOINTS

_ASSET = Path(__file__).parent.parent / "assets" / "character.npz"

_MID = "MID"   # midpoint(pelvis, neck) — the torso split point

# rig deform bone -> (start joint, end joint) it should span. Only the end joint
# is used: the head is carried by the parent (FK), and the bone rotates to aim at
# the end joint. Bones are NOT stretched to reach it — the rig is stylised (its
# thigh is 1.04 against a 1.54 shin, where a real thigh and shin are about equal)
# so stretching to match a real subject elongated the legs badly and crushed the
# torso. The character keeps its own proportions and just mimics the motion.
# The head/neck are intentionally NOT driven (they inherit the torso).
_DIRECT = {
    "spine": (Joint.PELVIS, _MID), "chest": (_MID, Joint.NECK),
    "upper_arm.L": (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    "forearm.L": (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    "upper_arm.R": (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    "forearm.R": (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    "thigh.L": (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    "shin.L": (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    "thigh.R": (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    "shin.R": (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
}

def _align(a, b):
    """3x3 rotation taking unit vector a to unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if c < -0.999999:
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
        x, y, z = axis
        return 2 * np.outer(axis, axis) - np.eye(3)  # 180° about axis
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


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
        # hierarchy order (parents before children) for skin-matrix inheritance
        self.order = self._topo()
        # rest homogeneous verts (V,4)
        self.vh = np.hstack([self.verts0, np.ones((len(self.verts0), 1))])
        # rig reference for alignment
        self.rig_h = float(self.head[:, 2].max() - self.head[:, 2].min()) or 1.0
        self.hips_world = self.head[self.bidx["hips"]]
        lh = self.head[self.bidx["upper_arm.L"]]; rh = self.head[self.bidx["upper_arm.R"]]
        self.rig_right = np.array([rh[0] - lh[0], rh[1] - lh[1]])
        self.rig_right /= (np.linalg.norm(self.rig_right) + 1e-12)
        self.hips_idx = self.bidx["hips"]
        self._direct = {self.bidx[b]: se for b, se in _DIRECT.items()
                        if b in self.bidx}
        # The rig carries stray helper bones (leg IK targets like "shin.L.001")
        # that have no parent and no skin weight. Left alone they keep their rest
        # transform while the body moves, so they float away from it in the
        # exported armature. Make each follow the bone it is named after.
        self._orphan = {}
        for b, name in enumerate(names):
            if b == self.hips_idx or self.parent[b] >= 0:
                continue
            base = re.sub(r"\.\d+$", "", name)
            self._orphan[b] = self.bidx.get(base, self.hips_idx)

    def _topo(self):
        order, seen = [], set()
        def visit(b):
            if b in seen or b < 0:
                return
            if self.parent[b] >= 0:
                visit(self.parent[b])
            seen.add(b); order.append(b)
        for b in range(len(self.parent)):
            visit(b)
        return order

    def _skin_matrices(self, up_pose, valid):
        """Per-bone skin (deform) matrices in RIG space + the alignment used.

        Returns (skin (B,4,4), pelvis (3,), scale, Rz (3,3)) or (None,...) if the
        pose has no usable pelvis. skin[b] maps a rest vertex to its posed
        position; this is the single source of truth shared by pose() (LBS for
        the live view) and pose_bone_matrices() (drives the Blender export).
        """
        up_pose = np.asarray(up_pose, float).reshape(NUM_JOINTS, 3)
        j = Joint
        def J(i):
            return up_pose[int(i)] if valid[int(i)] else None
        pelvis = J(j.PELVIS)
        if pelvis is None:
            hips = [J(j.LEFT_HIP), J(j.RIGHT_HIP)]
            hips = [h for h in hips if h is not None]
            if not hips:
                return None, None, None, None
            pelvis = np.mean(hips, axis=0)
        # our height + shoulder line for alignment
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
                a0 = np.arctan2(our_right[1], our_right[0])
                a1 = np.arctan2(self.rig_right[1], self.rig_right[0])
                dt = a1 - a0
                ca, sa = np.cos(dt), np.sin(dt)
                Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])

        # A: up_pose space -> rig space
        def to_rig(p):
            return self.hips_world + (Rz @ (p - pelvis)) * scale

        def resolve(spec):
            if spec == _MID:
                a, b = J(Joint.NECK), J(Joint.PELVIS)
                return None if (a is None or b is None) else to_rig((a + b) / 2)
            p = J(spec)
            return None if p is None else to_rig(p)

        # Per-bone skin matrix, built as an FK chain: every bone's head follows
        # its parent, so neighbouring bones always stay joined and the mesh can't
        # tear; a mapped bone then rotates to aim at its keypoint and stretches
        # (clamped) toward it. Unmapped bones just follow the parent rigidly, so
        # hands trail the wrist and feet the ankle.
        # The pelvis is the root of the chain, so it has no parent to aim it.
        # Orient it from the hip line + the torso direction; left at rest it
        # stays upright while everything above it rotates, which creases the
        # mesh at the waist.
        skin = np.tile(np.eye(4), (len(self.rest), 1, 1))
        hips_head = self.head[self.hips_idx]
        hips_pos = to_rig(pelvis)
        R_hips = np.eye(3)
        mid = resolve(_MID)
        if mid is not None:
            rest_dir = self.tail[self.hips_idx] - hips_head
            want = mid - hips_pos
            if np.linalg.norm(rest_dir) > 1e-9 and np.linalg.norm(want) > 1e-9:
                # aim the pelvis at the same torso midpoint the spine targets, so
                # the two stay collinear and the waist doesn't crease. Rotation
                # only: stretching here would scale the legs, which hang off it.
                R_hips = _align(rest_dir, want)
        skin[self.hips_idx][:3, :3] = R_hips
        skin[self.hips_idx][:3, 3] = hips_pos - R_hips @ hips_head
        for b in self.order:
            if b == self.hips_idx or b in self._orphan:
                continue
            p = self.parent[b]
            base = skin[p] if p >= 0 else np.eye(4)
            end = resolve(self._direct[b][1]) if b in self._direct else None
            skin[b] = base if end is None else self._bone_fk(b, base, end)
        # parentless helper bones follow the bone they are named after; done last
        # so their target is already posed, whatever the bone ordering is.
        for b, target in self._orphan.items():
            skin[b] = skin[target]
        return skin, pelvis, scale, Rz

    def pose_bone_matrices(self, up_pose, valid):
        """Posed bone world matrices in rig space: {bone_name: (4,4) list}.

        M_posed[b] = skin[b] @ rest[b]. Blender's deform is
        pose_bone.matrix @ rest[b]^-1, so setting pose_bone.matrix = M_posed[b]
        reproduces this class's skinning EXACTLY — the export then matches the
        live 3D preview pose-for-pose. Returns None for an unusable pose.
        """
        skin, *_ = self._skin_matrices(up_pose, valid)
        if skin is None:
            return None
        return {name: (skin[b] @ self.rest[b]).tolist()
                for b, name in enumerate(self.bone_names)}

    def pose(self, up_pose, valid):
        """Return skinned vertices (V,3) placed in the SAME space as up_pose.

        up_pose: (NUM_JOINTS,3) upright pose (as shown in the view). valid mask.
        """
        skin, pelvis, scale, Rz = self._skin_matrices(up_pose, valid)
        if skin is None:
            return None, None

        out = np.zeros((len(self.verts0), 3))
        for k in range(self.w_idx.shape[1]):
            bi = self.w_idx[:, k]; wv = self.w_val[:, k]
            M = skin[bi]                                 # (V,4,4)
            v = np.einsum("vij,vj->vi", M, self.vh)[:, :3]
            out += wv[:, None] * v

        # map rig space back to up_pose space so it aligns with the skeleton
        out = pelvis + (Rz.T @ (out - self.hips_world).T).T / scale
        return out.astype(np.float32), self.faces

    def _bone_fk(self, b, base, end):
        """Skin matrix for bone b, hanging off its already-posed parent.

        `base` is the parent's skin matrix. The bone's head is carried by the
        parent (so the two never separate — this is what keeps the waist and
        shoulders from tearing), and the bone is then rotated about that head to
        aim at `end`. Rotation only: the bone keeps its rest length, so the mesh
        is never stretched and the character holds its own proportions.
        """
        R_par, t_par = base[:3, :3], base[:3, 3]
        head = R_par @ self.head[b] + t_par                  # posed head
        d_par = R_par @ (self.tail[b] - self.head[b])        # bone, carried by parent
        n_par = float(np.linalg.norm(d_par))
        d_want = end - head
        n_want = float(np.linalg.norm(d_want))
        if n_par < 1e-9 or n_want < 1e-9:
            return base
        R = _align(d_par / n_par, d_want / n_want)           # aim at the joint
        M = np.eye(4)
        M[:3, :3] = R @ R_par
        M[:3, 3] = head - R @ (R_par @ self.head[b])         # pin the head in place
        return M

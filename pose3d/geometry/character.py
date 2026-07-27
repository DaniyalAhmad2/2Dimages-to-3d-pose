"""Pose the bundled low-poly character to match a reconstructed skeleton.

Loads the baked character asset (mesh + skin weights + rest bones) and, for a
given upright pose, computes posed bone matrices by aiming each rig bone at the
corresponding joint (the same retarget the Blender export does), then linear-
blend-skins the mesh. Pure numpy, so it runs live in the 3D view.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pose3d.core.skeleton import Joint, NUM_JOINTS

_ASSET = Path(__file__).parent.parent / "assets" / "character.npz"

_MID = "MID"   # midpoint(pelvis, neck) — the torso split point

# rig deform bone -> (start joint, end joint) it should span. Bones rotate to
# aim at the keypoint and stretch toward it, but stretch is CLAMPED (below) so
# the model bends to the pose without grotesque elongation. The head/neck are
# intentionally NOT driven (they inherit the torso) so the head stays natural.
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
_STRETCH_MIN, _STRETCH_MAX = 0.8, 1.25   # clamp bone stretch (keep proportions)


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

        # per-bone skin matrix: directly place mapped bones between their two
        # keypoints (stretched to fit); other bones inherit their parent so the
        # mesh stays connected (hands follow the wrist, feet the ankle, ...).
        skin = np.tile(np.eye(4), (len(self.rest), 1, 1))
        skin[self.hips_idx][:3, 3] = to_rig(pelvis) - self.head[self.hips_idx]
        for b in self.order:
            if b in self._direct:
                s_spec, e_spec = self._direct[b]
                start, end = resolve(s_spec), resolve(e_spec)
                if start is not None and end is not None:
                    skin[b] = self._bone_delta(b, start, end)
                    continue
            if b == self.hips_idx:
                continue
            p = self.parent[b]
            if p >= 0:
                skin[b] = skin[p]
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

    def _bone_delta(self, b, start, end):
        """Skin matrix placing bone b from its rest to span start->end (with
        stretch along the bone so the tail reaches `end`)."""
        rest_head, rest_tail = self.head[b], self.tail[b]
        rd = rest_tail - rest_head
        rlen = float(np.linalg.norm(rd))
        dv = end - start
        dlen = float(np.linalg.norm(dv))
        D = np.eye(4)
        if rlen < 1e-9 or dlen < 1e-9:
            D[:3, 3] = start - rest_head
            return D
        u = dv / dlen
        R = _align(rd / rlen, u)                       # aim the bone at the joint
        s = np.clip(dlen / rlen, _STRETCH_MIN, _STRETCH_MAX)   # clamped stretch
        Sc = np.eye(3) + (s - 1.0) * np.outer(u, u)
        M = Sc @ R
        D[:3, :3] = M
        D[:3, 3] = start - M @ rest_head
        return D

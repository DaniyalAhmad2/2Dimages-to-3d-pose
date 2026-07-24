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

# rig deform bone -> canonical joint it should point at (aim / Damped-Track)
_TRACK = {
    "spine": Joint.NECK, "chest": Joint.NECK, "neck": Joint.HEAD,
    "upper_arm.L": Joint.LEFT_ELBOW, "forearm.L": Joint.LEFT_WRIST,
    "upper_arm.R": Joint.RIGHT_ELBOW, "forearm.R": Joint.RIGHT_WRIST,
    "thigh.L": Joint.LEFT_KNEE, "shin.L": Joint.LEFT_ANKLE,
    "thigh.R": Joint.RIGHT_KNEE, "shin.R": Joint.RIGHT_ANKLE,
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
        self.parent = d["parent"].astype(int)
        names = [str(n) for n in d["bone_names"]]
        self.bidx = {n: i for i, n in enumerate(names)}
        self.inv_rest = np.linalg.inv(self.rest)
        # local rest matrices (relative to parent)
        self.local = np.empty_like(self.rest)
        for b in range(len(self.rest)):
            p = self.parent[b]
            self.local[b] = self.inv_rest[p] @ self.rest[b] if p >= 0 else self.rest[b]
        # hierarchy order (parents before children)
        self.order = self._topo()
        # rest homogeneous verts (V,4)
        self.vh = np.hstack([self.verts0, np.ones((len(self.verts0), 1))])
        # rig reference for alignment
        self.rig_h = float(self.head[:, 2].max() - self.head[:, 2].min()) or 1.0
        self.hips_world = self.head[self.bidx["hips"]]
        lh = self.head[self.bidx["upper_arm.L"]]; rh = self.head[self.bidx["upper_arm.R"]]
        self.rig_right = np.array([rh[0] - lh[0], rh[1] - lh[1]])
        self.rig_right /= (np.linalg.norm(self.rig_right) + 1e-12)
        self._track = {self.bidx[b]: int(j) for b, j in _TRACK.items() if b in self.bidx}

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

    def pose(self, up_pose, valid):
        """Return skinned vertices (V,3) placed in the SAME space as up_pose.

        up_pose: (NUM_JOINTS,3) upright pose (as shown in the view). valid mask.
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
                return None, None
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

        # A: up_pose space -> rig space ;  targets in rig space
        def to_rig(p):
            return self.hips_world + (Rz @ (p - pelvis)) * scale
        targets = {ji: to_rig(J(ji)) for ji in set(self._track.values())
                   if J(ji) is not None}

        # posed bone world matrices (aim FK)
        posed = self.rest.copy()
        for b in self.order:
            p = self.parent[b]
            base = (posed[p] @ self.local[b]) if p >= 0 else self.rest[b].copy()
            if p < 0:                                  # root: translate to pelvis
                base[:3, 3] = to_rig(pelvis)
            tgt_j = self._track.get(b)
            if tgt_j is not None and tgt_j in targets:
                head = base[:3, 3]
                cur_y = base[:3, :3] @ np.array([0, 1.0, 0])
                desired = targets[tgt_j] - head
                if np.linalg.norm(desired) > 1e-9:
                    Rw = _align(cur_y, desired / np.linalg.norm(desired))
                    base[:3, :3] = Rw @ base[:3, :3]
            posed[b] = base

        # linear blend skinning (in rig space)
        skin = posed @ self.inv_rest                    # (B,4,4)
        out = np.zeros((len(self.verts0), 3))
        for k in range(self.w_idx.shape[1]):
            bi = self.w_idx[:, k]; wv = self.w_val[:, k]
            M = skin[bi]                                 # (V,4,4)
            v = np.einsum("vij,vj->vi", M, self.vh)[:, :3]
            out += wv[:, None] * v

        # map rig space back to up_pose space so it aligns with the skeleton
        out = pelvis + (Rz.T @ (out - self.hips_world).T).T / scale
        return out.astype(np.float32), self.faces

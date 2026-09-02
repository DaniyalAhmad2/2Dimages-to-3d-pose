"""Tiny software renderer (PIL): painter's algorithm + lambert, skeleton on top.

Reproduces View3D.set_pose's framing (de-tilt, centre on the mean of the valid
joints, ground on the character's lowest vertex) so the contact sheets show
what the client sees. matplotlib is NOT installed in this venv; PIL is.
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
sys.path.insert(0, HERE)
import metrics as M  # noqa
from pose3d.core.skeleton import BONES, Joint
from pose3d.geometry.character import Character
from pose3d.geometry.orient import sequence_up, de_tilt_matrix
from roll_prototype import RollCharacter

u = lambda v: v / (np.linalg.norm(v) + 1e-12)
W = Hpx = 260


class NoFudge(Character):
    def _head_aim_target(self, J, pelvis):
        return J(Joint.HEAD)


def cam(az_deg, el_deg=8.0):
    a, e = np.radians(az_deg), np.radians(el_deg)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = u(np.cross(fwd, [0, 0, 1.0]))
    return np.stack([right, np.cross(right, fwd), fwd])


def tile(verts, faces, cj, cap, valid, Rc, half, label):
    im = Image.new("RGB", (W, Hpx), (16, 18, 24))
    d = ImageDraw.Draw(im)
    def px(P):
        x = (P[..., 0] / half) * (W * 0.45) + W / 2
        y = Hpx * 0.93 - (P[..., 1] / half) * (W * 0.45)
        return np.stack([x, y], -1)
    if verts is not None:
        V = verts @ Rc.T
        tri = V[faces]
        order = np.argsort(-tri[:, :, 2].mean(1))
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        L = u(np.array([0.35, 0.45, -0.82]))
        sh = np.clip(0.22 + 0.78 * np.abs(n @ L), 0, 1)
        P = px(tri[:, :, :2] if False else tri)[:, :, :2]
        for i in order:
            c = int(255 * sh[i])
            d.polygon([tuple(q) for q in P[i]],
                      fill=(int(c * 0.72), int(c * 0.72), int(c * 0.80)))
    C = px(cap @ Rc.T)
    for a, b in BONES:
        if valid[int(a)] and valid[int(b)]:
            d.line([tuple(C[int(a)]), tuple(C[int(b)])], fill=(255, 168, 50), width=2)
    for j in range(len(C)):
        if valid[j]:
            d.ellipse([C[j][0] - 3, C[j][1] - 3, C[j][0] + 3, C[j][1] + 3],
                      fill=(255, 130, 0))
    if cj is not None:
        Jv = px(cj @ Rc.T)
        ok = ~np.isnan(cj).any(1)
        for j in range(len(Jv)):
            if ok[j]:
                d.ellipse([Jv[j][0] - 2.5, Jv[j][1] - 2.5, Jv[j][0] + 2.5, Jv[j][1] + 2.5],
                          fill=(63, 216, 255))
    d.text((5, 4), label, fill=(230, 230, 235))
    return im


def main():
    take = M.load_take("workspace/pose3d_projects/Imported_Session")
    poses = np.asarray(take["fitted3d"], float)
    R = de_tilt_matrix(sequence_up(poses))
    up = poses @ R.T
    variants = [("shipped", Character()), ("no-nose-fudge", NoFudge()),
                ("roll-proto", RollCharacter())]
    for _, c in variants:
        c.fit_to_subject(up)
    frames = [0, 5, 10, 15, 20, 25]
    for azi, tag in ((-70, "front"), (20, "side")):
        Rc = cam(azi)
        sheet = Image.new("RGB", (W * len(frames), Hpx * len(variants)), (16, 18, 24))
        for r, (name, ch) in enumerate(variants):
            for c, t in enumerate(frames):
                p = up[t].copy()
                valid = ~np.isnan(p).any(1)
                v = p.copy()
                v[:, 0] -= p[valid, 0].mean(); v[:, 1] -= p[valid, 1].mean()
                v[:, 2] -= p[valid, 2].min()
                verts, faces, cj = ch.pose_and_joints(
                    np.where(valid[:, None], v, np.nan), valid, None)
                if verts is not None:
                    dz = float(verts[:, 2].min())
                    verts = verts - np.array([0, 0, dz])
                    v = v - np.array([0, 0, dz])
                    if cj is not None:
                        cj = cj - np.array([0, 0, dz])
                ref = verts if verts is not None else v[valid]
                half = 0.62 * float(np.ptp(ref, axis=0).max())
                sheet.paste(tile(verts, faces, cj, v, valid, Rc, half,
                                 f"{name} f{t + 1:04d}"), (c * W, r * Hpx))
        out = os.path.join(HERE, f"contact_{tag}.png")
        sheet.save(out)
        print("wrote", out)


if __name__ == "__main__":
    main()

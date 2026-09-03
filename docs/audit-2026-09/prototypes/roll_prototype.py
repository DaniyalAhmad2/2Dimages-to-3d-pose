"""H9(b) prototype: aim + explicit ROLL reference for every bone.

Subclasses Character (no repo edit). Every bone still AIMS at its own captured
joint exactly as today, but its rotation about that aim axis is fixed by an
anatomical reference instead of being whatever _align's minimal rotation
happens to give:

  thigh/shin, upper_arm/forearm : the captured limb bend-plane normal
  hips                          : the captured HIP line
  chest                         : the captured SHOULDER line
  neck/head/clavicle            : unchanged

Feet and hands carry no aim of their own, so they inherit the corrected roll.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import metrics as M  # noqa
from pose3d.core.skeleton import Joint, NUM_JOINTS, JOINT_NAMES
from pose3d.geometry.character import Character, _align, _MID, _HEAD_AIM  # renamed from _EAR_MID (rigid neck+head chain, task 1)
from pose3d.geometry.orient import sequence_up, de_tilt_matrix

u = lambda v: v / (np.linalg.norm(v) + 1e-12)


def _rot(axis, ang):
    a = u(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


class RollCharacter(Character):
    """Character with an explicit roll reference per bone."""

    # role -> (joint a, joint m, joint b) whose cross product is the roll ref
    _BEND = {
        "upper_arm.L": (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
        "forearm.L":   (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
        "upper_arm.R": (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
        "forearm.R":   (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
        "thigh.L":     (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
        "shin.L":      (Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
        "thigh.R":     (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
        "shin.R":      (Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    }
    # role -> (joint a, joint b) whose difference vector is the roll ref
    _LINE = {
        "hips":  (Joint.LEFT_HIP, Joint.RIGHT_HIP),
        "chest": (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
        "spine": (Joint.LEFT_HIP, Joint.RIGHT_HIP),
    }

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        rj = self.rest_joints()
        self._ref0 = {}                 # bone idx -> rest reference vector
        for role, (ja, jm, jb) in self._BEND.items():
            if role not in self.role:
                continue
            n = np.cross(rj[int(jm)] - rj[int(ja)], rj[int(jb)] - rj[int(jm)])
            if np.linalg.norm(n) < 1e-9 or np.isnan(n).any():
                # degenerate at rest (straight limb): use the rig's rest pole
                ub = self.role[role.replace("forearm", "upper_arm")
                               .replace("shin", "thigh")]
                n = np.cross(rj[int(jb)] - rj[int(ja)], self._rest_pole.get(ub, np.array([0, 1.0, 0])))
            self._ref0[self.role[role]] = u(n)
        for role, (ja, jb) in self._LINE.items():
            if role in self.role:
                self._ref0[self.role[role]] = u(rj[int(jb)] - rj[int(ja)])

    def _roll_target(self, b, up_pose, valid, Rz):
        """Captured reference vector for bone b, in rig space (or None)."""
        inv = {v: k for k, v in self.role.items()}
        role = inv.get(b)
        if role in self._BEND:
            ja, jm, jb = self._BEND[role]
            if not all(valid[int(x)] for x in (ja, jm, jb)):
                return None
            n = np.cross(up_pose[int(jm)] - up_pose[int(ja)],
                         up_pose[int(jb)] - up_pose[int(jm)])
        elif role in self._LINE:
            ja, jb = self._LINE[role]
            if not (valid[int(ja)] and valid[int(jb)]):
                return None
            n = up_pose[int(jb)] - up_pose[int(ja)]
        else:
            return None
        if np.linalg.norm(n) < 1e-9:
            return None
        return u(Rz @ n)

    def _apply_roll(self, b, R, axis, ref_t):
        r0 = self._ref0.get(b)
        if r0 is None or ref_t is None:
            return R
        a = u(axis)
        cur = R @ r0
        cur = cur - np.dot(cur, a) * a
        tgt = ref_t - np.dot(ref_t, a) * a
        if np.linalg.norm(cur) < 1e-6 or np.linalg.norm(tgt) < 1e-6:
            return R
        cur, tgt = u(cur), u(tgt)
        ang = np.arctan2(float(np.dot(np.cross(cur, tgt), a)), float(np.dot(cur, tgt)))
        return _rot(a, ang) @ R

    # copy of Character._skin_matrices with the roll step added -------------
    def _skin_matrices(self, up_pose, valid, head_pts=None):
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
                a = J(Joint.NECK)
                return None if a is None else to_rig((a + pelvis) / 2.0)
            if spec == _HEAD_AIM:
                p = self._head_aim_target(J, pelvis)
                return None if p is None else to_rig(p)
            p = J(spec)
            return None if p is None else to_rig(p)

        skin = np.tile(np.eye(4), (len(self.rest), 1, 1))
        hips_head = self.head[self.hips_idx]
        hips_pos = to_rig(pelvis)
        R_hips = np.eye(3)
        mid = resolve(_MID)
        if mid is not None:
            want = mid - hips_pos
            if np.linalg.norm(want) > 1e-9:
                R_hips = _align(self._rest_torso, want)
                R_hips = self._apply_roll(self.hips_idx, R_hips, want,
                                          self._roll_target(self.hips_idx, up_pose, valid, Rz))
        skin[self.hips_idx][:3, :3] = R_hips
        skin[self.hips_idx][:3, 3] = hips_pos - R_hips @ hips_head

        solved = {}
        for b in self.order:
            if b == self.hips_idx:
                continue
            p = int(self._eparent[b])
            base = skin[p] if p >= 0 else np.eye(4)
            end = solved.pop(b, None)
            if end is None and b in self._ik:
                lb, mid_j, end_j = self._ik[b]
                if not valid[int(mid_j)]:
                    target = resolve(end_j)
                    if target is not None:
                        sol = self._solve_ik(b, lb, base, target)
                        if sol is not None:
                            end, solved[lb] = sol
            if end is None and b in self._direct:
                end = resolve(self._direct[b][1])
            if end is None:
                skin[b] = base
                continue
            head = base[:3, :3] @ self.head[b] + base[:3, 3]
            rest_d = self.tail[b] - self.head[b]
            nr = float(np.linalg.norm(rest_d))
            d_want = end - head
            nw = float(np.linalg.norm(d_want))
            if nr < 1e-9 or nw < 1e-9:
                skin[b] = base
                continue
            R = _align(rest_d / nr, d_want / nw)
            R = self._apply_roll(b, R, d_want / nw,
                                 self._roll_target(b, up_pose, valid, Rz))
            Mt = np.eye(4); Mt[:3, :3] = R
            Mt[:3, 3] = head - R @ self.head[b]
            skin[b] = Mt
        return skin, pelvis, scale, Rz


def evaluate(ch, up, H, label):
    per = {j: [] for j in range(NUM_JOINTS)}
    rolljump = {n: [] for n in ("foot.L", "foot.R", "hand.L", "hand.R")}
    axes_prev = {}
    ground = []
    for p in up:
        v = ~np.isnan(p).any(1)
        verts, faces, Jt = ch.pose_and_joints(p, v, None)
        if Jt is None:
            continue
        d = np.linalg.norm(Jt - p, axis=1)
        for k in range(NUM_JOINTS):
            if v[k] and np.isfinite(d[k]):
                per[k].append(d[k])
        skin, *_ = ch._skin_matrices(p, v, None)
        for n in rolljump:
            bi = ch.role[n]
            ax = u(skin[bi][:3, :3] @ (ch.tail[bi] - ch.head[bi]))
            if n in axes_prev:
                rolljump[n].append(np.degrees(np.arccos(np.clip(
                    float(np.dot(ax, axes_prev[n])), -1, 1))))
            axes_prev[n] = ax
        low_ank = float(np.nanmin([p[int(Joint.LEFT_ANKLE), 2], p[int(Joint.RIGHT_ANKLE), 2]]))
        ground.append((low_ank - float(verts[:, 2].min())) / H * 100)
    out = {"label": label,
           "per_joint_pct_height": {JOINT_NAMES[k]: round(100 * float(np.median(v)) / H, 2)
                                    for k, v in per.items() if v},
           "all_median_pct_height": round(100 * float(np.median(
               np.concatenate([v for v in per.values() if v]))) / H, 2),
           "leaf_axis_frame_to_frame_deg_median": {
               n: round(float(np.median(v)), 1) for n, v in rolljump.items()},
           "ground_offset_pct_height": {"median": round(float(np.median(ground)), 1),
                                        "spread": round(float(np.max(ground) - np.min(ground)), 1)}}
    return out


if __name__ == "__main__":
    take = M.load_take("workspace/pose3d_projects/Imported_Session")
    poses = np.asarray(take["fitted3d"], float)
    R = de_tilt_matrix(sequence_up(poses))
    up = poses @ R.T
    H = float(np.median([float(p[~np.isnan(p).any(1), 2].max()
                               - p[~np.isnan(p).any(1), 2].min()) for p in up]))
    res = {}
    a = Character(); a.fit_to_subject(up)
    res["before"] = evaluate(a, up, H, "shipped Character")
    b = RollCharacter(); b.fit_to_subject(up)
    res["after"] = evaluate(b, up, H, "RollCharacter (aim + roll reference)")

    # roll of the limb bones vs the captured bend plane, before/after
    def bendroll(ch):
        out = {}
        for role, (ja, jm, jb) in RollCharacter._BEND.items():
            bi = ch.role[role]
            errs = []
            for p in up:
                v = ~np.isnan(p).any(1)
                skin, _, _, Rz = ch._skin_matrices(p, v, None)
                ax = u(skin[bi][:3, :3] @ (ch.tail[bi] - ch.head[bi]))
                n = np.cross(p[int(jm)] - p[int(ja)], p[int(jb)] - p[int(jm)])
                if np.linalg.norm(n) < 1e-9:
                    continue
                tgt = u(Rz @ n); tgt = u(tgt - np.dot(tgt, ax) * ax)
                r0 = RollCharacter._BEND and None
                ref0 = b._ref0[bi]
                cur = skin[bi][:3, :3] @ ref0
                cur = u(cur - np.dot(cur, ax) * ax)
                errs.append(abs(np.degrees(np.arctan2(
                    float(np.dot(np.cross(cur, tgt), ax)), float(np.dot(cur, tgt))))))
            out[role] = {"median_deg": round(float(np.median(errs)), 1),
                         "max_deg": round(float(np.max(errs)), 1)}
        return out
    res["roll_vs_bendplane_before"] = bendroll(a)
    res["roll_vs_bendplane_after"] = bendroll(b)
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(os.path.join(HERE, "roll_prototype.json"), "w"),
              indent=1, default=float)

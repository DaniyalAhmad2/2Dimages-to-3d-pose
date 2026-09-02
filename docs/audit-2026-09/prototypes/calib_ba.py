"""Bundle-adjust the two camera poses + the tag layout from ALL frames x ALL tags.

World frame = tag 14 (so the result is directly comparable to the shipped rig).
Unknowns: rvec/tvec of each camera (world->cam), rvec/tvec of tags 13/15/17
(tag->world).  Residual: reprojection of every detected tag corner, Huber loss.

Also: (a) the per-frame single-tag relative-pose spread (= the uncertainty the
shipped one-frame rig inherits), (b) a HOLDOUT test that never uses the right
camera's tag-15 observations and then triangulates tag 15's corners cross-view,
scoring the reconstructed square against its true 0.050 m side.
"""
import json, os, sys, itertools
import cv2, numpy as np
from scipy.optimize import least_squares

REPO = os.environ.get("POSE3D_REPO", "/media/athena/hd3/Projects/pose3d-tool")
sys.path.insert(0, REPO)
OUT = os.path.dirname(os.path.abspath(__file__))
TAKE = os.path.join(REPO, "workspace/pose3d_projects/Imported_Session")
from pose3d.calib.extrinsics import marker_object_points

L = 0.05
OBJ = marker_object_points(L).astype(np.float64)
SIZE = {"left": (3072, 4080), "right": (1536, 2048)}
CAMS = ("left", "right")
TAGS_FREE = ("13", "15", "17")          # 14 is the world frame
tags = json.load(open(os.path.join(OUT, "tags.json")))
FIDS = sorted(tags)


def K_of(cam, scale=1.0):
    w, h = SIZE[cam]
    f = max(w, h) * scale
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


DIST = np.zeros((1, 5))


def rod(r):
    R, _ = cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))
    return R


def irod(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def solves(img_pts, K):
    n, rv, tv, err = cv2.solvePnPGeneric(OBJ, np.asarray(img_pts, np.float64),
                                         K, DIST, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = [dict(R=rod(rv[i]), t=tv[i].ravel(), err=float(err[i][0])) for i in range(n)]
    return sorted(out, key=lambda x: x["err"])


def obs_list(drop=()):
    """[(fid, cam, tag, corners 4x2)] minus the (cam,tag) pairs in `drop`."""
    o = []
    for fid in FIDS:
        for cam in CAMS:
            for tid, c in (tags[fid][cam] or {}).items():
                if (cam, tid) in drop:
                    continue
                o.append((fid, cam, tid, np.asarray(c, float)))
    return o


def ang(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def rot_ang(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(A).T @ B) - 1) / 2, -1, 1))))


def rel_pose(Rl, tl, Rr, tr):
    """left-cam -> right-cam transform: invariant to the world frame."""
    R = Rr @ Rl.T
    t = tr - R @ tl
    return R, t


# ------------------------------------------------------------------ BA core
def run_ba(scale, drop=(), verbose=False):
    Ks = {c: K_of(c, scale) for c in CAMS}
    obs = obs_list(drop)

    # --- init: camera poses = median tag-14 solve; tag poses from the left cam
    init_cam = {}
    for cam in CAMS:
        S = [solves(c, Ks[cam])[0] for fid in FIDS
             for tid, c in (tags[fid][cam] or {}).items() if tid == "14"]
        # median over frames (rotation: pick the solve closest to the mean t)
        tm = np.median([s["t"] for s in S], axis=0)
        best = min(S, key=lambda s: np.linalg.norm(s["t"] - tm))
        init_cam[cam] = (best["R"], best["t"])
    init_tag = {}
    for tid in TAGS_FREE:
        cand = []
        for fid in FIDS:
            for cam in CAMS:
                d = tags[fid][cam] or {}
                if tid not in d:
                    continue
                S = solves(d[tid], Ks[cam])
                ratio = S[1]["err"] / max(S[0]["err"], 1e-9) if len(S) > 1 else 1e9
                Rc, tc = init_cam[cam]
                # tag->world = cam->world  o  tag->cam
                Rw = Rc.T @ S[0]["R"]
                tw = Rc.T @ (S[0]["t"] - tc)
                cand.append((ratio, Rw, tw))
        cand.sort(key=lambda x: -x[0])      # least-ambiguous solve first
        init_tag[tid] = (cand[0][1], cand[0][2])

    p0 = np.concatenate([np.concatenate([irod(init_cam[c][0]), init_cam[c][1]]) for c in CAMS]
                        + [np.concatenate([irod(init_tag[t][0]), init_tag[t][1]]) for t in TAGS_FREE])

    def unpack(p):
        cam = {c: (rod(p[6 * i:6 * i + 3]), p[6 * i + 3:6 * i + 6]) for i, c in enumerate(CAMS)}
        off = 6 * len(CAMS)
        tag = {"14": (np.eye(3), np.zeros(3))}
        for j, t in enumerate(TAGS_FREE):
            q = p[off + 6 * j: off + 6 * j + 6]
            tag[t] = (rod(q[:3]), q[3:])
        return cam, tag

    def resid(p):
        cam, tag = unpack(p)
        r = []
        for fid, c, tid, pts in obs:
            Rt, tt = tag[tid]
            Xw = (Rt @ OBJ.T).T + tt                 # corners in world
            Rc, tc = cam[c]
            proj = cv2.projectPoints(Xw, irod(Rc), tc, Ks[c], DIST)[0].reshape(-1, 2)
            r.append((proj - pts).ravel())
        return np.concatenate(r)

    r0 = resid(p0)
    sol = least_squares(resid, p0, loss="huber", f_scale=2.0, method="trf",
                        xtol=1e-12, ftol=1e-12, max_nfev=400)
    cam, tag = unpack(sol.x)
    r1 = resid(sol.x)
    if verbose:
        print(f"  BA scale={scale:.4f} drop={drop}: init rms {np.sqrt(np.mean(r0**2)):.2f} px "
              f"-> {np.sqrt(np.mean(r1**2)):.2f} px, n_res={len(r1)}, nfev={sol.nfev}")
    return dict(cam=cam, tag=tag, K=Ks,
                rms_init=float(np.sqrt(np.mean(r0 ** 2))),
                rms=float(np.sqrt(np.mean(r1 ** 2))),
                med_abs=float(np.median(np.abs(r1))), n_res=len(r1), obs=obs)


# --------------------------------------------- shipped rig, for comparison
ship = json.load(open(os.path.join(TAKE, "calibration/extrinsics.json")))
SHIP = {c: (np.array(ship[c]["R"]), np.array(ship[c]["t"])) for c in CAMS}


def triangulate_corners(Rl, tl, Rr, tr, Kl, Kr, pl, pr):
    Pl = Kl @ np.hstack([Rl, tl.reshape(3, 1)])
    Pr = Kr @ np.hstack([Rr, tr.reshape(3, 1)])
    X = cv2.triangulatePoints(Pl, Pr, pl.T, pr.T)
    return (X[:3] / X[3]).T


def square_score(X):
    """side lengths + diagonals + planarity of a reconstructed tag square."""
    sides = [np.linalg.norm(X[i] - X[(i + 1) % 4]) for i in range(4)]
    diags = [np.linalg.norm(X[0] - X[2]), np.linalg.norm(X[1] - X[3])]
    n = np.linalg.svd(X - X.mean(0))[2][2]
    plan = float(np.max(np.abs((X - X.mean(0)) @ n)))
    return dict(sides_mm=[round(1000 * s, 2) for s in sides],
                mean_side_mm=round(1000 * float(np.mean(sides)), 2),
                side_err_pct=round(100 * (float(np.mean(sides)) - L) / L, 2),
                diag_ratio=round(float(max(diags) / min(diags)), 4),
                diag_over_side=round(float(np.mean(diags) / np.mean(sides)), 4),  # sqrt2=1.4142
                planarity_mm=round(1000 * plan, 3))


def holdout_eval(Rl, tl, Rr, tr, Kl, Kr, label):
    """Triangulate tag-15 corners from the two views (frames where right sees 15)."""
    out = []
    for fid in FIDS:
        dl, dr = tags[fid]["left"] or {}, tags[fid]["right"] or {}
        if "15" in dl and "15" in dr:
            X = triangulate_corners(Rl, tl, Rr, tr, Kl, Kr,
                                    np.asarray(dl["15"]), np.asarray(dr["15"]))
            out.append(dict(frame=fid, **square_score(X)))
    # tag 14 too (not a holdout for either rig, but shows scale)
    out14 = []
    for fid in FIDS:
        dl, dr = tags[fid]["left"] or {}, tags[fid]["right"] or {}
        if "14" in dl and "14" in dr:
            X = triangulate_corners(Rl, tl, Rr, tr, Kl, Kr,
                                    np.asarray(dl["14"]), np.asarray(dr["14"]))
            out14.append(dict(frame=fid, **square_score(X)))
    return {label: dict(tag15_holdout=out, tag14=out14,
                        tag15_mean_side_mm=round(float(np.mean([o["mean_side_mm"] for o in out])), 2) if out else None,
                        tag14_mean_side_mm=round(float(np.mean([o["mean_side_mm"] for o in out14])), 2) if out14 else None,
                        tag14_side_std_mm=round(float(np.std([o["mean_side_mm"] for o in out14])), 2) if out14 else None)}


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    R = {}

    # (A) per-frame single-tag relative-pose spread (what the shipped rig samples once)
    spread = {}
    for scale, key in ((1.0, "assumed"), (2833.0 / 4080.0, "exif")):
        Ks = {c: K_of(c, scale) for c in CAMS}
        rels = []
        for fid in FIDS:
            dl, dr = tags[fid]["left"] or {}, tags[fid]["right"] or {}
            if "14" not in dl or "14" not in dr:
                continue
            sl, sr = solves(dl["14"], Ks["left"])[0], solves(dr["14"], Ks["right"])[0]
            rels.append((fid, *rel_pose(sl["R"], sl["t"], sr["R"], sr["t"])))
        Rm = rels[0][1]
        spread[key] = dict(
            n_frames=len(rels),
            frames=[r[0] for r in rels],
            rel_rot_spread_deg=round(max(rot_ang(a[1], b[1]) for a in rels for b in rels), 3),
            rel_baseline_m=[round(float(np.linalg.norm(r[2])), 4) for r in rels],
            baseline_spread_mm=round(1000 * (max(np.linalg.norm(r[2]) for r in rels)
                                             - min(np.linalg.norm(r[2]) for r in rels)), 2),
            baseline_std_mm=round(1000 * float(np.std([np.linalg.norm(r[2]) for r in rels])), 2),
            frame0001_rot_to_mean_deg=round(np.mean([rot_ang(rels[0][1], r[1]) for r in rels]), 3),
        )
    R["A_single_tag_relpose_spread"] = spread

    # (B) BA, full and holdout, at both focals
    rigs, evals = {}, {}
    for scale, key in ((1.0, "assumed"), (2833.0 / 4080.0, "exif")):
        ba = run_ba(scale, verbose=True)
        bah = run_ba(scale, drop=(("right", "15"),), verbose=True)
        rigs[key] = ba
        Ks = ba["K"]
        (Rl, tl), (Rr, tr) = ba["cam"]["left"], ba["cam"]["right"]
        (hRl, htl), (hRr, htr) = bah["cam"]["left"], bah["cam"]["right"]
        evals.update(holdout_eval(Rl, tl, Rr, tr, Ks["left"], Ks["right"], f"BA_full_{key}"))
        evals.update(holdout_eval(hRl, htl, hRr, htr, Ks["left"], Ks["right"], f"BA_holdout_{key}"))
        evals.update(holdout_eval(*SHIP["left"], *SHIP["right"], Ks["left"], Ks["right"], f"SHIPPED_{key}"))
        R[f"BA_{key}"] = dict(
            rms_init_px=round(ba["rms_init"], 3), rms_px=round(ba["rms"], 3),
            median_abs_resid_px=round(ba["med_abs"], 3), n_residuals=ba["n_res"],
            rot_vs_shipped_deg={c: round(rot_ang(SHIP[c][0], ba["cam"][c][0]), 3) for c in CAMS},
            cam_centre_shift_mm={c: round(1000 * float(np.linalg.norm(
                (-SHIP[c][0].T @ SHIP[c][1]) - (-ba["cam"][c][0].T @ ba["cam"][c][1]))), 2) for c in CAMS},
            rel_rot_vs_shipped_deg=round(rot_ang(
                rel_pose(*SHIP["left"], *SHIP["right"])[0],
                rel_pose(*ba["cam"]["left"], *ba["cam"]["right"])[0]), 3),
            baseline_m_shipped=round(float(np.linalg.norm(rel_pose(*SHIP["left"], *SHIP["right"])[1])), 4),
            baseline_m_ba=round(float(np.linalg.norm(rel_pose(*ba["cam"]["left"], *ba["cam"]["right"])[1])), 4),
            ba_vs_holdout_rel_rot_deg=round(rot_ang(
                rel_pose(*ba["cam"]["left"], *ba["cam"]["right"])[0],
                rel_pose(*bah["cam"]["left"], *bah["cam"]["right"])[0]), 3),
            tag_layout=dict((t, dict(
                centre_m=[round(float(x), 4) for x in ba["tag"][t][1]],
                normal_deg_to_tag14=round(ang(ba["tag"][t][0] @ [0, 0, 1.], [0, 0, 1.]), 2),
                inplane_Y_deg=round(ang(ba["tag"][t][0] @ [0, 1., 0], [0, 1., 0]), 2)))
                for t in TAGS_FREE),
        )
    R["C_square_reconstruction"] = evals

    # (C) write calibration folders in the app's format
    from pose3d.calib.intrinsics import Intrinsics
    from pose3d.calib.extrinsics import Extrinsics
    from pose3d.pipeline import CalibratedRig
    from pose3d.calib.resolve import save_rig
    for scale, key in ((1.0, "assumed"), (2833.0 / 4080.0, "exif")):
        ba = rigs[key]
        intr = {c: Intrinsics(K=ba["K"][c], dist=np.zeros((1, 5)),
                              image_size=SIZE[c], source="assumed") for c in CAMS}
        rig = CalibratedRig(intr["left"], intr["right"],
                            Extrinsics(*ba["cam"]["left"]), Extrinsics(*ba["cam"]["right"]))
        d = os.path.join(OUT, f"calib_ba_{key}")
        save_rig(rig, d)
        print("wrote", d)
    # and a shipped-extrinsics-with-exif-K folder, to separate f from extrinsics
    for scale, key in ((2833.0 / 4080.0, "exif"),):
        intr = {c: Intrinsics(K=K_of(c, scale), dist=np.zeros((1, 5)),
                              image_size=SIZE[c], source="assumed") for c in CAMS}
        rig = CalibratedRig(intr["left"], intr["right"],
                            Extrinsics(*SHIP["left"]), Extrinsics(*SHIP["right"]))
        d = os.path.join(OUT, "calib_shipped_ext_exif_K")
        save_rig(rig, d)
        print("wrote", d)

    json.dump(R, open(os.path.join(OUT, "ba_report.json"), "w"), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))

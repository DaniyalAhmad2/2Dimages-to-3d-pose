"""Shared helpers: tag cache, rig building with arbitrary focals, BA."""
import json, os, sys, copy
import numpy as np, cv2
from scipy.optimize import least_squares

REPO = os.environ.get("POSE3D_REPO", "/media/athena/hd3/Projects/pose3d-tool")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))

import metrics as M  # noqa
from pose3d.calib.extrinsics import Extrinsics, marker_object_points  # noqa
from pose3d.calib.intrinsics import Intrinsics  # noqa
from pose3d.core.project import CAM_LEFT, CAM_RIGHT, CAMERAS  # noqa
from pose3d.core.skeleton import NUM_JOINTS  # noqa
from pose3d import pipeline as pl  # noqa

TAKE = os.path.join(REPO, "workspace/pose3d_projects/Imported_Session")
MARKER = 0.05
SIZE = {CAM_LEFT: (3072, 4080), CAM_RIGHT: (1536, 2048)}
DIAG = {c: float(np.hypot(*SIZE[c])) for c in CAMERAS}
F_SHIPPED = {CAM_LEFT: 4080.0, CAM_RIGHT: 2048.0}


def load_tags():
    d = json.load(open(os.path.join(HERE, "tags.json")))
    out = {}
    for name, v in d.items():
        cam = CAM_LEFT if name.startswith("left") else CAM_RIGHT
        idx = int(name.split("_")[1].split(".")[0]) - 1
        out[(cam, idx)] = {int(i): np.asarray(c, float)
                           for i, c in zip(v["ids"], v["corners"])}
    return out


def K_of(f, cam, cx=None, cy=None):
    w, h = SIZE[cam]
    return np.array([[f, 0, w / 2.0 if cx is None else cx],
                     [0, f, h / 2.0 if cy is None else cy],
                     [0, 0, 1.0]], float)


def intr_of(f, cam, dist=None, cx=None, cy=None):
    return Intrinsics(K=K_of(f, cam, cx, cy),
                      dist=np.zeros((1, 5)) if dist is None else np.asarray(dist, float).reshape(1, -1),
                      image_size=SIZE[cam], source="assumed")


def pnp_tag(corners, intr, flags=cv2.SOLVEPNP_IPPE_SQUARE):
    obj = marker_object_points(MARKER)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(corners, np.float32),
                                  intr.K, intr.dist, flags=flags)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return Extrinsics(R=R, t=tvec.ravel())


def rig_single_tag(f_left, f_right, tags, tag_id=14, frame=0,
                   dist_l=None, dist_r=None):
    """Reproduce resolve_calibration: independent IPPE_SQUARE PnP per camera on
    the first frame where both cameras see `tag_id`."""
    il, ir = intr_of(f_left, CAM_LEFT, dist_l), intr_of(f_right, CAM_RIGHT, dist_r)
    for i in range(26):
        cl = tags.get((CAM_LEFT, i), {}).get(tag_id)
        cr = tags.get((CAM_RIGHT, i), {}).get(tag_id)
        if cl is None or cr is None:
            continue
        el, er = pnp_tag(cl, il), pnp_tag(cr, ir)
        if el is not None and er is not None:
            return pl.CalibratedRig(il, ir, el, er), i
    return None, None


# ---------------------------------------------------------------- bundle
def _rt(p):
    R, _ = cv2.Rodrigues(np.asarray(p[:3], float).reshape(3, 1))
    return R, np.asarray(p[3:6], float)


def bundle(tags, f_left, f_right, free_f=False, world_tag=14, verbose=False):
    """Joint BA: 2 static camera poses + poses of the non-world tags.

    World frame = `world_tag`'s own frame. Residuals are the reprojection of
    every detected marker corner in every frame of both cameras.
    Returns dict with rig, per-camera rms, recovered focals.
    """
    obs = []           # (cam, tag, 4x2 corners)
    for (cam, i), td in sorted(tags.items()):
        for tid, c in td.items():
            obs.append((cam, tid, c))
    tag_ids = sorted({t for _, t, _ in obs} - {world_tag})

    # ---- init from left-camera PnP with the given focal
    il, ir = intr_of(f_left, CAM_LEFT), intr_of(f_right, CAM_RIGHT)
    intr = {CAM_LEFT: il, CAM_RIGHT: ir}
    poses_l = {}
    for tid in [world_tag] + tag_ids:
        es = [pnp_tag(td[tid], il) for (cam, i), td in tags.items()
              if cam == CAM_LEFT and tid in td]
        es = [e for e in es if e is not None]
        if es:
            poses_l[tid] = es[len(es) // 2]
    cam_l = poses_l[world_tag]
    er_list = [pnp_tag(td[world_tag], ir) for (cam, i), td in tags.items()
               if cam == CAM_RIGHT and world_tag in td]
    er_list = [e for e in er_list if e is not None]
    cam_r = er_list[len(er_list) // 2]

    x0 = []
    for tid in tag_ids:
        if tid in poses_l:
            # tag->world = (world->cam)^-1 (tag->cam)
            R = cam_l.R.T @ poses_l[tid].R
            t = cam_l.R.T @ (poses_l[tid].t - cam_l.t)
        else:
            R, t = np.eye(3), np.array([0.1, 0, 0])
        x0 += list(cv2.Rodrigues(R)[0].ravel()) + list(t)
    for e in (cam_l, cam_r):
        x0 += list(cv2.Rodrigues(e.R)[0].ravel()) + list(e.t)
    n_pose = len(x0)
    if free_f:
        x0 += [np.log(f_left), np.log(f_right)]
    x0 = np.asarray(x0, float)
    obj_local = marker_object_points(MARKER).astype(float)

    def unpack(x):
        tp = {world_tag: (np.eye(3), np.zeros(3))}
        for k, tid in enumerate(tag_ids):
            tp[tid] = _rt(x[6 * k:6 * k + 6])
        cl = _rt(x[6 * len(tag_ids):6 * len(tag_ids) + 6])
        cr = _rt(x[6 * len(tag_ids) + 6:n_pose])
        if free_f:
            fl, fr = np.exp(x[n_pose]), np.exp(x[n_pose + 1])
        else:
            fl, fr = f_left, f_right
        return tp, {CAM_LEFT: cl, CAM_RIGHT: cr}, {CAM_LEFT: fl, CAM_RIGHT: fr}

    def residuals(x):
        tp, cams, fs = unpack(x)
        r = []
        for cam, tid, c in obs:
            Rt, tt = tp[tid]
            Xw = obj_local @ Rt.T + tt
            Rc, tc = cams[cam]
            Xc = Xw @ Rc.T + tc
            f = fs[cam]
            w, h = SIZE[cam]
            u = f * Xc[:, 0] / Xc[:, 2] + w / 2.0
            v = f * Xc[:, 1] / Xc[:, 2] + h / 2.0
            r.append(np.stack([u, v], 1) - c)
        return np.concatenate(r).ravel()

    res = least_squares(residuals, x0, method="lm", max_nfev=20000)
    tp, cams, fs = unpack(res.x)
    r = residuals(res.x).reshape(-1, 2)
    per = {}
    k = 0
    for cam, tid, c in obs:
        per.setdefault(cam, []).append(r[k:k + 4])
        k += 4
    rms = {c: float(np.sqrt(np.mean(np.sum(np.concatenate(v) ** 2, 1))))
           for c, v in per.items()}
    intr_out = {c: intr_of(fs[c], c) for c in CAMERAS}
    ext = {c: Extrinsics(R=cams[c][0], t=cams[c][1]) for c in CAMERAS}
    rig = pl.CalibratedRig(intr_out[CAM_LEFT], intr_out[CAM_RIGHT],
                           ext[CAM_LEFT], ext[CAM_RIGHT])
    return {"rig": rig, "rms_px": rms, "f": fs, "tag_poses": tp,
            "cost": float(res.cost), "n_obs": len(obs), "success": bool(res.success)}


# ---------------------------------------------------------------- metrics
def evaluate_rig(rig, take, label=""):
    """Body-keypoint metrics for a candidate rig, all recomputed from the
    stored 2D keypoints (raw triangulation, no cross-view gate)."""
    from pose3d.geometry.triangulate import epipolar_distance
    kp = take["kp2d"]
    L = np.asarray(kp[CAM_LEFT], float)
    R = np.asarray(kp[CAM_RIGHT], float)
    epi = []
    for t in range(L.shape[0]):
        for j in range(NUM_JOINTS):
            e = epipolar_distance(L[t, j], R[t, j], rig.intr[CAM_LEFT],
                                  rig.intr[CAM_RIGHT], rig.ext[CAM_LEFT],
                                  rig.ext[CAM_RIGHT])
            if np.isfinite(e):
                epi.append(e)
    epi = np.asarray(epi)
    poses = M.triangulate(kp, rig)
    bs = M.bone_length_stats(poses)
    # cross-view gate outcome with the shipped threshold rule
    thr = pl.epipolar_threshold(rig)
    # subject geometry
    fin = np.isfinite(poses).all(2)
    heights = []
    for t in range(poses.shape[0]):
        p = poses[t][fin[t]]
        if p.shape[0] >= 2:
            d = np.linalg.norm(p[:, None] - p[None], axis=2)
            heights.append(float(d.max()))
    cen = np.nanmean(poses.reshape(-1, 3), 0)
    dl = float(np.linalg.norm(rig.ext[CAM_LEFT].camera_center - cen))
    dr = float(np.linalg.norm(rig.ext[CAM_RIGHT].camera_center - cen))
    return {
        "label": label,
        "f_left": float(rig.intr[CAM_LEFT].K[0, 0]),
        "f_right": float(rig.intr[CAM_RIGHT].K[0, 0]),
        "epi_median_px": float(np.median(epi)),
        "epi_p90_px": float(np.percentile(epi, 90)),
        "epi_max_px": float(np.max(epi)),
        "epi_thr_px": float(thr),
        "epi_frac_over_thr": float(np.mean(epi > thr)),
        "bone_cv_median_pct": bs["median_cv_pct"],
        "bone_cv_max_pct": bs["max_cv_pct"],
        "bone_std_mm_median": float(np.median(
            [b["std_m"] for b in bs["bones"].values() if np.isfinite(b["std_m"])]) * 1000),
        "sym_asym_pct": {k: v["asym_pct"] for k, v in bs["symmetry"].items()},
        "sym_asym_mean_pct": float(np.mean([v["asym_pct"] for v in bs["symmetry"].values()
                                            if np.isfinite(v["asym_pct"])])),
        "subject_span_m": float(np.median(heights)) if heights else float("nan"),
        "dist_left_m": dl, "dist_right_m": dr,
        "baseline_m": float(np.linalg.norm(rig.ext[CAM_LEFT].camera_center
                                           - rig.ext[CAM_RIGHT].camera_center)),
        "n_epi": int(epi.size),
    }

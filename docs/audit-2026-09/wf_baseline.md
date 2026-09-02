## BEFORE numbers — shipped pipeline on the real client take

`workspace/pose3d_projects/Imported_Session`, 26 frames, shipped `calibration/` (K = [[4080,0,1536],[0,4080,2040]], zero distortion; single-tag extrinsics). 0 corrections in the project.

**Sanity checks (all passed):**
- stored `fitted3d` vs `pipeline.fit_project(deepcopy, smooth=True)`: **max abs diff = 0.0 m**, NaN masks identical. `fitted` and `refit` are bit-identical, so they share a column below.
- re-triangulating the stored `kp2d` with `triangulate_points` reproduces the stored `pose3d` exactly (max abs diff 0.0 m, identical NaN mask) — the stored 2D is post-validation 2D.
- `pipeline.validate_cross_view` on the stored 2D would drop **0 of 390** observations (threshold 71.5 px; measured max body epipolar error is 29.4 px). The cross-view gate is inert on this take.
- coverage: left 2D 100.0 % present, right 2D 99.5 %, raw 3D joints 99.5 % (2 NaN joints total: RIGHT_KNEE @ frame 0012, LEFT_ANKLE @ frame 0021). `fitted`/`refit` report 100 % because the EMA fills NaNs from the previous frame.

#### T1 — Bone length across the 26 frames (rigid mannequin: perfect pipeline = 0)

| bone | raw med (m) | raw std (m) | raw CV% | fitted=refit med | std | CV% | refit-nosmooth med | std | CV% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spine (pelvis-neck) | 0.0411 | 0.0020 | 4.9 | 0.0409 | 0.0001 | 0.4 | 0.0411 | 0.0001 | 0.1 |
| neck-head | 0.0219 | 0.0021 | 9.5 | 0.0217 | 0.0001 | 0.7 | 0.0219 | 0.0001 | 0.3 |
| neck-Lshoulder | 0.0160 | 0.0008 | 5.1 | 0.0159 | 0.0001 | 0.5 | 0.0160 | 0.0000 | 0.3 |
| neck-Rshoulder | 0.0160 | 0.0008 | 5.0 | 0.0160 | 0.0001 | 0.5 | 0.0160 | 0.0000 | 0.2 |
| L upper arm | 0.0224 | 0.0012 | 5.4 | 0.0220 | 0.0011 | **5.1** | 0.0224 | 0.0000 | 0.2 |
| R upper arm | 0.0225 | 0.0009 | 3.9 | 0.0224 | 0.0001 | 0.6 | 0.0225 | 0.0000 | 0.1 |
| L forearm | 0.0214 | 0.0011 | 5.0 | 0.0208 | 0.0013 | **6.2** | 0.0214 | 0.0000 | 0.1 |
| R forearm | 0.0200 | 0.0011 | 5.5 | 0.0198 | 0.0001 | 0.7 | 0.0200 | 0.0000 | 0.1 |
| pelvis-Lhip | 0.0083 | 0.0006 | 7.2 | 0.0083 | 0.0001 | 0.8 | 0.0083 | 0.0000 | 0.3 |
| pelvis-Rhip | 0.0083 | 0.0006 | 7.2 | 0.0082 | 0.0001 | 1.3 | 0.0083 | 0.0000 | 0.4 |
| L thigh | 0.0372 | 0.0018 | 4.9 | 0.0367 | 0.0006 | 1.6 | 0.0372 | 0.0000 | 0.1 |
| R thigh | 0.0355 | 0.0021 | 5.8 | 0.0347 | 0.0011 | 3.1 | 0.0355 | 0.0000 | 0.1 |
| L shin | 0.0371 | 0.0012 | 3.1 | 0.0359 | 0.0009 | 2.4 | 0.0371 | 0.0000 | 0.1 |
| R shin | 0.0368 | 0.0021 | 5.7 | 0.0352 | 0.0017 | 4.7 | 0.0368 | 0.0000 | 0.1 |
| **median CV over bones** |  |  | **5.3** |  |  | **1.0** |  |  | **0.1** |
| **max CV over bones** |  |  | **9.5** |  |  | **6.2** |  |  | **0.4** |

Raw std is 0.6–2.1 mm on 8–37 mm bones. Note the smoother *re-breaks* bones the fit had just fixed: L upper arm 5.1 % and L forearm 6.2 % CV in the shipped `fitted3d` vs 0.2 %/0.1 % without smoothing. Worst frames (fitted vs nosmooth length, m): L upper arm 0013 0.0177/0.0223, 0010 0.0195/0.0223, 0014 0.0197/0.0224; L forearm 0013 0.0173/0.0214, 0010 0.0175/0.0213 — i.e. the shipped left forearm is **19 % short** in frame 0013. Neither of those frames contains a NaN joint, so this is pure EMA chord-shortening across a swinging limb.

#### T2 — Left/right limb symmetry (same mannequin limb on both sides)

| limb | raw L (m) | raw R (m) | raw L/R | raw asym% | fitted L/R | fitted asym% | nosmooth L/R | nosmooth asym% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thigh | 0.0372 | 0.0355 | 1.047 | 4.6 | 1.058 | 5.6 | 1.047 | 4.6 |
| shin | 0.0371 | 0.0368 | 1.007 | 0.7 | 1.021 | 2.1 | 1.007 | 0.7 |
| upper arm | 0.0224 | 0.0225 | 0.995 | 0.5 | 0.983 | 1.7 | 0.995 | 0.5 |
| forearm | 0.0214 | 0.0200 | 1.070 | 6.7 | 1.052 | 5.0 | 1.070 | 6.7 |

#### T3 — Body-keypoint epipolar (Sampson) error, px — depends only on kp2d + rig, so identical for all four sources

Threshold in use 71.5 px (`_EPI_THR_FRAC` 0.014 × 5107 px diagonal), 388 both-view pairs.

| joint | n | median px | p90 px | max px |
|---|---:|---:|---:|---:|
| HEAD | 26 | 5.1 | 10.5 | 20.2 |
| NECK | 26 | 1.5 | 3.9 | 4.1 |
| LEFT_SHOULDER | 26 | 5.1 | 8.6 | 11.2 |
| RIGHT_SHOULDER | 26 | 3.0 | 6.7 | 8.9 |
| LEFT_ELBOW | 26 | 4.1 | 10.0 | 29.4 |
| RIGHT_ELBOW | 26 | 3.8 | 7.0 | 8.6 |
| LEFT_WRIST | 26 | 7.6 | 16.2 | 19.7 |
| RIGHT_WRIST | 26 | 4.7 | 7.7 | 17.7 |
| PELVIS | 26 | 4.8 | 10.4 | 15.8 |
| LEFT_HIP | 26 | 6.0 | 9.4 | 13.5 |
| RIGHT_HIP | 26 | 9.1 | 16.9 | 20.2 |
| LEFT_KNEE | 26 | 5.7 | 10.6 | 23.7 |
| RIGHT_KNEE | 25 | 4.9 | 10.8 | 12.7 |
| LEFT_ANKLE | 25 | 10.4 | 16.0 | 28.8 |
| RIGHT_ANKLE | 26 | 7.5 | 18.3 | 25.8 |
| **ALL** | 388 | **4.9** | 12.5 | 29.4 |

ALL median = **0.096 % of the image diagonal**; **0.0 %** of pairs exceed the pipeline threshold.

#### T4 — Reprojection error, px (median per joint per camera)

| joint | raw L | raw R | fitted=refit L | fitted R | nosmooth L | nosmooth R |
|---|---:|---:|---:|---:|---:|---:|
| HEAD | 5.23 | 2.94 | 30.68 | 20.98 | 7.65 | 4.05 |
| NECK | 1.56 | 0.87 | 20.24 | 15.43 | 6.05 | 3.54 |
| LEFT_SHOULDER | 5.10 | 2.93 | 20.76 | 14.49 | 4.82 | 3.42 |
| RIGHT_SHOULDER | 3.11 | 1.72 | 19.40 | 16.62 | 5.86 | 3.30 |
| LEFT_ELBOW | 4.20 | 2.38 | 24.90 | 15.27 | 5.40 | 2.55 |
| RIGHT_ELBOW | 4.00 | 2.13 | 14.82 | 17.78 | 5.14 | 2.50 |
| LEFT_WRIST | 7.91 | 4.36 | 42.97 | 23.35 | 8.89 | 5.12 |
| RIGHT_WRIST | 4.99 | 2.63 | 19.58 | 18.49 | 5.14 | 2.87 |
| PELVIS | 4.94 | 2.73 | 15.94 | 14.17 | 8.64 | 3.76 |
| LEFT_HIP | 6.09 | 3.41 | 18.85 | 11.99 | 9.32 | 4.46 |
| RIGHT_HIP | 9.47 | 5.21 | 13.75 | 15.87 | 9.58 | 7.17 |
| LEFT_KNEE | 5.86 | 3.24 | 22.25 | 18.37 | 7.81 | 4.02 |
| RIGHT_KNEE | 5.05 | 2.77 | 31.13 | 22.10 | 9.30 | 5.02 |
| LEFT_ANKLE | 11.13 | 5.87 | 26.03 | 30.91 | 12.24 | 5.10 |
| RIGHT_ANKLE | 7.94 | 4.23 | 49.23 | 30.42 | 8.74 | 5.43 |
| **ALL median** | **5.11** | **2.78** | **21.65** | **17.18** | **7.38** | **4.01** |
| **ALL max** | 29.70 | 16.90 | 167.82 | 119.61 | 44.49 | 20.98 |

The shipped `fitted3d` reprojects **4.2× worse than raw** in the left view (21.65 vs 5.11 px) and **6.2× worse** in the right (17.18 vs 2.78 px), with a worst single joint of 167.8 px. Bone fitting alone costs only 5.11 → 7.38 px; the EMA smoother costs the remaining 7.38 → 21.65 px.

#### T5 — Retarget: bundled character's read-back joint vs the captured joint, % of body height

Posed exactly as the app does: `R = de_tilt_matrix(sequence_up(poses))`, `up = poses @ R.T`, `Character().fit_to_subject(up)` once, `Character.posed_joints(up[t], valid[t])` per frame. 26/26 frames posed for every source.

| joint | raw med% | raw max% | fitted med% | fitted max% | nosmooth med% | nosmooth max% |
|---|---:|---:|---:|---:|---:|---:|
| HEAD | 12.8 | 14.6 | 12.6 | 12.7 | 12.9 | 13.0 |
| NECK | 4.8 | 7.1 | 4.1 | 4.5 | 4.8 | 4.9 |
| LEFT_SHOULDER | 0.8 | 3.0 | 1.0 | 1.5 | 0.8 | 1.4 |
| RIGHT_SHOULDER | 0.8 | 2.7 | 1.0 | 1.6 | 0.7 | 1.3 |
| LEFT_ELBOW | 1.9 | 2.9 | 1.9 | 2.4 | 1.7 | 2.3 |
| RIGHT_ELBOW | 1.9 | 3.6 | 2.2 | 2.4 | 1.8 | 2.0 |
| LEFT_WRIST | 1.3 | 3.5 | 1.3 | 3.7 | 1.1 | 1.7 |
| RIGHT_WRIST | 0.7 | 2.8 | 0.7 | 1.0 | 0.3 | 0.5 |
| PELVIS | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| LEFT_HIP | 2.4 | 4.4 | 2.1 | 3.8 | 2.4 | 4.7 |
| RIGHT_HIP | 3.2 | 4.3 | 3.5 | 5.1 | 3.6 | 5.6 |
| LEFT_KNEE | 1.9 | 4.3 | 1.1 | 3.6 | 1.4 | 4.2 |
| RIGHT_KNEE | 4.1 | 7.3 | 3.4 | 7.9 | 2.5 | 6.1 |
| LEFT_ANKLE | 1.2 | 4.3 | 1.0 | 2.0 | 1.5 | 3.2 |
| RIGHT_ANKLE | 1.6 | 5.9 | 0.7 | 3.1 | 1.0 | 3.2 |
| **ALL** | **1.8** | p90 5.4 | **1.5** | p90 4.4 | **1.7** | p90 4.8 |

| | raw | fitted=refit | refit-nosmooth |
|---|---:|---:|---:|
| subject height, z-extent of the de-tilted pose (m) | **0.1180** | 0.1178 | 0.1188 |
| uniform character scale from `fit_to_subject` | 111.10 | 113.62 | 111.10 |
| ALL median distance (m) | 0.0021 | 0.0018 | 0.0020 |

Skinning is not the bottleneck: every joint except HEAD/NECK lands within ~1–4 % of body height. HEAD is a **systematic 12.6–12.9 %** (≈15 mm) offset in every source — the canonical HEAD joint is the *nose*, and the rig's head-bone point is not the nose, so this is a definitional offset, not per-frame noise (median ≈ max).

#### T6 — Bend-plane normal, captured vs character (deg)

| chain | raw med | raw max | fitted med | fitted max | nosmooth med | nosmooth max |
|---|---:|---:|---:|---:|---:|---:|
| L elbow | 1.4 | 7.8 | 2.1 | 5.0 | 1.3 | 5.7 |
| R elbow | 2.8 | 10.3 | 4.0 | 5.1 | 2.7 | 5.8 |
| L knee | 4.0 | 18.7 | 4.3 | 14.6 | 4.8 | 16.0 |
| R knee | 5.6 | 16.6 | 5.6 | 14.9 | 5.8 | 16.6 |
| **ALL** | **3.1** | 18.7 | **3.6** | 14.9 | **3.1** | 16.6 |

#### T7 — Displacement each post-triangulation stage adds (mm)

| joint | \|fitted3d − pose3d\| mean | max | \|refit − refit-nosmooth\| mean | max |
|---|---:|---:|---:|---:|
| HEAD | 5.96 | 14.23 | 5.63 | 13.73 |
| NECK | 4.96 | 12.41 | 4.68 | 13.06 |
| LEFT_SHOULDER | 4.79 | 12.16 | 4.62 | 12.28 |
| RIGHT_SHOULDER | 5.30 | 13.94 | 4.93 | 13.92 |
| LEFT_ELBOW | 5.40 | 13.82 | 5.56 | 15.55 |
| RIGHT_ELBOW | 5.07 | 15.31 | 5.11 | 15.38 |
| LEFT_WRIST | 8.55 | 25.47 | 8.48 | 25.14 |
| RIGHT_WRIST | 5.93 | 17.35 | 5.76 | 17.19 |
| PELVIS | 4.44 | 14.29 | 4.28 | 14.30 |
| LEFT_HIP | 4.40 | 14.80 | 4.22 | 13.89 |
| RIGHT_HIP | 4.37 | 13.42 | 4.38 | 14.50 |
| LEFT_KNEE | 5.92 | 15.50 | 5.61 | 14.97 |
| RIGHT_KNEE | 7.95 | 20.56 | 7.78 | 20.51 |
| LEFT_ANKLE | 8.53 | 19.06 | 8.57 | 19.37 |
| RIGHT_ANKLE | 11.85 | 25.65 | 11.62 | 25.65 |
| **ALL** | **6.22** | 25.65 | **6.07** | 25.65 |

**6.07 of the 6.22 mm (98 %)** that separate the shipped `fitted3d` from the raw triangulation come from the EMA smoother alone; the bone-length fit contributes ~0.15 mm net. As a fraction of the 0.118 m body height the shipped pose is displaced from the measurement by **5.27 % mean, 10.68 % p90, 21.73 % max**.

#### T8 — Rig geometry and scale (measured, client take)

| quantity | client take | Panoptic demo_project |
|---|---:|---:|
| camera baseline | 0.559 m | 390.3 (cm) |
| optical-axis convergence angle | 50.4 deg | 89.2 deg |
| camera→subject distance L / R | 0.556 / 0.510 m | 255.0 / 277.0 |
| ray parallax at the joints, median (min–max) | 62.5 deg (52.7–76.7) | 91.6 deg (79.7–119.5) |
| assumed focal | 4080 px | 1633 px |
| depth error per 1 px of 2D error | **0.136 mm** | 102 mm (cm-unit scene) |
| per-frame max joint-pair span, median | **0.1277 m** | — |
| nose-to-ankle distance, median (n=51) | **0.1197 m** | — |
| frame-to-frame raw joint displacement | median 9.4 mm, mean 12.4, p90 26.8, max 69.6 | — |

The stereo geometry is well conditioned (62 deg parallax); 1 px of 2D error is only 0.14 mm of depth. The 0.6–2.1 mm raw bone-length std therefore corresponds to roughly 5–15 px of 2D keypoint error, matching the measured 4.9 px median / 12.5 px p90 epipolar error. Triangulation conditioning is **not** the limiting factor.

#### T9 — Panoptic reference (`data/demo_project`, GT calibration, RTMPose 2D, source=raw, 10 frames, units = Panoptic cm)

| bone | n | median | std | CV% |
|---|---:|---:|---:|---:|
| spine (pelvis-neck) | 10 | 61.421 | 1.629 | 2.7 |
| neck-head | 10 | 25.617 | 1.961 | 7.7 |
| neck-Lshoulder | 10 | 18.855 | 0.555 | 2.9 |
| neck-Rshoulder | 10 | 18.536 | 0.581 | 3.1 |
| L upper arm | 3 | 33.099 | 2.981 | 9.0 |
| R upper arm | 10 | 28.973 | 1.016 | 3.5 |
| L forearm | 3 | 21.850 | 1.864 | 8.5 |
| R forearm | 10 | 25.991 | 2.103 | 8.1 |
| pelvis-Lhip | 10 | 9.524 | 0.357 | 3.7 |
| pelvis-Rhip | 10 | 9.472 | 0.311 | 3.3 |
| L thigh | 10 | 38.517 | 0.924 | 2.4 |
| R thigh | 10 | 41.098 | 1.177 | 2.9 |
| L shin | 0 | — | — | — |
| R shin | 0 | — | — | — |
| **median / max CV** |  |  |  | **3.4 / 9.0** |

| limb | L | R | L/R | asym% |
|---|---:|---:|---:|---:|
| thigh | 38.517 | 41.098 | 0.937 | 6.5 |
| shin | — | — | — | — |
| upper arm | 33.099 | 28.973 | 1.142 | 13.3 |
| forearm | 21.850 | 25.991 | 0.841 | 17.3 |

Epipolar ALL median **6.8 px** (p90 20.9) = **0.310 % of a 2203 px diagonal**; reprojection ALL median L 5.31 / R 4.34 px; retarget ALL median **5.0 %** of height (p90 9.8 %); bend-plane ALL median **18.5 deg** (max 48.9, elbows only — knees/ankles absent).

**Reference reading:** with *ground-truth* calibration and the same RTMPose detector, the raw bone-length CV is 3.4 % median and the epipolar error is 0.31 % of the image diagonal. The client take's raw pipeline is **5.3 % CV and 0.096 % of diagonal** — i.e. its 2D–geometry consistency is *better* than the known-good reference (see caveats: a human moves and the reference has only 3 valid left-arm frames), and its retarget error (1.8 % of height) is much better than the reference's 5.0 %. Nothing in the client take's *geometry* is anomalous relative to the reference; the anomaly is what happens **after** triangulation.


KEY_NUMBERS:
{"take":{"path":"workspace/pose3d_projects/Imported_Session","frames":26,"corrections":0,"subject_height_m_raw":0.1180,"subject_height_m_fitted":0.1178,"subject_height_m_refit_nosmooth":0.1188,"nose_to_ankle_median_m":0.1197,"per_frame_max_joint_span_median_m":0.1277},"sanity":{"fitted3d_vs_refit_max_abs_m":0.0,"fitted_equals_refit":true,"retriangulate_stored_kp2d_vs_stored_pose3d_max_abs_m":0.0,"validate_cross_view_would_drop":0,"validate_cross_view_total_obs":390,"epipolar_threshold_px":71.5},"bone_cv_pct":{"raw":{"median":5.3,"max":9.5},"fitted_refit":{"median":1.0,"max":6.2},"refit_nosmooth":{"median":0.1,"max":0.4}},"bone_std_m_raw_range":[0.0006,0.0021],"symmetry_asym_pct":{"raw":{"thigh":4.6,"shin":0.7,"upper_arm":0.5,"forearm":6.7},"fitted":{"thigh":5.6,"shin":2.1,"upper_arm":1.7,"forearm":5.0}},"epipolar_px":{"all_median":4.9,"all_p90":12.5,"all_max":29.4,"n_pairs":388,"pct_of_image_diag":0.096,"image_diag_px":5107,"frac_over_threshold":0.0,"worst_joints":{"LEFT_ANKLE":10.4,"RIGHT_HIP":9.1,"LEFT_WRIST":7.6,"RIGHT_ANKLE":7.5},"best_joint":{"NECK":1.5}},"reprojection_median_px":{"raw":{"left":5.11,"right":2.78},"fitted_refit":{"left":21.65,"right":17.18},"refit_nosmooth":{"left":7.38,"right":4.01}},"reprojection_max_px":{"raw":{"left":29.70,"right":16.90},"fitted_refit":{"left":167.82,"right":119.61},"refit_nosmooth":{"left":44.49,"right":20.98}},"retarget_pct_height":{"raw":{"all_median":1.8,"all_p90":5.4},"fitted_refit":{"all_median":1.5,"all_p90":4.4},"refit_nosmooth":{"all_median":1.7,"all_p90":4.8},"HEAD_median":{"raw":12.8,"fitted":12.6,"nosmooth":12.9},"NECK_median":{"raw":4.8,"fitted":4.1,"nosmooth":4.8}},"retarget_median_m":{"raw":0.0021,"fitted_refit":0.0018,"refit_nosmooth":0.0020},"character_scale":{"raw":111.10,"fitted_refit":113.62,"refit_nosmooth":111.10},"bend_plane_deg":{"raw":{"all_median":3.1,"all_max":18.7,"L_elbow":1.4,"R_elbow":2.8,"L_knee":4.0,"R_knee":5.6},"fitted_refit":{"all_median":3.6,"all_max":14.9,"L_elbow":2.1,"R_elbow":4.0,"L_knee":4.3,"R_knee":5.6},"refit_nosmooth":{"all_median":3.1,"all_max":16.6}},"stage_deltas_mm":{"fitted3d_minus_pose3d":{"mean":6.22,"max":25.65,"worst_joint":"RIGHT_ANKLE 11.85"},"refit_minus_refit_nosmooth":{"mean":6.07,"max":25.65,"worst_joint":"RIGHT_ANKLE 11.62"},"smoother_share_of_total_pct":98},"smoother_displacement_pct_of_height":{"mean":5.27,"p90":10.68,"max":21.73},"smoother_breaks_bones":{"L_forearm_frame_0013_m":0.0173,"L_forearm_nosmooth_m":0.0214,"shortening_pct":19,"L_upper_arm_frame_0013_m":0.0177,"L_upper_arm_nosmooth_m":0.0223},"rig_geometry":{"baseline_m":0.559,"convergence_deg":50.4,"dist_left_m":0.556,"dist_right_m":0.510,"parallax_median_deg":62.5,"parallax_range_deg":[52.7,76.7],"focal_px":4080,"depth_error_per_px_mm":0.136},"frame_to_frame_raw_motion_mm":{"median":9.36,"mean":12.39,"p90":26.83,"max":69.62},"coverage_pct":{"left_kp2d":100.0,"right_kp2d":99.5,"raw_3d_joints":99.5,"raw_nan_joints":["frame 0012 RIGHT_KNEE","frame 0021 LEFT_ANKLE"]},"panoptic_reference_raw":{"path":"data/demo_project","frames":10,"units":"cm","bone_cv_pct_median":3.4,"bone_cv_pct_max":9.0,"epipolar_median_px":6.8,"epipolar_p90_px":20.9,"epipolar_pct_of_diag":0.310,"image_diag_px":2203,"reprojection_median_px":{"left":5.31,"right":4.34},"retarget_all_median_pct_height":5.0,"retarget_all_p90_pct_height":9.8,"bend_plane_median_deg":18.5,"bend_plane_max_deg":48.9,"symmetry_asym_pct":{"thigh":6.5,"upper_arm":13.3,"forearm":17.3},"baseline":390.3,"parallax_median_deg":91.6},"headline":["fitted3d == pipeline.fit_project(deepcopy,smooth=True) bit-for-bit (max diff 0.0 m)","the EMA smoother contributes 6.07 of the 6.22 mm (98%) that separate the shipped pose from the measurement, = 5.3% of body height mean / 21.7% max","reprojection of the shipped fitted3d is 21.65/17.18 px vs 5.11/2.78 px raw: 4.2x/6.2x worse","the smoother re-breaks bones the fit had just enforced: L forearm 19% short in frame 0013 (0.0173 vs 0.0214 m), CV 6.2% vs 0.1%","body epipolar error is only 4.9 px median = 0.096% of the image diagonal, and 0/388 pairs exceed the 71.5 px gate: the single-tag calibration is self-consistent on this take and validate_cross_view is inert","stereo geometry is well conditioned (0.559 m baseline, 62.5 deg parallax, 0.136 mm depth error per px): triangulation conditioning is not the bottleneck","raw bone-length CV 5.3% median is driven by ~5-15 px of 2D detector error, consistent with the measured epipolar spread","skinning is accurate: every character joint except HEAD/NECK lands within 1-4% of body height; HEAD is a systematic 12.6-12.9% offset because canonical HEAD = nose, not the rig head point","reconstructed nose-to-ankle is 0.120 m on a mannequin described as ~20 cm tall: possible absolute-scale error of ~0.7x, unverified"]}

CAVEATS:
SCOPE / VALIDITY
1. `fitted` and `refit` are bit-identical (max abs diff exactly 0.0 m, identical NaN masks), so they are one column everywhere. The stored project was written by the current code with default `alpha=0.6`, `smooth=True`; no drift, no stale file. Likewise re-triangulating the stored `kp2d` reproduces the stored `pose3d` exactly, so the stored 2D is already the post-`validate_cross_view` 2D — I could not measure how many observations the gate dropped at import time, only that re-running it on the stored 2D drops 0 more.
2. The epipolar and reprojection numbers assume the SHIPPED calibration is the reference frame. Epipolar error tests self-consistency of (K, R, t) against the 2D, not correctness: a wrong focal that is wrong in the SAME way in both cameras (both use f=4080) largely cancels in F, so a low epipolar error does NOT clear the focal-length assumption. It only says the two views agree about correspondences. Treat T3 as a null result for the correspondence/rectification question, not as calibration validation.
3. Reprojection error (T4) is deliberately a weak metric for `raw` — a two-view DLT point reprojects near its own observations by construction, so raw's 5.11/2.78 px is close to a lower bound and mostly reflects the 2D residual after the least-squares. Its value is entirely in the COMPARISON: raw → nosmooth → fitted (5.11 → 7.38 → 21.65 px left). Do not quote raw reprojection as an accuracy figure.
4. Retarget error (T5/T6) is measured against the CAPTURED joints of the same source, so it is a "how faithfully does the skinning reproduce this input" metric, not an absolute accuracy metric. It cannot see error that is already in the input. That is why the shipped `fitted` scores BEST on retargeting (1.5 %) while being worst on reprojection: the character faithfully follows a pose that has drifted off the keypoints.
5. HEAD's 12.6–12.9 % retarget error is a definitional offset (canonical HEAD = nose; the rig's head joint is the head-bone point), evidenced by median ≈ max in all three sources. It is not per-frame noise and should not be counted as retargeting failure. NECK's 4.1–4.8 % is similarly partly definitional (2D shoulder-midpoint vs the rig's neck bone).
6. `head2d`/`head3d` are ABSENT from this project (it predates the face-keypoint feature), so `retarget_error` posed the character with `head_pts=None` — the head-orientation path in `character.py` was never exercised. Any head-orientation regression is invisible in these numbers.
7. Body height for the % figures is the z-extent of the de-tilted pose (0.118 m), not the full 3D span. The per-frame max joint-pair span is 0.128 m and nose-to-ankle 0.120 m. The context's "~0.19 m" is the bounding box over all frames AND all joints, which includes the mannequin's motion — not a per-frame size. If the mannequin really is ~20 cm tall, the reconstruction is under-scaled by roughly 0.7×; I did NOT verify the true mannequin dimensions and cannot attribute the discrepancy (note: a too-large assumed focal would over-scale, not under-scale, so the EXIF-focal story does not straightforwardly explain it). Treat this as an open question, not a finding.
8. Panoptic comparison is NOT apples-to-apples: `data/demo_project` is 10 frames of a MOVING HUMAN (bone lengths genuinely vary with soft tissue and detector bias), in Panoptic cm units, with 90 deg parallax and a 390-unit baseline; both shins and both ankles are entirely missing and the left arm has only 3 valid frames, so its "median CV 3.4 %" rests on 12 of 14 bones and its L/R asymmetry (13–17 %) is dominated by that 3-frame left arm. It is a sanity reference for "what a good rig's order of magnitude looks like", not a target. Its bend-plane median of 18.5 deg is elbows-only.
9. The client take's raw metrics look BETTER than the Panoptic reference on every geometric axis. Do not read that as "the client's calibration is better than ground truth" — the mannequin is rigid and small, the cameras never moved, and the epipolar test cancels shared-focal error (caveat 2). It does mean the shipped calibration is internally consistent enough that the client's complaint is unlikely to be explained by cross-view geometry alone.
10. Everything here is on ONE take with 26 frames and 0 user corrections; per-joint medians rest on 25–26 samples and the max columns on a single frame each. The manual-correction re-solve path (`pose3d/ui/model.py`) is untested by these numbers.
11. `metrics.py` needs the repo root on `sys.path`; it inserts `os.getcwd()` (or `$POSE3D_REPO`, or the first ancestor holding `pose3d/__init__.py`). Run it with cwd = the repo root as in the usage line. It imports `Character`, which loads the bundled rig asset — if that asset is missing, `retarget_error` raises and `--no-retarget` is the escape hatch. It never imports `pose3d.app` (that pulls in PySide6 at module scope, `pose3d/app.py:16`); `load_rig` duplicates `_load_rig`'s body from `pose3d/app.py:81`.
12. Nothing in the repository was modified; all outputs are under the scratchpad folder (`metrics.py`, `run_all.py`, `compose.py`, `rig_geom.py`, `scale_jitter.py`, `why_arm.py`, `report.md`, `combined.md`, `all.json`, `client_{raw,fitted,refit,refit-nosmooth}.json`).
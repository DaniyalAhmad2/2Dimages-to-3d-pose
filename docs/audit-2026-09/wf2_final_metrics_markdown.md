| metric | baseline (shipped) | best prototype result seen (finding) | target after the plan |
|---|---|---|---|
| Delivered pose vs its own bone-fitted measurement, median / max | 4.92 mm / 25.65 mm = 4.2 % / **21.8 %** of the 117.8 mm body height | 0.00 mm with `smooth=False` (F01) | **0.00 mm** |
| Share of post-triangulation displacement from the smoother | **98 %** of 6.22 mm | — (F01) | n/a — smoother removed |
| Lag directionality, cos(displacement, motion) | **−0.976** median, negative in 95.4 % of 371 samples | 0 with a zero-phase filter (F01) | **≥ −0.1** (CI assertion) |
| Reprojection of the delivered pose, median L / R | **21.65 / 17.18 px** (2.8 % / 4.2 % of figure height) | 7.38 / 4.01 px unsmoothed; raw floor 5.11 / 2.78 (F01) | **≤ 7.4 / 4.1 px** (0.95 % / 0.99 %), ratio to raw **≤ 1.5** (today 4.24 / 6.17) |
| Bone-length CV on the rigid mannequin, delivered pose, median / max | 1.04 % / **6.35 %** | 0.14 % / 0.37 % unsmoothed (F01) | **≤ 0.15 % / ≤ 0.40 %** |
| Raw bone-length CV (the detector floor, unchanged by this plan) | 5.3 % median / 9.5 % max | 3.4 / 9.0 on ground-truth Panoptic (baseline T9) | **unchanged** — gate at ≤ 6.0 / ≤ 10.5 |
| Left forearm length, frame 0013 | 0.0173 m = **19 % short** | 0.0214 m (F01) | **0.0214 m** |
| Invented joints shown as measured | 2 (F 0012 `RIGHT_KNEE`, F 0021 `LEFT_ANKLE`), 20.21 / 38.50 mm = 17.2 / 32.7 % of height off | 0 with `bonefit.py:140` deleted (F09) | **0 invented; 2 flagged interpolations**, each < 2 % of height from its neighbours |
| Zero-pixel drag jump | 4.92 mm median / **25.65 mm max**, 15 of 15 joints move | 0 once the drag path == the batch path (F10) | **< 0.05 mm** |
| Stale derived NECK/PELVIS after a 60 px shoulder drag | 30.0 px 2D in 208/208 cases; 3.31 mm median / 10.66 max (2.8 / 9.0 % of height) | 0 with dependents re-derived (F33) | **≤ 0.2 mm** |
| Sparse-frame bone fit | 746.78 ms (`trf`) | 21.47 ms (`lm`) (F44) | **≤ 40 ms** |
| Limb roll vs the captured bend plane — `upper_arm.L/R`, `thigh.L/R` (median) | 53.5 / 13.6 / 21.4 / 27.1 deg (max 72.8) | **0.0 median AND max** (`roll_prototype.py`, F02) | **≤ 5 deg median, ≤ 15 max** |
| Limb roll — `forearm.L/R`, `shin.L/R` (median) | 20.4 / 14.4 / 13.8 / 25.4 deg | 0.0 (F02) | recorded and **documented as a convention**, not a measurement |
| Max consecutive-frame roll change (continuity) | n/a (no roll term) | 55.1 deg unweighted; captured normal jumps 55.7 deg at 22.7 deg of bend (F02 refuter) | **≤ 15 deg** |
| `hand.L` posed-axis spread over the take | **125.9 deg** | forearm's own roll variation (F02) | **≤ 40 deg** |
| All-joint retarget error (positional cost of the roll fix) | 1.51 % of height | 1.62 % with roll (F02) | **≤ 1.70 %** |
| Character hip line vs captured hip line | **11.76 deg median / 21.18 max** | analytic ceiling with a perfect roll 4.505 / 14.705 (F12) | **≤ 5.0 / ≤ 15.0 deg** |
| Ground datum, peak-to-peak vs the fixed grid | 7.3 % median, **9.7–11.2 % peak-to-peak** of character height | 0 by construction with the rig's 0.71217-unit rest sole drop (F11) | **≤ 0.5 %** |
| Foot sole tilt while planted (median) | L 38.5 / R 42.7 deg | L 39.6 / R 43.8 with roll — **no improvement** (F02 refuter) | **no assertion**; needs real foot keypoints |
| Exported hips translation range | **exactly [0, 0, 0] rig units** (verified on all 771 motion rows) | pelvis travels 0.1362 m = **115.7 % of body height** (F13) | **the captured 0.1362 m**, default ON for BVH/FBX |
| Largest non-root bone translation channel in the BVH | `shin.R.001` **6.1287 rig units = 42.5 % of rig height** (verified) | 0 if pinned at rest | **≤ 1 % of rig height** |
| BVH-vs-3D-view agreement at the captured keyframes | **never measured in the project's history** | — | **< 1 % of body height** |
| BVH in-between overshoot past the bracketing poses | **3.795 deg max, p99 2.93, 0 of 25 transitions > 5 deg** (verified — not a defect) | — | **≤ 5 deg** (guard, passes today) |
| Preview-vs-export vertical placement gap | **25.5 % of rig height** | 0 with one shared placement rule (F13) | **0** |
| Export frame index of captured pose *k* | 1 + 30·*k*, undocumented | `schedule=None` already supported (`blender_job.py:444`) | ***k* + 1** by default for BVH/FBX |
| Head-bone orientation lost on every imported project | **15.5 deg median / 23.9 p90 / 28.5 max**; head mesh 6.3 % / 12.2 % of height | restored by calling `detect_project` (F16) | **feature alive on every project** |
| Corrections surviving a re-detect | **0 of 5** (and 5 `corrected` flags lie) | 5 of 5 with `respect_corrections` (F16) | **5 of 5** |
| HEAD retarget error, no face points / with face points | 12.65 % / 9.09 % of body height | **3.93 % / 2.92 %** with a native skull HEAD (F04) | **≤ 4.0 % / ≤ 3.0 %** |
| neck-head bone CV (worst bone in the baseline) | **9.53 %** | 3.85 % (F04) | **≤ 4.0 %** |
| neck-Lshoulder bone CV (the Halpe regression gate) | 5.15 % (COCO) | 8.13 % with a **native** NECK — a regression (F04) | **must not exceed 5.15 %** → NECK stays derived |
| Reconstructed body height (re-baseline trigger) | 0.1187 m | 0.1303 m with a skull HEAD, **+9.8 %** (F04) | re-baselined in the same commit |
| Pose Accuracy gauge on this take | **52.3 %, 26/26 frames "Low", 284/390 dots red (72.8 %)** | same take at 1080p reads 84.0 % with 21/26 green (F24) | **amber**; ≤ 50 % of dots red; ≥ 20/26 frames non-red |
| Same gauge on ground-truth Panoptic calibration | **~45 % "Low"** — worse than the client's suspect rig | — (F07, F24) | **green** |
| Gauge response to a 15 deg extrinsic error (19.13 mm = 16.2 % of height) | 52.3 → 49.4 %, never leaves "Low" | bone CV 5.27 → **7.42 %** (7× more responsive) (F07) | **≥ 30 % move in at least one displayed row** |
| Two cameras' reprojection medians, comparable? | 5.11 vs 2.78 px = **1.84× apart** (framing artefact) | 0.66 % vs 0.69 % of figure height — agree to 4 % (F25) | **normalised, per camera, never averaged** |
| Observations recoverable after a bad calibration is fixed | **0 of 209** (loss survives save/reload) | 100 % with `kp2d_raw` + a mask (F06) | **100 %** |
| Epipolar gate width | 71.5 px = **14.6× the 4.91 px median**, 3.0× p99; 0/388 dropped | ~25–30 px = 6× median, still 0/388 (F36) | **~25–30 px**, hallucination caught in ≥ 24/26 frames (today 15) |
| Damage from one blacked-out hip over a whole take | **71.6 mm median / 279.6 max = 237 % of body height** | 2.3 mm with proportional fallbacks (F18) | **≤ 2.5 mm** |
| Ambiguous-tag protection | none — first-hit lowest-id tag; runner-up branch for tag 14 destroys 367/780 (47 %) | tag 13 ratio 1.26 vs 5.37–13.80 for tags 14/15/17 — **zero overlap** (F23) | tag 13 rejected 26/26; **shipped rig reproduced to 0.000 deg / 0.000 mm** |
| Residual tilt of the delivered animation | `sequence_up` sits **15.20 deg** from the 3-proxy consensus (world frame is 89.34 deg off vertical) | proxies agree within 8.1–14.2 deg (F15) | **≤ 7 deg, reported with its uncertainty** |
| Per-frame lean (must be preserved) | 10.83 deg median / 21.10 max | identical after de-tilt to **1.5e-11** (F15 correction) | **unchanged** |
| Calibration provenance on disk | **nothing** — recoverable only by brute-forcing 97 IPPE solves | — (F30, F17) | `report.json` rebuilds the rig bit-for-bit |
| CI tests that would have caught any of the above | **0** (183 pass, 6 skip; Panoptic validators `--ignore`d; every geometry test uses one symmetric `Intrinsics`) | — (F32) | **4 assertions that fail today** + an asymmetric close-range rig + collected Panoptic |

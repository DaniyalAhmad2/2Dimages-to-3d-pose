# Improvement plan — nine phases, ordered by measured effect per line changed

Three of the phases are net deletions. The only genuinely new modules are one Qt-free metrics file, one pure roll-reference helper, and one gravity estimator. Every phase is independently shippable and has a rollback. Effort is for one developer.

Paths are relative to `/media/athena/hd3/Projects/pose3d-tool`. `<SP>` = `/tmp/claude-1000/-media-athena-hd3-Projects-pose3d-tool/0fcb8c9e-52c1-483d-9e0a-80896b078fa5/scratchpad`.

| # | phase | effort | ships |
|---|---|---|---|
| 0 | Pin the current behaviour (harness, fixture, export-fidelity script) | M | no user-visible change |
| 1 | Remove the causal EMA; flagged gap fill; drag == batch | S | the 98 % item |
| 1b | Make it reach existing takes | S | version stamp, non-destructive re-detect, face points |
| 2 | Roll references, hip line, ankle grounding | M | the visible angular fix |
| 3 | The delivered file | M | root motion, helper bones, no silent fallback |
| 4 | Readouts that describe the delivered pose | M | the client can see a fix |
| 5 | Native skull HEAD (gated) | M | the worst bone in the baseline |
| 6 | Latent robustness + provenance + recorded up | L | six independent commits |
| 7 | Deferred register | S | written decisions |

---

## Phase 0 — Pin the current behaviour

**Goal.** One committed fixture from the client's own take and one metrics module, so every later phase is a measured before/after; and the delivered BVH gets asserted on for the first time.

**Findings resolved.** F32, F45 (partly); closes the audit's own caveat 9 (Blender was never run).

**Changes.**

1. **NEW `pose3d/quality.py`** (Qt-free, importable by tests, the CLI and the sidebar alike — this is the one new module the plan allows). `take_quality(project, rig, character=None) -> TakeQuality` with:
   * `subject_height_m` — the median z-extent of the de-tilted pose (**one denominator, defined once**; 0.1178 m today), so every "% of height" in the repo means the same thing;
   * `figure_h_px[cam]` — median 2D bbox height of the finite `kp2d` in that camera (776 / 407 px today);
   * `reproj[cam]["measured"|"delivered"]` in px **and** as a fraction of `figure_h_px[cam]`, for `f.pose3d` and `f.fitted3d` respectively;
   * `epipolar` — Sampson median/p90/p99/max via `triangulate.epipolar_distance`, plus the symmetric per-image point-to-line distances `d_L`, `d_R` each normalised by **its own** image diagonal;
   * `bone_cv` — per-bone median/std/CV over frames from `bonefit.measure_bone_lengths` applied per frame to the **raw** triangulation (measuring after smoothing shrinks the skeleton by up to 3.9 %), plus median/max;
   * `symmetry` — L/R ratio and asym % for thigh, shin, upper arm, forearm;
   * `gaps` — list of `(frame_id, joint)` with no 3D, and counts of rejected/filled;
   * when a `Character` is passed: `retarget_pct_height` per joint, **limb roll error in degrees** (the angle about each bone's aim axis between its carried rest bend reference and the captured bend-plane normal — the F02 metric, which every positional metric in the repo is structurally blind to), foot sole tilt, ground-datum offset.
   Port the implementations **verbatim** from `<SP>/baseline/metrics.py` (`bone_length_stats`, `body_epipolar`, `reprojection`, `retarget_error`) so the numbers stay directly comparable with the audit's. Do not re-derive them.
2. **NEW `tools/measure_take.py`** — thin CLI over `pose3d.quality`, printing the dataclass as a table and as JSON.
3. **NEW `tests/fixtures/client_take/`** — `project.json` trimmed to the 26 frames' `kp2d`/`scores`/`corrected`/`pose3d`/`fitted3d`, `calibration/{left,right}_intrinsics.json` + `extrinsics.json`, and `aruco_corners.json` holding the detected 4×2 tag corners per (frame, camera, tag) — reuse the detection in `<SP>/verify-F23/detect.py`. **No images, no RTMPose at test time**; ~250 KB. `workspace/` is gitignored (`.gitignore:43`) so the take cannot be read in place. Commit the regeneration script beside it.
4. **NEW `tests/test_client_regression.py`.** Thresholds set **10–15 % above** today's measured values so they cannot flake (F32 warned that three of the auditor's proposed numbers sat within 5–15 % of today's): raw bone CV median ≤ 6.0 % / max ≤ 10.5 %; L/R asymmetry ≤ 7.5 % per pair; epipolar median ≤ 6 px, p90 ≤ 14 px; retarget all-joint median ≤ 2.0 % of height. Record today's number as a comment on every assertion. Two more are written now as **`xfail(strict=True)`** because they fail on today's code and are exactly Phase 1's gates: `reproj(fitted)/reproj(raw) ≤ 1.5` (today 4.24 L / 6.17 R) and `max|fitted − refit(no smoothing)| ≤ 3 % of height` (today 21.8 %).
5. **`tests/synth.py`** — add `close_range_two_cam(baseline=0.56, dist=0.53, f_left=4080, size_left=(3072,4080), f_right=2048, size_right=(1536,2048))`, an **asymmetric** close-range rig, and parametrise `test_triangulate.py` and `test_cross_view.py` over both. Every geometry test today passes one `Intrinsics` for both cameras at a 3.86 m baseline / 3 m distance, so an asymmetric-rig bug is structurally invisible — and the client's rig is asymmetric by exactly 2:1.
6. Rename `tests/validate_panoptic_{geometry,rtmpose}.py` → `test_panoptic_*.py`, add `tests/gates.py::needs_panoptic()` next to the existing `needs_blender()`, **delete the two `--ignore` lines at `.github/workflows/windows-test.yml:84-85`** (renaming alone leaves them dead), and commit `data/demo_project/calibration/` (currently untracked).
7. **NEW `tools/check_export_fidelity.py`** (dev script, not CI). Export the fixture with `render_video=False`, parse the BVH, evaluate FK in numpy and report: (a) deviation from `Character.posed_joints` at the captured keyframes as % of body height; (b) per in-between frame, each bone's angular deviation from the slerp arc between its bracketing keys and any overshoot past both endpoints; (c) consecutive keyframe quaternion pairs with `dot < 0`; (d) the largest non-`hips` translation channel as a fraction of the 14.4228 rig height; (e) what the fallback path ships, by pointing `character_blend()` at a missing file. **Baseline already measured from `assets/Imported_Session.bvh` and to be encoded as the fixture's expected values:** 772 frames / 30 fps / 26 poses; each pose's 22-frame hold bit-identical (0.0 deg across all 189 rotation channels); 8 in-between frames per transition with max overshoot **3.795 deg**, p99 2.93, **0 of 25 transitions above 5 deg** (no hemisphere flips); hips translation constant at `(0.000000, −0.131342, 2.851750)`; `shin.R.001` position range **6.1287 rig units = 42.5 % of rig height**.

**Remove.** Only the two `--ignore` lines and the `validate_` prefixes. No production code changes — that is the point.

**Tests.** As above. `tests/test_export_smoke.py` gains five assertions (hold flatness, in-between overshoot ≤ 5 deg, FK-vs-`posed_joints` ≤ 1 % of body height at every keyframe, largest non-root translation ≤ 1 % of rig height, frame count = `1 + (n−1)·30 + 21`). The helper-bone assertion is asserted **from the committed BVH facts in Phase 0** (a plain parse, no Blender) and becomes a live assertion in Phase 3 — do **not** leave it as `xfail(strict=True)` inside a `needs_blender`-gated test, which skips locally and never fires on the developer's machine.

**Acceptance.** `tools/measure_take.py workspace/pose3d_projects/Imported_Session` reproduces the baseline to two significant figures: reprojection 21.65 / 17.18 px fitted vs 5.11 / 2.78 raw; bone CV 1.04 % median / 6.35 % max fitted, 5.3 / 9.5 raw; epipolar 4.91 px median / 12.5 p90; retarget 1.5 % of height; 0 of 388 pairs over the 71.5 px gate. Exactly two `xfail(strict=True)` entries. The Panoptic validators run in CI.

**Risk & rollback.** Lowest of any phase: no production code is touched, so the metric invariance *is* the self-test. The one judgement call is 250 KB of fixture in git; if unacceptable, store `kp2d`/`scores` as base64 float32 and drop `pose3d`/`fitted3d` (both recomputable), roughly halving it. Rollback: delete the fixture directory and the two new test files.

---

## Phase 1 — Remove the causal EMA

**Goal.** The pose that is drawn, exported and measured is each frame's own bone-fitted triangulation; missing joints stay missing or are visibly labelled; and the manual-drag path and the batch path are the same computation by construction.

**Findings resolved.** F01 (critical), F09, F10, F14, F27, F33, F44.

**Changes.**

1. **Stop calling the smoother at the call sites, not at the default.** `smooth=True` is passed **explicitly** at `pose3d/ui/import_dialog.py:214`, `pose3d/ui/model.py:55` and `tests/build_demo_project.py:77` — flipping `pipeline.py:153` alone is a measured no-op. Remove the argument at all three sites and default `fit_project(..., smooth: bool = False)`.
2. **Keep a lag-free filter, off by default** (do not delete the capability while client question 7 is unanswered). Re-implement `smooth_temporal` **zero-phase**: forward EMA, then the same EMA over the reversed sequence, NaN-aware in both directions:
   ```python
   def smooth_temporal(p, alpha=0.6):        # zero-phase, NaN-preserving
       return _ema(_ema(p, alpha)[::-1], alpha)[::-1]
   ```
   Expose it as an import-dialog checkbox **defaulted off**, labelled "Smooth across frames (video-rate capture only; a stop-motion take has no temporal signal to filter)", persisted as `ProjectData.smoothing: str = "none"` (a top-level `project.json` key, defaulting to `"none"` on load). Note `alpha=1.0` is **not** an off switch: the NaN branch is alpha-independent and still fabricates joints.
3. **Delete `bonefit.py:140`** — `blend = np.where(np.isnan(cur), prev, blend)` — the line that resurrects joints the fit deliberately refused to invent. Keep line 141, which correctly restarts the filter after a gap.
4. **Explicit, flagged, symmetric gap fill replacing the accidental one.** New `pipeline.fill_gaps(project, max_gap=1) -> int`: per joint, find runs of ≤ `max_gap` consecutive NaN frames flanked by observed frames, set them to the **midpoint of the two neighbours** (no lag, no forward leak — the EMA's stale value decayed over ~8 frames and affected 21 joint-frames, up to 15.40 mm), and set a new **`Frame.filled` boolean array persisted by `io_project` alongside `corrected`**. Longer gaps and end-of-take gaps stay NaN. A filled joint counts as observed for the fit. Both `view3d._draw_skeleton` and `camera_view` draw filled joints **hollow/amber**, and the reconstructed-joint count reports them separately — today frames 0012 and 0021 report 15/15 while two joints are inventions. `blender_export._character_bone_frames` consumes the **same** array, so `blender_job`'s hold-last-known rule (`blender_job.py:454`) becomes unreachable for short gaps and view and export stop disagreeing about what a dropout means.
5. **One fit function.** Extract `pipeline.fit_frame(pose3d, bone_lengths)` and `pipeline.bone_length_targets(project)`; `fit_project`'s loop and `ProjectModel._resolve_joint` (`model.py:157-161`) both call them. Delete `ProjectModel._compute_bone_lengths` (`model.py:173-178`, a verbatim duplicate of `pipeline.py:157-161`) and the `self._bone_lengths` session cache. **F10 then dissolves by construction** — no replacement code. Keep the single-frame call on drag; do not re-run `fit_project` on every release.
6. **Derived joints follow the drag** (F33). In `_resolve_joint`, after the stack edit, re-derive `kp2d[cam][NECK|PELVIS]` from the current shoulder/hip 2D in the edited camera only, skipping any derived joint whose own `corrected[cam]` flag is set, then re-triangulate that index as well as the dragged one (`dependents = {L/R_SHOULDER: NECK, L/R_HIP: PELVIS}`). Record the derived change on the **same** `Edit` so one Ctrl+Z reverses the whole thing — the pattern already used for the HEAD/nose sync at `model.py:137-149`. Guard on `keypoint_model == "coco17"`.
7. **`fit_bone_lengths` solves only observed joints** plus any unobserved joint lying on a bone chain between two observed ones; freeze the rest at `x0` **and drop the bone residuals that touch a frozen joint** (otherwise a joint frozen at the centroid pulls the observed ones — the hazard `trf` currently avoids), NaN-ing them at the end when `fill_missing=False`. This keeps `method="lm"` in every realistic case (F44: 21.47 ms vs 746.78 ms). Signature and the `fill_missing` contract unchanged.
8. **`fit_project` returns a `FitReport(failed, first_error, fallback_bones, gaps_filled)`** instead of `print()`ing to a stdout a windowed build redirects to `pose3d-log.txt` (F27); `import_dialog` and `model.recompute_all` surface it the way `rejection_note` already is.
9. **Pre-flight probe, before choosing the default.** Re-triangulate the take ~20× under the bbox-jitter perturbation the detection auditor characterised (σ giving 1.26–2.79 px of keypoint movement) and report the per-joint 3D spread against the 9.36 mm median inter-frame motion. **Under ~1 mm** confirms that smoothing off introduces no jitter a viewer can see (and the export holds each pose for 22 of every 30 frames); **above ~3 mm**, ship the zero-phase filter on at a low alpha instead. This converts the plan's strongest assumption into a measurement.

**Remove.** The three `smooth=True` call sites; `bonefit.py:140`; `ProjectModel._compute_bone_lengths` and the `_bone_lengths` cache; the `print()` at `pipeline.py:178-180`; `tests/test_bonefit.py:53-63` (the `sm_var < raw_var` assertion was the **only** guard the smoother ever had, which is how a 21.8 %-of-height defect stayed green). The causal EMA implementation itself is replaced, not deleted.

**Tests.**
* `tests/test_smoothing.py::test_no_backward_lag` — on a synthetic sequence built from `synth.sample_skeleton_3d` translated linearly and rotated with `synth.rot_about`, median `cos(displacement, motion) >= −0.1`. **Fails on today's code at −0.976**; this is the mechanism-level regression guard.
* `::test_smoothing_is_off_by_default` — `fit_project` on a two-frame project leaves frame 1 equal to its own fit.
* `::test_nan_stays_nan` — a joint NaN in the middle frame is still NaN after `smooth_temporal`.
* `tests/test_gap_fill.py` — a 1-frame gap is filled to the midpoint **and flagged**; a 2-frame gap stays NaN; view and export read the same array.
* `tests/test_pipeline_fit.py::test_zero_pixel_drag_is_a_no_op` — drive the real `ProjectModel._resolve_joint` on the fixture with each joint's existing coordinates; every joint moves < 0.05 mm (today median 4.92, max 25.65, 15/15 joints).
* `::test_derived_joint_follows_a_shoulder_drag` — NECK's 2D lands on the new midpoint and its 3D error is ≤ 0.2 mm (today 30.0 px / 3.31 mm).
* `tests/test_bonefit.py::test_sparse_frame_uses_lm` — 5 NaN joints fit in < 100 ms.
* Both Phase-0 `xfail`s flip to real assertions.

**Acceptance** (client fixture, against the Phase-0 baseline).

| metric | baseline | target |
|---|---|---|
| reprojection median L/R | 21.65 / 17.18 px | **7.38 / 4.01 px** (ratio to raw 4.24/6.17 → ≤ 1.5) |
| bone-length CV median / max | 1.04 % / 6.35 % | **0.14 % / 0.37 %** |
| L forearm, frame 0013 | 0.0173 m (19 % short) | **0.0214 m** |
| delivered-vs-measured displacement | 4.92 mm med / 25.65 max | **0.00** |
| zero-pixel drag jump | 4.92 mm med / 25.65 max | **0.00 mm** |
| fabricated joints | 2 | **0** (2 gaps filled and flagged; the posed-joint jump at each must stay under 2 % of height — measure it, and if it exceeds that, ship the gap as a visible hole instead) |
| stale derived joint after a 60 px drag | 3.31 mm med / 10.66 max | **≤ 0.2 mm** |
| sparse-frame fit | 746.78 ms | **≤ 40 ms** |

Expect the all-joint retarget median to **rise** 1.51 % → ~1.7 % of height. That is correct and must be stated in the release note: the smoothed pose scored best on that metric only because the character faithfully followed a pose that had drifted off the keypoints. Cost, stated honestly: inter-frame motion rises 7.60 → 9.16 mm (+17 %) — that 17 % is the entirety of what the smoother was buying.

**Effort.** S. **Risk & rollback.** Technically low: one call removed, one line deleted. The real risk is perceptual, and item 9 bounds it before merge. NaN joints reach the retarget for the first time in this take's history (0012 `RIGHT_KNEE` takes the two-bone IK branch; 0021 `LEFT_ANKLE` is an end joint with no aim) — which is why the flagged fill ships in the same phase, not after it. Rollback: `project.smoothing = "ema0.6"` restores a filter (lag-free), and the fixture test tells you within seconds which side of the change you are on.

---

## Phase 1b — Make the fix reach existing takes

**Goal.** A fixed build changes what the client sees when they open the project they complained about, with no button press and no mystery jump; and the only route to face keypoints stops destroying hand corrections.

**Findings resolved.** F16; the delivery gap the audit never measured.

**Changes.**

1. **`io_project.save_project`** writes `"pipeline_version": N` at the top level of `project.json` (today's keys are only `name`/`fps`/`calibration_ref`/`frames`). `load_project` returns it, defaulting to **0** for files written before the key existed. Old projects load unchanged.
2. **`app.build_model` / `main_window._load_model`** — when `project.pipeline_version < CURRENT` **and** a rig is present, run `model.recompute_all()` once on open, then stamp the version, and show one dismissible line in the existing status bar with the **actual numbers**: *"Recomputed with solver v2: each frame now follows its own keypoints; the previous build blended every frame with the one before it — the pose moved 4.9 mm median, 25.7 mm max (4 % / 22 % of the figure's height). Restore stored pose."* Keep the stored `fitted3d` in memory for the Restore, and **never auto-save**. With no rig present, leave the stored pose alone and say so.
3. **`pipeline.detect_project` stops overwriting hand corrections** — two lines: skip the write where `frame.corrected[cam][j]` is set, behind `respect_corrections: bool = True`. Today it overwrites unconditionally and `model.redetect_all` (`model.py:61-72`) never consults `corrected`, so "Run Detection" silently discards every correction while leaving its flag `True` — and that is exactly the button F16's fix makes people press.
4. **Delete the inline detection loop at `import_dialog.py:199-206`** and call `pipeline.detect_project(project, det, cv2.imread, on_frame=cb)`; give `detect_project` an `on_frame(i, n)` callback for the progress dialog rather than duplicating the loop. The inline loop writes only `kp2d`/`scores` and drops `Detection.head_xy`/`head_scores`, so every project ever made through Import Images is headless.
5. **One new menu action, "Re-detect face points only"** → `detect_project(..., fields="head")`: writes only `head2d`/`head_scores` and the triangulated `head3d`, touching neither `kp2d`, `scores` nor `corrected`. This is the migration path for the client's existing project, whose `project.json` predates the head feature. (The loader default of `head2d = NaN` already exists at `project.py:65-69` / `io_project.py:145-153` — do not re-add it.)
6. **Do not touch `character.py:38 _NOSE_PITCH` or `_head_aim_target`** here: they are the legacy path a project without face keypoints still takes, and F03's refuter measured that unclamping makes the head slightly worse and triples the visible nose-to-mesh distance (1.56 → 4.34 mm).

**Remove.** `import_dialog.py:199-206`.

**Tests.** `tests/test_import.py::test_import_populates_head_keypoints` (stub detector with finite `head_xy`; `frame.head2d[cam]` finite and `head3d` finite after triangulation — today all NaN, no exception). `tests/test_pipeline_fit.py::test_redetect_keeps_corrections` (5 corrections survive, 0 flags lie — today 0/5 survive, 5 lie). `tests/test_detect.py::test_detect_head_only_touches_nothing_else`. `tests/test_io_project.py::test_version_stamp_round_trips` and `::test_legacy_project_without_version_loads_and_recomputes`.

**Acceptance.** Opening `workspace/pose3d_projects/Imported_Session` on the fixed build changes the drawn pose with no button press — median ~4.9 mm, max ~25.7 mm — and the status bar states why. A project saved by the fixed build reopens with no recompute. Head bone orientation on a freshly imported take differs from today's import by **15.5 deg median / 28.5 max**. Corrections surviving a re-detect: 0/5 → **5/5**.

**Effort.** S. **Risk & rollback.** Recompute-on-open is a visible jump on every legacy project; it is gated on the version stamp so it happens exactly once per project, is explained with real numbers, and offers an in-session Restore. Rollback: do not bump `CURRENT` — the stamp is inert data old builds ignore.

---

## Phase 2 — Roll references, hip line, ankle grounding

**Goal.** Every bone's rotation about its own axis is set by anatomy rather than by whatever the minimal rotation gives; the figure stops bobbing against the grid; the hip line follows the captured hips.

**Findings resolved.** F02, F12, F11, F43.

**Changes.**

1. **Rest references, `Character.__init__`.** Build `self._rest_ref: dict[bone_index, unit vector]` from `rest_joints()`:

   | bone | rest reference `r0` |
   |---|---|
   | `upper_arm.L`, `forearm.L` | `unit(cross(rj[L_ELBOW] − rj[L_SHOULDER], rj[L_WRIST] − rj[L_ELBOW]))` |
   | `upper_arm.R`, `forearm.R` | same on the right |
   | `thigh.L`, `shin.L` | `unit(cross(rj[L_KNEE] − rj[L_HIP], rj[L_ANKLE] − rj[L_KNEE]))` |
   | `thigh.R`, `shin.R` | same on the right |
   | `hips`, `spine` | `unit(rj[RIGHT_HIP] − rj[LEFT_HIP])` |
   | `chest` | `unit(rj[RIGHT_SHOULDER] − rj[LEFT_SHOULDER])` |
   | `neck`, `head`, `clavicle.L/R`, `hand.L/R`, `foot.L/R` | **none** |

   Both limb chains are well conditioned at rest (rest elbow bend 52.93 deg, rest knee 15.83 deg). `neck`/`head` get none because the head already receives a full measured orientation from the face basis and a roll would double-drive it; `hand.*`/`foot.*` get none because they have no aim and no keypoints — once the parent's roll is right, their inherited orientation is right too.

2. **`_bone_fk(self, b, base, end, ref_target=None, weight=1.0)`.** After `R = _align(rest_d/n_rest, d_want/n_want)`, with `a = d_want/n_want`:
   ```python
   cur = _unit(_proj_perp(R @ self._rest_ref[b], a))
   tgt = _unit(_proj_perp(ref_target, a))
   if cur is None or tgt is None: return M          # reference parallel to the aim
   ang = atan2(dot(cross(cur, tgt), a), dot(cur, tgt))
   R = _rot(a, weight * ang) @ R
   ```
   This is `<SP>/find-retarget/roll_prototype.py::_apply_roll`, which reproduces the shipped skin matrices to **max |diff| = 0.0** with the roll term off — so the delta is fully attributable and the rollback is exact. Canonical joint positions are provably unaffected (each limb child's head lies on the parent's axis; the prototype moved `posed_joints` by 0.0000 mm over 26 frames).

3. **Capture targets, supplied by `_skin_matrices` in RIG space** (directions only, so only `Rz` is applied — `to_rig`'s scale and translation are irrelevant): limb bones take `Rz @ cross(p[mid] − p[root], p[end] − p[mid])` with all three joints required valid; `hips`/`spine` take `Rz @ (p[RIGHT_HIP] − p[LEFT_HIP])`; `chest` takes `Rz @ (p[RIGHT_SHOULDER] − p[LEFT_SHOULDER])`.

4. **Sign convention — stateless, torso-based.** The cross-product's sign is not determined by joint order alone, so flip `n` so that `dot(n, Rz @ (p[RIGHT_SHOULDER] − p[LEFT_SHOULDER])) > 0` for arms and `dot(n, Rz @ (p[RIGHT_HIP] − p[LEFT_HIP])) > 0` for legs; **apply the identical rule to `r0` at rest** so the rest and capture references live in the same hemisphere. Anatomically the hinge normal lies along the limb's medial-lateral axis, so this is well defined *and* stateless — which is what keeps `pose_bone_matrices` a pure function of one frame and therefore keeps view == export for free.

5. **Straight-limb fallback — continuous, not a guard.** Do **not** use an on/off straightness threshold: the left elbow's *minimum* captured bend is 22.7 deg — above any 15 deg guard — and its bend-plane normal still jumps up to **55.7 deg** between consecutive frames, so a guard would never fire and the correction itself would pop 55.1 deg. Use
   ```python
   w = clip((bend_deg - 20.0) / 20.0, 0.0, 1.0)     # 0 below 20 deg, 1 above 40
   ```
   applied as `weight = w`. Below 20 deg the roll term vanishes and the bone falls back **continuously** to today's minimal rotation; at the measured 22.7 deg worst case `w ≈ 0.14`, which is what actually suppresses the jump. `hips`/`spine`/`chest` use `weight = 1.0`: the hip line is only a 16.3 mm segment, but a Monte-Carlo at this take's own 2–3 px noise perturbs it by just 2.3–3.4 deg against an 11.8 deg error, and its frame-to-frame coherence matches the shoulder line's — signal, not jitter.

6. **Ship gate and the named fallback.** Gate: **max consecutive-frame roll change ≤ 15 deg** on the client fixture. If it fails, add `pose3d/geometry/retarget.py::roll_references(poses3d, valid) -> (T, B, 3)` — a **pure function** doing the hemisphere fix plus a symmetric 3-tap smoothing of the *reference vectors* across frames — called identically by `view3d`'s per-take precompute and by `blender_export._character_bone_frames`, and passed in as `pose_bone_matrices(..., roll_ref=...)`. **Explicitly do not** carry per-bone roll in caller-owned temporal state: `view3d.set_pose` is driven per frame from `main_window.py:450` in whatever order the user scrubs, while `blender_export` iterates once in order, so caller state desyncs the view from the export.

7. **Grounding** (F11). Replace `view3d.py:157`'s `dz = float(verts[:, 2].min())` with
   ```python
   dz = min(posed[L_ANKLE].z, posed[R_ANKLE].z) - ANKLE_SOLE_DROP / scale
   ```
   where `ANKLE_SOLE_DROP` is the rig's constant rest ankle-to-sole height read from `character.npz` once — **verified 0.71217 rig units = 4.94 % of the 14.4228 rig height**. Fall back to the old rule when both ankles are NaN. Do **not** force the sole horizontal: the shin genuinely tilts 16–86 deg on a rigid-footed mannequin and the roll fix leaves sole tilt essentially unchanged (L 38.5 → 39.6, R 42.7 → 43.8 deg) — levelling would replace a measurement with a convention, and the ankle datum already fixes the bob.

8. **Antipodal determinism** (F43). With an explicit reference the near-180 deg branch of `_align` no longer decides anything, but make it deterministic: `axis = unit(cross(a, self._rest_ref[b]))` when a reference exists. Today a 1 deg wobble at exactly 180 deg flips the roll by 163.8 deg.

9. **Gaps do not silently inherit.** When a bone's aim target is missing, mark the joint in both views (Phase 1's `filled`/NaN states) and keep the **same** rule in view and export — today `blender_job` holds the previous whole pose via `last` while `character.py` inherits the parent.

**Remove.** `view3d.py:157`'s lowest-mesh-vertex grounding; the unconditional minimal rotation as the *only* orientation source in `_bone_fk`. Keep `_compute_rest_poles`/`_pole`/`_solve_ik` — the two-bone IK is the occlusion path and Phase 1's preserved NaNs make it *more* used.

**Tests.** `tests/test_retarget.py::`
* `test_limb_roll_matches_the_captured_bend_plane` — synthetic subject from the rig with a known 40 deg bend rotated 60 deg about the upper-arm axis; the carried rest reference lands within 2 deg of the captured bend normal.
* `test_roll_is_continuous_across_a_straightening_limb` — sweep 60 → 0 deg of bend in 1 deg steps; no per-step roll change exceeds 15 deg (the unweighted prototype hits 55.1).
* `test_roll_does_not_move_the_canonical_joints` — `posed_joints` unchanged to 1e-9. This is what makes the change safe to ship.
* `test_view_and_export_share_one_skinning` — joints read back from `pose_bone_matrices` equal `posed_joints` to 1e-9 on all 26 fixture frames.
* `test_hip_line_follows_capture`.
* `test_align_is_deterministic_near_180` — 179 / 180 / 181 deg changes the roll by < 5 deg.
* `tests/test_view_orientation.py::test_grounding_is_ankle_based` — ankle-to-grid height constant to within 0.5 % of body height across the fixture.
* Visual gate, not automated: regenerate the contact sheets with `<SP>/find-retarget/render.py` (shipped vs fixed, 6 frames × 2 azimuths) and review side by side before merge. The client judges this phase by eye.

**Acceptance.**

| metric | baseline | target |
|---|---|---|
| roll error, `upper_arm.L/R`, `thigh.L/R` (median) | 53.5 / 13.6 / 21.4 / 27.1 deg | **≤ 5 deg median, ≤ 15 max** |
| roll error, `forearm.L/R`, `shin.L/R` | 20.4 / 14.4 / 13.8 / 25.4 deg | recorded, **documented as a convention** — see risk |
| max consecutive-frame roll change | 55.1 deg (unweighted prototype) | **≤ 15 deg** |
| `hand.L` posed-axis spread over the take | 125.9 deg | **≤ 40 deg** |
| character hip line vs captured | 11.76 deg med / 21.18 max | **≤ 5.0 / ≤ 15.0** (analytic ceiling with a perfect roll is 4.505 / 14.705) |
| ground-datum peak-to-peak | 9.7–11.2 % of character height | **≤ 0.5 %** |
| all-joint retarget median | 1.51 % of height | **≤ 1.70 %** (prototype cost +0.11 pp = 0.13 mm) |
| foot sole tilt | 38.5 / 42.7 deg median | **no assertion** — it will not improve, and no test should claim it does |

**Effort.** M. **Risk & rollback.** Roll is a *measurement* for `upper_arm` and `thigh` (the elbow and knee are hinges, so the bend plane genuinely determines the parent bone's roll) and a **convention** for `forearm` and `shin` — pronation was never observed and there are no hand or foot keypoints. Say so in the code and the release note; a future reader must not read "0.0 deg" on a forearm as accuracy. The four limb `_BEND` entries are independently removable if the client dislikes the twist. Mesh displacement is 0.77 mm frame-median with a 14.5 mm worst concentrated in the feet, so expect a subtle change, not a new silhouette. Rollback is exact: ship the weight endpoints as module constants and set the weight to 0 to restore today's matrices bit-for-bit.

---

## Phase 3 — The delivered file

**Goal.** Stop the exported file losing the subject's translation, stop it carrying 42 % of rig height of translation on bones nothing can interpret, and stop it silently substituting a different retarget.

**Findings resolved.** F13, F31 (export half).

**Measured first, so the plan does not fix a non-problem.** `assets/Imported_Session.bvh` parses clean: holds bit-identical (0.0 deg across 189 rotation channels), in-between overshoot ≤ 3.795 deg with **0 of 25 transitions above 5 deg** and no hemisphere flips. **No interpolation rework belongs in this plan.** What is wrong is placement.

**Changes.**

1. **Root motion.** `Character.pose_bone_matrices(..., keep_root_motion=True, pelvis_ref=None)`: `_skin_matrices` already computes `to_rig`, so add `root_offset = Rz @ (pelvis − pelvis_ref) * scale` with `pelvis_ref` = the take's first (or mean) pelvis, and add it to the **hips** matrix translation only — every child reads `base = skin[parent]` (`character.py:639`), so it propagates through the FK chain for free. `_from_rig` (`character.py:674`) takes the same `pelvis_ref` so `posed_joints` stays in the capture's space and the character/capture overlay is unaffected. Apply the same offset in the fallback at `blender_job.py:524` and delete the `COPY_LOCATION` pin at `:535`. **Default ON for BVH/FBX, OFF for the turntable mp4**, per-export overridable. Today the exported hips translation range is exactly `[0,0,0]` against 0.1362 m = 115.7 % of body height of real pelvis travel.
2. **The view uses the same rule.** Centre once on the take's `pelvis_ref` instead of per frame on the mean of the valid joints (`view3d.py:145-146`, which varies 6.8 mm = 5.8 % of height), keep the z seat (Phase 2 replaced its rule), and widen `_framed` to size the grid and camera distance from the take's full 0.136 m pelvis bounding box so the figure cannot walk out of frame. Do not simply delete line 145 — line 146 also seats z.
3. **Helper bones: pin at rest, do not omit.** `shin.L.001`, `shin.R.001`, `thigh.L.001`, `thigh.R.001` are IK helpers the app never drives; two of them are written as top-level roots carrying position ranges up to **6.1287 rig units = 42.5 % of rig height**. Set them to their rest transform before keyframing. **Pinning, not omission**: removing a bone changes the exported armature topology, which `blender_job`'s name lookups, the FBX writer and the FK round-trip assertion all depend on, and a bone that disappears between versions is a structural surprise for any downstream consumer.
4. **No silent retarget substitution.** Replace the two blanket `except Exception: return None, None` at `blender_export.py:44-53` with a typed reason carried in `ExportResult` and shown in the export dialog; give `blender_job.main`'s outer `except` (line 655) one loud greppable line and a non-zero marker the host surfaces as a **failed** export. Refuse to write a file that is not the view unless the user explicitly opts in. That fallback is what writes `data/demo_project/export/demo.bvh` — a null `ROOT __0` with `CHANNELS 0` and 15 sibling joints, all motion in position channels.
5. **One frame per pose becomes the default for BVH/FBX.** `blender_job` already supports it (`schedule=None`, `blender_job.py:444-445`) — this **exposes a capability, not adds one** — and it makes export frame *k*+1 correspond to photograph *k*. The stepped hold/ease schedule stays for the mp4 and behind a switch. Write `<name>_frames.json` (`captured index -> [first, last]`) **only for the stepped/mp4 path**, where the 1+30·*k* offset is real; on the default path it is redundant.
6. **Fixed-camera mp4.** Add an option rendering from the LEFT camera's own known pose alongside (or instead of) the turntable. A turntable spin over the whole clip is precisely what makes "does the character follow the keypoints" unanswerable by eye, and the client's comparison is photo-to-frame.

**Remove.** The `COPY_LOCATION` hips pin (`blender_job.py:535`); the two blanket `except`s at `blender_export.py:48/:52`; the print-only silent fallback in `blender_job.main`; `view3d.py:145`'s per-frame mean-of-valid-joints recentring as the framing rule; the hard-coded 0.7 s hold / 0.3 s ease as the *only* schedule.

**Tests** (`tests/test_export_smoke.py`, gated `needs_blender`, plus the Phase-0 parse-only assertions):
* `test_root_motion_reaches_bvh` — with the flag on, the hips translation range equals the pelvis travel × the fitted scale to within 1 %; with it off, exactly 0.
* `test_helper_bones_are_pinned` — the largest non-`hips` translation channel range ≤ 1 % of rig height (today 42.5 %). **Plain assertion, not `xfail`.**
* `test_bvh_keyframes_match_the_view` — FK from the BVH at every captured keyframe vs `Character.posed_joints`, max deviation < 1 % of the 117.8 mm body height, after a single global similarity fit.
* `test_no_quaternion_flips` and `test_inbetween_stays_on_the_arc` — 0 pairs with `dot < 0`; overshoot ≤ 5 deg (both pass today; they are the guards).
* `test_missing_character_asset_reports` — point `character_blend()` at a missing file; `ExportResult.ok is False` with a message naming the asset, instead of a different animation.
* `test_one_frame_per_pose_is_default`.

**Acceptance.** Largest non-root translation range **42.5 % → ≤ 1 %** of rig height. Hips translation range **0 → the captured 0.1362 m (115.7 % of body height)** by default. BVH-vs-view agreement at the 26 captured poses: **currently unknown in the project's history → < 1 % of body height**, with the measurement itself as the phase's primary deliverable. Preview-vs-export vertical placement gap **25.5 % of rig height → 0**. Export frame index for captured pose *k*: **1+30·*k* → *k*+1**. Silent retarget substitution: **possible on any asset failure → impossible without an explicit opt-in**.

**Effort.** M. **Risk & rollback.** Root motion changes framing for anyone who wanted the character in place — hence the per-export flag and a release note. Pinning helper bones is the one change that could alter what Blender exports structurally; the FK round-trip assertion is the guard. All five Blender tests skip locally, so run them once with `POSE3D_REQUIRE_BLENDER=1` against the installed Blender before merging. Rollback: `keep_root_motion=False` plus the stepped schedule restores today's file byte-for-byte.

---

## Phase 4 — Readouts that describe the delivered pose

**Goal.** The numbers on screen describe the pose on screen, in units that do not depend on which phone shot the frame, and they move when the reconstruction is actually wrong.

**Findings resolved.** F07, F24, F25, F31 (app half), F30/F17 (display half), F34 (diagnostic half).

**Changes.**

1. **`ProjectModel._accuracy(idx) -> dict[cam, ndarray]`** — score **both** stages and return the two cameras separately, never `nanmean`d: `"measured"` from `f.pose3d` ("do the two views agree?") and `"delivered"` from `f.fitted3d` ("what you are looking at"). Their **difference** is the actionable signal and the permanent regression detector for every later phase; after Phase 1 they should coincide (today they are 4.24× / 6.17× apart). Normalise each by that camera's `figure_h_px` from `pose3d.quality` — raw pixels are incommensurable (776 px vs 407 px tall, medians 1.84× apart; 0.66 % vs 0.69 % once normalised).
2. **`panels.accuracy_pct(err_px, figure_h_px)`** — replace `100·exp(−err_px/6.0)` and the `ACC_HIGH/ACC_MED = 85/70` constants with a piecewise-linear map on `frac = err_px / figure_h_px` anchored so `acc_band`, `acc_color`, `acc_label` and `camera_view._joint_status` keep their signatures: 100 % at 0, 85 % at 0.004, 70 % at 0.010, 0 % at 0.030. `main_window._refresh_timeline_status` (`main_window.py:483`) swaps its literal `< 5` / `< 12` px for `< 0.004` / `< 0.010` of figure height. Today green requires ≤ 0.98 px absolute on a 3072×4080 frame.
3. **Sidebar CALIBRATION section gains three rows** fed by `pose3d.quality.take_quality`, recomputed once per recompute: **bone-length CV** (median/max, captioned "a rigid subject should read near 0 %"), **epipolar disagreement** (median px plus % of each image's own diagonal), and **L/R limb symmetry**. These cover what reprojection cannot and each other's blind spots: bone CV is ~7× more responsive than the gauge to a 15 deg extrinsic error (5.27 → 7.42 %) but *blind* to a shared focal error (f × 0.7 **lowers** it to 5.09 %), while epipolar does move on the focal (4.91 → 6.23 px).
4. **Symmetry wording** (F34): never "this points at the calibration". On the ground-truth Panoptic reference RTMPose alone produces 6.5 % thigh asymmetry against 1.4 % in ground truth. Say *"the two views place LEFT_WRIST less consistently than RIGHT_WRIST — check that keypoint in both images"*, show the endpoint residuals, and flag at ~5 % over 26 frames (Monte-Carlo p99 at this take's noise is 4.24 %), not 3 %.
5. **Metric facts the client can falsify with a ruler**: camera baseline in cm (0.559 m), reconstructed subject height in cm (0.118 m), and the marker size / tag id / source frame the calibration came from (persisted in Phase 6). Three `_InfoRow`s, no new widget.
6. **`camera_view._joint_status`** bands on that view's own normalised residual and paints "not measured", "interpolated" (Phase 1) and "rejected by the cross-view check" (Phase 6) as distinct, visible states. Remove the unreachable detector-confidence fallback (measured firing 0 of 390 times) and the dead `Detection.rag()`.
7. **Stop swallowing diagnostics** (F31, app half): `app._load_rig` (`app.py:87-96`) separates `FileNotFoundError` (quiet, genuinely uncalibrated) from `JSONDecodeError`/`KeyError`/`ValueError`, validates that `R` is 3×3 and orthonormal and `t` is length 3 (a 2×2 `R` currently builds a `CalibratedRig` and only explodes inside triangulation), and puts the reason in the sidebar. `Character.pose_and_joints` raises a typed `PoseUnavailable` for "no usable pelvis"; `View3D._skin` catches that narrowly and routes anything else to a new `View3D.characterError` signal shown in the 3D card header. `fit_to_subject`'s silent `None` return gets its own message — the per-frame height fallback it triggers pulses the character over a 37.9 % range.

**Remove.** The `/6.0` px decay constant and `ACC_HIGH/ACC_MED`; the literal 5/12 px timeline cuts; the `nanmean` over cameras; `camera_view`'s confidence fallback and `Detection.rag()`; the blanket catches at `view3d.py:99`, `view3d.py:205` and `app.py:95`.

**Tests.** `tests/test_camera_view_accuracy.py::`
* `test_a_good_take_is_not_majority_red` — ≤ 50 % of joint-frames red on the fixture (today 72.8 %) and ≥ 20 of 26 timeline frames non-red (today 0).
* `test_bands_are_resolution_invariant` — scale `K` and `kp2d` by 0.265 (a similarity leaving the 3D identical); banding unchanged (today 52.3 % → 84.0 %, 0 → 21 green frames).
* `test_the_readout_sees_a_lagged_pose` — a synthetic lagged pose scores materially worse (today bit-identical).
* `tests/test_quality.py::test_bone_cv_responds_to_a_wrong_rig` — rotate the right camera's `R` by 15 deg; bone CV median rises ≥ 30 % (5.27 → 7.42 %).
* `::test_epipolar_responds_to_a_wrong_focal` — scale both `K` by 0.7; the epipolar row moves (4.91 → 6.23 px) even though bone CV does not.
* `tests/test_ui_smoke.py::test_missing_rig_reports_a_reason`.

**Acceptance.** Client fixture after Phase 1 bands **amber rather than 26/26 red**; 284/390 red dots (72.8 %) → the fraction that is genuinely inconsistent; Panoptic ground-truth calibration ~45 % "Low" → green. A 15 deg extrinsic error (19.13 mm = 16.2 % of height), which today costs the gauge 2.9 points and never changes its band, must move at least one displayed row by > 30 %. The sidebar shows baseline 55.9 cm, subject height 11.8 cm and marker 50 mm.

**Effort.** M. **Risk & rollback.** Rebanding is a judgement call and the first thresholds will be wrong; print the raw normalised number beside the band so the client can see through it, choose cut points from the Panoptic reference rather than this one take, and ship this **after** Phases 1–3 so the number first moves for a real reason. Rollback: both mapping functions are pure and isolated.

---

## Phase 5 — Native skull HEAD, behind a gate table

**Goal.** Take the one large remaining measured win — the head and the worst bone in the baseline — **without** the skeleton redefinition that comes with a whole-model swap.

**Findings resolved.** F04 (the separable half). Explicitly **not** F03 (refuted — keep the nose-pitch clamp on the legacy path).

**Changes.**

1. **A policy dict, not a model flag.** `skeleton.map_halpe26(kp, scores, head="native", neck="derived", pelvis="derived")`. Take Halpe index 17 (the **skull vertex**) as `Joint.HEAD`; keep NECK and PELVIS as the per-view **2D midpoints**. The evidence for the split is specific: native HEAD gives retarget 12.65 → **3.93 %** of height with no face points and 9.09 → **2.92 %** with them, and neck-head bone CV 9.53 → **3.85 %** (the worst bone in the entire baseline); native NECK **regresses** neck-Lshoulder CV 5.15 → 8.13 % because Halpe's neck is not the shoulder midpoint, while the derived midpoints are measurably better conditioned (derived NECK epipolar 1.51 px vs its parents' 3.86; construction penalty 0.25 mm at NECK, 0.08 mm at PELVIS). Fix the inverted module docstring at `rtmpose.py:3-6` in the same change.
2. **Head semantics, so this and Phase 1b do not double-correct.** Add a persisted `project.json` key `"head_source": "nose" | "skull"` (absent → `"nose"`). `character.py:38 _NOSE_PITCH` and `character.py:360 _head_aim_target` are gated on it: with a skull HEAD the legacy branch returns `J(Joint.HEAD)` directly — no 45 deg fudge, no clamp (it only ever "worked" with a skull point because `max(rest_pitch + θ − 45°, 0)` saturates at 0). Legacy projects keep today's behaviour bit-for-bit.
3. **Re-check the read-back point.** `_JOINT_FROM_RIG[Joint.HEAD] = (("head", "mid"),)` (`character.py:110`) is a point on the skull **axis**; with a skull HEAD the correct read-back is plausibly the head bone's **tail**. This choice is worth the same order as the fix itself, so let the harness decide it and record the answer — but do **not** adopt the "aim straight at the point" variant on the head-mid metric alone: F03's refuter measured that it triples the visible nose-to-mesh distance (1.56 → 4.34 mm).
4. **Build plumbing is mandatory, not optional.** Drop the `--feet` guard at `tools/fetch_weights.py:69` (one extra 55.7 MB file; YOLOX is shared) and add the Halpe checkpoint to `pose3d/selftest.py`'s presence check, the PyInstaller spec and the Docker staging. The repo ships no `models/` directory, so without this the client build silently falls through to rtmlib's downloader — the exact failure `detect/models.py` exists to prevent.
5. **Re-baseline in the same commit.** Reconstructed body height 0.1187 → 0.1303 m (**+9.8 %**), character scale 111.11 → 110.09. Every recorded "% of body height" number in the fixture and in Phases 0–4's acceptance criteria shifts; Phase 0's harness makes this mechanical.
6. **Storage check (verified non-issue, do not "fix").** `io_project._json_to_arr`'s row truncation (`io_project.py:36-42`) cannot lose Halpe data, because `map_halpe26` reduces 26 keypoints to the 15 canonical joints before storage and `head2d` is 5 rows by construction. Add a round-trip test so the claim stays true.

**Remove.** The unconditional `_NOSE_PITCH` correction on the skull path; the `--feet` guard; the inverted docstring. `derive_joints` becomes the documented COCO-17 fallback.

**Tests.** `tests/test_skeleton.py::test_map_halpe26` (no test exists today) and `::test_head_neck_pelvis_policy`. `tests/test_io_project.py::test_head_source_round_trips` and `::test_halpe_frame_loses_nothing`. `tests/test_client_regression.py` **gate table — these decide whether the change ships**:

| gate | today | must be |
|---|---|---|
| HEAD retarget, no face points | 12.65 % of height | **≤ 4.0 %** |
| HEAD retarget, with face points | 9.09 % | **≤ 3.0 %** |
| neck-head bone CV | 9.53 % | **≤ 4.0 %** |
| neck-Lshoulder bone CV | 5.15 % (COCO baseline) | **must not exceed 5.15 %** — the naive swap gives 8.13 %, which is why NECK stays derived |
| any directly-mapped body joint | — | **no regression > 0.5 % of height** (they move 0.16–0.93 % in 2D; retarget 2.08 → 2.27 mm, so this gate is tight) |
| head aim error with the nose path disabled | — | **< 5 deg** |
| body epipolar median | 4.91 px | **no regression** (measured 4.80) |

**Acceptance.** The gate table above, all fixed before the run. **Effort.** M. **Risk & rollback.** The +9.8 % height change silently invalidates every recorded threshold — mitigated only by re-baselining in the same commit, which is why Phase 0 comes first. The bundle grows 55.7 MB. A head double-correction if the pitch path is left live on a skull HEAD — explicitly gated by item 2, and measured by the gate table rather than assumed. Rollback: the policy dict reverts HEAD to the nose in one line; the weights guard is a build-script flag.

---

## Phase 6 — Latent robustness, provenance, and a recorded vertical

**Goal.** Nothing here changes the client's current numbers. All of it stops one reframing, one ambiguous tag or one occluded joint from producing an empty or wrecked 3D view with no diagnostic and no way back. **Ship as six independent commits.**

**Findings resolved.** F06, F36, F18, F23, F30, F17, F29, F41, F42, F15, F20, F21, F28, F22, F39, F40.

### 6.1 Non-destructive cross-view gate (F06)

`core/project.py` `Frame` gains `kp2d_raw`, `scores_raw` and `rejected[cam]` (bool, `NUM_JOINTS`), all persisted by `io_project`; `load_project` back-fills `kp2d_raw = kp2d` when absent so legacy projects load and behave no worse. `detect_project` writes the raw arrays once. `validate_cross_view` sets `frame.rejected[cam][j] = True` **instead of** writing NaN into `kp2d` and 0.0 into `scores` (`pipeline.py:100-101`); `triangulate_project` masks a local copy; `recompute_all` **re-derives the mask from `kp2d_raw` every time**, so fixing the calibration genuinely reinstates observations. Today it cannot: damage → save → reload → recompute with a good rig recovers **0 of 209**, and the NaN'd joints are hidden in the 2D view (`camera_view.py:169-171`) so the user cannot drag them back or tell the loss from a detection failure. `frame.corrected` still overrides. Draw rejected joints in `COL_PURPLE` **with the tooltip naming the number**: *"rejected: the two views disagree by N px (gate M px)"* — a rejection the user can see the number for is a diagnosis; a purple dot alone is a new kind of silence.

### 6.2 Data-driven, per-image epipolar gate (F36)

`epipolar_threshold(rig, project)`: compute all valid pairs' Sampson distances once, take a robust scale, `thr = clip(6 × median, 25.0, 1.4 % of the SMALLER image's diagonal)` — ~25–30 px here against a measured 4.91 median / 23.97 p99 / 29.38 max, still dropping 0 of 388, where the shipped 71.5 px is 14.6× the median. Separately evaluate a symmetric point-to-line distance in **each** image against 1.4 % of **that** image's diagonal and **reject when either exceeds** — explicitly **not** "only if both exceed", which is strictly *more* permissive than today (`d_R` max is 33.8 px against a 35.8 px allowance) and inverts the intent. Report the chosen threshold and the observed distribution in the sidebar.

### 6.3 Proportional fallback bone lengths (F18)

`fallback_bone_lengths(reference_m)` returning **ratios** of an observed reference — prefer the PELVIS→NECK spine, else the largest measured bone (verified within 4 % even when the spine itself falls back). `pipeline.bone_length_targets` passes `measured[(PELVIS, NECK)]`. Flip `bonefit.py:62` to `fill_missing: bool = False` so the safe behaviour is the library default. This matters on the **shipped** path: the fallback residuals sit in the least-squares regardless of the flag, so blacking out one hip in one view for a whole take moves the still-**observed** joints by 71.6 mm median / 279.6 max = 237 % of body height; with a correctly scaled table the same blackout costs **2.3 mm**. `FitReport` names any bone that fell back.

### 6.4 IPPE branch scoring, tag admission and provenance (F23, F22, F30, F17, F29, F41)

`extrinsics.estimate_extrinsics_for_marker` uses `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` and returns **both** solutions with their reprojection errors. `resolve_calibration` (`resolve.py:134-153`) replaces its first-hit `for tid in sorted(common): ... return` with a scoring loop:
1. detect tags in **all** frames and both cameras (it already loads every image);
2. **admit** a tag only if its per-tag IPPE reprojection rms across frames is ≤ 1.5 px and ≤ 3× the median tag — this separates tag 13 (2.4–3.3 px, bent over the curved sweep) from tags 14/15/17 (0.2–0.8 px) with **no overlap**, and the discriminator is planarity residual, **not** viewing angle (tag 13's 33.5 deg obliquity is unremarkable);
3. for every admitted tag common to both views, score the **joint** `(left-branch, right-branch)` pair — never per camera — on `err1/err0` in each camera (reject < 2 unless nothing else exists; medians 1.26 for tag 13 vs 5.37–13.80 for the others, zero overlap) **and** on cross-frame consistency of the resulting relative pose (0.0–2.0 deg for the right choice, 68.4 deg for the wrong one). Both cameras choosing wrongly does **not** cancel — `L1R1` is the worst case at 103.93 deg of relative-pose error while still returning a plausible 0.553 m baseline, which defeats any baseline-only sanity check.

`save_rig` (`resolve.py:44-49`) then writes **`calibration/report.json`**: `{world_tag_id, dictionary, world_frame_id, marker_length_m, per_camera_ippe_ratio, per_tag_rms_px, tags_admitted, tags_rejected_with_reason, n_observations_used, n_frames_with_a_common_tag, baseline_m, convergence_deg, world_up, world_up_source, world_up_spread_deg, focal_source_per_camera, solver, residual_rms_px, camera_motion_check}` — sufficient to rebuild the rig bit-for-bit, and asserted as such. `project.json` records `marker_length` and `keypoint_model`. `Intrinsics.load` defaults `source` to `"unknown"` (with `looks_assumed` as tie-breaker) instead of `"measured"` — today loading and re-saving the client's own guessed file launders the mislabel into new projects. **Delete** `estimate_extrinsics`' multi-tag branch and `tag_world_positions` (`extrinsics.py:99-145`): zero callers, one-rotation-for-all-tags is wrong by up to 179.4 deg on this backdrop, and its one-known-tag path violates `IPPE_SQUARE`'s centred-square precondition (camera-centre errors 527–1020 mm).

**Scale (F17), stated correctly:** the character export is scale-**invariant** (bone matrices differ by 1.03e-3 rig units between marker 0.05 and 0.10 m, because `fit_to_subject` normalises it away), so exporting at true size is a **missing feature**, not an accuracy bug. Persist the spinbox value, show the ruler-checkable readout (Phase 4), and add "Set scale from a measured distance". Do **not** ship the inter-tag consistency check as a scale check — a uniformly wrong tag size gives it exactly zero residual.

**Focal (F20, F21, F28):** keep `f = max(w, h)` as the default and record `source="assumed"`. **Keep `focal_from_exif` and its test** — fix DigitalZoomRatio (tag 41988) so it computes ~3570 px instead of the 21 %-short 2833 px, and invert the test's assertion to record *why* the assumption still wins (4.91 px body epipolar, the best of everything tried; the sweep optimum (3672, 1638) scores 5.37, the EXIF pair 10.9). Deleting the function re-loses the fact that commit f4c92f9 reverted a number **computed wrong** rather than an approach that is wrong, which is exactly how that revert gets re-litigated. Refuse an EXIF focal for a camera with no EXIF (the right camera has none). Wire `calibrate_checkerboard` as the optional one-time override, gated on `Intrinsics.source`. Document the two intrinsics nobody can recover: distortion must only ever come from a checkerboard (k1 = 0.03 moves tags 10.9 px, the subject 0.27 px) and the principal point is assumed centred and unverifiable from tags (±200 px = 19 px of epipolar error).

**Failure message that names the camera (F42):** *"left camera saw tags 13, 14, 17 in 26/26 frames; right saw tag 14 in 11/26 and nothing in 15 — move the right camera so at least two tags are in frame."* Do **not** add an essential-matrix fallback: 5.7–13.8 deg of rotation error with scale unrecoverable.

### 6.5 A recorded vertical (F15)

New `pose3d/geometry/gravity.py::estimate_world_up(rig, tag_layout=None) -> (up, spread_deg, candidates)` with three proxies:
* (a) `unit(cross(x_left, x_right))` where `x_cam = R_cam.T @ [1,0,0]`;
* (b) `−mean(R_cam.T @ [0,1,0])`;
* (c) `unit(cross(tag_row_direction, mean_tag_normal))` when a pooled layout exists — the tag row is 0.298 m long with a straightness singular-value ratio of 0.093, and it is the only candidate independent of camera roll. **Do not use it until 6.4's admission gate has rejected tag 13**, whose IPPE normal is ambiguous in 96 % of solves.

Combine as a normalised weighted mean and **report the pairwise spread as the uncertainty** (8.1–14.2 deg on this rig — that is the floor, and the report must say so). **Sign:** choose the sense that puts the take's mean NECK above its mean ANKLE. That is a binary test, not an orientation estimate, so it cannot leak body lean into the axis. `save_rig` writes `world_up` + `world_up_source` + `spread_deg`; `main_window._apply_view_orientation` (`main_window.py:494-500`) and `blender_export.py:59-61` prefer it when the spread is under ~20 deg and fall back to `orient.sequence_up` when it is absent — so a legacy project's orientation is bit-for-bit unchanged until recalibrated. Replace the standing sidebar paragraph (`panels.py:216-219`) with the number and its uncertainty: *"Vertical: camera pair, ±9 deg vs the tag row."* **Carry the correction into the docs:** `de_tilt_matrix` is ONE constant rotation, so `sequence_up` removes only the **mean** lean — the per-frame lean (10.83 deg median, 21.10 max) is preserved to 1.5e-11 and must stay preserved. The manual "level the figure" control stays deferred (Phase 7).

**Tests.**
* `tests/test_cross_view.py::test_a_bad_rig_does_not_destroy_kp2d` — damage with a 12 deg extrinsic error (185/780 dropped), save, reload, recompute with the good rig; **every** observation returns (today 0).
* `::test_the_gate_uses_each_image_s_own_scale` — parametrised over `close_range_two_cam`; the fixture still drops 0; the ankle-on-knee hallucination the docstring names is caught in ≥ 24/26 frames (today 15).
* `tests/test_bonefit.py::test_fallback_lengths_scale_to_the_subject` — one hip blacked out for a whole take moves the other joints < 5 mm (today 279.6 mm).
* `tests/test_extrinsics.py::test_the_ambiguous_tag_is_rejected` — **blocking**: on the fixture's ArUco corners the resolver must still choose **tag 14 / frame 0001 / branch 0 for both cameras, matching the shipped extrinsics to < 0.01 deg and < 0.1 mm**, and reject tag 13 in 26/26 frames. This test lands **in the same commit as the scoring loop** — it is the only change in the plan that can pick a different calibration than today's.
* `::test_joint_branch_scoring` — the `L1R1` combination with its plausible baseline is rejected. `::test_multi_tag_branch_is_gone`.
* `tests/test_calib_report.py::test_report_json_rebuilds_the_rig`.
* `tests/test_intrinsics_exif.py::test_digital_zoom_is_applied` (~3569 ± 5 px) and `::test_keyless_file_does_not_become_measured`.
* `tests/test_orient_up.py::test_recorded_up_wins`, `::test_legacy_rig_unchanged` (today's view rotation bit-for-bit), `::test_per_frame_lean_is_preserved` (unchanged to 1e-9 deg), `::test_sign_is_stable`.
* **Invariance assertion, on every commit in this phase:** all Phase-0 shape metrics (bone CV, epipolar, symmetry, retarget %, height) are **bit-identical** on the client fixture before and after. A robustness or frame-of-reference change that moves the reconstruction has done something else. This is the cheapest possible proof that the phase did not silently re-tune anything.

**Acceptance.** Observations recoverable after a bad calibration **0 → 100 %**. Gate width 71.5 px (14.6× the median) → ~25–30 px, still 0/388 dropped. Whole-take blackout damage 225 mm (191 % of height) → **≤ 2.5 mm**. The fixture resolves to the extrinsics it ships with (0.000 deg, 0.000 mm) with tag 13 rejected. `report.json` present and sufficient. Residual tilt: `sequence_up` sits 15.20 deg from the three-proxy consensus → **≤ 7 deg, stated with its uncertainty**. Every Phase-0 shape metric bit-identical.

**Effort.** L (six commits). **Risk & rollback.** None of this improves the client's current reconstruction — extrinsics are bounded at 0.15 % of body height in shape — and someone will read it as "we fixed the calibration". Sell it as *"we made the next take diagnosable"*: only 11 of 26 frames have any tag common to both cameras, tag 13 returns the wrong branch in 35 % of solves and is harmless only because the right camera never sees it, and nothing in the app detects a camera that moved (add the camera-motion check to `report.json` while the scoring loop is being written). `project.json` roughly doubles (200 KB → ~400 KB) because the raw 2D is kept — acceptable, and the price of never destroying detector output again. Rollback: each commit independently; `resolve_calibration(..., legacy_first_hit=True)`, `validate_cross_view(..., legacy_destructive=True)`, and deleting the `world_up` key.

---

## Phase 7 — Deferred, with the gate that would unblock each

No code. The deferral register is the deliverable — write it into `PROJECT_CONTEXT.md` or a `DECISIONS.md`.

| deferred | measured payoff | gate / what it waits on |
|---|---|---|
| **Halpe native NECK/PELVIS and foot keypoints** | NECK retarget 4.84 → 1.95 %; real toe/heel points for the feet | neck-Lshoulder CV must not exceed its 5.15 % COCO baseline (naive swap gives 8.13 %, so the neck **rest direction** must be re-derived first); `foot2d`/`foot3d` parallel arrays so `NUM_JOINTS` stays 15 |
| **Manual "level the figure" control** | beats the ±7 deg proxy floor | new UI; Phase 6.5 ships the recorded/auditable half without it |
| **Per-bone axial stretch (F08)** | retarget 1.51 → 1.19 % of height; elbows 1.90/2.17 → 0.29/0.28 % | required k spans 0.75–1.41 (chest 0.75, spine 1.41) — a visible reproportioning of a rig whose proportions the client accepted; also leaves the largest non-head residual (RIGHT_HIP 3.5 %) untouched |
| **Symmetrised fit targets (F34)** | none — measured, it moves the pose 0.31 mm and makes reprojection **worse** (7.44/4.19 vs 7.38/4.01 px) | do not build; the finding stands as an instrument, not a remedy |
| **Export in real-world units** | metric FBX/BVH | client question 2 (ruler); the export is currently scale-invariant, so this is a feature request |
| **Tag bundle adjustment (F38)** | shape 0.158 mm = 0.13 % of height; body epipolar slightly **worse** | provenance and the tag layout for gravity candidate (c) only — never sell it as accuracy |
| **Weighted DLT (F37)** | unmeasured | needs a probe; the right camera contributes half the angular resolution |

**Do not re-open:** extrinsics, intrinsics, triangulator choice, detector resolution, RTMPose-vs-red-dots (F05), derived-joint perspective bias, the `(NECK, HEAD)` bone (F35), the nose-pitch clamp on legacy projects (F03), `rig_bone_lengths` (F19), detector confidence (F26), and the essential-matrix fallback. Each has a bounding number in the audit.

Write the Halpe NECK/PELVIS gates as a **skipped, documented test body** (`tests/test_detect_halpe_full.py`) so the acceptance criteria are code, not prose.

---

## Verification

### Before/after on the client take

```
cd /media/athena/hd3/Projects/pose3d-tool
.venv/bin/python tools/measure_take.py workspace/pose3d_projects/Imported_Session --json > before.json
#   ... land a phase ...
.venv/bin/python tools/measure_take.py workspace/pose3d_projects/Imported_Session --json > after.json
.venv/bin/python tools/measure_take.py --diff before.json after.json
```

Until `tools/measure_take.py` exists (Phase 0), the audit's harness produces the identical numbers:

```
cd /media/athena/hd3/Projects/pose3d-tool          # cwd must be the repo root
.venv/bin/python <SP>/baseline/metrics.py workspace/pose3d_projects/Imported_Session
#   --no-retarget escapes if the character asset is missing
```

The harness reports, per source (`raw` / `fitted` / `refit` / `refit-nosmooth`): per-bone length median/std/CV, L/R symmetry, Sampson epipolar per joint, reprojection per joint per camera, retarget error as % of height, bend-plane angles, per-stage displacements, and rig geometry. It never imports `pose3d.app` (that pulls PySide6 at module scope).

**Phases that must be shape-neutral** (6.1–6.5, and the de-tilt half of 6.5): assert every T1/T2/T3/T5 number is **bit-identical**. **Phases that must move numbers** (1, 2, 3, 4, 5): the acceptance table in each phase names which, in which direction, and by how much — including the deliberate 1.51 → ~1.7 % retarget rise in Phase 1.

### On Panoptic

`data/demo_project` (ground-truth calibration, same detector, 10 frames, Panoptic cm) is the sanity reference, not a target: raw bone CV **3.4 % median / 9.0 max**, epipolar **6.8 px = 0.310 % of a 2203 px diagonal**, reprojection 5.31 / 4.34 px, retarget **5.0 % of height**. Its role in this plan is single and specific: **Phase 4's banding thresholds are chosen so that this take reads green** (today it reads ~45 % "Low", *worse* than the client's suspect rig). Note the caveats before comparing anything else: a moving human, 90 deg parallax, both shins and ankles absent, and only 3 valid left-arm frames.

```
POSE3D_REQUIRE_PANOPTIC=1 .venv/bin/python -m pytest tests/test_panoptic_geometry.py tests/test_panoptic_rtmpose.py -q
```

### CI regression harness

```
.venv/bin/python -m pytest tests/ -q                       # fast: fixture runs no detector, no Blender
POSE3D_REQUIRE_BLENDER=1 .venv/bin/python -m pytest tests/test_export_smoke.py tests/test_export_fidelity.py -q
POSE3D_REQUIRE_WEIGHTS=1 .venv/bin/python -m pytest tests/test_detect_accuracy.py -q
```

`tests/gates.py` skips locally and **fails** when the matching `POSE3D_REQUIRE_*` is set, which the Windows workflow does after installing each dependency. Remove the two `--ignore` lines at `.github/workflows/windows-test.yml:84-85` in Phase 0 or the Panoptic tests stay dead after renaming.

**Threshold discipline.** Set every geometric gate at **1.5× today's value or a documented physical bound**, never at today's value (raw CV median 5.27 against a 5.5 gate is a 4 % margin and will flake), and record today's number as a comment. The assertions with teeth are the ones that **fail today**: `reproj(fitted)/reproj(raw) ≤ 1.5` (4.24/6.17), `max|fitted − refit| ≤ 3 % of height` (21.8 %), largest non-root BVH translation ≤ 1 % of rig height (42.5 %), and `cos(displacement, motion) ≥ −0.1` (−0.976). Those four are the plan's regression net; everything else is a tripwire.

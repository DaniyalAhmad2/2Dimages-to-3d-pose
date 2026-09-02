# Pose3D accuracy audit and improvement plan

## Context

The tool reconstructs a 3D pose of a hand-posed figure from two static, uncalibrated cameras and drives a bundled rigged character with it. Extrinsics come from ArUco tags visible in both views. The client reports that the 3D character does not follow the detected keypoints accurately.

This plan is the result of an audit of the full keypoint-to-character control path, run against the real client take on disk at `workspace/pose3d_projects/Imported_Session` (26 image pairs, a ~12 cm red-dotted mannequin in front of four 6x6 ArUco tags on a curved green sweep). Six Opus auditors each ran experiments on that take; every finding was then independently re-derived by a second agent trying to refute it (35 confirmed, 5 refuted, 5 minor ones unverified). Three independent improvement proposals were then scored by two judges and merged. All numbers are measured on the client take unless stated. The full evidence (findings with corrected claims, baseline tables, auditor notes, the three proposals, judge scores, the synthesized long-form plan) is in the session scratchpad `/tmp/claude-1000/-media-athena-hd3-Projects-pose3d-tool/0fcb8c9e-52c1-483d-9e0a-80896b078fa5/scratchpad/` as `wf_*.md` and `wf2_*.md`; Phase 0 copies them into the repo so they outlive the session.

Objective metrics that need no ground truth (the mannequin is rigid): per-bone length spread across frames, left/right limb symmetry, body-keypoint epipolar error (independent of the tags), character-vs-capture joint error as % of body height, limb roll angle vs the anatomical bend plane. The audit's harness is `scratchpad/baseline/metrics.py`.

### Facts about the client take the plan depends on

- Left camera: 3072x4080 Pixel 10 Pro with EXIF (35mm-equivalent 24 mm, DigitalZoomRatio 1.26). True focal ≈ 3400–3600 px; the app assumed 4080 (+14–18 %).
- **Right camera: 1536x2048, no EXIF.** Assumed focal 2048 is right within ~3 %. The rig is asymmetric 2:1; anything that assumes one K or one pixel scale for both cameras is wrong.
- Shipped extrinsics = tag 14, frame 0001, first IPPE branch, both cameras. Not ambiguous there (second branch 9–12× worse) and within 1.3° of every other frame's solve. Cameras never moved (per-tag camera-centre std 1–5 mm).
- Left sees tags 13, 14, 17 every frame (15 in 6 frames); right sees only 14 (11 frames) and 15 (2 frames). Tag 13 is non-planar (3.3 px IPPE residual) and returns the wrong planar-pose branch in 35 % of frames; it is harmless only because the right camera never sees it. Tags are rotated 90°/113°/179° in-plane relative to tag 14 and are not coplanar.
- Metric scale hinges solely on the 0.05 m marker spinbox: nose-to-ankle is 0.120 m = 2.395 tag widths in both 3D and the raw image. The client calls the figure "about 20 cm" (client question 2). The character export is scale-invariant (`fit_to_subject` normalises it away), so this affects only the skeleton's metric units.
- The project predates the face-keypoint feature: `head2d`/`head3d` are absent, so the head-orientation code never runs on it.

## Audit result: why the character does not follow the keypoints

Ordered by measured contribution. Calibration and triangulation are at the noise floor; the damage happens **after** triangulation and in the retarget.

1. **The causal EMA smoother lags every frame behind the keypoints (F01, F09, F14, F10).** `fit_project` ends with `smooth_temporal(alpha=0.6)` on frames that are discrete hand-posed shots. It is 98 % of the difference between the shipped `fitted3d` and the bone-fitted measurement: 4.9 mm median / 25.7 mm max = 4.2 % / 21.8 % of body height, pure backward lag (cosine with motion −0.976). Reprojection of the displayed pose is 21.7/17.2 px vs 7.4/4.0 px without it; it re-breaks the bone lengths the fit just enforced (left forearm 19 % short in frame 13) and resurrects joints the fit deliberately left missing by copying the previous frame (20–39 mm off). A manual drag then replaces that frame with an unsmoothed single-frame refit, so a zero-pixel drag jumps every joint by up to 25.7 mm.
2. **Bone roll is undefined in the retarget (F02, F11, F12).** `_bone_fk` uses the minimal rotation from rest to aim direction, so twist about each bone is arbitrary: 13.6–53.6° off the anatomical bend plane. Hands and feet have no aim and inherit the parent's matrix, so feet render as spikes (sole tilt 30° median, 108° max) and hands splay; the view grounds on the lowest mesh vertex (a foot vertex), so the figure bobs 11.2 % of its height across the take; the hip line is 11.8° off because global yaw comes from the shoulders only. Positional metrics cannot see any of this. The auditor's `RollCharacter` prototype (aim + explicit roll reference) drives roll errors to 0° at a 0.1 pp positional cost; the contact-sheet renders confirm the feet and hands become plausible.
3. **The head feature never runs on imported takes, and HEAD is the nose (F16, F04).** `ImportDialog._process` has its own detection loop that writes only `kp2d`/`scores`, dropping the face keypoints `pipeline.detect_project` stores, so every imported project takes the legacy nose-aim path. The COCO-17 branch is always used (`feet=False`); Halpe-26 (already mapped in `skeleton.py`, weights in the local cache) has a native skull HEAD. Taking only that cuts HEAD retarget error 12.6 % → 3.9 % of height and neck-head length CV 9.5 % → 3.9 %; taking Halpe's native NECK regresses neck-shoulder CV 5.2 % → 8.1 %, so NECK/PELVIS stay derived.
4. **The exported file has zero root translation (F13, verified on `assets/Imported_Session.bvh`).** `to_rig` pins the pelvis at the rig's rest hips every frame: the BVH hips position is constant on all 771 motion rows while the pelvis travels 116 % of body height; the view discards the same motion by a different rule (preview/export placement gap up to 25 % of height). Two orphan IK helper bones (`shin.L/R.001`) carry up to 42.5 % of rig height of translation; the silent fallback on any export exception writes a null-rooted, position-channel-only skeleton (`data/demo_project/export/demo.bvh`). The in-between interpolation is fine (holds bit-identical, overshoot ≤ 3.8°, no quaternion flips).
5. **No gravity reference (F15).** The world frame is tag 14's (89° off vertical). `sequence_up` levels the take on its mean body axis, leaving a ~15° residual tilt vs three independent gravity proxies (cross of camera x-axes, mean camera up, tag row × mean normal, which agree within 6–14°). Per-frame lean is preserved and must stay so.
6. **The accuracy signals are blind (F07, F24, F25).** The gauge reprojects the raw two-view DLT point: bit-identical with the smoother on or off, and 52 % → 49 % under a 15° rig error. Thresholds are absolute pixels, so 73 % of joints read red on a good take and the ground-truth Panoptic reference reads red too. The two cameras' pixel units are averaged and painted on both views.
7. **Robustness defects that did not bite this take but will (F06, F23, F36, F18, F30, F17, F29, F21, F28, F41, F31, F27, F33, F42).** The cross-view gate NaNs detections in place and the loss survives save/reload (0 of 209 recoverable after a bad calibration is fixed); calibration takes the first frame's lowest common tag with no ambiguity or cross-frame check; the epipolar gate is 14.6× the observed error and sized from the left image only; fallback bone lengths are adult metres on a 4 cm limb (one blacked-out hip moves the *observed* joints 72 mm median); no provenance or marker size is persisted; `Intrinsics.load` labels guessed files "measured"; the EXIF focal ignores DigitalZoomRatio (computes 2833 instead of ~3570) and has no callers; the multi-tag path assumes one rotation for all tags and has no callers; blanket excepts in the view; fit failures go to print; drags leave derived NECK/PELVIS stale.
8. **Nothing can catch any of the above (F32, F45).** The synthetic test rig is a 3 m baseline at 720p with one symmetric K; no collected test runs the reconstruct path with an accuracy threshold; the Panoptic validators are not collected.
9. **Delivery gap (critic).** Opening a project never recomputes, nothing is versioned, and "Run Detection" overwrites hand corrections while leaving their flags set. A fixed build would show the client exactly what they complained about until they press Recalibrate.

### What is NOT the cause (measured; do not re-open)

- **Extrinsics.** Pooling all 97 tag observations into a bundle adjustment changes the shape by 0.13 % of body height; the rig is uncertain at ≤1.1°; a held-out 50 mm tag square reconstructs at 50.29 mm. (F38)
- **Intrinsics.** A best-fit fundamental matrix over the body keypoints reaches 3.80 px against the shipped rig's 4.91 px, so all calibration error combined explains ≤1.1 px; the true focal changes the shape by 0.44 % of span. Worth fixing for provenance, not for this complaint. (F20, F21)
- **Triangulator choice** (<0.4 mm), **detector resolution** (1 model px = 0.6–0.8 % of figure height), **derived-joint perspective bias** (0.25 mm), **bone-fit weights and root drift** (targets hit to 0.01 mm).
- **RTMPose vs the red dots** is a ~4 mm definitional offset against surface markers; snapping to the dots makes bone-length CV worse (5.3 % → 8.9 %). No marker detector. (F05 refuted)
- **The legacy head clamp** is not what puts the head 12.6 % off; unclamping triples the visible nose-to-mesh distance. Keep it on the legacy path. (F03 refuted)
- **Detector confidence** is inert, not misleading. (F26 refuted) **Symmetrising fit targets** moves the pose 0.3 mm and worsens reprojection. (F34) **Essential-matrix fallback** for a tag-less camera: 5.7–13.8° rotation error. (F42)

## Recommended plan

Winner of the design panel: the minimal-shippable proposal, with nine grafts from the other two applied (zero-phase filter kept off by default, visible gap flags, skull-HEAD-only Halpe, recorded gravity, two-number gauge, root motion on by default, per-image gate trap, antipodal test, report.json). Every phase is independently shippable with a rollback. Effort is for one developer.

| # | phase | effort | ships |
|---|---|---|---|
| 0 | Pin current behaviour: harness, fixture, export-fidelity script | M | no user-visible change |
| 1 | Remove the causal EMA; flagged gap fill; drag == batch | S | the 98 % item |
| 1b | Make it reach existing takes; stop dropping face keypoints | S | version stamp, non-destructive re-detect |
| 2 | Roll references, hip line, ankle grounding | M | the visible angular fix |
| 3 | The delivered file: root motion, helper bones, no silent fallback | M | export matches the view |
| 4 | Readouts that describe the delivered pose | M | the client can see a fix |
| 5 | Native skull HEAD, behind a gate table | M | worst bone in the baseline |
| 6 | Latent robustness, provenance, recorded vertical (six commits) | L | next take is diagnosable |
| 7 | Deferred register | S | written decisions |

If the client answers question 1 with "the exported FBX/BVH", move Phase 3 ahead of Phase 2.

### Phase 0 — Pin the current behaviour (F32, F45)

- Copy the audit evidence (`wf_*.md`, `wf2_*.md`, `baseline/metrics.py`, `find-retarget/roll_prototype.py`, `find-retarget/render.py`, `find-extrinsics/calib_ba.py`) from the scratchpad into `docs/audit-2026-09/` so it outlives the session.
- **New `pose3d/quality.py`** (Qt-free): `take_quality(project, rig, character=None) -> TakeQuality` with one body-height denominator (median z-extent of the de-tilted pose, 0.1178 m today), `figure_h_px[cam]`, reprojection of `pose3d` ("measured") and `fitted3d` ("delivered") per camera in px and as a fraction of figure height, Sampson epipolar median/p90/p99/max plus per-image point-to-line distances normalised by each image's own diagonal, per-bone length CV on the **raw** triangulation, L/R symmetry, gap counts, and with a `Character`: retarget % height, **limb roll error in degrees** (angle about the aim axis between the carried rest bend reference and the captured bend-plane normal), sole tilt, ground datum. Port the implementations verbatim from `metrics.py` so numbers stay comparable. Thin CLI `tools/measure_take.py [--json] [--diff a b]`.
- **New `tests/fixtures/client_take/`**: the 26 frames' `kp2d`/`scores`/`corrected`, the shipped calibration, and detected ArUco corners per (frame, camera, tag) (~250 KB, no images, no RTMPose at test time; `workspace/` is gitignored). Commit the regeneration script.
- **New `tests/test_client_regression.py`** with thresholds 10–15 % above today's values: raw bone CV median ≤ 6.0 % / max ≤ 10.5 %; L/R asymmetry ≤ 7.5 %; epipolar median ≤ 6 px, p90 ≤ 14 px; retarget median ≤ 2.0 % of height. Two assertions as `xfail(strict=True)` that Phase 1 flips: `reproj(fitted)/reproj(raw) ≤ 1.5` (today 4.24/6.17) and `max|fitted − refit(no smoothing)| ≤ 3 % of height` (today 21.8 %).
- `tests/synth.py`: add an **asymmetric** `close_range_two_cam(baseline=0.56, dist=0.53, f_left=4080, size_left=(3072,4080), f_right=2048, size_right=(1536,2048))` and parametrise `test_triangulate.py` / `test_cross_view.py` over both rigs.
- Rename `tests/validate_panoptic_*.py` → `test_panoptic_*.py`, add `gates.needs_panoptic()`, delete the two `--ignore` lines in `.github/workflows/windows-test.yml`, commit `data/demo_project/calibration/`.
- **New `tools/check_export_fidelity.py`** (dev script): export the fixture without video, parse the BVH, evaluate FK in numpy, and report deviation from `Character.posed_joints` at the captured keyframes, in-between overshoot, quaternion sign flips, largest non-hips translation channel, and what the fallback path ships. Encode today's facts as plain (non-Blender) assertions in `test_export_smoke.py`: 772 frames, holds bit-identical, overshoot ≤ 5°, hips constant, `shin.R.001` 42.5 % of rig height.
- Acceptance: `measure_take.py` reproduces the baseline to two significant figures. Risk: none to production code.

### Phase 1 — Remove the causal EMA (F01, F09, F10, F14, F27, F33, F44)

- Remove `smooth=True` at the **call sites** (`pose3d/ui/import_dialog.py:214`, `pose3d/ui/model.py:55`, `tests/build_demo_project.py:77`) and default `fit_project(..., smooth=False)`; flipping the default alone is a no-op.
- Re-implement `bonefit.smooth_temporal` **zero-phase** (forward EMA then the same EMA over the reversed sequence, NaN-aware), exposed as an import-dialog checkbox defaulted off ("video-rate capture only"), persisted as a top-level `project.json` key `smoothing: "none"`.
- **Delete `bonefit.py:140`** (`np.where(np.isnan(cur), prev, blend)`); keep line 141.
- **New `pipeline.fill_gaps(project, max_gap=1)`**: single-frame gaps flanked by observations are set to the neighbours' midpoint and flagged in a new persisted `Frame.filled` bool array; longer gaps stay NaN; filled joints draw hollow/amber in `view3d._draw_skeleton` and `camera_view` and are counted separately. `blender_export._character_bone_frames` consumes the same array so view and export agree on dropouts.
- **One fit function**: extract `pipeline.fit_frame(pose3d, bone_lengths)` and `pipeline.bone_length_targets(project)`; `fit_project` and `ProjectModel._resolve_joint` both call them; delete `ProjectModel._compute_bone_lengths` and the `_bone_lengths` cache. F10 dissolves by construction.
- Derived joints follow a drag (F33): in `_resolve_joint`, after the stack edit, re-derive NECK/PELVIS 2D from the current shoulder/hip 2D in the edited camera (unless that derived joint has its own `corrected` flag), re-triangulate it too, and record it on the same `Edit` so one Ctrl+Z reverses both (the pattern already used for the HEAD/nose sync at `model.py:137-149`).
- `fit_bone_lengths` solves only observed joints plus unobserved joints between two observed ones, freezes the rest and **drops bone residuals touching a frozen joint**, so `method="lm"` always applies (F44: 747 ms → ~21 ms).
- `fit_project` returns `FitReport(failed, first_error, fallback_bones, gaps_filled)` surfaced like `rejection_note` (F27).
- **Pre-merge probe (P2)**: re-triangulate the take ~20× under the detection auditor's bbox-jitter perturbation (1.3–2.8 px of 2D movement) and compare the per-joint 3D spread with the 9.36 mm median inter-frame motion. Under ~1 mm confirms no visible jitter; above ~3 mm, ship the zero-phase filter on at low alpha instead.
- Remove: the three call sites, `bonefit.py:140`, `_compute_bone_lengths`, the `print()`, `tests/test_bonefit.py:53-63` (`sm_var < raw_var`, the smoother's only guard).
- Tests: `tests/test_smoothing.py` (`test_no_backward_lag`: median cos(displacement, motion) ≥ −0.1 on a synthetic moving sequence, fails today at −0.976; `test_smoothing_is_off_by_default`; `test_nan_stays_nan`), `tests/test_gap_fill.py`, `tests/test_pipeline_fit.py::test_zero_pixel_drag_is_a_no_op` (real `ProjectModel`, every joint < 0.05 mm), `::test_derived_joint_follows_a_shoulder_drag`, `tests/test_bonefit.py::test_sparse_frame_uses_lm` (< 100 ms).
- Acceptance: reprojection 21.65/17.18 → **7.38/4.01 px**; bone CV 1.04 %/6.35 % → **0.14 %/0.37 %**; forearm frame 13 0.0173 → **0.0214 m**; zero-pixel drag 25.65 mm → **0**; invented joints 2 → **0** (2 flagged fills, each < 2 % of height from its neighbours); sparse fit ≤ 40 ms. Expect the retarget median to rise 1.51 % → ~1.7 % (the character now follows a pose that is on the keypoints) and inter-frame motion to rise 17 %; state both in the release note.
- Rollback: `project.smoothing = "ema"` restores a lag-free filter.

### Phase 1b — Make the fix reach existing takes (F16 + delivery gap)

- `io_project.save_project` writes top-level `pipeline_version: N`; `load_project` defaults it to 0.
- `app.build_model` / `main_window._load_model`: when `pipeline_version < CURRENT` and a rig exists, run `recompute_all()` once, stamp the version, and show a dismissible status line with the real numbers ("pose moved 4.9 mm median, 25.7 mm max; Restore stored pose"). Keep the stored `fitted3d` in memory for Restore; never auto-save.
- `pipeline.detect_project(..., respect_corrections=True)`: skip writes where `frame.corrected[cam][j]` is set (today "Run Detection" erases every correction).
- Delete the inline loop at `import_dialog.py:199-206` and call `pipeline.detect_project(project, det, cv2.imread, on_frame=cb)`.
- New menu action "Re-detect face points only" → `detect_project(..., fields="head")`, writing only `head2d`/`head_scores`/`head3d`. This is the migration path for the client's existing project. Do not touch `_NOSE_PITCH`/`_head_aim_target` (legacy path, F03).
- Tests: `test_import.py::test_import_populates_head_keypoints`, `test_pipeline_fit.py::test_redetect_keeps_corrections` (5/5 survive, 0 flags lie), `test_io_project.py::test_version_stamp_round_trips`, `::test_legacy_project_without_version_loads_and_recomputes`.
- Acceptance: opening the client project on the fixed build changes the pose with no button press and says why; head bone orientation on a fresh import differs from today's by 15.5° median.

### Phase 2 — Roll references, hip line, ankle grounding (F02, F12, F11, F43)

- **Rest references in `Character.__init__`** from `rest_joints()`: `upper_arm.*`/`forearm.*` → `unit(cross(elbow−shoulder, wrist−elbow))`; `thigh.*`/`shin.*` → `unit(cross(knee−hip, ankle−knee))`; `hips`/`spine` → rest hip line `unit(rj[R_HIP]−rj[L_HIP])`; `chest` → rest shoulder line; none for `neck`/`head` (the face basis already gives a full orientation), `clavicle.*`, `hand.*`, `foot.*` (no keypoints; once the parent's roll is right, theirs is).
- **`_bone_fk(b, base, end, ref_target=None, weight=1.0)`**: after `R = _align(rest_dir, aim)`, project both `R @ rest_ref[b]` and `ref_target` perpendicular to the aim axis `a`, take `ang = atan2(dot(cross(cur, tgt), a), dot(cur, tgt))`, and apply `R = rot(a, weight·ang) @ R`. This is `find-retarget/roll_prototype.py::_apply_roll`, which reproduces today's matrices bit-for-bit at weight 0, and it does not move any canonical joint (each child's head lies on the parent's axis).
- Capture targets supplied by `_skin_matrices` in rig space (`Rz` only; directions are scale/translation invariant): limb bones `Rz @ cross(mid−root, end−mid)` with all three joints valid; `hips`/`spine` the captured hip line; `chest` the captured shoulder line.
- **Stateless sign convention**: flip the normal so `dot(n, Rz @ shoulder_line) > 0` for arms and `dot(n, Rz @ hip_line) > 0` for legs; apply the same rule to the rest references. Stateless is what keeps `pose_bone_matrices` a pure function and view == export.
- **Continuous straight-limb weight**, not a guard: `w = clip((bend_deg − 20)/20, 0, 1)`. The left elbow's minimum captured bend is 22.7° and its bend normal jumps 55.7° between frames there; at 22.7° this gives w ≈ 0.14, which suppresses the pop. `hips`/`spine`/`chest` use weight 1 (Monte-Carlo at this take's noise perturbs the hip line 2.3–3.4° against an 11.8° error).
- **Ship gate**: max consecutive-frame roll change ≤ 15° on the fixture. If it fails, add a pure `pose3d/geometry/retarget.py::roll_references(poses3d, valid) -> (T,B,3)` doing the hemisphere fix plus a symmetric 3-tap smoothing of the reference vectors, called identically by the view's per-take precompute and by `blender_export`, passed as `pose_bone_matrices(..., roll_ref=...)`. Never carry per-bone roll in caller-owned temporal state: `view3d.set_pose` is driven per frame in scrub order.
- **Grounding**: replace `view3d.py:157` (`dz = verts[:,2].min()`) with `dz = min(posed_ankle_z) − ANKLE_SOLE_DROP/scale`, where the rig's rest ankle-to-sole height is 0.71217 rig units (4.94 % of rig height); fall back to the old rule when both ankles are NaN. Do **not** force soles horizontal (the shin genuinely tilts 16–86° on this rigid-footed mannequin).
- Antipodal determinism (F43): when a reference exists, `_align`'s 180° branch uses `axis = unit(cross(a, rest_ref[b]))`.
- Keep `_compute_rest_poles`/`_pole`/`_solve_ik` (the occlusion path, now more used).
- Tests in `tests/test_retarget.py`: roll matches a known 40° bend rotated 60° about the upper-arm axis within 2°; roll continuous across a 60° → 0° straightening sweep (≤ 15° per step); `posed_joints` unchanged to 1e-9; joints read back from `pose_bone_matrices` equal `posed_joints` on all 26 fixture frames; hip line follows capture; 179/180/181° changes roll < 5°; `test_view_orientation.py::test_grounding_is_ankle_based`. Visual gate: regenerate the contact sheets with `render.py` and review before merge (the defect is angular; no positional CI number sees it).
- Acceptance: roll error on `upper_arm.*`/`thigh.*` 53.5/13.6/21.4/27.1° → **≤ 5° median, ≤ 15° max**; `forearm.*`/`shin.*` recorded but **documented as a convention** (pronation is never observed); max consecutive-frame roll change ≤ 15°; `hand.L` axis spread 125.9° → ≤ 40°; hip line 11.8°/21.2° → ≤ 5°/≤ 15°; ground datum peak-to-peak 11.2 % → ≤ 0.5 %; retarget median ≤ 1.70 %; sole tilt: no assertion.
- Rollback: weight endpoints as module constants; weight 0 restores today's matrices bit-for-bit.

### Phase 3 — The delivered file (F13, F31 export half)

- **Root motion**: `Character.pose_bone_matrices(..., keep_root_motion=True, pelvis_ref=None)` adds `root_offset = Rz @ (pelvis − pelvis_ref) · scale` to the hips translation only (children inherit through the FK chain); `_from_rig` takes the same `pelvis_ref` so `posed_joints` and the overlay are unaffected; apply the same offset in `blender_job.py`'s fallback and delete the `COPY_LOCATION` hips pin. Default **on** for BVH/FBX, off for the turntable mp4.
- **The view uses the same rule**: centre once on the take's `pelvis_ref` instead of per frame on the mean of valid joints (`view3d.py:145`), keep the ankle z seat from Phase 2, size the grid and camera from the take's pelvis bounding box.
- **Helper bones pinned at rest, not omitted** (`shin.L/R.001`, `thigh.L/R.001`): omission changes armature topology.
- **No silent substitution**: replace the blanket excepts at `blender_export.py:44-53` with a typed reason in `ExportResult`; give `blender_job.main`'s outer except a loud marker the host reports as a failed export; refuse to write a file that is not the view without an explicit opt-in.
- **One frame per pose** becomes the BVH/FBX default (`schedule=None` is already supported at `blender_job.py:444`), so export frame k+1 = photograph k; the stepped hold/ease schedule stays for the mp4. Add a fixed-camera mp4 rendered from the left camera's pose alongside the turntable.
- Tests (`test_export_smoke.py`, `needs_blender`): root motion reaches the BVH (range = pelvis travel × scale within 1 %); helper bones ≤ 1 % of rig height (plain assertion); FK from the BVH at every captured keyframe vs `posed_joints` < 1 % of body height; no quaternion flips; overshoot ≤ 5°; missing character asset → `ExportResult.ok is False` with a message; one-frame-per-pose default. Run once with `POSE3D_REQUIRE_BLENDER=1` before merging.
- Acceptance: largest non-root translation 42.5 % → ≤ 1 % of rig height; hips travel 0 → the captured 0.136 m; preview-vs-export placement gap 25.5 % → 0; BVH-vs-view agreement measured for the first time and < 1 %.
- Rollback: `keep_root_motion=False` + stepped schedule reproduces today's file byte-for-byte.

### Phase 4 — Readouts that describe the delivered pose (F07, F24, F25, F31 app half, F34 diagnostic half, display halves of F30/F17)

- `ProjectModel._accuracy(idx) -> dict[cam, ndarray]`: **both** "measured" (`pose3d`) and "delivered" (`fitted3d`) reprojection, per camera, never averaged, each normalised by that camera's `figure_h_px`. Their difference is the permanent regression detector; after Phase 1 they coincide.
- `panels.accuracy_pct(err_px, figure_h_px)`: piecewise-linear on `err/figure_h` (100 % at 0, 85 % at 0.004, 70 % at 0.010, 0 % at 0.030), replacing `exp(−err/6)` and the 85/70 constants; `_refresh_timeline_status` swaps `< 5`/`< 12` px for `< 0.004`/`< 0.010` of figure height. Tune the cut points so the client fixture (after Phase 1) and the Panoptic reference read green and a 15° injected rig error reads amber/red (P6).
- Sidebar CALIBRATION section gains three rows from `take_quality`: bone-length CV ("a rigid subject should read near 0 %"), epipolar disagreement (px and % of each image's diagonal), L/R symmetry (flag at ~5 %, wording "check that keypoint in both images", never "this points at the calibration"). Plus ruler-checkable facts: baseline (55.9 cm), subject height (11.8 cm), marker size / tag id / source frame.
- `camera_view._joint_status` bands on its own view's normalised residual and paints "not measured", "interpolated" and "rejected by the cross-view check" as distinct states; remove the unreachable confidence fallback and dead `Detection.rag()`.
- Stop swallowing diagnostics: `app._load_rig` separates `FileNotFoundError` from parse/shape errors and validates `R` is 3×3 orthonormal; `Character.pose_and_joints` raises a typed `PoseUnavailable` for "no usable pelvis"; `View3D._skin` catches that narrowly and routes anything else to a `characterError` signal shown in the 3D card header; `fit_subject` failure gets a message.
- Tests: `test_camera_view_accuracy.py` (a good take is not majority red; bands are resolution-invariant under a K/kp2d similarity; a lagged pose scores worse), `test_quality.py` (bone CV rises ≥ 30 % under a 15° rig error; epipolar moves under a 0.7× focal), `test_ui_smoke.py::test_missing_rig_reports_a_reason`.
- Acceptance: fixture reads amber not 26/26 red; Panoptic ground truth reads green; a 15° extrinsic error moves at least one displayed row by > 30 %.

### Phase 5 — Native skull HEAD, behind a gate table (F04 separable half)

- `skeleton.map_halpe26(kp, scores, head="native", neck="derived", pelvis="derived")` as a policy dict: Halpe index 17 (skull vertex) as HEAD; NECK/PELVIS stay 2D midpoints (native NECK regresses neck-shoulder CV 5.2 % → 8.1 %). Fix the inverted docstring at `rtmpose.py:3-6`.
- Persist `project.json` `head_source: "nose" | "skull"` (absent → nose). Gate `_NOSE_PITCH`/`_head_aim_target` on it: skull HEAD aims directly, no 45° fudge, no clamp; legacy projects unchanged bit-for-bit, so Phase 1b's restored face keypoints and this cannot double-correct.
- Re-check `_JOINT_FROM_RIG[Joint.HEAD]` (head-bone mid vs tail) with the harness, but not on the head-mid metric alone (F03's refuter showed that metric misleads).
- **Build plumbing is mandatory**: drop the `--feet` guard in `tools/fetch_weights.py:69` (+55.7 MB), bump the CI cache key `rtmlib-weights-balanced-v1`, add the Halpe checkpoint to `selftest.py`'s presence check, the PyInstaller spec and Docker staging; otherwise the client build silently falls through to rtmlib's downloader.
- Re-baseline every % of height threshold in the same commit (height 0.1187 → 0.1303 m, +9.8 %).
- Gate table (all fixed before the run): HEAD retarget 12.65 % → ≤ 4.0 % (no face points) / 9.09 % → ≤ 3.0 % (with); neck-head CV 9.53 % → ≤ 4.0 %; neck-Lshoulder CV must not exceed 5.15 %; no body joint regresses > 0.5 % of height; head aim error < 5° with the nose path disabled; epipolar median no regression.
- Tests: `test_skeleton.py::test_map_halpe26` and `::test_head_neck_pelvis_policy` (none exist today), `test_io_project.py::test_head_source_round_trips`, `::test_halpe_frame_loses_nothing`.
- Rollback: the policy dict reverts HEAD to the nose in one line.

### Phase 6 — Latent robustness, provenance, recorded vertical (six independent commits)

All shape-neutral: assert every Phase 0 shape metric is **bit-identical** on the fixture before and after each commit.

- **6.1 Non-destructive gate (F06)**: `Frame` gains `kp2d_raw`, `scores_raw`, `rejected[cam]` (persisted; `load_project` back-fills `kp2d_raw = kp2d`); `validate_cross_view` sets the mask instead of NaN-ing; `triangulate_project` masks a local copy; `recompute_all` re-derives the mask from the raw arrays. Rejected joints draw in purple with the tooltip "rejected: the two views disagree by N px (gate M px)".
- **6.2 Data-driven, per-image gate (F36)**: `thr = clip(6 × median Sampson, 25 px, 1.4 % of the smaller image's diagonal)` (~25–30 px here, still 0/388 dropped); evaluate a point-to-line distance in each image against 1.4 % of that image's own diagonal and reject when **either** exceeds (the "both exceed" form is strictly more permissive than today). Report the threshold and distribution in the sidebar.
- **6.3 Proportional fallback bone lengths (F18)**: `fallback_bone_lengths(reference_m)` returning ratios of the measured pelvis-neck spine (else the largest measured bone); flip `bonefit.py:62` to `fill_missing=False`; `FitReport` names any bone that fell back. Whole-take blackout damage 225 mm → ≤ 2.5 mm.
- **6.4 IPPE branch scoring, tag admission, provenance (F23, F22, F30, F17, F29, F41, F42)**: `estimate_extrinsics_for_marker` uses `solvePnPGeneric` and returns both branches with errors; `resolve_calibration` detects tags in all frames, admits a tag only if its per-tag IPPE rms ≤ 1.5 px and ≤ 3× the median tag (tag 13 at 2.4–3.3 px vs 0.2–0.8 px, no overlap), scores the **joint** (left-branch, right-branch) pair on `err1/err0` per camera (reject < 2) and on cross-frame consistency of the relative pose (0–2° right vs 68° wrong; `L1R1` is 104° off with a plausible baseline, so baseline alone cannot check it). `save_rig` writes `calibration/report.json` (world tag/frame/dictionary, marker length, per-camera branch ratio, per-tag rms and admission reason, observation counts, baseline, convergence, world up + source + spread, focal source per camera, camera-motion check) sufficient to rebuild the rig bit-for-bit; `project.json` records `marker_length` and `keypoint_model`. `Intrinsics.load` defaults `source` to "unknown". Delete `estimate_extrinsics`' multi-tag branch. Failure message names which camera saw which tags in how many frames; no essential-matrix fallback. **Blocking test in the same commit**: on the fixture's corners the resolver still picks tag 14 / frame 0001 / branch 0 for both cameras (< 0.01°, < 0.1 mm from the shipped rig) and rejects tag 13 in 26/26 frames.
- Focal (F20, F21, F28): keep `f = max(w, h)` as the default with `source="assumed"` (it scores best on body epipolar, 4.91 px); fix `focal_from_exif` to apply DigitalZoomRatio (~3570 px), refuse it for a camera without EXIF, keep its test inverted to record why the assumption still wins; wire `calibrate_checkerboard` as the optional one-time override gated on `Intrinsics.source`; document that distortion only ever comes from a checkerboard and the principal point is unverifiable from tags. Scale (F17): persist the spinbox value; add "Set scale from a measured distance"; do not ship an inter-tag consistency check as a scale check.
- **6.5 Recorded vertical (F15)**: new `pose3d/geometry/gravity.py::estimate_world_up(rig, tag_layout=None)` from three proxies (`cross(x_left, x_right)`, `−mean(camera up)`, `cross(tag_row, mean_tag_normal)` once 6.4 has rejected tag 13), combined as a normalised weighted mean with the pairwise spread reported as the uncertainty (8–14° here); sign chosen so the take's mean NECK sits above its mean ANKLE (a binary test that cannot leak lean). `save_rig` writes `world_up`/`world_up_source`/`spread_deg`; `main_window._apply_view_orientation` and `blender_export.py:59-61` prefer it when the spread is under ~20° and fall back to `sequence_up` when absent (legacy projects bit-for-bit unchanged). Sidebar shows "Vertical: camera pair, ±9° vs the tag row". Document that `de_tilt_matrix` is one constant rotation, so per-frame lean is preserved and must stay so.
- Tests: `test_cross_view.py::test_a_bad_rig_does_not_destroy_kp2d` (12° error, save, reload, recompute, every observation returns), `::test_the_gate_uses_each_image_s_own_scale`, `test_bonefit.py::test_fallback_lengths_scale_to_the_subject`, `test_extrinsics.py::test_the_ambiguous_tag_is_rejected` (blocking), `::test_joint_branch_scoring`, `test_calib_report.py::test_report_json_rebuilds_the_rig`, `test_intrinsics_exif.py::test_digital_zoom_is_applied`, `test_orient_up.py::test_recorded_up_wins` / `::test_legacy_rig_unchanged` / `::test_per_frame_lean_is_preserved`.
- Acceptance: recoverable observations 0 → 100 %; gate 71.5 → ~25–30 px still dropping 0/388; blackout damage ≤ 2.5 mm; fixture resolves to its shipped extrinsics with tag 13 rejected; residual tilt 15.2° → ≤ 7° stated with uncertainty; every shape metric bit-identical.
- Sell it as "the next take is diagnosable", not "we fixed the calibration" (extrinsics are bounded at 0.15 % of height). `project.json` roughly doubles from keeping the raw 2D.

### Phase 7 — Deferred register (write into `DECISIONS.md`)

| deferred | measured payoff | gate |
|---|---|---|
| Halpe native NECK/PELVIS and foot keypoints | NECK 4.8 → 2.0 %; real toe/heel | neck-shoulder CV must not exceed 5.15 % (naive swap gives 8.1 %); `foot2d`/`foot3d` as parallel arrays |
| Manual "level the figure" control | beats the ±7° proxy floor | new UI |
| Per-bone axial stretch (F08) | retarget 1.51 → 1.19 % | needs k = 0.75–1.41, a visible reproportioning the client accepted not to have |
| Export in real-world units | metric FBX/BVH | client question 2 (ruler); currently a feature request, not a bug |
| Tag bundle adjustment (F38) | 0.13 % of height | provenance / tag layout only; never sell as accuracy |
| Weighted DLT (F37) | unmeasured | needs a probe |

Do not re-open: extrinsics, intrinsics, triangulator, detector resolution, RTMPose-vs-dots (F05), perspective bias, the (NECK, HEAD) bone (F35), the nose clamp on legacy projects (F03), `rig_bone_lengths` (F19), detector confidence (F26), symmetrised targets (F34), essential-matrix fallback (F42).

## Verification

Before/after on the client take, per phase:

```
cd /media/athena/hd3/Projects/pose3d-tool
.venv/bin/python tools/measure_take.py workspace/pose3d_projects/Imported_Session --json > before.json
# ... land a phase ...
.venv/bin/python tools/measure_take.py workspace/pose3d_projects/Imported_Session --json > after.json
.venv/bin/python tools/measure_take.py --diff before.json after.json
# until Phase 0 lands, the audit harness gives the identical numbers:
.venv/bin/python <scratchpad>/baseline/metrics.py workspace/pose3d_projects/Imported_Session [--source raw|fitted|refit|refit-nosmooth]
```

Phases 1–5 must move the numbers named in their acceptance tables in the stated direction; Phase 6 must leave every shape metric bit-identical. Panoptic (`data/demo_project`, ground-truth calibration) is the sanity reference and the anchor for Phase 4's banding, not a target (moving human, 90° parallax, shins absent).

CI: `.venv/bin/python -m pytest tests/ -q` (fixture runs no detector, no Blender); `POSE3D_REQUIRE_BLENDER=1 pytest tests/test_export_smoke.py`; `POSE3D_REQUIRE_WEIGHTS=1 pytest tests/test_detect_accuracy.py`; `POSE3D_REQUIRE_PANOPTIC=1 pytest tests/test_panoptic_*.py`. Every geometric gate sits at 1.5× today's value or a physical bound, with today's number as a comment. The assertions with teeth are the four that fail today: `reproj(fitted)/reproj(raw) ≤ 1.5`, `max|fitted − refit| ≤ 3 % of height`, largest non-root BVH translation ≤ 1 % of rig height, `cos(displacement, motion) ≥ −0.1`.

Visual gates (no CI number sees them): Phase 2's contact sheets (`render.py`, shipped vs fixed, 6 frames × 2 azimuths) and Phase 3's fixed-camera render against the photographs.

## Questions for the client (defaults in brackets)

1. **Which file were you judging: the live 3D view, the exported FBX/BVH, or the mp4?** Export has zero root motion and puts pose k at frame 1+30k; the mp4 is a turntable. [Live view; if export, Phase 3 moves ahead of Phase 2.]
2. **Printed tag edge and mannequin height, measured with a ruler.** Every millimetre figure hangs on the 0.05 m spinbox; the reconstruction is 2.395 tag widths tall. Do you need the export in real units? [Keep 50 mm, persist it, show baseline and height in the sidebar, no real-units export.]
3. **Can the right camera see at least two tags?** The whole calibration hangs on one tag in one of 11 frames. [Ship Phase 6.4 anyway with the camera-naming failure message.]
4. **Can the tags be mounted flat and rigid?** Tag 13 is bent over the curve and ambiguous in 96 % of solves. [Auto-reject by planarity residual.]
5. **One-time checkerboard per camera, ever?** Buys provenance, not a visible improvement on this take (≤1.1 px of 4.91). [Keep the assumed focal, `source="assumed"`, checkerboard as an unused override.]
6. **Should the exported character travel, one frame per photograph, and a fixed-camera render?** [Root motion on for BVH/FBX, one frame per pose, fixed-camera render alongside the turntable.]
7. **Stop-motion only, or video-rate takes too?** [Smoothing off; zero-phase filter kept as an unchecked option.]
8. **Are feet and hands worth a 56 MB checkpoint and a skeleton re-baseline?** [Take the head half now, defer feet.]
9. **What are you filming next** (this mannequin, real people, camera moves, several figures)? [Same rig; thresholds tuned on this take plus Panoptic.]
10. **What would "accurate enough" be, as a number or a photograph?** [Delivered pose reprojects as well as the raw triangulation; the ~5 % detector floor is out of scope.]
11. **Is a one-off pose change on existing projects acceptable** (~5 mm typical, 26 mm worst, explained, with Restore)? [Yes, recompute on open, never auto-save.]

## Open probes the plan does not settle

- **P1 export fidelity**: FK from the delivered BVH vs `Character.posed_joints` at the 26 keyframes has never been computed; Phase 0's script measures it before Phase 3. Blender is at `/home/athena/Downloads/blender-5.1.1-linux-x64/blender`.
- **P2 jitter**: the noise-floor probe in Phase 1 (bbox-jitter re-triangulation vs inter-frame motion); definitive answer needs two photographs of one held pose from the client.
- **P3 NaN frames on the retarget**: pose frames 0012/0021 with the NaN preserved and report bone direction change vs neighbours (> 10° or > 5 % of height means gaps ship as visible holes with the bone held at its previous local rotation).
- **P4 roll continuity**: run `roll_prototype.py` with the weight and sign rule over all 26 frames before Phase 2; the decisive check is the contact sheet.
- **P5 absolute scale**: tabulate marker 50/62/75 mm against EXIF SubjectDistance 0.470 m and the client's "~20 cm"; confirm by experiment that scaling the triangulation 1.5× leaves the BVH bit-identical.
- **P6 banding honesty**: sweep cut points against the fixture after Phase 1, Panoptic, and an injected 15° error; if no pair satisfies all three, band on a second axis (bone CV).
- **P7 representativeness**: one take, 26 frames, zero corrections, cameras never moved. A second take with the right camera seeing two tags and one pose photographed twice; meanwhile add a camera-motion check (per-tag camera-centre std, flag > ~2 cm) to `report.json`.
- **P8 the real widget**: a `pytest-qt` smoke test driving `CameraView` on the fixture (60 px shoulder drag → whole-take refit to 1e-9, NECK re-derived, one Ctrl+Z restores both, no re-entrancy warning).

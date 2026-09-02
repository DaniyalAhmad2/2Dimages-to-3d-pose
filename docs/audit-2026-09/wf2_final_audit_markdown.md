# Audit — why the character does not follow the keypoints

Take under test: `workspace/pose3d_projects/Imported_Session`, 26 stop-motion frames, 0 corrections, shipped single-tag calibration. All percentages are of the **117.8 mm** de-tilted body height (z-extent of `fitted3d`) unless stated. Every number below is measured, not estimated; finding ids refer to `wf_confirmed_full.md`.

---

## 1. Root causes, ordered by contribution

### 1.1 The causal EMA smoother — 98 % of everything the pipeline adds after triangulation (F01, F09, F14, F10)

`bonefit.smooth_temporal` (`pose3d/geometry/bonefit.py:129-143`), called from `pipeline.fit_project:183`, blends 40 % of the **previous, unrelated hand-posed frame** into every frame. On a stop-motion take there is no temporal signal to filter, so the entire effect is lag:

| | measured |
|---|---|
| displacement of the delivered pose from its own bone-fitted measurement | 4.92 mm median, 6.07 mean, 12.77 p90, **25.65 mm max** = 4.2 / 5.2 / 10.8 / **21.8 % of body height** |
| share of the total post-triangulation displacement (6.22 mm) | **98 %** (bone fit contributes ~0.15 mm net) |
| direction | pure backward lag: cos(displacement, motion) median **−0.976**, negative in 95.4 % of 371 samples; magnitude matches the closed-form EMA steady-state lag to 0.7 % |
| reprojection onto the source photos | **21.65 / 17.18 px** (L/R) vs 7.38 / 4.01 unsmoothed and 5.11 / 2.78 raw — 4.2× / 6.2× worse; worst single joint 167.8 px |
| bone-length CV on a rigid mannequin | **1.04 % median / 6.35 % max** vs 0.14 % / 0.37 % unsmoothed — it re-breaks the bones the fit enforced one line earlier (left forearm **19 % short** in frame 0013: 0.0173 m vs 0.0214 m) |
| bone **directions** — the quantity `character.py` aims each rig bone along | rotate 4.4 deg median, 11.5 p90, **27.8 deg max** away from the captured direction |
| benefit bought | 17 % less inter-frame motion (9.16 → 7.60 mm) on a take whose 9.36 mm median inter-frame motion is deliberate re-posing, not noise |

Two aggravating consequences of the same function:

* **It invents joints** (F09). `bonefit.py:140` substitutes the previous frame wherever the current frame is NaN. On this take that fabricates 2 joints (frame 0012 `RIGHT_KNEE`, frame 0021 `LEFT_ANKLE`), each a bitwise copy landing **20.21 mm / 38.50 mm (17.2 % / 32.7 % of height)** from the next real observation, displayed as measured, and it suppresses the two-bone IK fallback that exists for exactly this case.
* **It makes the manual-correction path inconsistent** (F10). A drag re-fits one frame with *no* smoothing (`model.py:157-160`), so a **zero-pixel drag moves every joint** — 4.92 mm median, 25.65 mm max, 15 of 15 joints — and Ctrl+Z does not restore the smoothed pose. This is not a separate defect: with the smoother gone the drag path and the batch path are the same computation.

Do **not** mistake re-ordering for the fix (F14): smoothing *before* the fit repairs the bone metric (CV 0.139 %/0.298 %) and leaves the lag untouched (6.09 mm vs 6.07 mm).

### 1.2 Undefined bone roll — the largest *angular* error, invisible to every positional metric in the repo (F02, F12, F11, F43)

`_bone_fk` (`character.py:488`) orients each bone with `_align(rest_dir, aim)` — the **minimal** rotation — so rotation about the bone's own axis is whatever the cross product happens to give. On the executed path of both the 3D view (`view3d.py:204`) and the export (`blender_export.py:87`).

| bone | roll error vs the captured bend plane, median / max (deg) |
|---|---|
| upper_arm.L | **53.5 / 72.8** |
| thigh.R | 27.1 / 54.0 |
| shin.R | 25.4 / 48.3 |
| thigh.L | 21.4 / 37.3 |
| forearm.L | 20.4 / 49.0 |
| forearm.R | 14.4 / 30.1 |
| shin.L | 13.8 / 23.4 |
| upper_arm.R | 13.6 / 41.2 |

`hand.L/R` and `foot.L/R` have no `_DIRECT` entry and inherit the parent matrix verbatim in 26/26 frames (`character.py:667`), so they inherit all of it — `hand.L`'s posed axis spreads **125.9 deg** across the take. The canonical joints move **0.0000 mm** when roll is corrected, which is exactly why no metric in the repo, and no acceptance number in the baseline, can see this. The mesh moves 0.77 mm frame-median, 14.5 mm worst (12.3 % of height), concentrated in feet and hands.

Two companions in the same mechanism:

* **Hip line** (F12): the character's hip line is **11.76 deg median / 21.18 max** off the captured hip line, in 23 of 26 frames, because the global yaw `Rz` is fitted to the **shoulders only** (`character.py:561-571`) and the hips bone takes a minimal rotation — measured |character hip yaw − character shoulder yaw| = 0.30 deg, i.e. the hips simply carry the shoulders' yaw, while the captured hip line is genuinely twisted 10.0 deg median (21.7 max) from the shoulders. That twist is signal: a realistic 2–3 px 2D perturbation moves it only 2.3–3.4 deg.
* **The ground datum** (F11): `view3d.py:157` grounds on `verts[:,2].min()`, which is a **foot vertex in 26 of 26 frames** and correlates r = 0.87 with the grounding-side shin's tilt (which genuinely ranges 16–86 deg). The whole scene therefore translates against the fixed grid by 7.3 % of character height median, **9.7–11.2 % peak-to-peak**. This — not the foot's orientation — is what produces the "spike feet resting nowhere" impression; a roll fix leaves sole tilt essentially unchanged (L 38.5 → 39.6, R 42.7 → 43.8 deg).

### 1.3 The delivered file loses the subject's translation and carries channels nothing can interpret (F13, F31)

Measured directly from this take's own export, `assets/Imported_Session.bvh` (I parsed it; 772 frames, 30 fps, 26 poses):

* **Hips translation is constant at `0.000000 −0.131342 2.851750` on all 771 motion rows — exactly zero root travel** — while the de-tilted pelvis moves **0.1362 m = 115.7 % of the subject's own body height**, monotonically +129.6 mm in x across the take, present identically in the raw triangulation (so it is real subject motion, not drift). No export path in the repo carries root translation: `character.py:573-574` pins it and `blender_job.py:535` pins the fallback with a `COPY_LOCATION`.
* **Two orphan IK helper bones ship live translation channels.** `shin.L.001` and `shin.R.001` are written as top-level roots (siblings of `hips`) carrying position ranges of 2.17/4.18/1.77 and 1.27/6.13/3.02 rig units — up to **42.5 % of the 14.4228 rig height** — on bones the app never drives and that carry no skin weight.
* **The view and the export place the figure by different rules.** The view re-centres per frame on the mean of the valid joints (varies 6.8 mm = 5.8 % of height) and re-grounds on the lowest mesh vertex, whose rig-space swing is 3.6735 units = **25.5 % of rig height**; the export grounds nothing.
* **A missing character asset silently ships a different animation** (F31). `blender_export.py:44-53` returns `(None, None)` on any exception; `blender_job.py:616-625` then substitutes an aim-only Damped-Track retarget, and `blender_job.main`'s outer `except` writes the shape visible in `data/demo_project/export/demo.bvh` — a null `ROOT __0` with `CHANNELS 0` and 15 sibling joints, all motion in position channels, which no DCC retargeter can consume. The documented "the export is pose-identical to the view" invariant can be broken with no message.

**Measured and NOT a defect:** Blender's in-between interpolation. Each captured pose holds **bit-identically for 22 frames (0.0 deg deviation across all 189 rotation channels)**, and the 8 eased in-between frames overshoot their bracketing keyframes by at most **3.795 deg** (p99 2.93; **0 of 25 transitions above 5 deg**, i.e. no quaternion hemisphere flips). Do not budget interpolation rework. What *is* wrong is that captured pose *k* lands at frames 1+30*k*…22+30*k* with nothing in the deliverable saying so, so a photo-to-frame comparison is misaligned by construction.

### 1.4 The head feature is dead on every imported project (F16)

`import_dialog.py:199-206` runs its **own** detection loop that writes only `kp2d`/`scores` and drops `Detection.head_xy`/`head_scores` on the floor, while the canonical `pipeline.detect_project` (`pipeline.py:38-46`) writes all four. Every project created through Import Images therefore has an all-NaN `head2d`, `head3d` is all-NaN, and the ear/eye head basis added in commit 5fdd919 has never run on an imported take. Restoring it rotates the head bone by **15.5 deg median / 23.9 p90 / 28.5 max** and moves the head mesh **6.3 % median / 12.2 % max of body height**, leaving the body pose bit-identical (0.0000 m). The only in-app route to face keypoints today is "Run Detection", which overwrites every hand-corrected 2D point while leaving its `corrected` flag `True`.

### 1.5 A canonical HEAD that is the nose (F04)

`RTMPoseDetector` defaults `feet=False` (`rtmpose.py:26`; its own module docstring at lines 3-6 asserts the exact inverse), so HEAD is the COCO nose. HEAD retarget error is a **systematic 12.6–12.9 % of body height** in every source (median ≈ max, so definitional, not noise), and the **neck-head bone-length CV on a rigid mannequin is 9.53 % — the worst bone in the entire baseline**. A native skull HEAD takes it to 3.93 % (2.92 % with face keypoints) and the CV to 3.85 %.

### 1.6 The client cannot see any of this, and neither could we (F07, F24, F25, F32)

The Pose Accuracy gauge scores `f.pose3d` (`model.py:180-191`) — the **raw** triangulation — while everything drawn (`main_window.py:450`) and exported (`main_window.py:355`) is `f.fitted3d`. The 6 mm mean / 26 mm max the smoother added was invisible **by construction**. On top of that the gauge is `100·exp(−err_px/6)` in **absolute pixels** (`panels.py:33`) with `< 5 px / < 12 px` timeline cuts (`main_window.py:483`), so on a 3072×4080 phone frame green requires ≤ 0.98 px: this take reads **52.3 % "Low", 26 of 26 frames red, 284 of 390 joint dots red (72.8 %)** — and the *ground-truth-calibrated* Panoptic reference reads **worse (~45 %)**. The identical reconstruction rescaled to 1080p reads 84.0 % with 21 of 26 frames green. Meanwhile the two cameras' raw pixels (5.11 vs 2.78 px) are incommensurable — the figure is 776 px tall left, 407 px right — and `_accuracy` `nanmean`s them into one number painted on both views. Normalised by figure height they agree to 4 % (0.66 % vs 0.69 %). Finally (F32) no test in CI runs calibration→retarget on realistic data with an accuracy threshold; every geometry test passes **one** `Intrinsics` for both cameras, so the client's 2:1 asymmetric rig is structurally untestable, and the two Panoptic validators are explicitly `--ignore`d in `.github/workflows/windows-test.yml`.

---

## 2. What is NOT the cause

| Suspect | Bounding measurement | Verdict |
|---|---|---|
| **Extrinsics** | Pooling all 97 tag observations in a bundle changes the reconstruction's **shape by 0.158 mm = 0.13 % of body height**; the rig is uncertain at ≤ 1.1 deg; a tag-15 cross-view holdout square reconstructs at **50.29 mm against 50.00 nominal** (F38, F22) | Not the cause. The shipped tag-14/frame-0001/branch-0 choice is, on this scene, the exhaustively-verified right answer |
| **Intrinsics / focal** | All calibration error combined explains **≤ 1.11 px of the 4.91 px epipolar median**, against a 3.80 px best-fit-fundamental noise floor; the assumed f = max(w,h) scores 4.91 px, better than every physically-motivated alternative tried (best sweep optimum (3672, 1638) = 5.37 px; EXIF pair 10.9 px). True-focal reconstruction differs by 0.44 % of span (F20, F21) | Not the cause |
| **Triangulator choice** | Three triangulators differ by **< 0.4 mm**; Hartley–Sturm makes bone CV worse | Not the cause |
| **Stereo conditioning** | 0.559 m baseline, 62.5 deg median ray parallax, **0.136 mm of depth error per pixel** of 2D error | Not the cause |
| **Detector resolution** | 1 model pixel = 0.63–0.84 % of figure height; crop and box-jitter probes move keypoints **< 1 %** | Not the cause |
| **RTMPose vs the moulded red dots** | ~4 mm mostly-uniform definitional offset; the dots are **1.8× noisier** as a reference (bone CV 9.18 % vs 5.07 %) | Refuted (F05) |
| **Derived NECK/PELVIS construction** | Triangulating the midpoint-of-projections instead of the midpoint of triangulated endpoints costs **0.25 mm at NECK, 0.08 mm at PELVIS** | Not the cause |
| **Cross-view gate** | Inert on this take: **0 of 388 pairs** exceed the 71.5 px threshold | Not the cause *today* — see F06/F36 for why it is a hazard |

The remaining irreducible floor is RTMPose's own 2D error on a grey mannequin: **raw bone-length CV 5.3 % median / 9.5 % max**, driven by roughly 5–15 px of 2D keypoint error and consistent with the measured 4.91 px epipolar median. Nothing in this plan moves that floor. For scale, the ground-truth-calibrated Panoptic reference with the same detector reads 3.4 % median / 9.0 % max.

---

## 3. Confirmed findings

Severity is refuter-adjusted. "Impact" is the measured magnitude on the client take.

| id | file:line | sev | corrected claim (one line) | measured impact | fix (one line) |
|---|---|---|---|---|---|
| **F01** | `pipeline.py:183` | critical | The causal EMA lags the delivered pose behind the keypoints; `smooth=True` is passed **explicitly** at both call sites, so flipping the default is a no-op | 4.92 mm med / 25.65 max = 4.2 / 21.8 % of height; 98 % of all post-triangulation displacement; reproj 21.65/17.18 vs 7.38/4.01 px | Stop calling it at `import_dialog.py:214` and `model.py:55`; keep a zero-phase filter off by default |
| **F02** | `character.py:488` | major | Every bone's roll about its aim axis is unconstrained; hands/feet inherit it; invisible to every positional metric (joints move 0.0000 mm) | limb roll 13.6–53.5 deg median, 72.8 max; `hand.L` axis spread 125.9 deg; mesh 14.5 mm worst | Explicit per-bone roll reference from the rest/captured bend plane, applied about the aim axis |
| **F04** | `rtmpose.py:26` | major | `feet=False` everywhere, so HEAD is the nose and NECK/PELVIS are 2D midpoints; the docstring states the inverse | HEAD retarget 12.65 → 3.93 % of height; neck-head CV **9.53 → 3.85 %** (worst bone); neck-Lshoulder **regresses** 5.15 → 8.13 % | Native skull HEAD via a head/neck/pelvis policy dict; keep derived NECK/PELVIS |
| **F06** | `pipeline.py:100` | major | `validate_cross_view` NaNs `kp2d` in place; the loss survives save/reload and recalibration cannot undo it | 12 deg extrinsic error drops 185/780; **0 recovered** by recalibrate; rejected joints hidden in the 2D view | `kp2d_raw` + a `rejected` mask re-derived on every recompute |
| **F07** | `model.py:187` | major | The gauge scores `f.pose3d`, not the `f.fitted3d` that is drawn and exported, and is blind to a wrong rig | gauge 52.3 % on this take; a 15 deg extrinsic error (19.13 mm = 16.2 % of height) moves it 52.3 → 49.4 and never leaves "Low" | Score the delivered pose; add bone-CV + epipolar rows |
| **F10** | `model.py:160` | major | The drag path writes an unsmoothed single-frame refit into a smoothed sequence | zero-pixel drag: 4.92 mm med / 25.65 max, 15/15 joints move | Dissolves once F01 lands: one shared `fit_frame` |
| **F13** | `character.py:574` | major | Both export paths pin the root; the view discards the same motion by a different rule | hips translation range **exactly [0,0,0]** vs 0.1362 m pelvis travel = 115.7 % of height; view/export vertical gap 25.5 % of rig height | `keep_root_motion` offset on the hips matrix; one shared placement rule |
| **F14** | `pipeline.py:182` | major | Re-ordering smoothing before the fit repairs bone lengths but not the lag | lag 6.09 mm vs 6.07 mm — unchanged | Do not re-order; remove the smoother |
| **F15** | `orient.py:78` | major | No gravity reference; `sequence_up` is the subject's own body line. (Corrected: it removes only the **mean** lean — per-frame lean is preserved to 1.5e-11) | world up 89.34 deg off vertical; `sequence_up` 15.20 deg from the 3-proxy consensus; proxies disagree 8.1–14.2 deg | Record an explicit `world_up` + source + spread; fall back to `sequence_up` |
| **F16** | `import_dialog.py:204` | major | The import dialog's private detection loop drops face keypoints, so the head feature is dead on every imported project | head bone 15.5 deg median / 28.5 max; head mesh 6.3 / 12.2 % of height; body pose unchanged | Call `pipeline.detect_project`; add a face-points-only migration |
| **F18** | `bonefit.py:46` | major | `_FALLBACK_LENGTHS` are adult metres on a 37 mm limb, and they sit in the least-squares regardless of `fill_missing` | 12× too large; one blacked-out hip moves the **still-observed** joints 71.6 mm med / 279.6 max = **237 % of height**; 2.3 mm with a proportional table | Proportions of a measured reference; default `fill_missing=False` |
| **F23** | `resolve.py:143` | major | First-hit lowest-id common tag in the first frame, no IPPE ambiguity or cross-frame check | tag 13 ambiguous in 96 % of solves, wrong branch in 9/26; taking the runner-up branch for tag 14 drops **367/780 (47 %)** permanently | Score both branches jointly across cameras and frames; admit tags by planarity rms |
| **F24** | `panels.py:33` | major | Absolute-pixel accuracy bands saturate red at phone resolution | 26/26 frames "Low", 284/390 dots red (72.8 %); same take at 1080p reads 84.0 % with 21/26 green; Panoptic GT reads ~45 % | Band on error / figure height |
| **F32** | `tests/synth.py:47` | major | No CI test runs calibration→retarget with an accuracy threshold; the synthetic rig is symmetric and far-field; the Panoptic validators are `--ignore`d | none of the defects above move a single CI number | Commit the client fixture; add an asymmetric close-range rig; collect the validators |
| **F08** | `character.py:353` | minor | Only a uniform scale is fitted; rig segment ratios differ from the mannequin's by up to 25 % | retarget 1.51 → 1.19 % of height with per-bone stretch, but required k spans 0.75–1.41 (visible reproportioning) | Defer; optional axial-stretch flag at most |
| **F09** | `bonefit.py:140` | minor | The smoother resurrects joints the fit refused to invent and shows them as measured | 2 joints; 20.21 / 38.50 mm = 17.2 / 32.7 % of height off the next real observation | Delete the line; ship an explicit, flagged interpolation |
| **F11** | `view3d.py:157` | minor | Grounding uses the lowest mesh vertex (a foot vertex 26/26) so the figure bobs | 7.3 % median, 9.7–11.2 % peak-to-peak of character height; r = 0.87 with shin tilt | Ground on the ankle minus the rig's constant 0.7122-unit rest sole drop |
| **F12** | `character.py:564` | minor | The character's hip line carries the shoulders' yaw | 11.76 deg median / 21.18 max; captured hips genuinely 10.0 deg median off the shoulders | Hips roll reference = captured hip line |
| **F17** | `import_dialog.py:108` | minor | Metric scale rests on an unvalidated 0.05 m spinbox that is never persisted and cannot be checked | reconstruction 0.118 m vs a figure described as ~20 cm; **the export is scale-invariant** (bone matrices differ 1.03e-3 rig units between 0.05 and 0.10 m) | Persist it; show baseline + subject height in cm; real-units export is a feature, not a bug |
| **F20** | `resolve.py:116` | minor | Focal source is decided per camera and can mix provenance across two different devices | mixing produced the 26.6 px epipolar failure reverted in f4c92f9 | Rig-level focal ladder; refuse to mix provenance |
| **F21** | `intrinsics.py:99` | minor | `focal_from_exif` ignores DigitalZoomRatio, so its 2833 px was **21 % short** of the true ~3570 px — f4c92f9 rejected a number computed wrong | assumed 4080 still scores best (4.91 px) | Apply tag 41988; keep the assumption as default; record `source` |
| **F22** | `resolve.py:81` | minor | Tag 13 does not image as a planar square | IPPE residual 2.4–3.3 px vs 0.2–0.8 px for tags 14/15/17; obliquity is *not* the discriminator | Admit tags by per-tag rms before any bundle |
| **F25** | `model.py:180` | minor | The two cameras' pixel errors are averaged and painted on both views | 5.11 vs 2.78 px raw = 1.84× framing artefact; 0.66 % vs 0.69 % once normalised | Per-camera normalised arrays |
| **F27** | `pipeline.py:179` | minor | Fit failures are `print()`ed to a stdout a windowed build redirects | silent on the affected takes | `FitReport` surfaced like `rejection_note` |
| **F28** | `resolve.py:70` | minor | `calibrate_checkerboard` has zero callers; nothing can ever improve the intrinsics | the "intrinsics were assumed" warning is permanent by construction | Wire it as the optional one-time override |
| **F29** | `intrinsics.py:58` | minor | `Intrinsics.load` defaults `source="measured"` | re-saving the client's own guessed file stamps `"measured"` onto disk | Default `"unknown"`/`"assumed"` |
| **F30** | `resolve.py:44` | minor | No calibration provenance is persisted | the shipped rig's provenance was only recoverable by brute-forcing 97 IPPE solves | `calibration/report.json` |
| **F31** | `view3d.py:99` | minor | Blanket `except`s make a missing asset look like a bad reconstruction — and in `blender_export` swap the exported retarget | `fit_to_subject` returning `None` silently pulses the character over a 37.9 % range | Typed failures + a reported reason |
| **F33** | `model.py:150` | minor | Drags leave the derived NECK/PELVIS stale; no recompute repairs it | 60 px drag → NECK 30.0 px stale in 208/208 cases, 3.31 mm med / 10.66 max (2.8 / 9.0 % of height) | Re-derive dependents on the same stack edit |
| **F34** | `pipeline.py:158` | minor | Fit targets are per-take medians, so upstream asymmetry is baked in | 6.7 % forearm / 4.6 % thigh L/R asymmetry. **Do not symmetrise** — measured, it moves the pose 0.31 mm and makes reprojection worse | Report it as a diagnostic instead |
| **F36** | `pipeline.py:53` | minor | The epipolar gate is 14.6× the observed median and scaled from the left image only | 71.5 px vs 4.91 median / 23.97 p99; catches the ankle-on-knee hallucination its own docstring names in only 15/26 frames | Data-driven threshold; per-image units, reject if **either** view exceeds |
| **F39** | `resolve.py:89` | minor | Zero distortion is nearly free for the subject but not for the tags | k1 = 0.03 moves tags 10.9 px, subject 0.27 px | Never fit k1 from tags; validate by subject displacement |
| **F40** | `resolve.py:88` | minor | The principal point is unrecoverable from the tags and the geometry is most sensitive to it | ±50 px = 1.4 px epipolar; ±200 px = 19 px | Never free it in a bundle; say so in `check_rig` |
| **F41** | `extrinsics.py:132` | minor | The multi-tag branch assumes one shared rotation for all tags; zero callers | wrong by up to 179.4 deg on this curved sweep; camera-centre errors 527–1020 mm | Delete it; a bundle estimates the layout |
| **F45** | `tests/test_detect.py:43` | minor | No test measures 2D keypoint accuracy | shapes and plumbing only | Gated accuracy test on 2–3 client frames |
| F37 | `triangulate.py:44` | minor, unverified | The right camera contributes half the angular resolution and the DLT is unweighted | — | Weighted DLT or documented null result |
| F38 | `resolve.py:134` | minor, unverified | The rig is solved from 1 of 97 tag observations; pooling is cheap but changes shape by 0.13 % | 0.158 mm | Bundle for provenance/robustness, not accuracy |
| F42 | `resolve.py:155` | minor, unverified | A camera that never sees a tag has no recovery path, and the essential-matrix fallback is not accurate enough | 5.7–13.8 deg rotation error, scale unrecoverable | Keep the failure, fix the message |
| F43 | `character.py:136` | minor, unverified | `_align`'s near-180 deg branch is non-deterministic (not reached on this take) | a 1 deg wobble at exactly 180 deg flips roll 163.8 deg | Subsumed by an explicit roll reference; add a regression test |
| F44 | `bonefit.py:121` | minor, unverified | The `trf` fallback for sparse frames is 34.8× slower than `lm` | 746.78 ms vs 21.47 ms per frame | Solve only observed + bridged joints |

---

## 4. Refuted — do not re-open

* **F03** — the legacy nose-pitch clamp is *not* what produces the 12.6 % head error, and removing it measurably makes the head **worse** (12.66 % vs 12.58 %) while tripling the visible nose-to-mesh distance from 1.56 mm to 4.34 mm. Keep the clamp on the legacy nose path.
* **F05** — RTMPose vs the moulded red dots is a ~4 mm **definitional** offset, roughly uniform at every joint (knees, the one marker verifiably at a pivot, score ~0 offset), with no bone-length distortion; the dots are 1.8× noisier (CV 9.18 % vs 5.07 %) than the thing they would replace.
* **F19** — `rig_bone_lengths` is live inside `fit_to_subject`; commit 02e8a44 says the opposite of what the finding attributed to it; forcing rig proportions onto the skeleton displaces it 3.6 % of height and makes reprojection 39 %/66 % worse.
* **F26** — detector confidence is nearly **inert**, not misleading: the UI bands on the reprojection residual (the confidence fallback fires 0 of 390 times), `kpt_thr` fired 0/780, and confidence's correlation with error is weak but correctly signed.
* **F35** — the (NECK, HEAD) bone's 9.5 % CV is a normalisation artefact of a short segment; dropping it moves the posed head axis 0.52 deg median, and `BONES` is shared with the 2D overlay, the 3D view and the export topology.

---

## 5. Interaction effects

**Between stages (why the audit's own metrics disagree).** Retarget error is measured against the captured joints of the *same* source, so it cannot see error already in its input — which is why the **smoothed** pose scores *best* on retargeting (1.5 %) while being worst on reprojection (21.65 px). Expect the all-joint retarget median to **rise** 1.51 % → ~1.7 % when the smoother is removed, at the same time as reprojection falls 3×. That combination is the signature of a real improvement, and any gate that only watches retarget error will read it as a regression. Likewise, epipolar error cancels a focal that is wrong the same way in both cameras, and bone-length CV is blind to a shared focal error (f × 0.7 *lowers* it to 5.09 %) while being ~7× more responsive than the gauge to a 15 deg extrinsic error (5.27 → 7.42 %). No single number covers the space; ship bone-CV **and** epipolar **and** normalised reprojection.

**Between fixes — the ones that must be sequenced or measured together:**

1. **Smoother removal → the retarget's gap behaviour.** With F01 and F09 both applied, NaN joints reach the retarget for the first time in this take's history: frame 0012 `RIGHT_KNEE` is a chain **mid** joint (two-bone IK fires correctly) and frame 0021 `LEFT_ANKLE` is an **end** joint (the shin has no aim and inherits the thigh). The retarget was tested only for exceptions, never for pops. The flagged single-frame gap fill must ship in the same phase.
2. **Smoother removal → the drag path.** F10 is not a separate fix; it dissolves. Do not "fix" it independently.
3. **Roll → the export.** The roll fix changes exactly the quaternions the exporter keyframes. Keeping the roll **stateless** (a pure function of one frame plus a per-take precomputed reference) is what preserves `pose_bone_matrices` as the single source of truth. Any caller-owned temporal state breaks view/export parity, because `view3d.set_pose` is driven per frame from `main_window.py:450` in **scrub order**, while `blender_export` iterates once in order.
4. **Roll → grounding.** The roll fix does *not* level the feet (measured: 38.5 → 39.6 deg). Grounding on the ankle does fix the bob. Do not let one be justified by the other.
5. **Halpe HEAD → the head basis → the nose fudge.** Once F16 restores face keypoints, `character.py:38 _NOSE_PITCH = 45 deg` and `_head_aim_target` operate on a HEAD joint that may no longer be the nose. The two fixes double-correct unless the pitch path is gated on a persisted head-source key.
6. **Halpe → every threshold.** A native skull HEAD moves the reconstructed body height 0.1187 → 0.1303 m (**+9.8 %**) and the character scale 111.11 → 110.09. Every "% of body height" number in this audit and in the plan's acceptance criteria must be re-baselined in the same commit.
7. **Halpe NECK → the bone targets, the uniform scale and the retarget root.** Native NECK/PELVIS change what the bone-length targets measure and regress neck-Lshoulder CV 5.15 → 8.13 %. Take the HEAD; leave NECK/PELVIS derived.
8. **Root motion → the view's auto-framing.** Restoring 116 % of body height of travel means a take-fixed centre; `_framed` must size the grid and camera distance from the take's full 0.136 m travel or the figure walks out of frame.
9. **Non-destructive gate → gate tightening.** Items must ship together: with the gate reversible, a wrong threshold is a visible annotation; without it, a wrong threshold is permanent data loss.
10. **Delivery mechanics gate everything.** `app.build_model` (`app.py:68-76`) hands the **stored** `fitted3d` straight to the view and never recomputes; `fit_project` runs only behind "↻ Recalibrate 3D". Every fix in this plan is invisible on the client's existing take until they happen to press that button — at which point the pose moves by up to 21.8 % of body height with no version stamp and no explanation. And the only route to head keypoints on an old project is the button that erases their corrections.

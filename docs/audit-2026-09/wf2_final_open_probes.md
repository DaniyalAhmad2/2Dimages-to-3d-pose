## Open probes — what the plan does not settle

Each item is a gap the critic raised that the plan either defers, assumes, or cannot answer from this take. The probe is what would close it. None of these blocks Phase 1.

---

### P1. Whether the client was judging the exported file, and whether the exported file matches the view at all

**What is unsettled.** The audit's entire error budget stops at `Character.pose_bone_matrices`; Blender was never run. Phase 0 ships `tools/check_export_fidelity.py` and Phase 3 fixes what I could already measure from `assets/Imported_Session.bvh` (zero root motion; a helper bone carrying 42.5 % of rig height of translation; the silent fallback that writes a null-rooted skeleton). But the load-bearing number — **does forward kinematics from the delivered BVH reproduce `Character.posed_joints` at the 26 captured keyframes?** — has never been computed in this project's history. If it does not, the ranking of this plan changes and Phase 3 becomes Phase 1.

**Probe.** `POSE3D_BLENDER=/home/athena/Downloads/blender-5.1.1-linux-x64/blender`, run `export_animation` on `Imported_Session` into the scratchpad with `render_video=False`, then from the produced BVH: (1) evaluate FK in numpy and compare world joint positions at every captured keyframe against `Character.posed_joints` for the same 26 frames after a single global similarity fit — **settled if max deviation < 1 % of the 117.8 mm body height**; anything above says the export is not the view even at the keyframes. (2) Repeat with the character asset renamed, to confirm exactly what the fallback path ships. Run this **before** Phase 3, and re-run it as the phase's acceptance. (Note: the interpolation half of this question is already answered — holds are bit-identical at 0.0 deg and in-between overshoot maxes at 3.795 deg with no hemisphere flips, so no interpolation rework is warranted.)

---

### P2. Whether removing the smoother is visible as jitter

**What is unsettled.** Every metric that condemns the EMA (reprojection, bone CV, displacement-from-measurement) is a fidelity-to-measurement metric, so all of them are maximised at α = 1 **by construction**. Nothing in the repo or the audit measures perceived temporal quality, and the take contains **no repeatability data at all** — the detector is deterministic, so re-running gives 0.00 px; you would need two photographs of one held pose, which this take does not contain. "Is the residual noise visible as jitter?" is currently *unanswered*, not answered in the negative.

**Probe** (Phase 1, item 9, before merge). Re-triangulate the take ~20× with the bbox-jitter 2D perturbation the detection auditor already characterised (σ giving 1.26–2.79 px of keypoint movement), and report the per-joint 3D spread in mm against the **9.36 mm median inter-frame motion**. Under ~1 mm (≈10 %) means α = 1 introduces no jitter a viewer can see, especially since the export holds each pose for 22 of every 30 frames — default stays off. Above ~3 mm means ship the zero-phase filter **on** at a low alpha instead. The definitive version needs one thing only the client can supply: **two photographs of a single held pose**, which would give a true repeatability floor.

---

### P3. What the character actually does on a frame with a preserved NaN

**What is unsettled.** With F01 and F09 both applied, NaN joints reach the retarget for the first time in this take's history: frame 0012 `RIGHT_KNEE` (a chain **mid** joint — two-bone IK fires) and frame 0021 `LEFT_ANKLE` (an **end** joint — the shin has no aim and inherits the thigh, the same path that produces the 30–108 deg sole tilt). The retarget auditor tested 130 sparse variants only for **exceptions** (0 raised, 52 `None` returns), never for what the character *looks like*. F09's "downstream already tolerates NaN" is a no-crash claim, not a no-pop claim. Phase 1's flagged single-frame fill covers exactly these two joints — but the plan asserts, rather than measures, that the fill is enough.

**Probe.** Pose the character on frames 0012 and 0021 with the NaN preserved (fill disabled) and report each affected bone's direction change versus its two neighbouring frames in degrees, plus the posed-joint jump as % of height. **> 10 deg or > 5 % of height** means a longer gap must be shipped as a visible hole with the bone held at its previous *local* rotation, not merely interpolated. Separately, diff the view's and the export's hold-last-known rules on the same NaN frame (`view3d.py:138-159` vs `blender_job.py:454`) and report the resulting joint discrepancy as % of height.

---

### P4. Whether the roll fix pops, and whether the stateless sign convention holds

**What is unsettled.** The roll design is stateless on purpose (that is what keeps view == export for free), and its continuity relies on one untested assumption: that weighting by `clip((bend−20)/20, 0, 1)` plus a torso-based hemisphere rule suppresses the measured **55.7 deg** frame-to-frame jump in the captured bend normal. The prototype that drives roll error to 0 was never run with either. If the ≤ 15 deg gate fails, the named fallback (a per-take `roll_references` pure function with 3-tap reference smoothing) costs about a day and has not been prototyped.

**Probe.** Run `<SP>/find-retarget/roll_prototype.py` over all 26 frames with the weight and the sign rule added, and report per bone: max consecutive-frame roll change, the number of sign flips, and the resulting all-joint retarget median. Then render the contact sheets (`<SP>/find-retarget/render.py`, 6 frames × 2 azimuths, shipped vs fixed) and review by eye — **the decisive check for this phase has no CI number**, because the defect is angular and every positional metric in the repo is blind to it (canonical joints move 0.0000 mm).

---

### P5. Absolute scale — the audit's one absolute check is circular

**What is unsettled.** The tag-15 holdout square reconstructs at 50.29 mm against a "true" 50.00 mm that is *the same assumed marker length*, so it validates tag-to-tag consistency, not scale. The only genuinely external number anywhere in the audit is EXIF `SubjectDistance` (median 0.470 m), and it was used to argue about focal, never to close the marker-size question — yet distance scales linearly with marker size, so it constrains both jointly. Nobody has measured the printed tag or the mannequin with a ruler. Every millimetre figure inherits whatever that error is; every "% of height" figure does not.

**Probe.** (1) Tabulate, for marker = 50 / 62 / 75 mm at the shipped focal, the implied camera-to-subject distance and the implied crown-to-sole figure height, against EXIF `SubjectDistance` 0.470 m (range 0.439–0.584) and the client's "~20 cm"; the marker size where both agree is the answer. At 50 mm the distance lands near 0.50 m and the figure near 13–14 cm, so if 20 cm is right the autofocus distance must be wrong by ~50 %. (2) Confirm by experiment what the plan asserts from code reading — pose and export the same take with the triangulation scaled by 1.0 and 1.5 and diff the BVH `OFFSET`/channel values; **bit-identical output proves metric scale never reaches the deliverable** and turns F17 from an accuracy bug into a missing feature. (3) One ruler photo from the client settles it outright — client question 2.

---

### P6. Whether a rebanded gauge is still honest once the smoother is gone

**What is unsettled.** Phase 4 shows two numbers (measured vs delivered) because their difference is the actionable signal. After Phase 1 they should coincide, which removes that diagnostic value on a healthy take — and the plan's green/amber cut points (0.4 % / 1.0 % of figure height) are **chosen, not derived**. The only external anchor is the Panoptic reference, which is a moving human at 90 deg parallax with two limbs missing.

**Probe.** Sweep the thresholds against three references at once — the client fixture after Phase 1, the Panoptic ground-truth project, and the client fixture under a deliberately injected 15 deg extrinsic error — and pick cut points that put the first two green and the third amber/red. Report the margin. If no single pair of cut points does all three, the banding needs a second axis (bone CV), not a re-tune.

---

### P7. Whether the reconstruction is representative of its own successor

**What is unsettled.** One take, 26 frames, one rig, zero corrections, two cameras that never moved. Per-joint medians rest on 25 samples; every "max" is a single frame; the live-edit findings come from a Qt-free reimplementation rather than the real widget. And the take is not representative of itself: only 11 of its 26 frames have any tag common to both cameras, tag 13 returns the wrong IPPE branch in 35 % of solves and is harmless only by luck, and **nothing in the app detects a camera that moved**. The "not the cause" verdicts on extrinsics and intrinsics are verdicts about *this scene's shape error*, not certificates that the calibration is right.

**Probe.** Two things, in order of value. (1) **A second take** — ideally with the right camera seeing two tags, and ideally with one pose photographed twice (which also closes P2). (2) Meanwhile, add a camera-motion check to `report.json` while the scoring loop of Phase 6.4 is being written: per-tag camera-centre standard deviation across frames (1–5 mm on this take, so a clean threshold exists) and flag anything above ~2 cm as "a camera moved during this take".

---

### P8. The live-edit findings have never touched the real widget

**What is unsettled.** F10, F33 and the correction-preservation numbers were all produced by a Qt-free reimplementation of `ProjectModel`. The tests the plan adds drive the real `ProjectModel._resolve_joint`, which is the right fix — but nothing in the plan exercises the actual `CameraView` drag path, the `itemChange` re-entrancy that `_load_images` comments warn about, or the undo/redo interaction with the derived-joint edit that Phase 1 item 6 adds to the same `Edit`.

**Probe.** One `pytest-qt` smoke test (the suite already has `tests/test_ui_smoke.py`): construct the real `MainWindow` on the fixture offscreen, programmatically move a shoulder joint item by 60 px, and assert (a) `fitted3d` equals a whole-take refit on the same 2D to 1e-9, (b) NECK's 2D is the new midpoint, (c) one Ctrl+Z restores **both** the dragged and the derived point, and (d) no signal re-entrancy warning is emitted. If `pytest-qt` is not viable on the CI runner, gate it like `needs_blender`.

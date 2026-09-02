## Questions for the client

Three of these can re-rank the plan. Numbers 1 and 2 are worth more than any further internal experiment.

---

### 1. Which file were you judging?

*"The 3D character does not follow the keypoints accurately"* — were you looking at the **live 3D view in the app**, the **exported FBX/BVH opened in your own software**, or the **rendered mp4**?

**Why it matters.** The three are produced by different code and fail differently. The live view has the smoother's lag (up to 22 % of the figure's height) and the limb twist. The exported file additionally has **exactly zero root translation** while your subject slides 116 % of its own height, and its timeline is 772 frames at 30 fps for your 26 photographs — captured pose *k* sits at frames 1+30·*k* through 22+30·*k*, held perfectly still, with a 9-frame ease between poses. If you were comparing photograph 5 to frame 5, everything after the first pose was misaligned by construction and nothing was wrong with the animation. The mp4 is a turntable spin over the whole clip, which makes a photo-by-photo comparison impossible. If you can, send the exact file, the program you opened it in, and the frame where it looks worst.

**Default if unanswered.** We assume the **live 3D view**, and keep the order above (smoother → roll → export). If the answer turns out to be the exported file, Phase 3 moves to the front.

---

### 2. How big are the printed tags, and how tall is the mannequin — measured with a ruler?

Measure one printed tag edge to edge (the **black square**, not the white border) to the nearest millimetre, and the mannequin from crown to sole. One photo of a ruler beside each settles it.

**Why it matters.** Every millimetre figure in this project hangs on the "ArUco marker size" box being 0.05 m; nothing on disk records it and the tool cannot detect it being wrong. We reconstruct nose-to-ankle at 120 mm, which at a 50 mm tag makes the figure roughly 13–14 cm; you describe it as about 20 cm. The reconstruction is exactly 2.395 tag widths tall in the photograph *and* 2.395 tag widths tall in 3D, so tag size is the only free parameter. Note that shape conclusions are unaffected either way, and that **the exported FBX/BVH is currently normalised to the rig's own size regardless of this number** — so tell us also whether you need the export in real-world units.

**Default if unanswered.** We keep 50 mm, persist it in the project file, and display baseline (55.9 cm) and figure height (11.8 cm) in the sidebar so you can falsify them with a ruler at any time. We do **not** build real-units export.

---

### 3. Can the right camera be moved so it sees at least two tags?

**Why it matters.** Your left camera sees tags 13, 14 and 17 in all 26 frames; the right camera sees only tag 14, in 11 of 26 frames, and nothing at all in 15. The entire calibration therefore hangs on one tag in one frame. It happened to be the right answer on this take, but a slightly different framing would have produced an empty 3D view with no useful diagnostic. Two tags visible to both cameras in most frames would let us cross-check the solve and detect a camera that moved — which nothing in the app can currently do.

**Default if unanswered.** We ship the branch-scoring, tag-admission and provenance work anyway (Phase 6) and add a failure message that names which camera saw which tags in how many frames.

---

### 4. How are the tags mounted — can they be flat, rigid and all the same way up?

**Why it matters.** Tag 13 does not image as a flat square (corner-fit residual 2.4–3.3 px against 0.2–0.8 px for tags 14, 15 and 17), consistent with it being taped over the curve of the green sweep. Its pose solution is genuinely ambiguous in 96 % of frames and returns the wrong one of two possibilities in about a third of them. It is harmless today **only because the right camera never sees it**. The four tags are also rotated 90.8, 114.7 and 179.4 degrees relative to tag 14 and are not coplanar, which tells us the layout is not deliberate. Flat, rigid mounting on the flat part of the backdrop removes the hazard outright.

**Default if unanswered.** We admit tags by planarity residual and reject tag 13 automatically; the take still calibrates.

---

### 5. Would you shoot a one-time checkerboard, once per camera, ever?

Roughly 15–20 photos per camera, once, never repeated per session.

**Why it matters.** Both cameras' intrinsics are currently guessed from the image size (focal = the long edge), and the principal point is assumed to be the exact image centre — where ±200 px costs 19 px of epipolar error, which no amount of tag work can recover. **Be clear about the payoff:** on this take, all calibration error combined accounts for **at most 1.11 px of the 4.91 px cross-view disagreement**, so a checkerboard buys provenance and insurance for future sessions rather than a visible improvement on this one. It stays optional and nothing about the per-take workflow changes.

**Default if unanswered.** We keep the assumed focal (it measures best of everything we tried), record `source="assumed"` on disk, and wire the checkerboard path as an available-but-unused override.

---

### 6. Should the exported character travel, and one frame per photograph?

**Why it matters.** Your mannequin's pelvis slides 0.136 m — **116 % of its own height** — across the 26 poses, monotonically, and both export paths discard all of it, so the character animates its limbs on the spot. That alone can read as "the character does not follow". Separately: do you want **one exported frame per photograph** (frame 3 = photo 3, which is what a retargeter wants) or the current 30 frames per photograph with interpolation between (which is what a viewer wants)? And would a **fixed-camera render framed to match your left camera** be more useful than the turntable for checking against your photographs?

**Default if unanswered.** Root motion **on** for BVH/FBX and off for the mp4; **one frame per pose** as the BVH/FBX default with the stepped timeline behind a switch; the fixed-camera render offered alongside the turntable, not instead of it.

---

### 7. Stop-motion, or will any take be video-rate?

**Why it matters.** Every frame of this take is a separately hand-posed shot, which is why cross-frame smoothing is doing harm rather than good — it lags each frame behind its own photograph by up to 22 % of the figure's height. We are turning it off. If some future takes are shot at video rate, we keep a **lag-free** filter available as a checkbox rather than removing the capability.

**Default if unanswered.** Smoothing off by default, the zero-phase filter retained and exposed as an unchecked import option labelled "video-rate capture only".

---

### 8. Are the feet and hands important enough to be worth 56 MB?

**Why it matters.** The detector we ship has no foot or hand keypoints, so the character's feet inherit the shin's direction and its hands inherit the forearm's. We are fixing the twist and the ground contact, but the sole itself will still tilt with the shin (which genuinely tilts 16–86 deg on a rigid-footed mannequin). Real toe and heel points would fix the feet properly, at the cost of one 56 MB checkpoint in the installer and a re-measurement of the whole skeleton.

**Default if unanswered.** We take the **head** half of that upgrade now (it removes the worst-behaved bone in the whole reconstruction) and defer the feet.

---

### 9. What are you filming next?

Always this mannequin, or real people or other figures? How many poses per take? Will the cameras stay fixed between takes, and will both always be phones of different resolutions? Will you ever pose more than one figure, or move a camera mid-take?

**Why it matters.** Everything measured so far comes from **one 26-frame take of one 12 cm mannequin, two cameras that never moved, and zero manual corrections**. Every per-joint median rests on 25 samples and every "max" is a single frame. The detector is trained on real people; a 1.8 m person at 3 m is a different regime and every threshold we are about to record would need re-deriving.

**Default if unanswered.** We assume the same rig and subject, tune thresholds on this take plus the Panoptic reference, and commit the fixture so the next surprise is caught by a number instead of by you.

---

### 10. What would "accurate enough" look like to you?

A number we can put in the test suite — *"each joint within X mm of where I would put it"*, or *"it should match this photograph at this frame"*.

**Why it matters.** Every gate in this plan is set from measurement rather than from your requirement. One sentence from you anchors the whole thing, and would tell us whether the remaining detector floor (raw bone-length variation of about 5 % on a rigid figure, which no part of this plan moves) is already good enough or is the next project.

**Default if unanswered.** We target "the delivered pose reprojects onto your photographs as well as the raw triangulation does" (reprojection ratio ≤ 1.5, i.e. under 1 % of the figure's on-image height) and treat the detector floor as out of scope.

---

### 11. Is a one-off pose change on your existing projects acceptable?

The first time you open an existing project on the fixed build, its pose will change by roughly 5 mm on a typical joint and up to 26 mm at worst (4 % and 22 % of the figure's height) — that is the lag being removed. We plan to do this automatically on open, tell you the size of the change, and offer a one-click restore of the old pose. Nothing is auto-saved.

**Why it matters.** Without it, you install the fixed build, open the project you complained about, and see exactly what you complained about — the app never recomputes on open today.

**Default if unanswered.** Recompute on open, with the banner and the Restore button, and never auto-save.

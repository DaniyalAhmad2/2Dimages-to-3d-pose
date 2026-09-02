# Getting better 3D poses from the two-camera setup — a guide for the capture side

This is what the audit of your take (26 poses, September 2026) says you can change on
the capture side to get a more accurate character, without any per-session calibration
ritual. Items are ordered by how much they buy. None of them require a checkerboard.

## 1. Tags: bigger, flat, and visible to BOTH cameras

What we measured: your left camera saw three tags in every shot, but the right camera
saw only one tag (id 14) in 11 of 26 shots and no tag at all in the other 15. The whole
3D reconstruction hung on one tag in one photograph. It happened to be right; a slightly
different framing would have produced an empty 3D view.

- Put **at least two tags where both cameras see them in every shot** — ideally all four.
  Two shared tags let the software cross-check the camera positions and detect a camera
  that moved.
- Mount tags **flat and rigid** (glued to foam board or card), not taped over the curve of
  the backdrop. Tag 13 in your take is bent over the curve and its pose is ambiguous in
  96 % of shots; the software will now reject such tags automatically, but a flat tag is
  free accuracy.
- Mount all tags **the same way up** (same rotation). It does not change accuracy, but it
  lets the software use the row of tags as a "which way is up" reference.
- **Bigger tags help**: 8–10 cm tags instead of 5 cm give a proportionally better camera
  pose and scale. Print at 100 % and **measure the black square edge to edge with a ruler**;
  type that number into the "ArUco marker size" box every time. Nothing on disk can tell
  a 5 cm tag from a 7.5 cm tag — the whole metric scale is that one number.
- Put one or two tags **at the figure's own depth** (for example on the table surface
  beside its base), not only on the backdrop behind it. Tags near the subject constrain
  the geometry where it matters.

## 2. Phones: no digital zoom, locked focus, static

- Your left phone shot at **1.26× digital zoom**. Digital zoom changes the effective focal
  length and the metadata we read is then wrong. Use the main camera at exactly 1× and
  move the phone instead.
- **Lock focus and exposure** (tap-and-hold on most phones) before the first shot and do
  not touch it during the take. Autofocus changes the focal length between shots.
- Keep both phones on tripods and **do not touch them** between shots. (In your take they
  did not move — good — and the software will now report if they do.)
- Use the **same phone model, or at least the same resolution**, for both cameras if you
  can. Your right camera was 1536×2048 with no photo metadata; the left was 3072×4080.
  Different devices are supported, but the lower-resolution one contributes half the
  angular precision.

## 3. Framing: make the figure big in the frame

The figure spans about 800 of the 4080 pixels of height in your left photos and about
400 of 2048 in the right ones. The joint detector's precision is a fixed fraction of the
figure's on-screen size, so **filling half the frame with the figure roughly halves the
error**. Move the cameras closer or use the full sensor; keep every tag you need in frame.
Keep the two cameras roughly **60–90° apart** (yours are ~50–60°, fine) and near the
figure's height rather than looking steeply down.

## 4. Capture habits that pay off

- Keep the **face (nose and both ears) visible** to both cameras; the head orientation
  comes from them.
- Keep the gooseneck clamp from crossing the arms or legs in either view.
- **Photograph one pose twice** at some point in a take (two shots, no change). That gives
  us the noise floor of the detector on your rig, which is the one number the audit could
  not measure.
- Take one photo with a **ruler beside the figure** and one beside a tag, once. It settles
  the "how big is it really" question (the reconstruction says the figure is about 12 cm
  crown to sole if the tags are 5 cm; you described it as about 20 cm).
- Shoot with the figure standing on a **flat, level surface**; the software levels the
  scene using the cameras and the tag row, and a level floor makes that unambiguous.

## 5. What the software will do without a checkerboard

You asked for robustness with uncalibrated cameras. The audit measured that the camera
intrinsics are **not** what limits accuracy on your take: replacing the assumed focal
length with the true one changes the reconstruction by under 0.5 % of the figure's size.
The updates make the tag-based path robust instead:

- both branches of every tag's pose are scored across all frames, bent or ambiguous tags
  are rejected, and the calibration is written to a report you can read;
- the focal length can be refined from the tags themselves across the whole take when
  two or more tags are visible to a camera in most frames (on your take this recovered
  the left phone's focal to within 0.2 % of the value implied by its metadata);
- the vertical direction is estimated from the cameras and the tag row, with its
  uncertainty shown;
- the accuracy readouts will show numbers you can check with a ruler (camera separation,
  figure height) and a bone-length consistency figure that a rigid figure should keep
  near zero.

If you ever want the last few percent: the same four tags glued to one flat board, held
at three or four tilts in front of each phone once (about 20 seconds per phone), gives
the software a full lens model without a checkerboard. It is optional.

## 6. What to expect after the fixes

- The character will sit on each photograph's keypoints instead of lagging one pose
  behind (the old build blended each frame with the previous one by 40 %).
- Limbs will twist the way the photographed limb does; feet and hands will no longer
  spike or splay; the figure will not bob against the floor.
- The exported FBX/BVH will move across the scene as the figure did (root motion), with
  one exported frame per photograph by default.
- The remaining floor is the joint detector itself: about 5 % variation in measured
  limb lengths on a rigid figure. Items 1–3 above are what reduce that floor.

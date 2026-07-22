# Project Context: 2D Image to 3D Pose Desktop Tool

Last updated: 2026-07-21
Owner: Daniyal Ahmad Khan (Computer Vision Engineer)
Client: Prav (Upwork)
Status: Scoping complete, awaiting one final answer before build (see Open Questions)

---

## 1. One-line summary

A desktop application that takes 2D photos of a posed subject shot from two camera
angles, detects the humanoid joints in each view, triangulates them into a metric 3D
pose, lets the user manually correct joints, and exports the result as BVH, FBX and an
mp4 for downstream animation work (Blender / Mixamo rig).

---

## 2. What we are building (end-to-end flow)

1. Open the desktop app (one-click launch, no terminal).
2. Import two image batches, one per camera angle (left / right).
3. Auto-match frames across the two cameras by filename (L_1 <-> R_1, L_2 <-> R_2, ...).
4. Detect the joints in each image.
5. Show the skeleton/bones over the original image with a per-joint confidence and a
   RAG (red/amber/green) status.
6. Triangulate the joints across the two views into a 3D pose.
7. Let the user step frame by frame and manually drag incorrect joints in 2D.
8. Recompute the 3D pose live after each correction.
9. Preview the 3D pose/animation.
10. Export to BVH, FBX and mp4. Save the project and reopen it later.

---

## 3. Scope

### v1 (this contract) - price 1,000 USD

Everything in the brief EXCEPT the machine-learning feedback loop:

- Desktop app, one-click launch (packaged, e.g. PyInstaller).
- Batch import from two camera folders.
- Frame matching across the two views by filename convention.
- Camera calibration handling (see section 6).
- Joint detection with per-joint confidence + RAG overlay on the original images.
- Frame-by-frame navigation.
- Manual 2D joint correction with live 3D re-solve.
- Bone-length constrained 3D reconstruction (fit to a fixed-length skeleton / Mixamo rig).
- 3D preview.
- Save / reload a project.
- Export: BVH + FBX + mp4. (FBX pulled into v1 at client request.)
- Correction data is stored on disk (needed for undo now, and seeds the v2 learning loop).

### v2 (future, separate) - client referenced ~1,500 USD extra

- The correction-learning feedback loop: corrections logged as labeled training data,
  periodic fine-tuning of the detection model, model versioning with rollback.
- Client has "some good ideas" for this but wants a solid v1 first.

Deliberately NOT promised: real-time capture (tool imports images, does not take them),
an analytics platform, multi-person, cloud anything.

---

## 4. Client context and preferences

- Budget-sensitive: target for a working version was under 1,000 USD. Wants proven value
  before investing in ML.
- Workflow is essentially miniature / stop-motion style: a poseable figure is hand-posed,
  held still, and photographed from two angles per frame.
- Capture: 2 webcams/cameras, two photos taken at the same time (sometimes with a
  colleague's help), files renamed L_1, R_1, L_2, R_2, etc.
- Cameras sit at the left and right corners of the scene, roughly 75-90 degrees apart,
  occasionally wider for a bigger view.
- Markers: four 5cm x 5cm ArUco tags are in every shot (taped to the backdrop). Client
  uses these specifically to avoid a tedious per-session calibration.
- Head tracking must be SIMPLE. A single head/nose point, not a 5-point face mesh.
  (5-point is too fiddly to adjust across many frames.)
- Export: FBX is the preferred format for the client's colleagues, alongside BVH and mp4.
- The dashboard/UI layout already exists as a PNG image (to be used as the template, not
  invented from scratch).

---

## 5. The capture setup (from the client's photo)

- Chroma-green backdrop, curved sweep. Good for figure segmentation.
- A small grey artist mannequin (S.H. Figuarts style) posed mid-stride, held in place by a
  black gooseneck clamp mounted on a base.
- The mannequin has RED DOTS stuck on every joint (head, shoulders, elbows, hands, hips,
  knees, feet). These are physical joint markers.
- Four ArUco tags are taped flat along the top of the green backdrop, in a row, at roughly
  the same depth and orientation (effectively coplanar).
- Small working volume (desktop scale).

Implications drawn from the photo:

- The subject is STATIC per frame (held by the gooseneck), so the two camera shots do not
  need to be perfectly simultaneous. The frame-sync problem largely disappears for this
  workflow.
- The gooseneck arm is in frame and attaches near the torso/arm. It must be masked/ignored
  so it is not treated as part of the body.
- The red dots are effectively an optical-mocap marker set (see section 7, detection).

---

## 6. Calibration approach (decided)

Split the problem in two:

- Extrinsics (where each camera sits): solved for free every shot via the ArUco tags and
  solvePnP. Both cameras localize against the same tag world-frame, so they automatically
  share one coordinate system. Cameras can even be repositioned between setups. The known
  5cm tag size gives real metric scale, so bone lengths come out in real-world units.
- Intrinsics (focal length, principal point, lens distortion): still required, and this is
  the catch. The tags in the actual setup are coplanar, which is fine for extrinsics but
  NOT enough to estimate intrinsics from the image (one view of one plane is insufficient).

Decision: do a ONE-TIME per-camera intrinsic calibration (30-second checkerboard/ChArUco
capture per camera), reused across all sessions, redone only if a camera is swapped. This
is NOT the per-session calibration the client wanted to avoid, and it fits the current rig
without changes.

Critical gotcha: these are webcams. AUTOFOCUS must be disabled/locked, otherwise focal
length drifts between shots and invalidates the one-time calibration. Each camera (L and R
may be different models) needs its own intrinsics.

Fallback if the client refuses even the one-time calibration: self-calibrate intrinsics
from the tags per shot, but only works if the tags are spread at different depths/angles
(they are not, currently), and distortion is estimated poorly, so accuracy near image
edges suffers. Not preferred.

---

## 7. Detection approach (OPEN - blocks architecture)

The mannequin has red joint markers, which changes the best approach:

- Option A (marker-based): detect the red dots directly (HSV color blob detection on the
  green background, near-perfect separation). This is far more reliable and accurate than a
  learned pose model, trivially ignores the gooseneck, and is basically classic optical
  mocap. Strongly preferred IF the real workflow always uses a marked figure.
- Option B (markerless pose model): run a human 2D pose model per view. Needed only if the
  end goal is unmarked real humans. Note: human pose models (MediaPipe, RTMPose, YOLO-pose)
  are trained on real people and are unreliable on a plain grey mannequin, so the current
  mannequin is a poor test subject for this path. If we go markerless, prefer RTMPose or
  ViTPose over MediaPipe (they handle non-upright/inverted poses far better, which matters
  for the gymnastics/martial-arts cases the client raised).

>> BLOCKER: confirm with client whether the final subject is ALWAYS a marked figure like
   this, or whether this is a stand-in for capturing real (unmarked) humans later. The
   answer decides A vs B. (Question already sent to client.)

Design the detector behind a common keypoint interface either way, so the model/method can
be swapped without touching the triangulation and fitting layers (client noted the model
may change over time).

Head tracking: single head/nose point only, per client. No 5-point face mesh.

---

## 8. Reconstruction, correction, and export

- Triangulation: undistort each view, then per-joint DLT triangulation across the two
  camera rays into a 3D point.
- Bone-length fit: raw triangulation is jittery, so fit the 3D joints to a fixed-bone-length
  skeleton (the Mixamo rig) with a small optimization plus light temporal smoothing. This is
  the step that makes the output usable animation instead of noise. Client explicitly wants
  joint/bone-length constraints, mapped to a Mixamo rig.
- Manual correction: corrections happen in 2D on the images. Dragging a joint in one view
  updates that observation and re-triangulates just that joint (corrected ray + other
  camera's ray), instant. Re-run the bone-length fit after each edit. If a joint is only
  fixable in one view (occluded in the other), constrain it along the viewing ray using the
  parent bone length. Every correction is a discrete, reversible edit.
- Extreme poses (gymnastics, martial arts): the guarantee is not that detection always
  nails it, but that no frame is silently dropped and everything is manually fixable. RAG
  status flags low-confidence/occluded joints for review rather than hiding them.
- Export backend: standardize on headless Blender (bpy). One rig produces BVH, FBX and the
  mp4 render. FBX rig retargeting is the fiddliest export piece but comes off the same rig
  as BVH, so it is a modest delta once the Blender path exists.

---

## 9. Proposed tech stack

- Language / env: Python, managed with uv.
- Desktop UI: PyQt6 or PySide6 (rebuild from the client's PNG layout).
- CV / geometry: OpenCV (ArUco, calibration, undistort, triangulation), NumPy, SciPy.
- Detection: OpenCV HSV blob detection (marker path) OR RTMPose/ViTPose via
  mmpose/onnxruntime (markerless path). Decided by the open question.
- 3D preview: pyqtgraph.opengl or VTK embedded in the Qt app.
- Skeleton/export: Blender (bpy) headless, Mixamo target rig, exports BVH/FBX/mp4.
- Project storage: per-project folder (images + JSON pose/correction data); SQLite optional
  for the correction log that feeds v2.
- Packaging: PyInstaller for one-click launch.

---

## 10. Open questions / pending client answers

1. [BLOCKER] Marked figure forever, or real humans eventually? (decides detection path)
2. Will the client do the one-time per-camera intrinsic calibration, and can they lock
   webcam autofocus?
3. Format of the existing dashboard PNG layout (just the image confirmed; need it to rebuild).
4. Expected number of frames per animation / per session (affects UI and performance).
5. Roughly how many joints in the target skeleton, and is the Mixamo standard rig the exact
   target?

---

## 11. Risks and mitigations

- Webcam autofocus drift invalidates calibration -> require focus lock; marker-per-shot
  extrinsics still adapt.
- Human pose model unreliable on the mannequin -> prefer marker-based detection if figures
  are always marked.
- FBX rig retargeting is finicky -> prototype the Blender export path EARLY, not at the end.
- Only two cameras -> joints occluded in both views need manual placement; RAG flagging +
  bone-length constraint keep this honest.
- Coplanar tags cannot self-calibrate intrinsics -> one-time calibration is the plan.

---

## 12. Commercials

- v1 (this contract): 1,000 USD, includes BVH + FBX + mp4 export. ML loop excluded.
- v2 (future): correction-learning feedback loop, client referenced ~1,500 USD additional.
- Reference-only internal rate anchor: ~25 USD/hr.

---

## 13. Relevant prior work to reuse (from proposal-kb)

- PTZ-camera CV library: PyQt6 desktop app with Zhang's-method calibration (intrinsics +
  extrinsics) and stereo depth. Directly reusable for the calibration + geometry core.
- pyqt-deepstream-counter: production PyQt5 desktop CV app pattern.
- YOLOv11-Pose player tracker (17 keypoints), jump-height keypoint tool, custom 24-keypoint
  model trained from scratch (OKS): the pose/keypoint experience if we go markerless.
- omnia360: staged, resumable folder-processing pipeline pattern (good analogue for a
  batch import + per-stage recompute tool).

# Implementation Plan: 2D Image → 3D Pose Desktop Tool (v1)

Owner: Daniyal Ahmad Khan · Client: Prav (Upwork) · Target: working v1 by next weekend
Companion doc: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) · Layout template: [Animation Dashboard.png](frontend-sample-desgin/Animation%20Dashboard.png)

## Decisions locked for this plan

| Decision | Choice | Rationale |
|---|---|---|
| Detection path (v1) | **Markerless** (RTMPose via `rtmlib`), behind a swappable `KeypointDetector` interface | Client's call; interface keeps the marker path a drop-in later |
| Build order | **Layered full-breadth** around a central shared data model | Client wants steady visible progress; layers stay decoupled via the model |
| Export backend | **Local Blender 5.1.1** headless via `subprocess` (`/home/athena/Downloads/blender-5.1.1-linux-x64/blender`) | No bpy wheel; probed and validated against the actual binary |
| UI framework | **PySide6** (LGPL) | Clean commercial licensing for a delivered paid app |
| 3D preview | **pyqtgraph.opengl** `GLViewWidget` | Simplest embeddable rotating skeleton (points+lines) |
| Env / packaging | **uv** for env, **PyInstaller** for one-click app | Global rule (uv); brief requires one-click launch |

### Two caveats to hold visible throughout the build
1. **Markerless will be flaky on the grey mannequin test rig** (RTMPose is trained on real people) and on extreme gym/martial-arts poses. Therefore the **manual-correction path is a load-bearing v1 feature (Phase 7), not end-of-project polish.** The guarantee is "no frame silently dropped, every joint always editable," not "detection is always right."
2. **COCO-17 ≠ the mockup's joint list.** RTMPose emits nose/eyes/ears/shoulders/elbows/wrists/hips/knees/ankles. The mockup shows Nose, **Head**, **Neck**, **Hips** — which COCO lacks. `core/skeleton.py` (Phase 1) *derives* neck = shoulder-midpoint, pelvis/hips = hip-midpoint, head = nose. Confirm the exact target rig joint set with the client when Q5 lands; the skeleton module isolates any change.

---

## Phase 0 — Allowed APIs (verified; do not deviate without re-checking source)

Every item below was verified against official docs or by probing the installed Blender binary. **Treat anything not listed here as unverified — check the cited source before using it.**

### OpenCV geometry (cv2 4.8–4.10) — `docs.opencv.org/4.10.0`
- **ArUco is class-based now.** `cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)`, `cv2.aruco.DetectorParameters()` **(constructor, NOT `.create()`)**, `cv2.aruco.ArucoDetector(dict, params)`, `detector.detectMarkers(gray) -> (corners, ids, rejected)`. `corners` = list of `(1,4,2)` float32.
- **`estimatePoseSingleMarkers` is DEPRECATED.** Replacement: build 3D object points and `cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)`. Corner order from `detectMarkers` is **TL, TR, BR, BL** — object points MUST match that order. Do not reshuffle.
- **Intrinsics:** `cv2.findChessboardCorners(gray,(cols,rows),flags=...)` → `cv2.cornerSubPix(...)` → `cv2.calibrateCamera(objpoints,imgpoints,imageSize,None,None)` returns exactly `(rms, K, dist, rvecs, tvecs)`. ChArUco alt: `cv2.aruco.CharucoBoard((sx,sy),sq,mk,dict)` + `cv2.aruco.CharucoDetector(board).detectBoard(gray)`; watch the 4.6.0 `setLegacyPattern(True)` break for pre-4.6 printed boards.
- **Extrinsics:** `cv2.solvePnP(...)` then optional `cv2.solvePnPRefineLM(...)`. `cv2.Rodrigues(rvec)` → 3×3 R. `X_cam = R @ X_obj + t`; camera center `C = -R.T @ tvec`.
- **Undistort:** `cv2.undistortPoints(src, K, dist)` with **P omitted → normalized** coords; **P=K → pixel** coords. `src` shape `(N,1,2)` float32.
- **Triangulation:** `cv2.triangulatePoints(P1, P2, pts1_2xN, pts2_2xN) -> 4×N`; then `X = (pts4[:3]/pts4[3]).T`. **Consistency rule (never mix):** normalized points ⇒ `P=[R|t]`; pixel points ⇒ `P=K[R|t]`. Always undistort before triangulating. **This plan uses normalized points + `P=[R|t]`.**

### Markerless detection — `rtmlib` 0.0.15 (`github.com/Tau-J/rtmlib`)
- Pure `onnxruntime`, **no mmpose/mmcv/torch/CUDA needed.** `from rtmlib import Body; body = Body(mode='balanced', backend='onnxruntime', device='cpu'); keypoints, scores = body(img_bgr)`. Output: `keypoints (N,17,2)` pixels, `scores (N,17)` in [0,1]. Pick person 0 (or highest mean score).
- `mode`: `'lightweight'` | `'balanced'` (RTMPose-M) | `'performance'`. Force CPU by installing the plain `onnxruntime` wheel (not `-gpu`) and `device='cpu'`.
- **COCO-17 order:** `nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`.
- **Derived joints:** `head = kp[0]`; `neck = (kp[5]+kp[6])/2`; `pelvis = (kp[11]+kp[12])/2`. Derived-joint confidence = `min` of parents.
- **RAG bands (tune on real data):** green ≥ 0.6 · amber 0.35–0.6 · red < 0.35.
- **Offline/PyInstaller:** models auto-download to `~/.cache/rtmlib/hub/checkpoints`. For a bundled app, use the low-level `from rtmlib import RTMPose; RTMPose(onnx_model='/abs/path.onnx', model_input_size=(192,256), backend='onnxruntime', device='cpu')` with shipped `.onnx` files (bundle the YOLOX detector ONNX too), OR pre-populate the cache / point `TORCH_HOME` at a bundled folder.

### Blender 5.1.1 headless — probed live from the installed binary (bundled Python 3.13.9)
- Invoke: `blender --background --python script.py -- <args>`. Read args: `argv = sys.argv[sys.argv.index('--')+1:]`.
- **Rig driving: use FK direct rotations, NOT IK.** Per bone, rotate its rest axis (+Y for Mixamo/FBX) onto the child-joint direction; `pose_bone.rotation_quaternion = rest_dir.rotation_difference(target_dir)`; `pose_bone.keyframe_insert(data_path='rotation_quaternion', frame=f)`. Root world motion keyframed on the object: `arm_obj.keyframe_insert(data_path='location', frame=f)`. Pose bones default `rotation_mode='QUATERNION'`. Robust alt: `bpy.ops.import_scene.fbx(...)` a Mixamo template rig and drive `pose.bones['mixamorig:Hips']` etc.
- **Frame/fps:** `scene.frame_start/frame_end`, `scene.render.fps` (int), `scene.render.fps_base=1.0`.
- **BVH:** `bpy.ops.export_anim.bvh(filepath=..., frame_start=..., frame_end=..., rotate_mode='NATIVE', root_transform_only=False, global_scale=1.0)`. Select + make active the armature first. **`frame_step` was removed in 5.x — do not pass it.** `io_anim_bvh` enabled by default.
- **FBX:** `bpy.ops.export_scene.fbx(filepath=..., use_selection=True, object_types={'ARMATURE','MESH'}, add_leaf_bones=False, bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False, primary_bone_axis='Y', secondary_bone_axis='X', axis_forward='-Z', axis_up='Y')`. `io_scene_fbx` enabled by default; defensive `addon_utils.enable('io_scene_fbx')` is harmless.
- **⚠️ mp4 render — THE one 5.x breaking change:** you must set `scene.render.image_settings.media_type='VIDEO'` **BEFORE** `file_format='FFMPEG'` or it raises `TypeError: enum "FFMPEG" not found`. Then `ffmpeg.format='MPEG4'`, `ffmpeg.codec='H264'`, `ffmpeg.constant_rate_factor='MEDIUM'`, `ffmpeg.audio_codec='NONE'`. `bpy.ops.render.render(animation=True)`. **Set `scene.camera` and a light** or frames render black; engine `'BLENDER_EEVEE_NEXT'` for speed.

### PySide6 UI — `doc.qt.io/qtforpython-6`
- **Draggable joints:** `QGraphicsView`+`QGraphicsScene`, `QGraphicsPixmapItem` background, one `JointItem(QGraphicsEllipseItem)` per joint with flags `ItemIsMovable | ItemIsSelectable | ItemSendsGeometryChanges` (**the last is required or `itemChange` gets no position events**). Override `itemChange(self, change, value)`; emit on `GraphicsItemChange.ItemPositionHasChanged` (value = QPointF). `QGraphicsItem` is not a `QObject` → host a small `QObject` with a `Signal` for emission. Bones = `QGraphicsLineItem.setLine(x1,y1,x2,y2)`, updated in the joint's move handler.
- **3D panel:** `pyqtgraph.opengl.GLViewWidget` (a QWidget, `layout.addWidget` it) + `GLScatterPlotItem(pos=Nx3)` joints + `GLLinePlotItem(pos=..., mode='lines')` bones; refresh via `setData(pos=...)`. Deps: `uv add pyqtgraph PyOpenGL numpy`.
- **Filmstrip:** `QListView` `ViewMode.IconMode`, `Flow.LeftToRight`, `setWrapping(False)`, `setUniformItemSizes(True)`, `SingleSelection`; custom `QStyledItemDelegate.paint()` draws thumbnail + RAG status dot; `selectionModel().currentChanged` drives frame load.
- **Sync hub:** one `ProjectModel(QObject)` owning state, exposing `Signal`s (`joint2dChanged`, `pose3dChanged`, `accuracyChanged`). Drag → model slot updates 2D → re-triangulate → `emit` → 3D panel + accuracy list redraw. Panels never call each other.
- **Theme:** `app.setStyleSheet(open('dark.qss').read())`; hook widgets via `setObjectName(...)` + `#id` QSS selectors.
- **Packaging:** `uv run pyinstaller --windowed --clean --name pose3d main.py`. Bundle the Qt `platforms` plugin (blank-window bug if missing), bundle `PyOpenGL`, `--exclude-module PyQt5 --exclude-module PyQt6` (PyInstaller ≥6.5 forbids mixed Qt bindings). Prefer `--onedir` while debugging.

### Anti-patterns to reject on sight (invented / stale APIs)
- `cv2.aruco.estimatePoseSingleMarkers(...)` ❌ · `DetectorParameters.create()` ❌ · `aruco.CharucoBoard.create()` ❌
- Triangulating **distorted** points, or mixing normalized points with `K[R|t]` ❌
- Blender: `file_format='FFMPEG'` before `media_type='VIDEO'` ❌ · passing `frame_step=` to `export_anim.bvh` ❌ · rendering with no `scene.camera` ❌
- IK rig solving when joint world positions are already known ❌ (use FK direct rotation)
- `onnxruntime-gpu` in a portable app ❌ · `QGraphicsEllipseItem` without `ItemSendsGeometryChanges` then expecting move callbacks ❌
- Any second Qt binding (PyQt5/6) present in the packaging env ❌

---

## Phase 1 — Scaffold + shared data model (the spine everything hangs off)

**Goal:** repo skeleton, uv env, and the central `ProjectModel` + canonical skeleton + project persistence that every later layer reads/writes.

**Files**
- `pyproject.toml` (uv), `pose3d/__init__.py`, `pose3d/app.py` (entry stub)
- `pose3d/core/skeleton.py` — canonical joint enum, bone list (parent→child pairs), COCO-17 index map, `derive_joints(kp, scores)` producing head/neck/pelvis with min-parent confidence, and the Mixamo bone-name mapping table.
- `pose3d/core/project.py` — `ProjectModel(QObject)`: per-frame structure `{frame_id, left_img, right_img, kp2d[cam][joint], score[cam][joint], pose3d[joint], corrections[]}`; Signals `joint2dChanged`, `pose3dChanged`, `accuracyChanged`; slot `set_joint_2d(cam, joint, QPointF)` (stubbed re-triangulate hook for now).
- `pose3d/core/io_project.py` — save/reload a project **folder** (`images/`, `poses.json`, `calibration/`, `corrections.sqlite`). JSON schema for poses; SQLite table `corrections(frame, cam, joint, old_xy, new_xy, ts)` (seeds v2).
- `tests/test_skeleton.py`, `tests/test_io_project.py`

**Implement (copy the verified patterns)**
- COCO-17 map + derived-joint formulas exactly from Phase 0 (rtmlib section).
- `Signal`/`Slot` pattern exactly from Phase 0 (PySide6 sync-hub).

**Verify**
- `uv run pytest tests/test_skeleton.py` — derived neck/pelvis/head equal hand-computed midpoints; derived confidence == min of parents.
- Round-trip: save a synthetic 3-frame project, reload, assert deep-equal.
- `grep -rn "estimatePoseSingleMarkers\|\.create()" pose3d/` returns nothing.

**Anti-pattern guards:** no geometry/detection logic leaks into `project.py` (model holds data + signals only); skeleton definition lives in exactly one place.

---

## Phase 2 — Calibration layer

**Goal:** one-time per-camera intrinsics + per-shot ArUco extrinsics in a shared world frame with metric scale.

**Files**
- `pose3d/calib/intrinsics.py` — checkerboard capture→`calibrateCamera`; save/load `K`, `dist` per camera (L and R separate). Include the **autofocus-lock reminder** in the operator-facing docstring/UI note (webcam focal drift invalidates this).
- `pose3d/calib/extrinsics.py` — `ArucoDetector.detectMarkers` → per-marker `solvePnP(SOLVEPNP_IPPE_SQUARE)` with the 5cm object points → both cameras localized to the shared ArUco world frame; return `R,t` per camera. Metric scale from the known 5cm tag.
- `assets/checkerboard.md` (spec of the board to print) · `tests/test_extrinsics.py`

**Implement (copy verified patterns):** all four ArUco/solvePnP/calibrate snippets from Phase 0 OpenCV section, exact corner order and flags.

**Verify**
- Synthetic test: project known 3D tag corners into two virtual cameras, run extrinsics, recover relative pose within tolerance.
- Reprojection RMS from `calibrateCamera` logged and asserted `< 1.0 px` on a real capture set.
- `grep` confirms `DetectorParameters()` used, no `.create()`.

**Anti-pattern guards:** never estimate intrinsics from the coplanar scene tags (Section 6 of context — insufficient); intrinsics come only from the checkerboard/ChArUco capture.

---

## Phase 3 — Detection layer (markerless, swappable)

**Goal:** per-view 2D keypoints + confidence behind an interface, feeding the model with RAG status.

**Files**
- `pose3d/detect/base.py` — `KeypointDetector` ABC: `detect(image_bgr) -> (kp Nx2, score N)` in canonical joint order.
- `pose3d/detect/rtmpose.py` — wraps `rtmlib`; picks best person; maps COCO-17 → canonical + derives head/neck/pelvis; assigns RAG per joint.
- `pose3d/detect/manual.py` — trivial "detector" that returns user-corrected points (lets the correction path reuse the same interface).
- `pose3d/detect/models/` — bundled ONNX (for offline packaging) · `tests/test_rtmpose_smoke.py`

**Implement (copy verified patterns):** `Body(mode=..., backend='onnxruntime', device='cpu')` call and output handling; derived-joint + RAG-band logic from Phase 0.

**Verify**
- Smoke test on a real sample image: shape `(17,2)`/`(17,)`, canonical joints populated, RAG assigned.
- Force-CPU check: session providers == `['CPUExecutionProvider']`.
- Offline check: with network blocked, `RTMPose(onnx_model=local_path)` still loads.

**Anti-pattern guards:** no triangulation here — detector returns 2D only; keep the interface pure so the marker path drops in later. Do not hardcode `to_openpose=True` (we do our own mapping).

---

## Phase 4 — Geometry layer (the metric core)

**Goal:** undistort + two-view DLT triangulation → bone-length-constrained fit to the Mixamo rig → smoothing. This is what turns 2D + calibration into usable 3D.

**Files**
- `pose3d/geometry/triangulate.py` — `cv2.undistortPoints` (normalized) → `cv2.triangulatePoints(P1,P2,...)` with `P=[R|t]` → `X=(pts4[:3]/pts4[3]).T`. Per-joint single-point triangulation function (for live re-solve of one dragged joint).
- `pose3d/geometry/bonefit.py` — SciPy least-squares fit of the raw 3D joints onto fixed bone lengths (Mixamo rig), light temporal smoothing; one-in-one-view fallback (constrain occluded joint along its viewing ray using parent bone length).
- Wire `ProjectModel.set_joint_2d` → re-triangulate that joint → re-run fit → `emit pose3dChanged`.
- `tests/test_triangulate.py`, `tests/test_bonefit.py`

**Implement (copy verified patterns):** the **normalized-points + `P=[R|t]`** combination exactly (Phase 0 triangulation rule). Never the pixel/K mix.

**Verify**
- Synthetic: known 3D point → project into two calibrated views → triangulate → recover within `< 1e-3` (perfect data), `< few mm` with pixel noise.
- Bone-fit test: jittered input → output bone lengths equal rig lengths within tolerance; temporal smoothing reduces frame-to-frame variance.
- Single-joint re-solve returns instantly and changes only that joint.

**Anti-pattern guards:** undistort before triangulate, always; no mixing of point-space and projection-matrix-space.

---

## Phase 5 — Export layer (SPIKE THIS EARLY — start a throwaway version during Phase 2)

**Goal:** one Blender headless job that takes per-frame 3D joints and emits BVH + FBX + mp4. Per the risk register, FBX retargeting is the fiddliest piece — prototype the Blender path early rather than at the end.

**Files**
- `pose3d/export/blender_export.py` — Python-side: writes joints to a temp JSON, calls `subprocess.run([BLENDER, '--background', '--python', blender_job.py, '--', '--in', json, '--out', dir])`, checks return code + output files.
- `pose3d/export/blender_job.py` — runs **inside** Blender: reads `--` args, builds/imports the Mixamo armature, FK-drives bones per frame, keyframes, exports BVH + FBX, renders mp4.
- `assets/mixamo_template.fbx` (optional template rig) · `tests/test_export_smoke.py`

**Implement (copy verified patterns):** the FK-rotation driving, BVH/FBX operator calls, and the **`media_type='VIDEO'` before `file_format='FFMPEG'`** mp4 sequence — all exactly from Phase 0 Blender section. Hardcode `BLENDER = /home/athena/Downloads/blender-5.1.1-linux-x64/blender` as a config default.

**Verify**
- End-to-end smoke: a 5-frame synthetic walk → produces a non-empty `.bvh`, `.fbx`, and a playable `.mp4`.
- Re-import the exported FBX into Blender headless → armature present, bone count matches, animation has keyframes (no black mp4 → confirms camera/light set).
- `grep` the job for the correct mp4 ordering and no `frame_step=`.

**Anti-pattern guards:** no bpy pip wheel; call the installed binary. Don't set FFMPEG format before media_type. Ensure `scene.camera` set.

---

## Phase 6 — UI shell (rebuild the dashboard from the PNG)

**Goal:** the dark dashboard laid out to match [Animation Dashboard.png](frontend-sample-desgin/Animation%20Dashboard.png), wired to `ProjectModel` (real geometry already works from Phases 3–5).

**Files**
- `pose3d/ui/main_window.py` — top bar (project name/status), left sidebar (project/calibration/processing/display/tools), center L+R camera panels, right 3D preview + Pose Accuracy + Joint Accuracy list + Selected Joint, bottom filmstrip.
- `pose3d/ui/camera_view.py` — `QGraphicsView` + `JointItem` (draggable) + bones.
- `pose3d/ui/view3d.py` — `GLViewWidget` skeleton preview.
- `pose3d/ui/timeline.py` — `QListView` IconMode filmstrip + delegate with RAG dots.
- `pose3d/ui/panels.py` — sidebar, joint-accuracy list, RAG legend, Selected Joint panel.
- `pose3d/ui/dark.qss` — theme matched to the mockup palette.

**Implement (copy verified patterns):** every widget from Phase 0 PySide6 section (flags, `itemChange`, `GLViewWidget`, `QListView` config, QSS load).

**Verify**
- App launches, loads a saved project, shows both views with joint overlays, 3D preview, and a populated filmstrip.
- Selecting a filmstrip frame loads that frame in both camera panels.
- Layout visually matches the PNG (side-by-side screenshot check).

**Anti-pattern guards:** panels talk only through `ProjectModel` signals; no cross-panel direct calls. Don't invent the layout — follow the PNG.

---

## Phase 7 — Manual correction path (load-bearing, first-class)

**Goal:** the safety net that makes flaky detection acceptable — drag a joint in 2D → live 3D re-solve → reversible edit → persisted for v2.

**Files**
- `pose3d/core/corrections.py` — apply/undo/redo stack; each correction a discrete reversible edit logged to `corrections.sqlite`.
- Wire `JointItem` drag → `ProjectModel.set_joint_2d` → single-joint re-triangulate (Phase 4) → re-fit → `pose3dChanged` → 3D + accuracy update. "Auto Recalculate 3D" toggle, Undo/Redo, Save Corrections buttons (all present in the mockup).
- One-view-only fix: constrain the joint along its viewing ray using parent bone length.
- `tests/test_corrections.py`

**Implement (copy verified patterns):** `itemChange`→signal→slot chain from Phase 0; single-joint re-solve from Phase 4.

**Verify**
- Drag a joint → 3D updates within one frame; only that joint moves.
- Undo restores exact prior 2D+3D state; redo re-applies; correction row written to SQLite.
- Occluded-in-one-view joint stays on its parent-bone-length sphere along the ray.
- No frame can be "dropped": every joint is draggable even at 0 confidence (RAG red still editable).

**Anti-pattern guards:** corrections must be discrete/reversible (no in-place mutation without a log entry); never silently discard a low-confidence joint.

---

## Phase 8 — Packaging (one-click launch)

**Goal:** a packaged app the client double-clicks; Blender stays external (documented dependency).

**Files:** `build.spec` (or scripted `pyinstaller` invocation), `README_RUN.md` (Blender path config, autofocus-lock instructions, checkerboard print).

**Implement:** `uv run pyinstaller --windowed --clean --name pose3d main.py`; bundle Qt `platforms` plugin + `PyOpenGL`; `--exclude-module PyQt5 --exclude-module PyQt6`; bundle detection ONNX or first-run cache warm.

**Verify:** fresh-machine (or clean user) launch works; 3D viewport renders (OpenGL context OK); detection loads bundled ONNX offline; export finds configured Blender.

**Anti-pattern guards:** no second Qt binding in the env; don't rely on `~/.cache/rtmlib` being pre-warmed on the client machine.

---

## Phase 9 — Final verification

1. **Docs match code:** re-grep for every Phase 0 anti-pattern across `pose3d/` — all must be absent:
   `grep -rn "estimatePoseSingleMarkers\|\.create()\|onnxruntime-gpu\|frame_step" pose3d/`
2. **mp4 ordering guard:** `grep -n "media_type" pose3d/export/blender_job.py` appears before `file_format='FFMPEG'`.
3. **Full pipeline on a real capture set:** import L/R batch → detect → calibrate/extrinsics → triangulate → fit → correct one joint → export BVH/FBX/mp4 → save/reload → outputs open in Blender.
4. `uv run pytest` — all phase tests green.
5. **Client-facing acceptance:** matches the 10-step flow in PROJECT_CONTEXT §2 and the mockup; BVH+FBX+mp4 all produced.

---

## Open questions to close with the client (do not block the build; isolate the impact)
1. **[Q5]** Exact target skeleton / is Mixamo standard rig the precise target? → only touches `core/skeleton.py` + `blender_job.py` mapping.
2. **[Q2]** Will they do the one-time intrinsic calibration and lock webcam autofocus? → if refused, fallback self-calibration from tags (poor; needs tags at varied depth — currently coplanar). Keep `intrinsics.py` swappable.
3. **[Q4]** Frames per session → sizing for filmstrip virtualization + export time; `QListView` `UniformItemSizes` already handles large N.
4. **[Q1 revisited]** If the client later says "marked figures forever," the `KeypointDetector` interface lets us add an HSV red-dot marker detector as a pure Phase-3 addition — no change to Phases 4–8.

## Suggested build cadence for the "by next weekend" target
Phases 1→2→(spike 5)→3→4→(finish 5)→6→7→8, demoing after Phase 4 (geometry proven headless) and after Phase 6 (visible dashboard) to keep the client updated as promised.

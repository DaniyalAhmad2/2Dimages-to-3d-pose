# Toe joints: the foot points where the toes point

_Design, 2026-09-21. Answers the client's build-17 question "does the program not detect toe
position?" (Prav, Woodlea Games). Scope agreed with the owner: ONE point per foot — the big toe —
so the foot's direction is measured instead of inherited from the shin. Two new joints, nothing
else about the feet._

## Goal

The character's feet point where the figure's toes point, in the 3D view and in the exported
BVH/FBX alike, driven by the big-toe keypoints the shipped detector already produces and the app
throws away. The toes are correctable like any other joint. Everything the client already sees —
frame colours, the accuracy dial, figure height, the joint-count messages, the floor — is unchanged
on any take, whether or not its feet are in frame.

## Non-goals (recorded, not forgotten)

- Heels and small toes (Halpe-26 22–25): foot pitch and roll. Not this cut.
- A heel/toe-based floor datum. The floor stays the lower ankle minus the rig's sole drop
  (`placement.ground_datum`), as ruled on 2026-09-21; a foot pointing straight down can dip below
  the grid, as it can today.
- Toe bones in the exported hierarchy. The rig's existing `foot.L/R` bones are driven; the
  BVH/FBX joint list does not change, so the client's Blender retarget is untouched.
- Any change to the detector, its weights, or `kpt_thr`.

## Facts the design rests on (verified in the tree at 16c1595)

- `core/skeleton.py`: `Joint` ends at `RIGHT_ANKLE = 14`; `NUM_JOINTS = len(Joint)`; `BONES` has
  14 edges, ankles are leaves; `map_halpe26` reads Halpe indices 0–16 only — 20 (L big toe) and
  21 (R big toe) are dropped on purpose (`:276-278`); `bonefit._FALLBACK_LENGTHS` has one entry
  per `BONES` edge.
- Face keypoints are addressed as `NUM_JOINTS + k` everywhere: `core/corrections.py:38-42,72,89`,
  `ui/model.py:707,726`, `ui/camera_view.py:98-108` (`FACE_KP_IDS`), and the bare `joint`
  integer column of `corrections.sqlite` (`io_project.py:243-246`). Changing `NUM_JOINTS` would
  re-read every saved eye/ear correction as a body joint.
- `io_project.load_project` pads arrays saved with fewer joints with NaN/False and preserves
  order (`:36-67`, written "to tolerate files saved with more joints (e.g. feet)") — except
  `corrected` (`:204`), which truncates and does not pad.
- The rig (`assets/character.npz`, 19 bones) has `foot.L/R` (head = ankle, tail = toe end), both
  leaves. `character._skin_matrices` gives them no target: `end is None → skin[b] = base`, the
  shin's matrix verbatim (`:1325-1326`). `_JOINT_FROM_RIG` reads the ankle from the foot's head;
  nothing reads the foot's tail. `fit_to_subject` is a weighted least squares over `BONES`.
- The export takes `pose_bone_matrices()` for all 19 bones (`blender_export.py:491`,
  `blender_job.py:517+`); the foot bones are driven already, with the shin's rotation.
- Aggregates: `ui/model.worst_per_joint`, `_states`, `_accuracy`, `figure_h_px`
  (`quality.figure_height_px`, a bbox over finite keypoints), the timeline status and Show
  filter, `main_window.py:1011` ("Only N of NUM_JOINTS joints") all run over every joint.
- Tests pinning the count/order: `test_skeleton.py:58-61,112-115`, `synth.sample_skeleton_3d`
  (15 literal rows), `test_integration_seams.py:170` (`eye = NUM_JOINTS + 1`),
  `test_integration_contracts.py:116-121`, `test_client_regression.py:190-214` (aggregates over
  `NUM_JOINTS`), `test_export_smoke.py:404` (FK vs view over `_joint_src`).

## Design

### 1. Skeleton — `pose3d/core/skeleton.py`, `pose3d/geometry/bonefit.py`

- `Joint` gains `LEFT_TOE = 15`, `RIGHT_TOE = 16` — appended, never inserted, so a saved 15-row
  project keeps every index. `NUM_JOINTS` becomes 17; `JOINT_NAMES` follows.
- `BONES` gains `(LEFT_ANKLE, LEFT_TOE)`, `(RIGHT_ANKLE, RIGHT_TOE)` (16 edges;
  `len(BONES) == NUM_JOINTS - 1` still holds).
- `CORE_JOINTS = tuple(j for j in Joint if j not in (LEFT_TOE, RIGHT_TOE))` and
  `EXTREMITY_JOINTS = (LEFT_TOE, RIGHT_TOE)`: the set every existing aggregate keeps using.
- `_DIRECT_FROM_HALPE26` maps 20 → `LEFT_TOE`, 21 → `RIGHT_TOE`; scores come with them;
  `reject()` in `detect/rtmpose.py` NaNs them below `kpt_thr` or outside the frame exactly as it
  does any joint. `derive_joints` (COCO-17) leaves them NaN. `MIXAMO_BONE` gains
  `mixamorig:LeftToeBase/RightToeBase`. Docstrings at `:77` and `:276` change to say the big toes
  are mapped and why the rest are not.
- `bonefit._FALLBACK_LENGTHS` gains the two foot edges (a foot from the ankle to the big toe is
  about 0.45 of the shin entry).
- `datasets/panoptic.py` maps no feet; its toes stay NaN (correct: Panoptic COCO-19 has none).

### 2. Face-point ids — `pose3d/core/skeleton.py`, `core/corrections.py`, `ui/model.py`, `ui/camera_view.py`, `core/io_project.py`

- `FACE_KP_BASE = 100` in `skeleton.py`, with `face_kp_id(k)`, `is_face_kp(joint)`,
  `face_kp_index(joint)`. Every `joint >= NUM_JOINTS` / `joint - NUM_JOINTS` becomes a call to
  these; `FACE_KP_IDS = [face_kp_id(k) for k in range(NUM_HEAD_KP)]`. Body joints will never
  reach 100; the constant never changes again.
- `corrections.sqlite` gets `PRAGMA user_version = 2`. On open, a file at version < 2 is migrated
  once: `UPDATE corrections SET joint = 100 + (joint - 15) WHERE joint >= 15`, with the 15 frozen
  as `_LEGACY_NUM_JOINTS` (never `NUM_JOINTS`), then the version is stamped. Idempotent; a new
  file is written at version 2 and never rewritten. The migration runs inside `load_project`
  before any row is read, and first copies the untouched file to `corrections.sqlite.pre-toes`
  beside it (skipped if that backup already exists).
- `io_project.py:204`: `corrected` is padded like every other array.
- `PIPELINE_VERSION` → 3 (informational; the loader keys nothing on it).

### 3. The character's foot — `pose3d/geometry/character.py`

- `_JOINT_FROM_RIG` gains `Joint.LEFT_TOE: (("foot.L", "tail"),)`, `RIGHT_TOE` likewise, so
  `posed_joints`, `fitted3d` and the export-vs-view gate all read the toe from the foot's tail.
- `_skin_matrices`: `foot.L/R` gain a target. When the toe joint is finite, the foot's matrix is
  the shin's matrix rotated **minimally** (the shortest rotation taking the inherited foot axis
  onto ankle→toe) — rotation only, rest length preserved, head fixed at the ankle. When the toe is
  NaN or rejected, exactly today's `skin[b] = base`. No roll reference: the foot keeps the shin's
  twist, which is what one point can determine.
- `fit_to_subject`: the two foot edges enter the least squares with weight 1 (not the ankles' 2);
  a NaN median (no toes in the take) contributes nothing, so a take without feet gets the scale
  it gets today, to the last digit.

### 4. Aggregates keep the core set — `pose3d/ui/model.py`, `ui/timeline.py`, `ui/panels.py`, `ui/main_window.py`, `pose3d/quality.py`

- `worst_per_joint`, `_states`' frame verdict, `_accuracy`/`frame_stat`, the timeline status and
  Show filter, `figure_height_px`, `rejection_note`'s denominator and the "Only N of M joints"
  messages iterate `CORE_JOINTS`. The toes keep their per-joint status (`panels.joint_status`) on
  the camera handles and the 3D dots, and nothing else. `test_client_regression` numbers do not
  move; the dial reads what it read yesterday.

### 5. UI — `pose3d/ui/camera_view.py`, `ui/main_window.py`

- Handles, bones, tooltips and placeholders come from `NUM_JOINTS`/`BONES`/`JOINT_NAMES` and need
  no change beyond `FACE_KP_IDS`. One nicety: `_placeholder_pos` parks a toe that was never seen
  in this view just below its ankle (the ankle's last position + a fraction of the figure
  height), not at the image centre.
- On opening a project whose toes are NaN in every frame while `keypoint_model == "halpe26"`,
  the status bar says once: "This project was detected before toe points existed — Run Detection
  adds them." Nothing is re-detected uninvited.

### 6. Export — `pose3d/export/blender_job.py`

- Character path: no change — the driven `foot.L/R` matrices now carry the aim. The BVH/FBX
  hierarchy is the same 19 bones. `test_helper_bones_are_pinned` holds (rotation only).
- Stick-figure path (no character asset): `build_armature` grows the two toe bones from `BONES`
  automatically; `rest_positions`' zero-fill fallback already covers NaN toes.

## Testing

Each change starts with its failing test; offscreen only.

- `test_skeleton.py`: `NUM_JOINTS == 17`, two appended members, 16 edges, `map_halpe26` puts
  Halpe 20/21 in the toes with their scores and nothing else moves (the inverse of the current
  "feet do not leak" test); `CORE_JOINTS` is the old 15 in order.
- `test_io_project.py`: a 15-row file loads with NaN toes and a **padded** `corrected`; a
  version-0 `corrections.sqlite` with `joint` 15–19 rows is migrated to 100–104 once (the file's
  `user_version` then reads 2; opening again changes nothing); a version-2 file is left alone;
  a body-joint row at 16 (a toe) is not touched.
- `test_corrections.py` / seams: face ids through `face_kp_id`; `test_integration_seams.py:170`
  uses `face_kp_id(1)`.
- `test_character.py` / `test_retarget.py`: with a finite toe the posed foot axis points at it
  (angle < 1°) and the foot's rest length is preserved; with a NaN toe the foot matrix equals
  the shin's; `posed_joints()[LEFT_TOE]` is the foot's tail; bones stay connected.
- `test_export_smoke.py`: the FK-vs-view gate runs over all 17 joints on a fixture with toes; the
  BVH still has 19 bones (no toe bone) and stays Y-up.
- `test_client_regression.py`: unchanged thresholds pass on the 15-row client fixture (toes NaN),
  proving the core-set rule.
- UI: `test_integration_contracts.py` (17 handles, ids `0..16`); a toe placeholder parks below
  the ankle; the timeline status of a frame with both toes missing and every core joint good is
  not "missing"; the dial ignores toes; the status hint appears once for an old project and not
  for a new one.
- `synth.sample_skeleton_3d` gains two toe rows (forward of the ankles).

## Compatibility and risk

- Old projects: open unchanged, toes as placeholders, one hint. Saved corrections: migrated once,
  reversible only by the sqlite backup the migration writes beside the file
  (`corrections.sqlite.pre-toes`). Cost if the migration is wrong: face corrections land on the
  wrong points — the backup and the test against a real pre-change file guard it.
- Older builds reading a 17-row project skip the extra rows (`io_project.py:39`).
- The accuracy dial, frame colours and figure height are computed over the core set by
  construction, so a take with cropped feet reads exactly as before — the promise to the client.
- The foot aim uses one point: pitch and yaw of the foot are measured, roll is inherited. A
  toe detected on the wrong foot (a Halpe left/right swap on a crossed-leg pose) aims the foot at
  the other foot's toe — the cross-view gate rejects it when the two cameras disagree, and the
  handle is there to fix it when they agree on the wrong thing.

## Docs and delivery

README ("The skeleton is 15 joints" → 17, with the toes named), `docs/client/RELEASE_NOTES_v1.md`
line ("Toe position: the foot now points where the big toe is detected; a toe the detector could
not see is a red dashed handle you can place"), `docs/DECISIONS.md` rulings (two joints; core set
unchanged; floor stays ankle-based; the fixed face-id base), `CLIENT_COMPLAINTS` entry.

Effort: about two working days including tests; ships as its own build after the client has
tested build-19.

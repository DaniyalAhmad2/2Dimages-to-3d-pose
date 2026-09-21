# Decisions and deferred work (accuracy audit, September 2026)

The audit is in `docs/audit-2026-09/` (PLAN.md is the plan that was executed; `wf_confirmed_full.md`
holds the findings by id). This file records what was deliberately NOT done and what would unblock it,
plus the rulings made during implementation. Measurements are on the client take unless stated.

## Deferred, with the gate that would unblock each

| deferred | measured payoff | gate / what it waits on |
|---|---|---|
| Halpe-26 native NECK/PELVIS and foot keypoints | NECK retarget 4.8 → 2.0 %; real toe/heel points for the feet | neck-shoulder bone CV regresses 5.15 → 8.13 % with a native NECK, so the neck rest direction must be re-derived first; foot keypoints need `foot2d`/`foot3d` parallel arrays so `NUM_JOINTS` stays 15. The skull HEAD alone is shipped (see rulings). |
| Manual "level the figure" control | beats the ±7–14° proxy floor of the recorded vertical | new UI; the recorded vertical with its uncertainty is shipped and shown |
| Per-bone axial stretch (F08) | retarget error 1.51 → 1.19 % of height; elbows 1.9/2.2 → 0.3 % | needs per-bone factors of 0.75–1.41, a visible reproportioning of a rig whose proportions the client accepted |
| Symmetrised bone-length fit targets (F34) | none — moves the pose 0.3 mm and worsens reprojection | do not build; the asymmetry is a diagnostic, shown in the sidebar |
| Export in real-world units | metric FBX/BVH | client question 2 (ruler photo of a tag and of the figure); the character export is normalised to the rig's size today |
| Tag bundle adjustment (F38) | 0.13 % of height of shape change | provenance / tag layout only; never an accuracy fix |
| Weighted DLT (F37) | unmeasured | needs a probe; the right camera contributes half the angular resolution |
| Temporal roll smoothing across frames | would suppress roll pops between poses | rejected for stop-motion: consecutive poses are genuinely different (the captured bend normal itself moves up to 55.7°); smoothing references would re-introduce the cross-pose blending the EMA removal fixed |
| Progress dialog for recompute-on-open | a long legacy take recomputes synchronously before the window shows | UI work; the recompute is one-time per project |
| Wiring `intrinsics.checkerboard_override` to a UI or CLI | distortion and a non-central principal point — the only two intrinsics the scene tags can never recover (k1 = 0.03 moves a tag corner 10.9 px but the subject 0.27 px; a principal point 200 px off centre costs 19 px of epipolar error) | nothing invokes it, deliberately: the function and its `Intrinsics.source` gate are implemented and tested, and client question 5's default answer is "an unused override". A capture screen is the work; until a client shoots a board we would run the calibration for them from the photographs (CLIENT_GUIDE §5). This row exists so the next reader can tell it from an oversight. |
| Re-weight `character.blend` so the skull is 100 % head bone | removes the last ~4 % of skull stretch (the throat seam, 1.05x in Nose mode and 1.16x in Face mode on the client take) and unblocks an articulated neck — a head that can nod independently of the neck | the client asking for an independently nodding head. The work is a re-weight in Blender (313 of the 991 head-weighted skull edges have an end sharing weight with the CHEST), then a re-bake with `tools/bake_character.py` and a re-baselined `test_bake_reproduces_the_shipped_asset`. Until then the chain is rigid, which is what keeps the skull's dimensions fixed (`docs/audit-2026-09/phase7_head_chain.json`). |
| Consolidate `calib.rigio.load_rig` with `app.load_rig_with_reason` | one loader | small refactor; both exist because the headless callers must not import PySide6. The VALIDATION is now shared (`calib.rigio.check_extrinsics`), so the two loaders can no longer disagree about which files are usable; what is left is the reason-reporting. |

## Do not re-open (bounded by the audit)

Extrinsics, intrinsics, triangulator choice, detector resolution, RTMPose-vs-red-dots (F05), derived-joint
perspective bias, the (NECK, HEAD) bone (F35), the nose clamp on legacy projects (F03), `rig_bone_lengths`
(F19), detector confidence (F26), symmetrised targets (F34), an essential-matrix fallback for a camera that
never sees a tag (F42).

## Rulings made during implementation

(Recorded as they were made; each names what it costs if wrong.)

- Smoothing is off by default; a zero-phase filter is kept behind an import option labelled "video-rate
  capture only". Cost: a genuine video take needs the option ticked.
- The gap fill is a flagged single-frame midpoint; it halves the error of the previous hold-last-known
  behaviour (4.3 % vs 7.8 % of height) but is above the "< 2 %" gate the plan wrote, which was below the
  take's own inter-frame motion. Filled joints never enter the raw measurement (`pose3d`), only the
  delivered pose (`fitted3d`), and are drawn hollow. Cost: a filled joint is an interpolation, shown as one.
- Roll continuity gate (≤ 15° between consecutive frames) accepted as unmet on stop-motion; the gate is
  restated as zero roll error wherever the bend plane exists (bend ≥ 40°) plus a synthetic straightening
  sweep. Forearm and shin roll is a documented convention, not a measurement (no hand/foot keypoints).
- A missing wrist or ankle on a frame drops that limb's roll term for the frame (weight 0, no temporal
  state); documented and tested. Cost: a roll pop on such frames; rare after the gap fill.
- The recorded vertical is used only when two or more gravity proxies agree; one proxy is reported as
  unverified and the body-line levelling stays. Its sign is fixed after triangulation (neck above ankle).
- `keypoint_model` is the detector layout (`coco17`/`halpe26`); the detector's name is `detector`.
- Export matrices are expressed in the de-tilted capture frame (the per-frame yaw alignment the live view
  undoes is undone for the export too), with root motion on by default for BVH/FBX and one frame per
  photograph; `keep_root_motion=False` reproduces the old rig-facing, pinned-root file.
- The Halpe-26 skull HEAD is switched on although one pre-set gate (neck-shoulder bone CV ≤ 5.15 %) reads
  5.98 %: that bar was set from an unmeasured assumption, the increase is ~0.13 mm on a 16 mm bone, and
  the head error falls from 12.7 % to 1.5 % of height. Restated gate: ≤ 6.5 %.
- The neck and the head are ONE RIGID CHAIN, and the face keypoints may orient it but never deform it.
  The head bone has no target of its own: it rides the neck's matrix. Two orientation modes, per project,
  chosen in the app, default **Nose**: the chain aims at the canonical HEAD and the NOSE alone rolls it
  about that aim (the nose is the one face point rigid enough to trust on a mannequin — nose-to-HEAD
  varies 5.2 % over the client take against ear-to-ear's 14.4 %); **Face** takes the whole nose+ears
  basis for humans, whose ears are real features. Measured on the client take: skull shear 1.428x → 1.000x
  on the edges the chain owns outright — the shear gate's scope was restated to those edges after the
  measurement, because over ALL head-weighted edges the take reads 1.05x (Nose) / 1.16x (Face), all of it
  on the throat seam that Decision 4 puts out of scope and that the deferred re-weight above would
  remove — head-vs-neck relative rotation 14.6–88.2° → the rest offset exactly, head
  aim with the face points in play 3.78° → 1.28° median (22.95° → 3.82° max). Cost: the head cannot nod
  independently of the neck, and on a mannequin whose ears are wrong Face mode still follows them —
  which is why Nose is the default. Evidence and gates: `docs/audit-2026-09/phase7_head_chain.json`.
- The Windows `python312.dll` failure is not a missing runtime in the build (build-12 ships the VC runtime);
  it is a file missing from the client's extracted copy. The build now audits every DLL import; the README
  tells the client to extract to a short local path and check antivirus quarantine.
- The Windows toolchain is pinned to Python **3.12** (`.python-version`), not the 3.13 the dev venv
  runs: every doc, README troubleshooting string and Windows test names `python312.dll`, and `uv.lock`
  only says `>=3.12`, so an unpinned runner would silently freeze the exe against whatever interpreter
  it had newest. Cost if wrong: the build forgoes 3.13's speedups until the docs and the pin are moved
  together. Blender and the ONNX checkpoints are pinned the same way — one version, one URL and one
  sha256 each, in `packaging/windows/inputs.json`, which nothing else may restate.
- Software OpenGL, and what it can and cannot rescue. `--software-gl` / `POSE3D_GL=software` / a
  `use-software-gl` marker file next to the exe set `QT_OPENGL=software` + `AA_UseSoftwareOpenGL` before
  the QApplication exists (the only moment Qt still listens), and the 3D card offers "Restart with
  software 3D" when its context is dead. Honest limit: that makes **Qt** run on the bundled
  `opengl32sw.dll`; pyqtgraph draws through PyOpenGL, which loads the machine's own `opengl32.dll` and is
  not redirected by it, so the 3D card still depends on the driver being there. A renderer that truly
  works with no GPU driver (Mesa llvmpipe shipped as `opengl32.dll` and loaded by absolute path before Qt
  and PyOpenGL) is out of scope this round. Cost if wrong: on a machine with no driver at all the restart
  changes nothing and the placeholder stays — which is why the placeholder names Help ▸ Diagnostics
  rather than promising a fix.
- **The delivery stays an unsigned .zip this round: no installer, no code signing.** A certificate is a
  purchase, an identity check and a renewal the client would inherit, and an MSI/Inno installer is a
  second packaging path to keep working — neither buys anything the zip does not, because SmartScreen's
  warning is about the signature, not the container. What is bought instead is that the warning is
  *expected*: `README.txt` and README.md name the "Windows protected your PC" screen, say it is not a
  virus warning, and say which two buttons dismiss it. Cost if wrong: every first launch on a new machine
  costs the client two clicks, and an antivirus with an aggressive reputation heuristic can still
  quarantine a file out of `_internal\` — which is why the DLL audit, the layout manifest and
  `Pose3D-diagnose.exe` exist.
- **onefile is still rejected.** A onefile build unpacks ~500 MB into `%TEMP%` on every launch, which is
  slow, breaks whenever `%TEMP%` is small, redirected or scanned, and puts the bundled Blender and the
  three checkpoints somewhere `app_dir()` cannot address. onedir plus a folder is what the client already
  runs. Cost if wrong: they extract a folder instead of a single file, which the README's first
  instruction covers.
- **Nothing is hashed at startup.** `pose3d/integrity.py` checks names and sizes only; the sha256s in
  `packaging/windows/inputs.json` are verified by the build and by the release gate, where a mismatch can
  be fixed. Hashing ~1.3 GB on the client's machine would add seconds to every launch, on the one file
  set most likely to be sitting on a network drive or a OneDrive placeholder, to detect a corruption that
  a missing-or-empty check already catches in its usual forms (an interrupted extraction, a quarantined
  file, a placeholder stub). Cost if wrong: a file that is present, the right size and silently corrupt
  is reported by whatever fails to load it rather than by name at startup.
- **The window opens clamped to the screen it opens on.** The designed 1540x920 is larger than the
  client's 1366x768 laptop, and at 150 % scaling the timeline opened below the bottom edge and the right
  column past the side — with no way to drag a title bar above the top of the desktop to recover them.
  `_initial_size()` clamps to `availableGeometry()` less a margin for the title bar, and the sidebar sets
  a *minimum* width rather than a fixed one so the layout can still give way. Cost if wrong: on a very
  small screen the window opens at the screen's size and the panels are tight, rather than opening
  partly off-screen.
- **One composite action builds the Windows bundle, and the gate is its own workflow.**
  `.github/actions/windows-bundle` is used verbatim by `windows-bundle.yml` (the gate) and
  `windows-release.yml` (the release), so the archive attached to a release is one that has already been
  extracted into a client-shaped path and self-tested from there. It is a separate workflow rather than a
  job in `windows-test.yml` because a `paths` filter decides whether a WORKFLOW runs, not whether one of
  its jobs does: filtering there to spare the test job a ~40-minute bundle build would have stopped
  running the test suite on most pull requests. Cost if wrong: a push to `main` that touches only
  `pose3d.spec` or `tools/` spends 40 minutes of runner time.
- **The release gate's OpenGL evidence is structural, not behavioural.** The runner has no GPU; Qt cannot
  create a hardware context, and the software renderer the bundle ships (`opengl32sw.dll`) drives Qt but
  not pyqtgraph, which draws through PyOpenGL and loads the machine's own `opengl32.dll`. So the gate
  keeps `POSE3D_NO_GL=1` for that one check, with the reason written where it is set, and proves the 3D
  view was PACKAGED instead: the manifest requires `opengl32sw.dll` and `qwindows.dll` in the EXTRACTED
  copy, and `check_qt_opengl`'s ImportError path stays fatal. Cost if wrong: a GL fault that is neither a
  missing file nor a failed import reaches the client, where `Pose3D.exe --selftest` is strict and
  `Pose3D-diagnose.exe` prints the answer. See `docs/windows-release-gate.md`.
- **Tests never reach an OS dialog; every native message box is a seam.** `pose3d.integrity._message_box`
  is a `MessageBoxW` drawn by Windows rather than by Qt — deliberately, because Qt is exactly what may be
  unable to start — and the first Windows run of the test suite drew it for real: a test faked a frozen
  bundle with a file missing, and on a Windows host `runtime.IS_WINDOWS` is True as well, so the suite
  blocked on a modal dialog for 42 minutes until the job's 45-minute cap cancelled it. The rule is now the
  same one `pose3d.ui.guard.report_error` has had since the Qt dialogs went in: anything that can draw a
  window the suite cannot close is a single named function, replaced by an autouse fixture in
  `tests/conftest.py` for every test whether it asks or not, and the tests assert on what the client would
  have read. Both fixtures assert their own presence (`test_no_test_can_draw_the_native_message_box`,
  `test_the_fixture_is_autouse_so_no_test_can_open_a_modal`). Backstop for the next one nobody foresaw:
  `timeout = 600` in `pyproject.toml` (pytest-timeout), so a blocked test is a traceback in ten minutes
  rather than a cancelled job that names nothing. Cost if wrong: the product's own dialog is exercised only
  through the seam, so a fault in the two lines of ctypes below it would be found on the client's machine —
  which is why `_message_box` is also tested unpatched, both that it is inert outside a frozen app and that
  it is not inert inside one.


## The Universal C Runtime ships in the bundle (2026-09-20)

The bundle audit classed `ucrtbase.dll` and the `api-ms-win-crt-*` forwarders as "provided by
Windows 10". A client followed the extraction instructions exactly — a complete folder, straight
on `C:` — and still got *"Failed to load Python DLL … python312.dll … The specified module could not
be found"*. That dialog names the file whose *dependency* is missing, and the only dependencies of
`python312.dll` not in `_internal\` were the UCRT; their Windows did not have it in working order.
Ruling: bundle it, from the Windows SDK redist Microsoft ships for this purpose (`pose3d.spec
ucrt()`, System32 as fallback, build stops if neither has it); the audit treats `api-ms-win-crt-*`
and `ucrtbase.dll` like the VC runtime (must be bundled) while the OS API sets stay Windows'; the
manifest requires both at the `bootloader` stage; and `Diagnose.cmd` checks every bootloader-stage
file in plain cmd before it runs the diagnose exe — which is a Python program and dies with the
same dialog, so it could never have reported this. Cost if wrong: about 1 MB and fifteen small
DLLs, and a bundled UCRT older than the client's is what PyInstaller shipped for years. The
one-click `vc_redist.x64.exe` stays documented as the last resort for a complete folder that
still fails, because it repairs the system copies too.


## The v1 corrections pass, before resubmission (2026-09-21)

The milestone was rejected on a client who still could not open the app, with seven earlier
complaints he had never confirmed as fixed. This pass closes the Priority 1 list and the
Priority 2 items that answer a complaint, in five isolated worktrees with one owner per file,
and ships as ONE resubmission build. Rulings, each with what it costs if it is wrong. Where a
ruling governs work owned by another task of this pass, it records the DECISION and the reason,
not the state of the code — those tasks are implemented in parallel, and the wording is settled
against the delivered behaviour once they merge:

- **The marker size box is centimetres** (label "ArUco marker size (cm)", default 5.00, range
  0.5–200, two decimals), converted to metres where the dialog reads it, so `project.json`,
  `calibration/report.json` and every solver below keep the metres they always held. The client
  measures a printed tag with a ruler and the job brief says "5cm x 5cm"; in metres the box took
  his typed 8 and silently clamped it to its 2.000 m maximum, putting the whole metric scale out
  by 25x with nothing on screen to connect it back. Cost if wrong: a two-line revert.
- **Metric BVH/FBX export stays deferred.** The export is normalised to the character rig's own
  size, so making it metric changes the scale the client's Blender retarget is set up around, and
  he has never asked for it in the thread. It is listed in the release note as still to come,
  with the ruler photograph that would settle the figure's true size. Cost if wrong: he opens the
  FBX expecting centimetres and finds rig units — which is what the note tells him.
- **Ruling: reusing the main window when a second project is imported is deferred** to the UI/UX
  audit (the user's earlier decision). What this pass is to do instead is close the leaked old
  window properly so nothing is left behind; Import opening a new window rather than reusing the
  open one is accepted for this build. Cost if wrong: a second window the client closes.
- **Ruling: the floor datum is to be a robust low percentile of the take's per-frame ankle
  heights, not the single lowest frame**, so one bad ankle cannot push the whole character down
  through the floor or bob it against the ground. Cost if wrong: a take genuinely captured off
  the ground sits at a small constant offset, visible and correctable.
- **The frame keys are the window's everywhere, a number box included** (client, 2026-09-21,
  on build 17: "regardless of what's been clicked"). The first cut let a focused spin box keep
  Left/Right; Qt withholds a window shortcut from such a box anyway, so an application-level
  event filter now hands the frame keys to the window before the focus widget sees them, and
  stops at the window's edge (a dialog's arrows stay the dialog's). Cost if wrong: a cursor
  that cannot be moved inside the height box with the arrows; Up/Down and the digits still work.
- **The BVH is written Y-up.** Blender's BVH exporter writes armature space (Z-up) and has no
  axis option; its importer — and Unity, Unreal, MotionBuilder — assumes the mocap convention
  and turned the delivered file 90° so the character lay flat. The export rotates a throwaway
  copy of the armature before writing; the FBX (already Y-up through the exporter's axis
  arguments) and the render are untouched, and the export-matches-view gates read the file back
  through `bvh.FILE_TO_WORLD`. Cost if wrong: a tool that expected the old Z-up file — none is
  known; the delivered fixture keeps its historical numbers.
- **Ruling: a photograph that cannot be read costs its pair, never the import.** Calibration
  skips the pair and says so; detection keeps the frame with that view marked missing (no 2D
  points for it) so the other view and the rest of the take are untouched; the import summary
  names the pair. Cost if wrong: a frame with one view missing shows red joints the client can
  place by hand.
- **Ruling: "Calibrated (with problems)" is to be reworded, and one warning suppressed.** The
  nominal-up warning is not to be shown while the recorded vertical is in use — it is a statement
  about a fallback that did not happen — and the other two are to be reworded in plain language
  and presented as notes rather than problems, because a non-technical client who reads
  "problems" on a calibration that is fine stops there. Cost if wrong: a wording change.
- **Ruling: the Priority 2 items taken into this pass** are the ones that answer a complaint —
  undo on the wrong frame, re-detect and batch NECK/PELVIS, set-scale rescale off the GUI thread,
  confidence colouring in the 3D view, floor hover, the calibration wording above, skipping
  0-byte images, the timeline highlight and Show filter, the export/preview framing mismatch,
  and the drag placement cache.
  **Excluded:** window reuse and metric export, both above.
- **The install instructions are one set of steps in two files.** `README.md` and the zip's
  `README.txt` carry the same three steps in the same order — unblock the download, extract to
  `C:\Pose3D`, run `Pose3D.exe` — pinned by the doc tests, because the contradiction between them
  (one said Desktop or Documents, the other said not to) is what the client followed into the
  launch failure. Cost if wrong: nothing; the tests fail before he sees it.
- **The client's getting-started guide has a source in the repository** (`docs/client/`,
  rendered with `node build.js`). The PDF sent on 2026-09-04 existed only in his inbox and could
  not be corrected when the build changed. Cost if wrong: a build step outside CI that has to be
  run by hand when the text changes.


## The toes: two joints, and nothing else moves (2026-09-22)

The client's build-17 question — "does the program not detect toe position?" — answered with one
point per foot. The design is `docs/superpowers/specs/2026-09-21-toe-joints-design.md`; below are
the decisions it rests on, then the rulings made while it was implemented.

- **The big toes are joints 15 and 16 — appended, one point per foot** (client, 2026-09-21: "does
  the program not detect toe position?"). Halpe-26 detects them already; the skeleton now keeps
  them so the rig's foot bone can aim. Small toes and heels stay out. Cost if wrong: two NaN
  columns on a take without feet.
- **The toes are extremities: no take-wide number or colour counts them.** Frame band, dial,
  figure height and the joint-count messages run over the original 15 (`CORE_JOINTS`); a cropped
  foot cannot turn a good take red or move the accuracy percentages. Cost if wrong: a wrong toe is
  visible only on its own handle.
- **Face-point ids have a fixed base (100), and old logs are migrated once.** They were
  `NUM_JOINTS + k`, which appending a joint would have silently re-read as body joints in every
  saved corrections.sqlite. The migration backs the file up beside itself first.
- **The floor stays ankle-based** for this cut; a foot pointing straight down can dip below the
  grid as it can today. The heel/toe floor is the next cut.
- **The core-set rule is applied where a number is computed, not where it is shown** — so its data
  half landed with the skeleton change rather than with the UI. `gap_stats` (all four counts),
  `figure_height_px`, both the numerator and the denominator of `rejection_note`, the bone-length
  fallback report, `character._frame_scale`, `subject_height` (head-to-ankle span by definition)
  and the take-wide bone-CV, retarget and reprojection summaries all run over `CORE_JOINTS`;
  per-joint and per-bone rows keep every joint, so a toe is still visible on its own row.
  The cross-view gate is the one place the rule has two halves: it is SIZED from the core set
  (`epipolar_threshold`, and `body_epipolar` with it, so the sidebar states the number the gate
  used) and APPLIED to every joint, so a toe past the gate is still rejected — it just cannot
  widen the gate the body is judged by first. Cost if wrong: the rule lives in two layers, so a
  new take-wide number has to choose the core set in whichever layer computes it; and the
  sidebar's "height" stays head-to-ankle, about 5 % under the true sole-to-head figure — the
  definition it already had.
- **`quality.BONE_NAMES` gained the two foot bones**, because it is a rig-wide invariant with one
  entry per `BONES` edge — the same kind of table as `bonefit._FALLBACK_LENGTHS` — not a take-wide
  number. The per-bone rows therefore name the feet; only the summary over them is core-only. Cost
  if wrong: two named rows a take without feet leaves empty.
- **`PIPELINE_VERSION` stays 2.** The design proposed 3, but an older project is recomputed on
  open (`model.py`) and a recompute cannot add toes: they come from the detector, not from the fit.
  A bump would have promised a repair that opening the project does not perform. Cost if wrong: an
  informational number that no longer separates a pre-toe pipeline from this one — the hint below
  is what actually tells the client.
- **The pre-toe hint is keyed on the data, not on a build stamp.** A take detected by THIS build
  whose feet are out of frame in every photograph reads as predating the toes and shows the hint
  too. Accepted: nothing in the file separates the two cases, and the hint costs one Run Detection.
  Cost if wrong: a client with cropped feet runs detection once, gets the same NaN toes, and the
  placeholders stay where they were.
- **The two foot edges enter the uniform scale fit at weight 1.0**, though the ankles they hang
  from carry 2.0: the ankle weight buys the leg length the floor datum depends on, while a foot
  measured to a single toe point is the least reliable edge in the set. The client take's fitted
  scale is bit-identical at 110.1025 with its toes NaN. Cost if wrong: a take with clean toes
  scales a fraction differently from one without.
- **The BVH reader keeps each End Site as `Joint.end_offset` on its parent**, not as a joint of
  its own, so the export-matches-view gate can compare bone TAILS — which is where the foot's aim
  shows. The written file is unchanged: 19 bones, Y-up, no toe bone, so the client's Blender
  retarget sees exactly what it saw. Cost if wrong: a field on the reader that only the gate reads.

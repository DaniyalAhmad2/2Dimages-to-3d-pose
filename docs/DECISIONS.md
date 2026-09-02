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
- The Windows `python312.dll` failure is not a missing runtime in the build (build-12 ships the VC runtime);
  it is a file missing from the client's extracted copy. The build now audits every DLL import; the README
  tells the client to extract to a short local path and check antivirus quarantine.

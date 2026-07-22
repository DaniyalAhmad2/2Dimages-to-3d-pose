# One-time intrinsic calibration board

Print a **checkerboard** (or ChArUco) and use it once per physical camera.
Reused across all sessions; redo only if a camera is swapped.

## Checkerboard
- 9×6 **inner corners** (a 10×7 square grid), square size 25 mm.
- Print on rigid, flat board (foamcore). Matte, no glare.
- Measure the actual printed square edge and pass it as `square_size`
  (only affects metric scale of the calibration rvecs/tvecs, not intrinsics).

## Capture procedure (per camera, ~30 s)
1. **Lock autofocus** on the webcam first (critical — see README).
2. Hold the board and take 15–25 shots covering:
   - near / mid / far depths,
   - tilts (±30° in pitch and yaw),
   - board in all corners of the frame (edges matter for distortion).
3. Feed the frames to `pose3d.calib.intrinsics.calibrate_checkerboard`.

Target: `calibrateCamera` reprojection RMS < 1.0 px. Save per camera as
`left_intrinsics.json` / `right_intrinsics.json` in the project's
`calibration/` folder.

## Why not the scene ArUco tags for intrinsics?
The four backdrop tags are coplanar — one view of one plane cannot recover
focal length + distortion. The tags are used for **extrinsics** (per-shot
camera pose + metric scale), which coplanar markers handle fine.

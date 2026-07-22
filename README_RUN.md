# Pose3D — Running & Building

A desktop tool that reconstructs a metric 3D pose from two camera views,
lets you correct joints by dragging, and exports BVH / FBX / mp4.

## 1. Requirements

- Python 3.13 (managed by `uv`).
- Blender 5.1.1 for export (external; not bundled). Default path:
  `/home/athena/Downloads/blender-5.1.1-linux-x64/blender`. Override with the
  `POSE3D_BLENDER` environment variable, or put `blender` on your PATH.
- For markerless detection: RTMPose ONNX models auto-download on first use to
  `~/.cache/rtmlib/hub/checkpoints` (~140 MB). For an offline/packaged app,
  pre-warm that cache or bundle the ONNX under `pose3d/detect/models/`.

## 2. Environment setup

```bash
uv venv --python 3.13
uv pip install \
  numpy scipy opencv-contrib-python onnxruntime pillow \
  pyside6-essentials shiboken6 pyqtgraph PyOpenGL tqdm pytest
uv pip install --no-deps rtmlib      # avoid pulling duplicate opencv wheels
```

> **Network note (this machine):** the local DNS resolver returns unreachable
> IPv6 addresses for the PyPI wheel CDN and the OpenMMLab model host, which
> hangs `uv`/`pip`/`curl`. Prefix network commands with `RES_OPTIONS="no-aaaa"`
> to force IPv4 (glibc ≥ 2.36). A pre-downloaded `wheelhouse/` is included; add
> `--find-links wheelhouse` to install the heavy wheels without the network.
> Use `opencv-contrib-python` (has `cv2.aruco`); do **not** also install
> `opencv-python`/`-headless` — three cv2 copies conflict.

## 3. Run the app

Use the venv's Python directly. Do **not** use plain `uv run` here: it re-syncs
the venv against `pyproject.toml` on every launch, which (a) undoes the
imperative install below and (b) hangs on this machine's IPv6/DNS issue. Either
run the interpreter directly, or pass `uv run --no-sync`.

```bash
.venv/bin/python -m pose3d.app                     # empty session
.venv/bin/python -m pose3d.app data/demo_project   # load a saved project
# equivalently: uv run --no-sync python -m pose3d.app data/demo_project
```

## 4. Validate the pipeline on real data (CMU Panoptic)

```bash
# geometry core vs real dome calibration + real 3D poses (no images needed)
.venv/bin/python -m tests.validate_panoptic_geometry
# full pipeline: RTMPose on real images -> triangulate -> vs ground truth
.venv/bin/python -m tests.validate_panoptic_rtmpose
# rebuild the demo project + dashboard screenshot + export
QT_QPA_PLATFORM=offscreen .venv/bin/python -m tests.build_demo_project
```

## 5. Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q \
  --ignore=tests/validate_panoptic_geometry.py \
  --ignore=tests/validate_panoptic_rtmpose.py
```

## 6. Package (one-click app)

```bash
.venv/bin/python -m PyInstaller pose3d.spec --noconfirm --clean
# result: dist/pose3d/pose3d
```

Notes: the spec excludes other Qt bindings (PyInstaller ≥ 6.5 forbids mixing),
bundles PyOpenGL for the 3D view, and ships `dark.qss` + `blender_job.py`.
Blender stays external — set `POSE3D_BLENDER` on the target machine.

## 7. Client setup checklist (capture rig)

- Lock each webcam's **autofocus** (focal drift invalidates the one-time
  intrinsic calibration).
- Do the one-time per-camera checkerboard calibration (see `assets/`).
- Keep the four ArUco tags visible in every shot (they give per-shot
  extrinsics + metric scale).
- Name files so left/right pairs match (`left_0001.jpg` / `right_0001.jpg`).

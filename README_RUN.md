# Pose3D — Running & Building

A desktop tool that reconstructs a metric 3D pose from two camera views,
lets you correct joints by dragging, and exports BVH / FBX / mp4.

## 1. Requirements

- Python 3.13 (managed by `uv`).
- Blender 5.x for export. Resolved in this order (`pose3d/config.py`):
  `POSE3D_BLENDER` → a `blender/` folder beside the executable → a standard
  install (`C:\Program Files\Blender Foundation\Blender *` on Windows,
  `/opt/blender` elsewhere) → `PATH`. The shipped builds bundle it; a checkout
  needs one of the others.
- RTMPose ONNX weights. Found via `POSE3D_MODELS` → a `models/` folder beside
  the executable → `pose3d/assets/models/` → rtmlib's cache
  (`~/.cache/rtmlib/hub/checkpoints`). If none of those has them, rtmlib
  downloads ~150 MB on first use — fine for a checkout, fatal for a windowed
  build, which is why both deliveries ship them. `tools/fetch_weights.py`
  stages them.

## 2. Environment setup

```bash
uv sync --frozen --all-groups        # installs exactly what uv.lock pins
```

`uv.lock` is committed, and CI and the Docker image install from it, so all
three deliveries carry the same versions the tests ran against. Do not resolve
fresh unless you mean to — see `pyproject.toml` for why opencv must be the
*contrib* build (`cv2.aruco`, for calibration), *headless* (a second Qt breaks
the PyInstaller bundle), and held below 5.0, and why `rtmlib` is declared as
having no dependencies of its own.

> That last override is a **uv resolver** feature. It does not follow the
> package into pip: `pip install -r` a uv-exported requirements file without
> `--no-deps` and pip re-reads rtmlib's own metadata and reinstates plain
> `opencv-python`, shadowing contrib and taking `cv2.aruco` with it.

> **Network note (this machine):** the local DNS resolver returns unreachable
> IPv6 addresses for the PyPI wheel CDN and the OpenMMLab model host, which
> hangs `uv`/`pip`/`curl`. Prefix network commands with `RES_OPTIONS="no-aaaa"`
> to force IPv4 (glibc ≥ 2.36).

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
.venv/bin/python -m tests.test_panoptic_geometry
# full pipeline: RTMPose on real images -> triangulate -> vs ground truth
.venv/bin/python -m tests.test_panoptic_rtmpose
# rebuild the demo project + dashboard screenshot + export
QT_QPA_PLATFORM=offscreen .venv/bin/python -m tests.build_demo_project
```

## 5. Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q \
  --ignore=tests/build_demo_project.py
```

The Panoptic validations above are collected as tests too, so they run
whenever the dataset has been downloaded and skip when it has not.

Tests needing Blender, the character rig or the ONNX weights **skip** when
those are absent, which is right on a laptop and wrong in CI — the entire
export path would report green having never run. Set the matching variable to
turn a skip into a failure (see `tests/gates.py`); the Windows workflow sets
all three (`POSE3D_REQUIRE_PANOPTIC=1` does the same for the dataset above):

```bash
POSE3D_REQUIRE_BLENDER=1 POSE3D_REQUIRE_ASSETS=1 POSE3D_REQUIRE_WEIGHTS=1 …
```

## 5b. Self-test

The same checks the container and the Windows bundle are verified with —
Blender version, the rig loads, weights present offline, a real GL context, and
an actual export:

```bash
.venv/bin/python -m pose3d.selftest              # add --no-video to skip the render
docker run --rm --entrypoint python pose3d:latest /app/docker/smoketest.py
build/Pose3D-Windows/Pose3D.exe --selftest       # on Windows
```

A missing preview video is reported as *degraded*, not failed: rendering drives
EEVEE through whatever OpenGL exists, and a GPU-less CI runner has none.
`--require-video` (which `docker/build.sh` passes) promotes that to a failure,
because the image ships Mesa specifically so it works.

## 6. Package

### Linux / development

```bash
.venv/bin/python -m PyInstaller pose3d.spec --noconfirm --clean
# result: dist/pose3d/pose3d
dist/pose3d/pose3d --selftest --no-video
```

The spec excludes other Qt bindings (PyInstaller ≥ 6.5 forbids mixing), bundles
PyOpenGL and pyqtgraph's lazily-imported GL items, and ships `dark.qss`,
`blender_job.py` and the character rig. It **fails the build** if any declared
data file is missing, rather than producing a bundle that breaks at run time.

### Windows client bundle

Built by `.github/workflows/windows-release.yml` on a tag or via manual
dispatch — we have no Windows machine, so the runner is the only place this
happens. It builds the exe, assembles the bundle, then runs `Pose3D.exe
--selftest` **against the assembled bundle** with `POSE3D_BLENDER` and the
rtmlib cache deliberately unavailable, so it can only pass by finding what the
bundle itself ships.

Locally, given a Windows machine:

```powershell
uv run pyinstaller pose3d.spec --noconfirm --clean
uv run pwsh tools/make_windows_bundle.ps1 -Dist dist/Pose3D -Out build/Pose3D-Windows -Zip
```

Resulting layout (`pose3d.runtime.app_dir()` resolves all of it relative to the
executable):

```
Pose3D-Windows/
  Pose3D.exe + _internal/    the app
  blender/blender.exe        the export engine
  models/*.onnx              pose weights, so nothing downloads
  workspace/                 projects and exports default here
  README.txt
```

Roughly 1 GB zipped, 1.6 GB extracted — Blender alone is ~1 GB. Shipped
unsigned; `packaging/windows/README.txt` tells the client how to click through
SmartScreen.

`-Zip` archives the assembled folder through `tools/bundle_zip.py`, which
refuses any entry too deep for Windows to extract and writes
`build/Pose3D-Windows.zip` (beside the `-Out` folder, named after it).

`tools/fetch_weights.py` stages the ONNX weights for both the bundle and the
Docker image, reading the expected filenames from rtmlib's own tables so a
rtmlib upgrade cannot leave us shipping a checkpoint the app then ignores.
`tools/make_icon.py` regenerates `packaging/windows/pose3d.ico`.

## 7. Client setup checklist (capture rig)

- Lock each webcam's **autofocus** (focal drift invalidates the one-time
  intrinsic calibration).
- Do the one-time per-camera checkerboard calibration (see `assets/`).
- Keep the four ArUco tags visible in every shot (they give per-shot
  extrinsics + metric scale).
- Name files so left/right pairs match (`left_0001.jpg` / `right_0001.jpg`).

# Pose3D

Reconstructs a metric 3D pose from two synchronised camera views, lets you
correct any joint by hand, and exports the result as a rigged character
(FBX / BVH) and a video.

---

## Running it

There are two ways to run it, and on Windows the native one is better.

| | Windows `.exe` | Docker |
|---|---|---|
| Platforms | Windows 10/11 | Windows, macOS, Linux |
| 3D view | uses the graphics card | software-rendered, noticeably slower |
| Prerequisites | none | Docker Desktop |
| File access | anywhere on the machine | only pre-shared folders |
| Download | ~1 GB zip | ~1.5 GB image |

The 3D view is where most of the work happens — scrubbing frames, checking
joints — so on Windows the native build is the one to use. Docker remains the
answer for macOS and Linux, and for reproducing a problem in a known-identical
environment.

### Windows (native)

Download `Pose3D-Windows.zip` from the
[Releases page](https://github.com/DaniyalAhmad2/2Dimages-to-3d-pose/releases),
extract it somewhere writable — Desktop or Documents, **not** Program Files —
and run `Pose3D.exe`.

Blender, the pose models and everything else are inside the folder. No
installs, no internet.

The first launch shows a blue **"Windows protected your PC"** screen. That is
SmartScreen, and it appears for any application that has not been code-signed;
it is not a virus warning. Click **More info** → **Run anyway**. It appears
once.

If the app fails to start or an export fails, `pose3d-log.txt` is written next
to `Pose3D.exe` and will normally say why.

### Docker (any platform)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/).
Nothing else — Python, Blender and the pose model are all inside the image.

```bash
docker compose up
```

Then open **<http://localhost:8080>** in your browser. The application appears
there and behaves exactly like a desktop window — mouse, dragging, everything.

To stop it, press `Ctrl+C` in the terminal (or `docker compose down`).

On **Windows**, install Docker Desktop for Windows (it enables WSL 2 for you if
it isn't already on: Windows 10 21H2 or later, or Windows 11), put
`docker-compose.yml` in a folder, open PowerShell there and run the same
command. The first run downloads about 1.5 GB; after that it starts in seconds
and needs no internet. To use a different port:

```powershell
$env:POSE3D_PORT=8090; docker compose up
```

### macOS and Linux

Identical: `docker compose up`, then <http://localhost:8080>. To change the port,
`POSE3D_PORT=8090 docker compose up`.

### Which folders the app can see

The application runs in a container, which means it can **only** open files in
folders that are explicitly shared with it — the rest of your disk is invisible
to it. Two are shared:

| Inside the app | On your machine | |
|---|---|---|
| `/workspace` | the `workspace` folder next to this file | read + write |
| `/host` | the folder you launched from | read only |

To let it reach your images without moving them, point `POSE3D_FILES` at a
wider folder before starting:

```bash
POSE3D_FILES=$HOME docker compose up                      # Linux / macOS
```
```powershell
$env:POSE3D_FILES="C:\Users\You"; docker compose up      # Windows
```

That folder then appears as `/host` in the file picker, with a shortcut in its
sidebar.

### Where your files go

A `workspace` folder is created next to this file and is shared with the
application:

```
workspace/
  <your image folders>      put your camera images here
  pose3d_projects/          projects the app creates
```

Inside the app, that folder is `/workspace`. Anything you export there appears
on your own machine immediately.

---

## Using it

1. **Capture** — two cameras, fixed and synchronised. Name the pairs so they
   match up, e.g. `left_0001.jpg` / `right_0001.jpg`.
2. **Calibrate** — either upload the camera intrinsics/extrinsics, or include
   four ArUco markers (or a checkerboard) in view and the app will solve the
   camera positions itself. If neither is available it will tell you rather
   than produce silent nonsense.
3. **Import** — *Import images* in the app, point it at your folder. It detects
   the 2D joints in both views and triangulates them into 3D.
4. **Review and correct** — the timeline shows every frame. Drag any joint in
   either camera view; the 3D pose updates immediately.
5. **Export** — writes to the folder you choose:
   - `<name>.mp4` — video of the animated character
   - `<name>.fbx` — the character with a rigged, re-poseable armature
   - `<name>.bvh` — motion capture data

---

## What the system can and cannot do

**It reconstructs the 3D pose** — the skeleton, in metric 3D — from two
calibrated views, and drives a rigged character with it.

Two cameras impose real limits, which are worth knowing up front:

- A joint hidden in **both** views at once cannot be triangulated directly.
  This is what the manual correction step is for.
- Rotation *about* a limb's own axis (forearm twist, palm facing) is not
  recoverable from joint positions alone.
- Depth accuracy depends heavily on calibration quality. The cameras must be
  properly calibrated and must not move between shots.
- The skeleton is 15 joints — no fingers or facial detail.

It does **not** reconstruct the subject's actual body surface, face or
clothing. That is photogrammetry and needs many overlapping viewpoints
(typically dozens of cameras), not two. The character you see is a rigged model
driven by your capture, automatically scaled to the subject's proportions.

---

## Troubleshooting

### Windows `.exe`

**"Windows protected your PC".** SmartScreen, shown for any unsigned
application. **More info** → **Run anyway**. Once only.

**It closes immediately, or nothing happens.** Read `pose3d-log.txt` next to
`Pose3D.exe`. A windowed application has no console, so that file is where
crashes and library errors go.

**It worked, then stopped launching.** Anti-virus software sometimes
quarantines a file out of `_internal\` because the build is unsigned. Check the
quarantine list.

**"Failed to load Python DLL ... `_internal\python312.dll`. LoadLibrary: The
specified module could not be found."** A file is missing from `_internal\`.
Extract the `.zip` again to a short path on `C:` — `C:\Pose3D` — rather than
Downloads, Documents or anywhere OneDrive syncs, where "files on-demand" can
leave placeholders instead of real files, and check the anti-virus quarantine
for anything from `_internal\`. Only if it still fails, install the
[Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

**Export fails, or the app says a folder is read-only.** Export somewhere you
own — `workspace\` inside the extracted folder is the default and always works.
Program Files and most network drives do not.

**`Pose3D.exe --selftest`** runs the same checks the release build is verified
with — Blender, the rig, the pose models, OpenGL, and a real export — and
prints which one is wrong. Worth running before reporting a problem.

### Docker

**The page doesn't load.** Give it a few seconds after `docker compose up` —
the display server starts first. Check the terminal says
`serving UI on http://localhost:8080`.

**Port 8080 is in use.** Start it on another port:

```bash
POSE3D_PORT=8090 docker compose up      # then open http://localhost:8090
```

On Windows PowerShell: `$env:POSE3D_PORT=8090; docker compose up`

**"Read-only file system" when exporting.** You picked a folder under `/host`,
which is shared for reading your images only. Export to `/workspace` — that is
the `workspace` folder next to `docker-compose.yml`, so the files appear on your
machine straight away. The export dialog now opens there by default.

**The 3D view is slow.** The container renders in software, with no GPU. It is
fine for review; large captures are smoother in the exported video.

**The file picker doesn't show my files.** A container can only see folders
shared with it. By default that is the folder you launched from. Start it with
`POSE3D_FILES=$HOME docker compose up` (PowerShell:
`$env:POSE3D_FILES="C:\Users\You"`) to browse a wider folder — see
"Which folders the app can see" above.

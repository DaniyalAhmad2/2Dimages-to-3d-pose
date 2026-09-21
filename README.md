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
| Download | about 720 MB zip | ~1.5 GB image |

The 3D view is where most of the work happens — scrubbing frames, checking
joints — so on Windows the native build is the one to use. Docker remains the
answer for macOS and Linux, and for reproducing a problem in a known-identical
environment.

### Windows (native)

Download `Pose3D-Windows.zip` from the
[Releases page](https://github.com/DaniyalAhmad2/2Dimages-to-3d-pose/releases)
— the newest release — and then, in this order:

1. **Unblock the zip.** Right-click the downloaded file ▸ **Properties** ▸ tick
   **Unblock** ▸ **OK**. Windows tags anything downloaded, the tag survives
   extraction, and a tagged `_internal\` is one of the ways the app refuses to
   start.
2. **Extract the whole zip to a short path on `C:`** — `C:\Pose3D` is the one
   we test. Do **not** extract to Desktop, Documents or any other folder
   OneDrive syncs ("files on-demand" leaves placeholder stubs where the real
   files should be), not to Program Files (the app saves next to itself, and
   Windows blocks writing there), and never run it from inside the zip. Let
   the extraction finish completely.
3. **Run `Pose3D.exe`.**

These are the same steps, in the same order, as the `README.txt` inside the
zip.

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
2. **Calibrate** — calibration files are optional. Include ArUco markers in
   view and the app solves the camera positions itself; the one number you
   must set is the marker size, typed in centimetres (the black square, edge
   to edge — an 8 cm tag is typed as 8). If no calibration can be worked out
   it says so rather than producing silent nonsense.
3. **Import** — *Import images* in the app, point it at your folder. It detects
   the 2D joints in both views and triangulates them into 3D.
4. **Review and correct** — the timeline shows every frame. Drag any joint in
   either camera view; the 3D pose updates immediately.
5. **Export** — writes to the folder you choose:
   - `<name>.mp4` — video of the animated character
   - `<name>.fbx` — the character with a rigged, re-poseable armature
   - `<name>.bvh` — motion capture data (Y-up, the mocap convention: imports
     upright in Blender, Unity and Unreal with their default settings)

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
- The skeleton is 17 joints — the 15 body joints plus one big-toe point per
  foot; no fingers or facial detail.

It does **not** reconstruct the subject's actual body surface, face or
clothing. That is photogrammetry and needs many overlapping viewpoints
(typically dozens of cameras), not two. The character you see is a rigged model
driven by your capture, automatically scaled to the subject's proportions.

---

## Troubleshooting

### Windows `.exe`

**Start here: double-click `Diagnose.cmd`.** It runs
`Pose3D-diagnose.exe --diagnose` — the console build of the same checks the
release gate verifies every bundle with (the libraries, Blender, the character
rig, the pose models, OpenGL) — and then prints the log, in a window that waits
at the end instead of hiding the answer in a windowed process with no console.
It writes `pose3d-diagnostics.txt`; **send that file**. It names which files
the bundle found, where it looked, and what the machine reported, which usually
answers the question in one round trip. Double-clicking `Pose3D-diagnose.exe`
runs the same diagnosis on its own — with no arguments it diagnoses rather than
starting the app — and holds its window open at the end.

`pose3d-diagnostics.txt` and `pose3d-log.txt` are written next to `Pose3D.exe`
when that folder is writable, and in `%LOCALAPPDATA%\Pose3D\` when it is not
(any install under Program Files). The report names the path it used at the
top. If the app has run at all, send the log with it.

**"Windows protected your PC".** SmartScreen, shown for any unsigned
application. **More info** → **Run anyway**. Once only.

**It closes immediately, or nothing happens.** `Diagnose.cmd` above — it runs
a console application, so it can report a failure that kills the windowed one
before it opens a window. `pose3d-log.txt` is the other half: a windowed
application has no console, so crashes and library errors go there (next to
`Pose3D.exe`, or in `%LOCALAPPDATA%\Pose3D\` when that folder is read-only).

**It worked, then stopped launching.** Anti-virus software sometimes
quarantines a file out of `_internal\` because the build is unsigned. Check the
quarantine list; `Pose3D-diagnose.exe` names the missing file.

**"Failed to load Python DLL ... `_internal\python312.dll`. LoadLibrary: The
specified module could not be found."** Despite the wording, this is what
Windows says when a file that `python312.dll` *depends on* is missing — and
every one of those now ships inside `_internal\`: the Visual C++ runtime and,
since build 17, the Universal C Runtime (`ucrtbase.dll` + `api-ms-win-crt-*`),
which a client's Windows turned out not to have in working order. So it means
a file is missing from the extracted folder. **Double-click `Diagnose.cmd`**:
it checks those files in plain cmd, before any Python runs, and names the one
that is missing or empty. Usual causes: the extraction stopped early, the
anti-virus quarantined something from `_internal\`, or the folder is under
OneDrive ("files on-demand" placeholders). Extract again to `C:\Pose3D`; if
the zip was downloaded, right-click it → Properties → **Unblock** first. Only
if a *complete* folder still fails, install the
[Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

**Export fails, or the app says a folder is read-only.** Export somewhere you
own — `workspace\` inside the extracted folder is the default and always works.
Program Files and most network drives do not.

**What CI already proved about the bundle you have** is in
[`docs/windows-release-gate.md`](docs/windows-release-gate.md): the zip is
extracted with 7-Zip into a nested path with a space in it and self-tested from
there, with the bundled Blender and weights as the only ones available.

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

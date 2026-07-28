# Pose3D

Reconstructs a metric 3D pose from two synchronised camera views, lets you
correct any joint by hand, and exports the result as a rigged character
(FBX / BVH) and a video.

---

## Running it

You need [Docker](https://www.docker.com/products/docker-desktop/) installed.
Nothing else — Blender and the pose model are already inside the image.

```bash
docker compose up
```

Then open **<http://localhost:8080>** in your browser. The application appears
there; it behaves exactly like a desktop window.

To stop it, press `Ctrl+C` in the terminal (or `docker compose down`).

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

**The page doesn't load.** Give it a few seconds after `docker compose up` —
the display server starts first. Check the terminal says
`serving UI on http://localhost:8080`.

**Port 8080 is in use.** Start it on another port:

```bash
POSE3D_PORT=8090 docker compose up      # then open http://localhost:8090
```

On Windows PowerShell: `$env:POSE3D_PORT=8090; docker compose up`

**The 3D view is slow.** The container renders in software, with no GPU. It is
fine for review; large captures are smoother in the exported video.

**Images don't appear after import.** They must be inside the `workspace`
folder — the application can only see that folder, not the rest of your disk.

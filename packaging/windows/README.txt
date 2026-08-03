Pose3D — 3D pose reconstruction from two camera views
=====================================================

Everything needed is in this folder. There is nothing to install, and the app
does not need an internet connection.


Running it
----------

1. Extract this whole folder somewhere you can write to — your Desktop or
   Documents is ideal.

   Do NOT run it from inside the .zip, and do not put it in "Program Files":
   the app saves projects and exports next to itself, and Windows blocks
   writing there.

2. Double-click Pose3D.exe.

3. The first time, Windows will show a blue "Windows protected your PC"
   screen. This is SmartScreen, and it appears for any application that has
   not been code-signed — it is not a virus warning.

   Click "More info", then "Run anyway".

   You should only see this once.


What is in this folder
----------------------

  Pose3D.exe        the application
  _internal\        its libraries — do not move or delete
  blender\          Blender 5.1, used to write the BVH/FBX/MP4 exports
  models\           the pose detection models
  workspace\        where your projects and exports go by default
  README.txt        this file


Using it
--------

1. Click "Import Images" and choose your left and right camera images.
2. Pick the calibration for that camera setup.
3. Step through the frames. Where the detected skeleton is wrong, drag the
   joint in either 2D view — the 3D pose updates as you drag.
4. Click "Export" and choose a folder. You get:
     .bvh   joint rotations, for Blender / Maya / MotionBuilder
     .fbx   the same motion on a character rig
     .mp4   a preview video

The 3D view has a fullscreen button; in fullscreen you also get per-joint
accuracy on the right and a frame timeline along the bottom.


If something goes wrong
-----------------------

If the app closes unexpectedly or an export fails, a file called
pose3d-log.txt is written next to Pose3D.exe. Send that file over and it will
usually say exactly what happened.

Two things worth checking first:

* Exporting to a folder you cannot write to (a network drive, a locked
  folder) will be refused up front, with a message saying so. Export into
  workspace\ if in doubt.

* Anti-virus software occasionally quarantines files out of _internal\
  because the app is unsigned. If the app stops launching after having
  worked, that is the usual cause.

Pose3D — 3D pose reconstruction from two camera views
=====================================================

Everything needed is in this folder. There is nothing to install, and the app
does not need an internet connection.


Running it
----------

1. Extract this whole folder somewhere you can write to. A short path
   straight on your C: drive — C:\Pose3D — is the safest choice.

   Do NOT run it from inside the .zip, and do not put it in "Program Files":
   the app saves projects and exports next to itself, and Windows blocks
   writing there. Avoid folders OneDrive syncs (often Desktop and Documents):
   it can leave placeholder stubs instead of real files.

2. Double-click Pose3D.exe.

3. The first time, Windows will show a blue "Windows protected your PC"
   screen. This is SmartScreen, and it appears for any application that has
   not been code-signed — it is not a virus warning.

   Click "More info", then "Run anyway".

   You should only see this once.


What is in this folder
----------------------

  Pose3D.exe            the application
  Pose3D-diagnose.exe   run this if something goes wrong (see below)
  _internal\            its libraries — do not move or delete
  blender\              Blender, used to write the BVH/FBX/MP4 exports
  models\               the pose detection models
  workspace\            where your projects and exports go by default
  README.txt            this file


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

Double-click Pose3D-diagnose.exe.

It runs the same checks this build was verified with before it shipped — the
libraries, Blender, the character rig, the pose models, OpenGL — and says
which one is wrong. It writes everything it found to

  pose3d-diagnostics.txt

next to the .exe. Send that one file over: it names which files the bundle
found, where it looked and what your machine reported, which is almost always
enough to answer the question without a round of "what happens if you...".

(It prints into a console window, which Windows may close again as soon as it
finishes. That is fine — the file is what matters.)

If the app got as far as opening, one more file is worth sending with it:

  pose3d-log.txt         written next to Pose3D.exe whenever the app runs.
                         Crashes and library errors go here, because a
                         windowed application has no console to print to.

Three things worth checking first:

* Exporting to a folder you cannot write to (a network drive, a locked
  folder) will be refused up front, with a message saying so. Export into
  workspace\ if in doubt.

* Anti-virus software occasionally quarantines files out of _internal\
  because the app is unsigned. If the app stops launching after having
  worked, that is the usual cause.

* "Failed to load Python DLL ... _internal\python312.dll. LoadLibrary: The
  specified module could not be found." means a file is missing from
  _internal\, not that anything is wrong with your Windows.

  Extract the .zip again to a short path on your C: drive (C:\Pose3D), not
  Downloads, Documents or anywhere under OneDrive, and check your anti-virus
  quarantine for files from _internal\.

  Only if that does not fix it, install the Microsoft Visual C++
  Redistributable (x64) from Microsoft:
  https://aka.ms/vs/17/release/vc_redist.x64.exe

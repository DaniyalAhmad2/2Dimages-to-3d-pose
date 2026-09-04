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
  Diagnose.cmd          run this if something goes wrong (see below)
  Pose3D-diagnose.exe   what Diagnose.cmd runs; it works on its own too
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

Double-click Diagnose.cmd.

It runs Pose3D-diagnose.exe --diagnose and then prints the log, in a window
that waits for you at the end. Between them they cover the same checks this
build was verified with before it shipped — the libraries, Blender, the
character rig, the pose models, OpenGL — and say which one is wrong.

Everything found is written to

  pose3d-diagnostics.txt

Send that one file over: it names which files the bundle found, where it
looked and what your machine reported, which is almost always enough to
answer the question without a round of "what happens if you...".

Double-clicking Pose3D-diagnose.exe does the same diagnosis on its own (it
needs no arguments) and holds its console window open at the end; Diagnose.cmd
is the one to reach for because it also prints the log, and it still says
something useful when the .exe itself will not run.

If the app got as far as opening, one more file is worth sending with it:

  pose3d-log.txt         crashes and library errors go here, because a
                         windowed application has no console to print to.

Both files are written next to Pose3D.exe when that folder can be written to.
When it cannot — an install under "Program Files" is the usual case — they go
to

  %LOCALAPPDATA%\Pose3D\

instead. Paste that into the File Explorer address bar to open it; the
diagnostics report names the exact path it used, at the top.

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

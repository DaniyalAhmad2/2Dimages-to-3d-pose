Pose3D — 3D pose reconstruction from two camera views
=====================================================

Everything needed is in this folder. There is nothing to install, and the app
does not need an internet connection.


Running it
----------

1. Unblock the .zip before extracting it. Right-click the file you
   downloaded > Properties > tick Unblock > OK. Windows tags every downloaded
   file, the tag survives extraction, and a tagged _internal\ folder is one of
   the ways this app will not start.

2. Extract this whole folder to a short path straight on your C: drive —
   C:\Pose3D is the one we test. Let it finish completely.

   Do NOT run it from inside the .zip. Avoid Desktop, Documents and anything
   else OneDrive syncs: "files on-demand" can leave placeholder stubs instead
   of real files. Do not put it in "Program Files" either — the app saves
   projects and exports next to itself, and Windows blocks writing there.

3. Double-click Pose3D.exe.

4. The first time, Windows will show a blue "Windows protected your PC"
   screen. This is SmartScreen, and it appears for any application that has
   not been code-signed — it is not a virus warning.

   Click "More info", then "Run anyway".

   You should only see this once.

These are the same steps, in the same order, as the "Windows (native)"
section of the project README.


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

2. Type the ArUco marker size in centimetres: the black square on the printed
   tag, measured edge to edge with a ruler — for example 8. That one number
   sets the real-world scale of everything.

   The three calibration file boxes above it are optional, and you almost
   certainly have none: leave them empty and the app works out where the
   cameras are from the ArUco tags in your photographs.

3. Give the project a name and click "Process".

4. Step through the frames. Where the detected skeleton is wrong, drag the
   joint in either 2D view — the 3D pose updates as you drag.

5. Click "Export" and choose a folder. You get:
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
  specified module could not be found." Despite the wording, Windows says this
  when a file python312.dll DEPENDS ON is missing - and every one of those is
  inside _internal\ (the Visual C++ runtime and the Universal C Runtime), so
  it means a file is missing from your extracted folder, not that anything is
  wrong with your Windows.

  Double-click Diagnose.cmd: it checks those files before any Python runs and
  names the one that is missing or empty.

  Usual causes: the extraction stopped early, your anti-virus removed a file
  (Windows Security > Protection history, and restore it), or the folder is
  under OneDrive. Extract the .zip again to C:\Pose3D. If the .zip came from
  a download, right-click it > Properties > tick Unblock > OK, first.

  Only if a complete folder still fails, install the Microsoft Visual C++
  Redistributable (x64) from Microsoft:
  https://aka.ms/vs/17/release/vc_redist.x64.exe

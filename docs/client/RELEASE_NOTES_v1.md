# Pose3D — what changed in build-19

Everything you reported is answered below: what you said, what changed, and the
quickest way to check it for yourself. Nothing here needs a technical step.

## Installing it (do these three in this order)

1. Right-click `Pose3D-Windows.zip` → **Properties** → tick **Unblock** → **OK**.
2. Extract the whole zip to `C:\Pose3D`. Not Desktop, not Documents, not
   OneDrive, not Program Files. Let it finish completely.
3. Double-click `Pose3D.exe`. The blue "Windows protected your PC" screen is
   Windows asking about an unsigned app: **More info** → **Run anyway**, once.

If it still will not open, double-click `Diagnose.cmd` in that folder and send
me `pose3d-diagnostics.txt`. It names the exact file that is missing.

## Getting it to run

- **"It will not open — python312.dll could not be found."** The app now
  carries its own copy of the Microsoft runtime that was missing from your PC,
  so it no longer depends on what Windows has. *Check: it opens.*
- **"Path too long" while extracting.** The zip is now built so its internal
  paths stay inside the Windows limit, and extracting to `C:\Pose3D` keeps
  them short. *Check: the extraction finishes without a warning.*
- **Instructions that disagreed with each other.** The page you download from
  and the README inside the zip now carry the same three steps, in the same
  order, with the Unblock step first. *Check: they match.*

## The 3D pose itself

- **"Right hand raised instead of left."** Left and right are no longer
  swapped anywhere between the photographs, the 3D view and the export.
  *Check: raise the figure's right arm and watch the character's right arm.*
- **"Leaning further forward than my character."** The forward lean is gone;
  the character stands as the figure in your photographs stands. *Check: a
  side-on pose, compared with the photograph beside it.*
- **"Clipping below zero on the Z axis."** The character stands on the grid
  whenever both feet are down instead of sinking through it, and one badly
  detected ankle can no longer push the whole take underground. When the
  figure jumps or kicks, the character leaves the floor as the figure does.
  *Check: any standing pose, viewed from the side.*
- **"The neck and head are bent down in the image but upright in the preview."**
  The head and neck now follow the photographs. *Check: the bent-forward pose
  you sent.*
- **"The 3D preview is very different to the real pose."** Each frame is now
  built from its own photographs only — the old build blended every frame with
  the one before it, which dragged every pose toward its neighbour. *Check:
  step through the frames; each one matches its own pair of photographs.*
- **"The character is distorted."** The character keeps its own proportions
  and no longer stretches to reach the detected points. *Check: the preview
  and the exported video.*

## Working with the frames

- **"You have to be on the exact pixel of the joint to move it."** The joint
  handles are larger and easier to see and grab. *Check: drag a knee.*
- **"The left and right keys should move you between frames."** The arrow keys
  step through the frames wherever you last clicked — either camera view, the
  3D view, the film strip, even a number box — and Home/End jump to the first
  and last frame. *Check: press left and right.*
- **"The 78/73% is not fully visible."** The accuracy dial's two numbers (left
  camera / right camera) now fit inside the ring at any size. *Check: the dial
  reads "NN / NN%" with nothing cut off.*
- **Joints the detector could not find.** A missing joint now appears as a
  red dashed handle you can drag into place instead of simply not being there,
  and the 3D view colours every joint the same way the photographs do (green
  seen, amber unsure, red missing, purple corrected). *Check: a frame where an
  arm is hidden.*
- **"Does the program not detect toe position?"** It does now: one point per
  foot, the big toe, so the character's foot points where the figure's toes
  point. A toe the detector could not see is a red dashed handle under the
  ankle that you can drag into place. Projects made before this build show
  those handles until you press Run Detection once. *Check: a side-on frame —
  the foot follows the toe.*
- **Corrections that vanished.** Your corrections are saved with the project
  and are still there when you reopen it, including after the 3D is
  recalculated. *Check: correct a joint, close the app, open the project
  again.*

## Your project and what comes out of it

- **"The video and BVH are not from the images of the character kicking."**
  The export is written from the same poses you are looking at in the app, so
  the BVH, the FBX and the MP4 show the take you processed. *Check: export
  your kick take and open the BVH in Blender.*
- **"When importing into Blender it imports at a 90 degree angle."** The BVH
  is now written the way Blender (and Unity, Unreal) expect, so it imports
  standing upright with the default settings. *Check: File → Import → Motion
  Capture (.bvh), change nothing.*
- **Saving and reopening.** A project saves to a folder and reopens from
  **File → Open Project**, with your corrections and your settings as you left
  them. *Check: reopen yesterday's project.*
- **The marker size is now typed in centimetres.** The box is labelled
  **ArUco marker size (cm)**: measure the black square edge to edge with a
  ruler and type that number — an 8 cm tag is typed as 8. It used to be metres,
  so a typed 8 was silently refused and the scale came out wrong. *Check: the
  camera separation and figure height shown in the sidebar against a ruler.*

## Still to come

- **Export at true real-world size.** The exported FBX and BVH are sized to
  the character rig, not to your figure in centimetres. Changing that changes
  the scale everything downstream in Blender is set up for, so it waits for
  your go-ahead — and for one photograph of a ruler beside a printed tag and
  beside the figure, which settles how big the figure really is.
- **A second window when you import into an open project.** Harmless, and
  being tidied up with the rest of the window behaviour in the next pass.

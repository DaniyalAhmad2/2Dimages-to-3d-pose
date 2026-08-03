# PyInstaller spec for the Pose3D desktop app.
#
#   pyinstaller pose3d.spec --noconfirm --clean
#
# Produces a onedir build: Pose3D(.exe) plus an _internal/ folder. Onedir, not
# onefile, on purpose — onefile unpacks ~500 MB to a temp directory on every
# launch, and the app already ships next to Blender and the ONNX weights, so
# there is no single-file illusion to preserve anyway.
#
# What ships beside the executable rather than inside it (see tools/
# make_windows_bundle.ps1): Blender in blender/, the pose weights in models/,
# and an empty workspace/. pose3d.runtime.app_dir() resolves all three relative
# to the executable.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

IS_WINDOWS = sys.platform.startswith("win")

datas = [
    ("pose3d/ui/dark.qss", "pose3d/ui"),
    # run by Blender in a separate process, so PyInstaller cannot see it as an
    # import — it has to be carried as data or export dies with "no such file"
    ("pose3d/export/blender_job.py", "pose3d/export"),
    # the rigged character drives BOTH the 3D view and the FBX/mp4 export; if
    # these are missing the app silently falls back to a stick figure
    ("pose3d/assets/character.blend", "pose3d/assets"),
    ("pose3d/assets/character.npz", "pose3d/assets"),
]

# Weights staged into the source tree by tools/fetch_weights.py get bundled;
# otherwise they are shipped beside the exe and found at run time. Either way
# the app must not have to download them. See pose3d/detect/models.py.
datas += collect_data_files("pose3d.detect", includes=["models/*.onnx"])
datas += collect_data_files("pose3d", includes=["assets/models/*.onnx"])

for src, _ in datas:
    if not Path(src).exists():
        raise SystemExit(
            f"pose3d.spec: {src} is missing.\n"
            "The build would succeed and the app would then fail at run time, "
            "so stop here instead.")

hiddenimports = [
    "OpenGL",
    "pyqtgraph.opengl",
    # pyqtgraph imports its GL items lazily by name, so nothing in the source
    # tells PyInstaller they are needed
    "pyqtgraph.opengl.items.GLScatterPlotItem",
    "pyqtgraph.opengl.items.GLLinePlotItem",
    "pyqtgraph.opengl.items.GLMeshItem",
    "pyqtgraph.opengl.items.GLGridItem",
]
if not IS_WINDOWS:
    # PyInstaller's bundled hook-OpenGL.py already collects OpenGL.platform.win32
    # and all of OpenGL.arrays on Windows; naming egl there would bundle a
    # platform module that cannot load.
    hiddenimports.append("OpenGL.platform.egl")

a = Analysis(
    ["pose3d/app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    # PyInstaller >= 6.5 refuses to bundle two Qt bindings at once, and pulling
    # in a test framework or matplotlib would add hundreds of MB for nothing.
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "matplotlib", "pytest",
              "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Pose3D" if IS_WINDOWS else "pose3d",
    # No console window: this is a desktop app, and a black terminal behind it
    # reads as "something went wrong". The cost is that sys.stdout/stderr are
    # None, which pose3d.runtime.ensure_std_streams() repairs at startup.
    console=False,
    icon="packaging/windows/pose3d.ico" if IS_WINDOWS else None,
    version="packaging/windows/version_info.txt" if IS_WINDOWS else None,
    # UPX mangles Qt and onnxruntime DLLs; the saving is not worth a bundle
    # that crashes on launch.
    upx=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    upx=False,
    name="Pose3D" if IS_WINDOWS else "pose3d",
)

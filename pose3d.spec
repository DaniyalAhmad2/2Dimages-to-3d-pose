# PyInstaller spec for the Pose3D desktop app (Phase 8).
# Build: uv run pyinstaller pose3d.spec --noconfirm --clean
#
# Notes (from research):
# - PySide6 'platforms' plugin must ship or the window is blank; the modern
#   PySide6 PyInstaller hooks usually handle this. --onedir first for debugging.
# - Bundle PyOpenGL for the pyqtgraph 3D view.
# - Exclude other Qt bindings (PyInstaller >=6.5 forbids mixed Qt bindings).
# - Blender is an EXTERNAL dependency (not bundled); the app calls the local
#   binary via subprocess. Ship README_RUN.md with the Blender path setting.
# - RTMPose ONNX models: either bundle under detect/models and use the
#   low-level RTMPose(onnx_model=...) path, or warm ~/.cache/rtmlib on first run.

from PyInstaller.utils.hooks import collect_data_files

datas = [
    ("pose3d/ui/dark.qss", "pose3d/ui"),
    ("pose3d/export/blender_job.py", "pose3d/export"),
    # the rigged character drives BOTH the 3D view and the FBX/mp4 export; if
    # these are missing the app silently falls back to a stick figure
    ("pose3d/assets/character.blend", "pose3d/assets"),
    ("pose3d/assets/character.npz", "pose3d/assets"),
]
# include any bundled ONNX models if present
datas += collect_data_files("pose3d.detect", includes=["models/*.onnx"])

a = Analysis(
    ["pose3d/app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=["OpenGL", "OpenGL.platform.egl", "pyqtgraph.opengl"],
    excludes=["PyQt5", "PyQt6", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="pose3d", console=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    name="pose3d",
)

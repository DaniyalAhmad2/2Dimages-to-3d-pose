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
import os
import sys
from pathlib import Path

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
    # the bundle's own inventory. pose3d/integrity.py reads it from
    # _internal/ before QApplication exists and tells the client which file
    # their extracted copy is missing; without it that check finds no manifest
    # and silently checks nothing.
    ("packaging/windows/manifest.json", "packaging/windows"),
]

# The ONNX weights are NOT here. They ship beside the executable, in models\,
# where pose3d.runtime.app_dir() looks for them and where a 150 MB copy is not
# also carried inside _internal\. Two collect_data_files globs used to try
# both: they pointed at package directories a clean checkout does not have, so
# they matched nothing, silently, in every build ever made. A glob that
# quietly collects nothing looks exactly like one that quietly collects
# 150 MB. tools/fetch_weights.py stages them beside the exe and verifies them
# against packaging/windows/inputs.json; pose3d.selftest fails the build if
# they did not arrive.

for src, _ in datas:
    if not Path(src).exists():
        raise SystemExit(
            f"pose3d.spec: {src} is missing.\n"
            "The build would succeed and the app would then fail at run time, "
            "so stop here instead.")

# --- the Visual C++ runtime ---------------------------------------------
# python312.dll, and most of Qt, onnxruntime and OpenCV, link against the MSVC
# runtime. It is NOT part of Windows: a machine that has never installed a
# Visual C++ application does not have it, and every DLL that needs it then
# fails to load with "The specified module could not be found" — which is
# exactly the dialog the client saw for _internal\python312.dll.
#
# PyInstaller's dependency scan happens to collect these already, so today's
# bundle has them. Nothing made that a requirement, though, and the failure
# mode is invisible on the build machine, where the runtime is installed
# system-wide. So name them, and stop the build rather than ship without them.
# Redistributing them beside the application is what Microsoft's Visual C++
# redistributable licence permits.
#
# concrt140.dll is deliberately absent: nothing in this bundle imports it
# today, and tools/check_bundle_deps.py fails the build if that ever changes.
VC_RUNTIME = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")


def _vc_runtime_dirs():
    """Where to look, best source first."""
    # python-build-standalone (what uv installs, and what CI builds with)
    # ships the runtime next to python312.dll, which is the copy that DLL was
    # actually linked against and the one PyInstaller bundles. sys.base_prefix
    # is where that lives; a uv venv's Scripts\ may or may not have a copy too.
    yield Path(sys.executable).parent
    yield Path(sys.base_prefix)
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(var)
        if root:
            # e.g. .../Microsoft Visual Studio/2022/Enterprise/VC/Redist/MSVC/
            #      14.44.35112/x64/Microsoft.VC143.CRT — newest last, so
            #      reverse to prefer it.
            yield from sorted(
                (Path(root) / "Microsoft Visual Studio").glob(
                    "*/*/VC/Redist/MSVC/*/x64/Microsoft.VC*.CRT"),
                reverse=True)
    yield Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"


def vc_runtime():
    """The runtime DLLs, as (source, destination) pairs for `binaries`."""
    found, missing = [], []
    for name in VC_RUNTIME:
        for folder in _vc_runtime_dirs():
            dll = folder / name
            if dll.is_file():
                found.append((str(dll), "."))
                break
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            "pose3d.spec: cannot find the Visual C++ runtime "
            f"({', '.join(missing)}).\n"
            "Looked beside the interpreter, in the Visual Studio redist "
            "folders and in System32. The bundle would launch here and fail "
            "on a clean Windows machine, so stop here instead.\n"
            "Install the Visual C++ redistributable, or build with a "
            "python-build-standalone interpreter (uv's default), which ships "
            "the runtime beside python.exe.")
    return found


hiddenimports = [
    "OpenGL",
    "pyqtgraph.opengl",
    # pyqtgraph imports its GL items lazily by name, so nothing in the source
    # tells PyInstaller they are needed
    "pyqtgraph.opengl.items.GLScatterPlotItem",
    "pyqtgraph.opengl.items.GLLinePlotItem",
    "pyqtgraph.opengl.items.GLMeshItem",
    "pyqtgraph.opengl.items.GLGridItem",
    # detection imports these inside functions. PyInstaller does follow
    # function-level imports, so naming them is insurance rather than the
    # gate — but what it insures against is "detection does nothing", found
    # after delivery. The gate is the self-test's inference check.
    "onnxruntime",
    "rtmlib",
]
if not IS_WINDOWS:
    # PyInstaller's bundled hook-OpenGL.py already collects OpenGL.platform.win32
    # and all of OpenGL.arrays on Windows; naming egl there would bundle a
    # platform module that cannot load.
    hiddenimports.append("OpenGL.platform.egl")

a = Analysis(
    ["pose3d/app.py"],
    pathex=["."],
    binaries=vc_runtime() if IS_WINDOWS else [],
    datas=datas,
    hiddenimports=hiddenimports,
    # PyInstaller >= 6.5 refuses to bundle two Qt bindings at once, and pulling
    # in a test framework or matplotlib would add hundreds of MB for nothing.
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "matplotlib", "pytest",
              "IPython", "notebook"],
    noarchive=False,
)

if IS_WINDOWS:
    # PyInstaller's PyOpenGL hook copies the whole OpenGL/DLLS folder: freeglut
    # and gle built for three MSVC generations, 32- and 64-bit. Only one pair
    # can ever be opened — OpenGL.platform.win32 hardcodes vc = 'vc14' and
    # picks by pointer size — and the vc9/vc10 builds import msvcr90.dll and
    # msvcr100.dll, runtimes we do not ship and Windows does not provide.
    # Shipping DLLs whose dependencies cannot be satisfied is what stops a
    # bundle being auditable (tools/check_bundle_deps.py), so drop the ones
    # PyOpenGL would never open. Nothing here imports OpenGL.GLUT or OpenGL.GLE
    # in the first place; keeping the vc14 pair means that stays a source
    # decision rather than a packaging one.
    #
    # Both lists have to be filtered. The hook files these under `datas`
    # (`if is_win: datas = collect_data_files('OpenGL')`), but Analysis
    # reclassifies every collected file by content before it returns
    # (build_main.py, "binary vs. data reclassification"), and on Windows
    # anything that opens as a PE is moved to `binaries` — so by the time this
    # runs, every freeglut/gle DLL is in `a.binaries` and filtering `a.datas`
    # alone would drop nothing at all.
    def keep(dest):
        dest = dest.replace("\\", "/").lower()
        return "opengl/dlls/" not in dest or dest.endswith("64.vc14.dll")

    a.datas = [entry for entry in a.datas if keep(entry[0])]
    a.binaries = [entry for entry in a.binaries if keep(entry[0])]

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

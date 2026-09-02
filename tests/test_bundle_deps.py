"""The build-time audit that keeps the Windows bundle self-contained.

The client's bundle refused to launch with "Failed to load Python DLL ...
python312.dll. LoadLibrary: The specified module could not be found." — a
dependency of python312.dll that Windows could not find. The CI runner has the
Visual C++ runtime and Qt's dependencies installed system-wide, so it cannot
notice one missing from `_internal`; only an audit of the bundle against itself
can. These tests drive that audit's resolver with synthetic data, which is the
whole reason it is a pure function of (imports, bundled paths), and evaluate
pose3d.spec to check what a Windows build would actually put in the bundle.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_bundle_deps as deps  # noqa: E402


def test_a_bundled_dll_resolves():
    assert deps.unresolved([("_internal/python312.dll", "VCRUNTIME140.dll")],
                           ["python312.dll", "vcruntime140.dll"]) == []


def test_matching_ignores_case():
    """PyInstaller writes MSVCP140.dll in some places and msvcp140.dll in
    others; import tables spell it either way."""
    assert deps.unresolved([("_internal/python312.dll", "msvcp140.dll")],
                           ["_internal/MSVCP140.dll"]) == []


def test_a_package_may_load_a_dll_from_another_folder():
    """numpy, scipy and PySide6 all call os.add_dll_directory for their own,
    and Windows matches an already-loaded DLL by name whatever its path — this
    bundle does ten such loads correctly."""
    assert deps.unresolved(
        [("_internal/PySide6/plugins/imageformats/qgif.dll", "Qt6Gui.dll")],
        ["_internal/PySide6/Qt6Gui.dll"]) == []


def test_what_loads_first_cannot_reach_into_a_subfolder():
    """python312.dll is loaded by the bootloader with only _internal on the
    search path, before any package has run and before anything is loaded that
    Windows could match by name. PySide6 ships its own VCRUNTIME140.dll, and
    that copy is no use here — which is why a bundle-wide name match would
    have missed exactly the fault this script exists for."""
    found = deps.unresolved([("_internal/python312.dll", "VCRUNTIME140.dll")],
                            ["_internal/PySide6/VCRUNTIME140.dll"])
    assert found == [("vcruntime140.dll", ["_internal/python312.dll"])]
    assert deps.unresolved([("_internal/python312.dll", "VCRUNTIME140.dll")],
                           ["_internal/vcruntime140.dll"]) == []


def test_windows_own_dlls_are_not_expected_in_the_bundle():
    """Redistributing kernel32 is neither possible nor desirable."""
    imports = [("_internal/python312.dll", d)
               for d in ("KERNEL32.dll", "ADVAPI32.dll", "WS2_32.dll",
                         "VERSION.dll", "bcrypt.dll", "ucrtbase.dll")]
    assert deps.unresolved(imports, ["python312.dll"]) == []


def test_api_sets_are_allowed_without_a_file():
    """api-ms-win-* and ext-ms-* are virtual names the loader resolves through
    the OS schema — python312.dll imports api-ms-win-core-path-l1-1-0.dll and
    no such file exists on any Windows machine."""
    imports = [("_internal/python312.dll", "api-ms-win-core-path-l1-1-0.dll"),
               ("_internal/PySide6/Qt6Gui.dll", "ext-ms-win-ntuser-window-l1-1-0.dll")]
    assert deps.unresolved(imports, ["python312.dll"]) == []


def test_an_unbundled_dll_is_reported_with_who_wanted_it():
    found = deps.unresolved(
        [("_internal/python312.dll", "VCRUNTIME140.dll"),
         ("_internal/PySide6/Qt6Core.dll", "vcruntime140.dll")],
        ["_internal/python312.dll", "_internal/PySide6/Qt6Core.dll"])
    assert found == [("vcruntime140.dll",
                      ["_internal/PySide6/Qt6Core.dll",
                       "_internal/python312.dll"])]


def test_findings_are_sorted_and_deduplicated():
    found = deps.unresolved(
        [("a.pyd", "zlib1.dll"), ("b.pyd", "zlib1.dll"), ("b.pyd", "mkl.dll")],
        [])
    assert [dll for dll, _ in found] == ["mkl.dll", "zlib1.dll"]
    assert found[1][1] == ["a.pyd", "b.pyd"]


def test_the_visual_cpp_runtime_is_never_treated_as_system_provided():
    """It is exactly what the client machine turned out not to have. Letting
    the allowlist cover it would make the audit pass on an empty bundle."""
    for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
                "concrt140.dll", "msvcr90.dll", "msvcr100.dll"):
        assert not deps.is_system_dll(dll)


def test_a_bundle_without_internal_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="_internal"):
        deps.app_files(tmp_path)


def test_only_the_app_is_examined_not_blender(tmp_path):
    """blender/ ships its own C runtime in blender.crt, is not ours to audit,
    and is not on the app's DLL search path either; the exe and _internal are
    both."""
    (tmp_path / "_internal" / "PySide6").mkdir(parents=True)
    (tmp_path / "blender").mkdir()
    for rel in ("Pose3D.exe", "README.txt", "_internal/python312.dll",
                "_internal/PySide6/QtGui.pyd", "_internal/base_library.zip",
                "blender/blender.exe"):
        (tmp_path / rel).write_bytes(b"")
    found = {p.relative_to(tmp_path).as_posix()
             for p in deps.app_files(tmp_path)}
    assert found == {"Pose3D.exe", "README.txt", "_internal/python312.dll",
                     "_internal/PySide6/QtGui.pyd",
                     "_internal/base_library.zip"}


@pytest.mark.skipif(importlib.util.find_spec("pefile") is None,
                    reason="pefile ships with PyInstaller on Windows only")
def test_a_binary_that_cannot_be_parsed_is_an_error_not_an_empty_list():
    """A truncated or quarantined DLL that read as "imports nothing" would sail
    through the audit, which is the one outcome this script must not produce."""
    import pefile

    with pytest.raises(pefile.PEFormatError):
        deps.read_imports(ROOT / "pyproject.toml")


# --- the audit has to actually run in the two places that build the bundle ---

def test_the_bundle_script_runs_the_audit():
    text = (ROOT / "tools" / "make_windows_bundle.ps1").read_text()
    assert "tools/check_bundle_deps.py" in text


def test_the_release_workflow_runs_the_audit_before_the_self_test():
    text = (ROOT / ".github" / "workflows" / "windows-release.yml").read_text()
    assert text.index("tools/check_bundle_deps.py") < text.index("--selftest")


# --- the spec's Windows-only additions, evaluated ---------------------------

VC_RUNTIME = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")


def _eval_spec(hook_datas=()):
    """Run pose3d.spec with PyInstaller's build classes stubbed out.

    Everything the spec decides — which binaries to bundle, which of
    PyInstaller's own findings to drop — happens at module level, so evaluating
    it is the only way to check the decision rather than the source text.

    `hook_datas` stands in for what PyInstaller's hooks add to Analysis's data
    list during a real build; the spec filters that list afterwards.
    """
    import types

    def analysis(scripts, **kw):
        return types.SimpleNamespace(scripts=scripts, pure=[],
                                     datas=list(kw["datas"]) + list(hook_datas),
                                     binaries=list(kw["binaries"]))

    def stub(*args, **kw):
        return types.SimpleNamespace()

    ns = {"__name__": "pose3d_spec", "Analysis": analysis,
          "PYZ": stub, "EXE": stub, "COLLECT": stub}
    spec = ROOT / "pose3d.spec"
    exec(compile(spec.read_text(), str(spec), "exec"), ns)
    return ns["a"]


def _as_windows(monkeypatch, interpreter_dir):
    # PyInstaller collects data files through a subprocess started from
    # sys.executable, which the fake interpreter directory below would break.
    # Which .onnx files get bundled is not what these tests are about.
    monkeypatch.setattr("PyInstaller.utils.hooks.collect_data_files",
                        lambda *a, **kw: [])
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(interpreter_dir / "python.exe"))
    monkeypatch.setattr(sys, "base_prefix", str(interpreter_dir / "base"))
    # nothing may be found by falling back to the build machine's own install
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setenv("SystemRoot", str(interpreter_dir / "no-such-windows"))


@pytest.mark.skipif(importlib.util.find_spec("PyInstaller") is None,
                    reason="PyInstaller is a dev dependency")
def test_the_linux_build_adds_nothing_windows_only():
    """The Linux dev build has to keep working: no VC runtime search, no
    OpenGL/DLLS filtering, nothing that assumes a Windows layout."""
    assert _eval_spec().binaries == []


@pytest.mark.skipif(importlib.util.find_spec("PyInstaller") is None,
                    reason="PyInstaller is a dev dependency")
def test_the_windows_build_bundles_the_runtime_beside_the_interpreter(
        monkeypatch, tmp_path):
    """python-build-standalone — what uv installs, and what CI builds with —
    ships the runtime next to python.exe, and that is the copy python312.dll
    was linked against."""
    _as_windows(monkeypatch, tmp_path)
    for name in VC_RUNTIME:
        (tmp_path / name).write_bytes(b"")
    bundled = _eval_spec().binaries
    assert {Path(src).name for src, _ in bundled} == set(VC_RUNTIME)
    assert {dest for _, dest in bundled} == {"."}, "must land in _internal/"


@pytest.mark.skipif(importlib.util.find_spec("PyInstaller") is None,
                    reason="PyInstaller is a dev dependency")
def test_the_windows_build_falls_back_to_the_base_interpreter(
        monkeypatch, tmp_path):
    """A uv venv's Scripts\\ need not carry a copy of the runtime. The base
    interpreter does, next to the python312.dll PyInstaller bundles."""
    _as_windows(monkeypatch, tmp_path)
    base = tmp_path / "base"
    base.mkdir()
    for name in VC_RUNTIME:
        (base / name).write_bytes(b"")
    assert {Path(src).parent for src, _ in _eval_spec().binaries} == {base}


@pytest.mark.skipif(importlib.util.find_spec("PyInstaller") is None,
                    reason="PyInstaller is a dev dependency")
def test_a_windows_build_without_the_runtime_stops(monkeypatch, tmp_path):
    """Shipping without it would build here and fail on the client's machine
    with the dialog this whole change exists to prevent."""
    _as_windows(monkeypatch, tmp_path)
    (tmp_path / "vcruntime140.dll").write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        _eval_spec()
    message = str(exc.value)
    assert "vcruntime140_1.dll" in message and "msvcp140.dll" in message
    assert "vcruntime140.dll," not in message, "the one it found is not missing"


@pytest.mark.skipif(importlib.util.find_spec("PyInstaller") is None,
                    reason="PyInstaller is a dev dependency")
def test_the_windows_build_drops_the_pyopengl_dlls_that_can_never_load(
        monkeypatch, tmp_path):
    """PyOpenGL ships freeglut and gle built for three MSVC generations, 32-
    and 64-bit. The vc9 and vc10 ones import msvcr90.dll / msvcr100.dll, which
    neither the bundle nor Windows provides — they were the only unsatisfiable
    imports in the shipped bundle — and OpenGL.platform.win32 would never open
    any of them but the 64-bit vc14 pair."""
    _as_windows(monkeypatch, tmp_path)
    for name in VC_RUNTIME:
        (tmp_path / name).write_bytes(b"")
    collected = ["freeglut32.vc9.dll", "freeglut64.vc9.dll",
                 "freeglut32.vc10.dll", "freeglut64.vc10.dll",
                 "freeglut32.vc14.dll", "freeglut64.vc14.dll",
                 "gle32.vc14.dll", "gle64.vc14.dll"]
    hook = [("OpenGL\\DLLS\\" + name, "/wherever/OpenGL/DLLS/" + name, "DATA")
            for name in collected]

    datas = _eval_spec(hook).datas
    kept = [entry[0] for entry in datas if "DLLS" in entry[0]]

    assert kept == ["OpenGL\\DLLS\\freeglut64.vc14.dll",
                    "OpenGL\\DLLS\\gle64.vc14.dll"]

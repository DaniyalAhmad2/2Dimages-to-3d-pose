"""The bundle must contain what the app loads at run time.

These files are read by path, not imported, so nothing else in the codebase
fails when one goes missing from the build — the app just silently degrades on
the client's machine. The spec's asset lines had in fact never been exercised
by a build at all.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "pose3d.spec"


def _spec_datas():
    """The ("src", "dest") pairs listed literally in the spec."""
    text = SPEC.read_text()
    body = text.split("datas = [", 1)[1].split("]", 1)[0]
    return re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', body)


def test_every_declared_data_file_exists():
    missing = [src for src, _ in _spec_datas() if not (ROOT / src).exists()]
    assert not missing, f"pose3d.spec lists files that do not exist: {missing}"


def test_the_things_the_app_loads_by_path_are_declared():
    """Each of these is read at run time by filename. If it is not in the spec
    the build still succeeds and the failure surfaces only on the client's
    machine — the character silently becomes a stick figure, the UI loses its
    stylesheet, or export dies with 'no such file'."""
    declared = {src for src, _ in _spec_datas()}
    for required in ("pose3d/assets/character.blend",
                     "pose3d/assets/character.npz",
                     "pose3d/ui/dark.qss",
                     "pose3d/export/blender_job.py"):
        assert required in declared, f"{required} is not bundled by pose3d.spec"


def test_windows_resources_referenced_by_the_spec_are_present():
    text = SPEC.read_text()
    for rel in re.findall(r'"(packaging/windows/[^"]+)"', text):
        assert (ROOT / rel).exists(), f"pose3d.spec references a missing {rel}"


def test_the_icon_is_a_real_multi_size_ico():
    """Windows picks a different size for the taskbar, the title bar and the
    Explorer list; a single-size .ico is resampled badly in most of them."""
    Image = pytest.importorskip("PIL.Image", reason="pillow not installed")
    ico = ROOT / "packaging" / "windows" / "pose3d.ico"
    with Image.open(ico) as im:
        sizes = {s[0] for s in im.info["sizes"]}
    assert {16, 32, 48, 256} <= sizes, f"icon only has {sorted(sizes)}"


def test_no_second_qt_binding_can_reach_the_bundle():
    """PyInstaller >= 6.5 refuses to bundle two Qt bindings, and opencv's
    non-headless wheel drags one in — which is why the dependency is pinned to
    the headless contrib build."""
    text = SPEC.read_text()
    excludes = text.split("excludes=[", 1)[1].split("]", 1)[0]
    for other in ("PyQt5", "PyQt6", "PySide2"):
        assert other in excludes

    pyproject = (ROOT / "pyproject.toml").read_text()
    assert "opencv-contrib-python-headless" in pyproject
    assert re.search(r'^\s*"opencv-contrib-python[>=]', pyproject, re.M) is None
    assert re.search(r'^\s*"PySide6[>=]', pyproject, re.M) is None

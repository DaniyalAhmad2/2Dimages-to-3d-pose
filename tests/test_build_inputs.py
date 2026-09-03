"""One source of truth for what the Windows build pulls in from outside.

Three external things decide whether the client's bundle works: the Python the
exe is frozen against, the Blender that is unzipped beside it, and the ONNX
checkpoints copied next to that. None of the three was pinned and verified in
one place. The interpreter was not written down at all: `uv.lock` says only
>=3.12, so the build took whatever the runner had newest, while every document
and every Windows test named python312.dll.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

# Task E owns .github/**, and lands after this task. Until it does, the two
# assertions below describe a contract nothing satisfies yet — so they are
# expected to fail, strictly: the moment E's change makes one pass, pytest
# reports XPASS as a failure and the marker has to be deleted with it.
NOT_YET_TASK_E = pytest.mark.xfail(
    strict=True,
    reason="Task E owns .github/**; when E lands this XPASSes — delete this "
           "marker then")


# --- A. the toolchain is pinned ---------------------------------------------

def test_the_python_version_is_pinned():
    """3.12, not 3.13: every doc, README string and Windows test names
    python312.dll, and it is where the PySide6 and onnxruntime wheel coverage
    is widest. `uv.lock` only says >=3.12, so without this file the build takes
    whatever the runner's newest interpreter happens to be."""
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_the_documented_dll_matches_the_pin():
    """The client's launch failure was reported by DLL name. If the pin and the
    troubleshooting text disagree, the instructions send them looking for a
    file the bundle does not contain."""
    for doc in (ROOT / "README.md", ROOT / "packaging" / "windows" / "README.txt"):
        assert "python312.dll" in doc.read_text(encoding="utf-8"), doc


@NOT_YET_TASK_E
def test_both_workflows_take_their_interpreter_from_the_pin():
    for wf in WORKFLOWS:
        assert ".python-version" in wf.read_text(encoding="utf-8"), wf

"""The release gate, read as a file: does CI prove what the client receives?

Every Windows fault this project has shipped was invisible to CI because CI
tested something adjacent to the delivered artifact. The self-test ran from
`build/Pose3D-Windows`, a folder PyInstaller had just written, on a path with
no spaces, on a machine that had the Visual C++ runtime installed anyway. The
client ran an extracted copy of a zip, from `C:\\Users\\<name>\\Downloads\\`,
on a machine that had none of it — and got "Failed to load Python DLL", then
"path too long".

So the gate now mirrors the client: build, assemble, verify, zip, EXTRACT the
zip with 7-Zip into a nested path with a space in it, verify THAT copy against
the manifest, and run the self-test out of it with Blender and the weights
cache deliberately unavailable. These tests read the workflow files and assert
the mirror is still a mirror — the assertions are about the text of the gate,
because there is no second Windows machine here to run it on.

Read as text rather than parsed: PyYAML is not a dependency of this project,
and `uv sync --frozen` on the Windows runner installs the lockfile and nothing
else, so a test that imported it would skip on the one machine that matters.
The helpers below slice the file on indentation, which is enough to ask "which
step is this" and "is there a `paths:` under `pull_request:`".
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ACTION = ROOT / ".github" / "actions" / "windows-bundle" / "action.yml"
BUNDLE_WF = ROOT / ".github" / "workflows" / "windows-bundle.yml"
RELEASE_WF = ROOT / ".github" / "workflows" / "windows-release.yml"
TEST_WF = ROOT / ".github" / "workflows" / "windows-test.yml"
MANIFEST = ROOT / "packaging" / "windows" / "manifest.json"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _runs(text: str) -> str:
    """The same file with its comment lines dropped.

    These files explain themselves at length, and half of what they explain is
    what they deliberately do NOT do. An assertion that a name is absent has to
    read what runs, or the prose saying why it is absent trips it.
    """
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def _block(text: str, header: str) -> str:
    """The lines under `header`, up to the next line indented no further."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.rstrip() != header:
            continue
        indent = len(line) - len(line.lstrip())
        out = []
        for following in lines[i + 1:]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            out.append(following)
        return "\n".join(out)
    raise AssertionError(f"{header!r} is not in this file")


def _steps(text: str) -> dict[str, str]:
    """Every `- name: …` step, mapped to the rest of its own block."""
    lines = text.splitlines()
    marker = next((len(raw) - len(raw.lstrip()) for raw in lines
                   if re.match(r"\s*- name:", raw)), None)
    assert marker is not None, "no named steps in this file"

    steps: dict[str, list[str]] = {}
    current = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("- ") and len(line) - len(stripped) == marker:
            found = re.fullmatch(r"- name:\s*(.+?)\s*", stripped)
            current = found.group(1) if found else None
            if current is not None:
                steps[current] = []
            continue
        if current is not None:
            steps[current].append(line)
    return {name: "\n".join(body) for name, body in steps.items()}


def _the_step_that(text: str, needle: str) -> str:
    """The one step whose body contains `needle` — named by what it does."""
    matches = {name: body for name, body in _steps(text).items()
               if needle in body}
    assert len(matches) == 1, f"{needle!r} is in {sorted(matches)}"
    return next(iter(matches.values()))


# --- the gate extracts the zip, the way the client does ---------------------

def test_the_gate_extracts_to_a_path_shaped_like_the_clients():
    """A space and a nesting level are not decoration. The bundle's deepest
    entry is ~148 characters and MAX_PATH is 260, so the extraction root is the
    budget; and a space is where an unquoted path in a script splits in two.
    The client's own root — Downloads under a user profile — has both."""
    step = _the_step_that(_text(ACTION), "7z x")
    root = re.search(r"Join-Path\s+'[^']*'\s+'([^']+)'", step)
    assert root, "the extraction root is not built with Join-Path"
    assert "Downloads" in root.group(1)
    assert " " in root.group(1), "the client's path has a space in it"


def test_the_zip_is_opened_with_7z_and_never_expand_archive():
    """Expand-Archive is .NET's ZipArchive: minutes on 1.6 GB, and its own
    MAX_PATH behaviour rather than the one the client's tool has. The archive
    was written with 7-Zip; it is opened with 7-Zip."""
    assert "7z x" in _text(ACTION)
    assert "Expand-Archive" not in _runs(_text(ACTION))


def test_the_self_test_runs_from_the_extracted_copy_not_the_build_tree():
    """`build/Pose3D-Windows` is what PyInstaller and the bundle script just
    wrote. It has never been through a zip, so it cannot show a file the
    archive dropped or a path the extraction refused."""
    extract = _the_step_that(_text(ACTION), "7z x")
    assert "CLIENT_BUNDLE=" in extract, "the extracted copy has no name"

    selftest = _runs(_the_step_that(_text(ACTION), "Start-Process"))
    assert "--selftest" in selftest
    assert "$env:CLIENT_BUNDLE" in selftest
    assert "build/Pose3D-Windows" not in selftest
    assert "build\\Pose3D-Windows" not in selftest


def test_the_layout_is_checked_on_the_extracted_copy_as_well():
    """Twice, on purpose: once on what was assembled, so a bundling mistake is
    named before five minutes of zipping, and once on what came back out, so a
    file the archive or the extraction lost is named too."""
    checks = [body for body in _steps(_text(ACTION)).values()
              if "check_bundle_layout.py" in body]
    assert len(checks) == 2, "the assembled bundle and the extracted copy"
    assert any("$env:CLIENT_BUNDLE" in body for body in checks)
    assert any("build/Pose3D-Windows" in body for body in checks)


def test_the_zip_is_gated_on_the_length_the_client_can_extract():
    """The first bundle would not extract at all: "path too long". 180 leaves
    ~80 characters for the extraction root."""
    assert "--limit 180" in _the_step_that(_text(ACTION), "bundle_zip.py")


# --- the one check the runner cannot make, and what stands in for it --------

def test_the_runners_missing_gpu_is_named_where_gl_is_switched_off():
    """POSE3D_NO_GL degrades the check this app's 3D view depends on, so it is
    the one env var in the gate that has to justify itself where it is set —
    otherwise it reads as a check somebody switched off to get a green run."""
    text = _text(ACTION)
    assert 'POSE3D_NO_GL: "1"' in text

    comment = []
    for line in reversed(text[:text.index("POSE3D_NO_GL:")].splitlines()):
        if line.strip().startswith("#"):
            comment.insert(0, line.strip().lstrip("#").strip())
        elif line.strip():
            break
    assert comment, "POSE3D_NO_GL is set with no comment above it"
    assert "GPU" in " ".join(comment), " ".join(comment)


def test_the_manifest_carries_the_gates_structural_gl_evidence():
    """With the GL check degraded, the only thing that still proves the 3D
    view was PACKAGED is that the files it needs are in the extracted copy."""
    rules = json.loads(MANIFEST.read_text(encoding="utf-8"))["rules"]
    required = {rule["pattern"] for rule in rules if rule["kind"] == "required"}
    for name in ("opengl32sw.dll", "qwindows.dll"):
        assert any(pattern.endswith(name) for pattern in required), name


# --- one gate, used by both workflows ---------------------------------------

def test_the_gate_and_the_release_build_through_the_same_action():
    """A release built by steps of its own is a release nothing gated: the two
    would drift, and the one that ships is the one nobody runs on a PR."""
    for workflow in (BUNDLE_WF, RELEASE_WF):
        assert "./.github/actions/windows-bundle" in _text(workflow), workflow


def test_the_release_workflow_still_ships_what_the_gate_built():
    text = _text(RELEASE_WF)
    assert "actions/upload-artifact@v4" in text
    assert "softprops/action-gh-release@v2" in text
    # the archive's path comes from the action that wrote it, not retyped here
    assert text.count("outputs.zip") >= 2, "the zip's path is written twice"


def test_the_bundle_workflow_gives_the_build_time_to_finish():
    """A ~1 GB bundle, a 414 MB download, PyInstaller and two hashing passes.
    The default 360 would not fail it, but a tighter one would kill it in the
    middle and read as a flaky gate."""
    minutes = [int(m) for m in
               re.findall(r"timeout-minutes:\s*(\d+)", _text(BUNDLE_WF))]
    assert minutes and min(minutes) >= 90


def test_the_windows_test_workflow_still_runs_on_every_pull_request():
    """The gate carries the `paths` filter because a filter cannot scope one
    job of a workflow — it decides whether the whole workflow runs. Adding one
    to the test workflow to spare it the bundle build would silently stop
    running the test suite on most pull requests."""
    triggers = _block(_text(TEST_WF), "on:")
    assert "pull_request:" in triggers
    assert "paths" not in triggers

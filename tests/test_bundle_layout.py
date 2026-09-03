"""What the Windows bundle must contain, and the check that says whether it does.

`tools/check_bundle_deps.py` already audits the DLL *imports* of what is there.
Nothing asked the other question — is it all there? — and the answers matter
one at a time: without `platforms\\qwindows.dll` the QApplication constructor
aborts before a single line of this application runs; without
`imageformats\\qjpeg.dll` every photo loads as a grey rectangle; without an
onnxruntime DLL detection raises inside a worker thread; without the Blender
tree export fails at the end of a long job. Each of those has its own dialog,
its own support thread, and its own wrong diagnosis.

So the bundle carries an inventory — `packaging/windows/manifest.json` — and
this is the tool that compares a folder against it. `pathlib.glob` only, no
Windows API and no PE parsing, so the check runs on the build machine, in CI on
Linux, and inside the frozen app on the client's machine, and so these tests
can build a bundle-shaped tree in `tmp_path` and take it apart one file at a
time.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_bundle_layout as layout  # noqa: E402

MANIFEST = ROOT / "packaging" / "windows" / "manifest.json"
INPUTS = ROOT / "packaging" / "windows" / "inputs.json"

#: Stand-in size for an .onnx checkpoint. The real ones are 20-101 MB; the
#: manifest's min_size only has to be small enough to pass for every real
#: checkpoint and large enough to reject an empty file or an HTML error page
#: saved under a .onnx name.
MODEL_BYTES = 1_000_000

#: A complete extracted Pose3D-Windows folder, in miniature: every path the
#: manifest names, plus two decoys — an OpenGL DLL that is allowed to ship and
#: a Blender datafiles folder that is not the locale tree — so the `forbidden`
#: rules have something to be wrong about.
CONTENTS = (
    "Pose3D.exe",
    "Pose3D-diagnose.exe",
    "README.txt",
    "_internal/python312.dll",
    "_internal/vcruntime140.dll",
    "_internal/vcruntime140_1.dll",
    "_internal/msvcp140.dll",
    "_internal/PySide6/opengl32sw.dll",
    "_internal/PySide6/plugins/platforms/qwindows.dll",
    "_internal/PySide6/plugins/imageformats/qjpeg.dll",
    "_internal/PySide6/plugins/imageformats/qico.dll",
    "_internal/PySide6/plugins/styles/qmodernwindowsstyle.dll",
    "_internal/onnxruntime/capi/onnxruntime_providers_shared.dll",
    "_internal/pose3d/assets/character.blend",
    "_internal/pose3d/assets/character.npz",
    "_internal/pose3d/ui/dark.qss",
    "_internal/pose3d/export/blender_job.py",
    "_internal/OpenGL/DLLS/freeglut64.vc14.dll",
    "blender/blender.exe",
    "blender/5.1/datafiles/fonts/DejaVuSans.woff2",
)


def model_names() -> list[str]:
    """The checkpoint filenames the release bundle ships, from inputs.json."""
    data = json.loads(INPUTS.read_text(encoding="utf-8"))
    return [w["name"] for w in data["weights"]["balanced"]]


def fake_bundle(root: Path) -> Path:
    """A complete bundle as far as the manifest can tell: every named path
    present, non-empty, and the models big enough to be plausible."""
    for rel in CONTENTS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(model_names()):
        (models / name).write_bytes(bytes([i]) * MODEL_BYTES)
    (root / "workspace").mkdir(parents=True, exist_ok=True)
    return root


def fake_inputs(bundle: Path, path: Path) -> Path:
    """An inputs.json for the miniature bundle: the release checkpoint names,
    with the checksums the stand-in files actually have."""
    entries = [{"name": p.name, "sha256": layout.sha256(p),
                "size": p.stat().st_size}
               for p in sorted((bundle / "models").glob("*.onnx"))]
    path.write_text(json.dumps({"weights": {"balanced": entries}}),
                    encoding="utf-8")
    return path


@pytest.fixture
def bundle(tmp_path):
    """A complete bundle and an inputs.json that agrees with it."""
    root = fake_bundle(tmp_path / "Pose3D-Windows")
    return root, fake_inputs(root, tmp_path / "inputs.json")


def rule_for(pattern: str):
    return next(r for r in layout.load(MANIFEST) if r.pattern == pattern)


# --- the manifest itself ----------------------------------------------------

def test_every_rule_says_what_it_is_and_why():
    """The `why` is not documentation: it is what the problem line prints, and
    what someone reading a failed build (or a client reading a dialog) has to
    act on."""
    rules = layout.load(MANIFEST)
    assert rules
    for rule in rules:
        assert rule.kind in ("required", "forbidden"), rule
        assert rule.stage in ("bootloader", "pre-qt", "later"), rule
        assert len(rule.why) > 20, f"{rule.pattern}: {rule.why!r}"
        # ASCII: the same text is printed by pose3d/integrity.py into a
        # console that is cp1252 in Europe and cp932 in Japan, where one em
        # dash raises instead of printing. See test_integrity.py.
        rule.why.encode("ascii")


def test_the_manifest_names_the_files_each_failure_was_traced_to():
    """Every entry here is a delivery that broke, or a check that found one.
    Deleting a rule is allowed; deleting it by accident is what this stops."""
    patterns = {r.pattern for r in layout.load(MANIFEST)}
    assert patterns >= {
        "Pose3D.exe",
        "_internal/python312.dll",
        "_internal/PySide6/plugins/platforms/qwindows.dll",
        "_internal/PySide6/plugins/imageformats/qjpeg.dll",
        "_internal/PySide6/opengl32sw.dll",
        "_internal/onnxruntime*/**/*.dll",
        "_internal/pose3d/assets/character.blend",
        "_internal/pose3d/export/blender_job.py",
        "models/*.onnx",
        "blender/blender.exe",
    }


def test_the_bootloader_rules_are_the_ones_nothing_in_process_can_catch():
    """python312.dll and the Visual C++ runtime are loaded before any Python
    runs, so a check written in Python cannot report them: they are staged
    `bootloader` and belong to the build and to Diagnose.cmd, never to a
    startup check that would already be dead."""
    bootloader = {r.pattern for r in layout.load(MANIFEST)
                  if r.stage == "bootloader"}
    assert "_internal/python312.dll" in bootloader
    assert {"_internal/vcruntime140.dll", "_internal/vcruntime140_1.dll",
            "_internal/msvcp140.dll"} <= bootloader


def test_a_rule_with_no_why_is_refused(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"rules": [
        {"pattern": "Pose3D.exe", "kind": "required", "stage": "later"}]}),
        encoding="utf-8")
    with pytest.raises(layout.Malformed, match="why"):
        layout.load(bad)


def test_a_rule_with_a_typo_in_a_key_is_refused(tmp_path):
    """`min_bytes` instead of `min_size` would otherwise be a rule that checks
    nothing and passes every build."""
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"rules": [
        {"pattern": "Pose3D.exe", "kind": "required", "stage": "later",
         "why": "the application itself", "min_bytes": 10}]}),
        encoding="utf-8")
    with pytest.raises(layout.Malformed, match="min_bytes"):
        layout.load(bad)


def test_a_manifest_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(layout.Malformed, match="no such file"):
        layout.load(tmp_path / "nope.json")


# --- checking a folder against it -------------------------------------------

def test_a_complete_bundle_has_no_problems(bundle):
    root, inputs = bundle
    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert problems == []


def test_a_missing_qt_platform_plugin_is_named_with_its_reason(bundle):
    """The failure this exists for: without it, Qt aborts with "could not load
    the Qt platform plugin windows" before the app runs a line, and the client
    sees a dialog naming a DLL they have never heard of."""
    root, inputs = bundle
    rule = rule_for("_internal/PySide6/plugins/platforms/qwindows.dll")
    (root / rule.pattern).unlink()

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert len(problems) == 1, problems
    assert "qwindows.dll" in problems[0]
    assert rule.why in problems[0]


def test_a_zero_byte_model_is_a_problem_even_though_the_file_is_there(bundle):
    """What a OneDrive placeholder, an interrupted copy and an antivirus that
    quarantined the contents all look like from here."""
    root, inputs = bundle
    empty = sorted((root / "models").glob("*.onnx"))[0]
    empty.write_bytes(b"")

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert len(problems) == 1, problems
    assert empty.name in problems[0] and "0 bytes" in problems[0]


def test_a_model_that_is_not_the_one_inputs_json_names_is_a_problem(bundle):
    """Same name, same size, different file. Only a checksum sees this — and a
    checkpoint that decodes to garbage detects nothing, silently, on the
    client's machine."""
    root, inputs = bundle
    swapped = sorted((root / "models").glob("*.onnx"))[1]
    swapped.write_bytes(b"?" * MODEL_BYTES)

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert len(problems) == 1, problems
    assert swapped.name in problems[0] and "sha256" in problems[0]


def test_no_hashing_means_no_checksum_problems(bundle):
    """`hash_files=False` is what the startup check uses: 211 MB of SHA-256 on
    every launch would be seconds of spinning cursor for a fault that happens
    once."""
    root, inputs = bundle
    sorted((root / "models").glob("*.onnx"))[1].write_bytes(b"?" * MODEL_BYTES)

    assert layout.check(root, layout.load(MANIFEST),
                        hash_files=False, inputs=inputs) == []


def test_a_model_inputs_json_says_nothing_about_is_a_problem(bundle):
    """Nothing can verify it, and `min_count` alone would have counted it as
    one of the three that are supposed to be there."""
    root, inputs = bundle
    (root / "models" / "rtmpose-from-somewhere-else.onnx").write_bytes(
        b"z" * MODEL_BYTES)

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert len(problems) == 1, problems
    assert "rtmpose-from-somewhere-else.onnx" in problems[0]


def test_a_checksum_nobody_has_filled_in_yet_refuses_to_pass(bundle, tmp_path):
    """`FILL-FROM-CI` is the placeholder for a hash that could not be produced
    when the entry was written. Treating it as "verified" is the one thing it
    must never do."""
    root, _ = bundle
    unfilled = tmp_path / "unfilled.json"
    data = json.loads(fake_inputs(root, tmp_path / "tmp.json").read_text(
        encoding="utf-8"))
    data["weights"]["balanced"][0]["sha256"] = layout.UNFILLED
    unfilled.write_text(json.dumps(data), encoding="utf-8")

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=unfilled)
    assert len(problems) == 1, problems
    assert layout.UNFILLED in problems[0] and "CI" in problems[0]


def test_the_locale_tree_is_forbidden(bundle):
    """900 MB of Blender translations that no delivery needs, and the reason
    the bundle is a gigabyte. The purge that removes them ran with
    -ErrorAction SilentlyContinue, so it could fail without saying so."""
    root, inputs = bundle
    (root / "blender" / "5.1" / "datafiles" / "locale").mkdir(parents=True)

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert len(problems) == 1, problems
    assert "locale" in problems[0]


def test_the_opengl_dlls_that_can_never_load_are_forbidden(bundle):
    """PyOpenGL's vc9/vc10 freeglut builds import msvcr90.dll and msvcr100.dll,
    runtimes this bundle does not ship and Windows does not provide. The vc14
    pair beside them is the one that loads, and stays."""
    root, inputs = bundle
    dlls = root / "_internal" / "OpenGL" / "DLLS"
    (dlls / "freeglut64.vc9.dll").write_bytes(b"x")
    (dlls / "gle64.vc10.dll").write_bytes(b"x")

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert {p.split(" ", 1)[0] for p in problems} == {
        "_internal/OpenGL/DLLS/freeglut64.vc9.dll",
        "_internal/OpenGL/DLLS/gle64.vc10.dll"}, problems


def test_every_problem_is_reported_not_only_the_first(bundle):
    """A build that fails one file at a time costs one CI round trip each."""
    root, inputs = bundle
    (root / "_internal/PySide6/plugins/platforms/qwindows.dll").unlink()
    (root / "_internal/PySide6/plugins/imageformats/qjpeg.dll").unlink()
    (root / "blender/blender.exe").unlink()

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert len(problems) == 3, problems


def test_a_rule_can_ask_for_a_directory(bundle):
    """`workspace/` is a folder, not a file, and an empty one: a pattern with
    a trailing slash is the only rule kind that can notice it is gone."""
    root, inputs = bundle
    (root / "workspace").rmdir()

    problems = layout.check(root, layout.load(MANIFEST),
                            hash_files=True, inputs=inputs)
    assert len(problems) == 1, problems
    assert "workspace" in problems[0]


def test_a_folder_that_is_not_there_says_so_once(tmp_path):
    """Rather than one problem per rule for a path that was mistyped."""
    problems = layout.check(tmp_path / "nope", layout.load(MANIFEST))
    assert len(problems) == 1
    assert "nope" in problems[0]


def test_min_count_absorbs_the_drift_it_is_there_for(bundle):
    """onnxruntime moves its DLLs between releases and Qt renames its style
    plugin; what matters is that at least one arrived."""
    root, inputs = bundle
    extra = root / "_internal" / "onnxruntime" / "capi" / "onnxruntime.dll"
    extra.write_bytes(b"x")
    styles = root / "_internal/PySide6/plugins/styles"
    (styles / "qwindowsvistastyle.dll").write_bytes(b"x")

    assert layout.check(root, layout.load(MANIFEST),
                        hash_files=True, inputs=inputs) == []


# --- the command line -------------------------------------------------------

def test_the_cli_passes_a_complete_bundle(bundle):
    root, inputs = bundle
    got = subprocess.run(
        [sys.executable, "tools/check_bundle_layout.py", str(root),
         "--inputs", str(inputs)],
        cwd=ROOT, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr


def test_the_cli_lists_every_problem_and_exits_1(bundle):
    root, inputs = bundle
    (root / "_internal/PySide6/plugins/platforms/qwindows.dll").unlink()
    (root / "blender/blender.exe").unlink()

    got = subprocess.run(
        [sys.executable, "tools/check_bundle_layout.py", str(root),
         "--inputs", str(inputs)],
        cwd=ROOT, capture_output=True, text=True)
    assert got.returncode == 1
    assert "qwindows.dll" in got.stderr and "blender.exe" in got.stderr


def test_the_cli_can_skip_the_hashing(bundle):
    """`--no-hash` for the build step that has already verified the
    checkpoints; the release gate runs it without."""
    root, _ = bundle
    got = subprocess.run(
        [sys.executable, "tools/check_bundle_layout.py", str(root),
         "--no-hash"],
        cwd=ROOT, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr

"""One source of truth for what the Windows build pulls in from outside.

Three external things decide whether the client's bundle works: the Python the
exe is frozen against, the Blender that is unzipped beside it, and the ONNX
checkpoints copied next to that. All three used to be written down in several
places at once — `5.1.1` in the ps1 and in both workflows, the interpreter
nowhere at all — and none of them was verified against a hash. A silent
mismatch there does not fail the build; it ships.

So: `.python-version` pins the interpreter, `packaging/windows/inputs.json` is
the ONLY place a Blender version or a checksum is written, and these tests keep
both true.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INPUTS = ROOT / "packaging" / "windows" / "inputs.json"
PS1 = ROOT / "tools" / "make_windows_bundle.ps1"
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

# Task E owns .github/**, and lands after this task. Until it does, the two
# assertions below describe a contract nothing satisfies yet — so they are
# expected to fail, strictly: the moment E's change makes one pass, pytest
# reports XPASS as a failure and the marker has to be deleted with it.
NOT_YET_TASK_E = pytest.mark.xfail(
    strict=True,
    reason="Task E owns .github/**; when E lands this XPASSes — delete this "
           "marker then")


def _inputs():
    return json.loads(INPUTS.read_text(encoding="utf-8"))


def _blender_input(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "tools/blender_input.py", *args],
        cwd=cwd, capture_output=True, text=True)


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


# --- B. inputs.json is the only place a version or a hash is written --------

def test_inputs_json_parses_and_has_both_sections():
    data = _inputs()
    assert set(data) == {"blender", "weights"}
    assert set(data["blender"]) == {"version", "url", "sha256", "size"}


@pytest.mark.parametrize("mode", ["balanced", "lightweight"])
def test_every_weights_entry_says_exactly_what_fetch_weights_reads(mode):
    """`name`, `sha256`, `size` and nothing else, three per mode: the two pose
    checkpoints (COCO-17 and Halpe-26) plus the YOLOX detector they share. An
    entry that quietly lost a key would be staged and never verified."""
    import tools.fetch_weights as fw

    entries = _inputs()["weights"][mode]
    assert len(entries) == 3, [e.get("name") for e in entries]
    for entry in entries:
        assert set(entry) == set(fw.ENTRY_KEYS) == {"name", "sha256", "size"}


def test_every_hash_is_64_hex_and_every_size_is_a_positive_int():
    data = _inputs()
    entries = [data["blender"], *(w for m in data["weights"].values() for w in m)]
    for entry in entries:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry
        assert isinstance(entry["size"], int) and entry["size"] > 0, entry


def test_the_blender_url_names_the_version_it_claims():
    """A copy-paste that bumps the version but not the URL would download the
    old build and verify it against the old hash — green, and wrong."""
    blender = _inputs()["blender"]
    name = blender["url"].rsplit("/", 1)[-1]
    assert name == f"blender-{blender['version']}-windows-x64.zip"


@pytest.mark.parametrize("mode", ["balanced", "lightweight"])
def test_the_weights_listed_are_exactly_what_rtmlib_asks_for(mode):
    """Read out of rtmlib's own MODE tables rather than retyped, so an rtmlib
    upgrade that changes a checkpoint fails HERE — on our machine, with a
    diff to read — instead of on the client's, where the app would fall
    through to rtmlib's downloader."""
    from pose3d.detect import models

    expected = (set(models.required_files(mode, feet=False))
                | set(models.required_files(mode, feet=True)))
    listed = {w["name"] for w in _inputs()["weights"][mode]}
    assert listed == expected


def test_every_mode_the_build_asks_for_has_an_entry():
    """`fetch_weights.py --mode X` with no entry for X cannot verify anything.
    Covers what is written down (grep) and the default, which is what the
    bundle script's bare invocation actually uses."""
    import tools.fetch_weights as fw

    asked = set()
    for path in [*(ROOT / ".github").rglob("*"), *(ROOT / "tools").rglob("*")]:
        # the file types a command line can be written in; a .pyc matches
        # anything if you let it
        if path.is_file() and path.suffix in (".yml", ".yaml", ".py", ".ps1"):
            asked |= set(re.findall(r"--mode[=\s]+([\w-]+)",
                                    path.read_text(encoding="utf-8")))
    asked.add(fw.DEFAULT_MODE)
    assert asked <= set(_inputs()["weights"]), (
        f"no inputs.json entry for {sorted(asked - set(_inputs()['weights']))}")


def test_the_bundle_script_writes_no_blender_version_of_its_own():
    text = PS1.read_text(encoding="utf-8")
    assert "5.1.1" not in text and "blender-5." not in text


@NOT_YET_TASK_E
def test_no_workflow_writes_a_blender_version_of_its_own():
    for wf in WORKFLOWS:
        text = wf.read_text(encoding="utf-8")
        assert "5.1.1" not in text and "blender-5." not in text, wf


# --- tools/blender_input.py: the workflows' only door to inputs.json --------

@pytest.mark.parametrize("flag,expected", [
    ("--version", "version"),
    ("--url", "url"),
    ("--sha256", "sha256"),
])
def test_blender_input_prints_one_value(flag, expected):
    got = _blender_input(flag)
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == str(_inputs()["blender"][expected])
    assert "\n" not in got.stdout.strip()


def test_blender_input_derives_the_zip_name_and_the_cache_key():
    blender = _inputs()["blender"]
    assert _blender_input("--zip-name").stdout.strip() == \
        f"blender-{blender['version']}-windows-x64.zip"
    assert _blender_input("--cache-key").stdout.strip() == \
        f"blender-{blender['version']}-windows-x64-{blender['sha256'][:12]}"


def test_the_cache_key_changes_when_the_hash_does(tmp_path):
    """It is the cache key that makes a hash bump take effect: keyed on the
    version alone, a corrected checksum would keep restoring the old zip from
    the runner's cache and every build would fail on a file nobody can see."""
    data = _inputs()
    data["blender"]["sha256"] = "b" * 64
    other = tmp_path / "inputs.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    got = _blender_input("--cache-key", "--inputs", str(other))
    assert got.stdout.strip().endswith("-bbbbbbbbbbbb")


@pytest.mark.parametrize("content", ["", "{", '{"blender": {}}',
                                     '{"blender": {"version": "5.1.1", '
                                     '"url": "http://x/blender-9.9.9-windows-x64.zip", '
                                     '"sha256": "' + "a" * 64 + '", "size": 1}}'])
def test_blender_input_refuses_a_malformed_file(tmp_path, content):
    bad = tmp_path / "inputs.json"
    bad.write_text(content, encoding="utf-8")
    got = _blender_input("--version", "--inputs", str(bad))
    assert got.returncode == 1
    assert got.stderr.strip()


def test_blender_input_refuses_a_missing_file(tmp_path):
    got = _blender_input("--version", "--inputs", str(tmp_path / "nope.json"))
    assert got.returncode == 1
    assert "nope.json" in got.stderr


def test_an_unfilled_checksum_fails_with_an_instruction(tmp_path):
    """The placeholder means "nobody has hashed this yet". Printing it would
    make the build compare a real file against the word FILL-FROM-CI and report
    a mismatch, which reads as a corrupt download."""
    data = _inputs()
    data["blender"]["sha256"] = "FILL-FROM-CI"
    stub = tmp_path / "inputs.json"
    stub.write_text(json.dumps(data), encoding="utf-8")

    for flag in ("--sha256", "--cache-key"):
        got = _blender_input(flag, "--inputs", str(stub))
        assert got.returncode == 1
        assert "FILL-FROM-CI" in got.stderr and "inputs.json" in got.stderr
    # ...but the version and the URL are still knowable, so the download step
    # can run and produce the hash the maintainer needs.
    assert _blender_input("--url", "--inputs", str(stub)).returncode == 0


# --- tools/fetch_weights.py: what it staged is what inputs.json describes ---

def _weights_stub(monkeypatch, out, names, content=b"x"):
    """Make stage() write `names` into `out` without touching the network."""
    import tools.fetch_weights as fw

    def stage(out, mode, feet, allow_download):
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name in names:
            (out / name).write_bytes(content)
            written.append(out / name)
        return written

    monkeypatch.setattr(fw, "stage", stage)
    return fw


def test_a_staged_file_that_does_not_match_is_named_with_both_hashes(
        monkeypatch, tmp_path):
    """A mistyped hash fails every build until someone fixes it, so the message
    has to carry the actual value — otherwise the fix needs a Windows machine."""
    name = _inputs()["weights"]["balanced"][0]["name"]
    fw = _weights_stub(monkeypatch, tmp_path, [name])
    monkeypatch.setattr(sys, "argv", ["fetch_weights.py", "--out", str(tmp_path),
                                      "--no-download"])
    with pytest.raises(SystemExit) as exc:
        fw.main()
    message = str(exc.value)
    assert name in message
    assert _inputs()["weights"]["balanced"][0]["sha256"] in message
    # sha256 of b"x"
    assert "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881" \
        in message


def test_a_file_that_was_already_present_is_verified_too(monkeypatch, tmp_path):
    """The real stage() prints "have" and returns early for those, copying
    nothing — which is exactly the file a half-finished earlier run may have
    left truncated. Verifying only what this run copied would miss it."""
    import tools.fetch_weights as fw

    names = [w["name"] for w in _inputs()["weights"]["balanced"]]
    for name in names:
        (tmp_path / name).write_bytes(b"truncated")
    monkeypatch.setattr(sys, "argv", ["fetch_weights.py", "--out", str(tmp_path),
                                      "--no-download"])
    with pytest.raises(SystemExit) as exc:
        fw.main()          # the real stage(): every file is "have", no network
    assert any(name in str(exc.value) for name in names)


def test_a_mode_with_no_entry_is_a_named_failure():
    import tools.fetch_weights as fw

    with pytest.raises(SystemExit) as exc:
        fw.expected("performance")
    assert "performance" in str(exc.value)
    assert "inputs.json" in str(exc.value)


def test_only_the_modes_inputs_json_describes_can_be_asked_for(monkeypatch,
                                                               tmp_path):
    """`--mode performance` was an offered choice with no entry behind it, so
    the one thing it could do was stage 150 MB and refuse to verify it."""
    import tools.fetch_weights as fw

    monkeypatch.setattr(sys, "argv",
                        ["fetch_weights.py", "--out", str(tmp_path),
                         "--mode", "performance", "--no-download"])
    monkeypatch.setattr(fw, "stage", lambda *a, **kw: pytest.fail(
        "a mode with no entry must not reach the staging"))
    with pytest.raises(SystemExit) as exc:
        fw.main()
    assert exc.value.code == 2          # argparse's own usage error


@pytest.mark.parametrize("content,phrase", [
    (None, "is missing"),
    ("{ not json", "could not be read"),
    ('{"blender": {}}', "no 'weights' section"),
    ('{"weights": {"balanced": [{"name": "a.onnx", "size": 1}]}}',
     "missing sha256"),
])
def test_expected_names_the_file_and_the_problem(tmp_path, content, phrase):
    """`inputs.json` is the file someone is sent to correct, so every way it
    can be unusable has to say which file and what is wrong with it — not a
    FileNotFoundError, a JSONDecodeError or a KeyError out of a build."""
    import tools.fetch_weights as fw

    path = tmp_path / "inputs.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        fw.expected("balanced", inputs=path)
    assert "inputs.json" in str(exc.value)
    assert phrase in str(exc.value)


def test_verify_only_checks_what_is_there_and_stages_nothing(
        monkeypatch, tmp_path):
    import tools.fetch_weights as fw

    def refuse(*a, **kw):
        raise AssertionError("--verify-only must not stage")

    monkeypatch.setattr(fw, "stage", refuse)
    monkeypatch.setattr(sys, "argv", ["fetch_weights.py", "--out", str(tmp_path),
                                      "--verify-only"])
    with pytest.raises(SystemExit) as exc:
        fw.main()
    assert any(w["name"] in str(exc.value)
               for w in _inputs()["weights"]["balanced"])


# --- tools/make_windows_bundle.ps1 -----------------------------------------

def test_the_bundle_script_verifies_the_blender_download():
    """~414 MB over HTTP with no checksum: a truncated or replaced archive
    unzips into a blender\\ folder that fails on the client, not here."""
    text = PS1.read_text(encoding="utf-8")
    assert "tools/blender_input.py" in text
    assert "Get-FileHash" in text and "SHA256" in text
    # a bad archive must not survive to be "cached" into the next run
    assert re.search(r"Remove-Item[^\n]*\$archive", text)


def test_the_bundle_script_is_told_which_python_to_use():
    """`& python` is whatever is first on PATH — on a Windows runner that is
    often the store stub, which is not the venv the lockfile installed."""
    text = PS1.read_text(encoding="utf-8")
    assert "$Python" in text
    assert not re.search(r"&\s*python\b", text)


def test_the_locale_purge_fails_instead_of_silently_doing_nothing():
    """-ErrorAction SilentlyContinue on a path that has moved deletes nothing
    and says nothing: the bundle just quietly grows 100 MB."""
    text = PS1.read_text(encoding="utf-8")
    # anchored on the VARIABLE, not on the word: the prose above it mentions
    # the purge too, and a comment gaining the word "locale" would have moved
    # this window over a block that says nothing about deleting anything.
    purge = text.split("$locale", 1)[1].split("--- pose weights", 1)[0]
    assert "SilentlyContinue" not in purge
    assert "throw" in purge


def test_the_bundle_script_zips_through_the_tested_tool():
    """The inline -Zip block had no MAX_PATH gate and CI never ran it, so the
    only zip path anyone exercised was the one pasted into the workflow."""
    text = PS1.read_text(encoding="utf-8")
    assert "tools/bundle_zip.py" in text
    assert "Compress-Archive" not in text

"""Zipping the bundle, and the MAX_PATH gate that has to come with it.

The first Windows build the client received would not extract: "path too long".
Windows' limit is 260 characters for the whole path, Blender's deepest file is
already ~148 characters inside the bundle, and everything left over is the
folder the client extracts into. So the archive is built from the PARENT of the
bundle — entries begin at `Pose3D-Windows\\`, not `build\\Pose3D-Windows\\` —
and anything deeper than 180 characters fails the build instead of the client.

That gate existed only as PowerShell pasted into the release workflow, which is
Windows-only and therefore never ran here. This is the same rule as a pure
function, tested on synthetic names on any platform.
"""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import bundle_zip  # noqa: E402


def test_the_deepest_path_is_the_longest_name():
    assert bundle_zip.deepest_path(["a", "bbb", "cc"]) == 3


def test_an_empty_archive_has_no_depth():
    assert bundle_zip.deepest_path([]) == 0


def test_a_181_character_path_fails_and_says_which_one():
    """80 characters is a realistic extraction root: C:\\Users\\<name>\\
    Downloads\\ is most of it already."""
    long = "Pose3D-Windows/" + "a" * (181 - len("Pose3D-Windows/"))
    problems = bundle_zip.check_depth(["Pose3D-Windows/ok", long])
    assert len(problems) == 1
    assert long in problems[0]
    assert "181" in problems[0] and "180" in problems[0]


def test_179_characters_is_fine():
    name = "Pose3D-Windows/" + "a" * (179 - len("Pose3D-Windows/"))
    assert bundle_zip.check_depth([name]) == []


def test_the_limit_is_the_limit_not_one_past_it():
    assert bundle_zip.check_depth(["a" * 180]) == []
    assert bundle_zip.check_depth(["a" * 181]) != []


def test_a_tighter_limit_can_be_asked_for():
    assert bundle_zip.check_depth(["a" * 20], limit=10) != []


def test_an_empty_directory_that_is_too_deep_is_still_too_deep(tmp_path):
    """A zip stores an entry per directory, and an empty one is the only entry
    that can be too long with no file under it to notice — so a check that
    looked at files alone passed a bundle the client still could not
    extract."""
    folder = tmp_path / "build" / "Pose3D-Windows"
    deep = folder / "/".join(["dir" * 8] * 8)
    deep.mkdir(parents=True)
    (folder / "Pose3D.exe").write_bytes(b"x")

    with pytest.raises(SystemExit) as exc:
        bundle_zip.zip_bundle(folder, tmp_path / "out.zip")
    assert "dirdirdir" in str(exc.value)


def _bundle(tmp_path, *relative):
    """A stand-in for build/Pose3D-Windows, under a build/ folder like the real
    one, so a zip that carried its source path would be visible."""
    folder = tmp_path / "build" / "Pose3D-Windows"
    for rel in relative:
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return folder


def test_the_archive_starts_at_the_bundle_folder(tmp_path):
    """Every character of `build\\` here is a character taken off the client's
    extraction path. Extracting the first build failed for six of them."""
    folder = _bundle(tmp_path, "Pose3D.exe", "_internal/python312.dll")
    out = bundle_zip.zip_bundle(folder, tmp_path / "Pose3D-Windows.zip")

    names = zipfile.ZipFile(out).namelist()
    assert not [n for n in names if n.startswith("build/")]
    assert {n.rstrip("/") for n in names} >= {
        "Pose3D-Windows/Pose3D.exe",
        "Pose3D-Windows/_internal/python312.dll"}


def test_zipping_returns_the_archive_it_wrote(tmp_path):
    folder = _bundle(tmp_path, "Pose3D.exe")
    out = tmp_path / "somewhere" / "Pose3D-Windows.zip"
    assert bundle_zip.zip_bundle(folder, out) == out
    assert out.is_file()


def test_the_archive_lands_where_it_was_asked_to(tmp_path):
    """shutil.make_archive names the file itself, from a base it appends .zip
    to. An --out that is not already called .zip would otherwise be reported
    as written and not be there."""
    folder = _bundle(tmp_path, "Pose3D.exe")
    out = tmp_path / "delivery-2026-09"
    assert bundle_zip.zip_bundle(folder, out) == out
    assert out.is_file() and zipfile.is_zipfile(out)


def test_a_bundle_that_is_too_deep_is_refused_before_it_is_zipped(tmp_path):
    """Loud, and cheap: the check is on the names, so a 1.6 GB tree does not
    have to be compressed first to learn that it cannot be extracted."""
    deep = "/".join(["dir" * 8] * 8) + "/leaf.dll"
    folder = _bundle(tmp_path, "Pose3D.exe", deep)
    out = tmp_path / "Pose3D-Windows.zip"

    with pytest.raises(SystemExit) as exc:
        bundle_zip.zip_bundle(folder, out)
    assert "leaf.dll" in str(exc.value)
    assert not out.exists(), "nothing may be published from a refused bundle"


def test_a_missing_folder_is_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        bundle_zip.zip_bundle(tmp_path / "nope", tmp_path / "out.zip")
    assert "nope" in str(exc.value)


def test_the_cli_zips_beside_the_folder_by_default(tmp_path):
    folder = _bundle(tmp_path, "Pose3D.exe")
    got = subprocess.run([sys.executable, "tools/bundle_zip.py", str(folder)],
                         cwd=ROOT, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert (folder.parent / "Pose3D-Windows.zip").is_file()


@pytest.mark.skipif(sys.platform.startswith("win"),
                    reason="the stand-in 7z is a #!/bin/sh script, and a "
                           "Windows PATH lookup would not run it")
def test_a_7z_that_fails_is_one_line_and_not_a_traceback(tmp_path, monkeypatch,
                                                         capsys):
    """A full disk or a locked output file is an ordinary way for the release
    step to end. It used to end in a CalledProcessError traceback out of a
    build script, which says nothing the person reading the log can act on."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "7z"
    fake.write_text("#!/bin/sh\necho 'ERROR: Can not open output file' >&2\n"
                    "exit 2\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    folder = _bundle(tmp_path, "Pose3D.exe")
    assert bundle_zip.main([str(folder), "--out", str(tmp_path / "out.zip")]) == 1

    err = capsys.readouterr().err
    assert "ERROR: Can not open output file" in err
    assert "Traceback" not in err


def test_the_cli_reports_the_deep_path_and_exits_nonzero(tmp_path):
    folder = _bundle(tmp_path, "/".join(["dir" * 8] * 8) + "/leaf.dll")
    got = subprocess.run(
        [sys.executable, "tools/bundle_zip.py", str(folder),
         "--out", str(tmp_path / "out.zip"), "--limit", "180"],
        cwd=ROOT, capture_output=True, text=True)
    assert got.returncode == 1
    assert "leaf.dll" in got.stderr

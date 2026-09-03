"""No text file in `pose3d/` is read or written in the machine's code page.

`open()`, `Path.read_text()`, `Path.write_text()` and `gzip.open(..., "rt")`
default to `locale.getencoding()`. On the developer's Linux box that is UTF-8
and nothing ever goes wrong; on the client's Windows machine it is whatever
their ANSI code page is, and the same file then reads back differently:

* cp1252 (Western Europe) decodes UTF-8 bytes without complaining and yields
  mojibake — `dark.qss`'s em dash became `â€"`, silently, inside a comment;
* cp932/936/949/950 (JP/CN/KR Windows) raise `UnicodeDecodeError` on the same
  bytes, so the app dies before it shows a window.

This is a whole-package rule rather than a list of fixed call sites, because
the next one written without `encoding=` is a Windows-only bug that no test on
this machine would ever fail on. JSON *writes* happen to be safe
(`ensure_ascii=True` by default) — but the reads are not, and a rule with an
exception in it is a rule nobody applies.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "pose3d"

TEXT_METHODS = {"read_text", "write_text"}

#: Modules whose `open()` is a text-capable file open like the builtin's.
#: `Image.open`, `zipfile.ZipFile.open` and friends are binary and are not
#: checked — which is the scanner's one blind spot: `some_path.open("w")` on a
#: bare variable is not recognised as a path open either. `read_text` /
#: `write_text` are the idiom in this package, and both ARE checked.
OPEN_MODULES = {"gzip", "bz2", "lzma"}


def _mode_of(call: ast.Call) -> str:
    """The literal mode string passed to this call, "" if none/not literal."""
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return kw.value.value
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) \
            and isinstance(call.args[1].value, str):
        return call.args[1].value
    return ""


def _kind(func: ast.expr) -> str:
    """"read_text" / "write_text" / "open", or "" if this is not one of ours."""
    if isinstance(func, ast.Name):
        return "open" if func.id == "open" else ""
    if not isinstance(func, ast.Attribute):
        return ""
    if func.attr in TEXT_METHODS:
        return func.attr
    if func.attr == "open":
        base = func.value
        if isinstance(base, ast.Name) and base.id in OPEN_MODULES:
            return "open"
        if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) \
                and base.func.id == "Path":
            return "open"
    return ""


def _has_encoding(call: ast.Call) -> bool:
    return any(kw.arg == "encoding" for kw in call.keywords) or \
        any(kw.arg is None for kw in call.keywords)      # **kwargs: trust it


def offenders(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _kind(node.func)
        if not name:
            continue
        text_mode = True if name in TEXT_METHODS else "b" not in _mode_of(node)
        if text_mode and not _has_encoding(node):
            out.append(f"{path}:{node.lineno}: {name}() without encoding=")
    return out


#: Files whose call sites belong to another workstream and are fixed on its
#: branch (main_window.py:504,682 and model.py:344,348,363). Both read JSON we
#: wrote ourselves, which `json.dumps` keeps ASCII, so the exposure is small —
#: but the exemption is here to be deleted, not kept: remove a name from this
#: set as soon as that file's reads say `encoding="utf-8"`.
ELSEWHERE = {"ui/main_window.py", "ui/model.py"}


def test_every_text_read_and_write_in_the_package_names_its_encoding():
    skip = {str(PACKAGE / rel) for rel in ELSEWHERE}
    found = [p for f in sorted(PACKAGE.rglob("*.py"))
             if str(f) not in skip for p in offenders(f)]
    assert found == [], "\n".join(found)


def test_the_scan_would_actually_catch_one(tmp_path):
    """A scanner that finds nothing because it looks for nothing is worse than
    no scanner, so prove it fails on the shapes it is meant to catch."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from pathlib import Path\n"
        "Path('x').read_text()\n"
        "Path('x').write_text('y')\n"
        "open('x')\n"
        "open('x', 'w')\n"
        "gzip.open(p, 'rt')\n"
        "Path('x').open('w')\n", encoding="utf-8")
    assert len(offenders(bad)) == 6


def test_the_scan_does_not_flag_binary_or_encoded_calls(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text(
        "open('x', 'rb')\n"
        "open('x', mode='wb')\n"
        "open('x', encoding='utf-8')\n"
        "Path('x').read_text(encoding='utf-8')\n"
        "gzip.open(p, 'rt', encoding='utf-8')\n"
        "open(path, **kw)\n"
        "Image.open(p)\n"                    # PIL: binary, and not a path open
        "zf.open(name)\n", encoding="utf-8")
    assert offenders(ok) == []

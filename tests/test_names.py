"""Turning what the user typed into a name a folder can actually have.

The import dialog used to do `text.strip().replace(" ", "_")`, so a project
called `Take 3: front/back` produced a folder Windows refuses to create — and
the failure surfaced minutes later, inside Blender, as an unwritable output
path. Windows' rules are stricter than POSIX's in three separate ways
(forbidden characters, no trailing dot or space, and a list of device names
reserved since DOS), and all three have to be applied or none of it helps.
"""
import pytest

from pose3d.core.names import safe_name

CASES = [
    # (typed, expected, why)
    ("Client Take 3", "Client_Take_3", "spaces become underscores"),
    ("  padded  ", "padded", "surrounding whitespace is dropped"),
    ("a\tb\nc", "a_b_c", "tabs and newlines are whitespace too"),
    ("Take   3", "Take_3", "a run of whitespace is one underscore"),
    ('front/back', "frontback", "the path separator is removed"),
    ("C:\\shoot", "Cshoot", "drive colon and backslash are removed"),
    ('a<b>c:d"e|f?g*h', "abcdefgh", "every Windows-forbidden character"),
    ("bell\x07", "bell", "control characters are removed"),
    ("trailing.", "trailing", "Windows silently drops a trailing dot"),
    ("trailing...", "trailing", "…however many there are"),
    ("CON", "CON_", "a reserved DOS device name is not usable"),
    ("con", "con_", "the reservation is case-insensitive"),
    ("COM1", "COM1_", "COM1-9 are reserved"),
    ("LPT9", "LPT9_", "LPT1-9 are reserved"),
    ("NUL.take", "NUL_.take", "the reservation applies to the stem too"),
    ("COM0", "COM0", "COM0 is not reserved"),
    ("COM10", "COM10", "only single digits are reserved"),
    ("consult", "consult", "a name that merely starts with one is fine"),
    ("", "project", "nothing typed falls back"),
    ("///", "project", "nothing survives, so it falls back"),
    ("x" * 200, "x" * 64, "capped at 64 characters"),
]


@pytest.mark.parametrize("typed,expected,why", CASES,
                         ids=[c[0][:24] or "empty" for c in CASES])
def test_safe_name(typed, expected, why):
    assert safe_name(typed) == expected, why


def test_the_fallback_is_the_callers_to_choose():
    assert safe_name("", fallback="untitled") == "untitled"
    assert safe_name("<<<>>>", fallback="untitled") == "untitled"


def test_a_name_capped_at_the_limit_never_ends_in_a_dot():
    """Truncation can expose a trailing dot that was legal mid-name."""
    out = safe_name("y" * 63 + ".tail")
    assert len(out) <= 64
    assert not out.endswith(".") and not out.endswith(" ")


def test_the_non_ascii_name_the_client_actually_uses_survives():
    """Unicode is fine on NTFS; only the listed characters are the problem."""
    assert safe_name("Müller Take 2") == "Müller_Take_2"
    assert safe_name("照片 1") == "照片_1"

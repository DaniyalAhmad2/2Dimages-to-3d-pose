"""A user-typed name turned into one a file system will accept.

Windows is the strict one, and it is strict in three separate ways: nine
characters are forbidden outright, a name may not end in a dot or a space (the
API drops them silently, so `Take 3.` becomes `Take 3` and a later lookup by
the typed name misses), and a short list of DOS device names — `CON`, `NUL`,
`COM1`…`LPT9` — is reserved whatever extension follows. Getting any one of
them wrong shows up long after the user typed it: the import created the
project, and the export died inside Blender on a path that could not exist.
"""
from __future__ import annotations

import re

#: The characters Windows forbids in a file or folder name.
FORBIDDEN = '<>:"/\\|?*'

#: DOS device names, reserved with or without an extension.
RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)

MAX_LEN = 64

_WHITESPACE_RUN = re.compile(r"\s+")


def safe_name(s: str, fallback: str = "project") -> str:
    """`s` as a folder name, or `fallback` when nothing usable is left."""
    out = _WHITESPACE_RUN.sub("_", str(s).strip())
    out = "".join(c for c in out
                  if c not in FORBIDDEN and (ord(c) >= 32 and ord(c) != 127))
    out = out.rstrip(". ")[:MAX_LEN].rstrip(". ")
    if not out:
        return fallback
    stem, dot, ext = out.partition(".")
    if stem.upper() in RESERVED:
        # `CON`, and `CON.take` too: the reservation is on the stem, so a
        # suffix does not rescue it. One appended underscore is the smallest
        # change that keeps the name recognisable.
        out = f"{stem}_{dot}{ext}"[:MAX_LEN].rstrip(". ")
    return out or fallback

"""Reading an image file, on every path a Windows machine can produce.

`cv2.imread(path)` hands the path to OpenCV's C++ file layer, which on Windows
encodes it with the machine's ANSI code page. A user folder like
`C:\\Users\\Müller` or a Japanese file name does not survive that encoding, and
`imread` reports the failure by returning `None` — no exception, no message. The
photo still appeared in the UI, because Qt reads the same path as UTF-16, so the
symptom surfaced much later as `'NoneType' object has no attribute 'shape'`
inside detection.

`np.fromfile` + `cv2.imdecode` keeps the path in Python, where it is already
correct, and hands OpenCV bytes. Everything that can still go wrong — the file
is gone, it is a 0-byte OneDrive "online only" placeholder, it is truncated or
is not an image at all — is raised as `ImageReadError`, named, so the caller
reports which file rather than dying on a `None`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ImageReadError(OSError):
    """An image file could not be read. Subclasses OSError so call sites that
    already handle a missing file keep working."""


def read_image(path) -> np.ndarray:
    """The image at `path` as a BGR array, or raise `ImageReadError`."""
    import cv2

    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as e:
        raise ImageReadError(f"Cannot read the image '{p}': {e.strerror or e}."
                             ) from e
    if size == 0:
        raise ImageReadError(
            f"The image '{p}' is 0 bytes. If it is stored in OneDrive it may "
            f"still be online-only — open the folder in File Explorer and "
            f"choose 'Always keep on this device', then try again.")
    try:
        raw = np.fromfile(str(p), dtype=np.uint8)
    except OSError as e:
        raise ImageReadError(f"Cannot read the image '{p}': {e.strerror or e}."
                             ) from e
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageReadError(
            f"The image '{p}' could not be decoded — it is damaged, "
            f"incomplete, or not an image file.")
    return img

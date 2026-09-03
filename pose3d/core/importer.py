"""Build a ProjectData from two folders/lists of synced camera images.

Left/right frames are matched by the numeric token in the filename
(left_0001 <-> right_0001, L_1 <-> R_1, ...); if numbering doesn't line up,
fall back to matching by sorted order.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from pose3d.core.project import CAM_LEFT, CAM_RIGHT, Frame, ProjectData

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS)


def _num_key(path: Path) -> int | None:
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else None


def match_frames(left: list[str | Path], right: list[str | Path]) -> list[tuple[Path, Path]]:
    """Pair left/right images. Numeric-token match first, else sorted order."""
    left = [Path(p) for p in left]
    right = [Path(p) for p in right]

    lk = {_num_key(p): p for p in left}
    rk = {_num_key(p): p for p in right}
    numeric_ok = (
        None not in lk and None not in rk          # every file has a number
        and len(lk) == len(left) and len(rk) == len(right)  # numbers are unique
        and set(lk) & set(rk)                       # at least one common index
    )
    if numeric_ok:
        common = sorted(set(lk) & set(rk))
        return [(lk[k], rk[k]) for k in common]

    # fallback: sorted order, truncated to the shorter list
    ls, rs = sorted(left), sorted(right)
    n = min(len(ls), len(rs))
    return list(zip(ls[:n], rs[:n]))


def build_project(
    left: list[str | Path], right: list[str | Path],
    name: str = "Imported", fps: int = 30,
    copy_into: str | Path | None = None,
    on_progress=None,
) -> ProjectData:
    """Create a ProjectData from matched image pairs.

    If ``copy_into`` is given, images are copied into ``<copy_into>/images``
    and referenced by that path (so the project folder is self-contained).

    ``on_progress(done, total, label)`` is called BEFORE each pair is copied,
    because that is the granularity at which this stalls: a virus scanner or a
    OneDrive placeholder holds up one `copy2` at a time, and the import used
    to sit behind a modal dialog that could say nothing about which one. The
    label therefore names the pair being copied and ``done`` is how many are
    already there.
    """
    pairs = match_frames(left, right)
    project = ProjectData(name=name, fps=fps, calibration_ref="calibration")

    img_dir = None
    if copy_into is not None:
        img_dir = Path(copy_into) / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

    for i, (lp, rp) in enumerate(pairs):
        num = _num_key(lp)
        fid = f"{num:04d}" if num is not None else f"{i:04d}"
        lpath, rpath = Path(lp), Path(rp)
        # BEFORE the copy, not after it: the point of reporting per pair is to
        # name the pair the copy is stuck on, and a report that follows the
        # copy names the last one that finished.
        if on_progress is not None:
            on_progress(i, len(pairs),
                        f"Copying image pair {i + 1} of {len(pairs)}")
        if img_dir is not None:
            ldst = img_dir / f"left_{fid}{lpath.suffix.lower()}"
            rdst = img_dir / f"right_{fid}{rpath.suffix.lower()}"
            shutil.copy2(lpath, ldst)
            shutil.copy2(rpath, rdst)
            lpath, rpath = ldst, rdst
        project.frames.append(Frame(
            frame_id=fid,
            images={CAM_LEFT: str(lpath), CAM_RIGHT: str(rpath)}))
    return project

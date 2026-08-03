#!/usr/bin/env python3
"""Generate packaging/windows/pose3d.ico.

Checked in as a script rather than only as a binary so the icon can be changed
without hunting for whatever tool drew it. Run:

    python tools/make_icon.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "packaging" / "windows" / "pose3d.ico"
SIZES = [16, 24, 32, 48, 64, 128, 256]

BG = (24, 26, 32)
LIMB = (94, 200, 245)
JOINT = (255, 255, 255)
CAM = (245, 158, 66)

# A standing figure in a normalised 0..1 box: the app's own subject.
JOINTS = {
    "head": (0.50, 0.16), "neck": (0.50, 0.28), "hip": (0.50, 0.55),
    "sh_l": (0.34, 0.31), "sh_r": (0.66, 0.31),
    "el_l": (0.26, 0.45), "el_r": (0.74, 0.45),
    "wr_l": (0.22, 0.60), "wr_r": (0.78, 0.60),
    "hp_l": (0.41, 0.56), "hp_r": (0.59, 0.56),
    "kn_l": (0.38, 0.73), "kn_r": (0.62, 0.73),
    "an_l": (0.36, 0.89), "an_r": (0.64, 0.89),
}
BONES = [("head", "neck"), ("neck", "hip"), ("neck", "sh_l"), ("neck", "sh_r"),
         ("sh_l", "el_l"), ("el_l", "wr_l"), ("sh_r", "el_r"), ("el_r", "wr_r"),
         ("hip", "hp_l"), ("hip", "hp_r"), ("hp_l", "kn_l"), ("kn_l", "an_l"),
         ("hp_r", "kn_r"), ("kn_r", "an_r")]


def render(size: int) -> Image.Image:
    # 4x supersampling: at 16px the strokes are sub-pixel and alias badly
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=int(s * 0.22), fill=BG)

    def px(p, inset=0.13):
        x, y = p
        return (s * (inset + x * (1 - 2 * inset)),
                s * (inset + y * (1 - 2 * inset)))

    w = max(1, int(s * 0.035))
    for a, b in BONES:
        d.line([px(JOINTS[a]), px(JOINTS[b])], fill=LIMB, width=w, joint="curve")
    r = max(1, int(s * 0.022))
    for name, p in JOINTS.items():
        x, y = px(p)
        d.ellipse([x - r, y - r, x + r, y + r], fill=JOINT)

    # the two camera views the whole app is built around
    if size >= 32:
        cr = s * 0.055
        for cx, cy in ((s * 0.13, s * 0.87), (s * 0.87, s * 0.87)):
            d.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=CAM)
    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [render(n) for n in SIZES]
    frames[-1].save(OUT, format="ICO",
                    sizes=[(n, n) for n in SIZES], append_images=frames[:-1])
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, sizes {SIZES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

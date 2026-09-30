"""Render assets/seer.svg from the pixel grid in subseer/banner.py.

banner.py is the single source of truth for the art and its colors; edit it
there and rerun this script to refresh the README image.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subseer.banner import _ART, _BG, _PAL  # noqa: E402

PX = 12  # pixel size


def main():
    rows = _ART.splitlines()
    w = max(len(r) for r in rows)
    rects = []
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in _PAL:
                rects.append(f'<rect x="{x*PX}" y="{y*PX}" width="{PX}" height="{PX}" fill="{_PAL[ch][0]}"/>')
    W, H = w * PX, len(rows) * PX
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" shape-rendering="crispEdges">\n'
        f'<rect width="{W}" height="{H}" fill="{_BG}" rx="10"/>\n'
        + "\n".join(rects) + "\n</svg>\n"
    )
    out = ROOT / "assets" / "seer.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({W}x{H}, {len(rects)} pixels)")


if __name__ == "__main__":
    main()

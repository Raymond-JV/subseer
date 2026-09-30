"""The subseer seer as portable terminal pixel art.

A seer gazing into a crystal ball, drawn on a character grid (``_ART``). Each
character maps to an exact color (used for assets/seer.svg) and a hand-picked
xterm-256 color (used in the terminal). The terminal colors are picked by eye
rather than computed, because a nearest-color match turns several of these
tones muddy (the table goes olive, the robe goes flat red).

The terminal rendering uses the "upper half block" trick: each cell stacks two
vertical pixels (fg = top, bg = bottom) so pixels stay square and the art is
compact. The 256-color palette renders on nearly any terminal (tmux, SSH,
Windows Terminal, macOS, Linux). It auto-skips when output isn't a color TTY,
and never raises: a decorative banner must not break the tool.
"""

from __future__ import annotations

import os
import sys

# char -> (hex for the SVG, xterm-256 index for the terminal); "." = empty
_PAL = {
    "A": ("#eceef5", 255),  # hair light
    "a": ("#a8acc0", 145),  # hair mid
    "q": ("#5d6078",  60),  # hair dark
    "F": ("#d98c5f", 173),  # skin
    "L": ("#f7c08f", 216),  # skin lit
    "S": ("#9a5436", 130),  # skin shadow
    "D": ("#5a2a1a",  52),  # skin deep
    "W": ("#ffffff", 231),  # white
    "K": ("#120c18", 233),  # pupil
    "G": ("#ffc93c", 221),  # gold circlet
    "e": ("#3fe6ff",  81),  # gem
    "R": ("#a3162e", 124),  # robe
    "T": ("#d8324c", 161),  # robe light
    "d": ("#52091a",  52),  # robe shadow
    "Z": ("#eaa62a", 178),  # gold trim
    "C": ("#0c3a8e",  24),  # ball rim
    "O": ("#2583f5",  33),  # ball
    "o": ("#134fb8",  25),  # ball shadow
    "Q": ("#7fd0ff", 117),  # ball glow
    "U": ("#e2f9ff", 195),  # ball core
    "X": ("#ffb347", 215),  # spark
    "M": ("#6b3d1c",  94),  # table
    "m": ("#2e1a0c",  52),  # table shadow
    "N": ("#a8e4ff", 153),  # table glow
    "v": ("#173a78",  24),  # mist
    "V": ("#2e62b8",  25),  # mist light
    "x": ("#ff7a2a", 208),  # ember
    "y": ("#ffd060", 221),  # ember light
    "k": ("#07040b", 232),  # outline
}
_BG = "#07050d"  # SVG background

_ART = """\
................................................
.......x.........k..kk.kk.kk..k.................
...............kkakkAAkAAkAAkkakk...............
..............kakaAAAAaAAaAAAAakak......y.......
...VV........kaaAAAaAAAAAAAAaAAAaak........VV...
..V..v......kaAAaAAAAaAAAAaAAAAaAAak......v..V..
..V.v......kaAAAAaAAAAAAAAAAAAaAAAAak......v.V..
...vv......kaAAaASFFFFFFFFFFFFSAaAAak......vv...
..........kqaAAaAGGGGGGeeGGGGGGAaAAaqk..........
..........kaAAaAASFFFFFeeFFFFFSAAaAAak......x...
.........kqaAaAAaSAAAAaFFaAAAASaAAaAaqk.........
..Vv.....kaAAaAAaSFWKKaLLaKKWFSaAAaAAak.....vV..
.V..v....kaAaAAaASFSSSFLLFSSSFSAaAAaAak....v..V.
..vv.....kqaAAaAaSFFFFSLLSFFFFSaAaAAaqk.....vv..
........kqaAaAAaASFFFFDLLDFFFFSAaAAaAaqk........
...y..kkdaAAaAaAqSFaAAAAAAAAaFSqAaAaAAadkk......
.....kdRRqaAaAAaaAAaAAAAAAAAaAAaaAAaAaqRRdk.....
....kdRRTaAaAAqZaAaAAaAAAAaAAaAaZqAAaAaTRRdk....
...kdTRTRqaAaAqZqaAAaAAaaAAaAAaqZqAaAaqRTRTdk...
..VkdTRTRRaAaqRRZaAaAAaAAaAAaAaZRRqaAaRRTRTdkV..
.VkdRTTRRRqFARRRZqaAaCCCCCCaAaqZRRRAFqRRRTTRdkV.
..kdRTRRRFRLqFdddZaCCCOOOOCCCaZdddFqLRFRRRTRdk..
..kdRTRRRFRLRLdddZCCOWOOOWOOCCZdddLRLRFRRRTRdk..
..kdRTRFRFRLRLdddCCWWWXOOOOOOCCdddLRLRFRFRTRdk..
..kdRTRFRFRFRLdddCOWWOOOOOOOWOCdddLRFRFRFRTRdk..
..kdRTRFFFFFFFLdCCWWOOOOOOOOOOCCdLFFFFFFFRTRdk..
..kdRTRSFFFFFLLdCOOOOOOWOOOOOOOCdLLFFFFFSRTRdk..
.xkdRTRSFFFFFLLLCOOOOOOQQOOXOOOCLLLFFFFFSRTRdk..
..kdRRRRSFFFFLLdCOOOOOQQQQOOOOOCdLLFFFFSRRRRdk..
..kdRRRRRSSFFFddCOOOOQQUUQQOoooCddFFFSSRRRRRdk..
..kdRRRRRDSSFdddCCOOWQQUUQQOooCCdddFSSDRRRRRdk..
..kdRRRZZZZZZZZddCOOOQQUUQQoWoCddZZZZZZZZRRRdk..
.vkdRRRRRRRRRddddCCOOOQQQQoooCCddddRRRRRRRRRdkv.
..kdRRRdRRRRRdddddCCOOOOWoooCCdddddRRRRRdRRRdk..
..kdRRRRdRRRRddddddCCCOOOoCCCddddddRRRRdRRRRdk..
..kdZZZZZZZZZZZddddddCCCCCCddddddZZZZZZZZZZZdk..
..MMMMMMMMMMMMMMMNNNNNNNNNNNNNNMMMMMMMMMMMMMMM..
..mmmmmmmmmmmmmmmmmNNNNNNNNNNmmmmmmmmmmmmmmmmm..
....mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm....
................................................"""

_TAGLINE = "s u b s e e r   -   sees the subs you don't"

_IDX = {ch: idx for ch, (_hex, idx) in _PAL.items()}


def render() -> str:
    """256-color half-block pixel art of the seer."""
    rows = _ART.splitlines()
    w = max(len(r) for r in rows)
    rows = [r.ljust(w, ".") for r in rows]
    if len(rows) % 2:
        rows.append("." * w)
    out = []
    for y in range(0, len(rows), 2):
        line = []
        for x in range(w):
            top = _IDX.get(rows[y][x])
            bot = _IDX.get(rows[y + 1][x])
            if top is not None and bot is not None:
                line.append(f"\x1b[38;5;{top};48;5;{bot}m▀\x1b[0m")
            elif top is not None:
                line.append(f"\x1b[38;5;{top}m▀\x1b[0m")
            elif bot is not None:
                line.append(f"\x1b[38;5;{bot}m▄\x1b[0m")
            else:
                line.append(" ")
        out.append("".join(line).rstrip())
    return "\n".join(out)


def _emit(text: str, stream) -> None:
    """Write UTF-8 directly (block glyphs aren't in Windows' cp1252); never raise."""
    try:
        buf = getattr(stream, "buffer", None)
        if buf is not None:
            buf.write(text.encode("utf-8"))
            buf.flush()
        else:
            stream.write(text)
    except Exception:
        pass  # a decorative banner must never break the tool


def show(stream=None) -> None:
    """Print the pixel seer to ``stream`` (default stderr) on a color TTY.

    Skips when piped/redirected, NO_COLOR is set, or TERM=dumb -- so it never
    pollutes `--out -` pipes or logs.
    """
    stream = stream or sys.stderr
    if (os.environ.get("NO_COLOR")
            or os.environ.get("TERM") == "dumb"
            or not getattr(stream, "isatty", lambda: False)()):
        return
    _emit(render() + "\n  " + _TAGLINE + "\n\n", stream)


if __name__ == "__main__":
    _emit(render() + "\n", sys.stdout)

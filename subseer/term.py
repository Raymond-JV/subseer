"""Terminal output helpers: color when it's a TTY, plain text otherwise.

Errors/warnings get loud ERROR:/WARNING: tags on stderr; ``paint`` colors the
progress lines. NO_COLOR and TERM=dumb turn all color off.
"""

from __future__ import annotations

import os
import sys

RED, YELLOW, GREEN, CYAN, GRAY = "1;31", "1;33", "1;32", "1;38;5;81", "38;5;245"


def color_ok(stream) -> bool:
    return (getattr(stream, "isatty", lambda: False)() and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM") != "dumb")


def paint(text: str, style: str, stream=None) -> str:
    """``text`` in ``style`` (an SGR code like CYAN) if ``stream`` shows color."""
    stream = stream if stream is not None else sys.stdout
    return f"\x1b[{style}m{text}\x1b[0m" if color_ok(stream) else text


def _say(level: str, style: str, msg: str) -> None:
    print(f"{paint(level + ':', style, sys.stderr)} {msg}", file=sys.stderr)


def error(msg: str) -> None:
    _say("ERROR", RED, msg)


def warn(msg: str) -> None:
    _say("WARNING", YELLOW, msg)

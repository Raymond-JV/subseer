"""Loud, consistent error/warning lines on stderr (bold color on a TTY, plain otherwise)."""

from __future__ import annotations

import os
import sys

_STYLES = {"ERROR": "\x1b[1;31m", "WARNING": "\x1b[1;33m"}  # bold red / bold yellow


def _say(level: str, msg: str) -> None:
    stream = sys.stderr
    tag = f"{level}:"
    if (getattr(stream, "isatty", lambda: False)() and not os.environ.get("NO_COLOR")
            and os.environ.get("TERM") != "dumb"):
        tag = f"{_STYLES[level]}{tag}\x1b[0m"
    print(f"{tag} {msg}", file=stream)


def error(msg: str) -> None:
    _say("ERROR", msg)


def warn(msg: str) -> None:
    _say("WARNING", msg)

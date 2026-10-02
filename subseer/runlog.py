"""Automatic per-run records for debugging, appended to ~/.subseer/logs.jsonl.

One JSON object per line, one line per run (read with ``jq . ~/.subseer/logs.jsonl``).
Each record captures what a run did -- command, input, backend, the templates it
learned (with values), per-generator counts and where the output went -- so a
surprising result can be traced back later. Override the directory with
$SUBSEER_LOG_DIR. Past ~10 MB the file rolls over to logs.jsonl.1 (one old file
kept). Writing a record never raises: logging must not break a run.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import term


_MAX_BYTES = 10 * 1024 * 1024


def log_path() -> Path:
    env = os.environ.get("SUBSEER_LOG_DIR")
    return (Path(env).expanduser() if env else Path.home() / ".subseer") / "logs.jsonl"


class RunLog:
    def __init__(self, argv: list[str], version: str):
        self._t0 = time.monotonic()
        self.record: dict = {
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "subseer_version": version,
            "command": ["subseer", *argv],
            "cwd": os.getcwd(),
            "warnings": [],
        }

    def set(self, **fields) -> None:
        self.record.update(fields)

    def warn(self, message: str) -> None:
        self.record["warnings"].append(message)

    def write(self) -> None:
        """Append the record as one JSON line; on failure, warn and carry on."""
        try:
            self.record["duration_s"] = round(time.monotonic() - self._t0, 2)
            path = log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > _MAX_BYTES:
                path.replace(path.with_name(path.name + ".1"))
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(self.record, default=str) + "\n")
        except Exception as e:  # never let logging break the tool
            term.warn(f"could not write run log ({e})")

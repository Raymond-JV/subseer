"""Automatic per-run records for debugging: one JSON file per run under ~/.subseer/runs/.

Each record captures what a run did -- command, input, backend, the templates it
learned (with values), per-generator counts and where the output went -- so a
surprising result can be traced back later. Override the directory with
$SUBSEER_LOG_DIR. Writing a record never raises: logging must not break a run.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def log_dir() -> Path:
    env = os.environ.get("SUBSEER_LOG_DIR")
    return Path(env).expanduser() if env else Path.home() / ".subseer" / "runs"


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

    @property
    def started(self) -> bool:
        """True once the run got past input validation (worth recording)."""
        return "input" in self.record

    def write(self) -> Path | None:
        """Write the record to the log dir; returns its path, or None on failure."""
        try:
            self.record["duration_s"] = round(time.monotonic() - self._t0, 2)
            d = log_dir()
            d.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            label = re.sub(r"[^A-Za-z0-9._-]+", "_", self.record.get("input", {}).get("label", "run"))
            path = d / f"{stamp}_{label[:40] or 'run'}.json"
            n = 2
            while path.exists():  # two runs in the same second
                path = d / f"{stamp}_{label[:40] or 'run'}-{n}.json"
                n += 1
            path.write_text(json.dumps(self.record, indent=2, default=str) + "\n", encoding="utf-8")
            return path
        except Exception as e:  # never let logging break the tool
            print(f"warning: could not write run log ({e})", file=sys.stderr)
            return None

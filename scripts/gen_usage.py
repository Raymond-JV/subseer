"""Regenerate the README's Usage block from `subseer -h` (80 columns, for stable wrapping).

Run after changing any flag or help text:  python scripts/gen_usage.py
tests/test_cli.py fails if the README block and the real help output drift apart.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
START, END = "<!-- usage:start -->", "<!-- usage:end -->"


def help_text() -> str:
    os.environ["COLUMNS"] = "80"  # argparse wraps to the terminal width
    sys.path.insert(0, str(ROOT))
    from subseer.cli import build_parser

    return build_parser().format_help().rstrip() + "\n"


def usage_block() -> str:
    return f"{START}\n```\n{help_text()}```\n{END}"


def readme_block(text: str) -> str | None:
    m = re.search(re.escape(START) + r".*?" + re.escape(END), text, re.S)
    return m.group(0) if m else None


def main() -> None:
    text = README.read_text(encoding="utf-8")
    old = readme_block(text)
    if old is None:
        sys.exit(f"{README.name}: no {START} ... {END} block found")
    README.write_text(text.replace(old, usage_block()), encoding="utf-8")
    print(f"Updated the Usage block in {README}")


if __name__ == "__main__":
    main()

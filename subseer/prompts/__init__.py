"""LLM prompts, kept as plain text next to this file.

System prompts: enrich.txt, predict.txt. User-message templates (filled with
``render``): enrich_templates.txt, enrich_residual.txt, predict_user.txt,
predict_covered.txt. Templates use ``$name`` placeholders, so the ``{s1}`` slots in
hostnames need no escaping.
"""

from __future__ import annotations

import importlib.resources as ir
from string import Template


def load(name: str) -> str:
    """The prompt in ``<name>.txt``, minus the leading '#' comment lines."""
    text = ir.files(__name__).joinpath(f"{name}.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def render(name: str, **values) -> str:
    """``<name>.txt`` with its ``$placeholders`` filled; a missing value is an error."""
    return Template(load(name)).substitute({k: str(v) for k, v in values.items()})

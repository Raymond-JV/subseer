"""Glue: load subs, load/expand themes, write outputs."""

from __future__ import annotations

import json
from pathlib import Path

from .expand import cardinality, expand_all
from .models import Theme


def load_subdomains(path: str | Path) -> list[str]:
    """Read a subdomain list: one per line, lowercased, deduped, comments/blanks dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip().lower()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def load_themes(path: str | Path) -> list[Theme]:
    """Load themes from a JSON file (no API call).

    Accepts either a bare list of theme objects (as written to patterns.json) or
    a ``{"themes": [...]}`` wrapper. Any extra ``cardinality`` field is ignored.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "themes" in data:
        data = data["themes"]
    if not isinstance(data, list):
        raise SystemExit(
            f'{path}: expected a JSON list of themes or {{"themes": [...]}}.'
        )
    themes: list[Theme] = []
    for i, d in enumerate(data, 1):
        if not isinstance(d, dict):
            raise SystemExit(f"{path}: theme #{i} is not an object.")
        d = {k: v for k, v in d.items() if k != "cardinality"}
        try:
            themes.append(Theme(**d))
        except Exception as e:  # pydantic ValidationError or bad shape
            raise SystemExit(f"{path}: theme #{i} is invalid: {e}")
    return themes


def theme_to_dict(theme: Theme) -> dict:
    d = theme.model_dump()
    d["cardinality"] = cardinality(theme)
    return d


def write_patterns(themes: list[Theme], patterns_path: str | Path) -> None:
    """Write the theme specs as JSON (opt-in; caller decides whether to call)."""
    Path(patterns_path).write_text(
        json.dumps([theme_to_dict(t) for t in themes], indent=2),
        encoding="utf-8",
    )


def expand_themes(
    themes: list[Theme],
    known: set[str],
    per_theme_cap: int | None,
    max_candidates: int | None,
) -> list[str]:
    return expand_all(themes, known, per_theme_cap=per_theme_cap, max_candidates=max_candidates)

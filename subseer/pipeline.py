"""Glue: load subs, serialize/expand themes."""

from __future__ import annotations

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


def theme_to_dict(theme: Theme) -> dict:
    d = theme.model_dump()
    d["cardinality"] = cardinality(theme)
    return d


def expand_themes(
    themes: list[Theme],
    known: set[str],
    per_theme_cap: int | None,
    max_candidates: int | None,
) -> list[str]:
    return expand_all(themes, known, per_theme_cap=per_theme_cap, max_candidates=max_candidates)

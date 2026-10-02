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
    """A pattern as the run log shows it: only the fields that apply to it."""
    slots = []
    for s in theme.slots:
        d = {"name": s.name}
        if s.label:
            d["label"] = s.label
        if s.meaning:
            d["meaning"] = s.meaning
        if s.kind == "enum":
            d["values"] = s.values
        else:
            d.update(range=[s.min, s.max], **({"pad": s.pad} if s.pad else {}),
                     **({"step": s.step} if s.step != 1 else {}))
        slots.append(d)
    out = {"template": theme.template}
    if theme.name and theme.name != theme.template:  # mined patterns are named by template
        out["label"] = theme.name
    out.update(description=theme.description, slots=slots, cardinality=cardinality(theme))
    if theme.evidence:
        out["evidence"] = theme.evidence
    if theme.kind == "enumerate":
        out["candidates"] = theme.candidates
    return out


def expand_themes(
    themes: list[Theme],
    known: set[str],
    per_theme_cap: int | None,
    max_candidates: int | None,
) -> list[str]:
    return expand_all(themes, known, per_theme_cap=per_theme_cap, max_candidates=max_candidates)

"""Pure-Python review helpers — no API, no pydantic, so they're trivially testable.

Two cheap, free signals against a theme:
  - ``missing_evidence``: cited subdomains that aren't actually in the input
    (a strong fabrication tell).
  - ``filter_themes``: drop themes the critic marked unsupported, or whose cited
    evidence is entirely absent from the input.

Functions use plain attribute access so any object with the right fields works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Theme, Verdict


def cumulative_names(theme_lists) -> list[str]:
    """Deduped (case-insensitive) theme names across a list of theme lists.

    Used to tell each discovery pass which themes earlier passes already found.
    """
    seen: set[str] = set()
    names: list[str] = []
    for tl in theme_lists:
        for t in tl:
            key = t.name.strip().lower()
            if key not in seen:
                seen.add(key)
                names.append(t.name)
    return names


def missing_evidence(themes, known: set[str]) -> dict[str, list[str]]:
    """Map theme name -> cited evidence entries not present in the input list."""
    out: dict[str, list[str]] = {}
    for t in themes:
        miss = [e for e in t.evidence if e.strip().lower() not in known]
        if miss:
            out[t.name] = miss
    return out


def apply_refine(themes, judgments):
    """Filter, rename, and rank mined templates from the refiner's judgments.

    Drops templates the refiner marked ``keep=False``; renames the rest to the
    refiner's label; sorts by priority (5 first). Templates with no matching
    judgment are kept at a neutral priority.
    """
    by_template = {j.template: j for j in judgments}
    ranked = []
    for t in themes:
        j = by_template.get(t.template)
        if j is None:
            ranked.append((3, t))  # unjudged -> neutral, keep
            continue
        if not j.keep:
            continue
        try:
            t.name = j.name
        except Exception:
            pass
        ranked.append((j.priority, t))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [t for _, t in ranked]


def filter_themes(themes, verdicts, known: set[str] | None = None):
    """Return ``(kept, dropped)`` where dropped is a list of ``(theme, reason)``.

    A theme is dropped if the critic marked it ``supported=False``, or — when
    ``known`` is provided — if *every* piece of evidence it cited is absent from
    the input (fully fabricated), even if the critic missed it.
    """
    verdict_by = {v.name.strip().lower(): v for v in verdicts}
    miss = missing_evidence(themes, known) if known is not None else {}

    kept = []
    dropped = []
    for t in themes:
        v = verdict_by.get(t.name.strip().lower())
        fully_fabricated = (
            known is not None
            and bool(t.evidence)
            and len(miss.get(t.name, [])) == len(t.evidence)
        )
        if v is not None and not v.supported:
            dropped.append((t, v.reason))
        elif fully_fabricated:
            dropped.append((t, "all cited evidence is absent from the input list"))
        else:
            kept.append(t)
    return kept, dropped

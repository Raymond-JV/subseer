"""Pure-Python expansion of discovered themes into candidate subdomains.

This module contains zero "intelligence" — it never decides what a pattern is.
It mechanically expands the specs the LLM authored. Functions operate on plain
attribute access (``slot.kind``, ``theme.template`` ...) so they can be tested
with any object that has those attributes, with no Anthropic/Pydantic import.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:  # avoid a hard dependency at runtime / in tests
    from .models import Slot, Theme


def slot_values(slot: "Slot", cap: int | None = None) -> list[str]:
    """Concrete values a slot expands to, in order.

    ``cap`` stops after that many values — avoids materializing a huge numeric
    range (e.g. CDN node numbering) into memory when only a preview (the enrich
    prompt) or a capped expansion is needed. Use ``slot_size`` for the full count.
    """
    if slot.kind == "enum":
        vals = list(slot.values)
        return vals[:cap] if cap is not None else vals
    # range
    step = slot.step or 1
    if step <= 0 or slot.max < slot.min:
        return []
    out = []
    for i in range(slot.min, slot.max + 1, step):
        out.append(str(i).zfill(slot.pad) if slot.pad else str(i))
        if cap is not None and len(out) >= cap:
            break
    return out


def slot_size(slot: "Slot") -> int:
    """Number of values a slot yields, without materializing them."""
    if slot.kind == "enum":
        return len(slot.values)
    step = slot.step or 1
    if step <= 0 or slot.max < slot.min:
        return 0
    return (slot.max - slot.min) // step + 1


def cardinality(theme: "Theme") -> int:
    """How many candidates a theme implies — computed from the spec, no expansion."""
    if theme.kind == "enumerate":
        return len(theme.candidates)
    if not theme.slots:
        return 1 if theme.template else 0
    n = 1
    for s in theme.slots:
        n *= max(slot_size(s), 0)
    return n


def expand(theme: "Theme", limit: int | None = None) -> Iterator[str]:
    """Yield candidate strings for one theme, up to ``limit`` (None = all)."""
    if theme.kind == "enumerate":
        for i, c in enumerate(theme.candidates):
            if limit is not None and i >= limit:
                return
            yield c
        return

    # template
    names = [s.name for s in theme.slots]
    # cap each pool to the expansion limit: itertools.product's first `limit` combos
    # never need more than `limit` values from any one slot, so this preserves the
    # output while never materializing a huge range in full.
    pools = [slot_values(s, cap=limit) for s in theme.slots]
    count = 0
    for combo in itertools.product(*pools):
        if limit is not None and count >= limit:
            return
        out = theme.template
        for name, val in zip(names, combo):
            out = out.replace("{" + name + "}", val)
        count += 1
        yield out


def expand_all(
    themes: Iterable["Theme"],
    known: set[str],
    per_theme_cap: int | None = None,
    max_candidates: int | None = None,
) -> list[str]:
    """Expand every theme, drop names already in ``known``, dedupe, and cap.

    Generation order is preserved (the model orders slot values by likelihood),
    so truncating at ``max_candidates`` keeps the highest-confidence guesses.
    """
    seen: set[str] = set()
    results: list[str] = []
    for theme in themes:
        for name in expand(theme, limit=per_theme_cap):
            key = name.lower()
            if key in known or key in seen:
                continue
            seen.add(key)
            results.append(name)
            if max_candidates is not None and len(results) >= max_candidates:
                return results
    return results

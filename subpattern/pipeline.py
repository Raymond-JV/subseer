"""Glue: load -> discover (single or map-reduce) -> preview -> expand -> write."""

from __future__ import annotations

import json
from pathlib import Path

from .expand import cardinality, expand_all
from .models import Theme
from .review import cumulative_names


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


def _discover_once(
    subdomains: list[str],
    model: str,
    chunk_size: int,
    client,
    prior_names: list[str],
) -> list[Theme]:
    """One full discovery pass: a single call, or map-reduce over chunks."""
    from .discover import discover_chunk, merge_themes

    prior = prior_names or None
    if len(subdomains) <= chunk_size:
        return discover_chunk(subdomains, model=model, client=client, prior_theme_names=prior)

    chunks = [subdomains[i : i + chunk_size] for i in range(0, len(subdomains), chunk_size)]
    results = []
    for idx, chunk in enumerate(chunks, 1):
        print(f"    chunk {idx}/{len(chunks)} ({len(chunk)} subdomains)...")
        results.append(
            discover_chunk(chunk, model=model, client=client, prior_theme_names=prior)
        )
    return merge_themes(results)


def load_themes(path: str | Path) -> list[Theme]:
    """Load discovered themes from a JSON file (no API call).

    Accepts either a bare list of theme objects (as written to patterns.json) or
    a ``{"themes": [...]}`` wrapper (as a chat model is asked to return). Any
    extra ``cardinality`` field from a prior patterns.json is ignored.
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


def run_discovery(
    subdomains: list[str],
    model: str = "claude-opus-4-8",
    chunk_size: int = 20000,
    runs: int = 1,
    client=None,
) -> list[Theme]:
    """Discover naming themes, optionally over several deepening passes.

    Each pass is a single call (or map-reduce over chunks, if the list is large
    enough that one call wouldn't reliably reason over all of it). Passes after
    the first are told which themes were already found, so they hunt for what the
    obvious patterns overshadow instead of repeating. All passes are merged.
    """
    from .discover import merge_themes

    runs = max(1, runs)
    all_runs: list[list[Theme]] = []
    for r in range(1, runs + 1):
        prior_names = cumulative_names(all_runs)
        if runs > 1:
            extra = f" (deepening past {len(prior_names)} theme(s))" if prior_names else ""
            print(f"  discovery pass {r}/{runs}{extra}")
        all_runs.append(_discover_once(subdomains, model, chunk_size, client, prior_names))
    return merge_themes(all_runs)


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


def write_outputs(
    themes: list[Theme],
    candidates: list[str],
    patterns_path: str | Path,
    candidates_path: str | Path,
) -> None:
    write_patterns(themes, patterns_path)
    Path(candidates_path).write_text("\n".join(candidates) + "\n", encoding="utf-8")


def expand_themes(
    themes: list[Theme],
    known: set[str],
    per_theme_cap: int | None,
    max_candidates: int | None,
) -> list[str]:
    return expand_all(themes, known, per_theme_cap=per_theme_cap, max_candidates=max_candidates)

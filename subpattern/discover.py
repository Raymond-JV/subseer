"""The LLM discovery step: subdomains in -> themes (with specs) out.

This is the only part that calls the Anthropic API. ``anthropic`` is imported
lazily so the rest of the package (and the expander tests) work without it.
"""

from __future__ import annotations

from . import pricing
from .models import Discovery, Slot, Theme

SYSTEM_PROMPT = """\
You are an expert at analyzing DNS subdomain naming conventions for AUTHORIZED \
security reconnaissance (bug-bounty and penetration-testing engagements). You are \
given a list of subdomains already discovered for a single target organization. \
Your job is to discover the naming conventions and themes the organization uses — \
especially subtle, semantic, or thematic patterns that simple frequency analysis \
would miss: hosts named after a mythology or theme, internal project codenames, \
team or product prefixes, environment/region structures, versioning schemes, \
deprecation patterns, and so on.

For every theme you identify:
- Give it a short name and describe the rule.
- Cite evidence: actual subdomains FROM THE PROVIDED LIST that exhibit the pattern. \
Never propose a theme you cannot support with real entries from the list.
- Classify it as one of two kinds:
  - "enumerate": a finite, knowledge-driven set where extending it requires world \
knowledge (other gods in the same pantheon, other cities, other product names). \
Directly list plausible NEW members that are NOT already in the list, in `candidates`.
  - "template": a combinatorial, mechanical pattern. DO NOT list the combinations. \
Provide a `template` string with {placeholder} slots and define each slot as a \
`range` (numeric, with optional zero `pad` and `step`) or an `enum` (explicit \
values). Order enum values and ranges by how likely they are for THIS organization. \
Code will expand the template into the full list.

Prefer "template" whenever a pattern is mechanical and high-cardinality; reserve \
"enumerate" for sets that genuinely require world knowledge to extend. Slot `name` \
values must match the {placeholders} in `template`. Be precise and evidence-driven; \
do not invent conventions that the data does not support."""

USER_TEMPLATE = """\
Here are the {count} known subdomains for the target. Discover the naming \
conventions and return them in the required structured format.

{listing}"""

PRIOR_BLOCK = """\

A previous analysis pass already identified these conventions:
{names}

Do NOT simply restate them. Find ADDITIONAL conventions — especially subtler or \
rarer ones that the obvious patterns overshadow. You may also return one of the \
above if you have found stronger evidence or a more precise rule for it."""

# Used only by --print-prompt: when there's no structured-output enforcement
# (e.g. pasting into claude.ai), the model needs the target shape spelled out.
SCHEMA_HINT = """\
Return ONLY JSON of this exact shape — no prose, no markdown fences:

{
  "themes": [
    {
      "name": "Greek-god codenames",
      "description": "Hosts named after Olympian gods.",
      "evidence": ["zeus.example.com", "apollo.example.com"],
      "kind": "enumerate",
      "candidates": ["hera.example.com", "ares.example.com"]
    },
    {
      "name": "numbered web hosts",
      "description": "Service name followed by a zero-padded counter.",
      "evidence": ["web01.example.com", "node02.example.com"],
      "kind": "template",
      "template": "{name}{n}.example.com",
      "slots": [
        {"name": "name", "kind": "enum", "values": ["web", "node", "app"]},
        {"name": "n", "kind": "range", "min": 1, "max": 100, "pad": 2}
      ]
    }
  ]
}

Rules: "kind" is "enumerate" or "template". enumerate themes use "candidates";
template themes use "template" + "slots". Slot "kind" is "enum" (with "values")
or "range" (with "min","max", optional "pad" and "step"). Each slot "name" must
match a {placeholder} in "template". Only cite evidence that is actually in the list."""


def build_prompt(subdomains: list[str], prior_theme_names: list[str] | None = None):
    """Assemble the (system, user) discovery prompt. Shared by the live call and --print-prompt."""
    listing = "\n".join(subdomains)
    user = USER_TEMPLATE.format(count=len(subdomains), listing=listing)
    if prior_theme_names:
        user += PRIOR_BLOCK.format(names="\n".join(f"- {n}" for n in prior_theme_names))
    return SYSTEM_PROMPT, user


def _make_client(api_key: str | None = None):
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "The 'anthropic' package is required for discovery. "
            "Install it with:  pip install -e .\n"
            f"(import error: {e})"
        )
    try:
        return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Could not create the Anthropic client. Set ANTHROPIC_API_KEY in your "
            f"environment (see .env.example).\n(error: {e})"
        )


def discover_chunk(
    subdomains: list[str],
    model: str = "claude-opus-4-8",
    max_tokens: int = 16000,
    client=None,
    prior_theme_names: list[str] | None = None,
) -> list[Theme]:
    """Run one discovery call over a list of subdomains.

    ``prior_theme_names`` (from earlier passes) is fed back so the model deepens
    rather than repeats — sequential multi-run discovery.
    """
    client = client or _make_client()
    system, user = build_prompt(subdomains, prior_theme_names)

    try:
        resp = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},  # reasoning helps surface subtle themes
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=Discovery,
        )
    except AttributeError as e:  # pragma: no cover
        raise SystemExit(
            "Your 'anthropic' version is too old for messages.parse(). "
            f"Upgrade with:  pip install -U anthropic\n(error: {e})"
        )

    pricing.report(resp, model, "discovery")

    parsed = resp.parsed_output
    if parsed is None:  # refusal or truncation
        reason = getattr(resp, "stop_reason", "unknown")
        raise SystemExit(f"Model did not return structured output (stop_reason={reason}).")
    return parsed.themes


def merge_themes(theme_lists: list[list[Theme]]) -> list[Theme]:
    """Combine themes from several chunks (map-reduce), deduping by name.

    Same-named themes have their evidence / candidates / enum slot-values unioned.
    A future enhancement is an LLM reduce pass; this code-side merge is enough to
    keep the pipeline runnable and cost-free.
    """
    by_name: dict[str, Theme] = {}
    for themes in theme_lists:
        for t in themes:
            key = t.name.strip().lower()
            base = by_name.get(key)
            if base is None:
                by_name[key] = t.model_copy(deep=True)
                continue
            base.evidence = _dedupe(base.evidence + t.evidence)
            base.candidates = _dedupe(base.candidates + t.candidates)
            if base.kind == "template" and t.kind == "template":
                slots_by_name = {s.name: s for s in base.slots}
                for s in t.slots:
                    existing = slots_by_name.get(s.name)
                    if existing and existing.kind == "enum" and s.kind == "enum":
                        existing.values = _dedupe(existing.values + s.values)
    return list(by_name.values())


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


__all__ = ["discover_chunk", "merge_themes", "Slot", "Theme", "Discovery"]

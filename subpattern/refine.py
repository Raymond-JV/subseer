"""LLM cleanup pass over auto-detected templates.

The miner is exhaustive but noisy: it emits low-value templates (random-id
slots, patterns that only regenerate known hosts) and uses raw template strings
as names. This pass sends the compact template list (NOT the host list) to a
model and gets back, per template: a human name, keep/drop, and a priority — so
the output is a clean, ranked set. Cheap, because it reasons over ~100 templates,
not millions of hosts.
"""

from __future__ import annotations

from . import pricing
from .discover import _make_client
from .models import RefineReport

SYSTEM_PROMPT = """\
You are reviewing auto-detected subdomain naming templates for an AUTHORIZED \
security-reconnaissance target (bug bounty / pentest). Each input line is a \
template plus its slot domains. For EACH template, return one judgment:
- name: a short human-readable label for the convention.
- keep: false if the template is noise — slots filled by random hex / opaque IDs, \
or a pattern that would essentially only regenerate already-known hosts or emit \
implausible names. true if it could plausibly surface NEW valid hostnames.
- priority: an integer 1-5, where 5 = most likely to yield new live hosts (clear \
service / environment / region / numbering conventions) and 1 = marginal.
- note: a brief reason (optional).
Echo each template string VERBATIM so it can be matched back. Return one judgment \
per template, no more, no fewer."""


def _slot_str(slot) -> str:
    if slot.kind == "range":
        return f"{slot.name}=range[{slot.min}-{slot.max}]"
    vals = list(slot.values)[:12]
    more = " ..." if len(slot.values) > 12 else ""
    return f"{slot.name}=enum[{', '.join(vals)}{more}]"


def _serialize(themes) -> str:
    lines = []
    for t in themes:
        slots = "; ".join(_slot_str(s) for s in t.slots) if t.slots else "(no slots)"
        lines.append(f"{t.template}  |  {slots}")
    return "\n".join(lines)


def run_refine(
    themes,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 16000,
    client=None,
) -> RefineReport:
    """Ask the model to name / keep / rank each detected template."""
    if not themes:
        return RefineReport(judgments=[])
    client = client or _make_client()
    body = _serialize(themes)
    user = f"Review these {len(themes)} detected templates:\n\n{body}"

    resp = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        output_format=RefineReport,
    )
    pricing.report(resp, model, "refine")
    return resp.parsed_output or RefineReport(judgments=[])


__all__ = ["run_refine"]

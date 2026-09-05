"""Optional critic pass: re-read the discovered themes and flag invented ones.

Open-ended discovery occasionally names a "theme" the data doesn't actually
support. This pass feeds the themes back and asks the model to judge whether the
cited evidence genuinely exhibits each rule, before we spend expansion on it.
"""

from __future__ import annotations

from . import pricing
from .discover import _make_client
from .models import CriticReport, Theme

SYSTEM_PROMPT = """\
You are a skeptical reviewer auditing candidate subdomain naming-convention \
"themes" proposed for a single target organization. For each theme you are given \
its rule and the evidence subdomains cited for it. Decide whether the cited \
evidence GENUINELY exhibits the stated rule.

Mark `supported = false` when a theme is vague, post-hoc, a coincidence, or when \
the cited examples don't actually demonstrate the claimed convention — the \
proposer sometimes invents conventions that the data doesn't justify. Mark \
`supported = true` only when the evidence clearly and specifically supports the \
rule. Return exactly one verdict per theme, echoing the theme `name` verbatim, \
with a one-sentence reason."""

USER_TEMPLATE = "Review these {count} proposed themes:\n\n{body}"


def _serialize(themes: list[Theme]) -> str:
    lines: list[str] = []
    for i, t in enumerate(themes, 1):
        lines.append(f"{i}. name: {t.name}")
        lines.append(f"   kind: {t.kind}")
        lines.append(f"   rule: {t.description}")
        lines.append(f"   evidence: {', '.join(t.evidence)}")
        if t.kind == "template":
            lines.append(f"   template: {t.template}")
            for s in t.slots:
                if s.kind == "enum":
                    vals = ", ".join(s.values[:20])
                    lines.append(f"     slot {s.name} (enum): {vals}")
                else:
                    lines.append(
                        f"     slot {s.name} (range): {s.min}-{s.max} pad={s.pad} step={s.step}"
                    )
        elif t.candidates:
            lines.append(f"   sample candidates: {', '.join(t.candidates[:15])}")
    return "\n".join(lines)


def run_critic(
    themes: list[Theme],
    model: str = "claude-opus-4-8",
    max_tokens: int = 8000,
    client=None,
) -> CriticReport:
    """Ask the model to judge each theme's evidentiary support."""
    if not themes:
        return CriticReport(verdicts=[])
    client = client or _make_client()
    body = _serialize(themes)
    user = USER_TEMPLATE.format(count=len(themes), body=body)

    resp = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
        output_format=CriticReport,
    )
    pricing.report(resp, model, "critic")

    parsed = resp.parsed_output
    if parsed is None:  # refusal/truncation -> treat as "no objections"
        return CriticReport(verdicts=[])
    return parsed


__all__ = ["run_critic"]

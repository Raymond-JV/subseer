"""Live cost reporting from real API usage.

Each Anthropic response carries a ``usage`` object with the actual token counts
(adaptive-thinking tokens are already included in ``output_tokens``), so the
figures printed here are real, not estimates — they only drift if the published
prices below change. Pure stdlib; safe to import anywhere.
"""

from __future__ import annotations

# USD per 1,000,000 tokens: (input, output). Cache writes bill at 1.25x input,
# cache reads at 0.1x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

_session = {
    "calls": 0,
    "input": 0,
    "cache_write": 0,
    "cache_read": 0,
    "output": 0,
    "cost": 0.0,
}


def _rates(model: str) -> tuple[float, float] | None:
    if model in PRICES:
        return PRICES[model]
    for known, rates in PRICES.items():  # tolerate dated/suffixed ids
        if model.startswith(known):
            return rates
    return None


def usage_cost(usage, model: str):
    """Return ``(cost_usd, (input, cache_write, cache_read, output))`` or None.

    None means we have no price for ``model`` (so we can't compute a cost).
    """
    rates = _rates(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    inp = getattr(usage, "input_tokens", 0) or 0
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cost = (
        inp * in_rate
        + cw * in_rate * 1.25
        + cr * in_rate * 0.1
        + out * out_rate
    ) / 1_000_000
    return cost, (inp, cw, cr, out)


def report(resp, model: str, label: str) -> None:
    """Print the cost of one call and add it to the running session total."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    result = usage_cost(usage, model)
    if result is None:
        print(f"  [{label}] (no price on file for {model})")
        return
    cost, (inp, cw, cr, out) = result
    _session["calls"] += 1
    _session["input"] += inp
    _session["cache_write"] += cw
    _session["cache_read"] += cr
    _session["output"] += out
    _session["cost"] += cost
    extra = f", cache_read={cr:,}, cache_write={cw:,}" if (cr or cw) else ""
    print(f"  [{label}] in={inp:,}, out={out:,}{extra}  ~ ${cost:.4f}")


def print_session_summary() -> None:
    s = _session
    if s["calls"] == 0:
        return
    total_in = s["input"] + s["cache_read"] + s["cache_write"]
    print(
        f"\nAPI usage: {s['calls']} call(s), "
        f"{total_in:,} input + {s['output']:,} output tokens  "
        f"~ ${s['cost']:.2f} total"
    )


def reset() -> None:
    for k in _session:
        _session[k] = 0.0 if k == "cost" else 0

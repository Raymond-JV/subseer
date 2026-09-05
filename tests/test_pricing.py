"""Tests for cost math. Stdlib only; run directly or via pytest."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subpattern.pricing import usage_cost


def usage(inp=0, out=0, cw=0, cr=0):
    return NS(
        input_tokens=inp,
        output_tokens=out,
        cache_creation_input_tokens=cw,
        cache_read_input_tokens=cr,
    )


def test_opus_basic_cost():
    cost, parts = usage_cost(usage(inp=80_000, out=5_000), "claude-opus-4-8")
    # 80k * $5/M + 5k * $25/M = 0.40 + 0.125
    assert abs(cost - 0.525) < 1e-9
    assert parts == (80_000, 0, 0, 5_000)  # (input, cache_write, cache_read, output)


def test_haiku_cheaper_than_opus():
    h, _ = usage_cost(usage(inp=80_000, out=5_000), "claude-haiku-4-5")
    o, _ = usage_cost(usage(inp=80_000, out=5_000), "claude-opus-4-8")
    assert h < o


def test_cache_read_discounted_to_10pct():
    cost, _ = usage_cost(usage(cr=100_000), "claude-opus-4-8")
    # 100k * $5/M * 0.1 = 0.05
    assert abs(cost - 0.05) < 1e-9


def test_cache_write_premium_125pct():
    cost, _ = usage_cost(usage(cw=100_000), "claude-opus-4-8")
    # 100k * $5/M * 1.25 = 0.625
    assert abs(cost - 0.625) < 1e-9


def test_unknown_model_returns_none():
    assert usage_cost(usage(inp=1, out=1), "gpt-4") is None


def test_dated_suffix_matches_by_prefix():
    cost, _ = usage_cost(usage(inp=1_000_000), "claude-haiku-4-5-20251001")
    assert abs(cost - 1.0) < 1e-9  # $1/M input


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

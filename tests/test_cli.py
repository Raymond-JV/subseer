"""Tests for the CLI's auto-scaling helpers. Stdlib only."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subpattern.cli import (
    _auto_min_values,
    _auto_runs,
    _auto_sample,
    _resolve_auto,
)


def test_auto_sample_anchored_near_2k():
    assert _auto_sample(500) == 2000
    assert _auto_sample(20000) == 2000
    assert _auto_sample(80000) == 3000
    assert _auto_sample(773890) == 4000


def test_auto_runs_scales_with_size():
    assert _auto_runs(300) == 2
    assert _auto_runs(2000) == 2
    assert _auto_runs(4000) == 5
    assert _auto_runs(9000) == 8
    assert _auto_runs(50000) == 10


def test_auto_min_values_scales_with_size():
    assert _auto_min_values(300) == 2
    assert _auto_min_values(3000) == 3
    assert _auto_min_values(12000) == 4
    assert _auto_min_values(50000) == 5


def test_resolve_auto_int_passthrough():
    assert _resolve_auto("7", 9999, _auto_runs, "local-runs") == 7
    assert _resolve_auto(3, 9999, _auto_min_values, "min-values") == 3


def test_resolve_auto_keyword():
    assert _resolve_auto("auto", 4000, _auto_runs, "local-runs") == 5
    assert _resolve_auto("AUTO", 12000, _auto_min_values, "min-values") == 4


def test_resolve_auto_invalid_exits():
    try:
        _resolve_auto("banana", 100, _auto_runs, "local-runs")
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("expected SystemExit on invalid value")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

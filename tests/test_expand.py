"""Tests for the pure-Python expander.

Runs with plain `python tests/test_expand.py` (stdlib only) or under pytest.
Uses SimpleNamespace stand-ins so it needs neither anthropic nor pydantic.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subseer.expand import cardinality, expand, expand_all, slot_size, slot_values


def slot(name, kind, values=None, mn=0, mx=0, pad=0, step=1):
    return NS(name=name, kind=kind, values=values or [], min=mn, max=mx, pad=pad, step=step)


def template_theme(template, slots, kind="template"):
    return NS(kind=kind, template=template, slots=slots, candidates=[])


def enum_theme(candidates):
    return NS(kind="enumerate", template="", slots=[], candidates=candidates)


def test_range_padding():
    s = slot("n", "range", mn=1, mx=3, pad=2)
    assert slot_values(s) == ["01", "02", "03"]
    assert slot_size(s) == 3


def test_range_step():
    s = slot("n", "range", mn=0, mx=10, step=5)
    assert slot_values(s) == ["0", "5", "10"]
    assert slot_size(s) == 3


def test_range_no_pad():
    s = slot("n", "range", mn=8, mx=10)
    assert slot_values(s) == ["8", "9", "10"]


def test_enum_size():
    s = slot("env", "enum", values=["dev", "stg", "prod"])
    assert slot_values(s) == ["dev", "stg", "prod"]
    assert slot_size(s) == 3


def test_cardinality_product():
    t = template_theme(
        "{name}{n}.example.com",
        [
            slot("name", "enum", values=["web", "node", "app"]),
            slot("n", "range", mn=1, mx=1000, pad=2),
        ],
    )
    assert cardinality(t) == 3 * 1000


def test_cardinality_enumerate():
    assert cardinality(enum_theme(["a", "b", "c"])) == 3


def test_expand_template():
    t = template_theme(
        "{name}{n}.example.com",
        [
            slot("name", "enum", values=["web", "db"]),
            slot("n", "range", mn=1, mx=2, pad=2),
        ],
    )
    got = list(expand(t))
    assert got == [
        "web01.example.com",
        "web02.example.com",
        "db01.example.com",
        "db02.example.com",
    ]


def test_expand_template_no_slots():
    t = template_theme("vpn.example.com", [])
    assert list(expand(t)) == ["vpn.example.com"]
    assert cardinality(t) == 1


def test_expand_limit():
    t = template_theme(
        "{n}.example.com",
        [slot("n", "range", mn=1, mx=100)],
    )
    assert len(list(expand(t, limit=5))) == 5


def test_expand_enumerate():
    assert list(expand(enum_theme(["zeus", "hera"]))) == ["zeus", "hera"]


def test_expand_all_dedupes_and_drops_known():
    t1 = enum_theme(["zeus", "hera", "web01.example.com"])
    t2 = enum_theme(["hera", "apollo"])  # 'hera' duplicates across themes
    known = {"web01.example.com"}  # already known -> dropped
    out = expand_all([t1, t2], known)
    assert out == ["zeus", "hera", "apollo"]


def test_expand_all_global_cap():
    t = template_theme("{n}.example.com", [slot("n", "range", mn=1, mx=1000)])
    out = expand_all([t], known=set(), max_candidates=10)
    assert len(out) == 10


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

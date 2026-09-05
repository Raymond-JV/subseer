"""Tests for the pure review helpers. Stdlib only; run directly or via pytest."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subpattern.review import apply_refine, cumulative_names, filter_themes, missing_evidence


def jud(template, name, keep, priority):
    return NS(template=template, name=name, keep=keep, priority=priority, note="")


def named(template):
    return NS(template=template, name=template)


def test_apply_refine_drops_and_ranks():
    themes = [named("a"), named("b"), named("c")]
    judgments = [jud("a", "Alpha", True, 5), jud("b", "Beta", False, 1), jud("c", "Gamma", True, 2)]
    out = apply_refine(themes, judgments)
    assert [t.template for t in out] == ["a", "c"]  # b dropped; a(prio 5) before c(prio 2)
    assert out[0].name == "Alpha"


def test_apply_refine_keeps_unjudged():
    themes = [named("a"), named("z")]
    out = apply_refine(themes, [jud("a", "Alpha", True, 5)])
    assert {t.template for t in out} == {"a", "z"}  # z unjudged but kept


def theme(name, evidence=None):
    return NS(name=name, evidence=evidence or [])


def verdict(name, supported, reason="r"):
    return NS(name=name, supported=supported, reason=reason)


def test_missing_evidence_flags_absent_entries():
    known = {"web01.example.com", "web02.example.com"}
    themes = [theme("numbered", ["web01.example.com", "web99.example.com"])]
    assert missing_evidence(themes, known) == {"numbered": ["web99.example.com"]}


def test_missing_evidence_case_insensitive():
    known = {"web01.example.com"}
    themes = [theme("numbered", ["WEB01.EXAMPLE.COM"])]
    assert missing_evidence(themes, known) == {}


def test_filter_drops_unsupported_verdict():
    themes = [theme("good", ["a"]), theme("bad", ["a"])]
    verdicts = [verdict("good", True), verdict("bad", False, "post-hoc")]
    kept, dropped = filter_themes(themes, verdicts)
    assert [t.name for t in kept] == ["good"]
    assert dropped[0][0].name == "bad" and dropped[0][1] == "post-hoc"


def test_filter_drops_fully_fabricated_even_without_verdict():
    known = {"real.example.com"}
    themes = [theme("ghost", ["fake1.example.com", "fake2.example.com"])]
    kept, dropped = filter_themes(themes, verdicts=[], known=known)
    assert kept == []
    assert "absent from the input" in dropped[0][1]


def test_filter_keeps_theme_with_partial_real_evidence():
    known = {"real.example.com"}
    themes = [theme("mostly", ["real.example.com", "fake.example.com"])]
    kept, dropped = filter_themes(themes, verdicts=[], known=known)
    assert [t.name for t in kept] == ["mostly"]
    assert dropped == []


def test_filter_name_match_is_case_insensitive():
    themes = [theme("Numbered Hosts", ["a"])]
    verdicts = [verdict("numbered hosts", False, "vague")]
    kept, dropped = filter_themes(themes, verdicts)
    assert kept == [] and dropped[0][1] == "vague"


def test_cumulative_names_dedupes_across_runs():
    run1 = [theme("Numbered Hosts"), theme("Greek Gods")]
    run2 = [theme("numbered hosts"), theme("Region Codes")]  # 1 dup, 1 new
    names = cumulative_names([run1, run2])
    assert names == ["Numbered Hosts", "Greek Gods", "Region Codes"]


def test_cumulative_names_empty():
    assert cumulative_names([]) == []


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

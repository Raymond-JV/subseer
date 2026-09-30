"""Tests for the pure parts of --enrich (parsing + conversion). Stdlib only."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subseer.enrich import (
    _merge_theme_data,
    _parse_enrichment,
    build_enrich_prompt,
    enrichment_stats,
    enrichment_to_themes,
)


def test_merge_theme_data_tolerates_malformed_shapes():
    dst = {}
    items = [
        # slots as bare strings (the crash case) -> skipped, template still recorded
        {"template": "{s1}.example.com", "slots": ["memphis", "austin"]},
        # well-formed, plus a values-as-string and a junk slot
        {"template": "api-{s1}.example.com",
         "slots": [{"name": "s1", "values": "dev"}, "junk", {"name": "s2"}]},
        "not-a-dict-theme",  # skipped entirely
    ]
    _merge_theme_data(dst, items)  # must not raise
    assert dst["api-{s1}.example.com"]["slots"]["s1"] == ["dev"]   # string coerced to [dev]
    assert "{s1}.example.com" in dst                                # recorded despite bad slots
from subseer.expand import expand
from subseer.models import Slot, Theme


def _theme(template, slots):
    return Theme(name="t", description="d", evidence=[], kind="template",
                 template=template, slots=slots)


def test_parse_enrichment_valid():
    content = '{"themes": [{"template": "{s1}.example.com", "slots": [{"name": "s1", "values": ["a","b"]}]}]}'
    out = _parse_enrichment(content)
    assert len(out) == 1 and out[0]["template"] == "{s1}.example.com"


def test_parse_enrichment_salvages_truncated():
    # second object is cut off; first complete one must survive
    content = ('{"themes": [\n'
               '{"template": "{s1}.example.com", "slots": [{"name": "s1", "values": ["a"]}]},\n'
               '{"template": "{s2}.foo.example.com", "slots": [{"name": "s2", "values": ["x"')
    out = _parse_enrichment(content)
    assert [t["template"] for t in out] == ["{s1}.example.com"]


def test_enrichment_to_themes_expands_enum():
    data = [{"template": "{s1}.dleague.example.com",
             "slots": [{"name": "s1", "values": ["Memphis", "austin", "memphis", " "]}]}]
    themes = enrichment_to_themes(data, "example.com", mined_templates=set())
    assert len(themes) == 1
    t = themes[0]
    assert t.slots[0].kind == "enum"
    assert t.slots[0].values == ["memphis", "austin"]  # lowered, deduped, blanks dropped
    assert set(expand(t)) == {"memphis.dleague.example.com", "austin.dleague.example.com"}


def test_enrichment_to_themes_sanitizes_dns_labels():
    data = [{"template": "{s1}.dleague.example.com",
             "slots": [{"name": "s1", "values": ["golden state", "new_york", "L.A. Lakers", "boston"]}]}]
    themes = enrichment_to_themes(data, "example.com", set())
    vals = themes[0].slots[0].values
    assert vals == ["goldenstate", "new-york", "lalakers", "boston"]  # spaces removed, _->-, dots dropped


def test_enrichment_to_themes_numeric_becomes_range():
    data = [{"template": "g{s1}.example.com", "slots": [{"name": "s1", "values": ["1", "2", "3"]}]}]
    themes = enrichment_to_themes(data, "example.com", set())
    s = themes[0].slots[0]
    assert s.kind == "range" and s.min == 1 and s.max == 3


def test_enrichment_to_themes_appends_apex():
    data = [{"template": "api-{s1}", "slots": [{"name": "s1", "values": ["dev", "prod"]}]}]
    themes = enrichment_to_themes(data, "example.com", set())
    assert themes[0].template == "api-{s1}.example.com"


def test_enrichment_to_themes_skips_mismatched_slots():
    # template has {s1} and {s2} but only s1 is specified -> dropped
    data = [{"template": "{s1}.{s2}.example.com", "slots": [{"name": "s1", "values": ["a"]}]}]
    assert enrichment_to_themes(data, "example.com", set()) == []


def test_enrichment_to_themes_novel_flag_from_mined_set():
    data = [
        {"template": "{s1}.dleague.example.com", "slots": [{"name": "s1", "values": ["a"]}]},
        {"template": "{s1}.eu.example.com", "slots": [{"name": "s1", "values": ["a"]}]},
    ]
    mined = {"{s1}.dleague.example.com"}
    themes = enrichment_to_themes(data, "example.com", mined)
    by_tmpl = {t.template: t.name for t in themes}
    assert by_tmpl["{s1}.dleague.example.com"] == "enriched"    # in mined set
    assert by_tmpl["{s1}.eu.example.com"] == "discovered"      # not mined -> novel


def test_enrichment_stats_counts_added_values_and_kinds():
    mined = [_theme("{s1}.dleague.example.com",
                    [Slot(name="s1", kind="enum", values=["memphis", "austin"])])]
    enriched = [
        # Job A: same template, 2 old + 3 new values -> +3
        _theme("{s1}.dleague.example.com",
               [Slot(name="s1", kind="enum", values=["memphis", "austin", "boston", "miami", "phoenix"])]),
        # Job B: brand-new template with 4 values -> +4, and discovered++
        _theme("{s1}.eu.example.com",
               [Slot(name="s1", kind="enum", values=["a", "b", "c", "d"])]),
    ]
    stats = enrichment_stats(mined, enriched)
    assert stats == {"enriched": 1, "discovered": 1, "new_values": 7}


def test_build_enrich_prompt_includes_templates_and_residual():
    themes = [_theme("{s1}.dleague.example.com",
                     [Slot(name="s1", kind="enum", values=["memphis", "austin"])])]
    prompt = build_enrich_prompt(themes, ["oddball.example.com"], "example.com")
    assert "{s1}.dleague.example.com" in prompt
    assert "memphis, austin" in prompt
    assert "oddball.example.com" in prompt
    assert "example.com" in prompt


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

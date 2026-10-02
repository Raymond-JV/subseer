"""Tests for the offline pattern miner. Stdlib only."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subseer.mine import (
    covered_hosts,
    detect_apex,
    detect_patterns,
    detok,
    registrable_domain,
    residual_hosts,
    tokenize,
)


def test_tokenize_dot():
    assert tokenize("purchase.lab") == (["purchase", "lab"], ["", "."])


def test_tokenize_digit_boundary():
    assert tokenize("mx10") == (["mx", "10"], ["", ""])


def test_tokenize_mixed():
    toks, seps = tokenize("austin.gleague-dev")
    assert toks == ["austin", "gleague", "dev"]
    assert seps == ["", ".", "-"]


def test_detok_roundtrip():
    for s in ["purchase.lab", "mx10", "austin.gleague-dev", "c4-npv-p1899-r12"]:
        toks, seps = tokenize(s)
        assert detok(toks, seps) == s


def test_detect_apex():
    subs = ["a.example.com", "b.example.com", "c.x.org"]
    assert detect_apex(subs) == "example.com"


def test_registrable_domain_plain_and_ccsld():
    assert registrable_domain("foo.example.com") == "example.com"
    assert registrable_domain("api.dev.example.com") == "example.com"
    assert registrable_domain("foo.example.com.mx") == "example.com.mx"      # not com.mx
    assert registrable_domain("a.b.example.co.uk") == "example.co.uk"  # not co.uk
    assert registrable_domain("example.com") == "example.com"
    assert registrable_domain("localhost") == "localhost"


def test_detect_apex_ccsld():
    subs = ["a.example.com.mx", "b.example.com.mx", "shop.example.com.mx"]
    assert detect_apex(subs) == "example.com.mx"


def _find(themes, template):
    return next((t for t in themes if t["template"] == template), None)


def test_detects_service_env_cross():
    # purchase/audience/lp each in int/lab/qa/stage -> {s}.{env}
    subs = []
    for svc in ["purchase", "audience", "lp"]:
        for env in ["int", "lab", "qa", "stage"]:
            subs.append(f"{svc}.{env}.example.com")
    themes = detect_patterns(subs)
    t = _find(themes, "{s1}.{s2}.example.com")
    assert t is not None, [x["template"] for x in themes]
    svc_slot = next(s for s in t["slots"] if s["name"] == "s1")
    env_slot = next(s for s in t["slots"] if s["name"] == "s2")
    assert set(svc_slot["values"]) == {"purchase", "audience", "lp"}
    assert set(env_slot["values"]) == {"int", "lab", "qa", "stage"}


def test_detects_numeric_range():
    subs = ["mx10.example.com", "mx20.example.com", "mx30.example.com", "mx40.example.com"]
    themes = detect_patterns(subs)
    t = _find(themes, "mx{s1}.example.com")
    assert t is not None, [x["template"] for x in themes]
    slot = t["slots"][0]
    assert slot["kind"] == "range"
    assert slot["min"] == 10 and slot["max"] == 40


def test_skips_bare_flat_hosts():
    # api/www/ads etc. with no shared structure -> no giant "{hole}" enum theme
    subs = ["api.example.com", "www.example.com", "ads.example.com", "cdn.example.com"]
    themes = detect_patterns(subs)
    assert all(t["template"] != "{s1}.example.com" for t in themes)


def test_does_not_merge_unrelated_word_pairs():
    # fans.heat / api.china / m.store share W.W shape but NOT a value-set -> no cross
    subs = ["fans.heat.example.com", "api.china.example.com", "m.store.example.com"]
    themes = detect_patterns(subs)
    assert _find(themes, "{s1}.{s2}.example.com") is None


def test_cross_fills_overlapping_dimensions():
    # teams split across two leagues, NOT all combos present (idaho only in gleague)
    subs = [f"{t}.gleague.example.com" for t in ["austin", "reno", "texas", "idaho"]]
    subs += [f"{t}.dleague.example.com" for t in ["austin", "reno", "texas"]]
    themes = detect_patterns(subs)
    cross = [t for t in themes if len(t["slots"]) == 2]
    assert cross, [t["template"] for t in themes]
    league_slot = next(
        s for s in cross[0]["slots"] if set(s.get("values", [])) >= {"gleague", "dleague"}
    )
    team_slot = next(s for s in cross[0]["slots"] if "idaho" in s.get("values", []))
    # crossing them yields idaho.dleague, which was NOT in the input
    assert set(league_slot["values"]) == {"gleague", "dleague"}
    assert "idaho" in team_slot["values"]


def test_residual_excludes_templated_keeps_oneoffs():
    subs = [
        "web1.example.com", "web2.example.com", "web3.example.com",   # -> template web{n}
        "oxygen.example.com", "raptorsuprising.example.com",       # one-offs, no pattern
    ]
    themes = detect_patterns(subs)
    cov = covered_hosts(subs, themes)
    res = residual_hosts(subs, themes)
    assert "web1.example.com" in cov and "web1.example.com" not in res  # templated -> covered
    assert "oxygen.example.com" in res                              # one-off -> residual
    assert "raptorsuprising.example.com" in res


def test_residual_handles_enum_and_empty_value_slots():
    # template {team}.{league}{env} with env including "" must still match base hosts
    theme = {
        "template": "{team}.{league}{env}.example.com",
        "slots": [
            {"name": "team", "kind": "enum", "values": ["austin", "reno"]},
            {"name": "league", "kind": "enum", "values": ["gleague", "dleague"]},
            {"name": "env", "kind": "enum", "values": ["", "-dev", "-qa"]},
        ],
    }
    cov = covered_hosts(
        ["austin.gleague.example.com", "reno.dleague-dev.example.com", "humpty.example.com"], [theme]
    )
    assert "austin.gleague.example.com" in cov
    assert "reno.dleague-dev.example.com" in cov
    assert "humpty.example.com" not in cov


def test_mined_slots_get_a_label_from_known_vocabularies():
    from subseer.mine import guess_slot_label

    assert guess_slot_label({"kind": "enum", "values": ["dev", "prod", "qa"]}) == "env"
    assert guess_slot_label({"kind": "range", "min": 1, "max": 9}) == "number"
    assert guess_slot_label({"kind": "enum", "values": ["api", "shop", "blog"]}) == ""


def test_log_record_keeps_only_fields_that_apply():
    from subseer.models import Slot, Theme
    from subseer.pipeline import theme_to_dict

    t = Theme(name="nodes", description="d", evidence=[], kind="template",
              template="{s1}.{s2}.example.com",
              slots=[Slot(name="s1", kind="enum", values=["dev", "qa"], label="env"),
                     Slot(name="s2", kind="range", min=1, max=3)])
    d = theme_to_dict(t)
    assert d["slots"] == [{"name": "s1", "label": "env", "values": ["dev", "qa"]},
                          {"name": "s2", "range": [1, 3]}]
    assert "evidence" not in d and "candidates" not in d and d["cardinality"] == 6


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

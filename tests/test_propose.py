"""Tests for the pure parts of --propose (sampling + normalization)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subseer.propose import (
    _finalize,
    _parse_candidates,
    build_sample,
    normalize_candidate,
)


def test_parse_candidates_valid_json():
    content = '{"candidates": ["a.example.com", "b.example.com"]}'
    assert _parse_candidates(content) == ["a.example.com", "b.example.com"]


def test_parse_candidates_salvages_truncated():
    # a reply cut off by the model's output limit — array never closed
    content = '{\n  "candidates": [\n    "grafana.example.com",\n    "vault.example.com",\n    "jenk'
    assert _parse_candidates(content) == ["grafana.example.com", "vault.example.com"]  # partial last string dropped


def test_parse_candidates_empty_on_junk():
    assert _parse_candidates("sorry, I cannot help") == []


def test_build_sample_all_when_small():
    subs = ["b.example.com", "a.example.com", "a.example.com", " "]
    assert build_sample(subs, 600) == ["a.example.com", "b.example.com"]  # deduped, sorted


def test_build_sample_caps_and_deterministic():
    subs = [f"h{i}.example.com" for i in range(1000)]
    s1 = build_sample(subs, 50)
    s2 = build_sample(subs, 50)
    assert len(s1) == 50
    assert s1 == s2  # same seed -> reproducible


def test_build_sample_seed_varies_subset():
    subs = [f"h{i}.example.com" for i in range(1000)]
    a = build_sample(subs, 50, seed=0)
    b = build_sample(subs, 50, seed=1)
    assert a != b                 # different seed -> different slice
    assert len(a) == len(b) == 50
    # but a small list ignores the seed (all sent either way)
    small = ["b.example.com", "a.example.com"]
    assert build_sample(small, 50, seed=0) == build_sample(small, 50, seed=1)


def test_normalize_candidate():
    assert normalize_candidate("grafana", "example.com") == "grafana.example.com"      # bare label
    assert normalize_candidate("Grafana.example.com", "example.com") == "grafana.example.com"  # already full
    assert normalize_candidate("test.grafana", "example.com") == "test.grafana.example.com"  # multi-level
    assert normalize_candidate("vpn.example.com.", "example.com") == "vpn.example.com"     # trailing dot
    assert normalize_candidate("   ", "example.com") is None


def test_finalize_normalizes_dedups_drops_known():
    subs = ["api.example.com"]
    raw = ["grafana", "grafana.example.com", "api", "  ", "Vault.example.com"]  # bare, dup, known, blank, cased
    out = _finalize(raw, subs, "example.com")
    assert out == ["grafana.example.com", "vault.example.com"]  # grafana deduped, api dropped (known)


def test_user_message_is_rendered_from_the_prompt_files():
    from subseer.propose import _user_prompt

    msg = _user_prompt(["api.example.com"], "example.com", 2000, 5, ["{s1}.dev.example.com"])
    assert msg.startswith("Apex domain: example.com")
    assert "ALREADY covered" in msg and "{s1}.dev.example.com" in msg
    assert msg.endswith("Propose up to 5 plausible NEW subdomains under example.com, "
                        "most-likely first.")
    assert "ALREADY" not in _user_prompt(["api.example.com"], "example.com", 2000, 5, [])


def test_the_reply_example_in_the_predict_prompt_parses():
    from subseer.propose import SYSTEM_PROMPT

    example = next(l for l in SYSTEM_PROMPT.splitlines() if l.startswith('{"candidates"'))
    assert _parse_candidates(example) == ["jenkins.example.com", "vault.example.com"]


def test_calls_split_a_big_list_without_gaps_or_repeats():
    subs = [f"h{i}.example.com" for i in range(5000)]
    slices = [build_sample(subs, 2000, call=c) for c in range(3)]  # ceil(5000/2000) = 3
    seen = [h for s in slices for h in s]
    assert len(seen) == len(set(seen)) == 5000           # every host once, none twice
    assert all(len(s) <= 2000 for s in slices)
    assert build_sample(subs, 2000, call=3) != slices[0]  # a 4th call starts a reshuffled pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

"""Tests for the per-host FUZZ generator. Stdlib only."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subseer.fuzz import (
    affix_mutations,
    fuzz_host,
    generate,
    is_random,
    label_inserts,
    pluralize,
    separator_swaps,
    vocab_swap,
)


def test_fuzz_intra_label_words():
    got = fuzz_host("racecar.example.com", "example.com", words={"race", "car", "ace"})
    for expected in [
        "FUZZ.example.com",           # whole-sub brute (kept)
        "FUZZcar.example.com",        # word "race"
        "raceFUZZ.example.com",       # word "car"
        "rFUZZcar.example.com",       # word "ace" (overlapping)
    ]:
        assert expected in got, (expected, sorted(got))


def test_label_insertion_off_by_default():
    # generic-wordlist label insertion stays gone; nothing new prepended by default
    got = fuzz_host("account.example.com", "example.com", words=set())
    assert "FUZZ.account.example.com" not in got   # no generic prepend
    assert "account.FUZZ.example.com" not in got   # no generic pre-apex insert
    assert "FUZZ.example.com" in got               # whole-sub brute stays


def test_label_inserts_typed_prepend_and_preapex():
    got = label_inserts("api.example.com", "example.com")
    assert "{env}.api.example.com" in got     # typed prepend (new leading label)
    assert "api.{env}.example.com" in got     # typed pre-apex insert
    assert "{region}.api.example.com" in got  # each enabled vocab
    # multi-label sub keeps its structure
    got2 = label_inserts("api.dev.example.com", "example.com")
    assert "{env}.api.dev.example.com" in got2
    assert "api.dev.{env}.example.com" in got2


def test_label_insert_gated_by_config():
    cfg = dict(__import__("subseer.fuzz", fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG)
    cfg["label_insert"] = True
    got = fuzz_host("api.example.com", "example.com", words=set(), config=cfg)
    assert "{env}.api.example.com" in got     # on when enabled
    off = fuzz_host("api.example.com", "example.com", words=set())
    assert "{env}.api.example.com" not in off  # off by default


def test_apex_frozen_and_infra_skipped():
    got = fuzz_host("api.dev.example.com", "example.com", words=set())
    assert all(t.endswith(".example.com") for t in got)   # apex never touched
    assert "api.{env}.example.com" in got                 # dev -> {env}
    assert "{infra}.dev.example.com" not in got           # infra too broad -> not typed


def test_ip_host_only_whole_sub():
    got = fuzz_host("32.36.204.example.com", "example.com", words=set())
    assert got == {"FUZZ.example.com"}   # IP-like -> nothing to fuzz but the label brute


def test_top_n_caps_words():
    # 'racecar' contains race, car, ace (>=3) — cap at 1 keeps only the longest (race)
    got = fuzz_host("racecar.example.com", "example.com", words={"race", "car", "ace"}, top_n=1)
    assert "FUZZcar.example.com" in got          # race (len 4) kept
    assert "rFUZZcar.example.com" not in got     # ace dropped by the cap


def test_min_word_len():
    got = fuzz_host("beacon.example.com", "example.com", words={"be", "beacon"}, min_len=3)
    assert not any(t == "FUZZacon.example.com" for t in got)  # 'be' too short


def test_is_random():
    assert is_random("28fceb")
    assert is_random("a1b2c3d4")
    assert not is_random("racecar")
    assert not is_random("dev")


def test_vocab_swap_env():
    got = vocab_swap("api-dev.example.com", "example.com")
    assert "api-qa.example.com" in got
    assert "api-prod.example.com" in got
    assert "api-stage.example.com" in got


def test_vocab_swap_direction():
    got = vocab_swap("redirect-west.example.com", "example.com")
    assert "redirect-east.example.com" in got   # the west->east extrapolation
    assert "redirect-north.example.com" in got


def test_typed_slots_number_and_vocab():
    got = fuzz_host("47min3-iature.example.com", "example.com", words=set())
    assert "{num}min3-iature.example.com" in got   # first number typed
    assert "47min{num}-iature.example.com" in got  # second number typed


def test_typed_slot_env():
    got = fuzz_host("cms-dev.example.com", "example.com", words=set())
    assert "cms-{env}.example.com" in got          # dev -> {env}, not generic FUZZ


def test_typed_slot_region_whole_label():
    got = fuzz_host("us-east-1.gw.example.com", "example.com", words=set())
    assert "{region}.gw.example.com" in got        # multi-token region typed as a whole label


def test_generate_returns_typed_templates():
    hosts = ["cms-dev.example.com", "cms-prod.example.com"]
    templates, candidates = generate(hosts)
    assert "cms-{env}.example.com" in templates    # vocab now a typed template
    # concrete candidates are separator/plural/affix only
    assert "cms.example.com" in candidates          # affix strip of "cms-dev" -> "cms"


def test_separator_swaps():
    got = separator_swaps("api-dev.example.com", "example.com")
    assert "apidev.example.com" in got       # '-' -> glue
    assert "api.dev.example.com" in got      # '-' -> '.'


def test_pluralize():
    got = pluralize("asset.example.com", "example.com")
    assert "assets.example.com" in got
    assert "asset.example.com" not in got    # the input form isn't re-emitted
    got2 = pluralize("images.example.com", "example.com")
    assert "image.example.com" in got2       # strip trailing s


def test_affix_add_and_strip():
    got = affix_mutations("account.example.com", "example.com")
    assert "api-account.example.com" in got  # prepend
    assert "account-api.example.com" in got  # append
    stripped = affix_mutations("dev-account.example.com", "example.com")
    assert "account.example.com" in stripped  # strip leading affix


def test_custom_words_segment():
    # the built-in dict doesn't know "raptors"/"uprising"; a custom list does
    got = fuzz_host("raptorsuprising.example.com", "example.com", words={"raptors", "uprising"})
    assert "FUZZuprising.example.com" in got   # raptors -> FUZZ
    assert "raptorsFUZZ.example.com" in got    # uprising -> FUZZ


def test_load_wordlist(tmp_path=None):
    import os
    import tempfile

    from subseer.fuzz import load_wordlist

    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("raptors\nUprising\n\nbad word\nceltics\n")
    try:
        words = load_wordlist(path)
        assert words == {"raptors", "uprising", "celtics"}  # lowercased, blanks/non-alnum dropped
    finally:
        os.remove(path)


def test_expand_templates():
    from subseer.fuzz import expand_templates

    templates = ["cms-{env}.example.com", "raptorsFUZZ.example.com", "ns{num}.example.com"]
    out = expand_templates(templates, words={"api", "dev"}, num_min=1, num_max=3)
    assert "cms-prod.example.com" in out       # {env} filled from env list
    assert "raptorsapi.example.com" in out     # FUZZ filled from words
    assert "raptorsdev.example.com" in out
    assert "ns2.example.com" in out            # {num} filled from range
    assert "ns4.example.com" not in out        # range is 1..3


def test_iter_expand_templates_streams():
    from subseer.fuzz import iter_expand_templates

    templates = ["cms-{env}.example.com", "raptorsFUZZ.example.com", "ns{num}.example.com"]
    out = list(iter_expand_templates(templates, words={"api"}, num_min=1, num_max=2))
    assert "cms-prod.example.com" in out       # {env} filled
    assert "raptorsapi.example.com" in out     # FUZZ filled from words
    assert "ns1.example.com" in out and "ns2.example.com" in out
    assert "ns3.example.com" not in out        # range 1..2
    # streaming yields duplicates if templates overlap (no dedup) — that's expected
    dupes = list(iter_expand_templates(["a-{env}.example.com", "a-{env}.example.com"], words=set()))
    assert dupes.count("a-prod.example.com") == 2


def test_expand_skip_fuzz():
    from subseer.fuzz import expand_templates

    templates = ["cms-{env}.example.com", "raptorsFUZZ.example.com"]
    out = expand_templates(templates, words={"api", "dev"}, include_fuzz=False)
    assert "cms-prod.example.com" in out        # typed slot still filled
    assert not any("raptors" in x for x in out)  # FUZZ template skipped


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

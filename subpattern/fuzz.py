"""Per-host FUZZ generator - treats every host individually (no siblings needed).

Complements the cross-host miner: it catches the one-offs the miner skips
(`raptorsuprising`, `redirect-west`) by mutating each host on its own.

Two outputs, both apex-frozen (never touches the registered domain):
  - FUZZ templates: one FUZZ marker per template, at every position -
      * new label inserted at each dot boundary  (FUZZ.sub / sub.FUZZ / between)
      * a dictionary word substring replaced inside a label (top-N longest)
      * whole-sub replacement (FUZZ.apex)
    Feed these to a wordlist fuzzer (ffuf/altdns).
  - Vocab swaps: when a token is in a known closed set (env/region/version/...),
    emit the host with the OTHER values of that set - concrete candidates
    ("one dev -> qa/stage/prod").

Pure module (no pydantic/anthropic) so it's stdlib-testable.
"""

from __future__ import annotations

import re

from .mine import detect_apex, detok, tokenize

FUZZ = "FUZZ"

# Closed vocabularies: detect one value -> offer the whole set.
VOCABS: dict[str, list[str]] = {
    "env": ["dev", "qa", "qai", "stage", "staging", "stg", "prod", "prd", "test",
            "uat", "int", "sandbox", "demo", "preprod", "nonprod", "canary", "beta"],
    "region": ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1",
               "eu-west-2", "eu-central-1", "ap-south-1", "ap-southeast-1",
               "ap-northeast-1", "ca-central-1", "sa-east-1"],
    "dc": ["sjc", "rtp", "rcdn", "bgl", "iad", "use1", "usw2", "lhr", "fra", "syd", "nrt", "alln"],
    "cc": ["us", "uk", "de", "fr", "es", "it", "jp", "kr", "cn", "br", "mx", "ca",
           "au", "in", "ru", "nl", "se", "pl", "tr"],
    "version": ["v1", "v2", "v3", "v4", "v5"],
    "color": ["blue", "green", "red", "canary"],
    "instance": ["primary", "secondary", "master", "replica", "standby", "active", "passive"],
    "dir": ["north", "south", "east", "west"],
    "infra": ["api", "app", "web", "www", "mail", "smtp", "ns", "mx", "vpn", "cdn",
              "db", "gw", "lb", "proxy", "auth", "sso", "admin", "portal", "gateway",
              "cache", "static", "cms", "ci", "git"],
    "tech": ["grafana", "kibana", "jenkins", "gitlab", "jira", "splunk", "vault",
             "consul", "okta", "nexus", "sonar", "prometheus", "argo"],
}

# value -> category (first category wins on collision)
VALUE_VOCAB: dict[str, str] = {}
for _cat, _vals in VOCABS.items():
    for _v in _vals:
        VALUE_VOCAB.setdefault(_v, _cat)

_IP = re.compile(r"^[0-9.]+$")

# Every template type the fuzzer can emit, with its default enabled state.
# infra/tech are off by default (too broad - a specific service isn't
# interchangeable with its whole category). Override via a config file.
DEFAULT_CONFIG: dict[str, bool] = {
    "whole_sub": True,       # FUZZ.apex  (brute the first label)
    "word_fuzz": True,       # replace a dictionary word inside a label
    "num": True,             # {num}
    "env": True,             # {env}
    "region": True,          # {region}
    "dc": True,              # {dc}
    "cc": True,              # {cc}
    "version": True,         # {version}
    "color": True,           # {color}
    "instance": True,        # {instance}
    "dir": True,             # {dir}
    "infra": False,          # {infra}  (off - too broad)
    "tech": False,           # {tech}   (off - too broad)
    "separator_swaps": True,
    "pluralize": True,
    "affix_strip": True,
    "label_insert": False,   # insert a typed value as a new label (prepend / pre-apex) - off (noisier)
}


def load_wordlist(path: str) -> set[str]:
    """Load a segmentation dictionary (one word per line) to union with the built-in.

    Only alphanumeric tokens are kept; length filtering happens at match time via
    ``min_len``, so short entries are harmless.
    """
    words: set[str] = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w and w.isalnum():
                words.add(w)
    return words


def load_config(path: str | None = None):
    """Return (template_toggles, top_words, min_word_len) from a JSON config.

    Missing file or keys fall back to DEFAULT_CONFIG. Config shape:
        {"top_words": 4, "min_word_len": 3, "templates": {"env": true, ...}}
    """
    import json
    import os

    cfg = dict(DEFAULT_CONFIG)
    top, min_len = 4, 3
    if path and os.path.exists(path):
        data = json.loads(open(path, encoding="utf-8").read())
        cfg.update(data.get("templates", {}))
        top = data.get("top_words", top)
        min_len = data.get("min_word_len", min_len)
    return cfg, top, min_len

# Built-in segmentation dictionary (common english + recon tokens). Supply your
# own via generate(words=...) for better coverage.
WORDS: set[str] = {
    "race", "car", "ace", "cross", "over", "inside", "out", "outside", "up", "rising",
    "down", "star", "all", "ball", "court", "hoop", "team", "teams", "fan", "fans",
    "news", "blog", "blogs", "video", "videos", "image", "images", "game", "games",
    "live", "stream", "watch", "play", "player", "ticket", "tickets", "event", "events",
    "media", "mobile", "secure", "login", "auth", "cloud", "data", "store", "shop",
    "careers", "jobs", "press", "corp", "help", "support", "account", "portal", "admin",
    "api", "web", "app", "apps", "mail", "dev", "test", "stage", "prod", "beta",
    "connect", "global", "league", "fantasy", "draft", "pick", "picks", "vote", "sport",
    "sports", "legend", "legends", "be", "get", "go", "my", "the", "and", "for", "pro",
    "photo", "photos", "search", "feed", "feeds", "upload", "download", "assets", "asset",
    "content", "service", "services", "internal", "external", "gateway", "proxy", "cache",
    "static", "share", "link", "links", "page", "pages", "site", "sites", "home", "world",
    "champion", "championship", "green", "reimagine", "digital", "insights", "analytics",
    "meeting", "meetings", "report", "reports", "health", "total", "well", "being",
    "academy", "school", "camp", "combine", "draft", "rookie", "playoff", "playoffs",
}


_HEX = re.compile(r"^[0-9a-f]{6,}$")


def is_random(tok: str) -> bool:
    """A token that looks like a random id / hash (skip fuzzing it)."""
    if _HEX.match(tok):
        return True
    return len(tok) >= 10 and sum(c.isdigit() for c in tok) >= 4


def _word_fuzz(tok: str, words: set[str], top_n: int, min_len: int) -> list[str]:
    """Generic-FUZZ variants of one token: replace each dictionary-word substring
    (top-N longest, overlapping) with a FUZZ marker, keeping surrounding chars."""
    hits: set[tuple[int, int]] = set()
    for start in range(len(tok)):
        for end in range(start + min_len, len(tok) + 1):
            if tok[start:end] in words:
                hits.add((start, end))
    out = []
    for start, end in sorted(hits, key=lambda se: -(se[1] - se[0]))[:top_n]:
        out.append(tok[:start] + FUZZ + tok[end:])
    return out


def fuzz_host(host: str, apex: str, words: set[str] = WORDS, top_n: int = 4,
              min_len: int = 3, config: dict | None = None) -> set[str]:
    """Single-slot templates for one host, gated by ``config`` toggles.

    Emits whole-sub brute, typed slots ({num}/{env}/{region}/...), and generic
    FUZZ for unknown dictionary words - each only if enabled in config. Apex is
    never touched; IP-like hosts get nothing but the label brute.
    """
    config = config or DEFAULT_CONFIG
    suffix = "." + apex
    host = host.strip().lower()
    if not host.endswith(suffix):
        return set()
    sub = host[: -len(suffix)]
    if not sub:
        return set()
    out: set[str] = set()
    if config.get("whole_sub"):
        out.add(f"{FUZZ}.{apex}")
    if _IP.match(sub):
        return out

    labels = sub.split(".")
    for li, label in enumerate(labels):
        def emit(new_label: str) -> None:
            out.add(".".join(labels[:li] + [new_label] + labels[li + 1:]) + "." + apex)

        # whole-label closed-vocab match (multi-token values like regions)
        cat = VALUE_VOCAB.get(label)
        if cat and config.get(cat):
            emit("{" + cat + "}")

        toks, seps = tokenize(label)
        for ti, tok in enumerate(toks):
            tcat = VALUE_VOCAB.get(tok)
            if tok.isdigit():
                if config.get("num"):
                    emit(detok(toks[:ti] + ["{num}"] + toks[ti + 1:], seps))
            elif tcat and config.get(tcat):
                emit(detok(toks[:ti] + ["{" + tcat + "}"] + toks[ti + 1:], seps))
            elif not is_random(tok) and config.get("word_fuzz"):
                for variant in _word_fuzz(tok, words, top_n, min_len):
                    emit(detok(toks[:ti] + [variant] + toks[ti + 1:], seps))

    if config.get("label_insert"):
        out |= label_inserts(host, apex, config)
    return out


def label_inserts(host: str, apex: str, config: dict | None = None) -> set[str]:
    """Insert a typed value as a NEW label - only prepend and pre-apex positions.

    A controlled slice of altdns-style label insertion: rather than dropping any
    wordlist word at every dot (the pruned firehose), this emits TYPED templates
    at the two positions orgs actually use - a new leading label ({env}.host) and
    a label just before the apex (host.{env}). Bounded to the enabled vocabs, so
    it stays ~a dozen templates/host instead of exploding. Off by default.
    """
    config = config or DEFAULT_CONFIG
    suffix = "." + apex
    host = host.strip().lower()
    if not host.endswith(suffix):
        return set()
    sub = host[: -len(suffix)]
    if not sub or _IP.match(sub):
        return set()
    out: set[str] = set()
    for cat in VOCABS:
        if not config.get(cat):
            continue
        ph = "{" + cat + "}"
        out.add(f"{ph}.{sub}.{apex}")   # prepend a new leading label
        out.add(f"{sub}.{ph}.{apex}")   # insert a label just before the apex
    return out


def vocab_swap(host: str, apex: str) -> set[str]:
    """Concrete candidates from swapping a known closed-vocab token for its siblings."""
    suffix = "." + apex
    host = host.strip().lower()
    if not host.endswith(suffix):
        return set()
    sub = host[: -len(suffix)]
    labels = sub.split(".")
    out: set[str] = set()
    for li, label in enumerate(labels):
        toks, seps = tokenize(label)
        for ti, tok in enumerate(toks):
            cat = VALUE_VOCAB.get(tok)
            if not cat:
                continue
            for alt in VOCABS[cat]:
                if alt == tok:
                    continue
                new_label = detok(toks[:ti] + [alt] + toks[ti + 1:], seps)
                out.add(".".join(labels[:li] + [new_label] + labels[li + 1:]) + "." + apex)
    return out


AFFIXES = ["api", "dev", "test", "prod", "internal", "admin", "new", "old", "svc", "v2"]


def separator_swaps(host: str, apex: str) -> set[str]:
    """Swap the separator at each boundary between glue / '-' / '.' (one at a time)."""
    suffix = "." + apex
    host = host.strip().lower()
    if not host.endswith(suffix):
        return set()
    toks, seps = tokenize(host[: -len(suffix)])
    out: set[str] = set()
    for i in range(1, len(toks)):
        for alt in ("", "-", "."):
            if alt == seps[i]:
                continue
            out.add(detok(toks, seps[:i] + [alt] + seps[i + 1:]) + "." + apex)
    return out


def pluralize(host: str, apex: str) -> set[str]:
    """Toggle a trailing 's' on each word token (asset<->assets)."""
    suffix = "." + apex
    host = host.strip().lower()
    if not host.endswith(suffix):
        return set()
    toks, seps = tokenize(host[: -len(suffix)])
    out: set[str] = set()
    for ti, tok in enumerate(toks):
        if not tok.isalpha() or len(tok) < 3:
            continue
        alt = tok[:-1] if tok.endswith("s") else tok + "s"
        out.add(detok(toks[:ti] + [alt] + toks[ti + 1:], seps) + "." + apex)
    return out


def affix_mutations(host: str, apex: str) -> set[str]:
    """Prepend/append common affixes to the leftmost label, and strip existing ones."""
    suffix = "." + apex
    host = host.strip().lower()
    if not host.endswith(suffix):
        return set()
    labels = host[: -len(suffix)].split(".")
    first, rest = labels[0], labels[1:]
    out: set[str] = set()
    for aff in AFFIXES:
        out.add(".".join([f"{aff}-{first}"] + rest) + "." + apex)
        out.add(".".join([f"{first}-{aff}"] + rest) + "." + apex)
    parts = first.split("-")
    if len(parts) > 1:
        if parts[0] in AFFIXES:
            out.add(".".join(["-".join(parts[1:])] + rest) + "." + apex)
        if parts[-1] in AFFIXES:
            out.add(".".join(["-".join(parts[:-1])] + rest) + "." + apex)
    return out


def affix_strip(host: str, apex: str) -> set[str]:
    """Strip a leading/trailing affix from the first label (the base host)."""
    suffix = "." + apex
    host = host.strip().lower()
    if not host.endswith(suffix):
        return set()
    labels = host[: -len(suffix)].split(".")
    first, rest = labels[0], labels[1:]
    parts = first.split("-")
    out: set[str] = set()
    if len(parts) > 1:
        if parts[0] in AFFIXES:
            out.add(".".join(["-".join(parts[1:])] + rest) + "." + apex)
        if parts[-1] in AFFIXES:
            out.add(".".join(["-".join(parts[:-1])] + rest) + "." + apex)
    return out


def mutations(host: str, apex: str, config: dict | None = None) -> set[str]:
    """Concrete (non-template) mutations, gated by config: separator / plural / affix-strip."""
    config = config or DEFAULT_CONFIG
    out: set[str] = set()
    if config.get("separator_swaps"):
        out |= separator_swaps(host, apex)
    if config.get("pluralize"):
        out |= pluralize(host, apex)
    if config.get("affix_strip"):
        out |= affix_strip(host, apex)
    return out


_SLOT = re.compile(r"\{(\w+)\}")


def iter_expand_templates(templates, words, num_min: int = 0, num_max: int = 50,
                          include_fuzz: bool = True):
    """Stream concrete hosts from single-slot templates (no dedup, no buffering).

    Same fills as ``expand_templates`` but yields one host at a time, so a huge
    wordlist doesn't build a giant in-memory set. Feed the output straight to a
    file or a resolver (dnsx/puredns).
    """
    for t in templates:
        if FUZZ in t:
            if not include_fuzz:
                continue
            for w in words:
                yield t.replace(FUZZ, w, 1)
            continue
        m = _SLOT.search(t)
        if not m:
            yield t
            continue
        cat, ph = m.group(1), m.group(0)
        if cat == "num":
            vals = [str(n) for n in range(num_min, num_max + 1)]
        else:
            vals = VOCABS.get(cat, [])
        for v in vals:
            yield t.replace(ph, v, 1)


def expand_templates(templates, words, num_min: int = 0, num_max: int = 50,
                     include_fuzz: bool = True) -> list[str]:
    """Fill each single-slot template into concrete hosts (deduped, sorted).

    FUZZ  -> each word in ``words``  (skipped if include_fuzz=False)
    {num} -> num_min..num_max
    {cat} -> the closed vocabulary for that category (env/region/version/...)

    Holds everything in memory to dedup; for huge wordlists use
    ``iter_expand_templates`` (streaming) instead.
    """
    return sorted(set(iter_expand_templates(templates, words, num_min, num_max, include_fuzz)))


def generate(hosts, apex: str | None = None, words: set[str] = WORDS,
             top_n: int = 4, min_len: int = 3, config: dict | None = None):
    """Return (fuzz_templates, candidates), both sorted and deduped, honoring ``config``."""
    config = config or DEFAULT_CONFIG
    apex = apex if apex is not None else detect_apex(hosts)
    known = {h.strip().lower() for h in hosts if h.strip()}
    fuzz: set[str] = set()
    cand: set[str] = set()
    for h in hosts:
        h = h.strip().lower()
        if not h:
            continue
        fuzz |= fuzz_host(h, apex, words=words, top_n=top_n, min_len=min_len, config=config)
        cand |= mutations(h, apex, config=config)
    cand -= known
    return sorted(fuzz), sorted(cand)

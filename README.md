# subpattern

Discover the **naming conventions** an organization uses for its subdomains with an
LLM, then expand those conventions into a candidate wordlist **in code**. Built for
authorized recon (bug bounty / pentest): feed it the subdomains you already know,
get back plausible ones you don't.

## The idea

A plain frequency miner can only find patterns it can express as a rule — token
counts, numeric ranges, skeletons. It's structurally blind to *semantic* and
*thematic* conventions: hosts named after a mythology, internal codenames, a
deprecated `payments-v1` implied by `payments-v2`. That's the gap an LLM fills.

But an LLM is the wrong tool for *bulk enumeration* — output is capped (~128k
tokens ≈ ~10k names per call) and expensive. So the work is split:

```
known subdomains
      │
      ▼
  DISCOVER  (LLM — Opus 4.8, adaptive thinking)   ← all the intelligence
      │  returns compact JSON themes, each typed:
      │    • enumerate : a finite, knowledge-driven set (the model lists members)
      │    • template  : a combinatorial spec  {name}{n}.example.com + slot defs
      ▼
  PREVIEW cardinality  (multiply slot sizes — instant, no expansion)
      ▼
  EXPAND  (pure Python)   ← zero intelligence, just a printer
      │  range/enum product, ranked, capped, deduped against the input
      ▼
  candidates.txt   +   patterns.json
```

The LLM decides *what* the patterns are and *which* values fill each slot. Code
only multiplies them out. A template like `{name}{n}.example.com` is ~150 bytes
but expands to thousands of names — so the model's limited output budget goes to
finding *more themes*, never to typing repetitive strings. That's how you reach
millions of candidates without millions of output tokens.

## Two engines: code miner + LLM

There are two ways to get the themes, with different strengths — use either, or
both:

| | **Code miner** (`--mine`) | **LLM** (default / discovery) |
|---|---|---|
| Catches | structural / combinatorial templates (`{site}-ads-{n}`, `{service}.{env}`) | semantic / thematic patterns (codenames, `west`→`east`, team brands) |
| Coverage | **exhaustive** — provably every host | surfaces what it notices (can miss) |
| Reproducible | yes, deterministic | no, varies per run |
| Scale | any size (7M+), streaming | bounded by context window |
| Cost | **$0**, no API | scales with input size |

The miner tokenizes every host (on `.`, `-`, and letter/digit boundaries), finds
positions whose value-set *recurs across contexts*, and emits templates with real
enum value-sets and numeric `min/max` — all from 100% of the data, for free. It
**cannot** do semantics (a glued one-off like `raptorsuprising`, or inferring
`redirect-east` from a lone `redirect-west`); that's the LLM's job.

The best of both is **`--hybrid`**: the miner explains the structural bulk, then
the LLM runs on **only the residual** — the hosts no template covers. On a
systematic infra target the templates absorb the millions and the LLM sees a tiny
fraction, so you get full coverage at minimal LLM cost.

```
all hosts ─▶ MINE (code, free) ─▶ RESIDUAL (uncovered one-offs) ─▶ LLM ─▶ combine ─▶ expand
```

## Install

```bash
cd subpattern
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...               # or set in your shell / .env
```

## Use

```bash
# Discover + expand
subpattern examples/sample_subdomains.txt

# Just see the conventions and how many candidates each implies — no expansion, no spend on output
subpattern examples/sample_subdomains.txt --dry-run

# Thorough run: 3 deepening discovery passes + a critic pass to drop invented themes
subpattern examples/sample_subdomains.txt --runs 3 --critic

# Caps and model
subpattern big_target.txt --model claude-opus-4-8 --max-candidates 100000 --per-theme-cap 20000
```

### Discovery modes

```bash
# Code miner only — exhaustive structural templates, $0, no API:
subpattern target.txt --mine

# Hybrid — mine structure, then LLM only on the residual one-offs (best coverage/cost):
subpattern target.txt --hybrid --model claude-sonnet-4-6

# Mine, then a cheap LLM pass to name / rank / drop the detected templates:
subpattern target.txt --mine --refine --model claude-sonnet-4-6

# Mine and just inspect the detected templates (no expansion, no API):
subpattern target.txt --mine --dry-run
```

| Flag | What it does | Cost |
|---|---|---|
| `--mine` | Detect templates from the data in code — exhaustive, deterministic, scales to any size. Writes to `--patterns`. | **$0** |
| `--min-values N` | Miner: min distinct values for a position to become a slot (default 3). Lower = more recall, more noise. | — |
| `--hybrid` | Mine structure, then run LLM discovery on **only** the hosts no template covers. | residual-sized LLM call |
| `--refine` | LLM pass over the detected templates: gives each a human name, ranks 1–5, drops noise. Reasons over ~100 templates, not the hosts — cheap. | ~+1 cheap call |

### Develop for free (no API credits)

The discovery step just turns subdomains into a themes JSON — you don't have to
buy that JSON from the paid API while building. Two flags let you run the whole
pipeline for $0:

```bash
# 1. Run everything offline from a themes file (no API call, no cost):
subpattern examples/sample_subdomains.txt --themes examples/sample_patterns.json

# 2. To produce your own themes file for free, print the prompt and paste it into
#    claude.ai (covered by a Pro plan), then save the JSON it returns:
subpattern your_target.txt --print-prompt
#    ... paste output into claude.ai, save its JSON as themes.json, then:
subpattern your_target.txt --themes themes.json
```

`--themes` accepts either a `patterns.json` this tool wrote or a
`{"themes": [...]}` blob from a chat model. Note: the Anthropic SDK can't bill a
Claude Pro subscription — Pro and API credits are separate — so this prompt-and-paste
flow is the way to use your Pro plan during development. When you're done, drop the
flags and let it call the API live; no code changes.

### Quality flags

| Flag | What it does | Cost |
|---|---|---|
| `--runs N` | N **deepening** discovery passes. Each pass after the first is told which themes were already found and hunts for what they overshadow, then all are merged. Beats single-pass non-determinism. 2–3 is a good range. | +N−1 discovery calls |
| `--critic` | After discovery, a skeptical pass drops themes whose cited evidence doesn't actually support the rule (plus a free code check that drops themes citing subdomains absent from your input). | +1 call |

Outputs:
- `candidates.txt` — new subdomains, ranked, deduped against your input.
- `patterns.json` — every theme the model found, with evidence + computed cardinality.
  Read this first; it's also where you'll catch any convention the model invented.

## How scale is handled

| Input size | Behavior |
|---|---|
| ≤ `--chunk-size` (default 20k) | One discovery call over the whole file. |
| larger | **Map-reduce**: split into chunks, discover each, merge themes. Guarantees every subdomain is actually reasoned over rather than buried in one oversized context. |

A 237 KB file (~10k subdomains, ~80k tokens) is a single call. A genuine 100k+
list (several MB) is chunked automatically.

## Theme spec

```json
{
  "name": "numbered web hosts",
  "kind": "template",
  "template": "{name}{n}.example.com",
  "slots": [
    {"name": "name", "kind": "enum",  "values": ["web", "node", "app"]},
    {"name": "n",    "kind": "range", "min": 1, "max": 1000, "pad": 2}
  ],
  "evidence": ["web01.example.com", "node02.example.com"],
  "cardinality": 3000
}
```

`enumerate` themes carry a `candidates` list instead of `template`/`slots`.

## Model choice

Discovery is *creative inference about a target* (spotting themes, extrapolating
siblings), which is where the top tier earns its premium — default is
`claude-opus-4-8`. Swap with `--model` (`claude-fable-5` for the strongest pattern
intuition, `claude-sonnet-4-6` if you want lower cost and a `temperature` dial for
variance across re-runs).

## Tests (no API key needed)

All core logic is covered by stdlib-only tests — the miner, expander, residual
matching, review/refine filtering, and cost math:

```bash
python tests/test_mine.py      # tokenizer, pattern detection, residual matching
python tests/test_expand.py    # cardinality + template expansion
python tests/test_review.py    # critic filter, multi-run merge, refine apply
python tests/test_pricing.py   # cost math
# or, if installed:  pytest
```

## Notes / future work

- The miner can't split **glued** compound tokens (`bucksgaming` → `bucks`+`gaming`)
  — those need dictionary-based word segmentation. They fall to the residual / LLM.
- **Known-vocabulary expansion** would let a single observed value imply its set
  (one `-west` → `-east/-north/-south`; a partial env list → the full one).
- **DNS validation** is the biggest missing piece: resolve the candidates (with
  wildcard filtering) and feed hits back to re-mine — turning a wordlist generator
  into a discovery tool.
- This is dual-use recon tooling. Only run it against targets you're authorized to test.
```

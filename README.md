# subseer

Generate candidate subdomains from the ones you already know. Feed `subseer` a
list of known subdomains; it learns their **naming patterns** and returns plausible
new hosts to go resolve. Built for authorized recon (bug bounty / pentest).

Offline and free by default; add an LLM (OpenAI or a local Ollama model) only when
you want it to reach beyond what the data alone can show.

## The idea

Three generators, each good at something different:

| Generator | What it does | Needs |
|---|---|---|
| **mine** | Learn structural templates from your hosts and recombine the values you have (cross-fill, numeric ranges). | code, offline, free |
| **fuzz** | Mutate each host — separators, plurals, affixes, typed-slot swaps (`dev`→`qa/prod`), dictionary-word FUZZ. | code, offline, free |
| **AI** | Fill mined slots with real-world values you *don't* have (`memphis, austin` → all 30 teams), discover latent slots, and propose net-new names. | LLM (OpenAI or Ollama) |

The split matters: an LLM is great at *knowledge* but a bad, expensive bulk
enumerator. So the model only emits compact things — slot values, a few template
specs — and **code multiplies them out**. A template like `{team}.dleague.example.com`
is tiny but expands to dozens of hosts.

```
known subs ─▶ mine (code) ─┐
             fuzz (code) ──┤─▶ merge + dedup ─▶ candidates
             AI enrich/propose (optional) ─┘
```

## Install

```bash
pip install -e .                      # from a clone
# or, straight from GitHub:
pipx install git+https://github.com/Raymond-JV/subseer
```

Only needed for the AI layer:

```bash
export OPENAI_API_KEY=sk-...          # for --gpt
# --ollama just needs a local `ollama serve`
```

## Use

```bash
# Offline, free — mine + fuzz:
subseer subs.txt

# Add AI (OpenAI gpt-4o-mini): enrich mined slots + discover + propose:
subseer subs.txt --gpt

# AI via a local model (offline, private):
subseer subs.txt --ollama            # bare = qwen2.5

# Just mining, enriched by the model:
subseer subs.txt --mine --gpt

# Stream straight into your resolver (results to stdout, logs to stderr):
subseer subs.txt --gpt --out - | dnsx | httpx -sc
```

`subseer --help` groups every flag under the generator it belongs to.

## Generators

- **`--mine`** — tokenizes each host (on `.`, `-`, and letter/digit boundaries),
  finds positions whose value-set recurs across contexts, and emits templates with
  observed enum values and numeric ranges. Cross-fills combinations you haven't
  deployed. Exhaustive, deterministic, free. PSL-aware apex detection (handles
  `.co.uk`, `.com.mx`, …).
- **`--fuzz`** — per-host mutations: separator swaps, pluralize, affix strip, typed
  closed-vocab swaps, and dictionary-word `FUZZ`. Used **alone**, it *streams* the
  full expansion to `--out` (any wordlist size, no memory blowup):
  ```bash
  subseer subs.txt --fuzz --wordlist /usr/share/seclists/.../raft-medium.txt --out - | dnsx
  ```
- **`--propose`** — the model proposes net-new names templates can't produce (infra
  it infers, target-specific brands, theme continuation). Needs a backend.

With **no generator flag**, `subseer` runs **mine + fuzz** (and, if a backend is
set, also enrich + propose) and merges everything into one deduped file.

## AI backend

Pick one; a backend flag turns on the AI layer (enriches `--mine`, powers `--propose`):

| Flag | Backend | Default model |
|---|---|---|
| `--gpt [MODEL]` | OpenAI-compatible (reads `$OPENAI_API_KEY`) | `gpt-4o-mini` |
| `--ollama [MODEL]` | local Ollama (`ollama serve`) | `qwen2.5` |

`--api-base` points `--gpt` at any compatible endpoint (Groq, Together, vLLM,
Ollama's `/v1`). gpt-4o-mini is the sweet spot: far cheaper than a frontier model,
much stronger than a small local one.

**Enrich** runs the mined templates once (batched) and then does Monte-Carlo
**discovery** passes over the residual one-offs. **Propose** samples your list for
context. Both merge across `--ai-runs` calls.

## Auto-tuning

Three knobs default to `auto` and scale with your list size — you rarely set them:

| Flag | `auto` behavior |
|---|---|
| `--sample` | subs sent to the model as context (~2k, more for huge lists) |
| `--ai-runs` | model calls merged (more for bigger lists / residuals) |
| `--min-values` | miner's slot threshold (stricter on huge lists to cut noise) |

## Output

- `--out PATH` — candidates (default `candidates.txt`); `--out -` streams to stdout
  with logs on stderr, so it pipes cleanly into `dnsx`/`httpx`.
- `--patterns PATH` — opt-in: also write the discovered themes as JSON (for
  inspection, `--themes` reuse, or feeding a permutation tool like gotator's `-perm`).
- `--themes PATH` — expand a saved themes JSON to candidates with no model call.
- `--per-theme-cap` / `--max-candidates` — caps (0 = unlimited).
- `--dry-run` — preview template cardinalities, write nothing.

## Suggested pipeline

Generation is cheap; resolution is where you spend time — so DNS-filter before you
HTTP-probe:

```bash
subseer subs.txt --gpt --out candidates.txt
dnsx  -l candidates.txt -o resolved.txt          # keep only hosts that exist
httpx -l resolved.txt -sc -title -rl 10          # probe the survivors, politely
```

## Tests

Stdlib-only, no API key or network needed (the OpenAI path is stubbed):

```bash
for t in tests/test_*.py; do python "$t"; done   # or: pytest
```

## Note

Dual-use recon tooling. Only run it against targets you're authorized to test, and
respect each program's scope and rate limits.

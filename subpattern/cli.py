"""Command-line interface for subpattern."""

from __future__ import annotations

import argparse
import contextlib
import sys

from .expand import cardinality
from .pipeline import expand_themes, load_subdomains, write_patterns


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="subpattern",
        description="Generate candidate subdomains from a known list. By default runs "
        "the offline generators (mine + fuzz). Add --gpt or --ollama to also expand "
        "patterns with an LLM.",
    )
    p.add_argument("input", nargs="?", help="File of subdomains, one per line. Omit if using -d.")
    p.add_argument("-d", "--domain", help="A single domain instead of an input file.")

    # --- --mine ------------------------------------------------------------
    m = p.add_argument_group("--mine (offline template mining)",
                             "Detect naming templates with code. Runs by default (with --fuzz) "
                             "when no generator is chosen. Add a backend to enrich its slots.")
    m.add_argument("--mine", action="store_true",
                   help="Detect templates from the input with code (offline).")
    m.add_argument("--min-values", default="auto",
                   help="Min distinct values for a slot (int or 'auto').")

    # --- --fuzz ------------------------------------------------------------
    f = p.add_argument_group("--fuzz (offline per-host fuzzing)",
                             "Mutate each host into candidate hostnames. Alone, streams the "
                             "full expansion to --out (any wordlist size, no memory blowup).")
    f.add_argument("--fuzz", action="store_true",
                   help="Per-host fuzz generator (offline): mutations, typed-slot fills, "
                   "and FUZZ filled from --wordlist.")
    f.add_argument("--wordlist", metavar="PATH",
                   help="Word list to fill FUZZ and aid segmentation (e.g. a SecLists file).")

    # --- --propose ---------------------------------------------------------
    pr = p.add_argument_group("--propose (LLM net-new names)",
                              "Ask the model for names templates can't produce. Needs a backend.")
    pr.add_argument("--propose", action="store_true",
                    help="LLM proposes net-new names (needs --gpt or --ollama).")
    pr.add_argument("--propose-count", type=int, default=1000, help="Max guesses to request.")

    # --- AI backend (shared by --mine enrichment and --propose) ------------
    ai = p.add_argument_group("AI backend",
                              "Pick one to enrich --mine's slots and power --propose.")
    ai.add_argument("--gpt", nargs="?", const="gpt-4o-mini", default=None, metavar="MODEL",
                    help="Use OpenAI (bare = gpt-4o-mini; or --gpt gpt-4o). Reads $OPENAI_API_KEY.")
    ai.add_argument("--ollama", nargs="?", const="qwen2.5", default=None, metavar="MODEL",
                    help="Use a local Ollama model (bare = qwen2.5). Needs `ollama serve`.")
    ai.add_argument("--api-base", default="https://api.openai.com/v1",
                    help="Endpoint for --gpt (OpenAI, Groq, Together, vLLM, ...).")
    ai.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL.")
    ai.add_argument("--sample", default="auto",
                    help="Max subs sent to the model as context (int or 'auto').")
    ai.add_argument("--ai-runs", default="auto", metavar="N",
                    help="Model calls merged, for more coverage (int or 'auto').")

    # --- output ------------------------------------------------------------
    o = p.add_argument_group("output")
    o.add_argument("--out", default="candidates.txt",
                   help="Where to write candidates ('-' = stdout; logs go to stderr).")
    o.add_argument("--patterns", metavar="PATH", default=None,
                   help="Opt-in: also write the discovered themes as JSON here.")
    o.add_argument("--per-theme-cap", type=int, default=50000,
                   help="Max candidates expanded per theme (0 = unlimited).")
    o.add_argument("--max-candidates", type=int, default=200000,
                   help="Global cap on total candidates (0 = unlimited).")
    o.add_argument("--dry-run", action="store_true",
                   help="Mine and preview cardinalities only; write nothing.")
    o.add_argument("--themes", metavar="PATH",
                   help="Load themes JSON and expand only (no model call).")
    return p.parse_args(argv)


def _cap(value: int) -> int | None:
    return None if value == 0 else value


# Real stdout, captured before any `--out -` redirect so results still reach the
# pipe while logs go to stderr.
_REAL_STDOUT = sys.stdout


@contextlib.contextmanager
def _out_stream(path: str):
    """Open the results sink: the real stdout for '-', otherwise a file."""
    if path == "-":
        yield _REAL_STDOUT
    else:
        f = open(path, "w", encoding="utf-8")
        try:
            yield f
        finally:
            f.close()


def _auto_runs(n: int) -> int:
    """Model runs scaled to list size (bigger list -> more sampled slices covered)."""
    if n <= 2000:
        return 2
    if n <= 5000:
        return 5
    if n <= 10000:
        return 8
    return 10


def _auto_sample(n: int) -> int:
    """Context subs sent to the model - anchored near 2k, a bit more for huge lists."""
    if n <= 20000:
        return 2000
    if n <= 100000:
        return 3000
    return 4000


def _auto_min_values(n: int) -> int:
    """Miner slot threshold scaled to list size (bigger list -> stricter, less noise)."""
    if n < 500:
        return 2
    if n < 5000:
        return 3
    if n < 20000:
        return 4
    return 5


def _resolve_auto(value, n: int, auto_fn, label: str) -> int:
    """Turn an int-or-'auto' flag into an int, printing what 'auto' chose."""
    if str(value).strip().lower() == "auto":
        resolved = auto_fn(n)
        print(f"  {label}: auto -> {resolved} (for {n:,} subs)")
        return resolved
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"Invalid --{label} value {value!r}; expected an integer or 'auto'.", file=sys.stderr)
        raise SystemExit(2)


def _print_themes(themes) -> None:
    print(f"\n{len(themes)} theme(s):\n")
    total = 0
    for t in themes:
        n = cardinality(t)
        total += n
        print(f"  [{t.kind:9}] {t.name}  (~{n:,} candidates)")
    print(f"\n  total implied candidates: ~{total:,}")


def _propose(subs, covered, backend, args, runs, prog):
    """Dispatch a propose run to the selected backend."""
    kind, model = backend
    from .propose import run_propose_local, run_propose_openai

    if kind == "gpt":
        return run_propose_openai(
            subs, model=model, count=args.propose_count, sample=args.sample,
            covered_templates=covered, base_url=args.api_base, runs=runs, progress=prog,
        )
    return run_propose_local(
        subs, model=model, count=args.propose_count, sample=args.sample,
        covered_templates=covered, url=args.ollama_url, runs=runs, progress=prog,
    )


def _enrich(mined, subs, residual, backend, args, runs, prog):
    """Dispatch an enrich run to the selected backend."""
    kind, model = backend
    from .enrich import run_enrich_local, run_enrich_openai

    if kind == "gpt":
        return run_enrich_openai(
            mined, subs, residual=residual, model=model, sample=args.sample,
            base_url=args.api_base, runs=runs, progress=prog,
        )
    return run_enrich_local(
        mined, subs, residual=residual, model=model, sample=args.sample,
        url=args.ollama_url, runs=runs, progress=prog,
    )


def _run_fuzz_standalone(args, subs, src) -> int:
    """`--fuzz` alone: STREAM the full expansion to --out.

    Generates per-host mutations + typed-slot fills + FUZZ-from-wordlist and writes
    each host as it goes, so any wordlist size works without a memory blowup.
    Streaming means no global dedup (whatever consumes it -- dnsx/httpx -- dedups).
    """
    from .fuzz import WORDS, generate, iter_expand_templates, load_wordlist

    words = WORDS | (load_wordlist(args.wordlist) if args.wordlist else set())
    print(f"Loaded {len(subs):,} subdomain(s) from {src}")
    if not args.wordlist:
        print(f"Note: no --wordlist; filling FUZZ from the built-in {len(WORDS)}-word list.")
    fuzz_t, concrete = generate(subs, words=words)
    cap = _cap(args.max_candidates)
    n = 0
    with _out_stream(args.out) as f:
        for host in concrete:                                   # bounded mutations
            f.write(host + "\n")
            n += 1
        for host in iter_expand_templates(fuzz_t, words):       # streamed template fills
            f.write(host + "\n")
            n += 1
            if cap is not None and n >= cap:
                print(f"Hit --max-candidates cap ({cap:,}); stopping.")
                break
    print(f"Wrote {n:,} fuzz candidates to {args.out}")
    return 0


def main(argv=None) -> int:
    global _REAL_STDOUT
    args = parse_args(argv)

    # `--out -` streams results to stdout; reroute logs to stderr by swapping stdout.
    _REAL_STDOUT = sys.stdout
    if args.out == "-":
        sys.stdout = sys.stderr

    # Backend selection (at most one).
    if args.gpt is not None and args.ollama is not None:
        print("Use only one backend: --gpt or --ollama.", file=sys.stderr)
        return 2
    backend = ("gpt", args.gpt) if args.gpt is not None else \
              ("ollama", args.ollama) if args.ollama is not None else None

    # Input: a single domain (-d) or a file of subs (positional).
    if args.domain:
        subs, src = [args.domain.strip().lower()], args.domain
    elif args.input:
        subs, src = load_subdomains(args.input), args.input
    else:
        print("Provide a domain with -d, or a subs file as the argument.", file=sys.stderr)
        return 1
    if not subs:
        print(f"No subdomains found in {src}", file=sys.stderr)
        return 1

    args.min_values = _resolve_auto(args.min_values, len(subs), _auto_min_values, "min-values")
    args.sample = _resolve_auto(args.sample, len(subs), _auto_sample, "sample")
    known = {s.strip().lower() for s in subs if s.strip()}

    # Standalone: load themes JSON and expand only (no model).
    if args.themes:
        themes = _load_and_expand_themes(args, known, src)
        return themes

    # Standalone: --fuzz alone emits templates (for ffuf/altdns handoff).
    if args.fuzz and not (args.mine or args.propose):
        return _run_fuzz_standalone(args, subs, src)

    # --- unified candidate generation ---------------------------------------
    explicit = args.mine or args.fuzz or args.propose
    run_mine = args.mine or not explicit
    run_fuzz = args.fuzz or not explicit
    run_propose = args.propose or (not explicit and backend is not None)
    if run_propose and backend is None:
        print("--propose needs a backend: add --gpt or --ollama.", file=sys.stderr)
        return 2

    print(f"Loaded {len(subs):,} subdomains from {src}")
    from .mine import mine_themes

    candidates: set[str] = set()
    themes = []

    if run_mine:
        themes = mine_themes(subs, min_values=args.min_values)
        print(f"Mined {len(themes)} template(s).")
        if backend is not None:
            from .enrich import enrichment_stats
            from .mine import residual_hosts

            residual = residual_hosts(subs, themes)
            runs = max(1, _resolve_auto(args.ai_runs, len(residual), _auto_runs, "ai-runs"))
            print(f"Enriching {len(themes)} template(s) via {backend[1]} "
                  f"(+{runs} discovery pass(es) over {len(residual):,} residual) ...")
            prog = lambda i, n, tot: print(f"  call {i}/{n}: {tot} themes so far")
            enriched = _enrich(themes, subs, residual, backend, args, runs, prog)
            stats = enrichment_stats(themes, enriched)
            print(f"  AI added {stats['new_values']:,} value(s): {stats['enriched']} expanded, "
                  f"{stats['discovered']} discovered.")
            themes = list(themes) + enriched
        if args.dry_run:
            _print_themes(themes)
            print("\nDry run: nothing written.")
            return 0
        candidates |= set(expand_themes(themes, known, _cap(args.per_theme_cap), _cap(args.max_candidates)))

    if run_fuzz:
        from .fuzz import WORDS, expand_templates, generate, load_wordlist

        words = WORDS | (load_wordlist(args.wordlist) if args.wordlist else set())
        fuzz_t, fuzz_cand = generate(subs, words=words)
        typed = expand_templates(fuzz_t, words, include_fuzz=False)  # bounded (no FUZZxwordlist)
        candidates |= set(fuzz_cand) | set(typed)
        print(f"Fuzz added {len(fuzz_cand) + len(typed):,} candidate(s).")

    if run_propose:
        from .mine import mine_themes as _mt

        covered = [t.template for t in (themes or _mt(subs, min_values=args.min_values))]
        runs = max(1, _resolve_auto(args.ai_runs, len(subs), _auto_runs, "ai-runs"))
        print(f"Proposing net-new via {backend[1]} ({runs} run(s)) ...")
        prog = lambda i, n, tot: print(f"  run {i}/{n}: {tot:,} unique so far")
        try:
            guesses = _propose(subs, covered, backend, args, runs, prog)
            candidates |= set(guesses)
            print(f"Propose added {len(guesses):,} name(s).")
        except (SystemExit, Exception) as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  propose skipped ({msg[:80]}).")

    out = sorted(candidates - known)
    with _out_stream(args.out) as f:
        f.write("\n".join(out) + "\n")
    print(f"\nWrote {len(out):,} candidates to {args.out}")
    if args.patterns and themes:
        write_patterns(themes, args.patterns)
        print(f"Wrote {len(themes)} themes to {args.patterns}")
    return 0


def _load_and_expand_themes(args, known, src) -> int:
    """--themes: expand a themes JSON to candidates, no model."""
    from .pipeline import load_themes

    themes = load_themes(args.themes)
    print(f"Loaded {len(themes)} theme(s) from {args.themes} (no model call).")
    if args.dry_run:
        _print_themes(themes)
        print("\nDry run: nothing written.")
        return 0
    candidates = expand_themes(themes, known, _cap(args.per_theme_cap), _cap(args.max_candidates))
    with _out_stream(args.out) as f:
        f.write("\n".join(candidates) + "\n")
    print(f"\nWrote {len(candidates):,} candidates to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line interface for subpattern."""

from __future__ import annotations

import argparse
import contextlib
import sys

from . import pricing
from .expand import cardinality
from .pipeline import (
    expand_themes,
    load_subdomains,
    run_discovery,
    write_patterns,
)
from .review import apply_refine, filter_themes


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="subpattern",
        description="Discover subdomain naming conventions, then expand them into a "
        "candidate wordlist. Pick ONE generator mode (--mine / --fuzz / --propose / "
        "--all); default is LLM discovery. Add --ai to layer AI onto --mine.",
    )

    # --- input ---------------------------------------------------------------
    p.add_argument(
        "input",
        nargs="?",
        help="File of subdomains, one per line. Omit if using -d/--domain.",
    )
    p.add_argument(
        "-d", "--domain",
        help="A single domain/subdomain to process instead of an input file.",
    )

    # --- generator modes -----------------------------------------------------
    modes = p.add_argument_group(
        "generator modes", "Pick one. With none, the default is LLM discovery."
    )
    modes.add_argument(
        "--mine", action="store_true",
        help="Detect templates from the input with code (offline, free). "
        "Add --ai to enrich/expand slots with an LLM.",
    )
    modes.add_argument(
        "--fuzz", action="store_true",
        help="Per-host FUZZ generator (offline, no API): FUZZ templates + closed-vocab swaps.",
    )
    modes.add_argument(
        "--propose", action="store_true",
        help="LLM proposes net-new names (infra / target-specific / theme continuation) "
        "that templates can't produce.",
    )
    modes.add_argument(
        "--ai", action="store_true",
        help="Layer AI onto --mine (or alone): expand each mined slot's values with "
        "real-world members and discover new slots; code re-expands.",
    )
    modes.add_argument(
        "--hybrid", action="store_true",
        help="Mine with code, then LLM-discover ONLY the residual (best coverage, lowest cost).",
    )
    modes.add_argument(
        "--all", action="store_true",
        help="Run mine + fuzz + propose and merge into one deduped candidate file.",
    )
    modes.add_argument(
        "--enrich", action="store_true", help="Deprecated alias for `--mine --ai`.",
    )

    # --- --mine options ------------------------------------------------------
    mine_g = p.add_argument_group("--mine options")
    mine_g.add_argument(
        "--min-values", default="auto",
        help="Min distinct values for a position to become a slot. An integer, or "
        "'auto' (default) to scale with list size (small->2, large->4-5).",
    )

    # --- --fuzz options ------------------------------------------------------
    fuzz_g = p.add_argument_group("--fuzz options")
    fuzz_g.add_argument("--top-words", type=int, default=None, help="Max words per label (overrides config).")
    fuzz_g.add_argument("--min-word-len", type=int, default=None, help="Min word length (overrides config).")
    fuzz_g.add_argument(
        "--config", default="fuzz.config.json",
        help="JSON config toggling template types (default: fuzz.config.json).",
    )
    fuzz_g.add_argument(
        "--wordlist", metavar="PATH",
        help="Dictionary file for label segmentation + FUZZ fill, unioned with the "
        "built-in list. Use a big list (e.g. SecLists) for real coverage.",
    )
    fuzz_g.add_argument(
        "--expand", action="store_true",
        help="Also fill slots into concrete hosts (deduped, in-memory) -> <out>_expanded.txt.",
    )
    fuzz_g.add_argument(
        "--expand-file", metavar="PATH",
        help="Read a FUZZ-templates file and STREAM the expansion to --out (no subs "
        "needed, no memory blowup). Fills FUZZ from --wordlist, {num}/{env}/... from "
        "their lists. Honors --max-candidates. Use for huge wordlists that hang --expand.",
    )

    # --- --propose / --ai options (LLM backends + tuning) --------------------
    ai_g = p.add_argument_group(
        "--propose / --ai options", "AI backend (default Anthropic) and tuning."
    )
    ai_g.add_argument(
        "--openai", nargs="?", const="gpt-4o-mini", default=None, metavar="MODEL",
        help="Use an OpenAI-compatible model. Bare --openai uses gpt-4o-mini; pass a "
        "model to override (e.g. --openai gpt-4o). Reads $OPENAI_API_KEY.",
    )
    ai_g.add_argument(
        "--api-base", default="https://api.openai.com/v1",
        help="Base URL for --openai (Groq, Together, local vLLM, Ollama's /v1).",
    )
    ai_g.add_argument(
        "--local", metavar="MODEL",
        help="Use a local Ollama model - offline, free, private (e.g. --local qwen2.5). "
        "Needs `ollama serve`.",
    )
    ai_g.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Base URL of the Ollama server (default: http://localhost:11434).",
    )
    ai_g.add_argument("--model", default="claude-opus-4-8", help="Anthropic model id (default backend).")
    ai_g.add_argument("--propose-count", type=int, default=1000, help="Propose: max guesses to request.")
    ai_g.add_argument(
        "--sample", default="auto",
        help="Max subs sent as context - propose (sub sample) and enrich (residual "
        "sample). Lists at/below this send ALL; larger are sampled (seeded; varies per "
        "--ai-runs). An integer, or 'auto' (default).",
    )
    ai_g.add_argument(
        "--ai-runs", "--local-runs", dest="ai_runs", default="auto", metavar="N",
        help="--local/--openai: call the model N times with different seeds and merge "
        "(more distinct names; big lists sample a fresh slice per run). An integer, or "
        "'auto' (default) to scale with list size. (--local-runs is a deprecated alias.)",
    )

    # --- LLM discovery options (default mode / --hybrid) ---------------------
    disc_g = p.add_argument_group("LLM discovery options")
    disc_g.add_argument(
        "--chunk-size", type=int, default=20000,
        help="Subdomains per discovery call. Larger inputs are map-reduced.",
    )
    disc_g.add_argument(
        "--runs", type=int, default=1,
        help="Discovery passes; each after the first deepens on the previous. 2-3 is good.",
    )
    disc_g.add_argument(
        "--refine", action="store_true",
        help="After mining/discovery, a cheap LLM pass to name, rank, and drop low-value templates.",
    )
    disc_g.add_argument(
        "--critic", action="store_true",
        help="After discovery, drop themes whose evidence doesn't support them (one LLM call).",
    )
    disc_g.add_argument(
        "--print-prompt", action="store_true",
        help="Print the discovery prompt (to paste into claude.ai), then exit; feed its JSON via --themes.",
    )
    disc_g.add_argument(
        "--themes", metavar="PATH",
        help="Load themes from a JSON file and skip the API (expand/output only). "
        'Accepts a patterns.json or {"themes": [...]}.',
    )

    # --- output & limits -----------------------------------------------------
    out_g = p.add_argument_group("output & limits")
    out_g.add_argument(
        "--out", default="candidates.txt",
        help="Where to write candidates ('-' = stdout; logs go to stderr).",
    )
    out_g.add_argument(
        "--patterns", metavar="PATH", default=None,
        help="Opt-in: also write the discovered themes as JSON here (inspection, "
        "--themes reuse, gotator -perm feed). Omitted by default.",
    )
    out_g.add_argument(
        "--per-theme-cap", type=int, default=50000,
        help="Max candidates expanded per theme (0 = unlimited).",
    )
    out_g.add_argument(
        "--max-candidates", type=int, default=200000,
        help="Global cap on total candidates (0 = unlimited).",
    )
    out_g.add_argument(
        "--dry-run", action="store_true",
        help="Discover and preview cardinalities only; do not expand or write candidates.",
    )
    return p.parse_args(argv)


def _cap(value: int) -> int | None:
    return None if value == 0 else value


# The real stdout, captured before any `--out -` redirect so results can still
# reach the pipe while logs go to stderr.
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
    """Propose runs scaled to list size (bigger list -> more sampled slices covered)."""
    if n <= 2000:
        return 2
    if n <= 5000:
        return 5
    if n <= 10000:
        return 8
    return 10


def _auto_sample(n: int) -> int:
    """Context subs sent to the model - anchored near 2k, a bit more for huge lists.
    (Bigger isn't strictly better: more context costs tokens and can crowd the reply.)"""
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


def _resolve_auto(value: str, n: int, auto_fn, label: str) -> int:
    """Turn an int-or-'auto' flag string into an int, printing what 'auto' chose."""
    if str(value).strip().lower() == "auto":
        resolved = auto_fn(n)
        print(f"  {label}: auto -> {resolved} (for {n:,} subs)")
        return resolved
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"Invalid --{label} value {value!r}; expected an integer or 'auto'.",
              file=sys.stderr)
        raise SystemExit(2)


def _print_themes(themes) -> None:
    print(f"\nDiscovered {len(themes)} theme(s):\n")
    total = 0
    for t in themes:
        n = cardinality(t)
        total += n
        ev = ", ".join(t.evidence[:3])
        print(f"  [{t.kind:9}] {t.name}  (~{n:,} candidates)")
        print(f"              {t.description}")
        if ev:
            print(f"              e.g. {ev}")
    print(f"\n  total implied candidates: ~{total:,}")


def _run_all(args, subs, src) -> int:
    """mine + fuzz + propose -> one deduped candidate file."""
    from .fuzz import WORDS, expand_templates, generate, load_config, load_wordlist
    from .mine import mine_themes
    from .pipeline import expand_themes
    from .propose import run_propose

    known = {s.strip().lower() for s in subs if s.strip()}
    print(f"Loaded {len(subs):,} subdomains from {src}")

    # 1. mine - structural templates -> concrete candidates
    print("[1/3] mining structural templates ...")
    mined = mine_themes(subs, min_values=args.min_values)
    mine_cands = expand_themes(mined, known, _cap(args.per_theme_cap), _cap(args.max_candidates))

    # 2. fuzz - per-host mutations + typed-slot fills (no FUZZ x wordlist explosion here)
    print("[2/3] fuzzing per-host mutations ...")
    cfg, top, min_len = load_config(args.config)
    words = WORDS | (load_wordlist(args.wordlist) if args.wordlist else set())
    fuzz_t, fuzz_cand = generate(subs, words=words, top_n=top, min_len=min_len, config=cfg)
    typed = expand_templates(fuzz_t, words, include_fuzz=False)
    fuzz_all = set(fuzz_cand) | set(typed)

    # 3. propose - knowledge-based net-new (LLM; local Ollama, OpenAI, or Anthropic; skip gracefully)
    from .propose import run_propose_local, run_propose_openai

    guesses = []
    covered = [t.template for t in mined]
    runs = max(1, _resolve_auto(args.ai_runs, len(subs), _auto_runs, "ai-runs"))
    if args.local:
        suffix = f", {runs} runs merged" if runs > 1 else ""
        print(f"[3/3] proposing net-new with local model '{args.local}' (offline{suffix}) ...")
        _prog = lambda i, n, tot: print(f"      run {i}/{n}: {tot:,} unique so far")
        propose_call = lambda: run_propose_local(
            subs, model=args.local, count=args.propose_count, sample=args.sample,
            covered_templates=covered, url=args.ollama_url, runs=runs, progress=_prog,
        )
    elif args.openai:
        print(f"[3/3] proposing net-new with '{args.openai}' via {args.api_base} ...")
        propose_call = lambda: run_propose_openai(
            subs, model=args.openai, count=args.propose_count, sample=args.sample,
            covered_templates=covered, base_url=args.api_base, runs=runs,
        )
    else:
        pmodel = args.model if args.model != "claude-opus-4-8" else "claude-sonnet-4-6"
        print(f"[3/3] proposing net-new with {pmodel} ...")
        propose_call = lambda: run_propose(
            subs, model=pmodel, count=args.propose_count, sample=args.sample,
            covered_templates=covered,
        )
    try:
        guesses = propose_call()
    except (SystemExit, Exception) as e:  # no key/Ollama or API error -> keep mine + fuzz
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"  propose skipped ({msg[:80]}); merging mine + fuzz only.")

    merged = sorted((set(mine_cands) | fuzz_all | set(guesses)) - known)
    with _out_stream(args.out) as f:
        f.write("\n".join(merged) + "\n")
    print(f"\nMerged {len(merged):,} unique candidates -> {args.out}")
    print(f"  mine: {len(mine_cands):,}  |  fuzz: {len(fuzz_all):,}  |  propose: {len(guesses):,}")
    pricing.print_session_summary()
    return 0


def main(argv=None) -> int:
    global _REAL_STDOUT
    args = parse_args(argv)

    # `--out -` streams results to stdout; send all logs/progress to stderr so
    # they don't pollute a pipe. Every status print() goes through sys.stdout, so
    # swapping it here reroutes them without touching each call site.
    _REAL_STDOUT = sys.stdout
    if args.out == "-":
        sys.stdout = sys.stderr

    # Stream-expand a FUZZ-templates file into concrete hosts (no subs needed).
    if args.expand_file:
        from .fuzz import WORDS, iter_expand_templates, load_wordlist

        words = WORDS | load_wordlist(args.wordlist) if args.wordlist else WORDS
        with open(args.expand_file, encoding="utf-8") as f:
            templates = [ln.strip() for ln in f if ln.strip()]
        print(f"Loaded {len(templates):,} templates from {args.expand_file}")
        if not args.wordlist:
            print(f"Note: no --wordlist; filling FUZZ from the built-in {len(WORDS)}-word list.")
        cap = _cap(args.max_candidates)
        n = 0
        with _out_stream(args.out) as f:
            for host in iter_expand_templates(templates, words):
                f.write(host + "\n")
                n += 1
                if n % 1_000_000 == 0:
                    print(f"  wrote {n:,} ...")
                if cap is not None and n >= cap:
                    print(f"Hit --max-candidates cap ({cap:,}); stopping.")
                    break
        print(f"Wrote {n:,} expanded hosts to {args.out}")
        return 0

    # Input: a single domain (-d) or a file of subs (positional).
    if args.domain:
        subs = [args.domain.strip().lower()]
        src = args.domain
    elif args.input:
        subs = load_subdomains(args.input)
        src = args.input
    else:
        print("Provide a single domain with -d/--domain, or a subs file as the argument.", file=sys.stderr)
        return 1
    if not subs:
        print(f"No subdomains found in {src}", file=sys.stderr)
        return 1

    # Resolve int-or-'auto' flags to ints now that we know the list size, so every
    # downstream branch reads a plain int.
    args.min_values = _resolve_auto(args.min_values, len(subs), _auto_min_values, "min-values")
    args.sample = _resolve_auto(args.sample, len(subs), _auto_sample, "sample")
    # --ai-runs is resolved per-mode below (propose scales high, enrich stays modest).

    # Free, no-API: print the discovery prompt to paste into claude.ai / Pro.
    if args.print_prompt:
        from .discover import SCHEMA_HINT, build_prompt

        system, user = build_prompt(subs)
        print(
            "Paste the two blocks below into claude.ai (covered by your Pro plan), "
            "save the JSON it returns to a file, then run:\n"
            f"    subpattern {src} --themes <that-file.json>\n"
        )
        print("===== SYSTEM =====\n" + system)
        print("\n===== USER =====\n" + user)
        print("\n===== OUTPUT FORMAT =====\n" + SCHEMA_HINT)
        return 0

    # Per-host FUZZ generation - offline, standalone (own output, no expand chain).
    if args.fuzz:
        from .fuzz import WORDS, generate, load_config, load_wordlist

        if args.ai:
            print("Note: --ai with --fuzz (AI word segmentation) is not implemented yet; "
                  "running plain fuzz. Use --mine --ai for AI slot enrichment.")

        cfg, cfg_top, cfg_min = load_config(args.config)
        top = args.top_words if args.top_words is not None else cfg_top
        min_len = args.min_word_len if args.min_word_len is not None else cfg_min
        words = WORDS
        if args.wordlist:
            words = WORDS | load_wordlist(args.wordlist)
            print(f"Loaded {len(words):,} segmentation words ({args.wordlist} + built-in)")
        enabled = [k for k, v in cfg.items() if v]
        print(f"Loaded {len(subs):,} subdomain(s) from {src}")
        print(f"Generating FUZZ templates (offline, no API): {len(enabled)} template types enabled ...")
        fuzz_t, cand = generate(subs, words=words, top_n=top, min_len=min_len, config=cfg)
        with _out_stream(args.out) as f:
            f.write("\n".join(fuzz_t) + "\n")
        print(f"Wrote {len(fuzz_t):,} FUZZ templates to {args.out}")
        base = args.out.rsplit(".", 1)[0] if "." in args.out else args.out
        if cand:
            cpath = base + "_candidates.txt"
            with open(cpath, "w", encoding="utf-8") as f:
                f.write("\n".join(cand) + "\n")
            print(f"Wrote {len(cand):,} concrete candidates (separator+plural+affix) to {cpath}")
        if args.expand:
            from .fuzz import expand_templates

            if not args.wordlist:
                print("Note: --expand with no --wordlist fills FUZZ from the small built-in list.")
            expanded = expand_templates(fuzz_t, words)
            epath = base + "_expanded.txt"
            with open(epath, "w", encoding="utf-8") as f:
                f.write("\n".join(expanded) + "\n")
            print(f"Expanded slots into {len(expanded):,} concrete hosts to {epath}")
        return 0

    # Run all three generators and merge into one deduped candidate file.
    if args.all:
        return _run_all(args, subs, src)

    # LLM proposes net-new subdomains (knowledge-based). API (Sonnet) or local Ollama.
    if args.propose:
        from .mine import mine_themes
        from .propose import run_propose, run_propose_local, run_propose_openai

        used = min(len(subs), args.sample)
        # Mine first (free) so propose knows what structure is already covered.
        covered = [t.template for t in mine_themes(subs, min_values=args.min_values)]
        print(f"Loaded {len(subs):,} subdomains from {src}")
        runs = max(1, _resolve_auto(args.ai_runs, len(subs), _auto_runs, "ai-runs"))
        suffix = f", {runs} runs merged" if runs > 1 else ""
        _prog = lambda i, n, tot: print(f"  run {i}/{n}: {tot:,} unique so far")
        if args.local:
            print(f"Proposing net-new subdomains with local model '{args.local}' "
                  f"(offline{suffix}, context: {used:,} subs, avoiding {len(covered)} templates) ...")
            guesses = run_propose_local(
                subs, model=args.local, count=args.propose_count, sample=args.sample,
                covered_templates=covered, url=args.ollama_url,
                runs=runs, progress=_prog,
            )
        elif args.openai:
            print(f"Proposing net-new subdomains with '{args.openai}' via {args.api_base} "
                  f"({suffix.lstrip(', ') or '1 run'}, context: {used:,} subs, avoiding {len(covered)} templates) ...")
            guesses = run_propose_openai(
                subs, model=args.openai, count=args.propose_count, sample=args.sample,
                covered_templates=covered, base_url=args.api_base,
                runs=runs, progress=_prog if runs > 1 else None,
            )
        else:
            pmodel = args.model if args.model != "claude-opus-4-8" else "claude-sonnet-4-6"
            print(f"Proposing net-new subdomains with {pmodel} "
                  f"(context: {used:,} subs, avoiding {len(covered)} templates) ...")
            guesses = run_propose(
                subs, model=pmodel, count=args.propose_count, sample=args.sample,
                covered_templates=covered,
            )
        with _out_stream(args.out) as f:
            f.write("\n".join(guesses) + "\n")
        print(f"Wrote {len(guesses):,} proposed new subdomains to {args.out}")
        pricing.print_session_summary()
        return 0

    # Mine, then have the LLM expand slot values AND discover new slots; expand in code.
    # Triggered by `--mine --ai`, bare `--ai`, or the deprecated `--enrich` alias.
    if args.ai or args.enrich:
        from .enrich import (
            enrichment_stats,
            run_enrich,
            run_enrich_local,
            run_enrich_openai,
        )
        from .expand import expand_all
        from .mine import mine_themes, residual_hosts

        known = {s.strip().lower() for s in subs if s.strip()}
        print(f"Loaded {len(subs):,} subdomains from {src}")
        mined = mine_themes(subs, min_values=args.min_values)
        residual = residual_hosts(subs, mined)
        print(f"Mined {len(mined)} template(s); {len(residual):,} residual host(s).")
        # Discovery (job B) is Monte Carlo over the residual, so scale runs off the
        # RESIDUAL size (template enrichment / job A runs once regardless).
        runs = max(1, _resolve_auto(args.ai_runs, len(residual), _auto_runs, "ai-runs"))
        if args.local:
            print(f"Enriching slots with local model '{args.local}' (offline; "
                  f"{len(mined)} templates once + {runs} discovery pass(es)) ...")
            _prog = lambda i, n, tot: print(f"  call {i}/{n}: {tot} themes so far")
            enriched = run_enrich_local(
                mined, subs, residual=residual, model=args.local, sample=args.sample,
                url=args.ollama_url, runs=runs, progress=_prog,
            )
        elif args.openai:
            print(f"Enriching slots with '{args.openai}' via {args.api_base} "
                  f"({len(mined)} templates once + {runs} discovery pass(es)) ...")
            _prog = lambda i, n, tot: print(f"  call {i}/{n}: {tot} themes so far")
            enriched = run_enrich_openai(
                mined, subs, residual=residual, model=args.openai, sample=args.sample,
                base_url=args.api_base, runs=runs, progress=_prog,
            )
        else:
            pmodel = args.model if args.model != "claude-opus-4-8" else "claude-sonnet-4-6"
            print(f"Enriching slots with {pmodel} ...")
            enriched = run_enrich(
                mined, subs, residual=residual, model=pmodel, sample=args.sample,
            )
        stats = enrichment_stats(mined, enriched)
        print(f"AI added {stats['new_values']:,} new slot value(s): "
              f"{stats['enriched']} existing template(s) expanded, "
              f"{stats['discovered']} new template(s) discovered.")
        cands = expand_all(
            list(mined) + enriched, known,
            per_theme_cap=_cap(args.per_theme_cap), max_candidates=_cap(args.max_candidates),
        )
        with _out_stream(args.out) as f:
            f.write("\n".join(cands) + "\n")
        print(f"Wrote {len(cands):,} candidates (mined + enriched) to {args.out}")
        pricing.print_session_summary()
        return 0

    known = set(subs)
    print(f"Loaded {len(subs):,} unique subdomains from {src}")

    # Free, no-API: load themes produced elsewhere and skip discovery entirely.
    if args.themes:
        from .pipeline import load_themes

        themes = load_themes(args.themes)
        print(f"Loaded {len(themes)} theme(s) from {args.themes} (no API call).")
        if args.critic:
            print("Note: --critic needs the API; ignoring it in --themes mode.")
    # Free, no-API: detect patterns from the data with code.
    elif args.mine:
        from .mine import mine_themes

        print("Mining patterns offline (no API) ...")
        themes = mine_themes(subs, min_values=args.min_values)
        print(f"Detected {len(themes)} template(s).")
    # Hybrid: mine structure with code, then LLM-discover only the residual.
    elif args.hybrid:
        from .mine import mine_themes, residual_hosts

        print("Mining structural patterns (offline) ...")
        mined = mine_themes(subs, min_values=args.min_values)
        residual = residual_hosts(subs, mined)
        covered = len(subs) - len(residual)
        print(f"  {len(mined)} templates cover {covered:,} hosts; {len(residual):,} residual one-offs")
        discovered = []
        if residual:
            print(f"Running LLM discovery on {len(residual):,} residual hosts with {args.model} ...")
            discovered = run_discovery(
                residual, model=args.model, chunk_size=args.chunk_size, runs=args.runs
            )
        themes = mined + discovered
        print(f"  total: {len(themes)} themes ({len(mined)} mined + {len(discovered)} discovered)")
    else:
        print(f"Running discovery with {args.model} ...")
        themes = run_discovery(
            subs, model=args.model, chunk_size=args.chunk_size, runs=args.runs
        )
        if args.critic:
            from .critic import run_critic

            print("Running critic pass ...")
            report = run_critic(themes, model=args.model)
            themes, dropped = filter_themes(themes, report.verdicts, known=known)
            for t, reason in dropped:
                print(f"  dropped: {t.name} - {reason}")
            print(f"  kept {len(themes)} theme(s), dropped {len(dropped)}")

    if args.refine and not args.themes:
        from .refine import run_refine

        print("Refining templates (LLM name/rank/drop) ...")
        report = run_refine(themes, model=args.model)
        before = len(themes)
        themes = apply_refine(themes, report.judgments)
        print(f"  kept {len(themes)} of {before}, renamed + ranked")

    _print_themes(themes)

    if args.dry_run:
        if args.patterns:
            write_patterns(themes, args.patterns)
            print(f"\nDry run: wrote themes to {args.patterns} (no candidates generated).")
        else:
            print("\nDry run: no candidates generated. Pass --patterns PATH to save the specs.")
        pricing.print_session_summary()
        return 0

    candidates = expand_themes(
        themes,
        known,
        per_theme_cap=_cap(args.per_theme_cap),
        max_candidates=_cap(args.max_candidates),
    )
    with _out_stream(args.out) as f:
        f.write("\n".join(candidates) + "\n")
    print(f"\nWrote {len(candidates):,} new candidates to {args.out}")
    if args.patterns:
        write_patterns(themes, args.patterns)
        print(f"Wrote {len(themes)} themes to {args.patterns}")
    pricing.print_session_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

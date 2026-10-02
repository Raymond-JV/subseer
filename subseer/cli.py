"""Command-line interface for subseer."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from pathlib import Path

from . import term
from .pipeline import expand_themes, load_subdomains, theme_to_dict


def _auto_int(value: str):
    """argparse type: a positive int, or 'auto'."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer or 'auto', got {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


class _UsageError(SystemExit):
    """A rejected command line (exit 2); carries the message for the run log."""

    def __init__(self, message: str):
        super().__init__(2)
        self.message = message


class _Parser(argparse.ArgumentParser):
    """argparse, but usage errors print a loud ERROR line instead of the usage dump."""

    def error(self, message):
        term.error(message)
        print("Run 'subseer -h' for usage.", file=sys.stderr)
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="subseer",
        description="Generate new subdomains from the ones you already know. Runs "
        "offline by default (Mine + Fuzz); add --gpt or --ollama to use an LLM.",
    )
    p.add_argument("input", nargs="?", help="File of subdomains, one per line.")
    p.add_argument("-d", "--domain", help="A single domain instead of a file.")
    p.add_argument("-q", "--quiet", action="store_true", help="Results and errors only.")
    p.add_argument("-v", "--version", action="store_true", help="Show the version and exit.")

    m = p.add_argument_group("Mine (offline)",
                             "Fill the naming templates found in your list.")
    m.add_argument("--mine", action="store_true", help="Run Mine.")
    m.add_argument("--min-values", metavar="N", default="auto", type=_auto_int,
                   help="Distinct values a slot needs (default auto).")

    f = p.add_argument_group("Fuzz (offline)",
                             "Permute each host, like gotator or altdns.")
    f.add_argument("--fuzz", action="store_true", help="Run Fuzz.")
    f.add_argument("-w", "--wordlist", metavar="PATH",
                   help="Wordlist (default: bundled SecLists top 20k, "
                   "or $SUBSEER_WORDLIST).")

    pr = p.add_argument_group("Predict (needs an LLM)",
                              "An LLM infers likely subdomains from your list.")
    pr.add_argument("--predict", action="store_true", help="Run Predict.")
    pr.add_argument("--predict-count", metavar="N", type=int, default=None,
                    help="Max names to request (default 1000).")

    ai = p.add_argument_group("LLM", "Required for Predict; enhances Mine.")
    ai.add_argument("--gpt", nargs="?", const="gpt-4o-mini", default=None, metavar="MODEL",
                    help="OpenAI-compatible model (default gpt-4o-mini). "
                    "Reads $OPENAI_API_KEY.")
    ai.add_argument("--ollama", nargs="?", const="qwen2.5", default=None, metavar="MODEL",
                    help="Local Ollama model (default qwen2.5).")
    ai.add_argument("--api-base", metavar="URL", default=None,
                    help="LLM endpoint (default: OpenAI, or localhost:11434 for --ollama).")
    ai.add_argument("--sample", metavar="N", default=None, type=_auto_int,
                    help="Subs sent to the model per call (default auto).")
    ai.add_argument("--ai-runs", default=None, metavar="N", type=_auto_int,
                    help="Model calls to merge (default auto).")

    o = p.add_argument_group("Output")
    o.add_argument("-o", "--out", metavar="PATH", default="-",
                   help="Write to a file (default: stdout, progress on stderr).")
    o.add_argument("--limit", metavar="N", type=int, default=200000,
                   help="Max results (default 200000; 0 = no limit).")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args(argv)
    _check_args(p, args)
    args.sample = args.sample or "auto"
    args.ai_runs = args.ai_runs or "auto"
    args.predict_count = args.predict_count or 1000
    return args


# A bare hostname: labels of letters, digits, '-', '_' (and a leading '*'), with a dot.
_HOSTNAME = re.compile(r"^(\*\.)?[a-z0-9_-]+(\.[a-z0-9_-]+)+$")


def _check_args(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject bad flag combinations up front, before any input is read."""
    if args.domain and args.input:
        p.error("use -d DOMAIN or a subs file, not both")
    if not args.domain and not args.input and not args.version:
        p.error("provide a domain with -d, or a subs file as the argument")
    if args.domain:
        args.domain = args.domain.strip().lower().rstrip(".")
        if not _HOSTNAME.match(args.domain):
            p.error(f"-d expects a hostname like example.com, got {args.domain!r}")
    if args.input and not os.path.isfile(args.input):
        p.error(f"subs file not found: {args.input}")
    if args.gpt is not None and args.ollama is not None:
        p.error("use only one LLM: --gpt or --ollama")
    has_llm = args.gpt is not None or args.ollama is not None
    if args.predict and not has_llm:
        p.error("--predict requires --gpt or --ollama")
    if not has_llm:
        for flag, value in (("--api-base", args.api_base), ("--sample", args.sample),
                            ("--ai-runs", args.ai_runs)):
            if value is not None:
                p.error(f"{flag} requires --gpt or --ollama")
    if args.gpt is not None and not args.api_base and not os.environ.get("OPENAI_API_KEY"):
        p.error("--gpt needs $OPENAI_API_KEY (or --api-base for a keyless endpoint)")
    if args.predict_count is not None and not args.predict:
        p.error("--predict-count requires --predict")
    if args.wordlist and not os.path.isfile(args.wordlist):
        p.error(f"wordlist not found: {args.wordlist}")
    if args.predict_count is not None and args.predict_count < 1:
        p.error("--predict-count must be at least 1")
    if args.out != "-":
        out = Path(args.out)
        if not out.parent.is_dir():
            p.error(f"-o folder does not exist: {out.parent}")
        if out.is_dir():
            p.error(f"-o is a folder, expected a file path: {out}")
        if args.input and out.resolve() == Path(args.input).resolve():
            p.error("-o would overwrite the input file; pick another path")
    if args.limit < 0:
        p.error("--limit must be 0 (unlimited) or more")


# Built-in per-template cap so one runaway template can't eat the whole --limit.
_PER_TEMPLATE_CAP = 50000


def _cap(value: int) -> int | None:
    return None if value == 0 else value


# Real stdout, captured before logs are rerouted to stderr so results still reach
# the pipe (results go to stdout unless -o names a file).
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
    return int(value)


_DEFAULT_API_BASE = {"gpt": "https://api.openai.com/v1", "ollama": "http://localhost:11434"}


def _api_base(kind, args):
    """--api-base if given, else the selected backend's default endpoint."""
    return args.api_base or _DEFAULT_API_BASE[kind]


def _predict(subs, covered, backend, args, runs, prog):
    """Dispatch a --predict run to the selected backend."""
    kind, model = backend
    from .propose import run_propose_local, run_propose_openai

    if kind == "gpt":
        return run_propose_openai(
            subs, model=model, count=args.predict_count, sample=args.sample,
            covered_templates=covered, base_url=_api_base(kind, args), runs=runs, progress=prog,
        )
    return run_propose_local(
        subs, model=model, count=args.predict_count, sample=args.sample,
        covered_templates=covered, url=_api_base(kind, args), runs=runs, progress=prog,
    )


def _enrich(mined, subs, residual, backend, args, runs, prog):
    """Dispatch an enrich run to the selected backend."""
    kind, model = backend
    from .enrich import run_enrich_local, run_enrich_openai

    if kind == "gpt":
        return run_enrich_openai(
            mined, subs, residual=residual, model=model, sample=args.sample,
            base_url=_api_base(kind, args), runs=runs, progress=prog,
        )
    return run_enrich_local(
        mined, subs, residual=residual, model=model, sample=args.sample,
        url=_api_base(kind, args), runs=runs, progress=prog,
    )


def _out_desc(path: str) -> str:
    """How the run log records the output destination."""
    return "stdout" if path == "-" else str(Path(path).resolve())


def _out_name(path: str) -> str:
    """How progress messages name the output destination."""
    return "stdout" if path == "-" else path


# Bundled default DNS wordlist (SecLists top-20000, MIT; see subseer/data/README.md).
_BUNDLED_WORDLIST = "subdomains-top20000.txt"


def _resolve_words(args, log):
    """Return the fuzz/segmentation word set: the built-in tokens plus a DNS list.

    The DNS list is ``--wordlist`` if given, else ``$SUBSEER_WORDLIST`` if it points
    at a real file, else the bundled SecLists top-20000 list. Prints and logs the
    source. Falls back to just the built-in tokens only if the bundle is missing.
    """
    import importlib.resources as ir

    from .fuzz import WORDS, load_wordlist

    env = os.environ.get("SUBSEER_WORDLIST")
    path = args.wordlist or (env if env and Path(env).is_file() else None)
    if path:
        extra = load_wordlist(path)
        source = "custom" if args.wordlist else "env"
        print(f"Wordlist: {path} -- {len(extra):,} words + {len(WORDS)} built-in.")
        log.set(wordlist={"path": str(Path(path).resolve()), "words": len(extra), "source": source})
        return WORDS | extra
    try:
        res = ir.files("subseer").joinpath("data", _BUNDLED_WORDLIST)
        with ir.as_file(res) as p:
            extra = load_wordlist(str(p))
        print(f"Wordlist: bundled {_BUNDLED_WORDLIST} -- {len(extra):,} words + "
              f"{len(WORDS)} built-in (override with --wordlist or $SUBSEER_WORDLIST).")
        log.set(wordlist={"path": f"bundled:{_BUNDLED_WORDLIST}", "words": len(extra),
                          "source": "bundled"})
        return WORDS | extra
    except Exception as e:  # bundle missing/unreadable: still usable, just smaller
        print(f"Note: bundled wordlist unavailable ({e}); using the built-in "
              f"{len(WORDS)}-word list.")
        log.set(wordlist=None)
        return WORDS


def _run_fuzz_standalone(args, subs, src, log) -> int:
    """`--fuzz` alone: STREAM the full expansion to stdout or the -o file.

    Generates per-host mutations + typed-slot fills + FUZZ-from-wordlist and writes
    each host as it goes, so any wordlist size works without a memory blowup.
    Streaming means no global dedup (whatever consumes it -- dnsx/httpx -- dedups).
    """
    from .fuzz import generate, iter_expand_templates

    print(f"Loaded {len(subs):,} subdomain(s) from {src}")
    words = _resolve_words(args, log)
    fuzz_t, concrete = generate(subs, words=words)
    cap = _cap(args.limit)
    n = 0
    with _out_stream(args.out) as f:
        for host in concrete:                                   # bounded mutations
            f.write(host + "\n")
            n += 1
        for host in iter_expand_templates(fuzz_t, words):       # streamed template fills
            f.write(host + "\n")
            n += 1
            if cap is not None and n >= cap:
                print(f"Hit --limit ({cap:,}); stopping.")
                log.set(limit_hit=True)
                break
    print(f"Wrote {n:,} fuzz candidates to {_out_name(args.out)}")
    log.set(mode="fuzz-stream", generators=["fuzz"], counts={"fuzz": n},
            output={"path": _out_desc(args.out), "written": n})
    return 0


_WELCOME = (
    "usage: subseer SUBS_FILE [options]   or   subseer -d DOMAIN [options]\n"
    "       subseer --help   for all flags"
)


def main(argv=None) -> int:
    from . import __version__, banner

    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:  # bare `subseer`: the welcome screen
        banner.show()
        print(_WELCOME)
        return 0
    from .runlog import RunLog

    log = RunLog(argv, __version__)
    try:
        args = parse_args(argv)  # --help / usage errors exit here, before any banner
    except _UsageError as e:  # rejected flags: still worth a log line
        log.set(error=e.message, exit_code=2)
        log.write()
        raise
    if args.version:
        print(f"subseer {__version__}")
        return 0

    saved_stdout = sys.stdout
    rc = 1
    try:
        rc = _run(args, __version__, log)
        return rc
    except BaseException as e:  # record crashes / Ctrl-C too, then re-raise
        if isinstance(e, SystemExit):
            rc = e.code if isinstance(e.code, int) else 1
        log.set(error=f"{type(e).__name__}: {e}")
        raise
    finally:
        if sys.stdout not in (saved_stdout, sys.stderr):
            sys.stdout.close()  # the --quiet devnull
        sys.stdout = saved_stdout
        log.set(exit_code=rc)
        log.write()


def _run(args, version: str, log) -> int:
    global _REAL_STDOUT
    from . import banner

    # Results on stdout (the default) -> reroute progress logs to stderr.
    _REAL_STDOUT = sys.stdout
    if args.out == "-":
        sys.stdout = sys.stderr
    if args.quiet:  # results and errors only (errors go to stderr explicitly)
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    backend = ("gpt", args.gpt) if args.gpt is not None else \
              ("ollama", args.ollama) if args.ollama is not None else None

    # Input: a single domain (-d) or a file of subs (positional).
    if args.domain:
        subs, src = [args.domain], args.domain
    else:
        subs, src = load_subdomains(args.input), args.input
    if not subs:
        term.error(f"no subdomains found in {src}")
        log.set(error=f"no subdomains found in {src}")
        return 1
    if not args.quiet:
        banner.header(version)  # one line on stderr (TTY only); errors above stay clean
    log.set(
        input=({"kind": "domain", "source": args.domain, "count": 1, "label": args.domain}
               if args.domain else
               {"kind": "file", "source": str(Path(src).resolve()), "count": len(subs),
                "label": Path(src).stem}),
        backend={"kind": backend[0], "model": backend[1]} if backend else None,
    )

    args.min_values = _resolve_auto(args.min_values, len(subs), _auto_min_values, "min-values")
    args.sample = _resolve_auto(args.sample, len(subs), _auto_sample, "sample")
    log.set(settings={"limit": args.limit, "min_values": args.min_values, "sample": args.sample})
    known = {s.strip().lower() for s in subs if s.strip()}

    # Standalone: --fuzz alone streams the full expansion.
    if args.fuzz and not (args.mine or args.predict):
        return _run_fuzz_standalone(args, subs, src, log)

    # --- unified candidate generation ---------------------------------------
    explicit = args.mine or args.fuzz or args.predict
    run_mine = args.mine or not explicit
    run_fuzz = args.fuzz or not explicit
    run_predict = args.predict or (not explicit and backend is not None)

    print(f"Loaded {len(subs):,} subdomains from {src}")
    from .mine import mine_themes

    # Insertion-ordered so --limit keeps mined (confidence-ranked) names first.
    candidates: dict[str, None] = {}
    themes = []
    counts: dict[str, int] = {}
    log.set(mode="generate", counts=counts,
            generators=[g for g, on in (("mine", run_mine), ("fuzz", run_fuzz),
                                        ("predict", run_predict)) if on])

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
            log.set(enrichment=stats)
            themes = list(themes) + enriched
        log.set(templates=[theme_to_dict(t) for t in themes])
        mined = expand_themes(themes, known, _PER_TEMPLATE_CAP, _cap(args.limit))
        counts["mine"] = len(mined)
        candidates.update(dict.fromkeys(mined))

    if run_fuzz:
        from .fuzz import expand_templates, generate

        words = _resolve_words(args, log)
        fuzz_t, fuzz_cand = generate(subs, words=words)
        typed = expand_templates(fuzz_t, words, include_fuzz=False)  # bounded (no FUZZxwordlist)
        candidates.update(dict.fromkeys(sorted(fuzz_cand) + sorted(typed)))  # stable for --limit
        counts["fuzz"] = len(fuzz_cand) + len(typed)
        print(f"Fuzz added {counts['fuzz']:,} candidate(s).")

    if run_predict:
        from .mine import mine_themes as _mt

        covered = [t.template for t in (themes or _mt(subs, min_values=args.min_values))]
        runs = max(1, _resolve_auto(args.ai_runs, len(subs), _auto_runs, "ai-runs"))
        print(f"Predicting net-new names via {backend[1]} ({runs} run(s)) ...")
        prog = lambda i, n, tot: print(f"  run {i}/{n}: {tot:,} unique so far")
        try:
            guesses = _predict(subs, covered, backend, args, runs, prog)
            candidates.update(dict.fromkeys(guesses))
            counts["predict"] = len(guesses)
            print(f"Predict added {len(guesses):,} name(s).")
        except (SystemExit, Exception) as e:
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            term.warn(f"predict skipped ({msg[:80]})")
            log.warn(f"predict skipped: {msg}")

    out = [c for c in candidates if c not in known]
    cap = _cap(args.limit)
    if cap is not None and len(out) > cap:
        print(f"Hit --limit ({cap:,}); keeping the first {cap:,} of {len(out):,}.")
        log.set(limit_trimmed=len(out) - cap)
        out = out[:cap]
    out.sort()
    with _out_stream(args.out) as f:
        f.write("\n".join(out) + "\n")
    print(f"\nWrote {len(out):,} candidates to {_out_name(args.out)}")
    log.set(output={"path": _out_desc(args.out), "written": len(out)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

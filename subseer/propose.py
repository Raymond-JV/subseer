"""LLM proposes net-new subdomains that templates/fuzzing structurally can't.

The miner and fuzzer only recombine observed pieces. This asks the model to
generate subdomains from KNOWLEDGE - common infra the org likely runs, target-
specific names, and continuations of themes it sees - grounded on a small sample
of the known hosts so the guesses are tailored and non-duplicative.

Kept cheap by design: a bounded sample as context (not all hosts), Sonnet by
default, thinking off, modest output. ~$0.10-0.15 per target.
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request

from . import prompts
from .mine import detect_apex

SYSTEM_PROMPT = prompts.load("predict")  # subseer/prompts/predict.txt


def build_sample(subs, n: int = 3000, seed: int = 0) -> list[str]:
    """Deduped context sample: all of them if <= n, else a sample of n.

    ``seed`` makes the subset reproducible; a different seed per run (see
    ``run_propose_local``) surfaces a different slice of a large list so multiple
    runs explore more of it. Lists at or below ``n`` are unaffected - all are sent.
    """
    uniq = sorted({s.strip().lower() for s in subs if s.strip()})
    if len(uniq) <= n:
        return uniq
    return sorted(random.Random(seed).sample(uniq, n))


def normalize_candidate(c: str, apex: str) -> str | None:
    """Lowercase and ensure the guess is a full hostname under the apex."""
    c = c.strip().lower().rstrip(".")
    if not c:
        return None
    if not c.endswith("." + apex) and c != apex:
        c = c + "." + apex
    return c


def _user_prompt(subs, apex, sample, count, covered_templates, seed: int = 0) -> str:
    """The user message; wording lives in prompts/predict_user.txt (+ predict_covered.txt)."""
    samp = build_sample(subs, sample, seed=seed)
    covered = ""
    if covered_templates:
        patterns = "\n".join(list(covered_templates)[:80])
        covered = "\n" + prompts.render("predict_covered", patterns=patterns) + "\n"
    return prompts.render("predict_user", apex=apex, count=len(samp), hosts="\n".join(samp),
                          covered=covered, limit=count)


def _finalize(candidates, subs, apex) -> list[str]:
    """Normalize, dedup, and drop known - shared by the API and local paths."""
    known = {s.strip().lower() for s in subs if s.strip()}
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        norm = normalize_candidate(c, apex)
        if norm and norm not in known and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# JSON schema for Ollama's structured-output `format` field.
_OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {"type": "string"}}},
    "required": ["candidates"],
}


def _parse_candidates(content: str) -> list[str]:
    """Extract the candidate strings from a model reply.

    Prefers strict JSON, but a local model can hit its output limit and return a
    truncated (unparseable) array. In that case, salvage every complete quoted
    string so a cut-off reply still yields the names that finished."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
            return [c for c in obj["candidates"] if isinstance(c, str)]
    except (json.JSONDecodeError, AttributeError):
        pass
    # Salvage: pull complete "..." strings, skipping the leading "candidates" key.
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"', content)
    return [s for s in strings if s != "candidates"]


def _ollama_once(user: str, model: str, url: str, num_ctx: int, seed: int | None) -> list[str]:
    """One Ollama /api/chat call -> raw candidate strings (unfiltered)."""
    options = {"num_predict": 8192, "temperature": 0.5, "num_ctx": num_ctx}
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "stream": False,
        "format": _OLLAMA_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "options": options,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            data = json.loads(r.read().decode())
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach Ollama at {url} ({e.reason if hasattr(e, 'reason') else e}). "
            f"Start it with `ollama serve` and pull the model: `ollama pull {model}`."
        )
    return _parse_candidates(data.get("message", {}).get("content", ""))


def run_propose_local(subs, apex: str | None = None, model: str = "qwen2.5",
                      count: int = 1000, sample: int = 3000, covered_templates=None,
                      url: str = "http://localhost:11434",
                      num_ctx: int | None = None, runs: int = 1,
                      seed: int | None = None, progress=None) -> list[str]:
    """Propose net-new subdomains via a local Ollama model - offline, free, private.

    Requires Ollama running (`ollama serve`) with the model pulled
    (`ollama pull <model>`). Data never leaves the machine.

    ``num_ctx`` sets the model's context window. Ollama defaults to only 2048
    tokens, which silently truncates a large sub list (dropping the system
    prompt/schema and yielding zero candidates). When not given, it is sized to
    fit the whole prompt plus the reply.

    ``runs`` calls the model that many times, each with a different ``seed`` -
    which drives BOTH the model's sampling AND the context subset: when the sub
    list is larger than ``sample``, each run sees a different slice of it, so
    multiple runs explore more of a big list (and the model varies too). Lists at
    or below ``sample`` send all subs every run (only the model output varies).
    Set ``seed`` for a single reproducible run. ``progress(i, n, running_total)``
    is an optional callback invoked after each run.
    """
    apex = apex if apex is not None else detect_apex(subs)
    n_runs = max(1, runs)
    merged: list[str] = []
    seen: set[str] = set()
    for i in range(n_runs):
        # One seed per run drives both the sample slice and the model sampling;
        # honor an explicit single seed when the caller wants reproducibility.
        run_seed = seed if (n_runs == 1 and seed is not None) else i
        user = _user_prompt(subs, apex, sample, count, covered_templates, seed=run_seed)
        nctx = num_ctx
        if nctx is None:
            # ~4 chars/token heuristic + headroom, then round up to a sane floor.
            nctx = max(8192, (len(SYSTEM_PROMPT) + len(user)) // 4 + 8192 + 1024)
        for c in _ollama_once(user, model, url, nctx, run_seed):
            if c not in seen:
                seen.add(c)
                merged.append(c)
        if progress is not None:
            progress(i + 1, n_runs, len(_finalize(merged, subs, apex)))
    return _finalize(merged, subs, apex)


def run_propose_openai(subs, apex: str | None = None, model: str = "gpt-4o-mini",
                       count: int = 1000, sample: int = 3000, covered_templates=None,
                       base_url: str | None = None, api_key: str | None = None,
                       runs: int = 1, seed: int | None = None, progress=None) -> list[str]:
    """Propose net-new subdomains via an OpenAI-compatible model (e.g. gpt-4o-mini).

    Same per-run seeding/merging as ``run_propose_local`` (each run varies the
    sample slice and the model sampling). Reads $OPENAI_API_KEY unless ``api_key``
    is given; ``base_url`` targets any compatible endpoint.
    """
    from .openai_compat import DEFAULT_BASE_URL, chat_json

    apex = apex if apex is not None else detect_apex(subs)
    base_url = base_url or DEFAULT_BASE_URL
    n_runs = max(1, runs)
    merged: list[str] = []
    seen: set[str] = set()
    for i in range(n_runs):
        run_seed = seed if (n_runs == 1 and seed is not None) else i
        user = _user_prompt(subs, apex, sample, count, covered_templates, seed=run_seed)
        content = chat_json(model, SYSTEM_PROMPT, user, base_url=base_url,
                            api_key=api_key, max_tokens=16000, temperature=0.5, seed=run_seed)
        for c in _parse_candidates(content):
            if c not in seen:
                seen.add(c)
                merged.append(c)
        if progress is not None:
            progress(i + 1, n_runs, len(_finalize(merged, subs, apex)))
    return _finalize(merged, subs, apex)


__all__ = ["run_propose_local", "run_propose_openai",
           "build_sample", "normalize_candidate"]

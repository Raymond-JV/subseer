"""Minimal OpenAI-compatible chat client (stdlib only).

Talks to any endpoint that implements POST /chat/completions with JSON-object
response formatting - OpenAI (gpt-4o-mini, ...), Groq, Together, a local vLLM,
Ollama's /v1, etc. Used as a middle-tier AI backend: cheaper and stronger than a
small local model, far cheaper than a frontier API. Dependency-free (urllib), so
the tool keeps working with no SDK installed.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Retry knobs (overridable via env for aggressive concurrency tuning).
_MAX_RETRIES = int(os.environ.get("SUBPATTERN_OPENAI_RETRIES", "8"))
_BACKOFF_BASE = float(os.environ.get("SUBPATTERN_OPENAI_BACKOFF", "0.5"))  # seconds
_BACKOFF_CAP = 30.0
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_after_seconds(err: urllib.error.HTTPError, body: str) -> float | None:
    """How long the server told us to wait, if anything: the Retry-After header
    (seconds) or the "try again in 139ms"/"in 1.5s" hint OpenAI puts in the body."""
    hdr = err.headers.get("Retry-After") if err.headers else None
    if hdr:
        try:
            return float(hdr)
        except ValueError:
            pass
    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)\b", body, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return val / 1000.0 if m.group(2).lower() == "ms" else val
    return None


def _ensure_json_hint(system: str, user: str) -> str:
    """OpenAI's json_object mode 400s unless the messages contain the word 'json'.
    Append a short instruction to the user message when neither prompt has it."""
    if "json" in (system + " " + user).lower():
        return user
    return user + "\n\nRespond with a single JSON object."


def chat_json(model: str, system: str, user: str, *,
              base_url: str = DEFAULT_BASE_URL, api_key: str | None = None,
              max_tokens: int = 16000, temperature: float = 0.5,
              seed: int | None = None, timeout: int = 600) -> str:
    """One chat completion in JSON mode; returns the message content string.

    ``api_key`` falls back to $OPENAI_API_KEY. Raises SystemExit with a clear
    message on a missing key or an API/connection error (callers treat that as
    "backend unavailable" and keep any free results).
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key: set OPENAI_API_KEY to use --gpt "
            "(get one at https://platform.openai.com/api-keys)."
        )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": _ensure_json_hint(system, user)},
        ],
        "response_format": {"type": "json_object"},  # portable across compatible servers
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    # Retry transient failures (429 rate-limit, 5xx) with backoff. Honors the server's
    # Retry-After/"try again in Xms" hint when present, else exponential backoff + jitter.
    # This lets many concurrent callers self-throttle to the TPM ceiling instead of each
    # 429 permanently dropping that host's work.
    for attempt in range(_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:300] if hasattr(e, "read") else ""
            if e.code in _RETRY_STATUS and attempt < _MAX_RETRIES:
                # Floor the server's hint with escalating backoff: a "try again in 183ms" hint
                # is honored once, but if the TPM bucket stays saturated the wait grows
                # (0.5,1,2,4,8,16,30…s) so we actually give it time to drain instead of
                # burning all retries in one second. Jitter de-syncs concurrent callers.
                hint = _retry_after_seconds(e, detail) or 0.0
                backoff = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
                time.sleep(max(hint, backoff) + random.uniform(0, _BACKOFF_BASE))
                continue
            raise SystemExit(f"OpenAI API error {e.code} at {base_url}: {detail}")
        except urllib.error.URLError as e:
            reason = e.reason if hasattr(e, "reason") else e
            if attempt < _MAX_RETRIES:                        # transient network hiccup
                time.sleep(min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
                           + random.uniform(0, _BACKOFF_BASE))
                continue
            raise SystemExit(f"Could not reach {base_url} ({reason}).")
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


__all__ = ["chat_json", "DEFAULT_BASE_URL"]

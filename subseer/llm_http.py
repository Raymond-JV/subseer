"""HTTP for LLM calls: one JSON POST with retries, shared by the OpenAI and Ollama paths.

Retried, with exponential backoff (0.5s, 1s, 2s, ... up to 30s, plus jitter) that
honors the server's own wait hint: rate limits (429), server errors (5xx), timeouts,
dropped connections and garbled responses. Not retried, because waiting can't fix
them: a bad key or request (4xx), an account out of credit (429 insufficient_quota),
and nothing listening at the URL. Every retry and final failure is reported through
subseer.events; a final failure raises SystemExit with a readable message.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from . import events


def _env(name: str, old: str, default: str) -> str:
    """$SUBSEER_* setting, still honoring the pre-rename $SUBPATTERN_* name."""
    return os.environ.get(name) or os.environ.get(old) or default


MAX_RETRIES = int(_env("SUBSEER_OPENAI_RETRIES", "SUBPATTERN_OPENAI_RETRIES", "8"))
BACKOFF_BASE = float(_env("SUBSEER_OPENAI_BACKOFF", "SUBPATTERN_OPENAI_BACKOFF", "0.5"))
BACKOFF_CAP = 30.0
RETRY_STATUS = {429, 500, 502, 503, 504}
_DROPPED = (TimeoutError, ConnectionError, http.client.HTTPException, json.JSONDecodeError)


def retry_after_seconds(err: urllib.error.HTTPError, body: str) -> float | None:
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


def _backoff(attempt: int) -> float:
    return min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP) + random.uniform(0, BACKOFF_BASE)


def _retry(provider: str, reason: str, attempt: int, retries: int, wait: float) -> None:
    events.emit("retry", provider=provider, reason=reason, attempt=attempt + 1, of=retries,
                wait=round(wait, 1))
    time.sleep(wait)


def _fail(provider: str, message: str, attempt: int):
    events.emit("error", provider=provider, error=message, attempts=attempt + 1)
    raise SystemExit(message)


def post_json(url: str, body: dict, *, provider: str, headers: dict | None = None,
              timeout: int = 600, retries: int = MAX_RETRIES,
              unreachable_hint: str = "") -> dict:
    """POST ``body`` as JSON and return the parsed JSON reply, retrying as above."""
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="ignore") if hasattr(e, "read") else ""
            detail = body_text[:300]
            if e.code == 429 and "insufficient_quota" in body_text:
                _fail(provider, f"{provider} account is out of credit (insufficient_quota); "
                                f"add billing or use --ollama.", attempt)
            if e.code in RETRY_STATUS and attempt < retries:
                wait = max(retry_after_seconds(e, body_text) or 0.0, _backoff(attempt))
                why = "rate limited" if e.code == 429 else f"server error {e.code}"
                _retry(provider, why, attempt, retries, wait)
                continue
            _fail(provider, f"{provider} API error {e.code} at {url}: {detail}", attempt)
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, ConnectionRefusedError) or attempt >= retries:
                _fail(provider, f"Could not reach {provider} at {url} ({reason}).{unreachable_hint}",
                      attempt)
            _retry(provider, f"network error ({reason})", attempt, retries, _backoff(attempt))
        except _DROPPED as e:
            what = "timed out" if isinstance(e, TimeoutError) else f"{type(e).__name__}"
            if attempt >= retries:
                _fail(provider, f"{provider} call failed after {attempt + 1} tries ({what}).",
                      attempt)
            _retry(provider, what, attempt, retries, _backoff(attempt))
    raise AssertionError("unreachable")  # every path above returns or raises


def ollama_chat(url: str, payload: dict, model: str, timeout: int = 1200) -> dict:
    """One Ollama /api/chat call (a couple of retries: it's local, so failures are rarer)."""
    return post_json(url.rstrip("/") + "/api/chat", payload, provider="Ollama",
                     timeout=timeout, retries=2,
                     unreachable_hint=(f" Start it with `ollama serve` and pull the model: "
                                       f"`ollama pull {model}`."))

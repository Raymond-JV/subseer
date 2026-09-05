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
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.openai.com/v1"


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
            "No API key: set OPENAI_API_KEY to use --openai "
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300] if hasattr(e, "read") else ""
        raise SystemExit(f"OpenAI API error {e.code} at {base_url}: {detail}")
    except urllib.error.URLError as e:
        reason = e.reason if hasattr(e, "reason") else e
        raise SystemExit(f"Could not reach {base_url} ({reason}).")
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


__all__ = ["chat_json", "DEFAULT_BASE_URL"]

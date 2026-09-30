"""Tests for the OpenAI-compatible backend (propose + enrich), with the HTTP
call stubbed so no network/key is needed. Stdlib only."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subseer.openai_compat as oc
from subseer.enrich import run_enrich_openai
from subseer.propose import run_propose_openai


def _stub_chat(responses):
    """Return a chat_json stub that yields queued responses per call."""
    calls = {"n": 0}

    def stub(model, system, user, **kw):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[i]

    stub.calls = calls
    return stub


def test_propose_openai_parses_and_finalizes(monkeypatch=None):
    subs = ["api.example.com"]
    payload = json.dumps({"candidates": ["grafana", "api", "vault.example.com", "grafana"]})
    oc.chat_json = _stub_chat([payload])  # monkeypatch the module-level fn
    # propose imports chat_json lazily from the module, so patching the module works
    out = run_propose_openai(subs, apex="example.com", model="gpt-4o-mini")
    assert out == ["grafana.example.com", "vault.example.com"]  # normalized, known 'api' dropped, deduped


def test_propose_openai_multi_run_merges():
    subs = ["api.example.com"]
    r1 = json.dumps({"candidates": ["grafana"]})
    r2 = json.dumps({"candidates": ["vault", "grafana"]})
    oc.chat_json = _stub_chat([r1, r2])
    out = run_propose_openai(subs, apex="example.com", runs=2)
    assert set(out) == {"grafana.example.com", "vault.example.com"}


def test_enrich_openai_builds_themes():
    from subseer.models import Slot, Theme

    mined = [Theme(name="t", description="d", evidence=[], kind="template",
                   template="{s1}.dleague.example.com",
                   slots=[Slot(name="s1", kind="enum", values=["memphis"])])]
    resp = json.dumps({"themes": [
        {"template": "{s1}.dleague.example.com",
         "slots": [{"name": "s1", "values": ["memphis", "boston", "miami"]}]}
    ]})
    oc.chat_json = _stub_chat([resp])
    enriched = run_enrich_openai(mined, ["memphis.dleague.example.com"], apex="example.com")
    assert len(enriched) == 1
    assert set(enriched[0].slots[0].values) == {"memphis", "boston", "miami"}


def test_ensure_json_hint():
    from subseer.openai_compat import _ensure_json_hint
    # appended when neither prompt mentions json
    assert _ensure_json_hint("expand slots", "here are hosts").endswith("JSON object.")
    # left alone when json already present (case-insensitive)
    assert _ensure_json_hint("return JSON", "hosts") == "hosts"
    assert _ensure_json_hint("sys", "reply as json") == "reply as json"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

"""Retries and failure reporting for LLM HTTP calls, against a scripted local server.
Stdlib only."""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("SUBSEER_LOG_DIR", tempfile.mkdtemp(prefix="subseer-test-logs-"))

from subseer import events, llm_http  # noqa: E402

llm_http.BACKOFF_BASE = 0.01  # keep the tests fast

OK = {"choices": [{"message": {"content": '{"themes": []}'}}]}


@contextlib.contextmanager
def server(script):
    """Serve each request with the next (status, body, delay_s) in ``script``."""
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, body, delay = script[min(len(calls), len(script) - 1)]
            calls.append(status)
            time.sleep(delay)
            data = json.dumps(body).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass  # the client gave up (timeout test)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    finally:
        srv.shutdown()


@contextlib.contextmanager
def recorded():
    seen = []
    handler = lambda kind, fields: seen.append((kind, fields))  # noqa: E731
    events.subscribe(handler)
    try:
        yield seen
    finally:
        events.unsubscribe(handler)


def _post(url, **kw):
    return llm_http.post_json(url + "/chat", {"x": 1}, provider="OpenAI", **kw)


def test_rate_limits_are_retried_then_succeed():
    rate_limited = (429, {"error": {"message": "Rate limit. Please try again in 1ms."}}, 0)
    with server([rate_limited, rate_limited, (200, OK, 0)]) as (url, calls), recorded() as seen:
        assert _post(url) == OK
    assert calls == [429, 429, 200]
    assert [k for k, _ in seen] == ["retry", "retry"]
    assert seen[0][1]["reason"] == "rate limited" and seen[0][1]["attempt"] == 1


def test_out_of_credit_fails_fast_without_retrying():
    quota = (429, {"error": {"code": "insufficient_quota", "message": "You exceeded your quota"}}, 0)
    with server([quota]) as (url, calls), recorded() as seen:
        try:
            _post(url)
        except SystemExit as e:
            assert "out of credit" in str(e)
        else:
            raise AssertionError("expected SystemExit")
    assert calls == [429]
    assert [k for k, _ in seen] == ["error"]


def test_bad_requests_fail_fast():
    with server([(400, {"error": {"message": "bad"}}, 0)]) as (url, calls), recorded() as seen:
        try:
            _post(url)
        except SystemExit as e:
            assert "API error 400" in str(e)
    assert calls == [400] and seen[-1][0] == "error"


def test_server_errors_are_retried_until_the_limit():
    with server([(503, {}, 0)]) as (url, calls), recorded() as seen:
        try:
            _post(url, retries=2)
        except SystemExit as e:
            assert "API error 503" in str(e)
    assert calls == [503, 503, 503]  # first try + 2 retries
    assert [k for k, _ in seen] == ["retry", "retry", "error"]


def test_timeouts_are_retried():
    with server([(200, OK, 1.5), (200, OK, 0)]) as (url, calls), recorded() as seen:
        assert _post(url, timeout=0.5) == OK
    assert seen[0][0] == "retry" and seen[0][1]["reason"] == "timed out"


def test_nothing_listening_fails_fast_with_the_hint():
    with socket.socket() as s:  # grab a free port, then close it so nothing listens
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with recorded() as seen:
        try:
            llm_http.ollama_chat(f"http://127.0.0.1:{port}", {"x": 1}, "qwen2.5")
        except SystemExit as e:
            assert "Could not reach Ollama" in str(e) and "ollama serve" in str(e)
        else:
            raise AssertionError("expected SystemExit")
    assert [k for k, _ in seen] == ["error"] and seen[0][1]["attempts"] == 1


def test_a_gpt_run_shows_retries_and_logs_them():
    from subseer.cli import main

    rate_limited = (429, {"error": {"message": "Please try again in 1ms."}}, 0)
    log_dir = os.environ["SUBSEER_LOG_DIR"]
    log_path = os.path.join(log_dir, "logs.jsonl")
    before = open(log_path, encoding="utf-8").read().splitlines() if os.path.exists(log_path) else []
    with tempfile.TemporaryDirectory() as d, server([rate_limited, (200, OK, 0)]) as (url, _):
        subs = os.path.join(d, "subs.txt")
        with open(subs, "w", encoding="utf-8") as f:
            f.write("api.example.com\nshop.example.com\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            main([subs, "--mine", "--gpt", "--api-base", url + "/v1"])
    assert "OpenAI rate limited, retrying in" in err.getvalue()
    lines = open(log_path, encoding="utf-8").read().splitlines()
    rec = json.loads(lines[len(before)])
    assert rec["llm_events"][0]["event"] == "retry"
    assert rec["llm_events"][0]["provider"] == "OpenAI"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

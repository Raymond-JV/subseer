"""Tests for the CLI: auto-scaling helpers, banner behavior, -q / --version. Stdlib only."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

# Keep test runs out of the real ~/.subseer/runs.
_LOG_DIR = tempfile.mkdtemp(prefix="subseer-test-logs-")
os.environ["SUBSEER_LOG_DIR"] = _LOG_DIR

from subseer import __version__, banner
from subseer.cli import (
    _api_base,
    _auto_min_values,
    _auto_runs,
    _auto_sample,
    _resolve_auto,
    main,
    parse_args,
)


def test_auto_sample_anchored_near_2k():
    assert _auto_sample(500) == 2000
    assert _auto_sample(20000) == 2000
    assert _auto_sample(80000) == 3000
    assert _auto_sample(773890) == 4000


def test_auto_runs_scales_with_size():
    assert _auto_runs(300) == 2
    assert _auto_runs(2000) == 2
    assert _auto_runs(4000) == 5
    assert _auto_runs(9000) == 8
    assert _auto_runs(50000) == 10


def test_auto_min_values_scales_with_size():
    assert _auto_min_values(300) == 2
    assert _auto_min_values(3000) == 3
    assert _auto_min_values(12000) == 4
    assert _auto_min_values(50000) == 5


def test_resolve_auto_int_passthrough():
    assert _resolve_auto("7", 9999, _auto_runs, "local-runs") == 7
    assert _resolve_auto(3, 9999, _auto_min_values, "min-values") == 3


def test_resolve_auto_keyword():
    assert _resolve_auto("auto", 4000, _auto_runs, "local-runs") == 5
    assert _resolve_auto("AUTO", 12000, _auto_min_values, "min-values") == 4


def test_api_base_defaults_per_backend_and_overrides():
    args = parse_args(["-d", "x.example.com"])
    assert _api_base("gpt", args) == "https://api.openai.com/v1"
    assert _api_base("ollama", args) == "http://localhost:11434"
    args = parse_args(["-d", "x.example.com", "--api-base", "http://box:11434"])
    assert _api_base("ollama", args) == "http://box:11434"


def test_resolve_auto_invalid_exits():
    try:
        _resolve_auto("banana", 100, _auto_runs, "local-runs")
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("expected SystemExit on invalid value")


class _FakeTTY(io.StringIO):
    def isatty(self):
        return True


def _capture(fn, *args):
    """Run fn with stdout/stderr captured; returns (result, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = fn(*args)
    return result, out.getvalue(), err.getvalue()


def test_bare_invocation_is_the_welcome_screen():
    code, out, _ = _capture(main, [])
    assert code == 0
    assert "usage: subseer" in out


def test_version_prints_version_on_stdout():
    code, out, _ = _capture(main, ["--version"])
    assert code == 0
    assert out.strip() == f"subseer {__version__}"


def test_quiet_run_writes_results_but_no_logs():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.txt")
        code, out, err = _capture(main, ["-q", "-d", "api.dev.example.com", "--out", path])
        assert code == 0
        assert out == "" and err == ""
        with open(path, encoding="utf-8") as f:
            assert f.read().strip()


def test_wordlist_env_override_is_used_and_logged():
    with tempfile.TemporaryDirectory() as d:
        wl = os.path.join(d, "dns.txt")
        with open(wl, "w", encoding="utf-8") as f:
            f.write("vpn\ngateway\ninternal\n")
        before = set(_logs())
        old = os.environ.get("SUBSEER_WORDLIST")
        os.environ["SUBSEER_WORDLIST"] = wl
        try:
            code, _, err = _capture(main, ["-d", "api.dev.example.com", "--fuzz"])
        finally:
            if old is None:
                os.environ.pop("SUBSEER_WORDLIST")
            else:
                os.environ["SUBSEER_WORDLIST"] = old
        assert code == 0 and wl in err
    (new,) = [p for p in _logs() if p not in before]
    with open(new, encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["wordlist"]["source"] == "env" and rec["wordlist"]["words"] == 3


def test_short_w_flag_sets_the_wordlist():
    with tempfile.TemporaryDirectory() as d:
        wl = os.path.join(d, "dns.txt")
        with open(wl, "w", encoding="utf-8") as f:
            f.write("vpn\ngateway\n")
        code, _, err = _capture(main, ["-d", "api.dev.example.com", "--fuzz", "-w", wl])
    assert code == 0 and wl in err


def test_default_uses_the_bundled_wordlist():
    before = set(_logs())
    old = os.environ.get("SUBSEER_WORDLIST")
    os.environ["SUBSEER_WORDLIST"] = os.path.join(tempfile.gettempdir(), "does-not-exist-xyz.txt")
    try:
        code, _, err = _capture(main, ["-d", "api.dev.example.com", "--fuzz"])
    finally:
        if old is None:
            os.environ.pop("SUBSEER_WORDLIST")
        else:
            os.environ["SUBSEER_WORDLIST"] = old
    assert code == 0 and "bundled" in err
    (new,) = [p for p in _logs() if p not in before]
    with open(new, encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["wordlist"]["source"] == "bundled" and rec["wordlist"]["words"] > 15000


def test_results_go_to_stdout_by_default_and_progress_to_stderr():
    code, out, err = _capture(main, ["-d", "api.dev.example.com"])
    assert code == 0
    lines = [l for l in out.splitlines() if l]
    assert lines and all(" " not in l and "." in l for l in lines)  # hostnames only
    assert "Loaded" in err and "Wrote" in err


def test_limit_caps_the_written_output():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.txt")
        code, _, _ = _capture(main, ["-q", "-d", "api.dev.example.com", "--limit", "5", "-o", path])
        assert code == 0
        with open(path, encoding="utf-8") as f:
            assert len(f.read().split()) == 5


def _logs():
    return sorted(os.path.join(_LOG_DIR, f) for f in os.listdir(_LOG_DIR))


def test_every_run_writes_a_run_log():
    before = set(_logs())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.txt")
        _capture(main, ["-q", "-d", "api.dev.example.com", "-o", path])
        with open(path, encoding="utf-8") as f:
            written = len(f.read().split())
    new = [p for p in _logs() if p not in before]
    assert len(new) == 1
    with open(new[0], encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["command"][0] == "subseer" and "-q" in rec["command"]
    assert rec["input"]["kind"] == "domain" and rec["input"]["count"] == 1
    assert rec["output"]["written"] == written
    assert rec["exit_code"] == 0 and rec["mode"] == "generate"
    assert "templates" in rec and "fuzz" in rec["counts"]


def test_input_errors_are_not_logged():
    before = set(_logs())
    _capture(main, ["-q"])            # no input -> error, nothing ran
    assert set(_logs()) == before


def test_run_log_records_learned_templates():
    with tempfile.TemporaryDirectory() as d:
        subs = os.path.join(d, "subs.txt")
        with open(subs, "w", encoding="utf-8") as f:
            f.write("api.dev.example.com\napi.prod.example.com\napi.qa.example.com\n"
                    "web.dev.example.com\nweb.prod.example.com\n")
        before = set(_logs())
        _capture(main, ["-q", subs, "--mine", "-o", os.path.join(d, "a.txt")])
    (new,) = [p for p in _logs() if p not in before]
    with open(new, encoding="utf-8") as f:
        rec = json.load(f)
    assert rec["input"]["kind"] == "file" and rec["input"]["count"] == 5
    assert rec["templates"] and "template" in rec["templates"][0]


def test_header_is_one_line_on_a_tty_and_silent_otherwise():
    tty, pipe = _FakeTTY(), io.StringIO()
    old = os.environ.get("NO_COLOR")
    os.environ["NO_COLOR"] = "1"
    try:
        banner.header("9.9.9", tty)
        banner.header("9.9.9", pipe)
    finally:
        if old is None:
            os.environ.pop("NO_COLOR")
        else:
            os.environ["NO_COLOR"] = old
    assert tty.getvalue() == "subseer v9.9.9 - sees the subs you don't\n"
    assert pipe.getvalue() == ""


def test_full_art_skipped_when_not_a_tty():
    pipe = io.StringIO()
    banner.show(pipe)
    assert pipe.getvalue() == ""


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

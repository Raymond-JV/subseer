"""A tiny hook for reporting LLM retries and failures from deep in the call stack.

The HTTP layer ``emit``s events; the CLI ``subscribe``s to show them on screen and
record them in the run log. With no subscriber, events go nowhere.
"""

from __future__ import annotations

_handlers: list = []


def subscribe(handler) -> None:
    """``handler(kind, fields)`` is called for every event."""
    _handlers.append(handler)


def unsubscribe(handler) -> None:
    if handler in _handlers:
        _handlers.remove(handler)


def emit(kind: str, **fields) -> None:
    for handler in list(_handlers):
        try:
            handler(kind, fields)
        except Exception:  # reporting must never break a call
            pass

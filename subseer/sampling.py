"""Shuffle-and-split: how a list too big for one prompt is spread across model calls.

The list is shuffled once per pass and cut into consecutive, near-equal slices of at
most ``size`` items, one slice per call. The first ``passes_needed(n, size)`` calls
together send every item exactly once; later calls start a new pass with a fresh
shuffle, so the model sees the same items in new groupings.
"""

from __future__ import annotations

import random


def calls_to_cover(n: int, size: int) -> int:
    """Calls needed to send all ``n`` items at most ``size`` per call (at least 1)."""
    return max(1, -(-n // max(1, size)))


def shuffled_slice(items, size: int, call: int, seed: int = 0) -> list[str]:
    """The items sent on call number ``call`` (0-based), deduped and sorted.

    A list that fits in one slice is sent whole on every call.
    """
    uniq = sorted(set(items))
    n = len(uniq)
    if n <= size:
        return uniq
    per_pass = calls_to_cover(n, size)
    pass_no, k = divmod(call, per_pass)
    order = uniq[:]
    random.Random(seed * 1_000_003 + pass_no).shuffle(order)  # a fresh shuffle per pass
    chunk = -(-n // per_pass)                                  # near-equal slices
    return sorted(order[k * chunk:(k + 1) * chunk])

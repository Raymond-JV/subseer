"""Offline structural profiler — exhaustive over the WHOLE file, no API.

Streams the input once and aggregates:
  - sub-zone inventory  (everything under a deeper zone, e.g. *.vpn.corp.example.com)
  - templated skeletons for direct *.example.com hosts (digits -> #, hex runs -> @)
  - most common leading tokens
  - a reservoir sample of word-only hostnames (codename candidates)

This gives complete coverage of the structural conventions without holding the
file in memory or sending it to a model. Use its output to author patterns.json.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter

PATH = sys.argv[1]
APEX = sys.argv[2] if len(sys.argv) > 2 else "example.com"
SUFFIX = "." + APEX

TOP_ZONES, TOP_SKELS, TOP_LEADS, SAMPLE_N = 70, 200, 90, 200

_dig = re.compile(r"^[0-9]+$")
_hex = re.compile(r"^[0-9a-f]{4,}$")
_digrun = re.compile(r"[0-9]+")


def norm_label(label: str) -> str:
    out = []
    for seg in label.split("-"):
        if _dig.match(seg):
            out.append("#")
        elif _hex.match(seg):
            out.append("@")
        else:
            out.append(_digrun.sub("#", seg))
    return "-".join(out)


zones: Counter[str] = Counter()
skels: Counter[str] = Counter()
leads: Counter[str] = Counter()
word_sample: list[str] = []
word_total = 0
total = 0
random.seed(0)

with open(PATH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        h = line.strip().lower()
        if not h.endswith(SUFFIX):
            continue
        total += 1
        rest = h[: -len(SUFFIX)]
        if "." in rest:
            zones[rest.split(".", 1)[1] + "." + APEX] += 1
        else:
            leads[rest.split("-")[0]] += 1
            sk = norm_label(rest)
            if "#" in sk or "@" in sk:
                skels[sk] += 1
            else:
                word_total += 1
                if len(word_sample) < SAMPLE_N:
                    word_sample.append(rest)
                else:
                    j = random.randint(0, word_total - 1)
                    if j < SAMPLE_N:
                        word_sample[j] = rest

print(f"total {APEX} hosts: {total:,}")
print(f"\n== top {TOP_ZONES} sub-zones (host count) ==")
for z, c in zones.most_common(TOP_ZONES):
    print(f"{c:>10,}  {z}")
print(f"\n== top {TOP_SKELS} templated skeletons, direct *.{APEX} ==")
for s, c in skels.most_common(TOP_SKELS):
    print(f"{c:>10,}  {s}")
print(f"\n== top {TOP_LEADS} leading tokens (direct hosts) ==")
for t, c in leads.most_common(TOP_LEADS):
    print(f"{c:>10,}  {t}")
print(f"\n== word-only direct hosts: {word_total:,} total; random sample ==")
print(", ".join(sorted(word_sample)))

"""Offline pattern miner - detect templates from a subdomain list with no API.

Pipeline:
  1. tokenize each host (split on '.', '-', and letter/digit boundaries).
  2. one-hole pass: blank each position in turn and collect the value-set seen
     in that hole for that exact surrounding context.
  3. two-hole merge: when many one-hole contexts share the SAME hole value-set
     (e.g. {int,lab,qa,stage} recurs after purchase, audience, lp, ...), the
     differing position is itself a dimension -> emit a multi-slot template
     ({service}.{env}). The "same value-set" rule is what keeps junk out.
  4. emit: numeric holes -> range slots (min/max/pad); word holes -> enum slots.

Core functions return plain dicts (Theme-compatible) so they're testable with no
pydantic/anthropic import. ``mine_themes`` wraps them into Theme objects.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

HOLE_I = "\x00"  # marks a slot position inside a masked token tuple
MARK = "\x02"    # transient marker for the position being generalized in a merge pass

_TOK = re.compile(r"[a-z]+|[0-9]+|[^a-z0-9]+")
_SPLIT = re.compile(r"([.-])")


def tokenize(label: str) -> tuple[list[str], list[str]]:
    """Split a label into tokens + the separator before each (''=glued boundary).

    'purchase.lab'    -> (['purchase','lab'],        ['', '.'])
    'mx10'            -> (['mx','10'],               ['', ''])
    'austin.gleague-dev' -> (['austin','gleague','dev'], ['', '.', '-'])
    """
    tokens: list[str] = []
    seps: list[str] = []
    pieces = _SPLIT.split(label.lower())
    for k, part in enumerate(pieces):
        if k % 2 == 1:  # separator piece
            continue
        sep = "" if k == 0 else pieces[k - 1]
        for m, sub in enumerate(_TOK.findall(part)):
            tokens.append(sub)
            seps.append(sep if m == 0 else "")
    return tokens, seps


def detok(tokens: list[str], seps: list[str]) -> str:
    return "".join(seps[i] + t for i, t in enumerate(tokens))


# Common multi-part public suffixes (ccSLDs). The naive "last two labels" rule
# breaks on these - foo.example.com.mx would yield "com.mx" instead of "example.com.mx".
# Not the full PSL, but covers the country second-levels real targets use. The
# matcher handles any suffix length, so 3-part entries work too.
_MULTI_SUFFIXES: frozenset[str] = frozenset("""
co.uk org.uk me.uk ltd.uk plc.uk net.uk sch.uk ac.uk gov.uk mod.uk nhs.uk police.uk
com.au net.au org.au edu.au gov.au asn.au id.au
co.nz net.nz org.nz govt.nz ac.nz geek.nz gen.nz school.nz
co.za org.za net.za gov.za ac.za web.za
co.jp or.jp ne.jp ac.jp go.jp gr.jp ad.jp ed.jp lg.jp
co.kr or.kr ne.kr re.kr pe.kr go.kr ac.kr
com.cn net.cn org.cn gov.cn edu.cn ac.cn
co.in net.in org.in gen.in firm.in ind.in gov.in ac.in edu.in res.in
com.br net.br org.br gov.br edu.br
com.mx net.mx org.mx gob.mx edu.mx
com.ar net.ar org.ar gob.ar edu.ar
com.tr net.tr org.tr gov.tr edu.tr bel.tr k12.tr
com.tw net.tw org.tw gov.tw edu.tw idv.tw
com.hk net.hk org.hk gov.hk edu.hk idv.hk
com.sg net.sg org.sg gov.sg edu.sg per.sg
com.ru net.ru org.ru msk.ru spb.ru
com.ua net.ua org.ua in.ua kiev.ua
com.pl net.pl org.pl gov.pl edu.pl
com.es org.es gob.es edu.es
co.il org.il net.il ac.il gov.il muni.il idf.il
co.id or.id web.id go.id ac.id sch.id net.id my.id biz.id
com.my net.my org.my gov.my edu.my
com.ph net.ph org.ph gov.ph edu.ph
co.th net.th or.th go.th ac.th in.th
com.vn net.vn org.vn gov.vn edu.vn
com.sa net.sa org.sa gov.sa edu.sa med.sa sch.sa
co.ae net.ae org.ae gov.ae ac.ae sch.ae
com.eg net.eg org.eg gov.eg edu.eg
com.ng net.ng org.ng gov.ng edu.ng
co.ke or.ke ne.ke go.ke ac.ke sc.ke
com.pk net.pk org.pk gov.pk edu.pk
com.gr net.gr org.gr edu.gr gov.gr
com.pt edu.pt gov.pt org.pt net.pt
com.co net.co org.co gov.co edu.co
com.pe net.pe org.pe gob.pe edu.pe
com.uy net.uy org.uy gub.uy edu.uy
""".split())


def registrable_domain(host: str) -> str:
    """The registrable domain (eTLD+1), aware of multi-part suffixes.

    'foo.example.com'      -> 'example.com'
    'foo.example.com.mx'   -> 'example.com.mx'   (not 'com.mx')
    'a.b.example.co.uk'-> 'example.co.uk'
    """
    labels = [x for x in host.strip().lower().strip(".").split(".") if x]
    if len(labels) < 2:
        return ".".join(labels)
    # Longest public suffix wins -> scan from the left; keep one label before it.
    for start in range(len(labels) - 1):
        if ".".join(labels[start:]) in _MULTI_SUFFIXES and start >= 1:
            return ".".join(labels[start - 1:])
    return ".".join(labels[-2:])


def detect_apex(subs: list[str]) -> str:
    """Most common registrable domain across the input (PSL/ccSLD aware)."""
    c: Counter[str] = Counter()
    for s in subs:
        rd = registrable_domain(s)
        if rd and "." in rd:
            c[rd] += 1
    return c.most_common(1)[0][0] if c else ""


def _slot_from_values(name: str, values: Counter) -> dict:
    keys = list(values.keys())
    if keys and all(v.isdigit() for v in keys):
        nums = [int(v) for v in keys]
        pads = {len(v) for v in keys if len(v) > 1 and v[0] == "0"}
        return {
            "name": name,
            "kind": "range",
            "min": min(nums),
            "max": max(nums),
            "pad": max(pads) if pads else 0,
        }
    ordered = [v for v, _ in values.most_common()]
    return {"name": name, "kind": "enum", "values": ordered}


def _emit_theme(seps, masked, slot_pos: dict[int, Counter], apex, examples, support) -> dict:
    """Build a Theme-compatible dict from a masked token sequence + slot domains."""
    names: dict[int, str] = {}
    parts: list[str] = []
    slots: list[dict] = []
    n = 1
    for k in range(len(masked)):
        if k > 0:
            parts.append(seps[k])
        if k in slot_pos:
            nm = f"s{n}"
            n += 1
            names[k] = nm
            parts.append("{" + nm + "}")
            slots.append(_slot_from_values(nm, slot_pos[k]))
        else:
            parts.append(masked[k])
    stripped = "".join(parts)
    template = f"{stripped}.{apex}" if apex else stripped
    return {
        "name": template,
        "description": f"Auto-detected from {support} hosts.",
        "evidence": [f"{e}.{apex}" if apex else e for e in examples[:4]],
        "kind": "template",
        "template": template,
        "slots": slots,
    }


def detect_patterns(
    subs: list[str],
    apex: str | None = None,
    min_values: int = 3,
    min_support: int = 3,
    max_enum: int = 400,
    merge_overlap: float = 0.3,
) -> list[dict]:
    """Detect templates from a subdomain list. Returns Theme-compatible dicts."""
    apex = apex if apex is not None else detect_apex(subs)
    suffix = "." + apex if apex else ""

    tokenized = []
    for s in subs:
        s = s.strip().lower()
        if not s or (suffix and not s.endswith(suffix)):
            continue
        label = s[: -len(suffix)] if suffix else s
        if not label:
            continue
        toks, seps = tokenize(label)
        if toks:
            tokenized.append((toks, seps, label))

    # --- seed one-hole templates: blank each position, collect its value-set ---
    holes: dict[tuple, list] = {}
    for toks, seps, label in tokenized:
        for i in range(len(toks)):
            masked = tuple(HOLE_I if j == i else t for j, t in enumerate(toks))
            key = (tuple(seps), masked, i)
            rec = holes.get(key)
            if rec is None:
                holes[key] = [Counter({toks[i]: 1}), label]
            else:
                rec[0][toks[i]] += 1

    templates = [
        {"seps": seps, "masked": masked, "slots": {i: vals}, "examples": [ex]}
        for (seps, masked, i), (vals, ex) in holes.items()
        if len(vals) >= min_values and sum(vals.values()) >= min_support
    ]

    # --- iterative overlap merge: cross-fill dimensions ---
    # Two templates that differ in exactly one literal position, whose slot
    # domains overlap (Jaccard >= merge_overlap), fold that position into a new
    # slot and UNION the domains -> {team}.gleague + {team}.dleague => {team}.{league}.
    # Repeating to a fixpoint stacks a third dimension ({team}.{league}-{env}).
    for _ in range(8):
        templates, changed = _merge_pass(templates, merge_overlap)
        if not changed:
            break

    # --- emit, deduped by template string ---
    themes: list[dict] = []
    seen: set[str] = set()
    for t in templates:
        masked, slots = t["masked"], t["slots"]
        capped = {
            p: (dom if len(dom) <= max_enum else Counter(dict(dom.most_common(max_enum))))
            for p, dom in slots.items()
        }
        if len(slots) == 1:  # skip a lone non-numeric slot with no literal context
            p = next(iter(slots))
            numeric = all(v.isdigit() for v in capped[p])
            has_literal = any(k != p for k in range(len(masked)))
            if not (numeric or has_literal):
                continue
        support = max((sum(d.values()) for d in capped.values()), default=0)
        theme = _emit_theme(t["seps"], masked, capped, apex, t.get("examples", []), support)
        if theme["template"] in seen:
            continue
        seen.add(theme["template"])
        themes.append(theme)

    themes.sort(key=lambda t: sum(_slot_card(s) for s in t["slots"]), reverse=True)
    return themes


def _slot_card(slot: dict) -> int:
    if slot["kind"] == "enum":
        return len(slot["values"])
    return max(slot["max"] - slot["min"] + 1, 0)


def _jaccard(a: Counter, b: Counter) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _merge_pass(templates: list[dict], threshold: float):
    """One merge pass: fold a shared literal position into a new slot.

    Groups templates that are identical except at one literal position ``j`` and
    have the same slot positions; within each group, clusters those whose slot
    domains all overlap (Jaccard >= threshold) and merges each cluster of >= 2
    into a template with ``j`` promoted to a slot (domains unioned). Returns
    ``(new_templates, changed)``.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, t in enumerate(templates):
        slots = t["slots"]
        for j in range(len(t["masked"])):
            if j in slots:
                continue
            blanked = tuple(MARK if k == j else t["masked"][k] for k in range(len(t["masked"])))
            groups[(t["seps"], tuple(sorted(slots)), j, blanked)].append(idx)

    used = [False] * len(templates)
    result: list[dict] = []
    changed = False
    # Larger groups first so a template merges along its widest shared dimension.
    for (seps, stup, j, _blanked), idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        idxs = [i for i in idxs if not used[i]]
        if len(idxs) < 2:
            continue
        shared = list(stup)
        parent = {i: i for i in idxs}

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if all(
                    _jaccard(templates[ia]["slots"][p], templates[ib]["slots"][p]) >= threshold
                    for p in shared
                ):
                    parent[find(ia)] = find(ib)

        clusters: dict[int, list[int]] = defaultdict(list)
        for i in idxs:
            clusters[find(i)].append(i)

        for members in clusters.values():
            if len(members) < 2:
                continue
            new_slots = {p: Counter() for p in shared}
            new_slots[j] = Counter()
            examples: list[str] = []
            weight_pos = shared[0] if shared else None
            for mi in members:
                t = templates[mi]
                for p in shared:
                    new_slots[p].update(t["slots"][p])
                w = sum(t["slots"][weight_pos].values()) if weight_pos is not None else 1
                new_slots[j][t["masked"][j]] += w
                examples.extend(t.get("examples", []))
                used[mi] = True
            base = templates[members[0]]["masked"]
            new_masked = tuple(HOLE_I if k in new_slots else base[k] for k in range(len(base)))
            result.append(
                {"seps": seps, "masked": new_masked, "slots": new_slots, "examples": examples[:4]}
            )
            changed = True

    for i, t in enumerate(templates):
        if not used[i]:
            result.append(t)
    return result, changed


def _field(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _theme_regex(theme):
    """Compile a template (dict or Theme) into an anchored regex.

    Literals are escaped; enum slots become value alternations (longest first);
    range slots become ``[0-9]+``. A host is "covered" by the template iff it
    fully matches this regex.
    """
    template = _field(theme, "template")
    slots = {_field(s, "name"): s for s in _field(theme, "slots")}
    out = []
    for part in re.split(r"(\{[a-zA-Z0-9_]+\})", template):
        m = re.fullmatch(r"\{([a-zA-Z0-9_]+)\}", part)
        if m and m.group(1) in slots:
            s = slots[m.group(1)]
            if _field(s, "kind") == "enum":
                vals = sorted((re.escape(v) for v in _field(s, "values")), key=len, reverse=True)
                out.append("(?:" + "|".join(vals) + ")")
            else:
                out.append("[0-9]+")
        else:
            out.append(re.escape(part))
    return re.compile("^" + "".join(out) + "$")


def covered_hosts(subs, themes) -> set[str]:
    """Hosts that match at least one detected template."""
    regexes = [_theme_regex(t) for t in themes]
    cov: set[str] = set()
    for s in subs:
        s = s.strip().lower()
        for r in regexes:
            if r.match(s):
                cov.add(s)
                break
    return cov


def residual_hosts(subs, themes) -> list[str]:
    """Hosts that match NO template - the one-offs for the LLM to handle."""
    cov = covered_hosts(subs, themes)
    out = []
    seen = set()
    for s in subs:
        s = s.strip().lower()
        if s and s not in cov and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# Readable names for the fuzz vocabulary keys that are shorthand.
_LABEL_NAMES = {"infra": "service", "cc": "country", "dc": "datacenter", "dir": "direction"}


def guess_slot_label(slot: dict) -> str:
    """Best-effort name for a mined slot: 'number' for ranges, else the known
    vocabulary (env, region, version, ...) most of its values belong to, or ''."""
    if slot["kind"] == "range":
        return "number"
    from .fuzz import VALUE_VOCAB  # lazy: fuzz imports this module

    cats = Counter(VALUE_VOCAB[v] for v in slot["values"] if v in VALUE_VOCAB)
    if not cats:
        return ""
    cat, hits = cats.most_common(1)[0]
    if hits < 2 or hits * 2 < len(slot["values"]):
        return ""
    return _LABEL_NAMES.get(cat, cat)


def mine_themes(subs, apex=None, min_values=3, min_support=3):
    """Detect patterns and return Theme objects (wraps detect_patterns)."""
    from .models import Slot, Theme

    out = []
    for d in detect_patterns(subs, apex=apex, min_values=min_values, min_support=min_support):
        d = dict(d)
        d["slots"] = [Slot(**s, label=guess_slot_label(s)) for s in d["slots"]]
        out.append(Theme(**d))
    return out

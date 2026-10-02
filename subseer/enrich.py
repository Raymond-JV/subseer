"""LLM enriches MINED templates - the highest-leverage AI use in the tool.

Mining discovers structure from your data but is blind twice over:
  1. it fills a slot only with values it OBSERVED (>=3 times), so
     ``{s1}.dleague`` = {memphis, austin} misses the other 28 teams; and
  2. it never creates a slot it lacks >=3 examples for, so a lone ``api-dev``
     yields no ``{env}`` dimension at all.

This pass sends the mined templates (with their current slot values) plus the
RESIDUAL hosts (one-offs mining could not pattern) to the model and asks it to:
  A. EXPAND existing slots with real-world members of the same category, and
  B. DISCOVER new templates/dimensions latent in one example or implied by the
     stack it infers.

The model returns compact value lists; code does the unbounded expansion. Same
"LLM = knowledge, code = volume" split as the rest of the tool, applied to slots.
Works against the API or a local Ollama model (offline).
"""

from __future__ import annotations

import json
import random
import re
import urllib.error
import urllib.request

from . import prompts, term
from .expand import slot_size, slot_values
from .mine import detect_apex
from .models import Slot, Theme

SYSTEM_PROMPT = prompts.load("enrich")  # subseer/prompts/enrich.txt

_MAX_VALS = 400  # cap values per slot to keep expansion sane


# --- rendering the mined templates for the prompt -----------------------------

def _theme_block(theme, max_vals: int = 40) -> str:
    """Render one mined template + its slot values compactly for the prompt."""
    lines = [f"- {theme.template}"]
    for s in theme.slots:
        vals = slot_values(s, cap=max_vals)          # only materialize what we show
        total = slot_size(s)                          # true count, no materialization
        shown = ", ".join(vals)
        more = f" (+{total - len(vals)} more)" if total > len(vals) else ""
        tag = f" ({s.label})" if s.label else ""
        lines.append(f"    {{{s.name}}}{tag} = {shown}{more}")
    return "\n".join(lines)


def build_enrich_prompt(themes, residual, apex: str, sample: int = 150) -> str:
    """The user message for job A (templates) and/or job B (residual hosts).

    Wording lives in prompts/enrich_templates.txt and prompts/enrich_residual.txt.
    """
    parts = []
    if themes:
        parts.append(prompts.render("enrich_templates", apex=apex,
                                    templates="\n".join(_theme_block(t) for t in themes)))
    resid = residual[:sample]
    if resid or not themes:
        parts.append(prompts.render("enrich_residual", apex=apex, count=len(resid),
                                    hosts="\n".join(resid) or "(none)"))
    return "\n\n".join(parts)


# --- converting the model's output into expandable Theme objects --------------

def _field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


_LABEL_BAD = re.compile(r"[^a-z0-9-]")


def _dns_label(v: str) -> str:
    """Coerce a model value into a valid DNS label ('golden state' -> 'goldenstate').

    A subdomain label is a-z/0-9/hyphen. The model often returns display names
    with spaces; collapse those (spaces removed, underscores -> hyphen) and drop
    anything else, so the value produces a real hostname rather than a broken one.
    """
    v = str(v).strip().lower().replace(" ", "").replace("_", "-")
    v = _LABEL_BAD.sub("", v).strip("-")
    return v


def _slot_from_spec(name: str, values: list[str], label: str = "",
                    meaning: str = "") -> Slot | None:
    vals: list[str] = []
    seen: set[str] = set()
    for v in values:
        v = _dns_label(v)
        if v and v not in seen:
            seen.add(v)
            vals.append(v)
        if len(vals) >= _MAX_VALS:
            break
    if not vals:
        return None
    if all(v.isdigit() for v in vals):
        nums = [int(v) for v in vals]
        pads = {len(v) for v in vals if len(v) > 1 and v[0] == "0"}
        return Slot(name=name, kind="range", min=min(nums), max=max(nums),
                    pad=max(pads) if pads else 0, label=label, meaning=meaning)
    return Slot(name=name, kind="enum", values=vals, label=label, meaning=meaning)


def _add_to_mined(base: Theme, td) -> Theme | None:
    """Mine's pattern with the model's ADDITIONS merged into its full slot lists.

    Job A replies carry only new values (and may leave slots out), because the
    prompt shows at most 40 values per slot - merging here keeps every value Mine
    found, including ones the model never saw. None if nothing new was added.
    """
    specs = {}
    for sp in _field(td, "slots", []) or []:
        nm = (_field(sp, "name") or "").strip().lstrip("{").rstrip("}")
        if nm:
            specs[nm] = sp
    slots, added = [], False
    for slot in base.slots:
        sp = specs.get(slot.name)
        if sp is None:
            slots.append(slot)
            continue
        new = _slot_from_spec(slot.name, _field(sp, "values", []) or [])
        label = str(_field(sp, "label", "") or "").strip()[:40] or slot.label
        meaning = str(_field(sp, "meaning", "") or "").strip()[:200] or slot.meaning
        if slot.kind == "enum" and new is not None:
            extra = [v for v in slot_values(new, cap=_MAX_VALS) if v not in set(slot.values)]
            added = added or bool(extra)
            values = (slot.values + extra)[:max(_MAX_VALS, len(slot.values))]
            slots.append(slot.model_copy(update={"values": values, "label": label,
                                                 "meaning": meaning}))
        elif slot.kind == "range" and new is not None and new.kind == "range":
            lo, hi = min(slot.min, new.min), max(slot.max, new.max)
            added = added or (lo, hi) != (slot.min, slot.max)
            slots.append(slot.model_copy(update={"min": lo, "max": hi, "label": label,
                                                 "meaning": meaning}))
        else:
            slots.append(slot.model_copy(update={"label": label, "meaning": meaning}))
    if not added:
        return None
    label = (_field(td, "label") or "").strip()
    return base.model_copy(update={"name": label or base.name, "slots": slots,
                                   "description": "Mined pattern, extended by the LLM."})


def enrichment_to_themes(themes_data, apex: str, mined_templates) -> list[Theme]:
    """Convert model themes into expandable Theme objects; skip malformed ones.

    ``mined_templates`` maps template -> Mine's Theme (or is a plain set). For a
    mined template the reply is additions only, merged into Mine's pattern; any
    other template is a discovery and must cover every placeholder itself.
    """
    mined_by = mined_templates if isinstance(mined_templates, dict) else {}
    mined_templates = set(mined_templates)
    suffix = "." + apex if apex else ""
    out: list[Theme] = []
    seen: set[str] = set()
    for td in themes_data:
        template = (_field(td, "template") or "").strip().lower()
        if not template:
            continue
        if suffix and not template.endswith(suffix):
            template = template + suffix
        if template in mined_by:  # job A: additions to Mine's pattern
            if template not in seen:
                seen.add(template)
                merged = _add_to_mined(mined_by[template], td)
                if merged is not None:
                    out.append(merged)
            continue
        slots: list[Slot] = []
        for sp in _field(td, "slots", []) or []:
            nm = (_field(sp, "name") or "").strip().lstrip("{").rstrip("}")
            if not nm or ("{" + nm + "}") not in template:
                continue  # slot must correspond to a placeholder in the template
            slot = _slot_from_spec(nm, _field(sp, "values", []) or [],
                                   str(_field(sp, "label", "") or "").strip()[:40],
                                   str(_field(sp, "meaning", "") or "").strip()[:200])
            if slot:
                slots.append(slot)
        # every placeholder in the template must be covered by a slot
        placeholders = set(_placeholders(template))
        if placeholders != {s.name for s in slots}:
            continue
        if template in seen:
            continue
        seen.add(template)
        novel = bool(_field(td, "novel", False)) or template not in mined_templates
        label = (_field(td, "label") or "").strip()
        out.append(Theme(
            name=label or ("discovered" if novel else "enriched"),
            description=("AI-discovered dimension." if novel else "AI-expanded slot values."),
            evidence=[str(e) for e in (_field(td, "evidence", []) or [])][:10],
            kind="template",
            template=template,
            slots=slots,
        ))
    return out


def _placeholders(template: str) -> list[str]:
    return re.findall(r"\{([a-zA-Z0-9_]+)\}", template)


def enrichment_stats(mined, enriched, subs=None) -> dict:
    """Counts for reporting: templates enriched vs discovered, and new values added.

    ``new_values`` = values in enriched slots that the matching mined template did
    NOT already have (Job A), plus values in newly-discovered templates (Job B).
    With ``subs``, Job B counts only values not already in the input (a discovered
    ``{s1}.example.com`` listing your own ``api``/``www`` adds nothing new).
    """
    from .expand import slot_size, slot_values
    from .mine import tokenize

    seen: set[str] = set()  # every label and token in the input
    for h in subs or []:
        for label in h.strip().lower().split("."):
            seen.add(label)
            seen.update(tokenize(label)[0])

    def discovered_new(slot) -> int:
        if not subs or slot.kind != "enum":
            return slot_size(slot)                                 # count only, no materialization
        return sum(1 for v in slot.values if v.strip().lower() not in seen)

    mined_vals: dict[str, dict[str, set]] = {}
    for t in mined:
        # cap materialization - a range slot's members aren't what enrich adds
        # (it expands enum categories), so capping keeps this stat OOM-safe.
        mined_vals[t.template] = {s.name: set(slot_values(s, cap=_MAX_VALS)) for s in t.slots}

    n_enriched = n_discovered = new_values = 0
    for e in enriched:
        base = mined_vals.get(e.template)
        if base is None:
            n_discovered += 1
            new_values += sum(discovered_new(s) for s in e.slots)
        else:
            n_enriched += 1
            for s in e.slots:
                new_values += len(set(slot_values(s, cap=_MAX_VALS)) - base.get(s.name, set()))
    return {"enriched": n_enriched, "discovered": n_discovered, "new_values": new_values}


# --- API path -----------------------------------------------------------------

# --- local Ollama path --------------------------------------------------------

_OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "template": {"type": "string"},
                    "slots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "values": {"type": "array", "items": {"type": "string"}},
                                "label": {"type": "string"},
                                "meaning": {"type": "string"},
                            },
                            "required": ["name", "values"],
                        },
                    },
                    "label": {"type": "string"},
                    "novel": {"type": "boolean"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["template", "slots"],
            },
        }
    },
    "required": ["themes"],
}


def _parse_enrichment(content: str) -> list[dict]:
    """Parse the themes array, salvaging complete objects from a truncated reply."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and isinstance(obj.get("themes"), list):
            return [t for t in obj["themes"] if isinstance(t, dict)]
    except (json.JSONDecodeError, AttributeError):
        pass
    # Salvage: the theme objects sit inside "themes": [ ... ]. Scan from that
    # array's opening bracket and collect each complete {...} object at its top
    # level (a truncated final object is simply skipped).
    key = content.find('"themes"')
    scan_from = content.find("[", key) + 1 if key != -1 else 0
    out: list[dict] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i in range(max(scan_from, 0), len(content)):
        ch = content[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        d = json.loads(content[start:i + 1])
                        if isinstance(d, dict) and "template" in d:
                            out.append(d)
                    except json.JSONDecodeError:
                        pass
                    start = -1
    return out


def _ollama_enrich_once(user: str, model: str, url: str, num_ctx: int,
                        seed: int | None) -> list[dict]:
    options = {"num_predict": 8192, "temperature": 0.4, "num_ctx": num_ctx}
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "stream": False,
        "format": _OLLAMA_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "options": options,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            data = json.loads(r.read().decode())
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach Ollama at {url} ({e.reason if hasattr(e, 'reason') else e}). "
            f"Start it with `ollama serve` and pull the model: `ollama pull {model}`."
        )
    return _parse_enrichment(data.get("message", {}).get("content", ""))


def _slot_pairs(td: dict):
    """Yield (name, values-list) from a theme's slots, skipping malformed entries.

    A JSON-mode model isn't schema-bound, so ``slots`` can come back as strings,
    or values as a bare string instead of a list — tolerate all of it."""
    slots = td.get("slots")
    if not isinstance(slots, list):
        return
    for s in slots:
        if not isinstance(s, dict):
            continue
        nm = s.get("name")
        if not isinstance(nm, str) or not nm:
            continue
        vals = s.get("values", [])
        if isinstance(vals, str):
            vals = [vals]
        elif not isinstance(vals, list):
            continue
        yield nm, [v for v in vals if isinstance(v, (str, int))]


def _merge_theme_data(dst: dict[str, dict], items: list[dict]) -> None:
    """Union model themes across runs, keyed by template, unioning slot values."""
    for td in items:
        if not isinstance(td, dict):
            continue
        template = (td.get("template") or "").strip().lower()
        if not template:
            continue
        cur = dst.get(template)
        if cur is None:
            cur = dst[template] = {
                "template": template,
                "slots": {},
                "label": td.get("label", "") if isinstance(td.get("label"), str) else "",
                "novel": bool(td.get("novel", False)),
                "slot_info": {},
                "evidence": [],
            }
        ev = td.get("evidence")
        for e in (ev if isinstance(ev, list) else []):
            if isinstance(e, str) and e not in cur["evidence"] and len(cur["evidence"]) < 10:
                cur["evidence"].append(e.strip().lower())
        for s in td.get("slots") or []:  # first non-empty label/meaning per slot wins
            if isinstance(s, dict) and isinstance(s.get("name"), str):
                info = cur["slot_info"].setdefault(s["name"], {"label": "", "meaning": ""})
                for k in ("label", "meaning"):
                    if not info[k] and isinstance(s.get(k), str):
                        info[k] = s[k]
        for nm, vals in _slot_pairs(td):
            dest = cur["slots"].setdefault(nm, [])
            have = set(dest)
            for v in vals:
                v = str(v)
                if v not in have:
                    have.add(v)
                    dest.append(v)


# Mined templates per job-A call (a big single call truncates into unparseable JSON).
LOCAL_BATCH, OPENAI_BATCH = 6, 15


def planned_calls(n_templates: int, n_residual: int, runs: int, batch_size: int) -> int:
    """Model calls an enrich run makes: one per template batch, plus ``runs`` discovery
    passes when there are residual hosts."""
    return -(-n_templates // batch_size) + (max(1, runs) if n_residual else 0)


def _enrich_batched(themes, subs, apex, residual, sample, once, runs, seed,
                    progress, batch_size) -> list[Theme]:
    """Shared enrich loop, with the two jobs decoupled by their run-count needs:

    - Job A (expand mined templates): the templates are fixed, so this runs ONCE,
      in batches (a big single call truncates into unparseable JSON).
    - Job B (discover new slots from the residual): Monte Carlo - ``runs`` passes,
      each over a different ``sample``-sized slice of the residual, so more runs
      cover more of a big residual. No template re-work, so runs scale freely.

    ``once(user, seed) -> list[dict]`` does one model call and parses it.
    """
    apex = apex if apex is not None else detect_apex(subs)
    residual = residual if residual is not None else []
    themes = list(themes)
    mined = {t.template: t for t in themes}

    batches = [themes[i:i + batch_size] for i in range(0, len(themes), batch_size)]
    n_runs = max(1, runs)
    do_discovery = bool(residual)
    total_calls = len(batches) + (n_runs if do_discovery else 0)
    merged: dict[str, dict] = {}
    call = 0

    def _tick():
        nonlocal call
        call += 1
        if progress is not None:
            progress(call, total_calls, len(merged))

    def _try_call(user, run_seed):
        """One enrich call, but degrade gracefully: if the AI backend gives up
        (rate limit exhausted / unreachable), keep whatever we've merged so far
        (mined templates + any earlier passes) instead of killing the whole run."""
        try:
            _merge_theme_data(merged, once(user, run_seed))
        except SystemExit as exc:                       # backend unavailable after retries
            term.warn(f"enrich call skipped ({exc}); keeping mined + partial results")
        except Exception as exc:                        # parse/transport hiccup on one call
            term.warn(f"enrich call failed ({exc!r}); keeping mined + partial results")
        _tick()

    # Job A - enrich each template batch once (no residual in these prompts).
    base_seed = seed if seed is not None else 0
    for batch in batches:
        user = build_enrich_prompt(batch, [], apex, sample)
        _try_call(user, base_seed)

    # Job B - Monte-Carlo discovery over the residual, `runs` passes.
    if do_discovery:
        for r in range(n_runs):
            run_seed = seed if (n_runs == 1 and seed is not None) else r
            if len(residual) > sample:
                slice_ = random.Random(run_seed).sample(residual, sample)
            else:
                slice_ = residual
            user = build_enrich_prompt([], slice_, apex, sample)  # residual-only -> discovery
            _try_call(user, run_seed)

    flat = [
        {"template": d["template"], "label": d["label"], "novel": d["novel"],
         "evidence": d["evidence"],
         "slots": [{"name": nm, "values": vals, **d["slot_info"].get(nm, {})}
                   for nm, vals in d["slots"].items()]}
        for d in merged.values()
    ]
    return enrichment_to_themes(flat, apex, mined)


def run_enrich_local(themes, subs, apex: str | None = None, residual=None,
                     model: str = "qwen2.5", sample: int = 150,
                     url: str = "http://localhost:11434", num_ctx: int | None = None,
                     runs: int = 1, seed: int | None = None, progress=None,
                     batch_size: int = LOCAL_BATCH) -> list[Theme]:
    """Enrich + discover slots via a local Ollama model - offline, free, private.

    Mined templates are enriched in small BATCHES (a local model returns nothing
    for a big single call); the residual (job B / discovery) rides with the first
    batch of each run. ``runs`` re-rolls with a different seed and unions results.
    """
    def once(user, run_seed):
        nctx = num_ctx
        if nctx is None:
            nctx = max(8192, (len(SYSTEM_PROMPT) + len(user)) // 4 + 8192 + 1024)
        return _ollama_enrich_once(user, model, url, nctx, run_seed)

    return _enrich_batched(themes, subs, apex, residual, sample, once, runs, seed,
                           progress, batch_size)


def run_enrich_openai(themes, subs, apex: str | None = None, residual=None,
                      model: str = "gpt-4o-mini", sample: int = 150,
                      base_url: str | None = None, api_key: str | None = None,
                      runs: int = 1, seed: int | None = None, progress=None,
                      batch_size: int = OPENAI_BATCH) -> list[Theme]:
    """Enrich + discover slots via an OpenAI-compatible model (e.g. gpt-4o-mini).

    Batches like the local path (larger batches - a hosted model handles more per
    call), so many templates don't overflow one response into an unparseable blob.
    ``runs`` re-rolls with a different seed and unions results. Reads
    $OPENAI_API_KEY unless ``api_key`` is given.
    """
    from .openai_compat import DEFAULT_BASE_URL, chat_json

    base_url = base_url or DEFAULT_BASE_URL

    def once(user, run_seed):
        content = chat_json(model, SYSTEM_PROMPT, user, base_url=base_url,
                            api_key=api_key, max_tokens=16000, temperature=0.4, seed=run_seed)
        return _parse_enrichment(content)

    return _enrich_batched(themes, subs, apex, residual, sample, once, runs, seed,
                           progress, batch_size)


__all__ = [
    "run_enrich_local", "run_enrich_openai", "enrichment_to_themes",
    "enrichment_stats", "build_enrich_prompt", "SYSTEM_PROMPT",
]

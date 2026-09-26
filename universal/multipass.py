"""
Multi-pass agent runner — lets one small-context model act like a large one.

A 2K-context model (moondream) can't hold the 1,800-token system prompt, the
task prompt, a 729-token image and a 35-field answer at once. Truncating the
prompt to a fixed "compact" version throws away most of what the prompt
author wrote. Instead:

1. Prompt packing (every call, every model). The system and task prompts
   are split into blocks at their natural seams (═══ sections, paragraphs).
   Each block is scored for THIS call: the fields it asks for, the
   document's script/country (Ethiopic, Arabic, CJK...), and topics (MRZ,
   address, calendars, names). Mandatory blocks (identity, rules, output
   format, placeholders, return instructions) always go in; the rest fill
   the call up to the model's real budget:

       budget = context − image tokens − output reserve − safety margin

   measured against the model itself (universal/tokens.py), not guessed.

2. Pass planning. If everything relevant doesn't fit, the output schema
   is split into field groups (identity / document / location /
   machine-readable / metadata) across 2 or 3 calls of the same model.
   Each call carries the knowledge relevant to its own fields, so across
   passes the model receives the whole prompt. `auto` keeps one maximally
   packed call when everything relevant fits, and splits only when that
   makes every pass well-informed (AUTO_MIN_COVERAGE); 1/2/3 force it.

   Models whose chat template has no system slot (moondream) get the
   system knowledge folded into the user message — otherwise Ollama drops it.

3. Merge. Passes own disjoint fields, so results union cleanly into one
   schema-shaped answer, with per-pass token accounting in the agent log.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional, Union

from PIL import Image

from . import tokens as T
from .llm import chat_json

logger = logging.getLogger(__name__)

MAX_PASSES = 3
TOKENS_PER_FIELD = 26        # measured: 35-field answers run ~600-900 tokens
OUTPUT_OVERHEAD = 48
SAFETY_MARGIN = 64
TEMPLATE_OVERHEAD = 32       # chat-template tokens around system + user
EXACT_CHECK_ABOVE = 0.85     # verify estimates exactly once a call is this full
# `auto` splits an agent into passes only when that gets every pass to at
# least this share of its relevant knowledge. Measured: moondream run 3x at
# ~0.3 coverage each read FEWER fields than one maximally-packed call —
# splitting only pays when each pass actually becomes well-informed.
AUTO_MIN_COVERAGE = 0.8

FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity", ("full_name", "full_name_latin", "full_name_local", "first_name", "last_name",
                  "middle_name", "date_of_birth", "date_of_birth_orig", "gender", "nationality",
                  "place_of_birth", "blood_type", "surname_as_labelled", "given_names_as_labelled", "sex_as_printed",
                  "date_of_birth_as_printed")),
    ("document", ("document_type", "document_number", "issue_date", "expiry_date", "issuing_country",
                  "issuing_authority", "issuing_place", "personal_number", "document_series",
                  "issue_date_as_printed", "expiry_date_as_printed", "document_number_as_printed",
                  "other_numbers_with_labels")),
    ("location", ("address", "region", "district", "sub_district", "postal_code")),
    ("machine", ("mrz_line_1", "mrz_line_2", "mrz_line_3", "barcode_data")),
    ("meta", ("language_primary", "language_secondary", "script_type", "calendar_system",
              "image_quality", "orientation", "sides_shown", "quick_notes", "overall_confidence",
              "document_quality", "extraction_notes")),
)
ALL_FIELDS = tuple(f for _, fs in FIELD_GROUPS for f in fs)

SCRIPT_TERMS = {
    "Ethiopic": ("ethiopic", "amharic", "tigrinya", "ethiopian", "fayda", "woreda", "kebele", "kifle ketema", "meskerem"),
    "Arabic": ("arabic", "rtl", "urdu", "persian", "farsi", "hijri", "jalali", "pashto", "iqama", "saudi",
               "uae", "emirates", "egyptian", "governorate"),
    "CJK": ("cjk", "chinese", "japanese", "korean", "pinyin", "hepburn", "kanji", "hangul", "居民"),
    "Cyrillic": ("cyrillic", "russian", "ukrainian", "cis countries", "kazakh"),
    "Devanagari": ("devanagari", "hindi", "nepali", "aadhaar", "indian", "marathi", "pincode"),
    "Hebrew": ("hebrew",), "Greek": ("greek",), "Thai": ("thai",),
    "Latin": ("latin script", "european", "eu id", "aamva", "driver license"),
}
COUNTRY_SCRIPTS = {
    "ETH": "Ethiopic", "ERI": "Ethiopic", "CHN": "CJK", "TWN": "CJK", "HKG": "CJK", "JPN": "CJK", "KOR": "CJK",
    "RUS": "Cyrillic", "UKR": "Cyrillic", "KAZ": "Cyrillic", "BLR": "Cyrillic", "BGR": "Cyrillic", "SRB": "Cyrillic",
    "IND": "Devanagari", "NPL": "Devanagari", "ISR": "Hebrew", "GRC": "Greek", "THA": "Thai",
    **{c: "Arabic" for c in ("SAU", "ARE", "EGY", "IRQ", "JOR", "SYR", "LBN", "YEM", "OMN", "QAT", "KWT",
                             "BHR", "MAR", "DZA", "TUN", "LBY", "SDN", "IRN", "AFG", "PAK")},
}
TOPICS = (
    (("mrz", "check digit", "td1", "td3", "filler"),
     {"mrz_line_1", "mrz_line_2", "mrz_line_3", "document_number", "date_of_birth", "expiry_date"}),
    (("address", "woreda", "kebele", "district", "pincode", "governorate", "zip", "street"),
     {"address", "region", "district", "sub_district", "postal_code"}),
    (("calendar", "hijri", "gregorian", "dd.mm", "yymmdd", "date"),
     {"date_of_birth", "date_of_birth_orig", "issue_date", "expiry_date", "calendar_system"}),
    (("name", "transliterat", "romaniz", "family name"),
     {"full_name", "full_name_latin", "full_name_local", "first_name", "last_name", "middle_name"}),
    (("barcode", "qr code"), {"barcode_data"}),
)

_HEADER = re.compile(r"^═{5,}\n(.+?)\n═{5,}$")
_PLACEHOLDER = re.compile(r"\{(previous_extraction|verified_extraction|agent_outputs)\}")
_JSON_TEMPLATE = re.compile(r'\{\s*\n\s*"[a-z0-9_]+"\s*:\s*""')
_RETURN_LINE = re.compile(r"^\s*(RETURN|Return|Always return|Mark any field)")
_ALWAYS_SYSTEM_SECTIONS = {"", "YOUR EXTRACTION RULES", "OUTPUT FORMAT"}


# ─────────────────────────────────────────────────────────────
# Blocks
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Block:
    source: str              # "system" | "user"
    index: int
    section: str
    header: str
    text: str
    placeholder: str = ""
    json_prefix: Optional[str] = None   # text before an embedded JSON template
    group: int = 0                      # paragraph this block came from
    lead: str = ""                      # the paragraph's title line, shared by its items
    forced: bool = False                # item of a mandatory paragraph

    @property
    def mandatory(self) -> bool:
        if self.forced or self.placeholder or self.json_prefix is not None:
            return True
        if self.source == "user" and self.index == 0:
            return True
        if self.source == "system" and self.section in _ALWAYS_SYSTEM_SECTIONS:
            return True
        return bool(_RETURN_LINE.match(self.text))


_SPLIT_ABOVE_CHARS = 360     # paragraphs larger than this are split into list items


def _items(para: str) -> tuple[str, list[str]]:
    """'TIER 1 — ...:\n  Latin: ...\n     cont\n  CJK: ...' -> (lead, [items]).
    An item starts at the shallowest indentation; deeper lines continue it."""
    lines = para.split("\n")
    lead = lines[0] if lines[0].rstrip().endswith(":") else ""
    body = lines[1:] if lead else lines
    indents = [len(l) - len(l.lstrip()) for l in body if l.strip()]
    if not indents:
        return lead, [para]
    base = min(indents)
    items: list[list[str]] = []
    for line in body:
        if line.strip() and len(line) - len(line.lstrip()) == base or not items:
            items.append([line])
        else:
            items[-1].append(line)
    return lead, ["\n".join(i).rstrip() for i in items if "".join(i).strip()]


@lru_cache(maxsize=64)
def split_blocks(template: str, source: str) -> tuple[Block, ...]:
    """Paragraphs at ═══ sections and blank lines; large list paragraphs are
    further split into items so a single relevant line (e.g. 'Ethiopic:
    Amharic...') can be packed without its whole 300-token list."""
    section, header, out = "", "", []
    for group, para in enumerate(re.split(r"\n[ \t]*\n", template.strip())):
        stripped = para.strip("\n")
        m = _HEADER.match(stripped.strip())
        if m:
            section, header = m.group(1).strip(), stripped.strip()
            continue
        ph = _PLACEHOLDER.search(stripped)
        jm = _JSON_TEMPLATE.search(stripped)
        whole = Block(source, len(out), section, header, stripped,
                      ph.group(1) if ph else "", stripped[:jm.start()].rstrip() if jm else None, group)
        if whole.mandatory or len(stripped) <= _SPLIT_ABOVE_CHARS:
            out.append(whole)
            continue
        lead, items = _items(stripped)
        if len(items) < 2:
            out.append(whole)
            continue
        for item in items:
            out.append(Block(source, len(out), section, header, item, group=group, lead=lead))
    return tuple(out)


DOC_TYPE_TERMS = {
    "passport": ("passport",), "national_id": ("national id",), "drivers_license": ("driver licen", "driving"),
    "residence_permit": ("residence", "iqama", "green card"), "refugee_travel_document": ("refugee",),
    "military_id": ("military",), "visa": ("visa",),
}


@dataclass(frozen=True)
class Hints:
    scripts: frozenset = frozenset()
    doc_types: frozenset = frozenset()

    def with_doc_types(self, doc_types: list[str]) -> "Hints":
        return Hints(self.scripts, frozenset(d for d in doc_types if d in DOC_TYPE_TERMS))

    @classmethod
    def from_signals(cls, texts: list[str], countries: list[str]) -> "Hints":
        found = {"Latin"}          # MRZ / English data zones are near-universal
        blob = " ".join(t for t in texts if t).casefold()
        for script, terms in SCRIPT_TERMS.items():
            if script.casefold() in blob or any(t in blob for t in terms):
                found.add(script)
        for c in countries:
            if c in COUNTRY_SCRIPTS:
                found.add(COUNTRY_SCRIPTS[c])
        return cls(frozenset(found) if len(found) > 1 or texts else frozenset())


def score(block: Block, fields: frozenset, hints: Hints) -> float:
    text = block.text.casefold()
    s = 1.0 + (0.5 if block.source == "user" else 0.0)
    mentioned = {f for f in ALL_FIELDS if re.search(rf"\b{f}\b", text)}
    if mentioned:
        s += 3.0 if mentioned & fields else -2.0
    for terms, topic_fields in TOPICS:
        if any(t in text for t in terms):
            s += 2.0 if topic_fields & fields else -1.5
    scripts = {sc for sc, terms in SCRIPT_TERMS.items() if any(t in text for t in terms)}
    if scripts and hints.scripts:
        s += 3.0 if scripts & hints.scripts else -3.0
    if hints.doc_types:
        doc_types = {d for d, terms in DOC_TYPE_TERMS.items() if any(t in text for t in terms)}
        if doc_types:
            s += 2.0 if doc_types & hints.doc_types else -1.0
    return s


def _json_template(fields: tuple[str, ...]) -> str:
    return "{\n" + ",\n".join(f'  "{f}": ""' for f in fields) + "\n}"


def _subset(value: Any, fields: frozenset) -> str:
    if isinstance(value, dict):
        if value and all(isinstance(v, dict) for v in value.values()):
            return json.dumps({a: {k: v for k, v in d.items() if k in fields and v} for a, d in value.items()},
                              ensure_ascii=False, indent=1)
        return json.dumps({k: v for k, v in value.items() if k in fields and v not in ("", None)},
                          ensure_ascii=False)
    return "{}" if value is None else str(value)


def render_block(block: Block, fields: tuple[str, ...], values: dict[str, Any]) -> str:
    text = block.text
    if block.json_prefix is not None:
        text = f"{block.json_prefix}\n{_json_template(fields)}"
    if block.placeholder:
        text = text.replace("{" + block.placeholder + "}", _subset(values.get(block.placeholder), frozenset(fields)))
    return text


def assemble(blocks: list[Block], fields: tuple[str, ...], values: dict[str, Any]) -> str:
    parts: list[str] = []
    emitted_sections: set = set()
    last_group = None
    for b in sorted(blocks, key=lambda b: b.index):
        if b.header and b.section not in emitted_sections:
            parts.append(b.header)
            emitted_sections.add(b.section)
            last_group = None
        text = render_block(b, fields, values)
        if b.lead and b.group == last_group:
            parts[-1] += "\n" + text              # next item of the same list
        else:
            parts.append(f"{b.lead}\n{text}" if b.lead else text)
        last_group = b.group
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────

@dataclass
class PassPlan:
    fields: tuple[str, ...]
    system: str
    user: str
    num_predict: int
    text_budget: int
    prompt_tokens_est: int
    coverage: float                  # share of relevant optional knowledge included
    blocks_included: int
    blocks_omitted: list[str] = field(default_factory=list)
    exact_checked: bool = False


def partition(fields: tuple[str, ...], n: int) -> list[tuple[str, ...]]:
    """Split the schema into n contiguous buckets of whole field groups,
    balanced by field count."""
    groups = [tuple(f for f in fs if f in fields) for _, fs in FIELD_GROUPS]
    leftovers = tuple(f for f in fields if f not in ALL_FIELDS)
    groups = [g for g in groups + [leftovers] if g]
    n = max(1, min(n, len(groups)))
    total, target = sum(map(len, groups)), sum(map(len, groups)) / n
    buckets, current, done = [], [], 0
    for i, g in enumerate(groups):
        current.extend(g)
        done += len(g)
        remaining_groups = len(groups) - i - 1
        remaining_buckets = n - len(buckets) - 1
        if remaining_buckets and (done >= target * (len(buckets) + 1) or remaining_groups == remaining_buckets):
            buckets.append(tuple(current))
            current = []
    if current:
        buckets.append(tuple(current))
    return buckets


def _plan_pass(model: str, ctx: int, image_tok: int, fields: tuple[str, ...], sys_blocks, user_blocks,
               values: dict[str, Any], context: str, hints: Hints) -> PassPlan:
    fset = frozenset(fields)
    reserve = len(fields) * TOKENS_PER_FIELD + OUTPUT_OVERHEAD
    text_budget = ctx - image_tok - reserve - SAFETY_MARGIN
    all_blocks = list(sys_blocks) + list(user_blocks)
    scores = {id(b): score(b, fset, hints) for b in all_blocks}

    def cost(b: Block, sections: set) -> int:
        header_cost = T.estimate(model, b.header) if b.header and (b.source, b.section) not in sections else 0
        lead_cost = T.estimate(model, b.lead) if b.lead and (b.source, "group", b.group) not in sections else 0
        return T.estimate(model, render_block(b, fields, values)) + header_cost + lead_cost

    chosen: list[Block] = []
    sections: set = set()
    used = TEMPLATE_OVERHEAD + T.estimate(model, context)
    def take(b: Block) -> None:
        sections.add((b.source, b.section))
        sections.add((b.source, "group", b.group))
        chosen.append(b)

    for b in all_blocks:
        if b.mandatory:
            used += cost(b, sections)
            take(b)

    optional = sorted((b for b in all_blocks if not b.mandatory),
                      key=lambda b: (-scores[id(b)], b.source != "user", b.index))
    relevant_total = sum(T.estimate(model, b.text) for b in optional if scores[id(b)] > 0)
    relevant_in = 0
    omitted: list[Block] = []
    for b in optional:
        c = cost(b, sections)
        if used + c <= text_budget:
            used += c
            take(b)
            if scores[id(b)] > 0:
                relevant_in += T.estimate(model, b.text)
        else:
            omitted.append(b)

    fold_system = not T.supports_system(model)

    def build() -> tuple[str, str]:
        system = assemble([b for b in chosen if b.source == "system"], fields, values)
        user = assemble([b for b in chosen if b.source == "user"], fields, values)
        user = (user + context) if context else user
        if fold_system:
            return "", f"{system}\n\n{user}"
        return system, user

    system, user = build()
    exact = False
    if used > EXACT_CHECK_ABOVE * text_budget:
        try:
            actual = T.exact_chat_tokens(model, system, user, ctx)
            exact = True
            removable = sorted((b for b in chosen if not b.mandatory), key=lambda b: scores[id(b)])
            while actual > text_budget and removable:
                drop = removable.pop(0)
                chosen.remove(drop)
                omitted.append(drop)
                if scores[id(drop)] > 0:
                    relevant_in -= T.estimate(model, drop.text)
                system, user = build()
                actual = T.exact_chat_tokens(model, system, user, ctx)
            used = actual
        except Exception as e:
            logger.warning("exact token check failed for %s: %s", model, e)

    num_predict = max(64, min(reserve, ctx - image_tok - used - 16))
    return PassPlan(
        fields=fields, system=system, user=user, num_predict=num_predict, text_budget=text_budget,
        prompt_tokens_est=used + image_tok,
        coverage=round(relevant_in / relevant_total, 3) if relevant_total else 1.0,
        blocks_included=len(chosen),
        blocks_omitted=[(b.section or b.source) + ": " + b.text.split("\n", 1)[0][:60] for b in omitted],
        exact_checked=exact,
    )


def _image_size(b64: str) -> tuple[int, int]:
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        return im.size


def plan_agent(
    model: str, system_template: str, user_template: str, fields: tuple[str, ...],
    values: dict[str, Any], context: str, hints: Hints, images: Optional[list[str]],
    passes: Union[str, int] = "auto", max_passes: int = MAX_PASSES,
) -> list[PassPlan]:
    ctx = T.context_limit(model)
    image_tok = sum(T.image_tokens_for(model, *_image_size(img)) for img in images or [])
    sys_blocks = split_blocks(system_template, "system")
    user_blocks = split_blocks(user_template, "user")
    def plans_for(n: int) -> list[PassPlan]:
        return [_plan_pass(model, ctx, image_tok, bucket, sys_blocks, user_blocks, values, context, hints)
                for bucket in partition(fields, n)]

    if passes != "auto":
        return plans_for(max(1, min(int(passes), max_passes)))
    single = plans_for(1)
    if single[0].coverage >= 0.999:
        return single
    for n in range(2, max_passes + 1):
        plans = plans_for(n)
        if min(p.coverage for p in plans) >= AUTO_MIN_COVERAGE:
            return plans
    return single            # splitting wouldn't make passes well-informed: one packed call


# ─────────────────────────────────────────────────────────────
# Running
# ─────────────────────────────────────────────────────────────

def run_agent(
    model: str, system_template: str, user_template: str, fields: tuple[str, ...],
    values: dict[str, Any], context: str, hints: Hints, images: Optional[list[str]],
    passes: Union[str, int] = "auto",
) -> tuple[dict[str, str], dict[str, Any]]:
    """-> (merged fields, log entry). A failed pass leaves its fields empty;
    raises only if every pass fails."""
    plans = plan_agent(model, system_template, user_template, fields, values, context, hints, images, passes)
    ctx = T.context_limit(model)
    merged: dict[str, str] = {}
    pass_logs, errors = [], []
    for i, plan in enumerate(plans, start=1):
        started = time.time()
        entry: dict[str, Any] = {
            "pass": i, "fields": len(plan.fields), "field_group": _group_names(plan.fields),
            "prompt_tokens_planned": plan.prompt_tokens_est, "context_limit": ctx,
            "knowledge_coverage": plan.coverage, "blocks_included": plan.blocks_included,
            "blocks_omitted": len(plan.blocks_omitted), "exact_checked": plan.exact_checked,
        }
        try:
            data, meta = chat_json(model, plan.system, plan.user, images=images, fields=plan.fields,
                                   num_ctx=ctx, num_predict=plan.num_predict)
            entry.update(prompt_tokens=meta.get("prompt_tokens"), output_tokens=meta.get("output_tokens"),
                         complete_json=meta.get("complete_json"))
            for f in plan.fields:
                if data.get(f) and not merged.get(f):
                    merged[f] = data[f]
        except Exception as e:
            errors.append(str(e)[:200])
            entry["error"] = str(e)[:200]
        entry["duration_seconds"] = round(time.time() - started, 2)
        pass_logs.append(entry)
    if errors and len(errors) == len(plans):
        raise RuntimeError("; ".join(errors))
    log = {
        "passes": len(plans),
        "context_limit": ctx,
        "knowledge_coverage": round(min(p.coverage for p in plans), 3),
        "prompt_tokens": sum(p.get("prompt_tokens") or 0 for p in pass_logs),
        "output_tokens": sum(p.get("output_tokens") or 0 for p in pass_logs),
        "pass_details": pass_logs,
    }
    return {f: merged.get(f, "") for f in fields}, log


def _group_names(fields: tuple[str, ...]) -> str:
    names = [name for name, fs in FIELD_GROUPS if set(fs) & set(fields)]
    return "+".join(names) or "other"

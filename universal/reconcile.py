"""
Reconciliation — turns five agents' JSON plus the machine-verified sources
into one result with per-field value, confidence, provenance and flags.

Precedence (same discipline as idx/pipeline.py's CLAUDE.md table):

    barcode (tier 1)  >  MRZ check digit / ID-number checksum (tier 2)
                      >  agent consensus (tier 4, VLM — unverifiable)

Within the agent tier:
  1. The supervisor's pick wins — but only if at least one agent actually
     produced that value. A supervisor value no agent saw is a hallucination
     by its own Rule 4 ("never fabricate"), and is rejected in code.
  2. Otherwise the majority across agents; ties go to the field's
     specialist agent (SUPERVISOR_PROMPT Rule 2), then the field extractor.
  3. Confidence comes from agreement: how many agents independently read
     the same value, minus a penalty for self-reported uncertainty.

A verified source that disagrees with the agents never silently wins on a
critical field: the value switches to the verified one, the agents' read is
kept under `alternatives`, and human review is requested.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

from .dates import normalize_calendar_name, normalize_date
from .id_numbers import PRIMARY_DOCUMENT_SCHEMES, IDNumberCheck
from .normalize import (
    FLAG_LOW_CONFIDENCE, FLAG_UNCERTAIN, clean_name, comparison_key, detect_script,
    is_latin, normalize_blood_type, normalize_document_type, normalize_sex, resolve_country,
    resolve_nationality, split_bilingual_name, strip_identifier_label, strip_markers,
)
from .verify import VerifiedSource, clean_mrz_line

VISION_AGENTS = ("fast_scanner", "field_extractor", "accuracy_verifier", "layout_analyzer")

NAME_FIELDS = ("full_name", "full_name_latin", "full_name_local", "first_name", "last_name", "middle_name")
DATE_FIELDS = ("date_of_birth", "issue_date", "expiry_date")
IDENTIFIER_FIELDS = ("document_number", "personal_number")
CRITICAL_FIELDS = ("full_name", "document_number", "date_of_birth", "expiry_date", "nationality", "issuing_country")

# SUPERVISOR_PROMPT Rule 2, applied deterministically when the supervisor
# is unavailable or its pick is rejected.
SPECIALIST = {
    "document_number": "accuracy_verifier", "mrz_line_1": "accuracy_verifier",
    "mrz_line_2": "accuracy_verifier", "mrz_line_3": "accuracy_verifier",
    "full_name": "field_extractor", "full_name_local": "field_extractor",
    "full_name_latin": "field_extractor",
    "address": "layout_analyzer", "district": "layout_analyzer", "sub_district": "layout_analyzer",
    "region": "layout_analyzer", "issuing_authority": "layout_analyzer",
    "document_type": "fast_scanner", "language_primary": "fast_scanner",
}

# SUPERVISOR_PROMPT Rule 3 ("if field_extractor has Amharic/Arabic/CJK value
# -> keep it") — enforced in code because later stages copy each other.
SCRIPT_FIELDS = ("full_name_local",)

# MRZ fields that may override the agents, and the trust they need to.
# Names are deliberately absent: the MRZ transliterates (MÜLLER -> MUELLER)
# and truncates long names, so it corroborates the visual name, never replaces it.
_MRZ_OVERRIDE_MIN = {
    "document_number": 0.99, "date_of_birth": 0.99, "expiry_date": 0.99, "personal_number": 0.99,
    "gender": 0.9, "nationality": 0.9, "issuing_country": 0.9, "document_type": 0.9,
    "mrz_line_1": 0.9, "mrz_line_2": 0.9, "mrz_line_3": 0.9,
}


@dataclass
class FieldResult:
    value: str = ""
    confidence: float = 0.0
    source: str = ""
    agreeing_agents: list[str] = field(default_factory=list)
    verified_by: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    original: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 3)
        return {k: v for k, v in d.items() if v not in ([], "") or k in ("value", "confidence")}


@dataclass
class Context:
    country: str = ""          # best-known issuing country (ISO alpha-3)
    calendar: str = ""         # best-known calendar_system
    warnings: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)

    def review(self, reason: str) -> None:
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


# ─────────────────────────────────────────────────────────────
# Per-agent normalization
# ─────────────────────────────────────────────────────────────

# Field labels an agent sometimes returns in place of the value ("NAME",
# "NAME KEBBEDE"): never part of a person's name.
_NAME_LABEL_WORDS = {
    "NAME", "NAMES", "SURNAME", "SURNAMES", "GIVEN", "FIRST", "LAST", "FULL", "FAMILY", "FORENAME", "FORENAMES",
    "NOM", "NOMS", "PRENOM", "PRENOMS", "PRÉNOM", "PRÉNOMS", "NAAM", "VOORNAMEN", "VORNAME", "VORNAMEN",
    "NACHNAME", "COGNOME", "NOME", "APELLIDO", "APELLIDOS", "NOMBRE", "NOMBRES", "EFTERNAMN", "FÖRNAMN",
    "LN", "FN",
}


def _mixed_script(value: str) -> bool:
    """A name written in one script: Latin letters mixed into an Ethiopic /
    Arabic / CJK name ('pora ተጋባ ሕክታራ') are a garbled read. Parenthesised
    transliterations are handled separately."""
    if "(" in value:
        return False
    scripts = {detect_script(t) for t in value.split() if any(c.isalpha() for c in t)}
    return "Latin" in scripts and len(scripts) > 1


def _strip_label_words(value: str) -> str:
    tokens = [t for t in re.split(r"[\s/<>]+", value) if t.strip(".:,[]()")]
    bare = lambda t: t.upper().strip(".:,[]()")
    while tokens and bare(tokens[0]) in _NAME_LABEL_WORDS:
        tokens = tokens[1:]
    while tokens and bare(tokens[-1]) in _NAME_LABEL_WORDS:
        tokens = tokens[:-1]
    if all(bare(t) in _NAME_LABEL_WORDS for t in tokens):
        return ""
    return " ".join(tokens)


def _prompt_examples() -> frozenset[str]:
    """Example values quoted in the prompts file ('e.g. "15/Meskerem/1982 EC"',
    '"MESAY TADESSE"', '784-YYYY-XXXXXXX-X'). An agent returning one verbatim
    copied the prompt, not the document."""
    import universal_id_extraction_prompts as P
    text = " ".join(v for v in vars(P).values() if isinstance(v, str))
    quoted = re.findall(r'e\.g\.\s*"([^"]{4,})"|\("([^"()]{4,})"\)', text)
    examples = {a or b for a, b in quoted}
    examples |= set(re.findall(r'"([^"\n]{5,40})"\)', text))
    # Only personal-data-shaped examples (several words, or digits) — never a
    # generic word like "PASSPORT" that a document legitimately prints.
    return frozenset(e.strip().casefold() for e in examples
                     if len(e.split()) >= 2 or re.search(r"\d", e))


_PROMPT_EXAMPLES: Optional[frozenset] = None
_ECHO_CHECKED = set(NAME_FIELDS) | set(DATE_FIELDS) | set(IDENTIFIER_FIELDS) | {"date_of_birth_orig"}


def is_prompt_echo(value: str) -> bool:
    global _PROMPT_EXAMPLES
    if _PROMPT_EXAMPLES is None:
        _PROMPT_EXAMPLES = _prompt_examples()
    return value.strip().casefold() in _PROMPT_EXAMPLES


def normalize_value(name: str, raw: str, ctx: Context) -> tuple[str, set[str]]:
    value, flags = strip_markers(raw)
    if not value:
        return "", flags
    if name in _ECHO_CHECKED and is_prompt_echo(value):
        return "", flags | {"prompt_example_echoed"}
    if name in NAME_FIELDS:
        value = _strip_label_words(value)
        if not value:
            return "", flags | {"label_word_not_a_name"}
    if name == "full_name_local" and _mixed_script(value):
        return "", flags | {"mixed_script"}
    if name == "full_name_local" and is_latin(value) and "(" not in value:
        return "", flags | {"latin_in_local_script_field"}
    if name in NAME_FIELDS:
        return clean_name(value), flags
    if name in DATE_FIELDS:
        result = normalize_date(value, calendar_hint=ctx.calendar, country=ctx.country,
                                future_leaning=(name == "expiry_date"))
        if result.approximate:
            flags.add("approximate_date")
        if result.ambiguous:
            flags.add("ambiguous_day_month")
        if not result.iso:
            flags.add("unparsed_date")
        return result.iso, flags
    if name == "gender":
        sex, sex_flags = normalize_sex(value)
        return sex, flags | sex_flags
    if name == "nationality":
        code, demonym = resolve_nationality(value)
        return (demonym or value.upper()), flags
    if name == "issuing_country":
        code, _ = resolve_country(value)
        if not code:
            flags.add("unresolved_country")
        return code or value.upper(), flags
    if name == "document_type":
        return normalize_document_type(value), flags
    if name == "blood_type":
        return normalize_blood_type(value), flags
    if name.startswith("mrz_line"):
        line = clean_mrz_line(value)
        if len(line) < 28 or ("<" not in line and len(line) not in (30, 36, 44)):
            return "", flags | {"not_mrz_shaped"}
        return line, flags
    if name == "calendar_system":
        canonical = normalize_calendar_name(value)
        return (canonical.replace("_", " ").title() if canonical else value), flags
    if name in IDENTIFIER_FIELDS:
        return " ".join(strip_identifier_label(value).split()).upper(), flags
    return value, flags


def _key(name: str, value: str) -> str:
    if name == "nationality":
        code, _ = resolve_nationality(value)
        if code:
            return code
    if name in NAME_FIELDS:
        # Diacritics and token order ("SURNAME, GIVEN" vs "GIVEN SURNAME") aren't disagreements.
        folded = unicodedata.normalize("NFKD", value)
        value = "".join(c for c in folded if not unicodedata.combining(c))
        return " ".join(sorted(comparison_key(name, value).split()))
    if name.startswith("mrz_line"):
        return clean_mrz_line(value).rstrip("<")  # filler count is not a disagreement
    return comparison_key(name, value)


# ─────────────────────────────────────────────────────────────
# Agent consensus
# ─────────────────────────────────────────────────────────────

def _agent_confidence(n_agree: int, n_agents: int, flags: set[str]) -> float:
    if n_agents <= 1:
        base = 0.6
    else:
        base = 0.35 + 0.5 * (n_agree / n_agents)
    if flags & {FLAG_LOW_CONFIDENCE, FLAG_UNCERTAIN}:
        base -= 0.15
    if "approximate_date" in flags:
        base -= 0.05
    return max(0.1, min(0.85, base))


def consensus(
    name: str,
    agent_values: dict[str, tuple[str, set[str], str]],
    supervisor: Optional[tuple[str, set[str], str]],
    n_agents: int,
) -> FieldResult:
    """agent_values: agent -> (normalized value, flags, raw)."""
    groups: dict[str, list[str]] = defaultdict(list)
    display: dict[str, str] = {}
    for agent, (value, _, _) in agent_values.items():
        if not value:
            continue
        k = _key(name, value)
        groups[k].append(agent)
        display.setdefault(k, value)
    if not groups:
        return FieldResult()

    chosen: Optional[str] = None
    source = "consensus"
    extra_flags: set[str] = set()
    if supervisor and supervisor[0]:
        sk = _key(name, supervisor[0])
        if sk in groups:
            # Supervisor picks WHICH value; the text shown stays as an agent read it off the image.
            chosen, source = sk, "supervisor"
        else:
            extra_flags.add("supervisor_value_rejected")

    # SUPERVISOR_PROMPT Rule 3: the field extractor's native-script read is kept.
    if chosen is None and name in SCRIPT_FIELDS:
        extractor_value = agent_values.get("field_extractor", ("",))[0]
        if extractor_value and not is_latin(extractor_value):
            chosen, source = _key(name, extractor_value), "script_rule"

    if chosen is None:
        specialist = SPECIALIST.get(name, "field_extractor")

        def rank(k: str) -> tuple:
            members = groups[k]
            return (len(members), specialist in members, "field_extractor" in members, len(display[k]))
        chosen = max(groups, key=rank)

    agreeing = groups[chosen]
    if name in SPECIALIST and SPECIALIST[name] in agreeing and source != "supervisor":
        display[chosen] = agent_values[SPECIALIST[name]][0]
    flags = set().union(*(agent_values[a][1] for a in agreeing)) | extra_flags
    raw = next((agent_values[a][2] for a in agreeing if agent_values[a][2]), "")
    return FieldResult(
        value=display[chosen],
        confidence=_agent_confidence(len(agreeing), n_agents, flags),
        source=source,
        agreeing_agents=sorted(agreeing, key=VISION_AGENTS.index),
        flags=sorted(flags),
        alternatives=[display[k] for k in groups if k != chosen],
        original=raw if raw != display[chosen] else "",
    )


# ─────────────────────────────────────────────────────────────
# Verified sources
# ─────────────────────────────────────────────────────────────

def _names_agree(visual: str, mrz: str) -> bool:
    """MRZ names are transliterated and may be truncated: agree when every
    MRZ token prefixes some visual token."""
    def fold(s: str) -> list[str]:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c)).upper()
        return [t for t in "".join(c if c.isalpha() else " " for c in s).split() if t]
    v, m = fold(visual), fold(mrz)
    return bool(m) and all(any(vt.startswith(mt) or mt.startswith(vt) for vt in v) for mt in m)


def apply_verified(fields: dict[str, FieldResult], src: VerifiedSource, ctx: Context) -> None:
    for name, value in src.fields.items():
        if name not in fields or not value:
            continue
        conf = src.confidence_for(name)
        current = fields[name]
        if src.source == "mrz" and name in NAME_FIELDS:
            issuing_state = (src.fields.get("mrz_line_1", "")[2:5]).strip("<")
            readings = _mrz_name_readings(src, name)
            if readings and fields[name].value and any(_names_agree(fields[name].value, r) for r in readings):
                # The MRZ name line is disputed between OCR readings and the visual
                # read matches one of them: agreement, not a conflict.
                fields[name].verified_by.append("mrz_name_reading")
                continue
            _corroborate_name(fields, name, value, conf, ctx, issuing_state, readings=_mrz_name_readings(src, name))
            continue
        if name in NAME_FIELDS and current.value and not is_latin(current.value) and is_latin(value):
            # Latin barcode name vs a local-script read: different slots, not a conflict.
            continue
        if src.source == "mrz" and conf < _MRZ_OVERRIDE_MIN.get(name, 1.01):
            if current.value and _key(name, current.value) == _key(name, value):
                current.verified_by.append("mrz_unprotected_field")
            continue
        _override(current, name, value, src.source, conf, ctx)


def _same_with_check_digit(name: str, read: str, verified: str) -> bool:
    """Agents often copy the document number together with its MRZ check digit."""
    a, b = comparison_key(name, read), comparison_key(name, verified)
    return name == "document_number" and len(a) == len(b) + 1 and a.startswith(b)


def _override(current: FieldResult, name: str, value: str, source: str, conf: float, ctx: Context) -> None:
    if current.value and (_key(name, current.value) == _key(name, value)
                          or _same_with_check_digit(name, current.value, value)):
        current.verified_by.append(source)
        current.confidence = max(current.confidence, conf)
        current.source = source
        current.value = value if name in DATE_FIELDS or name in IDENTIFIER_FIELDS else current.value
        return
    if current.value:
        current.alternatives = [current.value] + [a for a in current.alternatives if a != current.value]
        ctx.warn(f"{name}: {source} value {value!r} replaced the agents' read {current.value!r}")
        if name in CRITICAL_FIELDS:
            ctx.review(f"{name}: agents' read disagrees with the {source}")
    current.value = value
    current.confidence = conf
    current.source = source
    current.verified_by = [source]
    current.agreeing_agents = []


def _looks_mrz_copied(value: str, mrz_value: str, issuing_state: str) -> bool:
    """'UTOERIKSSON ANNA MARIA': an agent copied MRZ line 1 with the issuing
    state glued to the surname (and fillers misread) into a name field."""
    tokens = value.upper().split()
    if not tokens or not issuing_state:
        return False
    glued = tokens[0].startswith(("P" + issuing_state, "I" + issuing_state, issuing_state))
    return glued and _names_agree(" ".join([tokens[0].split(issuing_state, 1)[-1]] + tokens[1:]), mrz_value)


def _small_spelling_gap(visual: str, mrz: str) -> bool:
    def fold(v: str) -> str:
        v = unicodedata.normalize("NFKD", v.upper())
        return "".join(c for c in v if c.isalpha())
    a, b = fold(visual), fold(mrz)
    if not a or not b or abs(len(a) - len(b)) > 2:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 0 < prev[-1] <= 2


def _mrz_name_readings(src: VerifiedSource, name: str) -> list[str]:
    """Each disputed MRZ name-line reading, rendered as this field's value."""
    out = []
    for line in src.details.get("name_line_readings", []):
        body = line[5:] if src.details.get("format") in ("TD2", "TD3") else line
        surname, _, given = body.partition("<<")
        surname, given = surname.replace("<", " ").strip(), given.replace("<", " ").strip()
        value = {"last_name": surname, "first_name": given}.get(name, f"{given} {surname}".strip())
        if value:
            out.append(value)
    return out


def _corroborate_name(fields: dict[str, FieldResult], name: str, mrz_value: str, conf: float, ctx: Context,
                      issuing_state: str = "", readings: Optional[list[str]] = None) -> None:
    target = fields[name]
    if not target.value:
        target.value, target.source, target.confidence = mrz_value, "mrz", min(conf, 0.85)
        target.verified_by = ["mrz"]
        return
    mrz_surname = fields["last_name"].value if name == "first_name" else ""
    if name == "first_name" and mrz_surname and _key(name, target.value) == _key(name, mrz_surname) \
            and _key(name, mrz_value) != _key(name, mrz_surname):
        # The agent put the SURNAME into first_name (e.g. a card back showing only the MRZ).
        # A disputed MRZ name line: prefer the reading the agents' own name reads support.
        agent_letters = " ".join(fields[n].value for n in ("full_name", "full_name_latin", "middle_name")).upper()
        for reading in readings or []:
            if all(tok in agent_letters for tok in reading.upper().split()):
                mrz_value = reading
                break
        ctx.warn(f"first_name {target.value!r} is the surname; the MRZ given names are {mrz_value!r}")
        target.alternatives = [target.value] + target.alternatives
        target.value, target.source, target.confidence = mrz_value, "mrz", conf
        target.verified_by = ["mrz"]
        return
    if _names_agree(target.value, mrz_value):
        target.verified_by.append("mrz")
        target.confidence = max(target.confidence, 0.9)
        return
    if name in ("last_name", "first_name") and _names_agree(" ".join(
            fields[n].value for n in ("full_name", "full_name_latin", "first_name", "last_name", "middle_name")), mrz_value):
        # The MRZ states which tokens are the surname (SURNAME<<GIVEN); the visual
        # zone only prints the names. Its split outranks an agent's positional guess.
        ctx.warn(f"{name}: {target.value!r} replaced by the MRZ's {name.replace('_', ' ')} {mrz_value!r}")
        target.alternatives = [target.value] + target.alternatives
        target.value, target.source, target.confidence = mrz_value, "mrz", max(conf, 0.9)
        target.verified_by = ["mrz"]
        return
    if _looks_mrz_copied(target.value, mrz_value, issuing_state):
        ctx.warn(f"{name}: {target.value!r} is a raw copy of the MRZ line — replaced with the MRZ name")
        target.alternatives = [target.value] + target.alternatives
        target.value, target.source, target.confidence = mrz_value, "mrz", min(conf, 0.9)
        target.verified_by = ["mrz"]
        return
    visual_tokens = set(_key(name, target.value).split())
    mrz_tokens = set(_key(name, mrz_value).split())
    if name in ("full_name", "full_name_latin") and visual_tokens and visual_tokens < mrz_tokens:
        # The visual read dropped a name (often the surname); the MRZ has every token plus it.
        ctx.warn(f"{name}: {target.value!r} completed from the MRZ as {mrz_value!r}")
        target.alternatives = [target.value] + target.alternatives
        target.value, target.source, target.confidence = mrz_value, "mrz", max(conf, 0.85)
        target.verified_by = ["mrz"]
        return
    full = fields["full_name"]
    if name == "full_name_latin" and full.value and is_latin(full.value) and _names_agree(full.value, mrz_value):
        # The visual full name already agrees with the MRZ; the Latin slot was mis-copied.
        target.alternatives = [target.value] + target.alternatives
        target.value, target.source, target.confidence = full.value, full.source, max(full.confidence, 0.9)
        target.verified_by = ["mrz"]
        return
    if is_latin(target.value) and conf >= 0.9 and _small_spelling_gap(target.value, mrz_value):
        # A cleanly-read MRZ (every check digit passing) and a visual read one or
        # two letters apart: the visual zone was misread ('LEMA' for 'LEMMA').
        ctx.warn(f"{name}: visual read {target.value!r} corrected to the MRZ spelling {mrz_value!r}")
        target.alternatives = [target.value] + target.alternatives
        target.value, target.source, target.confidence = mrz_value, "mrz", conf
        target.verified_by = ["mrz"]
        return
    if is_latin(target.value):
        ctx.warn(f"{name}: visual read {target.value!r} does not match MRZ {mrz_value!r}")
        if name in ("full_name_latin", "last_name"):
            ctx.review(f"{name}: visual zone and MRZ disagree")


def apply_id_number_checks(fields: dict[str, FieldResult], checks: list[IDNumberCheck], ctx: Context) -> list[dict]:
    reports = []
    for check in checks:
        if check.details.get("field") == "document_number" and check.details.get("role") == "personal":
            # A personal identity number (personnummer, national register number) is not the card's number.
            moved = fields["document_number"]
            if not fields["personal_number"].value:
                fields["personal_number"] = FieldResult(value=moved.value, confidence=moved.confidence,
                                                        source=moved.source, agreeing_agents=list(moved.agreeing_agents))
            alt = next((a for a in moved.alternatives if a and a != moved.value), "")
            fields["document_number"] = FieldResult(value=alt, confidence=0.5 if alt else 0.0,
                                                    source="alternative" if alt else "",
                                                    alternatives=[moved.value])
            check.details["field"] = "personal_number"
            ctx.warn(f"{moved.value!r} is a {check.country} {check.scheme} (personal number), not the document "
                     f"number — moved to personal_number")
        if (check.details.get("field") == "personal_number" and check.scheme in PRIMARY_DOCUMENT_SCHEMES
                and check.format_valid and check.checksum_valid is not False and not fields["document_number"].value):
            # The card's own number landed in personal_number; it is the document number.
            fields["document_number"], fields["personal_number"] = fields["personal_number"], FieldResult()
            check.details["field"] = "document_number"
            ctx.warn(f"document_number taken from personal_number: it is the card's {check.scheme}")
        target = fields.get(check.details.get("field", "document_number"))
        report = {
            "country": check.country, "scheme": check.scheme, "field": check.details.get("field"),
            "format_valid": check.format_valid, "checksum_valid": check.checksum_valid,
            "notes": check.notes,
            **{k: v for k, v in check.details.items() if k not in ("field",)},
        }
        if check.details.get("inferred"):
            country_field = fields["issuing_country"]
            if check.verified and (not country_field.value or "unresolved_country" in country_field.flags):
                ctx.warn(f"issuing_country set to {check.country}: the document number is a checksum-valid "
                         f"{check.country} {check.scheme}")
                country_field.alternatives = [v for v in [country_field.value] if v] + country_field.alternatives
                country_field.value, country_field.source = check.country, f"id_number:{check.scheme}"
                country_field.confidence, country_field.verified_by = 0.9, [f"checksum:{check.scheme}"]
                country_field.flags = [f for f in country_field.flags if f != "unresolved_country"]
            else:
                ctx.warn(f"number matches the {check.country} {check.scheme} format — issuing country may be {check.country}")
        if target is not None and check.verified:
            target.verified_by.append(f"checksum:{check.scheme}")
            target.confidence = max(target.confidence, 0.95)
        elif target is not None and check.checksum_valid is False:
            target.flags.append("checksum_failed")
            target.confidence = min(target.confidence, 0.3)
            ctx.review(f"{check.details.get('field')}: fails the {check.country} {check.scheme} checksum")
        _check_name_letters(fields, check, ctx)
        strong = check.verified
        structural = check.format_valid and check.checksum_valid is None
        if check.date_of_birth and (strong or structural):
            _cross_check(fields, "date_of_birth", check.date_of_birth, check.scheme, strong, ctx)
        if check.sex and (strong or structural):
            _cross_check(fields, "gender", check.sex, check.scheme, strong, ctx)
        if check.birth_year and fields["date_of_birth"].value[:4] not in ("", check.birth_year):
            ctx.warn(f"date_of_birth year differs from the {check.scheme} birth year {check.birth_year}")
            if strong:
                ctx.review("date_of_birth: disagrees with the birth year encoded in the ID number")
        reports.append(report)
    return reports


def _check_name_letters(fields: dict[str, FieldResult], check: IDNumberCheck, ctx: Context) -> None:
    """Licence numbers that encode the surname's first letters (DVLA, Ontario)
    verify the surname — and expose a surname/given-names swap."""
    prefix = check.details.get("surname_prefix", "")
    if not prefix:
        return
    last, first = fields["last_name"], fields["first_name"]
    fold = lambda v: "".join(c for c in unicodedata.normalize("NFKD", v.upper()) if c.isalpha())
    if fold(last.value).startswith(prefix):
        last.verified_by.append(f"id_number:{check.scheme}")
        last.confidence = max(last.confidence, 0.9)
    elif fold(first.value).startswith(prefix) and first.value:
        fields["last_name"], fields["first_name"] = first, last
        for fr in (first, last):
            fr.flags.append("swapped_by_id_number")
        ctx.warn(f"surname and given names swapped: the {check.scheme} encodes surname letters {prefix!r}")
    elif last.value:
        # One name token carries the encoded surname letters: it is the surname,
        # the others (in printed order) are the given names.
        order: list[str] = []
        for text in (fields["full_name"].value, first.value, last.value, *first.alternatives, *last.alternatives):
            for t in (text or "").replace(",", " ").split():
                if fold(t) and fold(t) not in {fold(o) for o in order}:
                    order.append(t)
        hits = [t for t in order if fold(t).startswith(prefix)]
        if len(hits) == 1 and len(order) > 1:
            given = " ".join(t for t in order if t != hits[0])
            full = fields["full_name"]
            if fold(hits[0]) not in fold(full.value):
                full.alternatives = [v for v in [full.value] if v] + full.alternatives
                full.value, full.source = f"{hits[0]} {given}", f"id_number:{check.scheme}"
            ctx.warn(f"name structure rebuilt from the {check.scheme}: surname {hits[0]!r} carries the encoded "
                     f"letters {prefix!r}")
            fields["last_name"] = FieldResult(value=hits[0], confidence=0.85, source=f"id_number:{check.scheme}",
                                              verified_by=[f"id_number:{check.scheme}"], alternatives=[last.value])
            fields["first_name"] = FieldResult(value=given, confidence=0.8, source=f"id_number:{check.scheme}",
                                               alternatives=[first.value] if first.value else [])
            return
        last.flags.append("disagrees_with_id_number")
        ctx.warn(f"last_name {last.value!r} doesn't start with {prefix!r} encoded in the {check.scheme}")
        ctx.review(f"last_name: disagrees with the surname letters encoded in the {check.scheme}")


def _cross_check(fields: dict[str, FieldResult], name: str, derived: str, scheme: str, strong: bool, ctx: Context) -> None:
    target = fields[name]
    source = f"id_number:{scheme}"
    if not target.value:
        target.value, target.source = derived, source
        target.confidence = 0.9 if strong else 0.7
        target.verified_by = [source]
        return
    if target.value == derived:
        target.verified_by.append(source)
        target.confidence = max(target.confidence, 0.97 if strong else 0.85)
        return
    ctx.warn(f"{name}: read {target.value!r} but the {scheme} number encodes {derived!r}")
    if strong:
        _override(target, name, derived, source, 0.9, ctx)
        ctx.review(f"{name}: disagrees with the value encoded in the checksum-valid ID number")
    else:
        target.flags.append("disagrees_with_id_number")
        target.confidence = min(target.confidence, 0.5)


# ─────────────────────────────────────────────────────────────
# Names
# ─────────────────────────────────────────────────────────────

def settle_names(fields: dict[str, FieldResult]) -> None:
    """Rule 3 of SYSTEM_PROMPT puts a transliteration in parentheses after a
    non-Latin name; split it out, and fill local/latin slots from full_name."""
    full = fields["full_name"]
    local, latin = split_bilingual_name(full.value)
    if local and latin:
        full.value = local
    if local and not fields["full_name_local"].value:
        fields["full_name_local"] = FieldResult(value=local, confidence=full.confidence,
                                                source=full.source, agreeing_agents=list(full.agreeing_agents))
    if latin and not fields["full_name_latin"].value:
        fields["full_name_latin"] = FieldResult(value=latin, confidence=full.confidence,
                                                source=full.source, agreeing_agents=list(full.agreeing_agents))
    for slot in ("full_name_local", "full_name_latin"):
        loc, lat = split_bilingual_name(fields[slot].value)
        if loc and lat:
            fields[slot].value = loc if slot == "full_name_local" else lat
    loc_field = fields["full_name_local"]
    if loc_field.value and is_latin(loc_field.value):
        if not fields["full_name_latin"].value:
            fields["full_name_latin"] = loc_field
        fields["full_name_local"] = FieldResult()
    if not full.value:
        best = fields["full_name_latin"] if fields["full_name_latin"].value else fields["full_name_local"]
        if best.value:
            fields["full_name"] = FieldResult(**{**asdict(best), "flags": list(best.flags) + ["derived"]})


_FAMILY_NAME_FIRST_SCRIPTS = ("CJK",)


def settle_name_order(fields: dict[str, FieldResult], ctx: Context) -> None:
    """SYSTEM_PROMPT: in CJK names the family name comes FIRST. Agents often
    assign first_name/last_name by position instead ('王芳' -> first=王,
    last=芳); when the parts concatenate to the full name in that order,
    they're swapped back."""
    full = fields["full_name_local"].value or fields["full_name"].value
    first, last = fields["first_name"], fields["last_name"]
    if not (full and first.value and last.value) or detect_script(full) not in _FAMILY_NAME_FIRST_SCRIPTS:
        return
    squash = lambda s: "".join(s.split())
    if squash(first.value) + squash(last.value) == squash(full):
        fields["first_name"], fields["last_name"] = last, first
        for fr in (last, first):
            fr.flags.append("reordered_family_name_first")
        ctx.warn("first_name/last_name swapped: CJK names print the family name first")


def _naming_culture(fields: dict[str, FieldResult], ctx: Context) -> str:
    """Whose naming convention applies: the HOLDER's nationality when it has a
    decisive order (an Ethiopian's UAE residence card still carries an
    Ethiopian given / father / grandfather name), else the issuing country."""
    from .names import FAMILY_FIRST_COUNTRIES, PATRONYMIC_COUNTRIES
    code, _ = resolve_nationality(fields["nationality"].value)
    if code in PATRONYMIC_COUNTRIES | FAMILY_FIRST_COUNTRIES:
        return code
    return ctx.country


def guard_local_script(fields: dict[str, FieldResult], ctx: Context) -> bool:
    """Drops a native-script name in a script the issuing country doesn't print.
    Returns True when the local name was rejected."""
    from .names import local_script_plausible
    local = fields["full_name_local"]
    if not local.value:
        return False
    script = detect_script(local.value)
    if local_script_plausible(script, ctx.country):
        return False
    ctx.warn(f"full_name_local {local.value!r} is {script} script, which {ctx.country} documents don't print — "
             "dropped as an invented transliteration")
    fields["full_name_local"] = FieldResult(alternatives=[local.value] + local.alternatives,
                                            flags=["script_not_printed_by_issuer"])
    return True


def cross_script_check(fields: dict[str, FieldResult], ctx: Context) -> list[dict]:
    """Ge'ez names verified letter by letter against the Latin name (universal/ethiopic.py)."""
    from .ethiopic import combine_readings, correct_name, is_ethiopic, spelling_score
    local, latin = fields["full_name_local"], fields["full_name"]
    if not (local.value and latin.value and is_latin(latin.value)):
        return []
    # Choose among every Ge'ez reading (transcriber, extractor, OCR) the one that
    # best spells the Latin name — a garbled read (wrong word count) loses.
    readings = [local.value] + [a for a in local.alternatives if a and is_ethiopic(a)]
    best = max(readings, key=lambda r: spelling_score(r, latin.value))
    combined = combine_readings([best] + [r for r in readings if r != best], latin.value)
    if spelling_score(combined, latin.value) > spelling_score(best, latin.value):
        best = combined                       # each word from the reading that spells it best
    if best != local.value and spelling_score(best, latin.value) > spelling_score(local.value, latin.value):
        ctx.warn(f"full_name_local: {best!r} spells the Latin name better than {local.value!r}")
        local.alternatives = [local.value] + [a for a in local.alternatives if a != best]
        local.value = best
        local.flags.append("chosen_by_latin_spelling")
    if is_ethiopic(local.value) and spelling_score(local.value, latin.value) < len(latin.value.split()):
        # Under one point per word: this Ge'ez text doesn't spell the holder's
        # name at all (seen: the place of birth 'አዲስ አበባ', a 6-word garble).
        ctx.warn(f"full_name_local {local.value!r} does not spell the Latin name {latin.value!r} — dropped")
        fields["full_name_local"] = FieldResult(alternatives=[local.value] + local.alternatives,
                                                flags=["does_not_spell_latin_name"])
        return [{"dropped": local.value}]
    fixed, changes = correct_name(local.value, latin.value)
    if not changes:
        if fixed == local.value and local.value:
            from .ethiopic import is_ethiopic, _spells
            if is_ethiopic(local.value) and len(local.value.split()) == len(latin.value.split()) and all(
                    _spells(w, "".join(c for c in l.lower() if c.isalpha()))
                    for w, l in zip(local.value.split(), latin.value.split())):
                local.verified_by.append("latin_spelling")      # every syllable agrees
        return []
    local.alternatives = [local.value] + local.alternatives
    local.value = fixed
    local.verified_by.append("latin_spelling")
    local.flags.append("corrected_by_latin_spelling")
    for old, new in changes:
        ctx.warn(f"full_name_local: {old!r} corrected to {new!r} — it must spell the Latin name")
    return [{"from": o, "to": n} for o, n in changes]


def settle_name_parts(fields: dict[str, FieldResult], ctx: Context) -> str:
    """Every word of the given names is its own name: first word -> first_name,
    the rest -> middle_name. The surname is never split. With no surname
    stated, order alone decides only where culture makes it decisive
    (patronymic, family-name-first). Returns the full given names."""
    from .names import UNSPLIT_GIVEN_COUNTRIES, name_words, parts_from_full_name, split_given
    from .names import PATRONYMIC_COUNTRIES
    latin_full = fields["full_name_latin"].value or fields["full_name"].value
    if latin_full and is_latin(latin_full) and any(
            fields[n].value and not is_latin(fields[n].value) for n in ("first_name", "middle_name", "last_name")):
        # An agent gave the name parts in the native script while the Latin name is
        # known: the parts fields are Latin, so rebuild them from the Latin name.
        for n in ("first_name", "middle_name", "last_name"):
            if fields[n].value and not is_latin(fields[n].value):
                fields[n] = FieldResult(alternatives=[fields[n].value], flags=["native_script_moved"])
        ctx.warn("native-script name parts set aside; parts rebuilt from the Latin name")
    first, middle, last = fields["first_name"], fields["middle_name"], fields["last_name"]
    full_words = name_words(fields["full_name_latin"].value or fields["full_name"].value)
    culture = _naming_culture(fields, ctx)
    # The "first word is the own name" guard is for documents ISSUED by a
    # patronymic country (which label only a full name). A foreign document's
    # surname label (a Belgian card for an Ethiopian national) is never overruled.
    if (ctx.country in PATRONYMIC_COUNTRIES and len(full_words) >= 3 and last.value
            and last.value.upper() == full_words[0].upper() and "mrz" not in last.verified_by):
        # In a patronymic name the FIRST word is always the person's own name; a
        # "surname" equal to it was guessed from a card that labels only "Full Name".
        ctx.warn(f"{last.value!r} is the own name in {ctx.country} naming order, not a surname — re-ordered")
        fields["first_name"], fields["middle_name"], fields["last_name"] = FieldResult(), FieldResult(), FieldResult()
        first, middle, last = fields["first_name"], fields["middle_name"], fields["last_name"]
    if not last.value and not first.value:
        full = fields["full_name_latin"].value or fields["full_name"].value
        parts = parts_from_full_name(full, culture)
        if parts.last:
            src = fields["full_name"].source or "full_name"
            fields["first_name"] = FieldResult(value=parts.first, confidence=0.75, source=src, flags=["name_order_convention"])
            fields["middle_name"] = FieldResult(value=parts.middle, confidence=0.75, source=src, flags=["name_order_convention"])
            fields["last_name"] = FieldResult(value=parts.last, confidence=0.75, source=src, flags=["name_order_convention"])
            ctx.warn(f"name parts from {culture} naming order: first {parts.first!r}, middle {parts.middle!r}, "
                     f"last {parts.last!r}")
            return parts.given
    if first.value and ctx.country not in UNSPLIT_GIVEN_COUNTRIES:
        words = name_words(first.value)
        if len(words) > 1:
            head, rest = split_given(first.value, ctx.country)
            # A separately-read middle name is appended only if it isn't another
            # reading of the same given names (no shared words).
            first_words = {w.upper() for w in first.value.split()}
            extra = middle.value if middle.value and not first_words & {w.upper() for w in middle.value.split()} else ""
            first.value = head
            fields["middle_name"] = FieldResult(
                value=" ".join(p for p in (rest, extra) if p), confidence=first.confidence, source=first.source,
                verified_by=list(first.verified_by), flags=["split_from_given_names"])
    given = " ".join(p for p in (fields["first_name"].value, fields["middle_name"].value) if p)
    _canonical_full_name(fields, ctx)
    return given


def _canonical_full_name(fields: dict[str, FieldResult], ctx: Context) -> None:
    """full_name in natural order: first + middle + last (family-name-first
    cultures: last + first + middle), built from the settled parts. Cards that
    print the surname first ('1 SURNAME / 2 GIVEN') otherwise put the last name
    in the middle name's place. The printed order is kept in `original`."""
    from .names import FAMILY_FIRST_COUNTRIES
    first, middle, last = (fields[n].value for n in ("first_name", "middle_name", "last_name"))
    if not (first and last):
        return
    full_now = fields["full_name"].value
    if full_now and is_latin(full_now) != (is_latin(first) and is_latin(last)):
        return                                  # parts and full name in different scripts
    order = (last, first, middle) if ctx.country in FAMILY_FIRST_COUNTRIES else (first, middle, last)
    canonical = " ".join(p for p in order if p)
    full = fields["full_name"]
    fold = lambda v: sorted(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKD", v).upper()).split())
    if full.value and not is_latin(full.value):
        return                                  # a native-script full name keeps its own order
    printed = full.value
    if printed and fold(printed) != fold(canonical):
        from .printed import HONORIFICS
        extra = set(fold(printed)) - set(fold(canonical)) - _NAME_LABEL_WORDS - HONORIFICS
        missing = set(fold(canonical)) - set(fold(printed))
        if extra and not missing:
            return          # printed name holds every part plus more real names — keep it
        # Otherwise the printed read is incomplete (dropped the surname, or picked
        # up a place of birth / honorific) and the settled parts are the truth.
    if printed != canonical:
        full.original = full.original or printed
        full.value = canonical
        if not full.source:
            full.source = "name_parts"
        if "ordered_first_middle_last" not in full.flags:
            full.flags.append("ordered_first_middle_last")
    latin = fields["full_name_latin"]
    if latin.value and fold(latin.value) == fold(canonical):
        latin.value = canonical


def prefer_latin_full_name(fields: dict[str, FieldResult], ctx: Context) -> None:
    """On a bilingual document the Latin-script name is the interoperable one
    (MRZ, travel, banking); the native-script name stays in full_name_local."""
    full, latin = fields["full_name"], fields["full_name_latin"]
    if (full.value and not is_latin(full.value) and latin.value
            and "transliteration_not_printed" not in latin.flags):
        if not fields["full_name_local"].value:
            fields["full_name_local"] = FieldResult(**{**asdict(full)})
        fields["full_name"] = FieldResult(**{**asdict(latin), "flags": list(latin.flags) + ["latin_primary"]})


def script_of_name(fields: dict[str, FieldResult]) -> str:
    return detect_script(fields["full_name_local"].value or fields["full_name"].value)

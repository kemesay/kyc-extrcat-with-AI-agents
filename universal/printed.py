"""
Interpretation of "as printed" transcriptions — the model reads, code decides.

Tested against 31 real documents (US/Canadian licences, EU ID cards and
residence permits, UK/Australian/Ethiopian licences, passports), the agents'
reading was good but their interpretation was not: surname taken as the last
token by position, 01/02/1984 read day-first on a US card, the 6-digit CAN or
a personal number taken as the document number, label text glued onto
values. The field extractor therefore also returns raw transcriptions
(see prompts.AS_PRINTED_FIELDS), and this module interprets them with rules
that don't depend on a model's judgement:

  names   label-anchored surname / given names; honorifics dropped (MR, MRS,
          DR, ATO, W/RO ...); "SURNAME, Given" comma form; an ALL-CAPS token
          run in a mixed-case name is the surname ("DOE Jane Mary").
  dates   parsed with the issuing country's convention (US month-first);
          a printed dual-calendar pair ("12/05/1982 | 20/01/1990") resolved
          by calendar arithmetic, which also settles day/month order.
  numbers per-country document-number templates repair look-alike
          characters (CA123450A -> CA12345OA) and reject non-document
          numbers (CAN, personal numbers, label words, issue suffixes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from . import calendars as cal
from .dates import COUNTRY_CALENDARS, normalize_date
from .normalize import clean_name, strip_identifier_label, strip_markers
from .names import parts_from_labels
from .reconcile import Context, FieldResult

HONORIFICS = {
    "MR", "MRS", "MS", "MISS", "MX", "DR", "SIR", "DAME", "PROF", "MME", "MLLE", "HERR", "FRAU",
    "SR", "SRA", "SRTA", "SIG", "SIGRA", "ATO", "WRO", "W/RO", "WT", "W/T", "SHEIKH", "SHAIKH",
}

# Document-number templates per issuing country: A letter, 9 digit, X either;
# anything else is a literal separator (optional when matching).
DOC_NUMBER_TEMPLATES: dict[str, tuple[tuple[str, frozenset], ...]] = {
    # country: ((template, document types it applies to), ...)
    "NLD": (("AAXXXXXX9", frozenset({"national_id", "passport"})),),
    "ITA": (("AA99999AA", frozenset({"national_id"})),),                          # CIE
    "GBR": (("XXXXX999999XX9AA", frozenset({"drivers_license"})),),               # DVLA (surname part may hold 9s)
    "ETH": (("AA9999999", frozenset({"passport"})),),
    "CAN": (("A9999-99999-99999", frozenset({"drivers_license"})),                 # Ontario licence
            ("999999-999", frozenset({"drivers_license"})),                        # Alberta licence
            ("9999-99999", frozenset({"national_id", "drivers_license"}))),        # Alberta ID card
    "ARE": (("999-9999-9999999-9", frozenset({"national_id", "residence_permit"})),),
    "SWE": (("AA9999999", frozenset({"national_id"})),),
    "BEL": (("9999999 99", frozenset({"residence_permit", "national_id"})),
            ("A 9999999 99", frozenset({"residence_permit"}))),
    "ESP": (("99999999A", frozenset({"national_id"})),),
}
_TO_DIGIT = str.maketrans("OQDILZSBG", "000112586")
_TO_LETTER = str.maketrans("012568", "OIZSGB")


def fit_template(value: str, template: str) -> Optional[str]:
    """Returns value reformatted to the template, repairing look-alike
    characters (O/0, I/1, S/5, B/8, Z/2, G/6), or None if it doesn't fit."""
    core = re.sub(r"[^A-Z0-9]", "", (value or "").upper())
    slots = [c for c in template if c in "A9X"]
    if len(core) != len(slots):
        return None
    out = []
    for ch, slot in zip(core, slots):
        if slot == "9" and not ch.isdigit():
            ch = ch.translate(_TO_DIGIT)
            if not ch.isdigit():
                return None
        elif slot == "A" and not ch.isalpha():
            ch = ch.translate(_TO_LETTER)
            if not ch.isalpha():
                return None
        out.append(ch)
    it = iter(out)
    return "".join(next(it) if c in "A9X" else c for c in template)


def _pieces(candidate: str) -> list[str]:
    """The candidate, then every contiguous run of its tokens — so a trailing
    issue number ('SMITH701150JJ9AA 02') or label word falls away."""
    tokens = (candidate or "").split()
    runs = [" ".join(tokens[i:j]) for i in range(len(tokens)) for j in range(len(tokens), i, -1)]
    return [candidate] + sorted(set(runs) - {candidate}, key=len, reverse=True)


def templates_for(country: str, doc_type: str) -> list[str]:
    """A template only applies to its own document types (an Ethiopian
    passport number format says nothing about an Ethiopian licence);
    with the type unknown, every template of the country is tried."""
    return [t for t, types in DOC_NUMBER_TEMPLATES.get(country, ()) if not doc_type or doc_type in types]


def best_template_fit(candidates: list[str], country: str, doc_type: str = "") -> Optional[str]:
    for template in templates_for(country, doc_type):
        for cand in candidates:
            for piece in _pieces(cand):
                fitted = fit_template(piece, template)
                if fitted:
                    return fitted
    return None


def clean_document_number(value: str) -> str:
    """'4D DLN 000012345678 CLASS C' -> '000012345678'. Letter-only tokens at
    either end are label text; groups of digits in the middle stay together
    (a FAN prints as '1234 5678 9012 3456')."""
    text, _ = strip_markers(value)
    pieces = [p for p in re.split(r"\s*[|;]\s*", text) if len(re.findall(r"\d", p)) >= 4]
    text = pieces[0] if pieces else text
    text = strip_identifier_label(text)
    text = re.sub(r"\s*-\s*", "-", text).upper()
    tokens = text.split()
    has_digit = [bool(re.search(r"\d", t)) for t in tokens]
    if not any(has_digit):
        return "" if len(tokens) <= 2 else text
    first = has_digit.index(True)
    last = len(tokens) - 1 - has_digit[::-1].index(True)
    kept = tokens[first:last + 1]
    # Leading field markers ("4D", "5") and label words ("DLN", "LIC#") before the number.
    while len(kept) > 1 and (not re.search(r"\d", kept[0]) or
                             (len(re.sub(r"\W", "", kept[0])) <= 2 and not kept[0].isdigit())):
        kept = kept[1:]
    return " ".join(kept)


_FIELD_MARKER = re.compile(r"\[[a-z]\]|\([a-z]\)", re.I)     # German '[a]' / '(b)' sub-field markers


def clean_person_name(value: str) -> str:
    text, _ = strip_markers(value)
    text = _FIELD_MARKER.sub(" ", text)
    tokens = [t for t in re.split(r"[\s,]+", text) if t]
    while tokens and tokens[0].upper().rstrip(".") in HONORIFICS:
        tokens = tokens[1:]
    return clean_name(" ".join(tokens))


def caps_surname_split(full: str) -> Optional[tuple[str, str]]:
    """'DOE Jane Mary' / 'ROE, Kim Lee' -> (surname, given):
    typography marks the surname when caps and mixed case are combined."""
    tokens = [t for t in re.split(r"[\s,]+", full or "") if t]
    if len(tokens) < 2:
        return None
    caps = [i for i, t in enumerate(tokens) if t.isupper() and len(re.sub(r"\W", "", t)) >= 2]
    mixed = [i for i, t in enumerate(tokens) if t[:1].isupper() and not t.isupper()]
    if not caps or not mixed or len(caps) + len(mixed) != len(tokens):
        return None
    contiguous = caps == list(range(caps[0], caps[-1] + 1))
    at_edge = caps[0] == 0 or caps[-1] == len(tokens) - 1
    if not (contiguous and at_edge):
        return None
    return " ".join(tokens[i] for i in caps), " ".join(tokens[i] for i in mixed)


def _first(outputs: dict[str, dict], key: str) -> str:
    for agent in ("transcriber", "field_extractor", "layout_analyzer"):
        value = strip_markers((outputs.get(agent) or {}).get(key, ""))[0]
        if value:
            return value
    return ""


def _is_verified(fr: FieldResult) -> bool:
    return bool({"mrz", "barcode"} & set(fr.verified_by)) or any(
        v.startswith(("checksum:", "id_number:")) for v in fr.verified_by)


def _set(fr: FieldResult, value: str, source: str, confidence: float, flag: str) -> None:
    if fr.value and fr.value != value:
        fr.alternatives = [fr.value] + [a for a in fr.alternatives if a != value]
    fr.value, fr.source, fr.confidence = value, source, max(confidence, 0.0)
    fr.verified_by = [v for v in fr.verified_by if v not in ("ocr",)] + [source]
    if flag not in fr.flags:
        fr.flags.append(flag)


# ─────────────────────────────────────────────────────────────
# Names
# ─────────────────────────────────────────────────────────────

_NUMBERED_NAME = re.compile(r"^\s*([12])\s*[.,]?\s*([A-Z][A-Z'\- ]{1,40})\s*$")
_COMMA_SURNAME_LINE = re.compile(r"^\s*([A-Z][A-Z'\-]+(?: [A-Z][A-Z'\-]+)?)\s*,\s*$")
_COMMA_NAME_LINE = re.compile(r"^\s*([A-Z][A-Z'\-]+)\s*,\s*([A-Z][A-Z ,'\-]+)$")
_GIVEN_LINE = re.compile(r"^\s*([A-Z][A-Z ,'\-]+)\s*$")


def _squash(text: str) -> str:
    return re.sub(r"[^A-Z]", "", (text or "").upper())


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _respace(ocr_value: str, agent_texts: list[str]) -> str:
    """OCR glues words ('KIML'); recover the spacing from an agent's reading
    of the same letters ('KIM L') when one exists."""
    target = _squash(ocr_value)
    for text in agent_texts:
        tokens = text.replace(",", " ").split()
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens) + 1):
                if _squash(" ".join(tokens[i:j])) == target:
                    return " ".join(tokens[i:j])
    return ocr_value.replace(",", " ").strip()


def ocr_name_structure(ocr_text: str, agent_texts: list[str]) -> Optional[tuple[str, str, str]]:
    """Physical label evidence in the OCR text: ISO 18013 numbered fields
    glued to their values ('1DOE' / '2JANE QUINN') or the
    'SURNAME,' line of a comma-format name. Accepted only when both parts are
    letters the agents also read. -> (surname, given, source)."""
    haystack = _squash(" ".join(agent_texts))
    lines = [l.strip() for l in (ocr_text or "").upper().splitlines() if l.strip()]
    agent_tokens = {_squash(t) for text in agent_texts for t in text.replace(",", " ").split() if len(_squash(t)) >= 2}

    def supported(value: str, seen: int) -> bool:
        # The agents read the same letters, read them within an edit or two
        # (they misread 'ROWAN' as 'ROWEN'), or both OCR engines saw the line.
        v = _squash(value)
        if len(v) < 2:
            return False
        if v in haystack or seen >= 2:
            return True
        return all(any(_edit_distance(tok, a) <= max(1, len(tok) // 4) for a in agent_tokens)
                   for tok in (_squash(t) for t in value.split()) if len(tok) >= 3)

    numbered: dict[str, list[str]] = {}
    for line in lines:
        m = _NUMBERED_NAME.match(line)
        if m:
            numbered.setdefault(m.group(1), []).append(m.group(2).strip())
    picks = {k: max(set(v), key=v.count) for k, v in numbered.items()}
    if "1" in picks and "2" in picks and all(supported(picks[k], numbered[k].count(picks[k])) for k in ("1", "2")):
        return _respace(picks["1"], agent_texts), _respace(picks["2"], agent_texts), "ocr_numbered_labels"
    if not haystack:
        return None
    for i, line in enumerate(lines):
        m = _COMMA_NAME_LINE.match(line)
        if m:
            surname, given = m.group(1), m.group(2)
        else:
            m = _COMMA_SURNAME_LINE.match(line)
            nxt = _GIVEN_LINE.match(lines[i + 1]) if m and i + 1 < len(lines) else None
            if not (m and nxt):
                continue
            surname, given = m.group(1), nxt.group(1)
        if _squash(surname) in haystack and _squash(given) in haystack and len(_squash(given)) >= 2:
            return _respace(surname, agent_texts), _respace(given, agent_texts), "ocr_comma_format"
    return None


def resolve_local_name(fields: dict[str, FieldResult], outputs: dict[str, dict], ctx: Context) -> Optional[dict]:
    """The name in the document's own script, from the transcriber's
    label-anchored native-script fields. Latin text in these fields is a
    transliteration, not the local name, and is ignored."""
    from .names import compose_local_name
    from .normalize import detect_script, is_latin
    local = lambda key: (lambda v: v if v and not is_latin(v) else "")(" ".join(_first(outputs, key).split()))
    surname, given, whole = local("surname_local_script"), local("given_names_local_script"), local("full_name_local_script")
    if surname and given and (surname == given or surname in given or given in surname):
        whole, surname, given = max(surname, given, key=len), "", ""       # the same name put in both slots
    elif bool(surname) != bool(given):
        whole, surname, given = whole or surname or given, "", ""           # one slot holds the whole name
    script = detect_script(surname or given or whole)
    composed = compose_local_name(surname, given, ctx.country, script) if surname and given else ""
    value = composed or whole
    if not value:
        return None
    fr = fields["full_name_local"]
    squash = lambda v: sorted("".join(v.split())) if script == "CJK" else sorted(v.split())
    if not fr.value:
        fields["full_name_local"] = FieldResult(value=value, confidence=0.75, source="transcriber_local_script",
                                                verified_by=["printed_text"])
    elif squash(fr.value) == squash(value):
        fr.value = value                         # same names: keep the natural order
        fr.verified_by.append("printed_text")
    else:
        # The focused, label-anchored read outranks the name buried in the
        # extractor's 35-field answer; OCR arbitration (grounding) can still
        # switch back if the OCR text supports the extractor's reading better.
        fields["full_name_local"] = FieldResult(value=value, confidence=0.75, source="transcriber_local_script",
                                                verified_by=["printed_text"],
                                                alternatives=[fr.value] + [a for a in fr.alternatives if a != value])
    return {"surname_local": surname, "given_names_local": given, "full_name_local": fields["full_name_local"].value,
            "script": script}


def resolve_names(fields: dict[str, FieldResult], outputs: dict[str, dict], ctx: Context,
                  ocr_text: str = "") -> Optional[dict]:
    surname = clean_person_name(_first(outputs, "surname_as_labelled"))
    given = clean_person_name(_first(outputs, "given_names_as_labelled"))
    raw_surname = _first(outputs, "surname_as_labelled")
    if "," in raw_surname and not given:
        before, after = raw_surname.split(",", 1)
        surname, given = clean_person_name(before), clean_person_name(after)
    source = "printed_labels"

    # The place of birth is often printed right under the name ('TORINO (ITA)'):
    # its words, and parenthesised country codes, are never part of the name.
    birthplace = {w.upper() for w in re.findall(r"[^\W\d_]+", fields["place_of_birth"].value or "")}
    def drop_birthplace(v: str) -> str:
        v = re.sub(r"\([^)]*\)", " ", v)
        kept = [w for w in v.split() if w.upper() not in birthplace] if birthplace else v.split()
        return " ".join(kept) if kept else v.strip()
    surname, given = drop_birthplace(surname), drop_birthplace(given)
    agent_texts = [fields[n].value for n in ("full_name", "full_name_latin", "last_name", "first_name", "middle_name")]
    agent_texts += [_first(outputs, "surname_as_labelled"), _first(outputs, "given_names_as_labelled")]
    ocr = ocr_name_structure(ocr_text, [t for t in agent_texts if t])
    if ocr and (_squash(ocr[0]) != _squash(surname) or _squash(ocr[1]) != _squash(given)):
        surname, given, source = clean_person_name(ocr[0]), clean_person_name(ocr[1]), ocr[2]

    # Typography outranks a model's label guess when both cases are mixed in the name.
    for full in (fields["full_name"].value, fields["full_name_latin"].value,
                 f"{raw_surname} {_first(outputs, 'given_names_as_labelled')}"):
        split = caps_surname_split(full)
        if split and split[0].upper() != surname.upper():
            surname, given, source = clean_person_name(split[0]), clean_person_name(split[1]), "caps_convention"
            break
    if not surname or not given:
        return None

    parts = parts_from_labels(surname, given, ctx.country)
    report = {"surname": surname, "given_names": given, "first_name": parts.first,
              "middle_name": parts.middle, "source": source}
    full = fields["full_name"]
    printed_letters = _squash(ocr_text)
    full_printed = full.value and all(_squash(t) in printed_letters for t in full.value.split() if len(_squash(t)) >= 3)
    if source.startswith("ocr_") and not full_printed and not _is_verified(full):
        _set(full, f"{surname} {given}", source, 0.8, "rebuilt_from_labels")
        report["full_name_rebuilt"] = True
    elif not full.value:
        _set(full, f"{given} {surname}", source, 0.7, "composed_from_labels")
    changed = False
    for name, value in (("last_name", parts.last), ("first_name", parts.first), ("middle_name", parts.middle)):
        if not value and name == "middle_name":
            if fields[name].value and fields[name].value.upper() in given.upper():
                continue   # already part of the given names
            if fields[name].value:
                fields[name].alternatives = [fields[name].value] + fields[name].alternatives
                fields[name] = FieldResult(flags=["not_on_document"])
            continue
        fr = fields[name]
        if fr.value.upper() != value.upper() and not _is_verified(fr):
            _set(fr, value, source, 0.8, "structured_from_labels")
            changed = True
        elif fr.value.upper() == value.upper():
            fr.verified_by.append(source)
            fr.confidence = max(fr.confidence, 0.8)
    if changed:
        ctx.warn(f"name structure taken from the document's labels ({source}): surname {surname!r}, "
                 f"given names {given!r}")
    return report


# ─────────────────────────────────────────────────────────────
# Dates
# ─────────────────────────────────────────────────────────────

def _numeric_readings(raw: str) -> list[date]:
    m = re.fullmatch(r"\s*(\d{1,4})[./\- ](\d{1,2})[./\- ](\d{1,4})\s*", raw)
    if not m:
        return []
    a, b, c = (int(x) for x in m.groups())
    out = []
    for y, mo, d in ((c, b, a), (c, a, b), (a, b, c)):
        if y < 100:
            continue
        try:
            out.append(date(y, mo, d))
        except ValueError:
            pass
    return out


def _resolve_pair(parts: list[str], country: str) -> Optional[tuple[str, str, str]]:
    """Two printed dates for one field: find the reading where one is the
    calendar conversion of the other. -> (gregorian ISO, original, calendar)."""
    calendars = [COUNTRY_CALENDARS[country]] if country in COUNTRY_CALENDARS else ["ethiopian", "persian"]
    for i, orig in enumerate(parts):
        m = re.fullmatch(r"\s*(\d{1,2})[./\- ](\d{1,2})[./\- ](\d{4})\s*", orig)
        if not m:
            continue
        d, mo, y = (int(x) for x in m.groups())
        for calendar in calendars:
            try:
                converted = {"ethiopian": cal.ethiopian_to_gregorian, "persian": cal.persian_to_gregorian,
                             "hijri": cal.hijri_to_gregorian}[calendar](y, mo, d)
            except (KeyError, ValueError):
                continue
            for j, other in enumerate(parts):
                if j != i and converted in _numeric_readings(other):
                    return converted.isoformat(), orig.strip(), calendar
    return None


def ocr_date_readings(ocr_text: str) -> set[str]:
    """Every valid reading (day-first, month-first, year-first) of every
    numeric date in the OCR text."""
    out = set()
    for m in re.finditer(r"(?<!\d)(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})(?!\d)", ocr_text or ""):
        out.update(d.isoformat() for d in _numeric_readings(m.group(1)))
    return out


def _calendar_conversion(printed: str, country: str) -> Optional[str]:
    calendar = COUNTRY_CALENDARS.get(country)
    m = re.fullmatch(r"\s*(\d{1,2})[./\- ](\d{1,2})[./\- ](\d{4})\s*", printed)
    if calendar not in ("ethiopian", "persian") or not m:
        return None
    d, mo, y = (int(x) for x in m.groups())
    try:
        fn = cal.ethiopian_to_gregorian if calendar == "ethiopian" else cal.persian_to_gregorian
        return fn(y, mo, d).isoformat()
    except ValueError:
        return None


_ROMAN = {"XII": "12", "XI": "11", "X": "10", "IX": "9", "VIII": "8", "VII": "7", "VI": "6", "V": "5",
          "IV": "4", "III": "3", "II": "2", "I": "1"}


def _roman_months(text: str) -> str:
    """Roman-numeral month between numeric day and year. 'II' is read as 11,
    not 2: on these documents it is a transcription of a printed '11' (two
    upright strokes), confirmed by the calendar pair it then forms."""
    def repl(m: re.Match) -> str:
        token = m.group(2)
        value = "11" if token == "II" else _ROMAN.get(token, token)
        return f"{m.group(1)}{value}{m.group(3)}"
    return re.sub(r"(\d{1,2}[./\- ])(XII|XI|IX|X|VIII|VII|VI|IV|V|III|II|I)([./\- ]\d{2,4})", repl, text)


_DATE_LABELS = (("issue_date", r"issue|iss\b|delivr|emission|afgifte|ausstell|expedi"),
                ("expiry_date", r"expir|exp\b|valid|scaden|geldig|gültig|venc"),
                ("date_of_birth", r"birth|dob|naissance|nascita|geboorte|geburt|nacimiento"))


def _labelled_dates(other_numbers: str) -> dict[str, str]:
    out = {}
    for piece in re.split(r"\s*\|\s*", other_numbers or ""):
        if ":" not in piece:
            continue
        label, value = piece.split(":", 1)
        if not re.search(r"\d{1,4}[./\- ]\d{1,2}[./\- ]\d{1,4}", value):
            continue
        for field, pattern in _DATE_LABELS:
            if re.search(pattern, label, re.I) and field not in out:
                out[field] = value.strip()
                break
    return out


def _swap(iso: str) -> str:
    try:
        d = date.fromisoformat(iso)
        return date(d.year, d.day, d.month).isoformat()
    except ValueError:
        return ""


def resolve_sex(fields: dict[str, FieldResult], outputs: dict[str, dict]) -> Optional[str]:
    from .normalize import normalize_sex
    sex, _ = normalize_sex(_first(outputs, "sex_as_printed"))
    fr = fields["gender"]
    if sex and not fr.value:
        _set(fr, sex, "printed_text", 0.75, "from_transcription")
    elif sex and fr.value == sex:
        fr.verified_by.append("printed_text")
    return sex or None


def resolve_dates(fields: dict[str, FieldResult], outputs: dict[str, dict], ctx: Context,
                  ocr_text: str = "") -> list[dict]:
    readings = ocr_date_readings(ocr_text)
    orphans: list[str] = []
    reports = []
    labelled = _labelled_dates(_first(outputs, "other_numbers_with_labels"))
    for name, key in (("date_of_birth", "date_of_birth_as_printed"), ("issue_date", "issue_date_as_printed"),
                      ("expiry_date", "expiry_date_as_printed")):
        printed = _first(outputs, key)
        if labelled.get(name) and labelled[name] not in printed:
            # The other calendar's half, transcribed under "other numbers" ("Issue Date: 8/4/2025").
            printed = f"{printed} | {labelled[name]}" if printed else labelled[name]
        if not printed:
            continue
        parts = [_roman_months(p) for p in re.split(r"\s*\|\s*", printed) if p.strip()]
        iso, note, pair_found = "", "", False
        if len(parts) > 1:
            pair = _resolve_pair(parts, ctx.country)
            if pair:
                iso, note = pair[0], f"{pair[2]} pair {pair[1]!r}"
                pair_found = True
                if name == "date_of_birth" and not fields["date_of_birth_orig"].value:
                    fields["date_of_birth_orig"] = FieldResult(value=pair[1], confidence=0.85,
                                                               source="printed_text")
        if not iso:
            # Only one calendar transcribed (e.g. the Ethiopian half); OCR caught the
            # other. The pair holds when the conversion is a reading of an OCR date.
            for part in parts:
                converted = _calendar_conversion(part, ctx.country)
                if converted and converted in readings:
                    iso, note = converted, f"{COUNTRY_CALENDARS[ctx.country]} date {part.strip()!r}, confirmed by OCR"
                    if name == "date_of_birth" and not fields["date_of_birth_orig"].value:
                        fields["date_of_birth_orig"] = FieldResult(value=part.strip(), confidence=0.85,
                                                                   source="printed_text")
                    break
        if not iso and len(parts) > 1:
            # Two unrelated Gregorian dates in one field: the transcriber put a
            # neighbouring field's date here. Keep the first; place the rest later.
            for extra in parts[1:]:
                r = normalize_date(extra, country=ctx.country)
                if r.iso:
                    orphans.append(r.iso)
            parts = parts[:1]
        if not iso:
            for part in parts:
                r = normalize_date(part, country=ctx.country, future_leaning=(name == "expiry_date"))
                if r.iso and r.calendar in ("gregorian", "japanese", "minguo", "thai"):
                    iso, note = r.iso, "ambiguous day/month" if r.ambiguous else ""
                    if r.ambiguous and name != "expiry_date":
                        # A birth/issue date can't be in the future; if only the swap is past, it's the swap.
                        swapped = _swap(iso)
                        if iso > date.today().isoformat() and swapped and swapped <= date.today().isoformat():
                            iso, note = swapped, "day/month order settled: the other reading is in the future"
                    break
        if not iso:
            continue
        fr = fields[name]
        report = {"field": name, "printed": printed, "resolved": iso}
        if fr.value == iso:
            fr.verified_by.append("printed_text")
            fr.confidence = max(fr.confidence, 0.8)
        elif _is_verified(fr):
            report["kept_verified"] = fr.value
        else:
            ctx.warn(f"{name}: printed {printed!r} is {iso} under the document's date convention"
                     + (f" ({note})" if note else "") + (f" — agents had {fr.value}" if fr.value else ""))
            _set(fr, iso, "printed_text", 0.8, "reinterpreted_from_printed")
        if pair_found or (note and "confirmed by OCR" in note):
            fr.verified_by.append("calendar_pair")   # day/month order settled by arithmetic
        reports.append(report)
    _place_orphans(fields, orphans, ctx, reports)
    return reports


def _place_orphans(fields: dict[str, FieldResult], orphans: list[str], ctx: Context, reports: list[dict]) -> None:
    """An orphan date fills an EMPTY issue/expiry slot when plausibility allows
    only that slot: issue is after birth and not in the future; expiry is after issue/birth."""
    today = date.today().isoformat()
    dob = fields["date_of_birth"].value
    for iso in orphans:
        if iso in (dob, fields["issue_date"].value, fields["expiry_date"].value):
            continue
        slot = ""
        if not fields["issue_date"].value and (not dob or iso > dob) and iso <= today:
            slot = "issue_date"
        elif not fields["expiry_date"].value and iso > today and (not dob or iso > dob):
            slot = "expiry_date"
        if slot:
            fields[slot] = FieldResult(value=iso, confidence=0.65, source="printed_text",
                                       flags=["placed_by_plausibility"])
            ctx.warn(f"{slot}: {iso} was transcribed next to another date; placed by plausibility")
            reports.append({"field": slot, "resolved": iso, "placed_by_plausibility": True})


# ─────────────────────────────────────────────────────────────
# Document number
# ─────────────────────────────────────────────────────────────

def resolve_document_number(fields: dict[str, FieldResult], outputs: dict[str, dict], ctx: Context,
                            ocr_text: str = "") -> Optional[dict]:
    fr = fields["document_number"]
    if _is_verified(fr):
        return None
    printed = clean_document_number(_first(outputs, "document_number_as_printed"))
    others = [p.split(":", 1)[-1].strip() for p in re.split(r"\s*\|\s*", _first(outputs, "other_numbers_with_labels"))
              if not re.match(r"\s*CAN\b", p, re.I)]
    current = clean_document_number(fr.value)
    candidates = [c for c in (printed, current, *fr.alternatives, *others) if c]
    doc_type = fields["document_type"].value if fields["document_type"].value not in ("other", "unknown") else ""
    if templates_for(ctx.country, doc_type):
        ocr_tokens = re.findall(r"[A-Z0-9][A-Z0-9\- ]{4,22}[A-Z0-9]", (ocr_text or "").upper())
        fitted = (best_template_fit(candidates, ctx.country, doc_type)
                  or best_template_fit(ocr_tokens, ctx.country, doc_type))
        if fitted:
            if re.sub(r"\W", "", fitted) != re.sub(r"\W", "", fr.value.upper()):
                ctx.warn(f"document_number: {fitted!r} fits the {ctx.country} document-number format "
                         f"(agents had {fr.value!r})")
                _set(fr, fitted, "format_template", 0.85, "document_number_format")
                fr.verified_by.append("format_template")
            else:
                fr.verified_by.append("format_template")
                fr.value = fitted
            return {"value": fitted, "source": "format_template"}
    chosen = printed or current
    if chosen and chosen != fr.value:
        _set(fr, chosen, "printed_text", max(fr.confidence, 0.7), "cleaned_document_number")
    return {"value": chosen, "source": "printed_text"} if chosen else None


# ─────────────────────────────────────────────────────────────
# Issuing country from the printed text
# ─────────────────────────────────────────────────────────────

_US_STATES = ("ALABAMA ALASKA ARIZONA ARKANSAS CALIFORNIA COLORADO CONNECTICUT DELAWARE FLORIDA GEORGIA HAWAII "
              "IDAHO ILLINOIS INDIANA IOWA KANSAS KENTUCKY LOUISIANA MAINE MARYLAND MASSACHUSETTS MICHIGAN "
              "MINNESOTA MISSISSIPPI MISSOURI MONTANA NEBRASKA NEVADA OHIO OKLAHOMA OREGON PENNSYLVANIA TENNESSEE "
              "TEXAS UTAH VERMONT VIRGINIA WASHINGTON WISCONSIN WYOMING").split() + [
              "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA",
              "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA", "WEST VIRGINIA", "DISTRICT OF COLUMBIA"]
_CA_PROVINCES = ["ONTARIO", "ALBERTA", "QUEBEC", "QUÉBEC", "MANITOBA", "SASKATCHEWAN", "NOVA SCOTIA",
                 "NEW BRUNSWICK", "NEWFOUNDLAND", "PRINCE EDWARD ISLAND", "BRITISH COLUMBIA", "YUKON", "NUNAVUT"]


_NATIONALITY_CONTEXT = re.compile(r"NATIONAL|NATIONALIT|CITTADINANZA|STAATSANGEH|NATIONALITEIT|NATIONALITET|"
                                  r"PLACE OF BIRTH|LIEU DE NAISSANCE|GEBURTSORT|BIRTH|الجنسية|ዜግነት", re.I)


def country_from_number(candidates: list[str]) -> str:
    """A checksum-valid number of a distinctive national format (Emirates ID
    '784-…', Chinese 18-digit resident ID, …) names its issuing country."""
    from .id_numbers import check_numbers
    for check in check_numbers("", {f"c{i}": c for i, c in enumerate(candidates) if c}):
        if check.verified:
            return check.country
    return ""


def country_from_text(text: str) -> str:
    """A state / province / country HEADER in the OCR text — used only when
    neither the agents nor a distinctive document number name the country.
    Lines that carry a nationality or birthplace label are never used: a UAE
    residence card says 'Nationality: Ethiopia'."""
    lines = [l for l in (text or "").splitlines() if l.strip() and not _NATIONALITY_CONTEXT.search(l)]
    header = "\n".join(lines[:6])
    upper = f" {header.upper()} "
    if any(re.search(rf"\b{re.escape(p)}\b", upper) for p in _CA_PROVINCES) or "CANADA" in upper:
        return "CAN"
    if any(re.search(rf"\b{re.escape(s)}\b", upper) for s in _US_STATES) and (
            "USA" in upper or "DRIVER" in upper or "IDENTIFICATION" in upper):
        return "USA"
    headers = {"KONINKRIJK DER NEDERLANDEN": "NLD", "NEDERLANDSE IDENTITEITSKAART": "NLD",
               "REPUBBLICA ITALIANA": "ITA", "BUNDESREPUBLIK DEUTSCHLAND": "DEU", "SVERIGE": "SWE",
               "ROYAUME DE BELGIQUE": "BEL", "KONINKRIJK BELGIË": "BEL", "REPUBLIQUE FRANCAISE": "FRA",
               "RÉPUBLIQUE FRANÇAISE": "FRA", "REINO DE ESPAÑA": "ESP", "UNITED ARAB EMIRATES": "ARE",
               "UNITED KINGDOM": "GBR", "FEDERAL DEMOCRATIC REPUBLIC OF ETHIOPIA": "ETH",
               "ETHIOPIAN DRIVING LICENSE": "ETH", "AUSTRALIAN CAPITAL TERRITORY": "AUS"}
    for key, code in headers.items():
        if key in upper:
            return code
    # Address formats: a Canadian postal code after a province code, or a US
    # state code + ZIP(+4) — unmistakable even when the logo is unreadable.
    body = (text or "").upper()
    if re.search(r"\b(AB|BC|MB|NB|NL|NS|NT|NU|ON|PE|QC|SK|YT)\s+[A-Z]\d[A-Z]\s?\d[A-Z]\d\b", body):
        return "CAN"
    if re.search(r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|"
                 r"NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)[\s,]+\d{5}(-\d{4})?\b", body):
        return "USA"
    try:
        from idx.countries import find_issuing_in_text
        hit = find_issuing_in_text(header)
        if hit is not None:
            return hit.code
    except Exception:
        pass
    return ""

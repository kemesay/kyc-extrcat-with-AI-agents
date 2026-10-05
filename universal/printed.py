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


_CALENDAR_MARKER = re.compile(r"E\.?\s?C\.?(?![A-Za-z])|ዓ\.?\s?ም|\bA\.?H\b|هـ|ه\.ش|\bSH\b", re.I)


# ── dual-calendar dates written with month names ────────────────────────
# "የካቲት 07, 1987 / FEB 14, 1995" (Ethiopian / Gregorian), "መጋቢት 10, 2018/Mar 192026"
# (day and year printed glued), "07,1987/FEB 14,1995" (the Ge'ez month lost in
# transcription). One date, two calendars: each half is read on its own and the
# halves must agree by calendar arithmetic. Nothing is guessed: a half counts
# only when it is a complete date, or when exactly one month makes a
# month-less Ethiopian half convert to the complete Gregorian half.

_LATIN_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
                 "oct": 10, "okt": 10, "nov": 11, "dec": 12, "dez": 12}
_LATIN_MONTH_RE = re.compile(r"(?<![A-Za-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|okt|nov|dec|dez)[a-z]*\.?"
                             r"[\s=:,\-]*(\d{1,2})(?:[\s.,]+|(?=(?:19|20)\d{2}))((?:19|20)\d{2})(?!\d)", re.I)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _ethiopic_month(text: str) -> Optional[int]:
    for token in re.findall(r"[\u1200-\u137F]+", text or ""):
        m = cal.ETHIOPIAN_MONTH_LOOKUP.get(token)
        if m:
            return m
    return None


def _dual_half(text: str) -> dict:
    """One half of a dual date -> {calendar, iso, day, year, complete}."""
    m = _LATIN_MONTH_RE.search(text)
    if m:
        try:
            iso = date(int(m.group(3)), _LATIN_MONTHS[m.group(1).lower()[:3]], int(m.group(2))).isoformat()
            return {"calendar": "gregorian", "iso": iso, "complete": True, "text": m.group().strip()}
        except ValueError:
            return {"complete": False}
    month = _ethiopic_month(text)
    years = _YEAR.findall(text)
    days = [int(d) for d in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", _YEAR.sub(" ", text))]
    if not years:
        return {"complete": False}
    year, day = int(years[-1]), (days[-1] if days else None)
    if month and day:
        try:
            iso = cal.ethiopian_to_gregorian(year, month, day).isoformat()
            return {"calendar": "ethiopian", "iso": iso, "complete": True, "day": day, "month": month,
                    "year": year, "text": f"{day:02d}/{month:02d}/{year} EC"}
        except ValueError:
            return {"complete": False}
    return {"complete": False, "day": day, "year": year}


def interpret_dual_date(printed: str, country: str) -> Optional[dict]:
    """-> {iso, verified, original, note} or None when the text is not a
    month-name dual date (the numeric logic below handles those)."""
    if not printed or not (_LATIN_MONTH_RE.search(printed) or _ethiopic_month(printed)):
        return None
    halves = [h for h in re.split(r"\s*[/|]\s*", printed) if _YEAR.search(h) or _LATIN_MONTH_RE.search(h)]
    if len(halves) < 2:
        whole = _dual_half(printed)
        if whole.get("complete"):
            return {"iso": whole["iso"], "verified": False, "original": whole["text"] if whole["calendar"] == "ethiopian"
                    else "", "note": f"{whole['calendar']} date {whole['text']!r}"}
        return None
    parsed = [_dual_half(h) for h in halves[:2]]
    greg = next((p for p in parsed if p.get("calendar") == "gregorian"), None)
    eth = next((p for p in parsed if p.get("calendar") == "ethiopian"), None)
    partial = next((p for p in parsed if not p.get("complete") and p.get("year")), None)
    if greg and eth:
        gap = abs((date.fromisoformat(greg["iso"]) - date.fromisoformat(eth["iso"])).days)
        if gap <= 1:
            return {"iso": greg["iso"], "verified": True, "original": eth["text"],
                    "note": f"Ethiopian {eth['text']} = Gregorian {greg['text']} by calendar arithmetic"}
        return {"iso": "", "verified": False, "original": eth["text"], "conflict": True,
                "note": f"the two calendars disagree: {eth['text']} = {eth['iso']} but {greg['text']}"}
    if greg and partial and partial.get("day") and country in ("ETH", "ERI", ""):
        target = date.fromisoformat(greg["iso"])
        months = [mo for mo in range(1, 14) if _ec_is(partial["year"], mo, partial["day"], target)]
        if len(months) == 1:
            original = f"{partial['day']:02d}/{months[0]:02d}/{partial['year']} EC"
            return {"iso": greg["iso"], "verified": True, "original": original,
                    "note": f"Ethiopian {original} (month read from the arithmetic) = Gregorian {greg['text']}"}
    if greg:
        return {"iso": greg["iso"], "verified": False, "original": "",
                "note": f"Gregorian {greg['text']} (the other calendar's half is not a complete date)"}
    if eth:
        return {"iso": eth["iso"], "verified": False, "original": eth["text"],
                "note": f"Ethiopian {eth['text']} converted by calendar arithmetic"}
    return None


# ── calendar-pair solver for OCR-damaged dual dates ─────────────────────
# Ethiopian cards print each date twice (Ethiopian d/m/y, Gregorian — often
# month-first). OCR and transcription damage them by DROPPING characters
# ('12/3/1991' -> '12/31991', '7/29/2025' -> '7/292025'), rarely adding any. A
# Gregorian date is accepted only when BOTH hold for the same date row:
#   1. an Ethiopian reading on the row converts to it by calendar arithmetic
#      (the reading's digits appear, in order, in that Ethiopian date), and
#   2. a Gregorian reading on the row supports it directly (same year, its
#      digits appear in order in one of the ways the date can be printed),
# and exactly ONE date satisfies both. Otherwise nothing is decided here.

def _subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(c in it for c in small)


def _gregorian_renderings(g: date) -> set[str]:
    out = set()
    for a, b in ((g.month, g.day), (g.day, g.month)):
        for fa in ("{}", "{:02}"):
            for fb in ("{}", "{:02}"):
                out.add(fa.format(a) + fb.format(b) + str(g.year))
    return out


def _ethiopian_candidates(token: str) -> dict[date, str]:
    digits = "".join(re.findall(r"\d", token or ""))
    m = re.search(r"((?:19|20)\d\d)$", digits)
    if not m or len(digits) < 5:
        return {}
    year, out = int(m.group(1)), {}
    for month in range(1, 14):
        for day in range(1, 31):
            forms = (f"{day}{month}{year}", f"{day:02}{month:02}{year}", f"{day}{month:02}{year}",
                     f"{day:02}{month}{year}")
            if any(_subsequence(digits, f) for f in forms):
                try:
                    out.setdefault(cal.ethiopian_to_gregorian(year, month, day), f"{day:02}/{month:02}/{year} EC")
                except ValueError:
                    pass
    return out


def _supports(token: str, g: date) -> bool:
    digits = "".join(re.findall(r"\d", token or ""))
    return len(digits) >= 6 and str(g.year) in digits and any(
        _subsequence(digits, r) for r in _gregorian_renderings(g))


def solve_calendar_pair(tokens: list[str], agent_values: list[str]) -> Optional[tuple[str, str]]:
    """-> (Gregorian ISO, Ethiopian original) when exactly one date is confirmed
    by both calendars on the row, else None."""
    ethiopian: dict[date, str] = {}
    for t in tokens:
        for g, orig in _ethiopian_candidates(t).items():
            ethiopian.setdefault(g, orig)
    agent_dates = set()
    for v in agent_values:
        try:
            agent_dates.add(date.fromisoformat(v))
        except (TypeError, ValueError):
            pass
    confirmed = [g for g in ethiopian if g in agent_dates or any(_supports(t, g) for t in tokens)]
    if len(confirmed) != 1:
        return None
    return confirmed[0].isoformat(), ethiopian[confirmed[0]]


def _ec_is(year: int, month: int, day: int, target: date) -> bool:
    try:
        return cal.ethiopian_to_gregorian(year, month, day) == target
    except ValueError:
        return False


def _date_key(text: str) -> tuple:
    """Identity of a printed date regardless of zero-padding and separators."""
    return tuple(int(n) for n in re.findall(r"\d+", text or ""))


def _readings(printed: str, country: str) -> set[str]:
    """Every ISO date the printed text can mean (both day/month orders, and
    the calendar conversion on documents printed in another calendar)."""
    # The digits read naively too: a model that ignores "E.C" turns 25/10/2016 E.C into 2016-10-25.
    core = _CALENDAR_MARKER.sub("", printed).strip()
    out = {d.isoformat() for d in _numeric_readings(_roman_months(core))}
    converted = _calendar_conversion(core, country)
    if converted:
        out.add(converted)
    r = normalize_date(printed, country=country)
    if r.iso:
        out.add(r.iso)
    return out


def _label_evidence(outputs: dict[str, dict], anchored: list) -> list:
    """Printed-label evidence for dates: the layout's anchored dates plus the
    transcriber's own 'LABEL: date' lists, whose labels the lexicon maps."""
    from .layout import transcribed_labelled_dates
    out = list(anchored or [])
    for key in ("dates_with_labels", "other_numbers_with_labels"):
        out += transcribed_labelled_dates(_first(outputs, key))
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
                  ocr_text: str = "", anchored: Optional[list] = None,
                  row_tokens: Optional[dict[str, list[str]]] = None) -> list[dict]:
    """The model reads the dates; the printed LABELS decide which field each
    one is. A date the transcriber filed under one field but the card labels
    as another ("Date Issued" filed as date of birth) moves to its labelled
    field; the existing calendar / day-month logic then settles its value."""
    readings = ocr_date_readings(ocr_text)
    orphans: list[str] = []
    reports = []
    evidence = _label_evidence(outputs, anchored)
    labelled: dict[str, list[str]] = {}
    strong: dict[tuple, set] = {}
    for a in evidence:
        key = _date_key(a.value)
        if not key:
            continue
        if a.value not in labelled.setdefault(a.field, []):
            labelled[a.field].append(a.value)
        # Only a label in the same box / on the same row, or the one the transcriber
        # copied, may overrule the model; a label above a value only fills a gap.
        if a.how in ("same_box", "same_row", "left_stack", "transcribed_label"):
            strong.setdefault(key, set()).add(a.field)
    # A date two strong sources label differently is ambiguous: nothing is moved.
    label_of = {k: next(iter(f)) for k, f in strong.items() if len(f) == 1}
    moved: dict[str, list[str]] = {}
    for name, key in (("date_of_birth", "date_of_birth_as_printed"), ("issue_date", "issue_date_as_printed"),
                      ("expiry_date", "expiry_date_as_printed")):
        for part in [p for p in re.split(r"\s*\|\s*", _first(outputs, key)) if p.strip()]:
            other = label_of.get(_date_key(part))
            if other and other != name:
                moved.setdefault(name, []).append(part)
                reports.append({"field": name, "printed": part, "labelled_as": other, "moved_by_label": True})
                ctx.warn(f"{name}: {part!r} is printed beside a {other.replace('_', ' ')} label — "
                         f"not taken as the {name.replace('_', ' ')}")
    # Dates the model placed (in any field): a label only fills a gap with a date
    # nobody has placed, so one printed date never ends up in two fields.
    placed = {_date_key(p) for k in ("date_of_birth_as_printed", "issue_date_as_printed", "expiry_date_as_printed")
              for p in re.split(r"\s*\|\s*", _first(outputs, k)) if p.strip()}
    solved: set[str] = set()
    if ctx.country in ("ETH", "ERI") and row_tokens:
        for name, key in (("date_of_birth", "date_of_birth_as_printed"), ("issue_date", "issue_date_as_printed"),
                          ("expiry_date", "expiry_date_as_printed")):
            tokens = list(row_tokens.get(name, [])) + [p for p in re.split(r"\s*\|\s*", _first(outputs, key)) if p]
            agent_values = [o.get(name, "") for o in outputs.values()]
            got = solve_calendar_pair(tokens, agent_values)
            fr = fields[name]
            if not got or _is_verified(fr):
                continue
            iso, orig = got
            if fr.value != iso:
                ctx.warn(f"{name}: {iso} — the Ethiopian {orig} and the Gregorian date printed on the same row "
                         f"agree by calendar arithmetic" + (f" (agents had {fr.value})" if fr.value else ""))
                fields[name] = FieldResult(value=iso, confidence=0.92, source="calendar_pair",
                                           verified_by=["calendar_pair", "printed_text"],
                                           alternatives=[v for v in [fr.value] + fr.alternatives if v and v != iso],
                                           flags=["solved_from_calendar_pair"])
            else:
                fr.verified_by += ["calendar_pair", "printed_text"]
                fr.confidence = max(fr.confidence, 0.92)
            if name == "date_of_birth":
                fields["date_of_birth_orig"] = FieldResult(value=orig, confidence=0.9, source="calendar_pair",
                                                           verified_by=["calendar_pair"])
            reports.append({"field": name, "resolved": iso, "ethiopian": orig, "solved_from_calendar_pair": True})
            solved.add(name)
    arrived: dict[str, list[str]] = {}
    for name, parts in moved.items():
        for part in parts:
            arrived.setdefault(label_of[_date_key(part)], []).append(part)
    for name, key in (("date_of_birth", "date_of_birth_as_printed"), ("issue_date", "issue_date_as_printed"),
                      ("expiry_date", "expiry_date_as_printed")):
        # The model's reading stays the value (OCR digits are the weaker read);
        # a part it filed under the wrong field arrives here with its reading.
        if name in solved:
            continue
        model_parts = [p for p in re.split(r"\s*\|\s*", _first(outputs, key))
                       if p.strip() and p not in moved.get(name, [])] + arrived.get(name, [])
        model_parts = list({_date_key(p) or p: p for p in model_parts}.values())
        label_parts = [] if model_parts else [v for v in labelled.get(name, [])
                                              if _date_key(v) not in placed][:2]
        printed = " | ".join(model_parts + label_parts)
        if not printed:
            continue
        parts = [_roman_months(p) for p in re.split(r"\s*\|\s*", printed) if p.strip()]
        iso, note, pair_found = "", "", False
        dual = interpret_dual_date(printed, ctx.country)
        if dual and dual.get("conflict"):
            ctx.review(f"{name}: {dual['note']}")
            reports.append({"field": name, "printed": printed, "conflict": dual["note"]})
            continue
        if dual:
            iso, note, pair_found = dual["iso"], dual["note"], dual["verified"]
            if name == "date_of_birth" and dual["original"]:
                fields["date_of_birth_orig"] = FieldResult(value=dual["original"], confidence=0.9,
                                                           source="printed_text",
                                                           verified_by=["calendar_pair"] if pair_found else [])
        if not iso and len(parts) > 1:
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
                # A calendar the card NAMES ("25/10/2016 E.C", "ዓ.ም", "AH") is converted by
                # arithmetic; an unmarked date is only read in the Gregorian family here.
                marked = r.calendar in ("ethiopian", "hijri", "persian") and _CALENDAR_MARKER.search(part)
                if marked and r.iso:
                    iso, note = r.iso, f"{r.calendar} calendar date {part.strip()!r} (marked on the card)"
                    if name == "date_of_birth" and not fields["date_of_birth_orig"].value:
                        fields["date_of_birth_orig"] = FieldResult(value=part.strip(), confidence=0.9,
                                                                   source="printed_text")
                    # '30/01/2018 EC | 30/01/2021 EC' in one field: the other marked dates are other fields'.
                    for extra in parts[parts.index(part) + 1:]:
                        rx = normalize_date(extra, country=ctx.country)
                        if rx.iso and rx.iso != iso and _CALENDAR_MARKER.search(extra):
                            orphans.append(rx.iso)
                    break
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
        if labelled.get(name):
            report["labelled"] = labelled[name]
        if fr.value == iso:
            fr.verified_by.append("printed_text")
            fr.confidence = max(fr.confidence, 0.8)
        elif _is_verified(fr):
            report["kept_verified"] = fr.value
        else:
            ctx.warn(f"{name}: printed {printed!r} is {iso} under the document's date convention"
                     + (f" ({note})" if note else "") + (f" — agents had {fr.value}" if fr.value else ""))
            _set(fr, iso, "printed_text", 0.8, "reinterpreted_from_printed")
        if pair_found or (note and ("confirmed by OCR" in note or "marked on the card" in note)):
            fr.verified_by.append("calendar_pair")   # day/month order settled by arithmetic
        if labelled.get(name) and any(iso in _readings(v, ctx.country) for v in labelled[name]):
            fr.verified_by.append("printed_label")
        reports.append(report)
    _clear_mislabelled(fields, labelled, ctx, reports, label_of)
    _place_orphans(fields, orphans, ctx, reports, readings)
    _flag_unstated_calendar(fields, outputs, ctx, ocr_text, reports, solved, labelled)
    return reports


def _gregorian_is(m: re.Match, iso: str) -> bool:
    d, mo, y = (int(x) for x in m.groups())
    return iso in {f"{y:04d}-{mo:02d}-{d:02d}", f"{y:04d}-{d:02d}-{mo:02d}"}


def _flag_unstated_calendar(fields: dict[str, FieldResult], outputs: dict[str, dict], ctx: Context,
                            ocr_text: str, reports: list[dict], solved: set,
                            labelled: Optional[dict] = None) -> None:
    """A date printed with NO calendar mark on a card whose other dates are marked
    Ethiopian ('30/01/2018 EC') is ambiguous: '02/07/1988' is 1988-07-02 if
    Gregorian but 1996-03-11 if Ethiopian. Neither is chosen — both go to review."""
    if ctx.country not in ("ETH", "ERI"):
        return
    printed = {k: _first(outputs, f"{k}_as_printed") for k in ("date_of_birth", "issue_date", "expiry_date")}
    all_text = " ".join([ocr_text or "", *printed.values(), _first(outputs, "dates_with_labels")])
    if not re.search(r"\d{4}\s*(?:E\.?\s?C\.?(?![A-Za-z])|ዓ\.?\s?ም)", all_text, re.I):
        return                                     # no date on this card is marked Ethiopian
    fr = fields["date_of_birth"]
    # The printed reading the date came from: the transcriber's, else the labelled layout's.
    text = printed["date_of_birth"] or " | ".join((labelled or {}).get("date_of_birth", []))
    if (not fr.value or "date_of_birth" in solved or _is_verified(fr) or "calendar_pair" in fr.verified_by
            or not text or _ethiopic_month(text) or _LATIN_MONTH_RE.search(text)):
        return
    # The birth date's OWN reading: the transcriber may put the card's other (marked) dates in
    # the same field ('02/07/1988 | 30/01/2018 EC | ...'), so the mark is checked on this date only.
    m = next((x for x in re.finditer(r"(\d{1,2})[./\- ](\d{1,2})[./\- ](\d{4})", text)
              if date(int(x.group(3)), 1, 1).isoformat()[:4] == fr.value[:4]
              or _gregorian_is(x, fr.value)), None)
    if not m or _CALENDAR_MARKER.match(text[m.end():].lstrip()):
        return
    d, mo, y = (int(x) for x in m.groups())
    try:
        as_ethiopian = cal.ethiopian_to_gregorian(y, mo, d).isoformat()
    except ValueError:
        return
    gregorian = fr.value
    ctx.review(f"date_of_birth: {text.strip()!r} carries no calendar mark while the card's other dates are "
               f"Ethiopian — {fr.value} if Gregorian, {as_ethiopian} if Ethiopian")
    fields["date_of_birth_orig"] = FieldResult(value=text.strip(), confidence=0.8, source="printed_text")
    fields["date_of_birth"] = FieldResult(value="", confidence=0.0, source="printed_text",
                                          alternatives=[fr.value, as_ethiopian] + fr.alternatives,
                                          flags=["calendar_not_printed"])
    reports.append({"field": "date_of_birth", "printed": text, "calendar": "not printed",
                    "if_gregorian": gregorian,
                    "if_ethiopian": as_ethiopian})


def _clear_mislabelled(fields: dict[str, FieldResult], labelled: dict[str, list[str]], ctx: Context,
                       reports: list[dict], label_of: Optional[dict] = None) -> None:
    """An agents' date that the card labels as a DIFFERENT field, with no label
    of its own supporting it, is not this field (the issue date offered as a
    birth date on a work ID that prints no birth date)."""
    for name in ("date_of_birth", "issue_date", "expiry_date"):
        fr = fields[name]
        if not fr.value or _is_verified(fr) or "printed_label" in fr.verified_by:
            continue
        own = any(fr.value in _readings(v, ctx.country) for v in labelled.get(name, []))
        strong_values = [v for f, values in labelled.items() for v in values
                         if f != name and (label_of or {}).get(_date_key(v)) == f]
        other = next(((label_of or {})[_date_key(v)] for v in strong_values
                      if fr.value in _readings(v, ctx.country)), None)
        if own or not other:
            continue
        ctx.warn(f"{name}: {fr.value} is the date printed beside the {other.replace('_', ' ')} label — "
                 f"no {name.replace('_', ' ')} is printed, so it is left empty")
        fields[name] = FieldResult(value="", confidence=0.0, source="printed_labels",
                                   alternatives=[fr.value] + fr.alternatives, flags=[f"labelled_as_{other}"])
        reports.append({"field": name, "cleared": True, "labelled_as": other})


def _place_orphans(fields: dict[str, FieldResult], orphans: list[str], ctx: Context, reports: list[dict],
                   readings: Optional[set] = None) -> None:
    """An orphan date (printed, transcribed next to another date) fills an issue/
    expiry slot when plausibility allows only that slot: issue is after birth
    and not in the future; expiry is after issue/birth. A slot counts as open
    when empty, or when it holds an unverified agents' date no OCR reading has."""
    today = date.today().isoformat()
    dob = fields["date_of_birth"].value

    def open_slot(name: str) -> bool:
        fr = fields[name]
        return not fr.value or (not _is_verified(fr) and not {"printed_text", "calendar_pair", "printed_label"}
                                & set(fr.verified_by) and readings is not None and fr.value not in readings)

    for iso in orphans:
        if iso in (dob, fields["issue_date"].value, fields["expiry_date"].value):
            continue
        slot = ""
        issue = fields["issue_date"].value
        if open_slot("issue_date") and (not dob or iso > dob) and iso <= today:
            slot = "issue_date"
        elif open_slot("expiry_date") and iso > today and (not dob or iso > dob) and \
                (not issue or iso > issue or open_slot("issue_date")):
            slot = "expiry_date"
        if slot and fields[slot].value:
            ctx.warn(f"{slot}: {fields[slot].value} (agents) is not printed; the printed {iso} replaces it")
        if slot:
            fields[slot] = FieldResult(value=iso, confidence=0.65, source="printed_text",
                                       flags=["placed_by_plausibility"],
                                       alternatives=[v for v in [fields[slot].value] if v])
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

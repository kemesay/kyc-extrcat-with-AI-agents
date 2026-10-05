"""
Understanding what the card says about itself — labels, document type,
languages — and applying it with the rule "the model reads, the code decides".

Called by the pipeline after the agents' consensus:

  read()                  the card's layout (RapidOCR boxes, read in the
                          background while the agents run, plus Tesseract
                          boxes for label scripts RapidOCR can't read) ->
                          every date and value anchored to its printed label
  decide_document_type()  what the card calls itself ("Employee ID Card",
                          "የሰራተኛ መታወቂያ", "Kaardii Eenyummaa Hojjetaa") plus
                          work-card fields (position, department, employee
                          number) beat an agent guess that has no printed support
  decide_languages()      printed evidence + every agent's vote (the fast
                          scanner's included) -> language_primary / _secondary
  apply_values()          labelled values fill EMPTY schema fields; work-card
                          fields go to additional_fields; on an employee ID the
                          labelled employee number is the document number
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from . import labels, languages, layout as layout_mod
from .normalize import normalize_sex, resolve_nationality
from .reconcile import Context, FieldResult

# Tesseract codes of scripts RapidOCR can't read; Latin labels come from RapidOCR.
_NON_LATIN_TESS = {"amh", "tir", "ara", "fas", "urd", "heb", "chi_sim", "chi_tra", "jpn", "kor", "rus", "ukr",
                   "bul", "srp", "mkd", "ell", "hin", "ben", "tha", "kat", "hye", "khm", "lao", "mya", "sin",
                   "tam", "tel", "kan", "mal", "guj", "pan", "nep", "mar", "bel", "kaz", "kir", "mon", "tgk", "div"}

# Documents that identify a person to a state: a birth date is expected.
GOVERNMENT_IDENTITY_TYPES = {"passport", "national_id", "drivers_license", "residence_permit", "visa",
                             "refugee_travel_document", "military_id", "voter_id", "work_permit"}
_WORK_FIELDS = ("employee_number", "position", "department", "branch")
# Card types the agents' vocabulary rarely offers: a printed title for one of
# these corrects the agents. A generic state-ID title only fills an empty type
# (it never overrides the agents' passport / licence / residence-permit call).
SPECIFIC_TYPES = {"employee_id", "student_id", "voter_id", "work_permit", "health_insurance_card", "tax_id"}


@dataclass
class Understanding:
    layout: layout_mod.Layout = field(default_factory=layout_mod.Layout)
    dates: list = field(default_factory=list)          # layout.Anchored
    values: list = field(default_factory=list)         # layout.Anchored
    tesseract_langs: str = ""
    date_rows: dict = field(default_factory=dict)      # field -> date-like tokens on that label's row

    def report(self) -> dict:
        return {"labelled_dates": [a.as_dict() for a in self.dates],
                "labelled_values": [a.as_dict() for a in self.values],
                "layout_lines": len(self.layout.lines), "label_ocr_languages": self.tesseract_langs}


def label_ocr_languages(country: str, script: str) -> str:
    from .grounding import _COUNTRY_LANGS, _SCRIPT_LANGS, _tesseract_languages
    installed = _tesseract_languages()
    wanted = _COUNTRY_LANGS.get(country, []) + _SCRIPT_LANGS.get(script, [])[:2]
    return "+".join(dict.fromkeys(l for l in wanted if l in _NON_LATIN_TESS and l in installed))


def read(image_paths: list[str], rapid_lines: Optional[list], country: str, script: str) -> Understanding:
    langs = label_ocr_languages(country, script)
    lines = list(rapid_lines) if rapid_lines is not None else layout_mod.rapid_lines(image_paths)
    if langs:
        lines += layout_mod.tesseract_lines(image_paths, langs)
    lay = layout_mod.Layout(lines)
    return Understanding(lay, layout_mod.anchored_dates(lay), layout_mod.anchored_values(lay), langs,
                         layout_mod.date_rows(lay))


# ── document type ───────────────────────────────────────────────────────

def decide_document_type(fields: dict[str, FieldResult], texts: list[str], und: Understanding,
                         ctx: Context) -> dict:
    score: Counter = Counter()
    titles = []
    for text in texts:
        for doc_type, lang, phrase in labels.document_type_titles(text):
            if (doc_type, phrase) not in {(t[0], t[2]) for t in titles}:
                titles.append((doc_type, lang, phrase))
                score[doc_type] += 3
    work = sorted({a.field for a in und.values if a.field in _WORK_FIELDS})
    score["employee_id"] += 2 * len(work)
    report: dict[str, Any] = {"printed_titles": [{"type": t, "language": labels.LANGUAGE_NAMES.get(l, l),
                                                 "title": p} for t, l, p in titles],
                              "work_card_fields": work, "scores": dict(score)}
    fr = fields["document_type"]
    current = fr.value
    if not score:
        return report
    best, best_score = score.most_common(1)[0]
    # The agents' type may be corrected when it is empty, when the title names a
    # specific card type, or when nothing printed supports the agents' type at all.
    may_override = current in ("", "other", "unknown") or best in SPECIFIC_TYPES or score.get(current, 0) == 0
    if best_score >= 3 and best != current and score.get(current, 0) < best_score and may_override \
            and "mrz" not in fr.verified_by:
        evidence = ", ".join([f"'{p}'" for t, _, p in titles if t == best][:3]
                             + ([f"{len(work)} work-card field(s)"] if best == "employee_id" and work else []))
        ctx.warn(f"document_type: the card prints {evidence} — {best}"
                 + (f" (agents had {current})" if current else ""))
        fields["document_type"] = FieldResult(value=best, confidence=0.9, source="printed_title",
                                              verified_by=["printed_title"],
                                              alternatives=[v for v in [current] if v],
                                              flags=["document_type_from_printed_title"])
        report["decided"] = best
    elif current and score.get(current, 0) >= 3:
        fr.verified_by.append("printed_title")
    return report


# ── languages ───────────────────────────────────────────────────────────

# ── issuing country from printed evidence ───────────────────────────────

_LABEL_LANGUAGE_COUNTRY = {"Oromo": "ETH", "Amharic": "ETH"}


def printed_countries(text: str) -> set[str]:
    """Countries NAMED on the card, in any language: 'Ethiopia', 'Itiyoophiyaa',
    'ኢትዮጵያዊ', 'Egypt', 'مصر', 'REPUBLIC OF KENYA' ..."""
    from .normalize import NATIVE_COUNTRY_NAMES, resolve_country
    found = set()
    norm_words = labels.normalize(text).split()
    for key, code in NATIVE_COUNTRY_NAMES.items():
        if labels.normalize(key) in norm_words or (not key.isascii() and key in text):
            found.add(code)
    for n in (1, 2, 3):
        for i in range(len(norm_words) - n + 1):
            phrase = " ".join(norm_words[i:i + n])
            if len(phrase) < 4:
                continue
            code, name = resolve_country(phrase)
            if code and name and labels.normalize(name) == phrase:
                found.add(code)
    return found


def decide_country(fields: dict[str, FieldResult], ctx: Context, text: str) -> dict:
    """The agents' issuing country stands unless the card contradicts it: the card
    names no such country, yet names exactly one other country — or its label
    languages belong to exactly one country (Oromo/Amharic labels: Ethiopia)."""
    fr = fields["issuing_country"]
    current = ctx.country
    if {"mrz", "barcode"} & set(fr.verified_by):
        return {}
    named = printed_countries(text)
    by_language = {_LABEL_LANGUAGE_COUNTRY[d["language"]] for d in languages.detect(text, "")
                   if d["language"] in _LABEL_LANGUAGE_COUNTRY and any(e.startswith("label") for e in d["evidence"])}
    evidence = named or by_language
    report = {"printed_countries": sorted(named), "label_language_countries": sorted(by_language)}
    if len(evidence) == 1 and current not in named:
        new = next(iter(evidence))
        if new != current:
            why = "the card names it" if named else "its labels are in that country's languages"
            ctx.warn(f"issuing_country: {new} — {why}" + (f"; the card never names {current} (agents' reading)"
                                                           if current else ""))
            fields["issuing_country"] = FieldResult(value=new, confidence=0.85, source="printed_text",
                                                    verified_by=["printed_text"],
                                                    alternatives=[v for v in [current] if v])
            ctx.country = new
            report["decided"] = new
    return report


def decide_languages(fields: dict[str, FieldResult], outputs: dict[str, dict], texts: list[str],
                     country: str) -> list[dict]:
    detected = languages.detect("\n".join(t for t in texts if t), country)
    votes = []
    for out in outputs.values():
        for key in ("language_primary", "language_secondary"):
            if out.get(key):
                votes.append(out[key])
    primary, secondary, langs = languages.decide(detected, votes)
    if not primary:
        return langs
    evidenced = bool(detected)
    for name, value in (("language_primary", primary), ("language_secondary", secondary)):
        fr = fields[name]
        if not value or fr.value == value:
            if value:
                fr.verified_by.append("printed_evidence")
            continue
        fields[name] = FieldResult(value=value, confidence=0.9 if evidenced else fr.confidence or 0.6,
                                   source="printed_evidence" if evidenced else "agent_votes",
                                   verified_by=["printed_evidence"] if evidenced else [],
                                   alternatives=[v for v in [fr.value] if v])
    return langs


# ── values ──────────────────────────────────────────────────────────────

def apply_values(fields: dict[str, FieldResult], und: Understanding, ctx: Context) -> dict[str, str]:
    """-> additional_fields (work-card and other labelled values)."""
    extra: dict[str, str] = {}
    for a in und.values:
        extra.setdefault(a.field, a.value)
    fills = {"nationality": lambda v: resolve_nationality(v)[1] and v,
             "place_of_birth": lambda v: v, "issuing_authority": lambda v: v,
             "blood_type": lambda v: v, "gender": None}
    for name, check in fills.items():
        if name == "gender":
            sex = normalize_sex(extra.get("sex", ""))[0]
            if sex and not fields["gender"].value:
                fields["gender"] = FieldResult(value=sex, confidence=0.75, source="printed_labels",
                                               verified_by=["printed_labels"])
            elif sex and fields["gender"].value == sex:
                fields["gender"].verified_by.append("printed_labels")
            continue
        value = extra.get(name, "")
        if value and check(value) and not fields[name].value:
            fields[name] = FieldResult(value=value, confidence=0.7, source="printed_labels",
                                       verified_by=["printed_labels"], flags=["from_labelled_layout"])
    # Address levels, as the card labels them (Ethiopia: Region > Sub-city/Zone > Woreda > Kebele).
    for name, level in (("region", "region"), ("district", "sub_city")):
        value = extra.get(level, "")
        fr = fields[name]
        if value and fr.value.strip().lower() != value.strip().lower() and "mrz" not in fr.verified_by:
            if fr.value:
                ctx.warn(f"{name}: the card prints {value!r} beside its {level.replace('_', '-')} label "
                         f"(agents had {fr.value!r})")
            fields[name] = FieldResult(value=value, confidence=0.85, source="printed_labels",
                                       verified_by=["printed_labels"], alternatives=[v for v in [fr.value] if v])
        elif value:
            fr.verified_by.append("printed_labels")
    lower = ", ".join(f"{label} {extra[k]}" for k, label in (("woreda", "Woreda"), ("kebele", "Kebele"))
                      if extra.get(k))
    if lower and fields["sub_district"].value.lower() != lower.lower():
        fields["sub_district"] = FieldResult(value=lower, confidence=0.8, source="printed_labels",
                                             verified_by=["printed_labels"],
                                             alternatives=[v for v in [fields["sub_district"].value] if v])
    orgs = organization_names(und.layout)
    if orgs:
        extra["organization"] = " / ".join(orgs)
        if fields["document_type"].value in ("employee_id", "student_id") and not fields["issuing_authority"].value:
            # A work / student card is issued by the employer / school it names.
            fields["issuing_authority"] = FieldResult(value=orgs[0], confidence=0.75, source="printed_header",
                                                      verified_by=["printed_header"])
    # Two printed numbers glued into one by the agents ("ID/KI/W04/02848628 AA0003239441"):
    # the value beside the card's own "ID No" label is the document number.
    labelled_number = extra.get("document_number", "")
    current = fields["document_number"].value
    if labelled_number and current and current != labelled_number and "mrz" not in fields["document_number"].verified_by:
        squash = lambda v: re.sub(r"[\s\-]", "", v).upper()
        if squash(labelled_number) in squash(current) and len(squash(current)) > len(squash(labelled_number)):
            ctx.warn(f"document_number: {current!r} joins several printed numbers — the card labels "
                     f"{labelled_number!r} as the ID number")
            fields["document_number"] = FieldResult(value=labelled_number, confidence=0.85, source="printed_labels",
                                                    verified_by=["printed_labels"], alternatives=[current])
    # On a work card the labelled employee number is the card number; "ID NO" on it is the same number.
    # The name printed beside its label ('Maqaa Guutuu | ሙሉ ስም' -> 'Kaasahuun Girmaa Wulatuu | ካሳሁን ...').
    named = [a.value for a in und.values if a.field == "full_name"]
    if named:
        from . import ethiopic
        parts = [p.strip() for v in named for p in v.split("|") if p.strip()]
        latin = next((p for p in parts if len(re.findall(r"[A-Za-z]{2,}", p)) >= 2
                      and not labels.looks_like_title(p)), "")
        local = next((p for p in parts if ethiopic.is_ethiopic(p) and len(p.split()) >= 2), "")
        fold = lambda v: " ".join(labels.normalize(v).split())
        if latin and fold(latin) != fold(fields["full_name"].value) and "mrz" not in fields["full_name"].verified_by:
            ctx.warn(f"full_name: the card prints {latin!r} beside its name label"
                     + (f" (agents had {fields['full_name'].value!r})" if fields["full_name"].value else ""))
            for n in ("full_name", "full_name_latin"):
                fields[n] = FieldResult(value=latin, confidence=0.85, source="printed_labels",
                                        verified_by=["printed_labels"],
                                        alternatives=[v for v in [fields[n].value] if v])
            for n in ("first_name", "middle_name", "last_name"):      # rebuilt from the printed name
                fields[n] = FieldResult(alternatives=[v for v in [fields[n].value] if v])
        if local and len(local.split()) == len(latin.split()) and not fields["full_name_local"].value:
            fields["full_name_local"] = FieldResult(value=local, confidence=0.7, source="printed_labels")
    # A card's title is never a person's name ('Waraqaa Eenyummaa Jiraattota' = 'Residents' ID card').
    drop_titles_as_names(fields, ctx)
    number = extra.get("employee_number", "") or extra.get("document_number", "")
    return _finish_values(fields, extra, number, ctx)


def drop_titles_as_names(fields: dict[str, FieldResult], ctx: Context) -> None:
    """A card's title is never a person's name — checked again after OCR
    arbitration, which can promote a title the agents misspelled."""
    for n in ("full_name", "full_name_latin"):
        if fields[n].value and labels.looks_like_title(fields[n].value) and "printed_labels" not in fields[n].verified_by:
            ctx.warn(f"{n}: {fields[n].value!r} is the card's title, not a name — dropped")
            ctx.review(f"{n}: the name could not be read (the agents offered the card's title)")
            fields[n] = FieldResult(alternatives=[fields[n].value], flags=["title_not_a_name"])
            for part in ("first_name", "middle_name", "last_name"):
                fields[part] = FieldResult(alternatives=[v for v in [fields[part].value] if v])


def _finish_values(fields: dict[str, FieldResult], extra: dict, number: str, ctx: Context) -> dict:
    if fields["document_type"].value == "employee_id" and number:
        fr = fields["document_number"]
        clean = "".join(number.split())
        if fr.value.replace(" ", "") != clean and "mrz" not in fr.verified_by:
            if fr.value:
                ctx.warn(f"document_number: the card labels {number!r} as the employee number (agents had {fr.value!r})")
            fields["document_number"] = FieldResult(value=number, confidence=0.85, source="printed_labels",
                                                    verified_by=["printed_labels"],
                                                    alternatives=[v for v in [fr.value] if v])
        else:
            fr.verified_by.append("printed_labels")
    # Only what the schema has no field for: registration / employee number, position, ...
    from .prompts import EXTRACTION_FIELDS
    return {k: v for k, v in extra.items() if k not in EXTRACTION_FIELDS and k != "sex"}


_ORG_WORDS = ("bank", "baankii", "ባንክ", "plc", "s c", "share company", "ltd", "limited", "corporation",
              "company", "university", "yuunivarsiitii", "ዩኒቨርሲቲ", "college", "hospital", "ministry", "ሚኒስቴር",
              "authority", "agency", "institute", "bureau", "biiroo", "ቢሮ", "enterprise", "dhaabbata", "ድርጅት")
_STATEMENT_WORDS = ("our", "this", "keenyaa", "kanaa", "ባንካችን", "employee", "hojjetaa", "bearer", "berarer")


def organization_names(lay: layout_mod.Layout) -> list[str]:
    """The issuing organisation as printed on the card (in each language),
    top of the card first: 'Bank of Oromia', 'Baankii Hojii Gamtaa Oromiyaa'."""
    names: list[tuple[float, str]] = []
    seen = set()
    from idx.ocr_engine import unglue_ocr_text
    for ln in lay.lines:
        if re.search(r"www|https?:|\.com\b|\.org\b|@", ln.text, re.I):
            continue
        text = unglue_ocr_text(ln.text).strip()             # 'BaankiiHojiiGamtaa' -> 'Baankii Hojii Gamtaa'
        norm = labels.normalize(text)
        words = norm.split()
        covered = sum(h.end - h.start for h in labels.find_labels(text, fuzzy=False))
        if not words or len(words) > 8 or covered >= 0.4 * len(norm):
            continue                                        # a label line, not a header
        squashed = norm.replace(" ", "")
        if any(w in words for w in _STATEMENT_WORDS) or not any(
                (k in words) if len(k) < 4 else (k.replace(" ", "") in squashed) for k in _ORG_WORDS):
            continue
        if norm not in seen and sum(c.isalpha() for c in text) >= 6:
            seen.add(norm)
            names.append((ln.page * 1e6 + ln.top, text))
    return [t for _, t in sorted(names)][:4]


def _printed_forms(nationality: str) -> set[str]:
    """How a nationality can appear on a card: code, country, demonym, native names."""
    from .normalize import NATIVE_COUNTRY_NAMES, resolve_country
    code, demonym = resolve_nationality(nationality)
    forms = {nationality.upper()}
    if code:
        forms |= {code, resolve_country(code)[1].upper(), (demonym or "").upper()}
        forms |= {k.upper() for k, v in NATIVE_COUNTRY_NAMES.items() if v == code}
    return {f for f in forms if f and len(f) >= 3}


def require_printed(fields: dict[str, FieldResult], printed_text: str, und: Understanding, ctx: Context,
                    ocr_substantial: bool) -> dict:
    """Fields whose value must be PRINTED on the card (or machine-verified): an
    agent's inference is not a reading. Checked only when OCR read the card
    substantially — with little OCR text nothing is judged."""
    report: dict[str, Any] = {}
    if not ocr_substantial:
        return report
    text = labels.normalize(printed_text).upper()
    # The printed-order name must itself be printed (an agent's 'Djibouti' is not a name).
    full = fields["full_name"]
    if full.original and full.value and full.original != full.value:
        words = [w for w in labels.normalize(full.original).upper().split() if len(w) >= 3]
        if words and not all(w in text for w in words):
            full.original = full.value
    squashed = text.replace(" ", "")
    machine = {"mrz", "barcode"}
    nat = fields["nationality"]
    labelled_nat = next((a.value for a in und.values if a.field == "nationality"
                         and resolve_nationality(a.value)[0]), "")
    if labelled_nat and not machine & set(nat.verified_by) and \
            resolve_nationality(labelled_nat)[0] != resolve_nationality(nat.value)[0]:
        # The card prints "Nationality: Ethiopian": that is the nationality, whatever an agent said.
        ctx.warn(f"nationality: the card prints {labelled_nat!r} beside its nationality label"
                 + (f" (agents had {nat.value!r})" if nat.value else ""))
        fields["nationality"] = FieldResult(value=labelled_nat.upper(), confidence=0.9, source="printed_labels",
                                            verified_by=["printed_labels"],
                                            alternatives=[v for v in [nat.value] + nat.alternatives if v])
        report["nationality"] = f"from the printed label: {labelled_nat}"
        nat = fields["nationality"]
    elif labelled_nat:
        nat.verified_by.append("printed_labels")
    if nat.value and not machine & set(nat.verified_by) and not any(
            v.startswith(("checksum:", "id_number:")) for v in nat.verified_by):
        labelled = any(a.field == "nationality" for a in und.values)
        forms = _printed_forms(nat.value)
        printed = any((f"{labels.normalize(f).upper()}" in text) or labels.normalize(f).upper().replace(" ", "")
                      in squashed for f in forms)
        if not printed and not labelled:
            ctx.warn(f"nationality: {nat.value!r} is not printed on the document (no nationality field, "
                     "no country name or code) — not inferred")
            fields["nationality"] = FieldResult(value="", confidence=0.0, alternatives=[nat.value] + nat.alternatives,
                                                flags=["not_printed"])
            report["nationality"] = "dropped: not printed"
    latin, full = fields["full_name_latin"], fields["full_name"]
    if latin.value and full.value and latin.value != full.value and "ocr" in full.verified_by \
            and not machine & set(latin.verified_by) and "ocr" not in latin.verified_by:
        from .reconcile import is_latin
        if is_latin(full.value) and is_latin(latin.value):
            # The card prints its own Latin name; an unprinted "Latin name" is not a transliteration.
            ctx.warn(f"full_name_latin: {latin.value!r} is not printed — the printed Latin name "
                     f"{full.value!r} is used")
            fields["full_name_latin"] = FieldResult(value=full.value, confidence=full.confidence, source="full_name",
                                                    verified_by=list(full.verified_by),
                                                    alternatives=[latin.value] + latin.alternatives,
                                                    flags=["replaced_unprinted_latin_name"])
            report["full_name_latin"] = "replaced by the printed Latin name"
    # A phone number is not a personal number: '0933566565' printed beside 'Tel' / 'ስልክ'.
    pn = fields["personal_number"]
    digits = re.sub(r"\D", "", pn.value)
    if digits and not machine & set(pn.verified_by):
        phone_labelled = any(a.field == "phone" and re.sub(r"\D", "", a.value).endswith(digits[-7:])
                             for a in und.values)
        def near_same(a: str, b: str) -> bool:     # one misread / extra digit tolerated
            it = iter(a if len(a) >= len(b) else b)
            short = b if len(a) >= len(b) else a
            return abs(len(a) - len(b)) <= 1 and len(short) >= 7 and all(c in it for c in short)
        on_phone_line = False
        for ln in printed_text.splitlines():
            m = re.search(r"(?i)(tel|te|phone|mob|ስልክ)\W*(\+?[\d\s()]{7,})", ln)
            if m and near_same(re.sub(r"\D", "", m.group(2)), digits):
                on_phone_line = True
        if phone_labelled or on_phone_line:
            ctx.warn(f"personal_number: {pn.value!r} is the phone number printed beside the phone label — "
                     "moved to additional_fields.phone")
            fields["personal_number"] = FieldResult(value="", alternatives=[pn.value] + pn.alternatives,
                                                    flags=["is_phone_number"])
            report["phone"] = pn.value
    # A number OCR can't confirm: show the reviewer OCR's near-identical reading (never swapped in).
    doc = fields["document_number"]
    if doc.value and "not_found_in_ocr" in doc.flags:
        target = re.sub(r"\W", "", doc.value).upper()
        upper = printed_text.upper()
        tokens = set(re.findall(r"[A-Z0-9]{4,}", upper)) | set(re.findall(r"\d{4,}", upper))
        near = sorted({t for t in tokens
                       if len(t) == len(target) and t != target and sum(a != b for a, b in zip(t, target)) == 1})
        if len(near) == 1:
            ctx.review(f"document_number: agents read {doc.value!r}, OCR reads {near[0]!r} — check the card")
            doc.alternatives = [near[0]] + [a for a in doc.alternatives if a != near[0]]
            report["document_number_ocr_reading"] = near[0]
    blood = fields["blood_type"]
    labelled_blood = next((a.value for a in und.values if a.field == "blood_type"), "")
    if labelled_blood:
        from .normalize import normalize_blood_type
        value = normalize_blood_type(labelled_blood) or labelled_blood
        if blood.value and normalize_blood_type(blood.value) != normalize_blood_type(labelled_blood) \
                and not machine & set(blood.verified_by):
            ctx.warn(f"blood_type: the card prints {labelled_blood!r} beside its blood-group label "
                     f"(agents had {blood.value!r})")
            blood = fields["blood_type"] = FieldResult(value="", alternatives=[blood.value] + blood.alternatives)
        if not blood.value:
            fields["blood_type"] = FieldResult(value=value, confidence=0.8, source="printed_labels",
                                               verified_by=["printed_labels"],
                                               alternatives=blood.alternatives,
                                               flags=["restored_from_labelled_layout"] if blood.alternatives else [])
            report["blood_type"] = f"from the printed label: {value}"
        elif normalize_blood_type(blood.value) == normalize_blood_type(labelled_blood):
            blood.verified_by.append("printed_labels")
    return report


def arbitrate_spelling_by_script(fields: dict[str, FieldResult], local_names: list[str], ctx: Context) -> list[dict]:
    """An OCR spelling correction of a Latin name word is kept only if the card's
    own Ge'ez spelling does not contradict it: OCR misreads Latin letters
    ('TEFERA' as 'TEPERA'), and the Amharic line printed beside it (ተፈራ =
    te-fe-ra) is independent evidence of the right spelling."""
    from . import ethiopic
    words = [w for name in local_names if name and ethiopic.is_ethiopic(name) for w in name.split()]
    changes = []
    if not words:
        return changes
    spells = lambda latin: max((ethiopic.word_score(w, latin) for w in words), default=0) == 2
    for name in ("full_name", "full_name_latin", "first_name", "middle_name", "last_name"):
        fr = fields[name]
        if "ocr_spelling_corrected" not in fr.flags or not fr.alternatives:
            continue
        before, after = fr.alternatives[0].split(), fr.value.split()
        if len(before) != len(after):
            continue
        restored = [b if (b.upper() != a.upper() and spells(b) and not spells(a)) else a for b, a in zip(before, after)]
        if restored != after:
            value = " ".join(restored)
            changes.append({"field": name, "ocr": fr.value, "kept": value})
            ctx.warn(f"{name}: OCR's {fr.value!r} contradicts the printed Ge'ez spelling — {value!r} kept")
            fr.value = value
            fr.flags.append("ocr_spelling_overruled_by_script")
    return changes


def birth_date_expected(document_type: str) -> bool:
    return document_type in GOVERNMENT_IDENTITY_TYPES or document_type in ("", "other", "unknown")

"""
Generic, format-agnostic OCR-only fallback -- the last free/offline tier
before this project gives up on a document entirely.

Every other path in this project either recognizes a specific STRUCTURE
(MRZ's fixed-width checksummed lines, AAMVA's numbered fields, Ethiopia
ID's QR/barcode) or falls back to Gemini vision when none of those match.
What was missing: when a document matches NONE of those structures *and*
AI vision is unavailable (no GEMINI_API_KEY, the daily budget/quota is
exhausted, or the API call itself failed) -- e.g. `_universal_ai_fallback`
returning nothing -- the pipeline previously returned a completely empty
result. This module is the rescue: it doesn't know what document family
it's looking at, but "necessary info" (name, date of birth, document
number, etc.) is very often printed next to a recognizably-labeled field
regardless of the document's specific layout -- so it searches the whole
OCR'd text for a broad set of label ALIASES per field (the same field is
"Surname" on one document, "Family name" on another, "Apellido" on a
third) and extracts whatever it can find, exactly the same windowed-
search technique already proven in `aamva.py`'s Convention 3 (full-word
labels) and its numbered-convention fallback patterns -- generalized here
to not assume any particular document family at all.

Deliberately conservative: a field is only ever populated when its own
label is actually found nearby (never inferred from position alone, the
way a numbered/structured convention can), and a genuinely ambiguous date
(neither segment unambiguously a 4-digit year, so the day/month order
can't be determined) is left unparsed rather than guessed -- same
"calibration over coverage" discipline as everywhere else in this
project. Confidence is always "low": nothing here is checksum-verified
or barcode-decoded, and the document FAMILY itself is unknown.
"""

from __future__ import annotations

from typing import Any, Optional

import concurrent.futures
import re

from PIL import Image

from . import dates, ontology

# See aamva.py's Convention 3 comment for why every label below is
# wrapped in a case-insensitive-but-SCOPED group, "(?i:...)", rather than
# compiling the whole pattern with re.IGNORECASE: a global flag would
# make the value capture's `[A-Z...]` classes match lowercase too, so an
# unrelated label elsewhere in the text (mixed case) could get swallowed
# into a preceding field's captured value. Each label also has a LEADING
# \b for the same reason Convention 3 needed one: without it, a label
# could match starting mid-word inside some other printed text.
_WINDOW = r"(?:.|\n){0,60}?"

# Each word must START with a capital (real ID-card names are never
# printed all-lowercase), but the REST of the word may be upper OR lower
# case -- confirmed necessary against a real Dutch national ID card
# ("Angesom"/"Ararat", Title Case), not just AAMVA's ALL-CAPS convention
# this pattern originally assumed. Restricting the first letter to a
# capital still keeps this from matching a lowercase label fragment
# (this card's own labels -- "naam/surname", "geboortedatum/date of
# birth" -- are printed all-lowercase, so they can never satisfy this
# shape and get mistaken for a value).
#
# The word-continuation separator is `[ \t]`, deliberately NOT `\s` (which
# would also match a newline) -- confirmed necessary by a regression this
# widening itself caused: allowing lowercase letters made "Place" (of a
# following line's OWN Title-Case label, "Place of Birth:") satisfy the
# continuation word shape the same way "ETHIOPIAN"'s all-caps continuation
# never could, so a value like "ETHIOPIAN" swallowed the next LINE's
# label text into one bogus multi-word match ("ETHIOPIAN\nPlace"). A
# genuine multi-word name/value is printed on one physical line; a
# same-line-only separator keeps the wider Title-Case shape from also
# reaching across a line break into unrelated label text.
_NAME_VALUE = r"[A-ZÀ-ÿ][A-Za-zÀ-ÿ'\-]{1,30}(?:[ \t][A-ZÀ-ÿ][A-Za-zÀ-ÿ'\-]{1,30}){0,3}"
# A nationality/demonym ("Ethiopian", "Nederlandse") is structurally
# different from a personal name: it is essentially always ONE word, never
# 2-4 like a given name can legitimately be. Reusing `_NAME_VALUE`'s
# optional multi-word continuation here let the label-anchor search
# overshoot past the real value into the NEXT field's own (glued) label --
# confirmed against a real Ethiopian driving license, where the search
# from "Nationality" greedily joined "Ethiopian" with the following
# "SexM" (itself "Sex" + "M" glued with no space -- see `ocr_engine.py`'s
# module docstring on RapidOCR's own space-dropping) into one bogus
# two-word match, since `search_from_anchor` always tries the longest
# phrase first. A single-word-only pattern removes the ambiguity instead
# of trying to detect "is the next token actually another field's label"
# (a `search_from_anchor`-level fix that's much harder to get right
# in general -- see that function's own "known limitation" note).
_NATIONALITY_VALUE = r"[A-Z][A-Za-z'\-]{1,30}"
# Requires at least one digit somewhere in the match -- a real unlabeled
# document/ID number (CLAUDE.md's own Belgian-permit example, "6191348
# 32") always contains one; a bare run of letters never should, but
# WITHOUT this constraint this shape is indistinguishable from an
# ordinary all-caps word. Confirmed a real false positive: an Ethiopian
# driving license's "TUJUBA" (part of the holder's own name) satisfied
# the old letters-only-permitted shape and won this field's tier-5
# signature sweep outright, since the real value ("952592") was glued to
# its own label with no space and never became an isolated token at all.
_ALNUM_VALUE = r"(?=[A-Z0-9\-]*\d)[A-Z0-9][A-Z0-9\-]{4,19}"
# Nordic / Belgian-style civil numbers: YYMMDD-XXXX (hyphen may be spaced
# by unglue_ocr_text). Not a generic digit run — those collide with CAN
# serials and compact dates.
_PERSONAL_VALUE = (
    r"(?:\d{2}\s*[.]\s*\d{2}\s*[.]\s*\d{2}\s*[-–]\s*\d{3}\s*[.]\s*\d{2}"
    r"|\d{6,8}\s*[-–]\s*\d{3,5})"
)
# Permit category on the same printed line as its label (``A. BEPERKT
# VERBLIJF``, ``LIMITED RESIDENCE``). Same-line only so the next field's
# label is not swallowed. Do not capture the English tail of the label
# itself (``OF PERMIT`` after ``TYPE``).
_PERMIT_VALUE = (
    r"(?!(?:OF|TYPE|SOORT|TITEL|PERMIT|CATEGORY)\b)"
    r"[A-Z0-9][A-Za-z0-9.' \t-]{1,40}"
)
# Tolerates '.', '/', '-', and a stray space next to any of them as the
# date separator -- the same OCR-noise tolerance already confirmed
# necessary in aamva.py against a real card (a thin separator character
# is exactly the kind of glyph OCR drops or smudges).
_DATE_SEP = r"[\s./-]{1,3}"
_ALPHA_MONTH = r"[A-Za-zÀ-ÿ]{3,4}"
_DATE_VALUE = (
    r"(?:\d{1,2}\s*" + _ALPHA_MONTH + r"\s*/\s*" + _ALPHA_MONTH + r"\s*\d{2,4}"
    r"|\d{1,2}\s+" + _ALPHA_MONTH + r"\s+\d{2,4}"
    r"|\d{1,4}" + _DATE_SEP + r"\d{1,2}" + _DATE_SEP + r"\d{1,4}"
    r"|\d{6,8})"
)
# `[MF]` alone matches the letter inside any nearby word (`AM`,
# `MELAT`, `MOTOR`). Require a whole token -- the printed code, or the
# full word form. Glued `SexM` is split by `unglue_ocr_text` first.
_SEX_VALUE = (
    r"(?:Male|Female|ወንድ|M\s*/\s*M|V\s*/\s*F|M\s*/\s*F|W\s*/\s*M"
    r"|(?<![A-Za-z])[MF](?![A-Za-z]))"
)
# Parenthesized local numbers (`(0912) 344858`) and compact `0912344858`.
_PHONE_VALUE = r"\(?\s*0\d{2,4}\s*[)）]?\s*\d{5,8}"
# Prefer the Latin grade on a bilingual slash line (`የህዝብ 1 / Public 1`).
_CLASS_VALUE = r"Public\s*[1-9A-Z]|Class\s+[A-Z0-9]+"


def _labeled(labels: str, value: str) -> re.Pattern[str]:
    """For the small number of fields not yet migrated into the shared
    ontology (see `_FIELD_PATTERNS`'s own comment) -- takes a literal
    label alternation directly, same as every field used to before
    `ontology.py` existed.
    """
    return re.compile(r"(?i:\b(?:" + labels + r")\b)" + _WINDOW + r"(" + value + r")")


def _labeled_by_field(field: str, value: str) -> re.Pattern[str]:
    """The label half now comes from `ontology.label_alias_pattern` (see
    docs/FIELD_ONTOLOGY.md) instead of a hardcoded literal here -- one
    source of truth for "what are this field's known labels". The VALUE
    shape stays defined here: value signatures aren't a translated-label
    concern, and combining a label match with the required-adjacent-value
    shape is this function's whole job.
    """
    label_pattern = ontology.label_alias_pattern(field).pattern
    return re.compile(label_pattern + _WINDOW + r"(" + value + r")")


# One regex per field, tried in order against the whole OCR'd text --
# first alias to match wins. `sex`/`full_name` aren't (yet) in the shared
# ontology (see concepts/fields.yaml) -- `full_name` because "name" alone
# is too generic a concept to be worth cataloguing per-language yet, `sex`
# because its aliases are still hardcoded below pending a real multi-
# language source for it. Both kept as local literals for now rather than
# forcing them into the ontology just to say every field lives there.
# The bare "name" alternative requires a following colon -- unlike
# "full name" (specific enough on its own), a bare `\bname\b` is one
# stray punctuation mark away from a false match: a noisy OCR pass
# reading "given names" as "given name!" puts a non-word character (the
# "!") right after "name", which is enough to satisfy `\b` even though
# the real word was "names", plural -- confirmed against a real Dutch ID
# card, where one of several preprocessing-variant OCR passes did exactly
# this and the resulting match swallowed the NEXT field's own value
# ("Ararat") as if it were this card's full name. A real standalone
# "Name" label is virtually always followed by a colon (or occasionally
# just ends the line); requiring the colon keeps the legitimate case
# (see the "Name: MOCKGIVEN MOCKSURNAME" test) while excluding this one.
_FIELD_PATTERNS = {
    "surname": _labeled_by_field("surname", _NAME_VALUE),
    "given_names": _labeled_by_field("given_names", _NAME_VALUE),
    "full_name": re.compile(r"(?i:\bfull\s*name\b|\bname\s*:)" + _WINDOW + r"(" + _NAME_VALUE + r")"),
    "date_of_birth": _labeled_by_field("date_of_birth", _DATE_VALUE),
    "date_of_issue": _labeled_by_field("date_of_issue", _DATE_VALUE),
    "date_of_expiry": _labeled_by_field("date_of_expiry", _DATE_VALUE),
    "document_number": _labeled_by_field("document_number", _ALNUM_VALUE),
    "personal_number": _labeled_by_field("personal_number", _PERSONAL_VALUE),
    "sex": _labeled_by_field("sex", _SEX_VALUE),
    "nationality": _labeled_by_field("nationality", _NATIONALITY_VALUE),
    "place_of_birth": _labeled_by_field("place_of_birth", _NAME_VALUE),
    "phone_number": _labeled_by_field("phone_number", _PHONE_VALUE),
    "license_class": _labeled_by_field("license_class", _CLASS_VALUE),
    "permit_type": _labeled_by_field("permit_type", _PERMIT_VALUE),
}
# Address needs the rest of its own line (a name/date-shaped single token
# isn't representative of what an address looks like), not the shared
# per-field value shape above.
_ADDRESS_LABEL_RE = re.compile(r"(?i:\baddress\b|\bplace of residence\b)")
_ADMIN_ADDRESS_RE = re.compile(
    r"(?i)\bRegion\s*:?\s*.+",
)
_DATE_FIELDS = ("date_of_birth", "date_of_issue", "date_of_expiry")


def _slash_date_to_iso(raw: str) -> str:
    """Unlike aamva.py's own version of this (which can assume MM/DD/YYYY
    for US-convention cards and YYYY/MM/DD for Ontario-style ones based on
    which document family it's parsing), this module has NO jurisdiction
    context at all -- both DD/MM/YYYY (most of the world) and MM/DD/YYYY
    (the US) are real, common conventions when the year is printed last,
    and nothing here says which one applies. Only resolves that case when
    the printed VALUES themselves make it unambiguous (one segment can't
    possibly be a month, so the other must be); returns "" (never guesses)
    when both non-year segments are <=12 and either order would be
    grammatically valid. Year-first (YYYY/MM/DD) is unambiguous on its
    own -- ISO order always puts month before day, regardless of country.
    Dual-calendar pairs are resolved by `dates.resolve_printed_dates`.
    """
    return dates.slash_date_to_iso(raw)


def infer_date_order(raw: str, known_iso: str) -> str:
    """Cross-field date-order inference (see GLOBAL_COVERAGE.md's section
    of the same name): `_slash_date_to_iso` alone refuses to resolve a
    raw slash-separated date when both non-year segments are <=12
    (genuinely ambiguous on its own, and there's no jurisdiction context
    here to guess a convention from). But when `raw` is already known to
    represent the SAME calendar date as `known_iso` -- e.g. a passport's
    printed (VIZ) date_of_birth is redundant with its own checksum-
    verified MRZ date_of_birth -- trying both plausible orderings of
    `raw` and checking which one reproduces `known_iso` resolves the
    document's date convention for free, with no risk of guessing wrong:
    by construction, only one ordering can match an already-independently
    -verified date.

    Returns "MDY" or "DMY" (the convention that reproduced `known_iso`),
    or "" when neither ordering matches (this anchor doesn't resolve it --
    still not a license to guess).
    """
    parts = [p.strip() for p in re.split(_DATE_SEP, raw) if p.strip()]
    if len(parts) != 3 or not all(p.isdigit() for p in parts) or len(parts[2]) != 4:
        return ""
    from datetime import date

    a, b, yyyy = parts
    if int(a) == int(b):
        # A degenerate date like "01/01/2028" matches EITHER ordering by
        # construction (day == month), so it can never actually tell the
        # two conventions apart -- confirmed against a real UAE Resident
        # Identity Card, whose expiry date happened to be day-equals-
        # month. Accepting it as "confirmation" of MDY (the first
        # alternative tried below, which trivially matches too) wrongly
        # inferred a document-wide MDY convention from a date carrying
        # zero real signal, then misapplied it to the card's genuinely
        # DD/MM-ordered date_of_birth and date_of_issue.
        return ""
    for order, (mm, dd) in (("MDY", (a, b)), ("DMY", (b, a))):
        try:
            if date(int(yyyy), int(mm), int(dd)).isoformat() == known_iso:
                return order
        except ValueError:
            continue
    return ""


def resolve_date_with_known_order(raw: str, order: str) -> str:
    """Parse `raw` using an already-resolved convention ("MDY" or "DMY"
    -- see `infer_date_order`) instead of guessing. Applies that SAME
    document's convention to a DIFFERENT field with no independent
    verification of its own (e.g. date_of_issue, which no MRZ format
    encodes at all) -- one document uses one convention throughout, so a
    convention pinned by one field's redundant MRZ/VIZ pair is real
    evidence for every other ambiguous date on the same document. Returns
    "" on any parse/validity failure, never falls back to guessing a
    different order.
    """
    parts = [p.strip() for p in re.split(_DATE_SEP, raw) if p.strip()]
    if len(parts) != 3 or not all(p.isdigit() for p in parts) or len(parts[2]) != 4:
        return ""
    from datetime import date

    a, b, yyyy = parts
    mm, dd = (a, b) if order == "MDY" else (b, a)
    try:
        return date(int(yyyy), int(mm), int(dd)).isoformat()
    except ValueError:
        return ""


def ocr_full_text(image_path: str) -> str:
    """OCRs the WHOLE image (unlike aamva.py/ethiopia_id.py's own OCR
    passes, there's no known field layout here to crop to -- a name/DOB/
    address label could be anywhere on an unrecognized document), across
    every preprocessing variant, joined into one text blob for
    `extract_generic_fields` to search. Parallelized through a thread
    pool for the same reason as every other OCR loop in this project
    (each variant's OCR call is a real OS subprocess/independent work,
    not in-process-shared) -- see `preprocess.py`'s `OMP_THREAD_LIMIT=1`
    for the prerequisite that makes this a genuine speedup instead of CPU
    oversubscription.
    """
    from .ocr_engine import ocr_text, ocr_tesseract_page
    from .preprocess import preprocess_variants

    variants = list(preprocess_variants(image_path).values())

    def _ocr_one(variant: Image.Image) -> Optional[str]:
        return ocr_text(variant)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(variants))) as executor:
        results = list(executor.map(_ocr_one, variants))
    parts = [r for r in results if r]
    # One extra Tesseract page as a vote (RapidOCR stays primary). Same
    # pattern as aamva._ocr_texts: spaces and 5/S on mixed IDs that
    # RapidOCR glues or confuses.
    if variants:
        tess = ocr_tesseract_page(variants[0])
        if tess:
            parts.append(tess)
    return "\n".join(parts)


def _extract_date_field(text: str, field: str) -> tuple[str, list[str]]:
    """All date-shaped tokens in a short window after EVERY occurrence of
    this field's label (ocr_full_text concatenates preprocessing variants,
    so the first hit may be a poor pass that only saw the Ethiopian
    calendar date). Also looks a short distance BEFORE the label: on an
    Ethiopian driving licence the EC date sits to the left of 'Issue Date'
    and the GC compact date to the right. Dual-calendar resolution then
    picks Gregorian.
    """
    found: list[str] = []
    for start, end in ontology.find_label_spans(text, field):
        pre = text[max(0, start - 80): start]
        last_end = 0
        last_was_date = False
        for other in _FIELD_PATTERNS:
            if other == field:
                continue
            for _s, e in ontology.find_label_spans(pre, other):
                if e >= last_end:
                    last_end = e
                    last_was_date = other in _DATE_FIELDS
        pre = "" if last_was_date else pre[last_end:]
        window = text[end: end + 160]
        cut = len(window)
        for other in _DATE_FIELDS:
            if other == field:
                continue
            for other_start, _other_end in ontology.find_label_spans(window, other):
                if 0 <= other_start < cut:
                    cut = other_start
        window = window[:cut]
        pre_raws = dates.date_tokens(pre)
        post_raws = dates.date_tokens(window)
        # Value is printed to the right or below the label. A date sitting
        # *above* this label is a different field (typically date_of_birth
        # above date_of_issue). The exception is a dual-calendar pair
        # (Ethiopian EC left of Issue Date, GC right of it).
        pre_iso = dates.resolve_printed_dates(pre_raws)
        post_iso = dates.resolve_printed_dates(post_raws)
        if (
            pre_iso and post_iso and pre_iso != post_iso
            and not dates.years_are_dual_calendar(pre_iso, post_iso)
        ):
            found.extend(post_raws)
        else:
            found.extend(pre_raws)
            found.extend(post_raws)
    return dates.resolve_printed_dates(found), found


def _value_after_label(text: str, field: str, value_re: str) -> str:
    compiled = re.compile(value_re)
    for _start, end in ontology.find_label_spans(text, field):
        rest_line, newline, below = text[end:].partition("\n")
        competing = False
        for other in _FIELD_PATTERNS:
            if other == field:
                continue
            if ontology.find_label_spans(rest_line, other):
                competing = True
                break
        window = (below[:100] if competing and newline else text[end: end + 80])
        for m in compiled.finditer(window):
            value = m.group(0).strip()
            if field == "permit_type":
                for raw in dates.date_tokens(value):
                    idx = value.find(raw)
                    if idx > 0:
                        value = value[:idx].strip()
                        break
                value = _normalize_permit_type(value)
                if not _plausible_permit_category(value):
                    continue
            elif field in ("surname", "given_names", "full_name", "place_of_birth"):
                if _name_value_is_poisoned(value):
                    continue
            return value
    return ""


_BARE_CLASS_LABEL_RE = re.compile(r"(?i)\bClass\b")


def _bare_class_value(text: str) -> str:
    """"Class" as its own plain-English label -- deliberately NOT a shared
    ontology alias (`_FIELD_PATTERNS`'s own module comment: it collides
    with AAMVA's numbered "9 CLASS"/"9CLASS" field marker), but a
    non-AAMVA card genuinely using this plain label still deserves to
    resolve -- confirmed against a real Australian (ACT) driver licence
    printing "Class C" with no numbered convention anywhere on the card.
    Excluded whenever a digit sits immediately before "Class" (allowing a
    little OCR-noise spacing), the shape of AAMVA's own field marker, not
    a plain label.
    """
    for m in _BARE_CLASS_LABEL_RE.finditer(text):
        # A SHORT (1-2 digit) field-number marker directly before "Class",
        # same line -- AAMVA's own "9 CLASS"/"9CLASS"/"12 CLASS" shape.
        # `(?<!\d)` keeps this from also matching the tail of an unrelated,
        # longer digit run (a document number, postal code, ...) ending
        # the line above; `[ \t]*$` (no `\n`) keeps it same-line-only.
        if re.search(r"(?<!\d)\d{1,2}[ \t]*$", text[: m.start()]):
            continue
        value_m = re.match(r"[ \t]*([A-Z0-9]{1,3})\b", text[m.end():], re.IGNORECASE)
        if value_m:
            return value_m.group(1).upper()
    return ""


def _apply_shared_date_order(out: dict[str, Any], date_raws: dict[str, list[str]]) -> None:
    """Once one date on the card is resolved (dual calendar or an
    unambiguous day>12), reuse that DMY/MDY convention for remaining
    ambiguous slash dates — see infer_date_order.
    """
    order = ""
    for field, raws in date_raws.items():
        iso = out.get(field, "")
        if not iso:
            continue
        for raw in raws:
            expanded = re.sub(r"[\s./-]+", "/", dates.expand_compact_date(raw)).strip("/")
            inferred = infer_date_order(expanded, iso)
            if inferred:
                order = inferred
                break
        if order:
            break
    if not order:
        return
    for field in _DATE_FIELDS:
        if out.get(field):
            continue
        for raw in date_raws.get(field) or []:
            expanded = re.sub(r"[\s./-]+", "/", dates.expand_compact_date(raw)).strip("/")
            iso = resolve_date_with_known_order(expanded, order)
            if iso:
                out[field] = iso
                break


def ambiguous_date_notes(text: str) -> list[str]:
    """Human-readable notes for date fields with real label-anchored
    printed evidence that could not be safely resolved (both non-year
    components <=12, so DD/MM vs MM/DD genuinely can't be told apart
    without jurisdiction context). CLAUDE.md's own "known open issue"
    policy: never guess, but don't silently drop the printed value either
    -- emit the raw reading so a human reviewer has something to check
    against the physical document, rather than an unresolved field with
    no trace it was ever printed at all. Confirmed against a real UAE
    Resident Identity Card, whose issuing date ("02/01/2026") is exactly
    this case.
    """
    from .ocr_engine import ocr_space_for_printed_word_gap, unglue_ocr_text

    text = _strip_indexed_viz_markers(text)
    text = dates.repair_glued_alpha_dates(text)
    text = ocr_space_for_printed_word_gap(unglue_ocr_text(text))
    notes = []
    for field in _DATE_FIELDS:
        iso, raws = _extract_date_field(text, field)
        if iso:
            continue
        # Only a genuinely date-shaped candidate is worth showing to a
        # reviewer -- `raws` can also carry an unrelated digit-run
        # fragment that merely LOOKS date-token-shaped (confirmed on the
        # same UAE card: "98-627469", a glued remnant of the birth year
        # plus part of the personal number, tied on raw-string frequency
        # with the real "06/10/1998" and winning the tie-break).
        plausible = [r for r in raws if _marks_a_real_date(r)]
        if not plausible:
            continue
        common = re.sub(r"\s*([/.-])\s*", r"\1", max(set(plausible), key=plausible.count))
        notes.append(
            f"{field}: printed as '{common}' but the day/month order is ambiguous on this "
            f"document (no jurisdiction context to resolve DD/MM vs MM/DD) -- verify against "
            f"the physical document."
        )
    return notes


def extract_generic_fields(text: str) -> dict[str, Any]:
    """Pure string -> dict, no image/OCR involved (mirrors
    `aamva._parse_front_ocr_text`'s own testing convention) -- searches
    `text` for every field's label aliases and returns whatever's found.
    Never raises; an unrecognized/blank input just yields an empty dict.
    """
    from .ocr_engine import ocr_space_for_printed_word_gap, unglue_ocr_text

    text = _strip_indexed_viz_markers(text)
    text = dates.repair_glued_alpha_dates(text)
    text = ocr_space_for_printed_word_gap(unglue_ocr_text(text))
    out: dict[str, Any] = {}
    date_raws: dict[str, list[str]] = {}
    for field, pattern in _FIELD_PATTERNS.items():
        if field in _DATE_FIELDS:
            iso, raws = _extract_date_field(text, field)
            date_raws[field] = raws
            if iso:
                out[field] = iso
            continue
        m = pattern.search(text)
        if not m:
            continue
        value = m.group(1).strip()
        if field == "sex":
            sex = _normalize_sex(value)
            if sex:
                out[field] = sex
        elif field == "personal_number":
            out[field] = _hyphenated_personal_id(value) or re.sub(r"\s+", "", value)
        elif field == "phone_number":
            out[field] = _normalize_phone(value)
        elif field == "license_class":
            out[field] = re.sub(r"\s+", " ", value)
        elif field == "permit_type":
            continue
        elif field == "nationality":
            from . import countries as countries_mod

            # Try every match, not just the first -- "Citizenship" is a
            # real nationality-field alias, but on a real UAE Resident
            # Identity Card it also appears inside an unrelated header
            # caption ("...IDENTITY & CITIZENSHIP, CUSTOMS AND PORT
            # SECURITY"), whose adjacent value-shaped word ("CUSTOMS")
            # doesn't resolve as a nationality and was silently rejected
            # by the check below -- but the REAL "Nationality Ethiopia"
            # line further down was never tried since `.search()` only
            # ever looks at the first match.
            for hit in pattern.finditer(text):
                cand = hit.group(1).strip()
                if countries_mod.resolve_nationality(cand) and not _name_value_is_poisoned(cand):
                    out[field] = cand
                    break
        elif field in ("surname", "given_names", "full_name", "place_of_birth"):
            chosen = ""
            for hit in pattern.finditer(text):
                cand = hit.group(1).strip()
                if _name_value_is_poisoned(cand):
                    continue
                chosen = cand
                break
            if chosen:
                out[field] = chosen
        else:
            out[field] = value

    from .ocr_engine import ocr_space_for_printed_word_gap

    for name_field in ("given_names", "full_name"):
        if name_field in out:
            out[name_field] = ocr_space_for_printed_word_gap(str(out[name_field]))

    if "document_number" not in out:
        value = _value_after_label(text, "document_number", _ALNUM_VALUE)
        if value and not _looks_like_printed_date(value):
            out["document_number"] = value
    if "full_name" not in out:
        value = _value_after_label(text, "full_name", _NAME_VALUE)
        if value:
            if not _name_value_is_poisoned(value):
                out["full_name"] = value
    if "surname" not in out:
        value = _value_after_label(text, "surname", _NAME_VALUE)
        if value and not _name_value_is_poisoned(value):
            out["surname"] = value
    if "given_names" not in out:
        value = _value_after_label(text, "given_names", _NAME_VALUE)
        if value and not _name_value_is_poisoned(value):
            out["given_names"] = value
    if "phone_number" not in out:
        value = _value_after_label(text, "phone_number", _PHONE_VALUE)
        if value:
            out["phone_number"] = _normalize_phone(value)
    if "license_class" not in out:
        value = _value_after_label(text, "license_class", _CLASS_VALUE)
        if value:
            out["license_class"] = re.sub(r"\s+", " ", value).strip()
    if "license_class" not in out:
        value = _bare_class_value(text)
        if value:
            out["license_class"] = value
    if "place_of_birth" not in out:
        value = _value_after_label(text, "place_of_birth", _NAME_VALUE)
        if value and not _name_value_is_poisoned(value):
            out["place_of_birth"] = value
    if "permit_type" not in out:
        value = _extract_permit_type(text)
        if value:
            out["permit_type"] = value

    _apply_shared_date_order(out, date_raws)

    if "surname" not in out and "given_names" not in out and out.get("full_name"):
        # No separately-labeled surname/given-names, but a generic "Name"
        # label was found -- split on whitespace as a last-resort guess
        # at which part is which (first token = given, rest = surname),
        # flagged by the caller via a warning rather than asserted with
        # unearned confidence. Restricted to EXACTLY two tokens: a
        # patronymic-only name (no family surname at all, confirmed
        # against a real UAE Resident Identity Card's "Name:Henok Feseha
        # Tilahun" -- an Ethiopian holder's own given name, father's
        # given name, and grandfather's given name, not a given+family
        # pair) has three-plus tokens, and "first = given, REST = one
        # surname" would glue two unrelated patronymic tokens together
        # into a fabricated "surname" that was never printed as such.
        # Two tokens has only one reasonable reading either way.
        parts = out["full_name"].split()
        if len(parts) == 2:
            out.setdefault("given_names", parts[0])
            out.setdefault("surname", parts[1])

    addr_m = _ADDRESS_LABEL_RE.search(text)
    if addr_m:
        # The label match ends right after "Address" itself, not at the
        # end of that printed line -- whatever's left on that SAME line
        # (e.g. a colon, or "Address: 123 Main St" printed all on one
        # line) needs handling separately from the lines that follow,
        # the same "header match ends mid-line" fix already needed in
        # aamva.py's own row-sharing parsers.
        rest = text[addr_m.end():]
        same_line, _, remainder = rest.partition("\n")
        same_line = same_line.strip().lstrip(":").strip()
        lines = ([same_line] if same_line else []) + [
            ln.strip() for ln in remainder.splitlines() if ln.strip()
        ]
        lines = lines[:2]
        if lines:
            from .ocr_engine import ocr_space_for_printed_word_gap

            out["address"] = ocr_space_for_printed_word_gap(", ".join(lines))

    if "address" not in out:
        admin = _ADMIN_ADDRESS_RE.search(text)
        if admin:
            out["address"] = re.sub(r"\s+", " ", admin.group(0).strip())

    _fill_unlabeled_identity_fields(
        text, out, document_number_labeled="document_number" in out, date_raws=date_raws,
    )
    if out.get("given_names"):
        cleaned = " ".join(
            p for p in str(out["given_names"]).split() if not _looks_like_date_ocr_junk(p)
        )
        if cleaned:
            out["given_names"] = _space_glued_allcaps_given(cleaned, text)
        else:
            out.pop("given_names", None)
    serials = _can_serials(text)
    if serials:
        out.setdefault("card_access_number", sorted(serials)[0])
    bilingual_sex = [
        _normalize_sex(token.replace(".", "/"))
        for token in re.findall(
            r"\b(M\s*/\s*M|V\s*/\s*F|M\s*/\s*F|W\s*/\s*M)\b", text, re.I
        )
    ]
    bilingual_sex = [token for token in bilingual_sex if token]
    if bilingual_sex:
        from collections import Counter

        out["sex"] = Counter(bilingual_sex).most_common(1)[0][0]
    return out


# The value-SHAPE half of each field's combined label+value pattern above,
# standalone -- what idx/sweep.py's `locate()` needs (it supplies its own
# label-anchor logic via the ontology, positionally, so it only wants the
# value shape, not another copy of the label alternation).
_VALUE_PATTERNS = {
    "surname": re.compile(_NAME_VALUE),
    "given_names": re.compile(_NAME_VALUE),
    "full_name": re.compile(_NAME_VALUE),
    "date_of_birth": re.compile(_DATE_VALUE),
    "date_of_issue": re.compile(_DATE_VALUE),
    "date_of_expiry": re.compile(_DATE_VALUE),
    "document_number": re.compile(_ALNUM_VALUE),
    "personal_number": re.compile(_PERSONAL_VALUE),
    "sex": re.compile(_SEX_VALUE, re.IGNORECASE),
    "nationality": re.compile(_NATIONALITY_VALUE),
    "place_of_birth": re.compile(_NAME_VALUE),
    "phone_number": re.compile(_PHONE_VALUE),
    "license_class": re.compile(_CLASS_VALUE, re.IGNORECASE),
    "permit_type": re.compile(_PERMIT_VALUE),
}

# Fields sharing `_NAME_VALUE`'s shape ("1-4 capitalized words") have no
# real unlabeled SIGNATURE at all -- practically any prominent capitalized
# text satisfies it (a document's own bilingual header, in the real Dutch
# ID card case that exposed this). `extract_generic_fields`'s own module
# docstring already states the rule this module should share: "a field is
# only ever populated when its own label is actually found nearby" --
# `sweep.locate`'s `allow_unlabeled=False` enforces that same discipline
# here for exactly these fields, while genuinely distinctive shapes
# (dates, alnum IDs, the sex code) keep step B, where it's actually a
# meaningful signature.
_NO_UNLABELED_SIGNATURE = {
    "surname", "given_names", "full_name", "nationality", "place_of_birth", "permit_type",
}

# Fallback only if document_vocabulary.yaml failed to load. Prefer the
# YAML file — that is the vocabulary that grows without a code change.
# Order still matters: licences before "card" / national_id.
_DOCUMENT_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("passport", ("passport",)),
    (
        "drivers_license",
        ("driving licence", "driving license", "driver's license", "driver's licence",
         "drivers license", "drivers licence"),
    ),
    ("residence_permit", ("residence permit", "resident permit", "residency permit")),
    ("national_id", ("identity card", "national id", "identification card", "national identity")),
)


def _looks_like_printed_date(value: str) -> bool:
    """Compact calendar dates must not win document_number. Digit runs of
    9+ (``882910732``) and 7-digit serials are not dates even when they
    contain ``19`` or ``20`` as a substring (``_year_of`` would otherwise
    treat that as a year).
    """
    compact = re.sub(r"\D", "", value or "")
    if len(compact) >= 9 or len(compact) <= 5:
        return False
    if len(compact) == 7:
        return bool(re.fullmatch(r"\d{1,3}(?:19|20)\d{2}", compact))
    year = dates._year_of(value)
    return year is not None and 1900 <= year <= 2100


def _normalize_phone(value: str) -> str:
    return re.sub(r"\s+", "", value).replace("）", ")").replace("（", "(")


def _normalize_sex(value: str) -> str:
    """Map a printed sex token onto M/F/X. Bilingual prints (M/M, V/F,
    ወንድ) are confirmed on real Belgian / Dutch / Ethiopian cards.
    Empty string when the token is not a known sex value — never guess.
    """
    stripped = value.strip()
    if stripped == "ወንድ":
        return "M"
    compact = stripped.upper().replace(" ", "").replace(".", "/")
    if compact in {"M", "MALE", "M/M", "W/M"}:
        return "M"
    if compact in {"F", "FEMALE", "V/F", "M/F"}:
        return "F"
    if compact == "X":
        return "X"
    return ""


# Words that are printed on almost every identity document header and
# must never be taken as a holder's name. Not a jurisdiction list.
_NAME_STOPWORDS = frozenset({
    "identity", "card", "kingdom", "republic", "federal", "national",
    "passport", "license", "licence", "driving", "driver", "permit",
    "given", "names", "surname", "family", "birth", "date", "issue",
    "issuing", "expiry", "sex", "male", "female", "the", "and", "for", "name",
    "identiteitskaart", "identite", "koninkrijk", "document", "number",
    "customer", "identifier", "address", "place", "class", "valid",
    "until", "gender", "nationality", "citizenship",
    "identitetskort", "nationellt", "sverige", "sweden", "holder",
    "signature", "height", "personal", "kortnr", "personnr",
    "efternamn", "fornamn", "förnamn", "nationalitet", "fodelsedatum",
    "födelsedatum", "giltighetsdag", "langd", "längd",
    "personalausweis", "bundesrepublik", "deutschland", "diplomatic",
    "provisional", "repubblica", "italiana", "ministero", "interno",
    "emirates", "united", "arab", "authority",
    "capital", "territory", "identification", "government", "conditions",
    "naam", "voornamen", "forenames", "forename", "geslacht",
    "nationaliteit", "geboortedatum", "opmerkingen", "remarks",
    "verblijfstitel", "geldig", "rijksregisternr",
    "residence", "soort", "titel",
    "vornamen", "geburtsname", "geburtstag", "geburtsort",
    "staatsangehorigkeit", "staatsangehörigkeit", "gultig", "gültig",
    "naissance", "prenoms", "prénoms", "lieu",
})
# Title Case (Dutch ID) or ALL CAPS (Swedish / AAMVA VIZ).
_NAME_TOKEN_RE = re.compile(r"\b([A-Z][a-z]{2,20}|[A-Z]{3,20})\b")
_GLUED_SEX_NATIONALITY_RE = re.compile(r"^(?:MM|VF|MF|WM)[A-Z]{3,}$")
_MIXED_DOC_ID_RE = re.compile(
    r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{8,16}\b",
    re.IGNORECASE,
)
_SEGMENTED_NATIONAL_ID_RE = re.compile(
    r"\b(\d{3}\s*[-–]\s*\d{4}\s*[-–]\s*\d{6,7}\s*[-–]\s*\d)\b"
)
_EU_VIZ_SURNAME_RE = re.compile(r"(?i)\b1[.\s]+([A-Z][A-Za-z'\-]{2,30})\b")
_EU_VIZ_GIVEN_RE = re.compile(
    r"\b2[.\s]+(?:(?:MR|MRS|MS|MISS|DR)\.?\s+)?([A-Z][A-Za-z'\- ]{2,40})",
    re.IGNORECASE,
)
_EU_VIZ_DOC_RE = re.compile(r"(?i)\b5[.\s]+([A-Z0-9]{8,16}(?:\s+\d{2})?)\b")
_PLACE_AFTER_DATE_RE = re.compile(
    r"\b\d{1,2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{2,4}\s+([A-Z][A-Za-z]{2,20}(?:\s+[A-Z][A-Za-z]{2,20})?)\b"
)
_PLACE_BEFORE_DATE_RE = re.compile(
    r"\b([A-Z][A-Za-z]{2,24})\s+\(\s*[A-Z]{3}\s*\)\s+\d{1,2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{2,4}\b"
)
_COMMA_NAME_RE = re.compile(
    r"\b([A-Z]{2,24}),\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\b"
)
# The same "ALL-CAPS surname, Title-Case given names" convention as
# `_COMMA_NAME_RE`, but with NO comma at all -- confirmed against a real
# Australian (ACT) driver licence printing "AGO Binyam Wendemu" as one
# bare, unlabeled line. Anchored to the WHOLE line (unlike `_COMMA_NAME_RE`,
# which is searched anywhere) since dropping the comma removes the one
# punctuation cue that made an unanchored search safe; each given-name
# token's second character must be lowercase (genuine Title Case), so an
# all-caps address/header word sitting next to the surname on its own
# line ("MAWSON ACT") can never satisfy the given-name shape.
_ALLCAPS_TITLECASE_NAME_LINE_RE = re.compile(
    r"^[ \t]*([A-Z][A-Z'\-]{1,30})[ \t]+"
    r"([A-Z][a-z][a-zA-Z'\-]{0,29}(?:[ \t][A-Z][a-z][a-zA-Z'\-]{0,29}){0,3})[ \t]*$",
    re.MULTILINE,
)
_HYPHEN_PERSONAL_RE = re.compile(r"\b(\d{6,8})\s*[-–]\s*(\d{3,5})\b")
_DOTTED_CIVIL_RE = re.compile(
    r"\b(\d{2})\s*[.]\s*(\d{2})\s*[.]\s*(\d{2})\s*[-–]\s*(\d{3})\s*[-.]?\s*(\d{2})\b"
)
_SPACED_DIGIT_DOC_RE = re.compile(r"\b(\d{6,8})\s+(\d{2})\b")
_DIGIT_PERSONAL_RE = re.compile(r"\b(\d{6,11})\b")
_ID_CONFUSABLES = {("S", "5"), ("5", "S"), ("O", "0"), ("0", "O"),
                   ("I", "1"), ("1", "I"), ("B", "8"), ("8", "B")}


def _fill_eu_numbered_viz(text: str, out: dict[str, Any]) -> None:
    """EU/UK driving-licence numbered VIZ (fields 1, 2, 4a, 4b, 5).

    Distinct from AAMVA: those cards print 4d as the credential number.
    """
    from . import countries

    if re.search(r"\b4d\b", text, re.I):
        return
    if not (re.search(r"\b4a\b", text, re.I) and re.search(r"\b4b\b", text, re.I)):
        return
    if "surname" not in out:
        m = _EU_VIZ_SURNAME_RE.search(text)
        if m and _keep_viz_name_token(m.group(1), countries):
            out["surname"] = m.group(1)
    if "given_names" not in out:
        m = _EU_VIZ_GIVEN_RE.search(text)
        if m:
            given = (m.group(1) or "").strip()
            given = re.sub(r"\s+", " ", given)
            first = given.split()[0] if given else ""
            if first and _keep_viz_name_token(first, countries):
                out["given_names"] = given
    if "document_number" not in out or (out.get("document_number") or "").isdigit():
        m = _EU_VIZ_DOC_RE.search(text)
        if m:
            value = re.sub(r"\s+", "", m.group(1))
            current = out.get("document_number", "")
            if not current or current.isdigit() or len(value) > len(current):
                out["document_number"] = value


def _can_serials(text: str) -> set[str]:
    """EU eID Card Access Number is the word CAN plus a short digit group.
    That serial is not a civil personal number and is not Canada.
    """
    return set(re.findall(r"\bCAN\s+(\d{4,8})\b", text, re.I))


def _hyphenated_personal_id(text: str) -> str:
    dotted = _DOTTED_CIVIL_RE.search(text)
    if dotted:
        return (
            f"{dotted.group(1)}.{dotted.group(2)}.{dotted.group(3)}"
            f"-{dotted.group(4)}.{dotted.group(5)}"
        )
    m = _HYPHEN_PERSONAL_RE.search(text)
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}"


def _name_value_is_poisoned(value: str) -> bool:
    """True when a labeled 'value' is actually another field's printed
    label (``Forenames`` after ``SURNAME``, ``GEBOORTEDATUM`` after
    ``NATIONALITY``). A later token like ``NAMES`` is a legitimate
    given-name word and is not treated as poison.
    """
    tokens = re.findall(r"[A-Za-zÀ-ÿ']+", value or "")
    if not tokens:
        return True
    if sum(len(t) for t in tokens) < 3:
        return True
    label_fields = (
        "surname", "given_names", "full_name", "sex", "nationality",
        "date_of_birth", "date_of_issue", "date_of_expiry", "document_number",
    )
    first = tokens[0]
    if first.casefold() in _NAME_STOPWORDS:
        return True
    if any(ontology.match_label(first, field) for field in label_fields):
        return True
    for tok in tokens[1:]:
        if any(
            ontology.match_label(tok, field)
            for field in ("surname", "given_names", "nationality", "sex", "date_of_birth")
        ):
            return True
    return False


def _is_label_token(token: str) -> bool:
    if not token:
        return True
    if token.casefold() in _NAME_STOPWORDS:
        return True
    return any(
        ontology.match_label(token, field)
        for field in (
            "surname", "given_names", "full_name", "sex", "nationality",
            "date_of_birth", "date_of_issue", "date_of_expiry", "permit_type",
        )
    )


def _normalize_permit_type(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip(" /")
    return re.sub(r"\b([A-Z])\s+\.", r"\1.", cleaned)


def _plausible_permit_category(value: str) -> bool:
    """A permit *category* (``A. LIMITED RESIDENCE``), not the label
    leftover ``OF PERMIT`` after ``TYPE``.
    """
    cleaned = _normalize_permit_type(value)
    if not cleaned:
        return False
    folded = cleaned.casefold()
    if folded.startswith("of ") or folded in {"permit", "type", "titel", "soort"}:
        return False
    if "geldig" in folded or "expir" in folded:
        return False
    if ontology.match_label(cleaned, "permit_type"):
        return False
    if ontology.match_label(cleaned, "date_of_expiry"):
        return False
    if re.match(r"^[A-Z]\s*\.", cleaned):
        return True
    if re.fullmatch(r"[A-ZÀ-ÿ]{3,24}", cleaned):
        return False
    if _name_value_is_poisoned(cleaned.split()[0]):
        return False
    return bool(re.search(r"[A-Za-zÀ-ÿ]{4,}", cleaned))


def _extract_permit_type(text: str) -> str:
    pattern = _FIELD_PATTERNS.get("permit_type")
    if pattern is not None:
        for match in pattern.finditer(text):
            candidate = _normalize_permit_type(match.group(1))
            for raw in dates.date_tokens(candidate):
                idx = candidate.find(raw)
                if idx > 0:
                    candidate = candidate[:idx].strip()
                    break
            if _plausible_permit_category(candidate):
                return candidate
    return _value_after_label(text, "permit_type", _PERMIT_VALUE)


def _unlabeled_digit_card_number(text: str) -> str:
    """Unlabelled credential numbers: ``6191348 32`` (6–8 digits, space,
    2-digit group) or a 9-digit token that is not a compact calendar date.
    """
    from collections import Counter

    spaced: list[str] = []
    for match in _SPACED_DIGIT_DOC_RE.finditer(text):
        joined = match.group(1) + match.group(2)
        if _looks_like_printed_date(joined) or _looks_like_printed_date(match.group(1)):
            continue
        spaced.append(f"{match.group(1)} {match.group(2)}")
    if spaced:
        return Counter(spaced).most_common(1)[0][0]
    nines: list[str] = [
        token for token in re.findall(r"\b\d{9}\b", text)
        if not _looks_like_printed_date(token)
    ]
    if not nines:
        return ""
    token, count = Counter(nines).most_common(1)[0]
    if count >= 2 or len(set(nines)) == 1:
        return token
    return ""


def compact_identifier(value: str) -> str:
    return re.sub(r"[\s\-]+", "", value or "").casefold()


def identifiers_equivalent(left: str, right: str) -> bool:
    """Spaced ``6191348 32`` and glued ``619134832`` are the same read."""
    a, b = compact_identifier(left), compact_identifier(right)
    return bool(a) and a == b


def sweep_overcaptured(whole: str, sweep_value: str) -> bool:
    """True when the sweep swallowed the whole-text value plus label
    leftovers (``KASSA`` vs ``Forenames KASSA Sirak Dejene``).
    """
    if not whole or not sweep_value:
        return False
    if identifiers_equivalent(whole, sweep_value):
        return True
    whole_toks = whole.split()
    sweep_toks = sweep_value.split()
    if not whole_toks or len(sweep_toks) <= len(whole_toks):
        return False
    needle = [t.casefold() for t in whole_toks]
    hay = [t.casefold() for t in sweep_toks]
    found = False
    for i in range(len(hay) - len(needle) + 1):
        if hay[i: i + len(needle)] == needle:
            found = True
            break
    if not found:
        return False
    extra = [t for t in sweep_toks if t.casefold() not in {w.casefold() for w in whole_toks}]
    return bool(extra) and any(_is_label_token(t) for t in extra)


def _is_identity_label_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for field in ("sex", "nationality", "date_of_birth", "date_of_expiry", "permit_type"):
        if ontology.match_label(stripped, field):
            return True
    folded = stripped.casefold()
    return any(
        hint in folded
        for hint in (
            "geslacht", "nationaliteit", "geboorte", "geldig",
            "opmerking", "soort titel", "type of permit",
        )
    )


def _fill_stacked_names_after_compound_label(text: str, out: dict[str, Any]) -> None:
    """Combined header (surname aliases and given-name aliases on one
    printed line) then two stacked value lines — ICAO VIZ, not a
    country-specific layout.
    """
    from . import countries

    spans = ontology.find_label_spans(text, "full_name")
    dual: list[tuple[int, int]] = []
    for start, end in spans:
        phrase = text[start:end]
        tokens = re.findall(r"[A-Za-zÀ-ÿ]+", phrase)
        has_surname = any(ontology.match_label(tok, "surname") for tok in tokens)
        has_given = any(ontology.match_label(tok, "given_names") for tok in tokens)
        if has_surname and has_given:
            dual.append((start, end))
    if not dual:
        return
    _, end = dual[0]
    lines: list[str] = []
    for raw in text[end:].splitlines():
        line = raw.strip().strip("/")
        if not line:
            continue
        if _is_identity_label_line(line):
            break
        if dates.date_tokens(line):
            break
        tokens = [
            tok for tok in _NAME_TOKEN_RE.findall(line)
            if _keep_viz_name_token(tok, countries)
        ]
        if not tokens:
            continue
        lines.append(" ".join(tokens))
        if len(lines) >= 2:
            break
    if len(lines) < 2:
        return
    out["surname"] = lines[0]
    out["given_names"] = lines[1]
    out.pop("full_name", None)


def _strip_indexed_viz_markers(text: str) -> str:
    """EU identity cards print ``[a]`` / ``[b]`` beside VIZ rows.
    OCR often glues the marker onto the value (``[a]HELFE``, ``[aJHELFE``).
    """
    out = re.sub(r"\[\s*[a-zA-Z0-9]{1,2}\s*\]", " ", text or "")
    out = re.sub(r"\[\s*[a-zA-Z]\s*J?(?=[A-Z]{3,})", " ", out)
    return out


def _plausible_nationality_token(token: str) -> bool:
    """``No`` is English OCR noise, not Norway. 3-letter codes must be
    all-caps (SWE, ETH). Demonyms are longer (Svensk, Nederlandse).
    ``AND`` / ``THE`` are English glue from country headers, not ISO codes.
    """
    if len(token) < 3:
        return False
    if dates.month_number(token):
        return False
    if token.upper() in {"AND", "THE", "FOR", "ARE", "WAS", "NOT", "ANY", "YOU"}:
        return False
    if len(token) == 3:
        return token.isupper()
    return True


def _looks_like_date_ocr_junk(token: str) -> bool:
    """OCR of bilingual month dates (``SEP/SEP`` → ``SPSPH``, ``SEPSPH``)
    is not a given name. All-caps VIZ names always contain a vowel.
    """
    if not token or not token.isupper():
        return False
    if dates.month_number(token):
        return True
    letters = re.sub(r"[^A-Z]", "", token)
    if 6 <= len(letters) <= 8:
        if dates.month_number(letters[:3]) and dates.month_number(letters[3:6]):
            return True
    if 4 <= len(letters) <= 6 and not any(c in "AEIOUY" for c in letters):
        return True
    return False


def _keep_viz_name_token(token: str, countries: Any) -> bool:
    folded = token.casefold()
    if folded in _NAME_STOPWORDS:
        return False
    if _GLUED_SEX_NATIONALITY_RE.match(token):
        return False
    if token.isupper() and dates.month_number(token):
        return False
    if _looks_like_date_ocr_junk(token):
        return False
    if countries.resolve(token):
        return False
    return True


def _space_glued_allcaps_given(given: str, text: str) -> str:
    """``FASILHUNDE`` → ``FASIL HUNDE`` when a spaced variant exists."""
    if " " in given or not given.isupper() or not (8 <= len(given) <= 18):
        return given
    for match in re.finditer(r"\b([A-Z]{3,10})\s+([A-Z]{3,10})\b", text):
        glued = match.group(1) + match.group(2)
        if glued == given:
            return f"{match.group(1)} {match.group(2)}"
    return given


def _marks_a_real_date(raw: str) -> bool:
    """True when `raw` is plausibly a printed date, even one this project
    deliberately refuses to fully resolve (`slash_date_to_iso` returns ""
    for a genuinely ambiguous DD/MM-vs-MM/DD date -- correct for that
    function's own job, but `_text_before_first_date` only needs to know
    WHERE a date sits in the text, not what it means). Confirmed against
    a real UAE Resident Identity Card, all of whose dates (06/10/1998,
    02/01/2026, 01/01/2028) have both non-year components <=12 and so
    never resolve -- without this, the search never found a stopping
    point at all and swept the document's own header caption text into
    the name-candidate pool.
    """
    if dates.slash_date_to_iso(raw):
        return True
    parts = [p for p in re.split(r"[\s./-]+", raw.strip()) if p]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return False
    year_idx = next((i for i, p in enumerate(parts) if len(p) == 4), None)
    if year_idx is None:
        return False
    others = [int(p) for i, p in enumerate(parts) if i != year_idx]
    return len(others) == 2 and all(1 <= o <= 31 for o in others) and any(o <= 12 for o in others)


def _text_before_first_date(text: str) -> str:
    for raw in dates.date_tokens(text):
        if _marks_a_real_date(raw):
            idx = text.find(raw)
            if idx > 0:
                return text[:idx]
            break
    return text


def _harmonize_mixed_ids(candidates: list[str]) -> str:
    """Vote on mixed letter+digit identifiers; prefer a digit where
    OCR variants disagree on a known lookalike (5/S, 0/O, 1/I).
    """
    if not candidates:
        return ""
    from collections import Counter

    lengths = Counter(len(c) for c in candidates)
    target_len = lengths.most_common(1)[0][0]
    pool = [c.upper() for c in candidates if len(c) == target_len]
    if not pool:
        return ""
    out: list[str] = []
    for i in range(target_len):
        chars = [p[i] for p in pool]
        digits = [c for c in chars if c.isdigit()]
        letters = [c for c in chars if c.isalpha()]
        chosen = _pick_identifier_char(i, chars, digits, letters, pool)
        if chosen:
            out.append(chosen)
            continue
        out.append(Counter(chars).most_common(1)[0][0])
    return "".join(out)


def _pick_identifier_char(
    index: int,
    chars: list[str],
    digits: list[str],
    letters: list[str],
    pool: list[str],
) -> str:
    """5/S and 0/O: prefer the digit. Leading 1/I on a mixed ID: prefer I
    (ICAO document numbers are letter-led; ``1W5J5R3K3`` is ``IW5…``).
    """
    from collections import Counter

    if not digits or not letters:
        return ""
    lookalike_letters = set(letters) <= {"S", "O", "I", "B", "G", "Z"}
    pairs = {(d, ltr) for d in set(digits) for ltr in set(letters)}
    i_vs_1 = set(letters) <= {"I"} and set(digits) <= {"1"}
    if i_vs_1:
        if index == 0:
            return Counter(letters).most_common(1)[0][0]
        left = pool[0][index - 1] if index else ""
        right = pool[0][index + 1] if index + 1 < len(pool[0]) else ""
        if left.isalpha() and right.isalpha():
            return Counter(letters).most_common(1)[0][0]
        return Counter(digits).most_common(1)[0][0]
    if pairs <= _ID_CONFUSABLES or lookalike_letters:
        return Counter(digits).most_common(1)[0][0]
    return ""


def merge_confusable_identifiers(left: str, right: str) -> str:
    """If two mixed IDs differ only by 0/O, 1/I, 5/S, return one repaired
    reading. Empty when they are genuinely different strings.
    """
    a, b = (left or "").strip().upper(), (right or "").strip().upper()
    if not a or not b or len(a) != len(b):
        return ""
    for ca, cb in zip(a, b, strict=True):
        if ca == cb:
            continue
        pair = {(ca, cb), (cb, ca)}
        if pair.isdisjoint(_ID_CONFUSABLES):
            return ""
    return _harmonize_mixed_ids([left, right])


def _fill_unlabeled_identity_fields(
    text: str,
    out: dict[str, Any],
    *,
    document_number_labeled: bool = False,
    date_raws: dict[str, list[str]] | None = None,
) -> None:
    """Recover fields whose printed labels were too small/low-contrast
    for OCR, using value signatures only: bilingual month dates, mixed
    document numbers, ICAO VIZ stacked names (surname then given names),
    standalone demonyms. Never overwrites a labeled read.

    `document_number_labeled` is the caller's own record of whether
    `out["document_number"]` (if present) came from an actual label match
    earlier in `extract_generic_fields`, rather than this function's own
    unlabeled signature guess below -- a purely-numeric value is not on
    its own evidence that the label didn't find the real thing. Confirmed
    against a real Australian (ACT) driver licence, whose genuinely
    labeled "Licence No. 5520875" was being silently replaced by an
    unrelated unlabeled vertical control-number printed elsewhere on the
    card, purely because "5520875" is all digits -- violating this
    function's own "never overwrites a labeled read" rule.

    `date_raws` (from the label-anchored search earlier in
    `extract_generic_fields`) marks a date field as "found but refused to
    guess" even when it never made it into `out` -- an ambiguous DD/MM-
    vs-MM/DD date the label search agreed on across every OCR pass is
    still stronger evidence than this function's own blind whole-text
    date sweep below, which has no idea WHICH field a stray date belongs
    to and no label to anchor it. Confirmed against a real UAE Resident
    Identity Card: one heavily garbled OCR pass misread the issuing date
    "02/01/2026" (label-agreed 3 times, but ambiguous) as "01/01/2020" --
    day==month, so unambiguous -- and the whole-text sweep confidently
    picked that WRONG value over honestly leaving the field unresolved.
    """
    from . import countries

    ambiguous_date_fields = {f for f, raws in (date_raws or {}).items() if raws}
    for field, iso in dates.assign_life_cycle_dates(dates.unique_resolved_dates(text)).items():
        if field in ambiguous_date_fields:
            continue
        out.setdefault(field, iso)
    dates.repair_duplicate_life_cycle(out, dates.unique_resolved_dates(text))

    if not document_number_labeled and (
        "document_number" not in out or (out.get("document_number") or "").isdigit()
    ):
        mixed = _MIXED_DOC_ID_RE.findall(text)
        mixed = [m for m in mixed if not _looks_like_printed_date(m)]
        chosen = _harmonize_mixed_ids(mixed)
        if chosen:
            previous = out.get("document_number", "")
            if previous.isdigit() and "personal_number" not in out:
                out["personal_number"] = previous
            out["document_number"] = chosen

    segmented = _SEGMENTED_NATIONAL_ID_RE.search(text)
    if segmented:
        out["document_number"] = re.sub(r"\s+", "", segmented.group(1).replace("–", "-"))

    digit_id = _unlabeled_digit_card_number(text)
    if digit_id:
        current = str(out.get("document_number", ""))
        if not current or _looks_like_printed_date(current) or current.isdigit():
            if not current or _looks_like_printed_date(current) or not re.search(r"[A-Za-z]", current):
                out["document_number"] = digit_id

    _fill_eu_numbered_viz(text, out)

    hyphenated = _hyphenated_personal_id(text)
    if hyphenated:
        out["personal_number"] = hyphenated
    elif "personal_number" not in out:
        skip = _can_serials(text)
        glued_doc = re.sub(r"\s+", "", str(out.get("document_number", "")))
        for m in _DIGIT_PERSONAL_RE.finditer(text):
            token = m.group(1)
            if token == glued_doc or token in skip:
                continue
            if _looks_like_printed_date(token):
                continue
            if len(token) == 4:
                continue
            out["personal_number"] = token
            break

    if "sex" not in out:
        sex_m = re.search(
            r"\b(V\s*/\s*F|M\s*/\s*M|M\s*[./]\s*M|M\s*/\s*F|W\s*/\s*M|Male|Female)\b",
            text,
            re.I,
        )
        if sex_m:
            sex = _normalize_sex(sex_m.group(1).replace(".", "/"))
            if sex:
                out["sex"] = sex
        if "sex" not in out:
            glued = re.search(r"\b(MM|VF|WM|MF)[A-Z]{3,}\b", text)
            if glued:
                mapped = {"MM": "M", "WM": "M", "VF": "F", "MF": "F"}
                out["sex"] = mapped[glued.group(1).upper()]

    if "nationality" not in out:
        from .countries import _fold

        header_codes = countries.header_iso_state_codes(text)
        issuing_header = header_codes[0] if header_codes else ""
        body = "\n".join([ln for ln in text.splitlines() if ln.strip()][5:])
        other_iso = False
        for match in re.finditer(r"\b([A-Z]{3})\b", body):
            later = countries.resolve_nationality(match.group(1))
            if later and later.code != issuing_header:
                other_iso = True
                break
        votes: dict[str, tuple[int, int, int]] = {}
        for match in _NAME_TOKEN_RE.finditer(text):
            token = match.group(1)
            if not _plausible_nationality_token(token):
                continue
            if token.upper() == "CAN" and re.search(r"\bCAN\s+\d{4,8}\b", text, re.I):
                continue
            if other_iso and token.upper() == issuing_header:
                continue
            # A code tucked in parentheses right after a place name is a
            # birth-country annotation ("ADEKENEY (ETH) 12.06.1984"), the
            # same "PLACE (COUNTRY) DATE" convention this module's own
            # place_of_birth parsing already stops before -- not a printed
            # nationality statement. Confirmed against a real Italian
            # Carta d'Identita, whose only country-code-shaped text in the
            # OCR sweep is this birth-country parenthetical (the card's
            # actual printed nationality, "ITA", sits under its own
            # CITTADINANZA/NATIONALITY label and is caught by the labeled
            # pattern above when present); without this guard the fallback
            # sweep below picked the birth country as if it were the
            # holder's nationality.
            if (
                re.search(r"\(\s*$", text[: match.start()])
                and re.match(r"^\s*\)", text[match.end():])
            ):
                continue
            entity = countries.resolve_nationality(token)
            if not entity or not entity.nationality:
                continue
            if _fold(token) == _fold(entity.country_name) and _fold(token) != _fold(entity.nationality):
                continue
            rank = 2 if token.upper() == entity.code else (
                1 if _fold(token) == _fold(entity.nationality) else 0
            )
            prev = votes.get(token, (0, 0, 0))
            votes[token] = (rank, prev[1] + 1, len(token))
        if votes:
            keys = list(votes)
            for short in keys:
                for long in keys:
                    if (
                        short != long
                        and long.casefold().startswith(short.casefold())
                        and len(long) > len(short)
                    ):
                        votes[long] = (-1, 0, 0)
            out["nationality"] = max(votes, key=lambda k: votes[k])

    _apply_viz_names(text, out)
    _fill_stacked_names_after_compound_label(text, out)

    if not out.get("date_of_birth") and out.get("personal_number"):
        compact = re.sub(r"\D", "", str(out["personal_number"]))[:6]
        iso = dates.iso_from_yymmdd(compact)
        if iso:
            out["date_of_birth"] = iso

    if out.get("date_of_birth") and not out.get("date_of_expiry"):
        later_dates = [
            d for d in dates.unique_resolved_dates(text)
            if d > out["date_of_birth"]
        ]
        if len(later_dates) == 1:
            out["date_of_expiry"] = later_dates[0]

    if "place_of_birth" not in out:
        place_m = _PLACE_BEFORE_DATE_RE.search(text) or _PLACE_AFTER_DATE_RE.search(text)
        if place_m:
            place = place_m.group(1).strip()
            # Check every TOKEN, not just the whole phrase -- a date
            # immediately followed by the NEXT field's own multi-word
            # label ("Expiry Date") is itself Title-Case and satisfies
            # `_PLACE_AFTER_DATE_RE`'s shape just as well as a genuine
            # place name would. Confirmed against a real UAE Resident
            # Identity Card, whose "Issuing Date ...\nExpiry Date ..."
            # rows sit back-to-back with no place-of-birth field at all
            # on this document family.
            tokens_are_clean = all(
                tok.casefold() not in _NAME_STOPWORDS for tok in place.split()
            )
            if tokens_are_clean and not dates.month_number(place):
                out["place_of_birth"] = place


def _viz_name_sequence(text: str) -> list[str]:
    from . import countries

    # A line carrying TWO OR MORE known institutional/bureaucratic caption
    # words (already-excluded stopwords like "authority", "federal",
    # "identity") is disqualified WHOLESALE, not just those words --
    # confirmed against a real UAE Resident Identity Card, whose header
    # line ("FEDERAL AUTHORITY FOR IDENTITY & CITIZENSHIP, CUSTOMS AND
    # PORT SECURITY") has three OTHER all-caps words on the SAME line
    # that aren't themselves registered stopwords, and those alone
    # outvoted the card's genuine (Title Case) name below it. Requiring
    # 2+ (not just 1) keeps an ordinary "Name Henok Feseha Tilahun" line
    # safe -- its single stopword ("Name") is already excluded on its own
    # by the per-token filter below, without throwing out the rest of
    # the line too.
    text = "\n".join(
        line for line in text.splitlines()
        if sum(tok.casefold() in _NAME_STOPWORDS for tok in _NAME_TOKEN_RE.findall(line)) < 2
    )
    prefix = _text_before_first_date(text)
    counts: dict[str, int] = {}
    for token in _NAME_TOKEN_RE.findall(text):
        if not _keep_viz_name_token(token, countries):
            continue
        counts[token] = counts.get(token, 0) + 1
    ordered: list[str] = []
    seen: set[str] = set()
    for token in _NAME_TOKEN_RE.findall(prefix):
        if token not in counts or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    min_votes = 2 if counts and max(counts.values()) >= 2 else 1
    names = [n for n in ordered if counts[n] >= min_votes]
    caps = [n for n in names if n.isupper()]
    if len(caps) >= 2:
        names = caps
    if (
        len(names) >= 4
        and names[1].isupper()
        and names[1] == names[2] + names[3]
    ):
        names = [names[0], names[2], names[3]]
    if len(names) > 4:
        names = names[:4]
    return names


def _unlabeled_address_after(text: str, end: int, out: dict[str, Any]) -> None:
    """The 1-2 lines directly below an unlabeled name line, when nothing
    ELSE has already claimed an address -- confirmed against a real
    Australian (ACT) driver licence, whose street and suburb/state/
    postcode lines carry no "Address" label anywhere on the card at all.
    Deliberately shape-agnostic (a street-number-then-name line, a
    suburb-state-postcode line, a multi-level admin line, ... vary far
    too much across countries to enumerate) -- takes whatever unlabeled
    text sits between the name and the next recognized field, the same
    "stop at the next known label" boundary this file already applies to
    the labeled "Address:" case just above.
    """
    if "address" in out:
        return
    from .ocr_engine import ocr_space_for_printed_word_gap

    lines = [ln.strip() for ln in text[end:].splitlines() if ln.strip()][:3]
    candidates: list[str] = []
    for line in lines:
        if line.casefold() in _NAME_STOPWORDS:
            break
        if any(ontology.find_label_spans(line, field) for field in _FIELD_PATTERNS):
            break
        candidates.append(line)
        if len(candidates) == 2:
            break
    if candidates:
        out["address"] = ocr_space_for_printed_word_gap(", ".join(candidates))


def _apply_viz_names(text: str, out: dict[str, Any]) -> None:
    comma = _COMMA_NAME_RE.search(text)
    if comma and comma.group(1).casefold() not in _NAME_STOPWORDS:
        out["surname"] = comma.group(1)
        out["given_names"] = comma.group(2)
        _unlabeled_address_after(text, comma.end(), out)
        return
    if "surname" not in out and "given_names" not in out and "full_name" not in out:
        for line_m in _ALLCAPS_TITLECASE_NAME_LINE_RE.finditer(text):
            candidate_tokens = [line_m.group(1)] + line_m.group(2).split()
            if all(tok.casefold() not in _NAME_STOPWORDS for tok in candidate_tokens):
                out["surname"] = line_m.group(1)
                out["given_names"] = line_m.group(2)
                _unlabeled_address_after(text, line_m.end(), out)
                return
    names = _viz_name_sequence(text)
    if len(names) < 2:
        return
    # Every candidate token sitting on ONE physical line (as opposed to
    # the ICAO-VIZ convention's own stacked "surname line, then given-
    # names line") means this document never signaled a surname/given
    # split at all -- only a single, undifferentiated name string.
    # Splitting it as "position 0 = surname" would invent a distinction
    # the card itself never made (CLAUDE.md's own "parse by the
    # document's own labels, never by position" rule) -- confirmed
    # against a real UAE Resident Identity Card printing "Name Henok
    # Feseha Tilahun" (an Ethiopian patronymic name, no family surname at
    # all) on one line with no comma; treating "Henok" as the surname
    # reordered full_name into a wrong "Feseha Tilahun Henok" once
    # composed given-then-surname downstream.
    if (
        "surname" not in out and "given_names" not in out and "full_name" not in out
        and any(all(n in _NAME_TOKEN_RE.findall(line) for n in names) for line in text.splitlines())
    ):
        out["full_name"] = " ".join(names)
        return
    viz_surname = names[0]
    viz_given = _space_glued_allcaps_given(" ".join(names[1:]), text)
    labeled_s = out.get("surname", "")
    labeled_g = out.get("given_names", "")
    all_caps_viz = viz_surname.isupper() and viz_given.isupper()
    if all_caps_viz:
        if not labeled_s or not labeled_s.isupper():
            out["surname"] = viz_surname
        if not labeled_g or not labeled_g.isupper():
            out["given_names"] = viz_given
        return
    if "surname" not in out and "given_names" not in out and "full_name" not in out:
        out["surname"] = viz_surname
        out["given_names"] = viz_given


def guess_document_category(text: str) -> str:
    """Best-effort document-type inference from the OCR'd text alone,
    reached only once no structural gate (MRZ, AAMVA barcode, Ethiopia
    ID QR) already classified the document -- see `_DOCUMENT_CATEGORY_
    KEYWORDS`'s comment for why this is safe to keep generic rather than
    per-country. Never invents a category from layout/position, only
    from text the document itself printed. Returns "" (never "unknown"
    itself -- that string is the caller's own explicit default) when
    nothing matches.
    """
    normalized = text.casefold()
    vocabulary = ontology.document_vocabulary() or _DOCUMENT_CATEGORY_KEYWORDS
    for category, keywords in vocabulary:
        if any(keyword.casefold() in normalized for keyword in keywords):
            return category
    return ""


def extract_fields_via_sweep(image_path: str) -> dict[str, Any]:
    """Field-first alternative to `ocr_full_text` + `extract_generic_
    fields`'s whole-text-blob regex search -- see idx/sweep.py's module
    docstring. Uses genuine per-token POSITIONS (label anchor: right,
    then below) instead of a bounded character-count window over one
    long joined string, so a value sitting further down the page than
    that window's reach can still be found by real geometry.

    Additive, not a replacement: `pipeline._generic_ocr_fallback` calls
    this ALONGSIDE the whole-text approach, merging results with the
    whole-text search's own findings taking priority (it's the proven,
    tested path) -- this only fills fields that search missed. Operates
    on the raw image directly (no `preprocess_variants` treatment, unlike
    `ocr_full_text`) -- a real, documented scope limitation for now, not
    an oversight (see docs/ARCHITECTURE.md).
    """
    from . import layout, scoring, sweep

    tokens = sweep.tokens_from_image(image_path)
    page = layout.build_page(tokens)
    out: dict[str, Any] = {}
    for field, value_pattern in _VALUE_PATTERNS.items():
        allow_unlabeled = (
            ontology.field_profile(field).allow_unlabeled
            and field not in _NO_UNLABELED_SIGNATURE
        )
        candidates = sweep.locate(field, tokens, value_pattern, allow_unlabeled=allow_unlabeled)
        if field == "document_number":
            candidates = [
                c for c in candidates
                if c.source == "label_anchor" or not _looks_like_printed_date(c.value)
            ]
        if not candidates:
            continue
        # Score every candidate (label proximity, same row, type, competing
        # labels, date plausibility) instead of taking the first / lowest
        # tier. A future date next to ISS must not win date_of_birth just
        # because it was seen first.
        best = scoring.pick_best(field, candidates, page)
        if best is None:
            continue
        if best.source != "label_anchor" and best.score < 15:
            continue
        value: Any = best.value
        if field in _DATE_FIELDS:
            labeled_raws = [c.value for c in candidates if c.source == "label_anchor"]
            value = dates.resolve_printed_dates(labeled_raws or [best.value])
        elif field == "sex":
            value = _normalize_sex(value)
        elif field in ("surname", "given_names", "full_name"):
            if _name_value_is_poisoned(str(value)):
                continue
            if any(_is_label_token(tok) for tok in str(value).split()):
                continue
        elif field == "permit_type":
            value = _normalize_permit_type(str(value))
            if not _plausible_permit_category(value):
                continue
        elif field == "nationality" and (
            value.casefold() in _NAME_STOPWORDS
            or "identitet" in value.casefold()
            or "identity" in value.casefold()
        ):
            continue
        if value:
            out[field] = value
    page_text = "\n".join(t.text for t in tokens)
    dates.repair_duplicate_life_cycle(out, dates.unique_resolved_dates(page_text))
    return out

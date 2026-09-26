"""
Ethiopian Digital ID Card ("Fayda") template.

Unlike a passport/MRZ-bearing national ID, this card carries no MRZ at
all. Instead it carries two independent machine-readable sources that are
far more trustworthy than OCR:

  - A QR code on the back whose payload is a plaintext, colon-delimited
    structure: `...:DLT:<full name>:<4 unlabeled single-char tokens>:
    <16-digit FAN>:<1 unlabeled token>:<Ethiopian-calendar DOB display>:
    SIGN:<signature blob>`. Confirmed against a real card during
    development. The unlabeled single-character tokens between the known
    fields have no documented meaning available to this parser and are
    deliberately left unparsed rather than guessed at -- same discipline
    as `mrz_td3.py` not inventing meaning for personal_number's internal
    filler. The `SIGN` block is a cryptographic signature; verifying it
    would need the issuer's public key (not available), so parsing simply
    stops there.
  - A Code128 barcode on the front encoding the same FAN in isolation --
    an independent cross-check against the QR's FAN with no OCR involved
    on either side.

Both are optional inputs (either image alone still yields a partial
result): the front alone gives the FAN (barcode) plus whatever OCR can
read; the back alone gives everything the QR encodes. Providing both
enables three independent cross-checks, recorded in `checks` (surfaced as
`verification_checks` on the final output):
  - `fan_qr_barcode_match` -- QR's FAN vs. the front barcode's FAN. No OCR
    on either side; the strongest signal this template has.
  - `full_name_qr_ocr_match` -- QR's full_name vs. front OCR's best-effort
    name read. Weaker (OCR is one of the two sides), but still real
    corroboration when both land on the same reading.
  - `dob_ethiopian_display_qr_ocr_match` -- QR's Ethiopian-calendar DOB
    display vs. the same value read from the front's dual-date printing.
    Informational only (date_of_birth itself always comes from the
    Gregorian reading, never this display value) -- see below.
Three more checks need only a SINGLE image (the front) -- see
`_dual_calendar_years_plausible` below -- `{field}_dual_calendar_
plausible` for date_of_birth/date_of_issue/date_of_expiry: whether that
field's own printed Ethiopian and Gregorian years are a real calendar
pair (exactly 7 or 8 years apart) rather than an OCR error on one side.

No Ethiopian-to-Gregorian calendar conversion is attempted anywhere in
this module -- confirmed across TWO real card generations that each date
field already prints both calendars, so parsing only ever has to pick the
right one, never compute it:
  - Older ("FAN") generation: alphabetic Gregorian month, e.g.
    "19/03/1989 | 1996/Nov/28" -- the plain DD/MM/YYYY (or YYYY/MM/DD)
    number is the Ethiopian-calendar display, the YYYY/<month-abbrev>/DD
    form (a Latin month abbreviation) is the true Gregorian date. See
    `_dates_from_one_pass`/`_ALPHA_MONTH_DATE_RE`.
  - Newer ("FCN") generation: an explicit two-letter calendar suffix
    directly after each plain-numeric date instead of a month name, e.g.
    "2024/07/03GC" / "2016/10/26EC". date_of_birth on this generation is a
    further special case, printing neither an alphabetic month NOR a
    suffix at all -- just two stacked plain-numeric dates in the same
    Ethiopian-then-Gregorian print order every other field and generation
    uses. See `_numeric_calendar_dates_from_one_pass`/`_resolve_gc_ec_pair`.
Never assume a fixed +7 or +8 year offset between the two calendars to
COMPUTE one from the other -- the exact offset depends on which side of
the Ethiopian new year (~Sept 11) a date falls on, confirmed by cross-
referencing a real card against its holder's checksum-verified passport
MRZ (see mrz_td3.py). `_dual_calendar_years_plausible` uses this only as a
weak, free CONSISTENCY check (the two years must be exactly 7 or 8 apart,
whichever it is) between a field's own printed pair, never to derive a
value that wasn't printed. Every Gregorian date is passed through exactly
as resolved from what's actually printed.

"FAN" (older generation) and "FCN" (newer) are the same concept -- the
front-of-card document number -- differing only by which literal label a
given card prints; `EthiopiaIDResult.fan` and `.card_generation` both
still populate correctly regardless of which one this specific card uses.

Date of Issue is a special case on BOTH generations: unlike DOB/expiry, it
prints rotated ~90 degrees along the card's left edge rather than inline
with the other date fields, so the normal-orientation OCR passes above
never see it. See `_issue_date_from_sidebar_ocr` for the free/offline
crop+rotate+OCR fallback this module tries (against both date formats)
before giving up and asking the caller to set GEMINI_API_KEY (which reads
the rotated text directly off the pixels, still the more reliable option
when available).

Some fields the card prints are explicitly qualified by the issuer as
holder-asserted, not independently verified -- confirmed on a real card:
"Nationality (Self Declared)". `nationality_qualifier` carries this
through structurally (see schema.py's own docstring for why), rather than
only as a warning a caller could miss.
"""

from __future__ import annotations

from typing import Any, Optional

import concurrent.futures
import re

from PIL import Image
from dataclasses import dataclass, field

import cv2

from .barcode import decode_barcodes
from .preprocess import enhance_contrast, load_image, preprocess_variants, upscale_if_small

_QR_MARKER = ":DLT:"
_FAN_RE = re.compile(r"\b\d{15,16}\b")
_ETHIOPIAN_DATE_RE = re.compile(r"\b\d{4}/\d{1,2}/\d{1,2}\b")

# "national\s+id" alone used to be enough on its own -- but that phrase is
# generic to essentially every country's national ID card, not specific
# to this one, so a bare OCR match on it (no QR/barcode decoded, the only
# case this header check is even consulted for) risked misidentifying a
# completely different country's card as Ethiopian and reporting a
# result built almost entirely from unvalidated OCR. Requiring "ethiopian"
# to appear ANYWHERE alongside an ID-label phrase keeps the real card's
# actual header ("Ethiopian Digital ID Card" / "... National ID",
# confirmed against a real card) matching, while a foreign card's own
# "National ID" text can no longer satisfy this alone.
_ETHIOPIAN_WORD_RE = re.compile(r"\bethiopian\b", re.IGNORECASE)
_HEADER_RE = re.compile(r"digital\s+id|national\s+id", re.IGNORECASE)
_SEX_RE = re.compile(r"\b(Male|Female)\b", re.IGNORECASE)
# Printed grouped into 4-digit clusters on a real card ("3726 0857 3503"),
# not always as one contiguous run -- `[\d ]{10,}` tolerates the internal
# spaces; the digit-count checks at the call site strip them back out
# before applying the "16 digits total = FAN, not FIN" length rule.
_FIN_DIGIT_RUN_RE = re.compile(r"\d[\d ]{9,}\d")
_PHONE_RE = re.compile(r"\b0\d{9}\b")
# Confirmed printed verbatim on a real card, directly under the Nationality
# field ("Nationality (Self Declared)") -- the issuer's own qualifier that
# this specific field is asserted by the holder, not independently
# verified. See schema.py's `nationality_qualifier` docstring for why this
# is carried through structurally rather than only as a warning.
_SELF_DECLARED_RE = re.compile(r"self[\s-]?declared", re.IGNORECASE)
# "FAN" and "FCN" are the same concept, differing only by card generation
# (CLAUDE.md's Axis 1 evidence) -- recording which literal label a specific
# card actually prints is free, real metadata about which generation
# produced this read. Word-bounded so it never matches inside an unrelated
# word.
_CARD_GENERATION_LABEL_RE = re.compile(r"\b(FAN|FCN)\b")
_NAME_BLOCKLIST = re.compile(
    r"ethiopian|digital|card|national|full|name|date|birth|expiry|issue|male|female|sex",
    re.IGNORECASE,
)
_NAME_LINE_RE = re.compile(r"^[A-Za-z]+(?:\s+[A-Za-z]+){1,4}$")
# A properly printed Latin name is consistently title-case per word. This is
# a stronger signal than raw OCR vote count for picking between candidate
# readings -- see `_pick_name` for why.
_NAME_TITLE_CASE_RE = re.compile(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,4}$")

# Layout modes beyond tesseract's default ("assume a uniform block of text")
# that empirically recover fields the default mode gets wrong on this
# card's low-contrast bilingual printing -- confirmed against a real photo
# during development: the default pass alone misread "Mesay" as "lesay" and
# a digit of the printed expiry year, while adding these two recovered the
# correct reading as the majority across all passes. Read `_parse_front_ocr`
# for how the votes are combined.
_FRONT_OCR_CONFIGS = ("", "--psm 6", "--psm 11")


@dataclass
class EthiopiaIDResult:
    full_name: str = ""
    fan: str = ""              # front-of-card document number ("card number")
    fin: str = ""               # back-of-card personal identifier
    dob_ethiopian_display: str = ""   # as printed on the QR/card, no conversion
    dob_gregorian: str = ""
    sex: str = ""
    date_of_issue: str = ""
    # As printed on the card (no conversion) -- see `date_of_issue_ethiopian_
    # display`'s docstring note on `dob_ethiopian_display` above for why this
    # is never used as a Gregorian value in its own right.
    date_of_issue_ethiopian_display: str = ""
    # True when date_of_issue came from the rotated-sidebar OCR fallback
    # rather than the normal-orientation front OCR pass -- see
    # `_issue_date_from_sidebar_ocr`. Lets callers flag it as extra-best-effort.
    date_of_issue_from_sidebar_ocr: bool = False
    date_of_expiry: str = ""
    date_of_expiry_ethiopian_display: str = ""
    phone_number: str = ""
    region: str = ""
    zone: str = ""
    woreda: str = ""
    # "self_declared" when the card itself prints that qualifier next to
    # nationality (confirmed on a real card generation: "Nationality (Self
    # Declared)") -- never asserted otherwise. See schema.py's
    # `nationality_qualifier` docstring for why this is carried through
    # rather than only surfaced as a warning: a caller relying on
    # nationality for KYC purposes needs this distinction structurally,
    # not buried in warning text.
    nationality_qualifier: str = ""
    # Which literal label this card generation prints for its front
    # document number -- "FAN" and "FCN" are the same concept (CLAUDE.md's
    # own Axis 1 evidence: differ only by card generation), but recording
    # which one was actually seen is a free, real signal for anything that
    # cares which generation produced a given read.
    card_generation: str = ""
    checks: dict[str, Any] = field(default_factory=dict[str, Any])
    raw_qr_present: bool = False
    raw_barcode_present: bool = False


def _parse_qr_payload(text: str) -> dict[str, Any]:
    """Pull out only the fields this parser can positively identify by
    shape (name, FAN, Ethiopian-display date) -- see module docstring for
    why the remaining tokens are left alone.
    """
    idx = text.find(_QR_MARKER)
    if idx == -1:
        return {}
    tail = text[idx + 1:]
    sign_idx = tail.find(":SIGN:")
    if sign_idx != -1:
        tail = tail[:sign_idx]
    segments = tail.split(":")[1:]  # drop the leading "DLT" tag itself

    out = {"full_name": segments[0].strip() if segments else ""}
    for seg in segments[1:]:
        if not out.get("fan") and _FAN_RE.fullmatch(seg):
            out["fan"] = seg
        elif not out.get("dob_ethiopian_display") and _ETHIOPIAN_DATE_RE.fullmatch(seg):
            out["dob_ethiopian_display"] = seg
    return out


def _ocr_texts(image_path: str, configs: tuple[str, ...] = ("",)) -> list[str]:
    """OCR every preprocessing variant under every requested tesseract
    layout config, returning each pass's text SEPARATELY (not merged) --
    callers that need to classify/vote per-pass (`_parse_front_ocr`) need
    that separation; callers that just want to regex-search everything at
    once (`_parse_back_ocr`, `looks_like_ethiopian_digital_id`) can still
    join the list themselves. This template's fields are best-effort/
    cross-check only (the QR/barcode are the trustworthy sources), so it's
    worth trying every variant/config combination rather than picking one,
    unlike the MRZ path where a wrong variant is caught by checksum and
    can be discarded cheaply.

    Every (variant, config) combination is an independent tesseract OS
    subprocess (`pytesseract.run_tesseract` uses `subprocess.Popen`), run
    through a thread pool for genuine wall-clock parallelism -- the same
    pattern as `mrz_ocr.find_mrz_candidates` and `aamva._ocr_texts`, and
    it depends on the same fix: `preprocess.py` caps tesseract's own
    internal OpenMP threading to 1 at import time, since running several
    tesseract subprocesses concurrently WITHOUT that cap oversubscribes
    the CPU and is slower than sequential, not faster (confirmed
    empirically on this exact pattern elsewhere in this project).
    """
    from .ocr_engine import ocr_text, using_rapidocr

    # A `config` string only changes tesseract's own page-segmentation
    # behavior -- RapidOCR ignores it entirely (see ocr_engine.py's module
    # docstring), so once RapidOCR is the active engine (drop-in single-
    # winner mode, confirmed with the user), iterating several configs
    # against the SAME variant would just OCR one image N identical times
    # and inflate its vote weight in `_parse_front_ocr`'s per-pass
    # majority vote, rather than genuinely sampling N different readings
    # the way it does under tesseract.
    effective_configs = configs[:1] if using_rapidocr() else configs
    tasks = [
        (variant, config)
        for variant in preprocess_variants(image_path).values()
        for config in effective_configs
    ]

    def _run(task: tuple[Image.Image, str]) -> Optional[str]:
        variant, config = task
        return ocr_text(variant, tesseract_config=config)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
        results = list(executor.map(_run, tasks))
    return [r for r in results if r is not None]


def _ocr_text(image_path: str, *, sparse: bool = False) -> str:
    """Single-string convenience wrapper around `_ocr_texts` for callers
    that just want to regex-search everything at once. `sparse=True` also
    runs tesseract's "sparse text" layout mode (--psm 11), which recovers
    isolated, oddly-placed text (e.g. the back-of-card phone number,
    confirmed against a real card) that the default "assume a uniform
    block" mode skips entirely.
    """
    configs = ("", "--psm 11") if sparse else ("",)
    return "\n".join(_ocr_texts(image_path, configs))


def _name_candidate(text: str) -> str:
    """The first name-shaped line in a single OCR pass's text (or "").
    Deliberately not the final answer on its own -- `_pick_name` combines
    one of these per pass across every pass to settle on a reading.
    """
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 4 or _NAME_BLOCKLIST.search(line):
            continue
        if _NAME_LINE_RE.match(line):
            return line
    return ""


def _pick_name(votes: dict[str, Any]) -> str:
    """Pick the best full_name reading from {candidate: vote_count}.

    Raw vote count alone isn't enough: a single-character OCR misread of
    the true capital letter (e.g. "Mesay" -> "lesay") can tie or even beat
    the correct reading in vote count if it happens on more preprocessing
    variants (confirmed against a real card during development -- "lesay
    Kebbede Lemma" and "Mesay Kebbede Lemma" tied 2-2 across OCR passes).
    A properly printed Latin name is consistently title-case per word,
    which is a structural fact about the document independent of how many
    OCR passes happened to agree -- so among the candidates, prefer ones
    that are fully title-case, and only fall back to the full pool (still
    ranked by vote count, then word count) if none are.
    """
    if not votes:
        return ""
    title_cased = {name: n for name, n in votes.items() if _NAME_TITLE_CASE_RE.match(name)}
    pool = title_cased or votes
    return max(pool, key=lambda name: (pool[name], name.count(" ")))


# OCR sometimes splits a 4-digit year with a stray space (e.g. "1 989") --
# collapse that before date matching rather than widening the date regex
# itself (which would risk swallowing unrelated adjacent digits).
_SPLIT_YEAR_RE = re.compile(r"(?<=\d) (?=\d{3}\b)")

# Every date on this card prints two calendars side by side, e.g.
# "19/03/1989 | 1996/Nov/28" -- the plain DD/MM/YYYY (or YYYY/MM/DD) number
# is the Ethiopian-calendar display, and the YYYY/<month-abbrev>/DD form is
# the true Gregorian date. Confirmed by cross-referencing a real card
# against its holder's checksum-verified passport MRZ: the numeric side is
# consistently exactly 7 years behind the alphabetic side, matching the
# real Ethiopian/Gregorian calendar offset for dates after the Ethiopian
# new year (~Sept 11) -- see module docstring. So only the alphabetic-month
# form is trustworthy as date_of_birth/date_of_issue/date_of_expiry; the
# plain numeric form is never used as a Gregorian value.
# The month group's first character also accepts a literal '0' -- confirmed
# against a real card's rotated sidebar OCR reading "Oct" as "0ct" (the
# classic O/0 confusable, worse here since the sidebar text is read via a
# crop+90-degree-rotate rather than upright, which tesseract handles less
# reliably than normal text). The rest of the token must still be letters,
# so this never matches a pure numeric date. `_alpha_month_to_iso` below
# normalizes the leading '0' back to 'o' before the month lookup.
#
# The year group's leading edge is `(?<!\d)`, not `\b` -- confirmed
# necessary against the same real card's sidebar crop, where stray OCR
# noise (a fragment of the printed "Date of Issue" label itself, glued on
# with no separating space/newline the crop preserved, e.g. "ot2025/0ct20")
# left NO word-boundary before the year: a letter immediately followed by
# a digit is not a `\b` transition (both are "word" characters), so `\b`
# silently rejected an otherwise-good match. `(?<!\d)` only forbids the
# genuinely unsafe case -- matching partway into a LONGER digit run (e.g.
# "12025/Oct/20" as if the year were "2025") -- while still allowing a
# letter or start-of-string immediately before, which `\b` also allowed.
#
# The separator between month and day is optional (`[/-]?`, not `[/-]`) --
# confirmed necessary against the same sidebar crop's "0ct20" (should be
# "Oct/20"): a thin punctuation mark is exactly what a 90-degree-rotated
# low-resolution crop most often loses. Still bounded by the day group's
# own `\d{1,2}\b`, so this can't silently swallow a longer trailing digit
# run either.
#
# The month group's minimum length is 2 characters total, not 3 -- the
# SAME real card's sidebar OCR sometimes drops the month abbreviation's
# third letter entirely ("Oct" -> "Oc"/"0c"), consistently enough across
# repeated OCR passes of the identical crop that raising resolution/PSM
# alone didn't fully fix it. A 2-character match is only ever ACCEPTED
# by `_alpha_month_to_iso` below when it's an unambiguous prefix of
# exactly one month abbreviation ("oc" -> October is the only fit) --
# "ma" (mar/may) and "ju" (jun/jul) are genuinely ambiguous at 2 letters
# and are never guessed. The regex itself just needs to be loose enough to
# capture the token; the lookup stays strict.
_ALPHA_MONTH_DATE_RE = re.compile(r"(?<!\d)(\d{4})[/-]([A-Za-z0][A-Za-z]{1,8})[/-]?(\d{1,2})\b")
# The plain-numeric counterpart printed immediately alongside each
# alphabetic-month date (e.g. the "19/03/1989" in "19/03/1989 | 1996/Nov/28")
# -- only ever used to recover the Ethiopian-calendar DISPLAY value for
# cross-checking against the QR's own dob_ethiopian_display, never as a
# Gregorian value in its own right (see module docstring).
_NUMERIC_DATE_RE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b")
# Same shape, but WITHOUT the trailing word boundary -- used where a
# numeric date can be immediately glued to a following calendar-suffix
# LETTER with no separating space (e.g. "2024/07/03GC"), confirmed on a
# real card. Plain `\b` fails there: a digit and a letter are both "word"
# characters, so there's no boundary between them at all. Still guarded
# against swallowing part of a LONGER digit run via the negative lookahead
# (a genuine truncation risk `\b` protects against for the plain form
# above) -- just permissive about what non-digit character follows.
_NUMERIC_DATE_TOKEN_RE = re.compile(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}(?!\d)")
_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# 2-letter prefixes that identify exactly ONE month abbreviation above --
# "ma" (mar/may) and "ju" (jun/jul) are excluded, genuinely ambiguous at
# 2 letters. Used only as a fallback in `_alpha_month_to_iso` when the
# full 3-letter abbreviation isn't available -- see `_ALPHA_MONTH_DATE_RE`.
_UNAMBIGUOUS_MONTH_PREFIX = {
    abbr[:2]: month for abbr, month in _MONTH_ABBR.items()
    if sum(1 for a in _MONTH_ABBR if a[:2] == abbr[:2]) == 1
}
# How far back from an alphabetic-month date match to look for a nearby
# field label ("birth" / "issu" / "exp") when classifying which field it
# belongs to.
_LABEL_WINDOW = 40


def _alpha_month_to_iso(year: str, month_name: str, day: str) -> str:
    normalized = "o" + month_name[1:] if month_name[:1] == "0" else month_name
    month = _MONTH_ABBR.get(normalized[:3].lower())
    if month is None and len(normalized) == 2:
        month = _UNAMBIGUOUS_MONTH_PREFIX.get(normalized.lower())
    if not month or not day.isdigit():
        return ""
    return f"{year}-{month:02d}-{int(day):02d}"


def _paired_ethiopian_display(text: str, alpha_start: int) -> str:
    window = text[max(0, alpha_start - 20):alpha_start]
    matches = list(_NUMERIC_DATE_RE.finditer(window))
    return matches[-1].group(0) if matches else ""


def _dates_from_one_pass(text: str) -> dict[str, Any]:
    """Single-OCR-pass date extraction -- `_parse_front_ocr` calls this once
    per pass and votes across the results, same reasoning as `_name_candidate`.
    """
    out = {}
    normalized = _SPLIT_YEAR_RE.sub("", text)

    dob_iso = expiry_iso = issue_iso = None
    dob_display = expiry_display = issue_display = None
    unclassified = []
    for m in _ALPHA_MONTH_DATE_RE.finditer(normalized):
        iso = _alpha_month_to_iso(*m.groups())
        if not iso:
            continue
        window = normalized[max(0, m.start() - _LABEL_WINDOW):m.start()].lower()
        if "birth" in window and dob_iso is None:
            dob_iso = iso
            dob_display = _paired_ethiopian_display(normalized, m.start())
        elif "issu" in window and issue_iso is None:
            issue_iso = iso
            issue_display = _paired_ethiopian_display(normalized, m.start())
        elif re.search(r"exp", window) and expiry_iso is None:
            expiry_iso = iso
            # Same pairing this function already does for DOB -- confirmed
            # against a real card missing THIS half entirely (only DOB's
            # own Ethiopian-calendar display was ever surfaced, even
            # though the card prints one for expiry too, in the same
            # "EC | GC" layout).
            expiry_display = _paired_ethiopian_display(normalized, m.start())
        else:
            unclassified.append((iso, m.start()))

    # Positional fallback for whatever an OCR-mangled label couldn't
    # classify -- DOB's pair always prints before expiry's pair in this
    # card's fixed body layout. Issue is deliberately NOT guessed
    # positionally: it prints separately along the card's edge, not in
    # this reading order, so an unlabeled hit here is more likely noise.
    for iso, start in unclassified:
        if dob_iso is None:
            dob_iso = iso
            dob_display = _paired_ethiopian_display(normalized, start)
        elif expiry_iso is None:
            expiry_iso = iso
            expiry_display = _paired_ethiopian_display(normalized, start)

    if dob_iso:
        out["dob_gregorian"] = dob_iso
        if dob_display:
            out["dob_ethiopian_display"] = dob_display
    if expiry_iso:
        out["date_of_expiry"] = expiry_iso
        if expiry_display:
            out["date_of_expiry_ethiopian_display"] = expiry_display
    if issue_iso:
        out["date_of_issue"] = issue_iso
        if issue_display:
            out["date_of_issue_ethiopian_display"] = issue_display
    return out


# A newer card generation (confirmed against a real "FCN"-labeled card, as
# opposed to the older "FAN"-labeled one `_ALPHA_MONTH_DATE_RE` above was
# built against) prints an explicit calendar suffix directly after each
# numeric date instead of an alphabetic Gregorian month name -- e.g.
# "2024/07/03GC" for date_of_expiry, "2016/10/26EC" for date_of_issue.
# date_of_birth on this SAME generation is a further special case: it
# prints NEITHER an alphabetic month NOR a suffix at all, just two stacked
# plain-numeric dates with no distinguishing text whatsoever (confirmed:
# "24/11/1987" then "1995/07/31") -- Ethiopian first, Gregorian second, the
# same print order every other field on both generations already uses.
# `_numeric_calendar_dates_from_one_pass` below is tried as a FALLBACK,
# only for whatever `_dates_from_one_pass` didn't find -- see
# `_parse_front_ocr` -- so the older generation's alphabetic-month reading
# is never second-guessed when it already succeeded.
_CALENDAR_SUFFIX_RE = re.compile(r"^\s{0,3}(EC|GC)\b", re.IGNORECASE)


def _calendar_suffix_after(text: str, end: int) -> Optional[str]:
    m = _CALENDAR_SUFFIX_RE.match(text[end:end + 6])
    return m.group(1).upper() if m else None


def _resolve_gc_ec_pair(tokens: list[re.Match[str]], text: str) -> tuple[Optional[str], Optional[str]]:
    """Given every plain-numeric date token found in some scope of text,
    picks out the GC (Gregorian) and EC (Ethiopian) raw strings -- by
    explicit suffix when present, falling back to print-order (first =
    Ethiopian, second = Gregorian) only when NEITHER token carries a
    suffix at all, never a guess between a suffixed and an unsuffixed one.
    """
    gc_raw = ec_raw = None
    for m in tokens:
        suffix = _calendar_suffix_after(text, m.end())
        if suffix == "GC" and gc_raw is None:
            gc_raw = m.group(0)
        elif suffix == "EC" and ec_raw is None:
            ec_raw = m.group(0)

    if gc_raw is None and ec_raw is None and len(tokens) >= 2:
        ec_raw, gc_raw = tokens[0].group(0), tokens[1].group(0)
    elif gc_raw is None and ec_raw is not None:
        remaining = [m.group(0) for m in tokens if m.group(0) != ec_raw]
        if remaining:
            gc_raw = remaining[0]
    elif ec_raw is None and gc_raw is not None:
        remaining = [m.group(0) for m in tokens if m.group(0) != gc_raw]
        if remaining:
            ec_raw = remaining[0]

    return gc_raw, ec_raw


def _numeric_calendar_dates_from_one_pass(text: str) -> dict[str, Any]:
    """Fallback date extraction for the newer EC/GC-suffix card generation
    -- see the block comment above. Tried only for fields `_dates_from_one_
    pass` didn't already resolve via the older alphabetic-month format.
    """
    from .generic_ocr import _slash_date_to_iso

    normalized = _SPLIT_YEAR_RE.sub("", text)
    out: dict[str, Any] = {}

    for label_pattern, iso_key, display_key in (
        (r"birth", "dob_gregorian", "dob_ethiopian_display"),
        (r"issu", "date_of_issue", "date_of_issue_ethiopian_display"),
        (r"exp", "date_of_expiry", "date_of_expiry_ethiopian_display"),
    ):
        label_m = re.search(label_pattern, normalized, re.IGNORECASE)
        if not label_m:
            continue
        window = normalized[label_m.end():label_m.end() + 60]
        tokens = list(_NUMERIC_DATE_TOKEN_RE.finditer(window))
        if not tokens:
            continue

        gc_raw, ec_raw = _resolve_gc_ec_pair(tokens, window)
        if gc_raw:
            iso = _slash_date_to_iso(gc_raw)
            if iso:
                out[iso_key] = iso
        if ec_raw:
            out[display_key] = ec_raw

    return out


def _dual_calendar_years_plausible(ethiopian_display: str, gregorian_iso: str) -> Optional[bool]:
    """Weak-but-free consistency check between a field's own printed
    Ethiopian and Gregorian dates: the two calendars' years are always
    exactly 7 or 8 years apart (the exact offset depends on which side of
    the Ethiopian new year, ~Sept 11, a date falls on -- this module
    deliberately never computes the precise conversion, see its own
    docstring, but a genuine pair's years are never apart by anything
    OTHER than 7 or 8). Returns None when either side is missing or
    unparseable -- nothing to check, not a guessed verdict either way.
    """
    if not ethiopian_display or not gregorian_iso:
        return None
    ec_year_m = re.search(r"\d{4}", ethiopian_display)
    if not ec_year_m:
        return None
    try:
        gc_year = int(gregorian_iso[:4])
        ec_year = int(ec_year_m.group(0))
    except ValueError:
        return None
    return (gc_year - ec_year) in (7, 8)


def _parse_front_ocr(texts: list[str]) -> dict[str, Any]:
    """Extracts each field independently from EVERY OCR pass, then takes
    a majority vote per field, instead of trusting whichever pass happens
    to run first. This card's low-contrast bilingual printing makes single-
    character misreads common enough (confirmed against a real card: a
    digit of the printed expiry year came out wrong on some passes but not
    others) that "first reading found" is not a safe default here the way
    it can be for a clean scan -- see `_pick_name` for the sharpest example.
    """
    name_votes: dict[str, int] = {}
    sex_votes: dict[str, int] = {}
    dob_votes: dict[str, int] = {}
    dob_display_votes: dict[str, int] = {}
    expiry_votes: dict[str, int] = {}
    expiry_display_votes: dict[str, int] = {}
    issue_votes: dict[str, int] = {}
    issue_display_votes: dict[str, int] = {}
    nationality_qualifier_votes: dict[str, int] = {}
    card_generation_votes: dict[str, int] = {}

    for text in texts:
        name = _name_candidate(text)
        if name:
            name_votes[name] = name_votes.get(name, 0) + 1
        sex_m = _SEX_RE.search(text)
        if sex_m:
            # Normalized to the single-letter "M"/"F" code here -- matching
            # AAMVA's and every MRZ format's own convention for this field,
            # rather than keeping this template's printed English word
            # ("Male"/"Female") as the schema value. Confirmed necessary
            # against a real card once `idx/validation.py`'s universal
            # plausibility check started rejecting "Male" outright (it only
            # ever accepts "M"/"F", the value every OTHER document family
            # already produces) -- the fix belongs here, at the one place
            # this template's own sex value is produced, not in relaxing
            # that check to accept a second, redundant representation.
            v = "M" if sex_m.group(1).upper().startswith("M") else "F"
            sex_votes[v] = sex_votes.get(v, 0) + 1
        dates = _dates_from_one_pass(text)
        # Newer card generation's EC/GC-suffix format -- only consulted for
        # whatever the alphabetic-month pass above didn't already resolve,
        # see `_numeric_calendar_dates_from_one_pass`'s own docstring.
        fallback_dates = _numeric_calendar_dates_from_one_pass(text)
        for key in (
            "dob_gregorian", "dob_ethiopian_display", "date_of_expiry",
            "date_of_expiry_ethiopian_display", "date_of_issue", "date_of_issue_ethiopian_display",
        ):
            dates.setdefault(key, fallback_dates.get(key))
        for value, votes in (
            (dates.get("dob_gregorian"), dob_votes),
            (dates.get("dob_ethiopian_display"), dob_display_votes),
            (dates.get("date_of_expiry"), expiry_votes),
            (dates.get("date_of_expiry_ethiopian_display"), expiry_display_votes),
            (dates.get("date_of_issue"), issue_votes),
            (dates.get("date_of_issue_ethiopian_display"), issue_display_votes),
        ):
            if value:
                votes[value] = votes.get(value, 0) + 1
        if _SELF_DECLARED_RE.search(text):
            nationality_qualifier_votes["self_declared"] = nationality_qualifier_votes.get("self_declared", 0) + 1
        card_gen_m = _CARD_GENERATION_LABEL_RE.search(text)
        if card_gen_m:
            label = card_gen_m.group(0).upper()
            card_generation_votes[label] = card_generation_votes.get(label, 0) + 1

    out = {}
    name = _pick_name(name_votes)
    if name:
        out["full_name"] = name
    if sex_votes:
        out["sex"] = max(sex_votes, key=lambda k: sex_votes[k])
    if dob_votes:
        out["dob_gregorian"] = max(dob_votes, key=lambda k: dob_votes[k])
    if dob_display_votes:
        out["dob_ethiopian_display"] = max(dob_display_votes, key=lambda k: dob_display_votes[k])
    if expiry_votes:
        out["date_of_expiry"] = max(expiry_votes, key=lambda k: expiry_votes[k])
    if expiry_display_votes:
        out["date_of_expiry_ethiopian_display"] = max(expiry_display_votes, key=lambda k: expiry_display_votes[k])
    if issue_votes:
        out["date_of_issue"] = max(issue_votes, key=lambda k: issue_votes[k])
    if issue_display_votes:
        out["date_of_issue_ethiopian_display"] = max(issue_display_votes, key=lambda k: issue_display_votes[k])
    if nationality_qualifier_votes:
        out["nationality_qualifier"] = "self_declared"
    if card_generation_votes:
        out["card_generation"] = max(card_generation_votes, key=lambda k: card_generation_votes[k])
    return out


# Title-case-per-word only ("Addis Ababa", "Kirkos", "Nifas Silk-Lafto")
# -- excludes the ALL-CAPS or all-lowercase noise OCR tends to produce
# when it's actually misreading the Amharic label line right above each
# value, which has no real English content at all. Each word may carry
# one internal hyphen joining two Title-Case parts (a real sub-city name
# on a newer-generation card) without opening the door to genuine
# ALL-CAPS noise -- "ADDIS" still fails since it has no lowercase letter
# after its first character.
_CLEAN_WORD_LINE_RE = re.compile(
    r"^[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s[A-Z][a-z]+(?:-[A-Z][a-z]+)?){0,2}$"
)
# Newer-generation cards print an actual English "Region"/"Zone" label
# (as part of a compound "Region/City Admin" / "Zone/City Admin/Sub
# City" header) rather than relying only on the Amharic label an older
# card has instead -- confirmed against a real card. Tried FIRST, since
# it's a direct label anchor; the position-after-"Ethiopian" fallback
# below still covers a card whose back has no such label at all.
_REGION_LABEL_RE = re.compile(r"\bregion\b[^\n]*", re.IGNORECASE)
_ZONE_LABEL_RE = re.compile(r"\bzone\b[^\n]*", re.IGNORECASE)


def _value_on_next_clean_line(text: str, after: int) -> str:
    for line in text[after:].splitlines():
        line = line.strip()
        if not line:
            continue
        return line if _CLEAN_WORD_LINE_RE.match(line) else ""
    return ""


def _parse_back_ocr(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Date of Expiry prints on the BACK on some card layouts (confirmed
    # against a real card) rather than alongside the other dates on the
    # front -- the same field, same dual-calendar EC/GC-suffix shape
    # `_dates_from_one_pass`/`_numeric_calendar_dates_from_one_pass`
    # already resolve for the front, just printed on the other side of
    # the card this time. `parse_ethiopia_digital_id` only uses this as
    # a fallback when the front didn't already have it.
    back_dates = _dates_from_one_pass(text)
    fallback_dates = _numeric_calendar_dates_from_one_pass(text)
    for key in ("date_of_expiry", "date_of_expiry_ethiopian_display"):
        back_dates.setdefault(key, fallback_dates.get(key))
        if back_dates.get(key):
            out[key] = back_dates[key]
    # FIN is a 12-digit number, but OCR sometimes glues a few extra digits
    # from a misread label onto its front (e.g. "FIN" -> "11", observed
    # against a real card) while reading the genuine 12 digits themselves
    # cleanly -- so take the last 12 digits of any long-enough run rather
    # than requiring an exact 12-digit match. A 16-digit run is a FAN
    # (this card's other identifier), not a FIN, so it's excluded.
    for run in _FIN_DIGIT_RUN_RE.findall(text):
        digits = run.replace(" ", "")
        if len(digits) != 16:
            out["fin"] = digits[-12:]
            break
    phone_m = _PHONE_RE.search(text)
    if phone_m:
        out["phone_number"] = phone_m.group(0)
    # "woreda" itself sometimes gets its trailing letter misread (e.g.
    # "Woreds") -- match the stable "wored" stem rather than the exact word.
    woreda_m = re.search(r"wored\w{0,2}\D{0,3}(\d+)", text, re.IGNORECASE)
    if woreda_m:
        out["woreda"] = f"Woreda {woreda_m.group(1)}"

    region_m = _REGION_LABEL_RE.search(text)
    if region_m:
        value = _value_on_next_clean_line(text, region_m.end())
        if value:
            out["region"] = value
    zone_m = _ZONE_LABEL_RE.search(text)
    if zone_m:
        value = _value_on_next_clean_line(text, zone_m.end())
        if value:
            out["zone"] = value

    # Fallback for an older-generation card whose back has no English
    # "Region"/"Zone" label of its own at all (only the Amharic label,
    # which OCR mangles into noise) -- the printed layout is a fixed
    # Nationality -> Region -> Zone -> Woreda stack, so take the first
    # two clean-looking English word lines after "Ethiopian"/nationality
    # as region then zone instead. Only fills whichever of the two the
    # label anchor above didn't already find.
    if "region" not in out or "zone" not in out:
        nat_m = re.search(r"ethiopian", text, re.IGNORECASE)
        if nat_m:
            candidates = [
                line.strip() for line in text[nat_m.end():].splitlines()
                if _CLEAN_WORD_LINE_RE.match(line.strip())
                and "woreda" not in line.lower()
            ]
            if candidates:
                out.setdefault("region", candidates[0])
            if len(candidates) > 1:
                out.setdefault("zone", candidates[1])
    return out


# Bounded grid of left-edge crop widths x rotation directions to search for
# the vertically-printed Date of Issue field -- neither the exact sidebar
# width nor which way it's rotated is known ahead of time, so this tries a
# small fixed set of both rather than one guessed value, same philosophy as
# barcode.py's bounded crop/scale search for small 1D barcodes.
_SIDEBAR_WIDTH_FRACTIONS = (0.10, 0.14, 0.18)
_SIDEBAR_ROTATIONS = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _locate_card_bbox(gray: "np.ndarray") -> Optional[tuple[int, int, int, int]]:
    """Finds the card's own bounding box within a photo that has real
    background margin around the card (a phone photo of a card lying on a
    table/desk, not a tight scan) -- confirmed necessary against a real
    "flat lay" photo where the card occupied roughly a fifth of the frame's
    area, off-center. `preprocess.rectify_card`'s stricter quad detector
    (exact 4-corner polygon, tuned to avoid false positives across every
    document family this project handles) fell back to the unrectified
    original on that same photo, which meant `_issue_date_from_sidebar_ocr`
    below was cropping a percentage of the FRAME's left edge -- table, not
    card -- and never finding the sidebar text at all.

    This is deliberately looser than `rectify_card`'s detector (bounding
    box of the largest non-border-touching edge-contour, not a strict
    4-point polygon) -- acceptable ONLY because this function is used
    exclusively as a targeted crop for one already-identified Ethiopian ID
    card (the caller already confirmed the document family via barcode/QR
    or header text), never to decide what a document IS in the first
    place. A caller that gets None falls back to treating the whole frame
    as the card, same as before this existed -- never a guess that could
    make the crop worse.
    """
    import numpy as np

    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[tuple[int, int, int, int]] = None
    best_area = 0.15 * w * h  # below this, not confidently "the card" -- see docstring
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        touches_border = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
        if touches_border or area <= best_area:
            continue
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if not (1.2 <= aspect <= 2.0):  # ID-1 is ~1.586:1, tolerant of perspective skew
            continue
        best, best_area = (x, y, cw, ch), area

    return best


def _issue_date_from_sidebar_ocr(front_path: str) -> tuple[str, str]:
    """Free/offline fallback for Date of Issue, which -- unlike DOB and
    expiry -- prints rotated ~90 degrees along the card's left edge rather
    than inline with the other date fields, so it never appears in the
    normal-orientation OCR passes `_parse_front_ocr` runs. Crops a narrow
    strip off the left edge, rotates it both ways, and OCRs each variant.
    Tries both this module's date formats -- the older generation's
    alphabetic Gregorian month, and the newer generation's explicit EC/GC
    suffix (confirmed on a real card: the sidebar prints "2016/10/26EC
    2024/07/03GC" with no alphabetic month at all) -- since the sidebar
    crop has no surrounding "issu" label text of its own to anchor a
    window on, `_resolve_gc_ec_pair` is applied directly against the whole
    strip's OCR text. Best-effort like every other OCR-only field in this
    module -- not as reliable as the GEMINI_API_KEY multimodal path (still
    preferred when available, since it reads the rotated text directly
    off the pixels instead of depending on a crop/rotate guess), but gives
    a free alternative to a permanently-blank field otherwise. Returns
    (gregorian_iso, ethiopian_display), either half empty if not found.
    """
    try:
        img = load_image(front_path)
    except FileNotFoundError:
        return "", ""

    # Card-bbox detection runs on the ORIGINAL image, before upscale/CLAHE
    # -- confirmed necessary against a real photo where CLAHE's local
    # contrast boost amplified shadow/reflection gradients enough to merge
    # the card's edge-contour into one blob touching the frame border
    # (rejected by _locate_card_bbox's own border check, so detection
    # found nothing at all on the enhanced image even though it worked
    # cleanly on the original). The bbox is then scaled by the same factor
    # `upscale_if_small` applies, so it still lines up on the upscaled
    # image used for the actual OCR crop.
    orig_h, orig_w = img.shape[:2]
    bbox = _locate_card_bbox(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    img = upscale_if_small(img, min_dim=1600)
    enhanced = enhance_contrast(img)

    # Crop to the card's own bounding box first when a photo has real
    # background margin around the card (see _locate_card_bbox) -- the
    # fixed-fraction sidebar strip below is meaningless if taken from the
    # PHOTO's edge instead of the CARD's edge. Small padding (2% of the
    # card's own width) keeps the crop from clipping the sidebar text if
    # the detected box is a few pixels tight on the true edge. Falls back
    # to the full (already upscaled+enhanced) frame when no confident card
    # region is found -- unchanged from before this existed.
    if bbox is not None:
        scale = img.shape[1] / orig_w
        x, y, cw, ch = (int(v * scale) for v in bbox)
        pad = max(2, int(cw * 0.02))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(enhanced.shape[1], x + cw + pad), min(enhanced.shape[0], y + ch + pad)
        enhanced = enhanced[y0:y1, x0:x1]

    _, w = enhanced.shape[:2]

    # 6 independent tesseract subprocess calls (3 widths x 2 rotations) --
    # OCR'd through a thread pool for the same reason/prerequisite as
    # `_ocr_texts` above (real wall-clock parallelism, requires
    # preprocess.py's OMP_THREAD_LIMIT=1 cap to be a net win rather than a
    # CPU-oversubscription loss). Rotated strips are all built up front
    # (cheap, no OCR) so only the actual tesseract calls go through the pool.
    #
    # Each strip is upscaled 2x AFTER rotation, right before OCR -- confirmed
    # necessary against a real card: even with a correctly-cropped card (see
    # _locate_card_bbox above), a ~10-18%-of-card-width strip is still only
    # ~100-190px wide, too low-resolution for tesseract to read "Oct"
    # reliably (it read "0ct", then "0c" -- losing another character -- on
    # consecutive runs of the SAME strip). 2x consistently recovered the
    # full "0ct/20" across 3 different PSM modes in testing; the strip is
    # already narrow, so this stays cheap relative to OCRing the whole card.
    rotated_strips = [
        cv2.resize(
            cv2.rotate(enhanced[:, 0:max(1, int(w * frac))], rotation),
            None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC,
        )
        for frac in _SIDEBAR_WIDTH_FRACTIONS
        for rotation in _SIDEBAR_ROTATIONS
    ]

    def _ocr_strip(strip: Image.Image) -> Optional[str]:
        from .ocr_engine import ocr_text

        return ocr_text(strip, tesseract_config="--psm 6")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(rotated_strips))) as executor:
        texts = list(executor.map(_ocr_strip, rotated_strips))

    from .generic_ocr import _slash_date_to_iso

    iso_votes: dict[str, int] = {}
    display_votes: dict[str, int] = {}
    for text in texts:
        if text is None:
            continue
        for m in _ALPHA_MONTH_DATE_RE.finditer(text):
            iso = _alpha_month_to_iso(*m.groups())
            if iso:
                iso_votes[iso] = iso_votes.get(iso, 0) + 1
                # The alphabetic-month (Gregorian) reading has its Ethiopian-
                # calendar counterpart printed immediately before it, same
                # pairing `_dates_from_one_pass` already applies for DOB/
                # expiry on the card's front face -- this path just never
                # populated `display_votes` for it before, so the sidebar's
                # own Ethiopian-display value was silently dropped even
                # when the Gregorian side resolved correctly.
                paired = _paired_ethiopian_display(text, m.start())
                if paired:
                    display_votes[paired] = display_votes.get(paired, 0) + 1
        numeric_tokens = list(_NUMERIC_DATE_TOKEN_RE.finditer(text))
        if numeric_tokens:
            gc_raw, ec_raw = _resolve_gc_ec_pair(numeric_tokens, text)
            if gc_raw:
                iso = _slash_date_to_iso(gc_raw)
                if iso:
                    iso_votes[iso] = iso_votes.get(iso, 0) + 1
            if ec_raw:
                display_votes[ec_raw] = display_votes.get(ec_raw, 0) + 1

    iso = max(iso_votes, key=lambda k: iso_votes[k]) if iso_votes else ""
    display = max(display_votes, key=lambda k: display_votes[k]) if display_votes else ""
    return iso, display


def _has_qr_marker(path: str) -> bool:
    return any(_QR_MARKER in hit.text for hit in decode_barcodes(path, formats=("QRCode",)))


def _has_fan_barcode(path: str) -> bool:
    return any(_FAN_RE.fullmatch(hit.text) for hit in decode_barcodes(path, formats=("Code128",)))


def classify_ethiopia_id_sides(path_a: str, path_b: str) -> tuple[str, str]:
    """Given two images of an Ethiopian Digital ID in unknown order, figure
    out which is front (FAN barcode, photo, "Ethiopian Digital ID Card"
    header) and which is back (QR code, FIN, region/zone/woreda) -- so a
    caller that can't guarantee upload order (e.g. two files attached
    under the same form field, in whichever order a person picked them)
    doesn't have to. Returns (front_path, back_path).

    Decode-based signals (QR / FAN barcode) are tried first since they're
    unambiguous and involve no OCR; the front/back header text is a
    fallback for a photo where the codes didn't decode cleanly. If nothing
    is decisive either way (most likely because this isn't actually an
    Ethiopian ID at all), the given order is returned unchanged --
    `parse_ethiopia_digital_id` will correctly return None in that case
    regardless of which path is passed as which.
    """
    a_qr, a_fan = _has_qr_marker(path_a), _has_fan_barcode(path_a)
    b_qr, b_fan = _has_qr_marker(path_b), _has_fan_barcode(path_b)

    if a_qr and not b_qr:
        return path_b, path_a
    if b_qr and not a_qr:
        return path_a, path_b
    if a_fan and not b_fan:
        return path_a, path_b
    if b_fan and not a_fan:
        return path_b, path_a

    try:
        a_is_front = looks_like_ethiopian_digital_id("\n".join(_ocr_texts(path_a)))
        b_is_front = looks_like_ethiopian_digital_id("\n".join(_ocr_texts(path_b)))
    except Exception:
        a_is_front = b_is_front = False
    if a_is_front and not b_is_front:
        return path_a, path_b
    if b_is_front and not a_is_front:
        return path_b, path_a

    return path_a, path_b


def looks_like_ethiopian_digital_id(front_text: str) -> bool:
    return bool(_ETHIOPIAN_WORD_RE.search(front_text) and _HEADER_RE.search(front_text))


def parse_ethiopia_digital_id(front_path: str, back_path: str | None = None) -> EthiopiaIDResult | None:
    """Returns None (like `_try_parse_td1` etc.) when this doesn't look like
    an Ethiopian Digital ID at all, so the pipeline can fall through to the
    next tier. Otherwise returns a best-effort result: fields sourced from
    the QR/barcode are trustworthy, fields sourced from OCR are not (no
    checksum exists for them), which is why `checks` only ever reflects
    the QR/barcode cross-source agreement, never OCR-derived fields.
    """
    front_texts = _ocr_texts(front_path, _FRONT_OCR_CONFIGS)
    qr_fields: dict[str, Any] = {}
    barcode_fan = ""
    raw_qr_present = False
    raw_barcode_present = False

    if back_path:
        for hit in decode_barcodes(back_path, formats=("QRCode",)):
            parsed = _parse_qr_payload(hit.text)
            if parsed:
                qr_fields = parsed
                raw_qr_present = True
                break

    for hit in decode_barcodes(front_path, formats=("Code128",)):
        if _FAN_RE.fullmatch(hit.text):
            barcode_fan = hit.text
            raw_barcode_present = True
            break

    if (
        not raw_qr_present and not raw_barcode_present
        and not looks_like_ethiopian_digital_id("\n".join(front_texts))
    ):
        return None

    result = EthiopiaIDResult(raw_qr_present=raw_qr_present, raw_barcode_present=raw_barcode_present)

    front_ocr = _parse_front_ocr(front_texts)
    back_ocr = _parse_back_ocr(_ocr_text(back_path, sparse=True)) if back_path else {}

    qr_name = qr_fields.get("full_name", "")
    ocr_name = front_ocr.get("full_name", "")
    qr_dob_display = qr_fields.get("dob_ethiopian_display", "")
    ocr_dob_display = front_ocr.get("dob_ethiopian_display", "")

    result.full_name = qr_name or ocr_name
    result.fan = qr_fields.get("fan") or barcode_fan
    result.dob_ethiopian_display = qr_dob_display or ocr_dob_display
    result.dob_gregorian = front_ocr.get("dob_gregorian", "")
    result.sex = front_ocr.get("sex", "")
    result.date_of_issue = front_ocr.get("date_of_issue", "")
    result.date_of_issue_ethiopian_display = front_ocr.get("date_of_issue_ethiopian_display", "")
    # Printed on the FRONT on some card layouts, the BACK on others
    # (confirmed against a real card) -- try both, front first since
    # that's the more common layout in this project's own corpus.
    result.date_of_expiry = front_ocr.get("date_of_expiry") or back_ocr.get("date_of_expiry", "")
    result.date_of_expiry_ethiopian_display = (
        front_ocr.get("date_of_expiry_ethiopian_display")
        or back_ocr.get("date_of_expiry_ethiopian_display", "")
    )
    result.nationality_qualifier = front_ocr.get("nationality_qualifier", "")
    result.card_generation = front_ocr.get("card_generation", "")
    result.fin = back_ocr.get("fin", "")
    result.phone_number = back_ocr.get("phone_number", "")
    result.region = back_ocr.get("region", "")
    result.zone = back_ocr.get("zone", "")
    result.woreda = back_ocr.get("woreda", "")

    if not result.date_of_issue:
        sidebar_iso, sidebar_display = _issue_date_from_sidebar_ocr(front_path)
        if sidebar_iso:
            result.date_of_issue = sidebar_iso
            result.date_of_issue_from_sidebar_ocr = True
        if sidebar_display and not result.date_of_issue_ethiopian_display:
            result.date_of_issue_ethiopian_display = sidebar_display

    # Cross-checks between independent sources for the same field --
    # QR (back) vs. front side where both exist, and the front's own
    # printed EC/GC pair where a Gregorian conversion was actually
    # resolved. None of these require a second image; the dual-calendar
    # ones are free on a front-only submission.
    if qr_fields.get("fan") and barcode_fan:
        result.checks["fan_qr_barcode_match"] = qr_fields["fan"] == barcode_fan
    if qr_name and ocr_name:
        result.checks["full_name_qr_ocr_match"] = qr_name.strip().lower() == ocr_name.strip().lower()
    if qr_dob_display and ocr_dob_display:
        result.checks["dob_ethiopian_display_qr_ocr_match"] = qr_dob_display == ocr_dob_display
    for check_name, display, gregorian in (
        ("dob_dual_calendar_plausible", result.dob_ethiopian_display, result.dob_gregorian),
        ("date_of_issue_dual_calendar_plausible", result.date_of_issue_ethiopian_display, result.date_of_issue),
        ("date_of_expiry_dual_calendar_plausible", result.date_of_expiry_ethiopian_display, result.date_of_expiry),
    ):
        plausible = _dual_calendar_years_plausible(display, gregorian)
        if plausible is not None:
            result.checks[check_name] = plausible

    return result

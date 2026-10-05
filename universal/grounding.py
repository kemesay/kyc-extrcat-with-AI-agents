"""
OCR grounding — independent pixel evidence for what the agents claim.

Vision LLMs fail in two ways a KYC system can't accept silently: they
misread (a digit, a calendar), and they hallucinate plausible values that
aren't on the document at all. Local CPU OCR reads the same image with no
language model in the loop, so it can arbitrate:

  - grounded      the agents' value is also present in the OCR text
  - not grounded  OCR read the document well but the value is nowhere in it
                  -> confidence cut; a lone agent's ungrounded value is
                     dropped into `alternatives` as a likely hallucination
  - calendar fix  OCR sees a printed dual-calendar pair (e.g. Ethiopian EC
                  next to Gregorian) whose halves agree by calendar
                  arithmetic; when an agent reported the original-calendar
                  date as if it were Gregorian, it is corrected by code
  - completion    an agent reported only one digit group of a grouped
                  number ("1234" of "1234 5678 9012 3456"); the full run the
                  OCR saw replaces it when it validates for the country

Engines: RapidOCR (idx.ocr_engine, Latin/CJK, runs in parallel with the
agents) + tesseract with the language packs matching the document's script
(amh, ara, fas, chi_sim, jpn, kor, rus, hin, heb, ...) when installed.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional

from PIL import Image, ImageOps

from . import calendars as cal
from .dates import COUNTRY_CALENDARS, normalize_date
from .id_numbers import SCHEMES, compact
from .normalize import detect_script, is_latin
from .reconcile import CRITICAL_FIELDS, DATE_FIELDS, Context, FieldResult

logger = logging.getLogger(__name__)

MIN_SUBSTANTIAL_CHARS = 40

_SCRIPT_LANGS = {
    "Ethiopic": ["amh", "tir"], "Arabic": ["ara", "fas", "urd"], "Hebrew": ["heb"],
    "CJK": ["chi_sim", "chi_tra", "jpn", "kor"], "Cyrillic": ["rus", "ukr", "bul", "srp", "kaz"],
    "Devanagari": ["hin", "nep", "mar"], "Greek": ["ell"], "Thai": ["tha"], "Georgian": ["kat"],
    "Armenian": ["hye"], "Bengali": ["ben"], "Tamil": ["tam"], "Telugu": ["tel"],
    "Gujarati": ["guj"], "Gurmukhi": ["pan"], "Sinhala": ["sin"], "Burmese": ["mya"],
    "Khmer": ["khm"], "Lao": ["lao"],
}
_COUNTRY_LANGS = {
    "ETH": ["amh"], "ERI": ["tir"], "SAU": ["ara"], "ARE": ["ara"], "EGY": ["ara"], "IRQ": ["ara"],
    "JOR": ["ara"], "MAR": ["ara", "fra"], "DZA": ["ara", "fra"], "TUN": ["ara", "fra"], "IRN": ["fas"],
    "AFG": ["fas"], "PAK": ["urd"], "CHN": ["chi_sim"], "TWN": ["chi_tra"], "HKG": ["chi_tra"],
    "JPN": ["jpn"], "KOR": ["kor"], "RUS": ["rus"], "UKR": ["ukr"], "KAZ": ["kaz", "rus"],
    "BGR": ["bul"], "SRB": ["srp"], "IND": ["hin"], "NPL": ["nep"], "ISR": ["heb"], "GRC": ["ell"],
    "THA": ["tha"], "GEO": ["kat"], "ARM": ["hye"], "BGD": ["ben"], "LKA": ["sin"], "MMR": ["mya"],
    "KHM": ["khm"], "LAO": ["lao"], "DEU": ["deu"], "FRA": ["fra"], "ESP": ["spa"], "ITA": ["ita"],
    "PRT": ["por"], "BRA": ["por"], "TUR": ["tur"], "VNM": ["vie"], "IDN": ["ind"], "POL": ["pol"],
}

# Numeric dates with ./- separators, or single spaces ("15 03 1992" on Belgian cards).
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}"
                           r"|\d{1,2} \d{1,2} \d{4})(?!\d)")
# Month words, incl. bilingual "OKT/OCT", glued "06MAR1994" and 2-digit years "SEP/SEP 75".
_WORD_DATE = re.compile(r"(?<!\d)\d{1,2}[\s.-]*[^\W\d_]{3,12}(?:/[^\W\d_]{3,12})?\.?[\s.-]*\d{2,4}(?!\d)"
                        r"|[^\W\d_]{3,12}\.?\s+\d{1,2},?\s+\d{4}")
_CJK_DATE = re.compile(r"\d{2,4}\s*[年년]\s*\d{1,2}\s*[月월]\s*\d{1,2}\s*[日일]?")
_DIGIT_RUN = re.compile(r"\d(?:\d|[ \-](?=\d))*\d")
# OCR sometimes drops the separator between side-by-side dates: "30/06/203023/10/2022".
_GLUED_DATES = re.compile(r"(\d{1,2}([./-])\d{1,2}\2\d{4})(?=\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")


@lru_cache(maxsize=1)
def _tesseract_languages() -> frozenset[str]:
    try:
        import pytesseract
        if shutil.which("tesseract") is None:
            # Server started without the conda env on PATH — use the env's binary.
            candidate = f"{sys.prefix}/bin/tesseract"
            if shutil.which(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
        return frozenset(pytesseract.get_languages())
    except Exception as e:
        logger.info("tesseract unavailable for grounding: %s", e)
        return frozenset()


@dataclass
class CalendarPair:
    gregorian: str          # ISO date
    original: str           # as printed
    calendar: str
    original_digits: tuple[int, int, int]  # the printed numbers, in printed order


@dataclass
class OcrEvidence:
    text: str = ""
    engines: list[str] = field(default_factory=list)
    gregorian_dates: dict[str, str] = field(default_factory=dict)   # iso -> raw
    pairs: list[CalendarPair] = field(default_factory=list)
    digit_runs: list[str] = field(default_factory=list)

    @property
    def compact(self) -> str:
        return re.sub(r"[^0-9A-Z]", "", _fold(self.text).upper())

    @property
    def substantial(self) -> bool:
        return len(self.compact) >= MIN_SUBSTANTIAL_CHARS

    def summary(self) -> dict:
        return {"engines": self.engines, "characters": len(self.compact), "substantial": self.substantial,
                "dates_seen": sorted(self.gregorian_dates), "calendar_pairs": [
                    {"gregorian": p.gregorian, "original": p.original, "calendar": p.calendar} for p in self.pairs]}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c))


def _load(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def rapid_ocr_text(image_paths: list[str]) -> str:
    """RapidOCR via idx.ocr_engine — CPU-only, safe to run beside the agents."""
    from idx.ocr_engine import ocr_text
    chunks = []
    for path in image_paths:
        try:
            chunks.append(ocr_text(_load(path)) or "")
        except Exception as e:
            logger.warning("RapidOCR failed on %s: %s", path, e)
    return "\n".join(chunks)


def early_ocr_text(image_paths: list[str]) -> str:
    """RapidOCR + an English tesseract pass, run beside the agents: the two
    engines segment lines differently ('ROE,' keeps its comma in tesseract),
    and the name/date interpreters need both before grounding runs."""
    rapid = rapid_ocr_text(image_paths)
    tess, _ = tesseract_text(image_paths, "", "Latin")
    return f"{rapid}\n{tess}" if tess else rapid


def tesseract_text(image_paths: list[str], country: str, script: str) -> tuple[str, str]:
    installed = _tesseract_languages()
    if not installed:
        return "", ""
    wanted = ["eng"] + _COUNTRY_LANGS.get(country, []) + _SCRIPT_LANGS.get(script, [])[:2]
    langs = "+".join(dict.fromkeys(l for l in wanted if l in installed))
    if not langs:
        return "", ""
    import pytesseract
    chunks = []
    for path in image_paths:
        try:
            chunks.append(pytesseract.image_to_string(_load(path), lang=langs, timeout=30))
        except Exception as e:
            logger.warning("tesseract failed on %s: %s", path, e)
    return "\n".join(chunks), langs


# ─────────────────────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────────────────────

def _numbers(raw: str) -> tuple[int, int, int]:
    a, b, c = (int(x) for x in re.split(r"[./-]", raw))
    return a, b, c


def _pair_candidates(orig: tuple[int, int, int], calendar: str) -> list[date]:
    """Gregorian dates the printed original-calendar numbers could denote."""
    a, b, c = orig
    ymd = (a, b, c) if a > 31 else (c, b, a)          # year-first or day-first
    y, m, d = ymd
    out = []
    try:
        if calendar == "ethiopian":
            out.append(cal.ethiopian_to_gregorian(y, m, d))
        elif calendar == "persian":
            out.append(cal.persian_to_gregorian(y, m, d))
        elif calendar == "hijri":
            out.append(cal.hijri_to_gregorian(y, m, d))
    except ValueError:
        pass
    return out


def _gregorian_readings(nums: tuple[int, int, int]) -> list[date]:
    a, b, c = nums
    out = []
    for y, m, d in ((c, b, a), (c, a, b), (a, b, c)):
        if y < 1000:
            continue
        try:
            out.append(date(y, m, d))
        except ValueError:
            pass
    return out


def build_evidence(text: str, engines: list[str], country: str) -> OcrEvidence:
    text = _GLUED_DATES.sub(r"\1 ", text)
    ev = OcrEvidence(text=text, engines=engines)
    numeric = []
    for m in _NUMERIC_DATE.finditer(text):
        raw = m.group(1)
        r = normalize_date(raw, country=country)
        if r.iso:
            ev.gregorian_dates.setdefault(r.iso, raw)
        try:
            numeric.append((raw, _numbers(raw), r.iso))
        except ValueError:
            continue
    for pattern in (_WORD_DATE, _CJK_DATE):
        for m in pattern.finditer(text):
            raw = m.group(0)
            # A 2-digit year is 19xx on a birth date but 20xx on an expiry: keep both readings.
            for future in ((False, True) if re.search(r"(?<!\d)\d{2}\s*$", raw) else (False,)):
                r = normalize_date(raw, country=country, future_leaning=future)
                if r.iso and r.calendar in ("gregorian", "japanese", "minguo"):
                    ev.gregorian_dates.setdefault(r.iso, raw)

    calendars = [COUNTRY_CALENDARS[country]] if country in COUNTRY_CALENDARS else ["ethiopian", "persian", "hijri"]
    calendars = [c for c in calendars if c in ("ethiopian", "persian", "hijri")]
    for raw_o, nums_o, _ in numeric:
        for raw_g, nums_g, _ in numeric:
            if raw_o == raw_g:
                continue
            # Every valid reading of the Gregorian half: some documents (Ethiopian
            # licences) print it month-first, and the pair arithmetic decides which.
            for target in _gregorian_readings(nums_g):
                matched = False
                for calendar in calendars:
                    tolerance = timedelta(days=1 if calendar == "hijri" else 0)
                    if any(abs(c - target) <= tolerance for c in _pair_candidates(nums_o, calendar)):
                        ev.pairs.append(CalendarPair(target.isoformat(), raw_o, calendar, nums_o))
                        ev.gregorian_dates[target.isoformat()] = raw_g
                        matched = True
                if matched:
                    break
    unique = {(p.gregorian, p.original, p.calendar): p for p in ev.pairs}
    ev.pairs = list(unique.values())
    # A date that is the original half of a pair is not a Gregorian date.
    for p in ev.pairs:
        for iso, raw in list(ev.gregorian_dates.items()):
            if raw == p.original:
                del ev.gregorian_dates[iso]

    ev.digit_runs = sorted({re.sub(r"[ \-]", "", m.group()) for m in _DIGIT_RUN.finditer(text)}, key=len, reverse=True)
    return ev


# ─────────────────────────────────────────────────────────────
# Grounding
# ─────────────────────────────────────────────────────────────

def _digits_as_iso_variants(nums: tuple[int, int, int]) -> set[str]:
    """ISO strings an agent might produce by copying printed numbers without
    converting the calendar (DMY, MDY, or year-first read)."""
    a, b, c = nums
    variants = set()
    for y, m, d in ((c, b, a), (c, a, b), (a, b, c)):
        try:
            variants.add(date(y, m, d).isoformat())
        except ValueError:
            continue
    return variants


_TEXT_FIELDS = ("full_name", "full_name_latin", "full_name_local", "first_name", "last_name", "middle_name",
                "place_of_birth", "issuing_authority", "issuing_place", "address", "region",
                "district", "sub_district")
# The two agents that read the image without seeing another agent's JSON.
# Later stages are shown earlier output, so their agreement isn't independent.
_INDEPENDENT_AGENTS = {"fast_scanner", "field_extractor"}


def _text_grounded(value: str, ev: OcrEvidence) -> Optional[bool]:
    """Latin text: True/False on 3+ char tokens found in the OCR text.
    Non-Latin text: True on a near-exact OCR line, else None — OCR for many
    scripts is too weak to prove absence."""
    if not is_latin(value):
        squashed = "".join(value.split())
        if len(squashed) >= 1 and squashed in "".join(ev.text.split()):
            return True  # covers partial names (surname 王 inside 王芳) and address parts
        ratio, _ = _best_line_match(value, ev.text)
        return True if ratio >= 0.9 else None
    tokens = [t for t in re.findall(r"[A-Z0-9]{3,}", _fold(value).upper())]
    if not tokens:
        return None
    haystack = ev.compact
    return sum(1 for t in tokens if t in haystack) / len(tokens) >= 0.8


def is_grounded(name: str, value: str, ev: OcrEvidence) -> Optional[bool]:
    """True = OCR saw it, False = OCR read the document and it isn't there,
    None = OCR can't judge this kind of value."""
    if not value:
        return None
    if name in DATE_FIELDS:
        if value in ev.gregorian_dates:
            return True
        return False if ev.substantial else None
    if name in ("document_number", "personal_number"):
        number = compact(value)
        if len(number) < 4:
            return None
        if number in ev.compact or _lookalike_fold(number) in _lookalike_fold(ev.compact):
            return True   # OCR's 'CA123450A' is the printed 'CA12345OA'
        return False if ev.substantial else None
    if name == "blood_type":
        # OCR glues the value to its label ('Blood GroupAB+VE'); '+' may be spaced.
        text = re.sub(r"\s*([+\-])\s*", r"\1", _fold(ev.text).upper())
        v = re.escape(re.sub(r"\s+", "", value.upper()))
        found = re.search(rf"(?:(?<![A-Z0-9])|(?<=GROUP)|(?<=TYPE)){v}(?![A-Z0-9])", text)
        return True if found else (False if ev.substantial else None)
    if name in _TEXT_FIELDS:
        grounded = _text_grounded(value, ev)
        if grounded is False and not ev.substantial:
            return None
        return grounded
    return None


def _best_line_match(value: str, text: str) -> tuple[float, str]:
    best, best_line = 0.0, ""
    target = " ".join(value.split())
    for line in text.splitlines():
        for part in re.split(r"\s*\|\s*", line):
            part = " ".join(part.split())
            if not part or is_latin(part):
                continue
            ratio = SequenceMatcher(None, target, part).ratio()
            if ratio > best:
                best, best_line = ratio, part
    return best, best_line


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _ocr_spell_correct(value: str, ev: OcrEvidence) -> Optional[str]:
    """Latin text with one or two tokens the OCR doesn't contain, each within
    a small edit distance of exactly one OCR token ('KEBDE' -> 'KEBEDE')."""
    ocr_tokens = set(re.findall(r"[A-Z]{3,}", _fold(ev.text).upper()))
    tokens = value.split()
    haystack = ev.compact
    fixed, changed = [], 0
    for token in tokens:
        key = _fold(token).upper()
        if len(key) < 3 or key in haystack:
            fixed.append(token)
            continue
        limit = 1 if len(key) < 7 else 2      # the match must also be unique (below)
        near = [t for t in ocr_tokens if abs(len(t) - len(key)) <= limit and _edit_distance(t, key) <= limit]
        if not near:
            # OCR glues neighbouring words ('KEBEDETESFAYE'): look inside glued tokens.
            near = list({t[i:i + len(key)] for t in ocr_tokens if len(t) > len(key) + 2
                         for i in range(len(t) - len(key) + 1)
                         if _edit_distance(t[i:i + len(key)], key) <= limit})
        if len(near) != 1:
            return None
        fixed.append(near[0] if token.isupper() else near[0].title())
        changed += 1
    if not changed or changed > 2:
        return None
    return " ".join(fixed)


def _all_readings(text: str) -> set[str]:
    from .printed import ocr_date_readings
    return ocr_date_readings(text)


_LOOKALIKES = str.maketrans("OQDILZSBG", "000112586")


def _lookalike_fold(text: str) -> str:
    return text.upper().translate(_LOOKALIKES)


def _swap_day_month(iso: str) -> str:
    try:
        d = date.fromisoformat(iso)
        return date(d.year, d.day, d.month).isoformat() if d.day != d.month else ""
    except ValueError:
        return ""


def _hamming_close(a: str, b: str, max_diff: int = 2) -> bool:
    return len(a) == len(b) and a != b and sum(x != y for x, y in zip(a, b)) <= max_diff


def _promote(fr: FieldResult, value: str, name: str, why: str, ctx: Context, confidence: float = 0.75) -> None:
    ctx.warn(f"{name}: OCR supports {value!r} over the agents' pick {fr.value!r} ({why})")
    fr.alternatives = [fr.value] + [a for a in fr.alternatives if a != value]
    fr.value, fr.confidence, fr.source = value, confidence, "ocr_arbitration"
    fr.verified_by = ["ocr"]
    fr.flags.append("ocr_selected_alternative")


def _mark(fr: FieldResult, grounded: bool, n_agents: int, name: str, ctx: Context) -> None:
    if grounded:
        fr.verified_by.append("ocr")
        fr.confidence = min(0.95, fr.confidence + 0.1)
        return
    fr.flags.append("not_found_in_ocr")
    fr.confidence = min(fr.confidence, 0.4)
    agents = set(fr.agreeing_agents)
    unsupported = len(agents) <= 1 or not agents & _INDEPENDENT_AGENTS
    if unsupported and n_agents >= 3 and name not in CRITICAL_FIELDS and fr.source in ("consensus", "supervisor"):
        fr.alternatives = [fr.value] + fr.alternatives
        fr.flags.append("dropped_possible_hallucination")
        ctx.warn(f"{name}: {fr.value!r} is not on the document per OCR and no independent agent read it — dropped")
        fr.value, fr.confidence = "", 0.0
    elif name in CRITICAL_FIELDS:
        ctx.warn(f"{name}: {fr.value!r} was not found by OCR on the document")
        # A second, independent read (the transcriber) or a format/checksum match
        # already corroborates it; OCR simply can't read this document's font.
        corroborated = {"printed_text", "format_template", "calendar_pair"} & set(fr.verified_by) or any(
            v.startswith(("checksum:", "id_number:")) for v in fr.verified_by)
        if not corroborated:
            ctx.review(f"{name}: not corroborated by OCR")


def _calendar_fix(fields: dict[str, FieldResult], name: str, ev: OcrEvidence, ctx: Context, report: dict) -> bool:
    fr = fields[name]
    pair = next((p for p in ev.pairs if fr.value in _digits_as_iso_variants(p.original_digits)), None)
    if pair is None:
        return False
    fr.alternatives = [fr.value] + fr.alternatives
    ctx.warn(f"{name}: agents reported the {pair.calendar}-calendar date {pair.original!r} as Gregorian "
             f"{fr.value} — corrected by calendar arithmetic to {pair.gregorian}")
    fr.value, fr.source, fr.confidence = pair.gregorian, "calendar_arithmetic", 0.9
    fr.verified_by = ["ocr", f"{pair.calendar}_calendar_pair"]
    fr.flags.append("calendar_corrected")
    report["calendar_corrections"].append({"field": name, "original": pair.original,
                                           "calendar": pair.calendar, "gregorian": pair.gregorian})
    if name == "date_of_birth" and not fields["date_of_birth_orig"].value:
        fields["date_of_birth_orig"] = FieldResult(value=pair.original, confidence=0.9,
                                                   source="ocr", verified_by=["ocr"])
    return True


def _script_arbitration(fr: FieldResult, name: str, ev: OcrEvidence, ctx: Context, report: dict) -> None:
    """Non-Latin names: OCR can't prove absence, but when one agent candidate
    is clearly closer to what tesseract read, prefer it; always surface the
    OCR reading for a reviewer when it differs."""
    value_ratio, line = _best_line_match(fr.value, ev.text)
    scored = [(a, _best_line_match(a, ev.text)[0]) for a in fr.alternatives if a and not is_latin(a)]
    best_alt, best_ratio = max(scored, key=lambda t: t[1], default=("", 0.0))
    if best_ratio >= 0.5 and best_ratio >= value_ratio + 0.15:
        _promote(fr, best_alt, name, f"OCR similarity {best_ratio:.2f} vs {value_ratio:.2f}", ctx, 0.6)
        value_ratio, line = _best_line_match(fr.value, ev.text)
    if 0.5 <= value_ratio < 0.9 and line and line != fr.value:
        fr.flags.append("ocr_reads_differently")
        if line not in fr.alternatives:
            fr.alternatives.append(line)
        report["script_differences"].append({"field": name, "agents": fr.value, "ocr": line,
                                             "similarity": round(value_ratio, 2)})


def ground_fields(fields: dict[str, FieldResult], ev: OcrEvidence, ctx: Context, n_agents: int) -> dict:
    report: dict = {"grounded": [], "not_grounded": [], "calendar_corrections": [], "completions": [],
                    "arbitrated": [], "script_differences": []}
    # Machine-read values, and values read beside the card's own printed label
    # (already an OCR reading) are not re-spelled by OCR.
    verified = {"barcode", "mrz", "printed_labels"}

    for name in DATE_FIELDS + ("document_number", "personal_number", "blood_type") + _TEXT_FIELDS:
        fr = fields[name]
        if not fr.value or verified & set(fr.verified_by):
            continue
        if name in ("document_number", "personal_number"):
            completion = _complete_number(compact(fr.value), ev, ctx.country, fields)
            if completion:
                ctx.warn(f"{name}: agents reported {fr.value!r}, one group of the number {completion!r} "
                         f"printed on the document — replaced with the full number")
                fr.alternatives = [fr.value] + fr.alternatives
                fr.value, fr.source, fr.confidence = completion, "ocr", 0.8
                fr.verified_by = ["ocr"]
                fr.flags.append("completed_from_ocr")
                report["completions"].append({"field": name, "value": completion})
                continue

        grounded = is_grounded(name, fr.value, ev)
        if (not grounded and name in DATE_FIELDS and {"calendar_pair", "printed_text"} & set(fr.verified_by)
                and fr.value in _all_readings(ev.text)):
            # Day/month order already settled by a calendar pair or the printed text.
            grounded = True
        if grounded:
            _mark(fr, True, n_agents, name, ctx)
            report["grounded"].append(name)
            continue
        if name in DATE_FIELDS and _calendar_fix(fields, name, ev, ctx, report):
            continue
        if name in DATE_FIELDS:
            swapped = _swap_day_month(fr.value)
            if swapped and swapped in ev.gregorian_dates:
                _promote(fr, swapped, name, f"printed as {ev.gregorian_dates[swapped]!r} — day and month were swapped", ctx, 0.85)
                fr.flags.append("day_month_corrected")
                report["arbitrated"].append(name)
                continue
        if name in _TEXT_FIELDS and not is_latin(fr.value):
            _script_arbitration(fr, name, ev, ctx, report)
            continue
        if grounded is None:
            continue

        # Not on the document: a near-miss spelling of what OCR read keeps the
        # agents' structure; otherwise prefer another agent's candidate that is printed.
        if name in _TEXT_FIELDS:
            corrected = _ocr_spell_correct(fr.value, ev)
            if corrected and is_grounded(name, corrected, ev):
                _promote(fr, corrected, name, "OCR spelling of the same name", ctx, 0.8)
                fr.flags.append("ocr_spelling_corrected")
                report["arbitrated"].append(name)
                continue
        alt = next((a for a in fr.alternatives if is_grounded(name, a, ev)), None)
        if alt is None and name == "full_name":
            other = fields["full_name_latin"].value
            if other and is_grounded(name, other, ev) and SequenceMatcher(
                    None, _fold(fr.value).upper(), _fold(other).upper()).ratio() >= 0.8:
                alt = other
        if alt is None and name == "document_number":
            other = fields["personal_number"].value
            if other and is_grounded(name, other, ev) and _hamming_close(compact(fr.value), compact(other)):
                alt = other
        if alt is not None:
            _promote(fr, alt, name, "the agents' own alternative is printed on the document", ctx)
            report["arbitrated"].append(name)
            continue
        if name == "full_name_latin" and not is_latin(fields["full_name_local"].value or "a"):
            # A romanization the prompt asks for (Pinyin, Hepburn, ...) — not printed, not a hallucination.
            fr.flags.append("transliteration_not_printed")
            fr.confidence = min(fr.confidence, 0.7)
            continue
        _mark(fr, False, n_agents, name, ctx)
        report["not_grounded"].append(name)
    if ev.pairs and fields["calendar_system"].value.lower() in ("", "gregorian"):
        calendars = sorted({p.calendar.title() for p in ev.pairs})
        fields["calendar_system"] = FieldResult(value=" + ".join(["Gregorian"] + calendars), confidence=0.9,
                                                source="ocr", verified_by=["calendar_pairs"])
    return report


def _complete_number(number: str, ev: OcrEvidence, country: str, fields: dict[str, FieldResult]) -> Optional[str]:
    """A 1-6 digit value that is one printed group of a longer number."""
    if not number.isdigit() or len(number) > 6:
        return None
    schemes = SCHEMES.get(country, [])
    agent_numbers = [compact(fields[f].value) for f in ("document_number", "personal_number")]
    for run in ev.digit_runs + agent_numbers:
        if len(run) < len(number) + 4 or not run.startswith(number):
            continue
        if run in agent_numbers or any(s(run) is not None and s(run).format_valid for s in schemes):
            return run
    return None


def collect_evidence(rapid_text: str, image_paths: list[str], country: str, script: str) -> OcrEvidence:
    engines = ["rapidocr"] if rapid_text.strip() else []
    tess, langs = tesseract_text(image_paths, country, script)
    if tess.strip():
        engines.append(f"tesseract[{langs}]")
    return build_evidence("\n".join(t for t in (rapid_text, tess) if t), engines, country)


__all__ = ["OcrEvidence", "collect_evidence", "ground_fields", "rapid_ocr_text", "detect_script"]

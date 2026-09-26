"""
Universal printed-date normalizer -> ISO YYYY-MM-DD (Gregorian).

Handles what the agents hand back from documents worldwide:
  ISO / year-first      1990-05-15, 1990/05/15, 1990.05.15, 19900515
  day-month-year        15.05.1990, 15/05/1990, 15-05-1990
  month-first (US etc.) 05/15/1990 — only for MDY countries or when day > 12
  month names           15 MAY 1990, 15 MAI/MAY 1990, May 15, 1990, 15-Mär-1990
                        (any language idx.dates.month_number knows, plus
                        Ethiopian, Hijri and Persian month names)
  CJK                   1990年5月15日, 1990년 5월 15일, 令和5年3月1日, 平成2年
  calendar markers      "EC"/"ዓ.ም" (Ethiopian), "AH"/"هـ" (Hijri),
                        "SH"/"ه.ش" (Persian), "BE"/"พ.ศ." (Thai), 民國 (ROC)

The calendar comes from, in order: an explicit marker in the string, a
calendar-specific month name, then `calendar_hint` (the document's
calendar_system) — but a hint is only applied when the year itself is
implausible as Gregorian or the field is an "_orig" field, since agents
are told to put the already-converted Gregorian date in the main field.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Optional

from idx.dates import month_number

from . import calendars as cal

# ISO 3166-1 alpha-3 countries that print numeric dates month-first.
MDY_COUNTRIES = {"USA", "PHL", "FSM", "PLW", "MHL", "BLZ", "GUM", "ASM", "PRI", "VIR", "MNP"}

_ETH_MARKERS = re.compile(r"(?:\bE\.?\s?C\.?\b|ዓ\.?\s?ም\.?|\bethiopian\b|\bAmete\s+Mihret\b)", re.I)
_HIJRI_MARKERS = re.compile(r"(?:\bA\.?\s?H\.?\b|هـ|\bhijri\b|\bhegira\b|\bhijra\b)", re.I)
_PERSIAN_MARKERS = re.compile(r"(?:\bS\.?\s?H\.?\b|ه\.?\s?ش|\bjalali\b|\bshamsi\b|\bsolar hijri\b)", re.I)
_THAI_MARKERS = re.compile(r"(?:\bB\.?\s?E\.?\b|พ\.?\s?ศ\.?|\bbuddhist\b)", re.I)
_MINGUO_MARKERS = re.compile(r"(?:民國|民国|\bROC\b|\bminguo\b)", re.I)
_JP_ERA = re.compile(r"(令和|平成|昭和|大正|reiwa|heisei|sh[oō]wa|taish[oō])\s*(\d{1,2}|元)\s*年?", re.I)
_CJK_YMD = re.compile(r"(\d{2,4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]?")
_YEAR_FIRST = re.compile(r"(?<!\d)(\d{4})[-/.\s](\d{1,2})[-/.\s](\d{1,2})(?!\d)")
_DAY_FIRST = re.compile(r"(?<!\d)(\d{1,2})([-/.\s])(\d{1,2})\2(\d{4}|\d{2})(?!\d)")
_COMPACT = re.compile(r"(?<!\d)(\d{8})(?!\d)")
_WORD = re.compile(r"[^\W\d_]+(?:[\s'-][^\W\d_]+)*", re.U)

_CALENDAR_ALIASES = {
    "gregorian": "gregorian", "western": "gregorian",
    "ethiopian": "ethiopian", "ethiopic": "ethiopian", "ec": "ethiopian", "geez": "ethiopian",
    "hijri": "hijri", "islamic": "hijri", "lunar hijri": "hijri", "umm al-qura": "hijri",
    "persian": "persian", "jalali": "persian", "solar hijri": "persian", "shamsi": "persian", "iranian": "persian",
    "thai": "thai", "buddhist": "thai",
    "minguo": "minguo", "roc": "minguo", "taiwanese": "minguo",
    "japanese": "japanese", "imperial": "japanese",
    "hebrew": "hebrew", "jewish": "hebrew",
    "nepali": "bikram_sambat", "bikram sambat": "bikram_sambat", "vikram samvat": "bikram_sambat",
}

_UNCONVERTIBLE = {"hebrew", "bikram_sambat"}

# The non-Gregorian calendar a country's documents print alongside (or
# instead of) Gregorian dates. Used when an agent mislabels calendar_system.
COUNTRY_CALENDARS = {
    "ETH": "ethiopian", "SAU": "hijri", "YEM": "hijri", "IRN": "persian", "AFG": "persian",
    "THA": "thai", "TWN": "minguo", "JPN": "japanese", "NPL": "bikram_sambat", "ISR": "hebrew",
}

# Year ranges in which a year is clearly NOT Gregorian for that calendar.
_NON_GREGORIAN_YEARS = {
    "hijri": range(1300, 1501),
    "persian": range(1280, 1451),
    "thai": range(2400, 2701),
    "minguo": range(1, 200),
}


@dataclass
class DateResult:
    iso: str = ""
    calendar: str = "gregorian"
    original: str = ""
    approximate: bool = False
    ambiguous: bool = False
    note: str = ""


def normalize_calendar_name(value: str) -> str:
    v = (value or "").strip().lower()
    if not v:
        return ""
    for alias, canonical in _CALENDAR_ALIASES.items():
        if alias in v:
            return canonical
    return ""


def _fold(token: str) -> str:
    token = unicodedata.normalize("NFKC", token).strip().lower().strip(".,")
    return " ".join(token.replace("-", " ").split())


_MONTH_TABLES = (("ethiopian", cal.ETHIOPIAN_MONTH_LOOKUP),
                 ("hijri", cal.HIJRI_MONTH_LOOKUP),
                 ("persian", cal.PERSIAN_MONTH_LOOKUP))


def _calendar_month(token: str, prefer: str = "") -> tuple[str, int] | None:
    """Month names overlap across calendars ("Tir" is Ethiopian month 5 and
    Persian month 4), so an already-known calendar's table is tried first."""
    folded = _fold(token)
    tables = sorted(_MONTH_TABLES, key=lambda t: t[0] != prefer)
    for calendar, table in tables:
        for name, month in table.items():
            if folded == _fold(name):
                return calendar, month
    return None


def _explicit_calendar(text: str) -> str:
    if _ETH_MARKERS.search(text):
        return "ethiopian"
    if _PERSIAN_MARKERS.search(text):
        return "persian"
    if _HIJRI_MARKERS.search(text):
        return "hijri"
    if _THAI_MARKERS.search(text):
        return "thai"
    if _MINGUO_MARKERS.search(text):
        return "minguo"
    return ""


def _expand_two_digit_year(yy: int, *, future_leaning: bool) -> int:
    if future_leaning:
        return 2000 + yy
    current = date.today().year % 100
    return (2000 if yy <= current else 1900) + yy


def _to_gregorian(calendar: str, y: int, m: int, d: int) -> tuple[date, bool]:
    """Returns (date, approximate)."""
    if calendar == "ethiopian":
        return cal.ethiopian_to_gregorian(y, m, d), False
    if calendar == "hijri":
        return cal.hijri_to_gregorian(y, m, d), True
    if calendar == "persian":
        return cal.persian_to_gregorian(y, m, d), False
    if calendar == "thai":
        return date(cal.thai_buddhist_to_gregorian_year(y), m, d), False
    if calendar == "minguo":
        return date(cal.minguo_to_gregorian_year(y), m, d), False
    return date(y, m, d), False  # gregorian, and japanese (era year already converted)


def _components(text: str, *, country: str, future_leaning: bool, prefer_calendar: str = "") -> Optional[tuple[int, int, int, str, bool]]:
    """(year, month, day, calendar_from_month_name, ambiguous) or None."""
    era = _JP_ERA.search(text)
    if era:
        era_year = 1 if era.group(2) == "元" else int(era.group(2))
        year = cal.japanese_era_to_gregorian_year(era.group(1), era_year)
        rest = text[era.end():]
        md = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})", rest)
        if md:
            return year, int(md.group(1)), int(md.group(2)), "japanese", False

    m = _CJK_YMD.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), "", False

    m = _YEAR_FIRST.search(text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), "", False

    # Month written as a word: "15 MAY 1990", "May 15, 1990", "15 Meskerem 1982".
    words = [w for w in _WORD.findall(text)]
    for word in words:
        month_cal = _calendar_month(word, prefer_calendar)
        month = month_cal[1] if month_cal else month_number(word)
        if not month and "/" in word:
            month = next((month_number(p) for p in word.split("/") if month_number(p)), None)
        if not month:
            continue
        numbers = [int(n) for n in re.findall(r"\d+", text)]
        years = [n for n in numbers if n >= 100]
        small = [n for n in numbers if n < 100]
        if years and small:
            return years[0], month, small[0], month_cal[0] if month_cal else "", False
        if len(small) == 2:
            day, yy = small
            return _expand_two_digit_year(yy, future_leaning=future_leaning), month, day, (month_cal[0] if month_cal else ""), False

    m = _DAY_FIRST.search(text)
    if m:
        a, sep, b, y = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        year = int(y) if len(y) == 4 else _expand_two_digit_year(int(y), future_leaning=future_leaning)
        if a > 12 and b <= 12:
            return year, b, a, "", False
        if b > 12 and a <= 12:
            return year, a, b, "", False
        ambiguous = a != b
        if country in MDY_COUNTRIES and sep == "/":
            return year, a, b, "", ambiguous
        return year, b, a, "", ambiguous

    m = _COMPACT.search(text)
    if m:
        s = m.group(1)
        y, mo, d = int(s[:4]), int(s[4:6]), int(s[6:])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return y, mo, d, "", False
    return None


def normalize_date(
    raw: str,
    *,
    calendar_hint: str = "",
    country: str = "",
    is_original_field: bool = False,
    future_leaning: bool = False,
) -> DateResult:
    text = (raw or "").strip()
    result = DateResult(original=text)
    if not text:
        return result

    explicit = _explicit_calendar(text)
    parts = _components(text, country=country, future_leaning=future_leaning, prefer_calendar=explicit)
    hinted = normalize_calendar_name(calendar_hint)
    if parts is None:
        if hinted in _UNCONVERTIBLE and is_original_field:
            result.calendar = hinted
            result.note = f"{hinted} calendar date — not auto-converted"
        else:
            result.note = "unparseable date"
        return result
    y, m, d, month_calendar, ambiguous = parts
    result.ambiguous = ambiguous

    calendar = explicit or month_calendar
    if not calendar:
        if hinted and hinted != "gregorian":
            implausible = y in _NON_GREGORIAN_YEARS.get(hinted, range(0))
            if is_original_field or implausible:
                calendar = hinted
    if not calendar and y in _NON_GREGORIAN_YEARS["thai"] and country == "THA":
        calendar = "thai"
    calendar = calendar or "gregorian"
    result.calendar = calendar

    if calendar in _UNCONVERTIBLE:
        result.note = f"{calendar} calendar date — not auto-converted"
        return result

    try:
        converted, approximate = _to_gregorian(calendar, y, m, d)
    except ValueError as e:
        result.note = str(e)
        return result
    result.iso = converted.isoformat()
    result.approximate = approximate
    if approximate:
        result.note = "tabular Hijri conversion — may differ from Umm al-Qura by 1-2 days"
    elif ambiguous:
        result.note = "day/month order ambiguous — resolved by the issuing country's convention"
    return result

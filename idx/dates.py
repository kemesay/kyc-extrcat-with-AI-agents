"""Printed-date parsing for the jurisdiction-agnostic OCR path.

Unlike AAMVA (which may assume MDY), this module never guesses a
convention from country. It only resolves a date when:

- the numbers themselves make the order unambiguous (day > 12), or
- both non-year segments are equal (``09.09.1997`` is the same day
  in DMY and MDY), or
- the printed separators are dots (``06.05.2035``) — ICAO/EU VIZ
  numeric dates, not US slash-MDY — or
- two dates on the same field differ by 7 or 8 years — the dual-calendar
  pair printed on Ethiopian (and some other) documents, where the later
  year is Gregorian. Ambiguous Gregorian twins then use DMY, which is
  how those cards print the numeric GC date (confirmed on a real
  Ethiopian driving licence: ``29/5/1987`` EC beside ``2/6/1995`` GC), or
- the month is printed as letters (``29 MAR 2007``, bilingual
  ``29 MAA/MAR 2007``). Two month abbreviations on either side of a slash
  must resolve to the *same* calendar month; disagreement is left unread.

Compact OCR glitches such as ``2951987`` (separators dropped) are
expanded before the same rules run.
"""

from __future__ import annotations

from datetime import date
import re
import unicodedata

# 3–4 letter month tokens as printed on ICAO-style VIZ dates (English plus
# the other Latin-script abbreviations that appear on the same bilingual
# slash line). Keys are NFKD-stripped, casefolded, no jurisdiction branch.
_MONTH_NUM: dict[str, int] = {
    "jan": 1, "ene": 1, "gen": 1, "january": 1, "januari": 1, "janvier": 1,
    "feb": 2, "fev": 2, "fév": 2, "february": 2, "februari": 2, "fevrier": 2,
    "mar": 3, "maa": 3, "mrt": 3, "mär": 3, "march": 3, "maart": 3, "mars": 3,
    "apr": 4, "avr": 4, "abr": 4, "april": 4, "avril": 4,
    "may": 5, "mei": 5, "mai": 5, "mag": 5,
    "jun": 6, "juin": 6, "giu": 6, "june": 6, "juni": 6,
    "jul": 7, "juil": 7, "lug": 7, "july": 7, "juli": 7, "juillet": 7,
    "aug": 8, "ago": 8, "aou": 8, "aoû": 8, "august": 8, "augustus": 8, "aout": 8,
    "sep": 9, "set": 9, "september": 9, "septembre": 9,
    "oct": 10, "okt": 10, "ott": 10, "out": 10, "october": 10, "oktober": 10, "octobre": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "dez": 12, "dic": 12, "déc": 12, "december": 12, "decembre": 12,
}

_BILINGUAL_MONTH_DATE_RE = re.compile(
    r"\b(\d{1,2})\s*([A-Za-zÀ-ÿ]{3,4})\s*/\s*([A-Za-zÀ-ÿ]{3,4})\s*(\d{2,4})\b",
    re.IGNORECASE,
)
_SINGLE_MONTH_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-zÀ-ÿ]{3,9})\s+(\d{2,4})\b",
    re.IGNORECASE,
)
_YEAR_MONTH_DAY_RE = re.compile(
    r"\b(\d{4})[/-]([A-Za-zÀ-ÿ]{3,9})[/-](\d{1,2})\b",
    re.IGNORECASE,
)

_DATE_TOKEN_RE = re.compile(
    r"\d{1,2}\s*[A-Za-zÀ-ÿ]{3,4}\s*/\s*[A-Za-zÀ-ÿ]{3,4}\s*\d{2,4}"
    r"|\d{1,2}\s+[A-Za-zÀ-ÿ]{3,9}\s+\d{2,4}"
    r"|\d{4}[/-][A-Za-zÀ-ÿ]{3,9}[/-]\d{1,2}"
    r"|\d{4}[\s./-]+\d{1,2}[\s./-]+\d{1,2}"
    r"|\d{1,2}[\s./-]+\d{1,2}[\s./-]+\d{4}"
    r"|\d{1,2}[\s./-]+\d{5,6}"
    r"|\b\d{6,8}\b"
)


def repair_glued_alpha_dates(text: str) -> str:
    """RapidOCR glues ``29OKT/OCT1972`` and reads O as 0 (``290KT/0CT1972``).

    Repair before letter/digit unglue, which would otherwise split
    ``290KT`` into ``290 KT`` and lose the day/month boundary.
    """
    out = re.sub(r"0(?=[Kk][Tt])", "O", text or "")
    out = re.sub(r"0(?=[Cc][Tt])", "O", out)
    out = re.sub(
        r"\b(\d{1,2})([A-Za-zÀ-ÿ]{3,4}\s*/\s*[A-Za-zÀ-ÿ]{3,4})(\d{2,4})\b",
        r"\1 \2 \3",
        out,
    )
    out = re.sub(
        r"\b(\d{1,2})([A-Za-zÀ-ÿ]{3,4}\s*/\s*[A-Za-zÀ-ÿ]{3,4})\b",
        r"\1 \2",
        out,
    )
    out = re.sub(r"(/[A-Za-zÀ-ÿ]{3,4}\s*)(\d)[Oo]\b", r"\g<1>\g<2>0", out)
    # Glued alphabetic dates as printed on some licences (``06MAR1994``).
    out = re.sub(
        r"\b(\d{1,2})([A-Za-zÀ-ÿ]{3})(\d{4})\b",
        r"\1 \2 \3",
        out,
    )
    return out


def date_tokens(text: str) -> list[str]:
    repaired = repair_glued_alpha_dates(text)
    return [m.group(0).strip() for m in _DATE_TOKEN_RE.finditer(repaired)]


def expand_compact_date(raw: str) -> str:
    """``2951987`` → ``29/5/1987``; ``842025`` → ``8/4/2025``. Leaves
    already-separated dates alone.
    """
    stripped = raw.strip()
    if len([p for p in re.split(r"[\s./-]+", stripped) if p]) >= 3:
        return stripped
    digits = re.sub(r"\D", "", stripped)
    if not (6 <= len(digits) <= 8):
        return stripped
    year, rest = digits[-4:], digits[:-4]
    try:
        y = int(year)
    except ValueError:
        return stripped
    if not (1900 <= y <= 2100):
        return stripped
    if len(rest) == 2:
        return f"{int(rest[0])}/{int(rest[1])}/{year}"
    if len(rest) == 3:
        return f"{int(rest[:2])}/{int(rest[2])}/{year}"
    if len(rest) == 4:
        return f"{int(rest[:2])}/{int(rest[2:])}/{year}"
    return stripped


def _fold_month_token(token: str) -> str:
    stripped = unicodedata.normalize("NFKD", token or "")
    return "".join(c for c in stripped.casefold() if c.isalpha())


def month_number(token: str) -> int | None:
    """Map a printed month word/abbreviation to 1–12, or None.

    Prefix matching is only for 3–4 letter abbreviations (``MAR``,
    ``MAART``-style ``MAAR``). Longer tokens (``Marta``, ``Marcella``)
    are names, not months.
    """
    key = _fold_month_token(token)
    if key in _MONTH_NUM:
        return _MONTH_NUM[key]
    if 3 <= len(key) <= 4 and key[:3] in _MONTH_NUM and key[:3] != "jui":
        return _MONTH_NUM[key[:3]]
    return None


def _iso_from_dmy(day: str, month: int, year: str) -> str:
    if not day.isdigit() or not year.isdigit() or not month:
        return ""
    y = int(year)
    if len(year) == 2:
        y += 2000 if y < 50 else 1900
    try:
        return date(y, month, int(day)).isoformat()
    except ValueError:
        return ""


def alphabetic_date_to_iso(raw: str) -> str:
    """``29 MAA/MAR 2007``, ``23SEP/SEP 2027``, ``16 JUN 97``, ``1996/Nov/28``."""
    text = repair_glued_alpha_dates(" ".join((raw or "").split()))
    glued = re.sub(r"\b(\d{1,2})([A-Za-zÀ-ÿ]{3,4}\s*/)", r"\1 \2", text)
    m = _BILINGUAL_MONTH_DATE_RE.search(glued)
    if m:
        day, left, right, year = m.groups()
        a, b = month_number(left), month_number(right)
        if a and b and a == b:
            return _iso_from_dmy(day, a, year)
        return ""
    m = _SINGLE_MONTH_DATE_RE.search(text)
    if m:
        day, month, year = m.groups()
        num = month_number(month)
        if num:
            return _iso_from_dmy(day, num, year)
    m = _YEAR_MONTH_DAY_RE.search(text)
    if m:
        year, month, day = m.groups()
        num = month_number(month)
        if num:
            return _iso_from_dmy(day, num, year)
    return ""


def slash_date_to_iso(raw: str, *, prefer_dmy: bool = False) -> str:
    """See generic_ocr._slash_date_to_iso — prefer_dmy is only True when
    a dual-calendar pair has already identified this token as Gregorian.
    """
    alpha = alphabetic_date_to_iso(raw)
    if alpha:
        return alpha
    expanded = expand_compact_date(raw.strip())
    parts = [p.strip() for p in re.split(r"[\s./-]+", expanded) if p.strip()]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return ""

    if len(parts[0]) == 4:
        yyyy, mm, dd = parts
        try:
            return date(int(yyyy), int(mm), int(dd)).isoformat()
        except ValueError:
            return ""

    if len(parts[2]) != 4:
        return ""
    yyyy = int(parts[2])
    a, b = int(parts[0]), int(parts[1])
    dotted = bool(re.search(r"\d\s*\.\s*\d", raw)) and "/" not in raw
    if a > 12 >= b:
        dd, mm = a, b
    elif b > 12 >= a:
        mm, dd = a, b
    elif a == b:
        dd, mm = a, b
    elif prefer_dmy or dotted:
        # ICAO/EU VIZ prints DD.MM.YYYY with dots. Slash dates stay
        # unresolved when both segments are months (US MDY vs world DMY).
        dd, mm = a, b
    else:
        return ""
    try:
        return date(yyyy, mm, dd).isoformat()
    except ValueError:
        return ""


def _year_of(raw: str) -> int | None:
    expanded = expand_compact_date(raw.strip())
    parts = [p.strip() for p in re.split(r"[\s./-]+", expanded) if p.strip()]
    if len(parts) == 3 and parts[2].isdigit() and len(parts[2]) == 4:
        return int(parts[2])
    if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 4:
        return int(parts[0])
    m = re.search(r"((?:19|20)\d{2})", expanded)
    return int(m.group(1)) if m else None


def resolve_printed_dates(raws: list[str]) -> str:
    """Pick the Gregorian ISO date from one or more printed forms next to
    the same label. Empty string when nothing is safely resolvable.
    """
    cleaned = [r.strip() for r in raws if r and r.strip()]
    if not cleaned:
        return ""

    years = [_year_of(r) for r in cleaned]
    gregorian_index: int | None = None
    for i, yi in enumerate(years):
        if yi is None:
            continue
        for j, yj in enumerate(years):
            if i == j or yj is None:
                continue
            if (max(yi, yj) - min(yi, yj)) in (7, 8):
                gregorian_index = i if yi > yj else j
                break
        if gregorian_index is not None:
            break

    if gregorian_index is not None:
        iso = slash_date_to_iso(cleaned[gregorian_index], prefer_dmy=True)
        if iso:
            return iso

    for raw in cleaned:
        iso = slash_date_to_iso(raw)
        if iso:
            return iso
    return ""


def unique_resolved_dates(text: str) -> list[str]:
    """Distinct ISO dates found anywhere in ``text``, calendar-sorted."""
    seen: list[str] = []
    for raw in date_tokens(text):
        iso = slash_date_to_iso(raw)
        if iso and iso not in seen:
            seen.append(iso)
    seen.sort()
    return seen


def assign_life_cycle_dates(
    iso_dates: list[str], *, today: date | None = None,
) -> dict[str, str]:
    """Map unlabeled dates onto birth / issue / expiry by calendar order.

    Those three fields have a meaning that does not depend on layout:
    a person is born before the document is issued, and the document
    expires on or after the issue date. Applied only when there are
    exactly three distinct resolvable dates that satisfy that order.
    Two dates are left unassigned — birth vs issue is then a guess.
    """
    today = today or date.today()
    uniq = sorted({d for d in iso_dates if d})
    if len(uniq) == 2:
        earlier, later = (date.fromisoformat(d) for d in uniq)
        if later < earlier:
            return {}
        age = (today - earlier).days / 365.25
        if 15 <= age <= 120:
            return {
                "date_of_birth": earlier.isoformat(),
                "date_of_expiry": later.isoformat(),
            }
        if 0 <= age < 15:
            return {
                "date_of_issue": earlier.isoformat(),
                "date_of_expiry": later.isoformat(),
            }
        return {}
    if len(uniq) != 3:
        return {}
    born, issued, expires = (date.fromisoformat(d) for d in uniq)
    age = (today - born).days / 365.25
    if not (0 <= age <= 120):
        return {}
    if issued < born or expires < issued:
        return {}
    if issued.year > today.year + 1:
        return {}
    return {
        "date_of_birth": born.isoformat(),
        "date_of_issue": issued.isoformat(),
        "date_of_expiry": expires.isoformat(),
    }


def years_are_dual_calendar(iso_a: str, iso_b: str) -> bool:
    """True when two ISO dates are 7 or 8 years apart (Ethiopian/Gregorian)."""
    try:
        ya, yb = int(iso_a[:4]), int(iso_b[:4])
    except (TypeError, ValueError):
        return False
    return abs(ya - yb) in (7, 8)


def repair_duplicate_life_cycle(fields: dict[str, str], iso_dates: list[str]) -> None:
    """Birth, issue, and expiry are never the same printed day. If two
    slots collapsed onto one ISO value, re-assign from the distinct
    dates on the card or drop the duplicate rather than keep a lie.
    """
    assigned = assign_life_cycle_dates(iso_dates)
    dob = fields.get("date_of_birth", "")
    iss = fields.get("date_of_issue", "")
    exp = fields.get("date_of_expiry", "")

    if dob and iss and dob == iss:
        replacement = assigned.get("date_of_issue", "")
        if replacement and replacement != dob:
            fields["date_of_issue"] = replacement
        else:
            fields.pop("date_of_issue", None)
        iss = fields.get("date_of_issue", "")

    if dob and exp and dob == exp:
        replacement = assigned.get("date_of_expiry", "")
        if replacement and replacement != dob:
            fields["date_of_expiry"] = replacement
        else:
            fields.pop("date_of_expiry", None)
        exp = fields.get("date_of_expiry", "")

    if iss and exp and iss == exp:
        if assigned.get("date_of_issue") and assigned.get("date_of_expiry"):
            fields["date_of_issue"] = assigned["date_of_issue"]
            fields["date_of_expiry"] = assigned["date_of_expiry"]


def iso_from_yymmdd(yymmdd: str, *, today: date | None = None) -> str:
    """YYMMDD as printed in a hyphenated personal id. Empty when both
    19xx and 20xx are plausible ages (do not guess the century).
    """
    today = today or date.today()
    digits = re.sub(r"\D", "", yymmdd or "")
    if len(digits) != 6 or not digits.isdigit():
        return ""
    yy, mm, dd = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
    plausible: list[date] = []
    for century in (1900, 2000):
        try:
            parsed = date(century + yy, mm, dd)
        except ValueError:
            continue
        age = (today - parsed).days / 365.25
        if 0 <= age <= 120:
            plausible.append(parsed)
    if len(plausible) != 1:
        return ""
    return plausible[0].isoformat()

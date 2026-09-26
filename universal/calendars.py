"""
Calendar arithmetic — so non-Gregorian dates are converted by code, not by
asking a 7B model to do date math (which it gets wrong often enough to
matter for KYC).

Exact conversions:     Ethiopian (Amete Mihret), Persian/Solar Hijri
                       (Jalali), Thai Buddhist Era, Minguo (ROC/Taiwan),
                       Japanese imperial eras.
Approximate:           Islamic (Hijri) — the arithmetic/tabular calendar.
                       Saudi Umm al-Qura and sighting-based calendars can
                       differ by 1-2 days, so results are flagged approximate.
Not converted:         Hebrew, Nepali Bikram Sambat (both need published
                       lookup tables) — reported with the original kept.

Ethiopian and Islamic epochs follow Reingold & Dershowitz, "Calendrical
Calculations", in R.D. fixed-day numbers — identical to Python's
date.toordinal(), which is what makes these one-liners.
"""

from __future__ import annotations

from datetime import date

_ETHIOPIC_EPOCH = 2796      # R.D. of 1 Meskerem 1 E.C. (Aug 29, 8 CE Julian)
_ISLAMIC_EPOCH = 227015     # R.D. of 1 Muharram 1 AH (Jul 16, 622 CE Julian)

ETHIOPIAN_MONTHS = {
    # Latin transliterations, several spellings each, then Ge'ez script.
    1: ("meskerem", "meskerm", "መስከረም"),
    2: ("tikimt", "tekemt", "tikemt", "tiqimt", "ጥቅምት"),
    3: ("hidar", "hedar", "ኅዳር", "ህዳር", "ሕዳር"),
    4: ("tahsas", "tahesas", "tahisas", "ታኅሣሥ", "ታህሳስ", "ታሕሳስ"),
    5: ("tir", "ter", "tirr", "ጥር"),
    6: ("yekatit", "yekatit", "yakatit", "የካቲት"),
    7: ("megabit", "magabit", "መጋቢት"),
    8: ("miyazya", "miazia", "miyazia", "ሚያዝያ"),
    9: ("ginbot", "genbot", "ግንቦት"),
    10: ("sene", "senie", "ሰኔ"),
    11: ("hamle", "hamile", "ሐምሌ", "ሃምሌ"),
    12: ("nehase", "nehasse", "nehasie", "ነሐሴ", "ነሃሴ"),
    13: ("pagume", "pagumen", "pagumie", "ጳጉሜ", "ጳጉሜን"),
}

HIJRI_MONTHS = {
    1: ("muharram", "moharram", "محرم"),
    2: ("safar", "صفر"),
    3: ("rabi al-awwal", "rabi al awwal", "rabi i", "rabia al awal", "ربيع الأول", "ربيع الاول"),
    4: ("rabi al-thani", "rabi al thani", "rabi ii", "rabi al-akhir", "ربيع الثاني", "ربيع الآخر"),
    5: ("jumada al-awwal", "jumada al awwal", "jumada i", "جمادى الأولى", "جمادى الاولى"),
    6: ("jumada al-thani", "jumada al thani", "jumada ii", "jumada al-akhirah", "جمادى الآخرة", "جمادى الثانية"),
    7: ("rajab", "رجب"),
    8: ("shaban", "sha'ban", "shaaban", "شعبان"),
    9: ("ramadan", "ramadhan", "رمضان"),
    10: ("shawwal", "shawal", "شوال"),
    11: ("dhu al-qadah", "dhul qadah", "dhu al-qi'dah", "zulqadah", "ذو القعدة"),
    12: ("dhu al-hijjah", "dhul hijjah", "zulhijjah", "ذو الحجة"),
}

PERSIAN_MONTHS = {
    1: ("farvardin", "فروردین"), 2: ("ordibehesht", "اردیبهشت"), 3: ("khordad", "خرداد"),
    4: ("tir", "تیر"), 5: ("mordad", "مرداد"), 6: ("shahrivar", "شهریور"),
    7: ("mehr", "مهر"), 8: ("aban", "آبان"), 9: ("azar", "آذر"),
    10: ("dey", "دی"), 11: ("bahman", "بهمن"), 12: ("esfand", "اسفند"),
}

JAPANESE_ERAS = {
    # era -> offset such that gregorian_year = offset + era_year
    "令和": 2018, "reiwa": 2018, "r": 2018,
    "平成": 1988, "heisei": 1988, "h": 1988,
    "昭和": 1925, "showa": 1925, "shōwa": 1925, "s": 1925,
    "大正": 1911, "taisho": 1911, "taishō": 1911, "t": 1911,
}


def _month_lookup(table: dict[int, tuple[str, ...]]) -> dict[str, int]:
    return {name: month for month, names in table.items() for name in names}


ETHIOPIAN_MONTH_LOOKUP = _month_lookup(ETHIOPIAN_MONTHS)
HIJRI_MONTH_LOOKUP = _month_lookup(HIJRI_MONTHS)
PERSIAN_MONTH_LOOKUP = _month_lookup(PERSIAN_MONTHS)


def ethiopian_to_gregorian(year: int, month: int, day: int) -> date:
    if not (1 <= month <= 13 and 1 <= day <= 30) or (month == 13 and day > 6):
        raise ValueError(f"invalid Ethiopian date {year}-{month}-{day}")
    fixed = _ETHIOPIC_EPOCH - 1 + 365 * (year - 1) + year // 4 + 30 * (month - 1) + day
    return date.fromordinal(fixed)


def gregorian_to_ethiopian(d: date) -> tuple[int, int, int]:
    fixed = d.toordinal()
    year = (4 * (fixed - _ETHIOPIC_EPOCH) + 1463) // 1461
    month = (fixed - ethiopian_to_gregorian(year, 1, 1).toordinal()) // 30 + 1
    day = fixed + 1 - ethiopian_to_gregorian(year, month, 1).toordinal()
    return year, month, day


def hijri_to_gregorian(year: int, month: int, day: int) -> date:
    """Tabular Islamic calendar — may differ from Umm al-Qura by 1-2 days."""
    if not (1 <= month <= 12 and 1 <= day <= 30):
        raise ValueError(f"invalid Hijri date {year}-{month}-{day}")
    fixed = (day + 29 * (month - 1) + (6 * month - 1) // 11 + (year - 1) * 354
             + (3 + 11 * year) // 30 + _ISLAMIC_EPOCH - 1)
    return date.fromordinal(fixed)


def persian_to_gregorian(year: int, month: int, day: int) -> date:
    """Solar Hijri (Jalali) — the standard 33-year-cycle arithmetic."""
    if not (1 <= month <= 12 and 1 <= day <= 31) or (month > 6 and day > 30):
        raise ValueError(f"invalid Persian date {year}-{month}-{day}")
    jy = year + 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + day
    days += (month - 1) * 31 if month < 7 else (month - 7) * 30 + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    month_lengths = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while gm < 12 and gd > month_lengths[gm]:
        gd -= month_lengths[gm]
        gm += 1
    return date(gy, gm + 1, gd)


def thai_buddhist_to_gregorian_year(year: int) -> int:
    return year - 543


def minguo_to_gregorian_year(year: int) -> int:
    return year + 1911


def japanese_era_to_gregorian_year(era: str, era_year: int) -> int:
    offset = JAPANESE_ERAS.get(era.strip().lower()) or JAPANESE_ERAS.get(era.strip())
    if offset is None:
        raise ValueError(f"unknown Japanese era {era!r}")
    return offset + era_year

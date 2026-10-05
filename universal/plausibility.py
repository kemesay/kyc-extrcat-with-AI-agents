"""
Cross-field plausibility — "is this value physically possible", never "is
this person eligible" (same line idx/validation.py draws). Also checks the
dual-calendar pair many documents print (Ethiopian EC + GC, Hijri + Gregorian):
the agents are asked for both, and the pair must agree once the original is
converted by code.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from .dates import COUNTRY_CALENDARS, normalize_calendar_name, normalize_date
from .reconcile import Context, FieldResult

MAX_AGE_YEARS = 120


def _iso(fields: dict[str, FieldResult], name: str) -> Optional[date]:
    try:
        return date.fromisoformat(fields[name].value) if fields[name].value else None
    except ValueError:
        return None


def check_dual_calendar(fields: dict[str, FieldResult], ctx: Context, printed_text: Optional[str] = None) -> Optional[dict]:
    orig = fields["date_of_birth_orig"].value
    if not orig:
        return None
    if "calendar_not_printed" in fields["date_of_birth"].flags:
        return {"original": orig, "note": "calendar not printed — left for review"}
    if printed_text is not None and "calendar_pair" not in fields["date_of_birth_orig"].verified_by \
            and fields["date_of_birth_orig"].source != "printed_text":
        # An agent's "original-calendar date" counts only if that date is printed on the card.
        key = tuple(int(n) for n in re.findall(r"\d+", orig))
        printed_keys = {tuple(int(n) for n in re.findall(r"\d+", m.group()))
                        for m in re.finditer(r"\d{1,2}\s?[./\-]\s?\d{1,2}\s?[./\-]\s?\d{2,4}", printed_text)}
        if key and key not in printed_keys:
            fields["date_of_birth_orig"].alternatives = [orig] + fields["date_of_birth_orig"].alternatives
            fields["date_of_birth_orig"].value = ""
            fields["date_of_birth_orig"].flags.append("not_printed")
            ctx.warn(f"date_of_birth_orig: {orig!r} is not printed on the document — not used")
            return {"original": orig, "note": "not printed on the document — ignored"}
    if "calendar_pair" in fields["date_of_birth"].verified_by:
        # Both calendars were read and agree by arithmetic: nothing left to check.
        return {"original": orig, "agrees_with_date_of_birth": True, "note": "settled by the printed calendar pair"}
    if re.search(r"(?<![A-Za-z])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", orig, re.I):
        # '07/FEB 1995': a Gregorian month name — an agent's mix of the two halves,
        # not an Ethiopian date. Converting it as one invents a birth date.
        fields["date_of_birth_orig"].alternatives = [orig]
        fields["date_of_birth_orig"].value = ""
        fields["date_of_birth_orig"].flags.append("not_an_original_calendar_date")
        return {"original": orig, "note": "contains a Gregorian month name — not an original-calendar date"}
    hint = normalize_calendar_name(ctx.calendar)
    if hint in ("", "gregorian"):
        hint = COUNTRY_CALENDARS.get(ctx.country, "")
    converted = normalize_date(orig, calendar_hint=hint, country=ctx.country, is_original_field=True)
    if converted.calendar == "gregorian":
        return None
    report = {"original": orig, "calendar": converted.calendar, "converted": converted.iso,
              "approximate": converted.approximate, "note": converted.note}
    dob = fields["date_of_birth"]
    if not converted.iso:
        return report
    expected = {COUNTRY_CALENDARS.get(ctx.country)} - {None, ""}
    # Only a country known to print this calendar may let it override anything;
    # agents append a bogus 'EC' to Gregorian dates on non-Ethiopian cards.
    plausible_calendar = bool(ctx.country) and converted.calendar in expected
    plain = normalize_date(re.sub(r"\b(E\.?C\.?|A\.?H\.?|S\.?H\.?)\b|ዓ\.?ም\.?", "", orig), country=ctx.country)
    if not plausible_calendar and plain.iso and plain.iso == dob.value:
        # The "original" is the Gregorian birth date itself with a calendar label tacked on.
        fields["date_of_birth_orig"].alternatives = [orig]
        fields["date_of_birth_orig"].value = ""
        fields["date_of_birth_orig"].flags.append("dropped_mislabelled_calendar")
        report["note"] = "the 'original-calendar' date was the Gregorian birth date with a calendar label added"
        return report
    dob_verified = bool({"mrz", "barcode"} & set(dob.verified_by)) or any(
        v.startswith(("checksum:", "id_number:")) for v in dob.verified_by)
    if not plausible_calendar:
        report["note"] = (f"{converted.calendar} calendar is not used by {ctx.country} documents — "
                          "original-calendar date ignored")
        fields["date_of_birth_orig"].flags.append("implausible_calendar_for_country")
        return report
    labelled_other = [f for f in ("issue_date", "expiry_date") if fields[f].value == converted.iso]
    if any(flag.startswith("labelled_as_") for flag in dob.flags) or labelled_other:
        # The 'original-calendar birth date' is another printed date (the issue date's
        # "25/10/2016 E.C" on a card that prints no birth date): it is not a birth date.
        fields["date_of_birth_orig"].alternatives = [orig]
        fields["date_of_birth_orig"].value = ""
        fields["date_of_birth_orig"].flags.append("is_another_printed_date")
        report["note"] = (f"{orig!r} is the card's {(labelled_other or ['labelled'])[0].replace('_', ' ')}, "
                          "not a birth date")
        return report
    if not dob.value:
        dob.value, dob.source, dob.confidence = converted.iso, "calendar_conversion", 0.7
        dob.verified_by = ["calendar_conversion"]
        ctx.warn(f"date_of_birth converted by code from the {converted.calendar} date {orig!r}")
        return report
    try:
        gap = abs((date.fromisoformat(dob.value) - date.fromisoformat(converted.iso)).days)
    except ValueError:
        return report
    tolerance = 2 if converted.approximate else 1
    report["agrees_with_date_of_birth"] = gap <= tolerance
    if gap <= tolerance:
        dob.verified_by.append("dual_calendar")
        dob.confidence = max(dob.confidence, 0.9)
    elif dob_verified:
        # A checksum/barcode-verified DOB outranks an agent's original-calendar read.
        fields["date_of_birth_orig"].flags.append("inconsistent_with_verified_date_of_birth")
        fields["date_of_birth_orig"].confidence = min(fields["date_of_birth_orig"].confidence, 0.3)
        ctx.warn(f"date_of_birth_orig {orig!r} contradicts the verified date_of_birth {dob.value}")
    else:
        dob.flags.append("dual_calendar_mismatch")
        ctx.warn(f"date_of_birth {dob.value} does not match the {converted.calendar} date "
                 f"{orig!r} (= {converted.iso} by calendar arithmetic)")
        if converted.calendar == "ethiopian" or not converted.approximate:
            # The agents' own EC->GC conversion is the likeliest error; prefer arithmetic.
            dob.alternatives = [dob.value] + dob.alternatives
            dob.value, dob.source, dob.confidence = converted.iso, "calendar_conversion", 0.8
            ctx.review("date_of_birth: agents' Gregorian date contradicts the printed original-calendar date")
    return report


def check_plausibility(fields: dict[str, FieldResult], ctx: Context, today: Optional[date] = None) -> dict:
    today = today or date.today()
    if fields["issue_date"].value and fields["issue_date"].value == fields["date_of_birth"].value:
        # An agent copying the birth date into the issue slot — a document isn't issued at birth.
        fr = fields["issue_date"]
        ctx.warn(f"issue_date {fr.value} duplicates date_of_birth — dropped")
        fr.alternatives, fr.flags = [fr.value] + fr.alternatives, fr.flags + ["duplicates_date_of_birth"]
        fr.value, fr.confidence = "", 0.0
    dob, issue, expiry = _iso(fields, "date_of_birth"), _iso(fields, "issue_date"), _iso(fields, "expiry_date")
    printed = lambda n: bool({"printed_text", "calendar_pair", "printed_label", "mrz", "barcode"}
                             & set(fields[n].verified_by))
    if issue and expiry and expiry <= issue and printed("issue_date") and not printed("expiry_date") \
            and "not_found_in_ocr" in fields["expiry_date"].flags:
        # The contradiction is the unprinted date's: the printed issue date stands.
        fr = fields["expiry_date"]
        ctx.warn(f"expiry_date {fr.value} is before the printed issue date and not on the document per OCR — dropped")
        fr.alternatives, fr.flags = [fr.value] + fr.alternatives, fr.flags + ["dropped_impossible_unprinted"]
        fr.value, fr.confidence, expiry = "", 0.0, None
    impossible_issue = issue and (issue > today or (expiry and expiry <= issue) or (dob and issue < dob))
    if impossible_issue and "not_found_in_ocr" in fields["issue_date"].flags and not printed("issue_date"):
        # Physically impossible AND not printed anywhere per OCR: an invented value, not a finding.
        fr = fields["issue_date"]
        ctx.warn(f"issue_date {fr.value} is impossible and not on the document per OCR — dropped")
        fr.alternatives, fr.flags = [fr.value] + fr.alternatives, fr.flags + ["dropped_impossible_unprinted"]
        fr.value, fr.confidence, issue = "", 0.0, None
    problems: list[str] = []

    if dob:
        if dob > today:
            problems.append("date_of_birth is in the future")
        elif (today - dob).days > MAX_AGE_YEARS * 365.25:
            problems.append(f"date_of_birth implies an age over {MAX_AGE_YEARS}")
    if issue and issue > today:
        problems.append("issue_date is in the future")
    if dob and issue and issue < dob:
        problems.append("issue_date is before date_of_birth")
    if issue and expiry and expiry <= issue:
        problems.append("expiry_date is not after issue_date")
    if dob and expiry and expiry <= dob:
        problems.append("expiry_date is not after date_of_birth")

    for p in problems:
        ctx.warn(p)
        ctx.review(p)

    result: dict = {"problems": problems}
    if expiry:
        result["document_currently_valid"] = expiry >= today
        result["validity_checked_at"] = today.isoformat()
        if expiry < today:
            ctx.warn(f"document expired on {expiry.isoformat()}")
    return result

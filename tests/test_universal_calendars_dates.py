from datetime import date

import pytest

from universal import calendars as cal
from universal.dates import normalize_date


@pytest.mark.parametrize("ec, gc", [
    ((2016, 1, 1), date(2023, 9, 12)),    # Enkutatash before a Gregorian leap year
    ((2017, 1, 1), date(2024, 9, 11)),
    ((2012, 1, 1), date(2019, 9, 12)),
    ((2017, 4, 29), date(2025, 1, 7)),    # Genna
    ((1982, 5, 12), date(1990, 1, 20)),
])
def test_ethiopian(ec, gc):
    assert cal.ethiopian_to_gregorian(*ec) == gc
    assert cal.gregorian_to_ethiopian(gc) == ec


@pytest.mark.parametrize("fa, gc", [
    ((1403, 1, 1), date(2024, 3, 20)),
    ((1400, 1, 1), date(2021, 3, 21)),
    ((1379, 10, 11), date(2000, 12, 31)),
])
def test_persian(fa, gc):
    assert cal.persian_to_gregorian(*fa) == gc


@pytest.mark.parametrize("ah, umm_al_qura", [
    ((1446, 1, 1), date(2024, 7, 7)),
    ((1445, 9, 1), date(2024, 3, 11)),
    ((1444, 10, 1), date(2023, 4, 21)),
])
def test_hijri_within_tabular_tolerance(ah, umm_al_qura):
    assert abs((cal.hijri_to_gregorian(*ah) - umm_al_qura).days) <= 1


def test_invalid_ethiopian_pagume():
    with pytest.raises(ValueError):
        cal.ethiopian_to_gregorian(2016, 13, 7)


@pytest.mark.parametrize("raw, kwargs, iso, calendar", [
    ("1990-05-15", {}, "1990-05-15", "gregorian"),
    ("15.05.1990", {}, "1990-05-15", "gregorian"),
    ("05/15/1990", {}, "1990-05-15", "gregorian"),
    ("03/04/1990", {"country": "USA"}, "1990-03-04", "gregorian"),
    ("03/04/1990", {"country": "ETH"}, "1990-04-03", "gregorian"),
    ("15 MAY 1990", {}, "1990-05-15", "gregorian"),
    ("15 MAI/MAY 1990", {}, "1990-05-15", "gregorian"),
    ("May 15, 1990", {}, "1990-05-15", "gregorian"),
    ("15-Mär-1990", {}, "1990-03-15", "gregorian"),
    ("1990年5月15日", {}, "1990-05-15", "gregorian"),
    ("1990년 5월 15일", {}, "1990-05-15", "gregorian"),
    ("令和5年3月1日", {}, "2023-03-01", "japanese"),
    ("民國79年5月15日", {}, "1990-05-15", "minguo"),
    ("19 Meskerem 1989 EC", {}, "1996-09-29", "ethiopian"),
    ("12/05/1982 ዓ.ም", {}, "1990-01-20", "ethiopian"),
    ("01 Farvardin 1403", {}, "2024-03-20", "persian"),
    ("15 Tir 1402 SH", {}, "2023-07-06", "persian"),     # Persian Tir = month 4
    ("15 Tir 1982 EC", {}, "1990-01-23", "ethiopian"),   # Ethiopian Tir = month 5
    ("15/05/2533", {"country": "THA"}, "1990-05-15", "thai"),
    ("12/05/1982", {"calendar_hint": "Ethiopian", "is_original_field": True}, "1990-01-20", "ethiopian"),
])
def test_normalize_date(raw, kwargs, iso, calendar):
    result = normalize_date(raw, **kwargs)
    assert (result.iso, result.calendar) == (iso, calendar)


def test_hint_not_applied_to_plausible_gregorian_main_field():
    # Agents are told to put the converted Gregorian date in the main field.
    assert normalize_date("12/05/1990", calendar_hint="Ethiopian").calendar == "gregorian"


def test_hijri_flagged_approximate():
    result = normalize_date("15/03/1411", calendar_hint="Hijri")
    assert result.calendar == "hijri" and result.approximate


def test_hebrew_reported_not_converted():
    result = normalize_date("15 Nisan 5750", calendar_hint="Hebrew", is_original_field=True)
    assert result.iso == "" and result.calendar == "hebrew"

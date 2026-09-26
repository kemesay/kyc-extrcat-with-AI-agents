"""
TD2 MRZ parser — ID card / visa format, 2 lines x 36 chars.
ICAO Doc 9303 Part 6. Field layout (0-indexed):

Line 1 (36):
  0      document code (I/P/A/C)
  1      document code variant
  2-4    issuing state (3)
  5-35   name field: SURNAME<<GIVEN<NAMES<<<<... (31 chars)

Line 2 (36):
  0-8    document number (9)
  9      check digit — document number
  10-12  nationality (3)
  13-18  date of birth YYMMDD
  19     check digit — DOB
  20     sex (M/F/<)
  21-26  date of expiry YYMMDD
  27     check digit — expiry
  28-34  optional data (7)
  35     composite check digit over [0:10] + [13:20] + [21:35]
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

from .mrz_checksum import verify, strip_filler
from .countries import resolve_country_code, resolve_nationality_code

LINE_LEN = 36


@dataclass
class TD2Result:
    document_code: str = ""
    issuing_state: str = ""
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    nationality_name: str = ""
    date_of_birth: str = ""
    sex: str = ""
    date_of_expiry: str = ""
    optional_data: str = ""
    checks: dict[str, Any] = field(default_factory=dict[str, Any])
    valid: bool = False
    raw_lines: tuple[Any, ...] = ()


def looks_like_td2(line1: str, line2: str) -> bool:
    return (
        len(line1) == LINE_LEN and len(line2) == LINE_LEN
        and line1[0] in ("I", "P", "A", "C")
    )


def parse_td2(line1: str, line2: str) -> TD2Result:
    line1 = (line1 or "").upper().ljust(LINE_LEN, "<")[:LINE_LEN]
    line2 = (line2 or "").upper().ljust(LINE_LEN, "<")[:LINE_LEN]

    doc_number = line2[0:9]
    doc_number_check = line2[9]
    nationality = line2[10:13]
    dob = line2[13:19]
    dob_check = line2[19]
    sex = line2[20]
    expiry = line2[21:27]
    expiry_check = line2[27]
    optional_data = line2[28:35]
    composite_check = line2[35]

    name_raw = line1[5:36].rstrip("<")
    parts = name_raw.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given_names = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    issuing_state, _ = resolve_country_code(line1[2:5])
    nationality_code, nationality_name = resolve_nationality_code(nationality)

    checks = {
        "document_number": verify(doc_number, doc_number_check),
        "date_of_birth": verify(dob, dob_check),
        "date_of_expiry": verify(expiry, expiry_check),
        "composite": verify(
            line2[0:10] + line2[13:20] + line2[21:35], composite_check
        ),
    }
    # Unlike TD3 (see mrz_td3.py), composite is NOT relaxed out of `valid`
    # here — see the matching comment in mrz_td1.py for why: TD2 shares an
    # identical byte layout with TD3 for document_number/nationality/DOB/
    # expiry, so composite is what actually distinguishes the formats
    # rather than just re-checking noisy optional data.

    return TD2Result(
        document_code=line1[0:2].rstrip("<"),
        issuing_state=issuing_state,
        surname=surname,
        given_names=given_names,
        document_number=strip_filler(doc_number),
        nationality=nationality_code,
        nationality_name=nationality_name,
        date_of_birth=dob,
        sex=sex if sex in ("M", "F") else "",
        date_of_expiry=expiry,
        optional_data=strip_filler(optional_data),
        checks=checks,
        valid=all(checks.values()),
        raw_lines=(line1, line2),
    )

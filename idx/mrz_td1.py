"""
TD1 MRZ parser — national ID card format, 3 lines x 30 chars.
ICAO Doc 9303 Part 5. Field layout (0-indexed):

Line 1 (30):
  0      document code (I/A/C)
  1      document code variant
  2-4    issuing state (3)
  5-13   document number (9)
  14     check digit — document number
  15-29  optional data 1 (15)

Line 2 (30):
  0-5    date of birth YYMMDD
  6      check digit — DOB
  7      sex (M/F/<)
  8-13   date of expiry YYMMDD
  14     check digit — expiry
  15-17  nationality (3)
  18-28  optional data 2 (11)
  29     composite check digit over line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]

Line 3 (30): SURNAME<<GIVEN<NAMES<<<<...
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

from .mrz_checksum import verify, strip_filler
from .countries import resolve_country_code, resolve_nationality_code

LINE_LEN = 30


@dataclass
class TD1Result:
    document_code: str = ""
    issuing_state: str = ""
    document_number: str = ""
    optional_data_1: str = ""
    date_of_birth: str = ""
    sex: str = ""
    date_of_expiry: str = ""
    nationality: str = ""
    nationality_name: str = ""
    optional_data_2: str = ""
    surname: str = ""
    given_names: str = ""
    checks: dict[str, Any] = field(default_factory=dict[str, Any])
    valid: bool = False
    raw_lines: tuple[Any, ...] = ()


def looks_like_td1(line1: str, line2: str, line3: str) -> bool:
    return (
        len(line1) == LINE_LEN and len(line2) == LINE_LEN and len(line3) == LINE_LEN
        and line1[0] in ("I", "A", "C")
    )


def parse_td1(line1: str, line2: str, line3: str) -> TD1Result:
    line1 = (line1 or "").upper().ljust(LINE_LEN, "<")[:LINE_LEN]
    line2 = (line2 or "").upper().ljust(LINE_LEN, "<")[:LINE_LEN]
    line3 = (line3 or "").upper().ljust(LINE_LEN, "<")[:LINE_LEN]

    doc_number = line1[5:14]
    doc_number_check = line1[14]
    optional_1 = line1[15:30]

    dob = line2[0:6]
    dob_check = line2[6]
    sex = line2[7]
    expiry = line2[8:14]
    expiry_check = line2[14]
    nationality = line2[15:18]
    optional_2 = line2[18:29]
    composite_check = line2[29]

    surname, given_names = "", ""
    name_raw = line3.rstrip("<")
    parts = name_raw.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given_names = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""

    composite_field = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    issuing_state, _ = resolve_country_code(line1[2:5])
    nationality_code, nationality_name = resolve_nationality_code(nationality)

    checks = {
        "document_number": verify(doc_number, doc_number_check),
        "date_of_birth": verify(dob, dob_check),
        "date_of_expiry": verify(expiry, expiry_check),
        "composite": verify(composite_field, composite_check),
    }
    # Unlike TD3 (see mrz_td3.py), composite is NOT relaxed out of `valid`
    # here: TD1/TD2/TD3 share an identical byte layout for document_number/
    # nationality/DOB/expiry, so a genuine TD3 passport reinterpreted at
    # TD1/TD2 length will pass those same three checks by construction —
    # composite (which covers different trailing content per format) is
    # what actually distinguishes "this really is this format" from "this
    # is some other format's data accidentally satisfying a shared prefix".
    # Losing that safety net let real passports get misidentified as
    # national IDs during doc_type="national_id"/"auto" discovery.

    return TD1Result(
        document_code=line1[0:2].rstrip("<"),
        issuing_state=issuing_state,
        document_number=strip_filler(doc_number),
        optional_data_1=strip_filler(optional_1),
        date_of_birth=dob,
        sex=sex if sex in ("M", "F") else "",
        date_of_expiry=expiry,
        nationality=nationality_code,
        nationality_name=nationality_name,
        optional_data_2=strip_filler(optional_2),
        surname=surname,
        given_names=given_names,
        checks=checks,
        valid=all(checks.values()),
        raw_lines=(line1, line2, line3),
    )

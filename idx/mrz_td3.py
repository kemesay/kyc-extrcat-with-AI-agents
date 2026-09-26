"""
TD3 MRZ parser — passport bio-data page, 2 lines x 44 chars.
ICAO Doc 9303 Part 4. Field layout (0-indexed):

Line 1 (44):
  0      document code ('P', or 'P<' variants for diplomatic/service)
  1      document code variant
  2-4    issuing state (3)
  5-43   name field: SURNAME<<GIVEN<NAMES<<<<... (39 chars)

Line 2 (44):
  0-8    document number (9)
  9      check digit — document number
  10-12  nationality (3)
  13-18  date of birth YYMMDD
  19     check digit — DOB
  20     sex (M/F/<)
  21-26  date of expiry YYMMDD
  27     check digit — expiry
  28-41  personal number (14, optional, '<'-padded)
  42     check digit — personal number ('<' if unset)
  43     composite check digit over [0:10] + [13:20] + [21:43]

TD3Result.personal_number is "" when the 14-char field is entirely filler
(the common case -- ICAO marks this field optional and most issuers leave
it unused). Otherwise it's the full raw 16-char slice [28:44] -- field +
its own check digit + the composite check digit, completely unstripped --
since this field's own checksum is too weak on its own (see below) to
justify any processing of the value beyond "unused vs. not".
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field

from .mrz_checksum import verify, strip_filler
from .countries import resolve_country_code, resolve_nationality_code

LINE_LEN = 44


@dataclass
class TD3Result:
    document_code: str = ""
    issuing_state: str = ""
    issuing_state_inferred: bool = False  # see parse_td3: line1 carries no check digit at all
    surname: str = ""
    given_names: str = ""
    document_number: str = ""
    nationality: str = ""
    nationality_name: str = ""
    date_of_birth: str = ""   # YYMMDD as printed; caller resolves century
    sex: str = ""
    date_of_expiry: str = ""
    personal_number: str = ""
    checks: dict[str, Any] = field(default_factory=dict[str, Any])   # field -> bool (checksum passed)
    valid: bool = False
    raw_lines: tuple[Any, ...] = ()
    # Populated by the caller (pipeline._try_parse_td3), which alone has
    # access to every independent OCR pass -- parse_td3 only ever sees one
    # line1/line2 pair. personal_number's own check digit is too weak (1
    # digit over up to 14 chars) to trust alone, but independent OCR passes
    # agreeing on the same value is much stronger corroborating evidence.
    personal_number_agree: int = 0
    personal_number_total: int = 0


def _parse_name_field(raw: str) -> tuple[str, str]:
    raw = raw.rstrip("<")
    parts = raw.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    return surname, given


def looks_like_td3(line1: str, line2: str) -> bool:
    return (
        len(line1) == LINE_LEN and len(line2) == LINE_LEN
        and line1[0] in ("P", "V")
    )


def parse_td3(line1: str, line2: str) -> TD3Result:
    """Parse+validate a TD3 (passport) MRZ. Tolerates a missing/short trailing
    line2 char (some OCR passes drop the final composite digit) by padding.
    """
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
    personal_number_field = line2[28:42]
    personal_number_check = line2[42]
    composite_check = line2[43]
    # This ICAO-optional field is unused (all '<') on most real-world
    # passports -- treat it as genuinely blank whenever the 14-char data
    # portion itself carries no non-filler character, regardless of what
    # ends up in the check-digit positions.
    personal_number_has_data = bool(personal_number_field.strip("<"))

    surname, given_names = _parse_name_field(line1[5:44])
    issuing_state, issuing_state_name = resolve_country_code(line1[2:5])
    nationality_code, nationality_name = resolve_nationality_code(nationality)

    # Line 1 (issuing state + name) carries NO check digit anywhere in the
    # TD3 format — unlike every line 2 field, there is no checksum that can
    # ever catch a misread here (e.g. OCR dropping one character shifts
    # every field after it, silently corrupting issuing_state/surname with
    # no signal in `checks`). Ordinary passports have issuing_state ==
    # nationality in the overwhelming majority of cases, so when
    # issuing_state doesn't resolve to a real country but nationality does,
    # use nationality as a best-effort correction rather than surfacing an
    # unresolvable code. This is a heuristic, not a checksum guarantee —
    # flagged via issuing_state_inferred so callers can warn about it.
    issuing_state_inferred = False
    if not issuing_state_name and nationality_name:
        issuing_state = nationality_code
        issuing_state_inferred = True

    checks = {
        "document_number": verify(doc_number, doc_number_check),
        "date_of_birth": verify(dob, dob_check),
        "date_of_expiry": verify(expiry, expiry_check),
        "personal_number": verify(personal_number_field, personal_number_check),
        "composite": verify(
            line2[0:10] + line2[13:20] + line2[21:43], composite_check
        ),
    }
    # `valid` gates on the identity-bearing fields only. personal_number is
    # optional, often blank/unused, and sits in the sparsest, hardest-to-OCR
    # part of the line (a long run of near-invisible '<' filler) — the least
    # reliable part of a real phone-photo read and the least useful for
    # identity purposes. composite is mathematically downstream of
    # personal_number, so it fails whenever personal_number does even if
    # document_number/DOB/expiry are perfectly read. Gating overall validity
    # on it would reject documents whose actual identity data is solid.
    # Both are still reported in `checks` for anyone who wants to see them.
    #
    # personal_number's check digit is also far weaker evidence than the
    # others: it's a single digit covering up to 14 characters, so wildly
    # different contents can share the same check digit (confirmed on a
    # real document during development: a 1-character reading and the true
    # 10-character value produced the identical check digit). A passing
    # check here does NOT mean the value is complete or correct, and a
    # failing one does NOT mean it's worthless -- a partial OCR read that
    # fails checksum can still be more informative (real leading/trailing
    # digits, '<' marking genuinely-unread positions) than a shorter reading
    # that happens to validate by coincidence. So unlike the other fields,
    # personal_number is never blanked based on `checks` -- shown fully raw
    # (14-char field + its own check digit + the composite check digit, no
    # stripping at all) whenever it carries real data, so callers can see
    # exactly what was on the document rather than a processed guess.
    core_valid = checks["document_number"] and checks["date_of_birth"] and checks["date_of_expiry"]

    return TD3Result(
        document_code=line1[0:2].rstrip("<"),
        issuing_state=issuing_state,
        issuing_state_inferred=issuing_state_inferred,
        surname=surname,
        given_names=given_names,
        document_number=strip_filler(doc_number),
        nationality=nationality_code,
        nationality_name=nationality_name,
        date_of_birth=dob,
        sex=sex if sex in ("M", "F") else "",
        date_of_expiry=expiry,
        personal_number=(
            personal_number_field + personal_number_check + composite_check
            if personal_number_has_data else ""
        ),
        checks=checks,
        valid=core_valid,
        raw_lines=(line1, line2),
    )

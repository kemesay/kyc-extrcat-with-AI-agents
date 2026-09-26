"""
Sanity checks for extracted ID document fields.

Local quantized vision models (qwen2.5vl:7b, llava:13b, moondream) were found
to sometimes fail in two ways that look like a normal "success" response:
  1. Repetition-loop garbage — a field (or several) get stuck echoing the same
     short pattern (e.g. a date repeated across otherwise-distinct fields).
  2. Hallucination — a fully-formed, plausible-looking but entirely wrong
     identity, uncorrelated with the actual document.

These checks don't fix either problem (that's a model-capability limit, not
a code bug) — they give the API caller a `validation` block so a hallucinated
or garbled result isn't silently reported as trustworthy, which matters for
identity-verification use cases.
"""

import re
from datetime import datetime

MRZ_LINE_LEN = 44


def _char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    if c.isalpha():
        return ord(c.upper()) - 55  # A=10 ... Z=35
    return 0


def mrz_check_digit(data: str) -> int:
    """ICAO Doc 9303 check-digit algorithm (weights 7,3,1 repeating)."""
    weights = (7, 3, 1)
    total = sum(_char_value(c) * weights[i % 3] for i, c in enumerate(data))
    return total % 10


def validate_td3_mrz(mrz_code: str) -> dict:
    """Validate a TD3 (passport-style, 2x44) MRZ's embedded check digits."""
    if not mrz_code:
        return {"applicable": False, "well_formed": True}  # empty is not itself suspicious

    lines = [l for l in re.split(r"[\r\n]+", mrz_code.strip()) if l]
    if len(lines) != 2 or any(len(l) != MRZ_LINE_LEN for l in lines):
        return {"applicable": False, "well_formed": False}

    line2 = lines[1]
    passport_no, passport_chk   = line2[0:9],   line2[9]
    birth_date,  birth_chk      = line2[13:19], line2[19]
    expiry_date, expiry_chk     = line2[21:27], line2[27]
    personal_no, personal_chk   = line2[28:42], line2[42]
    composite_chk                = line2[43]

    composite_input = passport_no + passport_chk + birth_date + birth_chk + \
        expiry_date + expiry_chk + personal_no + personal_chk

    checks = {
        "passport_number": mrz_check_digit(passport_no) == _char_value(passport_chk),
        "birth_date":      mrz_check_digit(birth_date) == _char_value(birth_chk),
        "expiry_date":     mrz_check_digit(expiry_date) == _char_value(expiry_chk),
        "personal_number": mrz_check_digit(personal_no) == _char_value(personal_chk),
        "composite":       mrz_check_digit(composite_input) == _char_value(composite_chk),
    }
    return {
        "applicable":       True,
        "well_formed":      True,
        "valid":            all(checks.values()),
        "checks":           checks,
        "passport_number":  passport_no,
        "birth_date":       birth_date,
        "expiry_date":      expiry_date,
    }


_REPEAT_PATTERN = re.compile(r"(.{2,10}?)\1{2,}")


def has_internal_repetition(value: str) -> bool:
    """Catches repetition-loop garbage like 'E-P-T-H-L-E-M-AE-P-T-H-L-E-M-A...'."""
    return bool(value) and bool(_REPEAT_PATTERN.search(value))


def find_cross_field_duplicates(fields: dict, min_len: int = 4) -> list[str]:
    """Catches a repetition loop that fills several distinct fields with the
    same value (e.g. date_of_birth == issue_date == expiry_date == '02.02.19')."""
    seen: dict[str, list[str]] = {}
    for name, value in fields.items():
        value = (value or "").strip()
        if len(value) < min_len:
            continue
        seen.setdefault(value, []).append(name)
    return [
        f"identical value {value!r} in fields {names}"
        for value, names in seen.items()
        if len(names) >= 3
    ]


# Fields that should semantically never hold the same text as a person's name.
# If they do, a model has very likely misread document-header/institutional
# text (e.g. "MINISTRY OF FOREIGN AFFAIRS") as the bearer's name.
_NAME_FIELDS = ("full_name", "full_name_local")
_DISTINCT_FROM_NAME = ("issuing_authority", "document_type", "nationality", "address")

# Words that overwhelmingly indicate institutional/document-header text rather
# than a person's name — real human names essentially never contain these.
_INSTITUTIONAL_KEYWORDS = (
    "ministry", "authority", "department", "government", "republic",
    "immigration", "nationality affairs", "national id", "passport office",
    "directorate", "administration", "bureau", "commission",
)


def find_name_field_issues(fields: dict) -> list[str]:
    reasons = []
    for name_field in _NAME_FIELDS:
        value = (fields.get(name_field) or "").strip()
        if not value:
            continue
        lowered = value.lower()
        hit = next((kw for kw in _INSTITUTIONAL_KEYWORDS if kw in lowered), None)
        if hit:
            reasons.append(
                f"{name_field} {value!r} contains institutional keyword {hit!r} — "
                "likely misread document-header text instead of a person's name"
            )
        for other_field in _DISTINCT_FROM_NAME:
            other_value = (fields.get(other_field) or "").strip()
            if other_value and other_value.lower() == lowered:
                reasons.append(
                    f"{name_field} is identical to {other_field} ({value!r}) — "
                    "a person's name should never match this field"
                )
    return reasons


def sanity_check(extracted_data: dict) -> dict:
    """Returns {suspicious, reasons, mrz}. Does not modify extracted_data."""
    reasons = []

    non_empty = sum(1 for v in extracted_data.values() if isinstance(v, str) and v.strip())
    if non_empty == 0:
        reasons.append("no fields were extracted — all pipeline stages likely failed or returned empty")

    for field, value in extracted_data.items():
        if field == "mrz_code":
            continue  # MRZ padding is legitimately long runs of '<' — checked via validate_td3_mrz instead
        if isinstance(value, str) and has_internal_repetition(value):
            reasons.append(f"{field} looks like repetition-loop garbage: {value!r}")

    reasons.extend(find_cross_field_duplicates(extracted_data))
    reasons.extend(find_name_field_issues(extracted_data))

    mrz_result = validate_td3_mrz(extracted_data.get("mrz_code", ""))
    if not mrz_result.get("well_formed", True):
        reasons.append(
            f"mrz_code is present but not a valid TD3 MRZ (expected 2 lines of 44 chars): "
            f"{extracted_data.get('mrz_code')!r} — likely garbled or hallucinated"
        )
    if mrz_result.get("applicable") and not mrz_result.get("valid"):
        failed = [k for k, ok in mrz_result["checks"].items() if not ok]
        reasons.append(f"MRZ checksum mismatch on: {failed} — extracted fields may not match the document")

    if mrz_result.get("applicable"):
        mrz_passport_no = mrz_result["passport_number"].rstrip("<")
        id_number = (extracted_data.get("id_number") or "").strip().upper()
        if id_number and mrz_passport_no and id_number != mrz_passport_no:
            reasons.append(
                f"id_number {id_number!r} does not match MRZ passport number {mrz_passport_no!r} "
                "— likely a hallucinated or misread field"
            )

    dob, issue, expiry = (
        extracted_data.get("date_of_birth"),
        extracted_data.get("issue_date"),
        extracted_data.get("expiry_date"),
    )
    parsed = {}
    for label, raw in (("dob", dob), ("issue", issue), ("expiry", expiry)):
        parsed[label] = _try_parse_date(raw)
    if parsed["dob"] and parsed["issue"] and parsed["dob"] >= parsed["issue"]:
        reasons.append("date_of_birth is not before issue_date")
    if parsed["issue"] and parsed["expiry"] and parsed["issue"] >= parsed["expiry"]:
        reasons.append("issue_date is not before expiry_date")

    return {
        "suspicious": bool(reasons),
        "reasons":    reasons,
        "mrz":        mrz_result,
    }


_DATE_FORMATS = (
    "%Y-%m-%d", "%d %b %y", "%d %b %Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%y", "%d.%m.%Y",
    "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y", "%m/%d/%Y", "%Y/%m/%d",
)


def _try_parse_date(value):
    if not value or not isinstance(value, str):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None

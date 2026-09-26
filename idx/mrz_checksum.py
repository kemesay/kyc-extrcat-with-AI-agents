"""
ICAO Doc 9303 MRZ check-digit algorithm — shared by TD1/TD2/TD3.

Char value mapping: 0-9 -> itself, A-Z -> 10-35, '<' -> 0.
Weights cycle 7, 3, 1 across the field being checked.

Verified against the canonical ICAO/Wikipedia TD3 worked example
(document holder ERIKSSON ANNA MARIA, issuing state UTO — the fictional
"Utopia" test country used throughout the ICAO 9303 spec):

    P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<<
    L898902C36UTO7408122F1204159ZE184226B<<<<<10

All four check digits (document number, DOB, expiry, composite) were
hand-verified against this vector during development — see tests/test_mrz_checksum.py.
"""

from __future__ import annotations

_WEIGHTS = (7, 3, 1)


def char_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {ch!r}")


def check_digit(field: str) -> int:
    """Compute the ICAO 9303 check digit for a field (weights 7,3,1 repeating)."""
    total = 0
    for i, ch in enumerate(field):
        total += char_value(ch) * _WEIGHTS[i % 3]
    return total % 10


def strip_filler(field: str) -> str:
    """Remove ALL '<' padding characters from a single-token MRZ field
    (document number, personal number, optional data) — '<' is never
    meaningful inside these, only as padding, unlike the name field where
    '<' intentionally separates words and becomes a space instead. Using
    rstrip() here instead of this would leave internal padding in the
    output whenever OCR read fewer trailing '<' than the true field width
    (common on real photos), e.g. '0<<<<<<<<02<<<' should display as '002',
    not '0<<<<<<<<02'.
    """
    return field.replace("<", "")


def verify(field: str, expected_digit: str) -> bool:
    """True if `field`'s computed check digit matches `expected_digit`.

    A '<' expected digit is only accepted when `field` itself is genuinely
    all filler (some issuers print '<' instead of a computed digit for a
    truly-unset optional field, e.g. personal number). It must NOT be
    accepted just because OCR happened to read '<' at the check-digit
    position while the field itself contains real characters — that was a
    real bug: a misaligned/corrupted OCR read of a non-blank field (trailing
    filler run partially dropped, shifting real characters into the wrong
    columns) was passing checksum validation purely because the garbled
    check-digit character came out as '<', silently hiding the corruption.
    """
    if expected_digit == "<":
        return all(ch == "<" for ch in field)
    if not expected_digit.isdigit():
        return False
    return check_digit(field) == int(expected_digit)

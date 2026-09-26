"""
AAMVA DL/ID Card Design Standard parser — the PDF417 barcode printed on
essentially every US/Canadian driver's license and state/province ID card.
Ported from ID-Document-Extractor's idx/aamva.py (barcode-payload parser
portion only — see module docstring below for what was deliberately NOT
ported and why).

Trust model differs from MRZ on purpose: a successful PDF417 decode is
treated as fully trustworthy on its own, with no extra checksum/corroboration
step. PDF417 carries its own Reed-Solomon error correction as part of the
barcode symbology itself — `zxingcpp` only reports a successful decode when
the recovered bytes already satisfy that code's redundancy, so by the time a
payload reaches this parser it has already passed a much stronger integrity
check than any single MRZ check digit provides.

Payload structure (AAMVA DL/ID Card Design Standard):
  "@\\n\\x1e\\rANSI " + 6-digit IIN + 2-digit AAMVA version + ... header,
  then the mandatory "DL"/"ID" subfile, optionally followed by one or more
  jurisdiction-specific subfiles (e.g. California's "ZC") — each a run of
  lines where the first 3 characters are a data element ID immediately
  followed by its value (no delimiter), lines separated by a newline,
  subfiles separated by a segment terminator (carriage return).

WHAT WAS NOT PORTED: the source project's `parse_front_only` (~2400 lines)
reads the front of a card visually when no barcode decodes at all, via three
hand-tuned printed-layout conventions (abbreviated/numbered/full-word
labels). Its comments repeatedly cite specific real card photos it was
iteratively tuned against ("confirmed against a real Washington card",
"confirmed against a real Virginia license", ...) — that is not complexity
that can be responsibly reproduced without those fixtures; a fabricated
version would silently misread real cards with false confidence, which is
exactly the failure mode this whole architecture exists to avoid. Until this
is built against real fixture photos, `parse_front_only` here returns None
and the pipeline falls through to the generic OCR sweep / Ollama VLM tiers
for a driver's license whose barcode doesn't decode — a real degradation in
coverage for that specific case, not a silent gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_HEADER_RE = re.compile(r"ANSI ")

# The subfile type marker ("DL" for a driver's license, "ID" for an
# identification card) is immediately followed by the first element record
# with NO newline in between — locating this marker (rather than the full
# fixed-width offset/length header table, which varies by AAMVA version) is
# what lets a plain newline-split correctly recover the first element.
_SUBFILE_START_RE = re.compile(r"(DL|ID)(?=[A-Z]{3})")

# Every subfile AFTER the mandatory first one has the exact same "glued
# marker" quirk, just with its own 2-letter jurisdiction code (e.g.
# California's "ZC").
_SUBFILE_CONTINUATION_RE = re.compile(r"^[A-Z]{2}(?=[A-Z]{3})")

_SEX_MAP = {"1": "M", "2": "F", "M": "M", "F": "F"}

# AAMVA D20's standard 3-letter eye-color code for brown is "BRO", but the
# commonly PRINTED form (and what jurisdiction-specific subfile duplicates
# like California's ZCA use) is "BRN" — normalize to the latter.
_EYE_COLOR_CANON = {"BRO": "BRN"}


def _load_aamva_ontology() -> tuple[dict[tuple[str, str], str], dict[str, Any], dict[str, int]]:
    """Per-jurisdiction AAMVA data lives in ontology/aamva_jurisdictions.yaml
    as DATA, never as `if state == "..."` branches in Python."""
    import yaml

    path = Path(__file__).parent / "ontology" / "aamva_jurisdictions.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    class_code_canonical: dict[tuple[str, str], str] = {}
    for key, value in raw.get("class_code_canonical", {}).items():
        jurisdiction, letter = key.split(":", 1)
        class_code_canonical[(jurisdiction, letter)] = value

    return class_code_canonical, raw.get("field_profile", {}), raw.get("standard_term_years", {})


_CLASS_CODE_CANONICAL, JURISDICTION_FIELD_PROFILE, _STANDARD_TERM_YEARS = _load_aamva_ontology()

# Reused by idx/signatures.py as the "height" value signature. The
# foot/inch separator is either an apostrophe (optionally followed by a
# dash, "5'-09\"") or a bare dash alone ("5-09\""); "cm" may have a
# stray-period or space before it ("175.Cm", "175 cm").
_HEIGHT_SEP = r"(?:'-?|-)"
_HEIGHT_CM_SEP = r"[\s.]*"
_HEIGHT_SHAPE = r"\d{1,3}(?:" + _HEIGHT_SEP + r"\d{1,2}\"?|" + _HEIGHT_CM_SEP + r"cm)"


@dataclass
class AAMVAResult:
    surname: str = ""
    given_names: str = ""
    full_name: str = ""
    sex: str = ""
    date_of_birth: str = ""     # ISO YYYY-MM-DD
    date_of_issue: str = ""
    date_of_expiry: str = ""
    document_number: str = ""
    address: str = ""
    # The holder's mailing-address state/province — AAMVA has no separate
    # "issuing state" element distinct from the holder's address, and for a
    # driver's license the two coincide in the overwhelming majority of cases.
    address_state: str = ""
    country: str = ""
    license_class: str = ""
    license_class_printed_blank: bool = False
    license_restrictions: str = ""
    license_restrictions_printed_blank: bool = False
    license_endorsements: str = ""
    license_endorsements_printed_blank: bool = False
    subfile_type: str = "DL"    # "DL" or "ID" — see module docstring
    # Fields with no home in the project's common schema — folded into
    # ExtractedIDDocument.extra rather than dedicated top-level fields.
    height: str = ""
    weight: str = ""
    eye_color: str = ""
    eye_color_raw: str = ""
    hair_color: str = ""
    document_discriminator: str = ""
    inventory_control_number: str = ""
    compliance_type: str = ""   # "F" = REAL ID compliant, "N" = not (DDA)
    card_revision_date: str = ""
    address_county: str = ""
    fields: dict[str, Any] = field(default_factory=dict)   # every raw element ID -> value


def _mmddccyy_to_iso(raw: str) -> str:
    if not raw or len(raw) != 8 or not raw.isdigit():
        return ""
    mm, dd, ccyy = int(raw[0:2]), int(raw[2:4]), int(raw[4:8])
    try:
        return date(ccyy, mm, dd).isoformat()
    except ValueError:
        return ""


def looks_like_aamva(payload: str) -> bool:
    return payload.startswith("@") and bool(_HEADER_RE.search(payload[:30]))


def parse_aamva(payload: str) -> AAMVAResult | None:
    """Parse a decoded AAMVA PDF417 payload string. Returns None if the
    payload doesn't have the AAMVA header at all — "not this format, let the
    caller move on" contract, same as mrz_td3.parse_td3's looks_like_td3 guard."""
    if not looks_like_aamva(payload):
        return None

    normalized = payload.replace("<LF>", "\n").replace("<CR>", "\r")

    start = _SUBFILE_START_RE.search(normalized)
    body = normalized[start.end():] if start else normalized
    subfile_type = start.group(1) if start else "DL"

    fields: dict[str, str] = {}
    for i, segment in enumerate(body.split("\r")):
        if i > 0:
            segment = _SUBFILE_CONTINUATION_RE.sub("", segment)
        for line in segment.split("\n"):
            line = line.strip()
            if len(line) < 4 or not line[:3].isalpha() or not line[:3].isupper():
                continue
            element_id, value = line[:3], line[3:].strip()
            if value:
                fields[element_id] = value

    result = AAMVAResult(fields=fields, subfile_type=subfile_type)
    result.surname = fields.get("DCS", "")
    middle_name = fields.get("DAD", "")
    if middle_name.strip().upper() == "NONE":
        middle_name = ""
    result.given_names = " ".join(p for p in (fields.get("DAC", ""), middle_name) if p)
    result.full_name = f"{result.given_names} {result.surname}".strip()
    result.date_of_birth = _mmddccyy_to_iso(fields.get("DBB", ""))
    result.date_of_expiry = _mmddccyy_to_iso(fields.get("DBA", ""))
    result.date_of_issue = _mmddccyy_to_iso(fields.get("DBD", ""))
    result.document_number = fields.get("DAQ", "")
    result.sex = _SEX_MAP.get(fields.get("DBC", ""), "")
    result.country = fields.get("DCG", "")
    result.license_class = fields.get("DCA", "")
    result.license_restrictions = fields.get("DCB", "")
    result.license_endorsements = fields.get("DCD", "")
    result.address_state = fields.get("DAJ", "")

    address_parts = [fields.get(k, "") for k in ("DAG", "DAI", "DAJ", "DAK")]
    result.address = ", ".join(p for p in address_parts if p)

    result.height = fields.get("DAU", "")
    result.weight = fields.get("DAW", "")
    result.eye_color_raw = fields.get("DAY", "")
    result.eye_color = _EYE_COLOR_CANON.get(result.eye_color_raw, result.eye_color_raw)
    result.hair_color = fields.get("DAZ", "")
    result.document_discriminator = fields.get("DCF", "")
    result.inventory_control_number = fields.get("DCK", "")
    result.compliance_type = fields.get("DDA", "")
    result.card_revision_date = _mmddccyy_to_iso(fields.get("DDB", ""))

    return result


def parse_front_only(front_path: str) -> AAMVAResult | None:
    """No barcode decoded — see module docstring for why this returns None
    rather than a fabricated OCR-based reading. Caller (pipeline.py) falls
    through to the generic OCR sweep / Ollama VLM tier instead."""
    return None

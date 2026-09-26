"""
Machine-verifiable sources, reusing idx's deterministic readers read-only.

MRZ: the agents transcribe MRZ lines; a 7B VLM's copy of 88 filler-heavy
characters is often one or two characters off. So the agents' lines are
re-validated with idx's ICAO 9303 TD1/TD2/TD3 parsers, and when no agent
produced a checksum-valid copy, idx's local OCR MRZ search (RapidOCR +
check-digit error correction) re-reads the zone from pixels. Only a
checksum-valid MRZ is ever treated as verified.

Barcodes: decoded locally with zxing-cpp via idx.barcode. AAMVA PDF417
payloads (US/CA licences and ID cards) and Ethiopian Fayda QR payloads are
parsed into fields; any other barcode's text is surfaced as barcode_data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from idx.aamva import parse_aamva
from idx.barcode import decode_barcodes
from idx.ethiopia_id import _parse_qr_payload
from idx.mrz_ocr import find_mrz_candidates
from idx.mrz_td1 import parse_td1
from idx.mrz_td2 import parse_td2
from idx.mrz_td3 import parse_td3

from .normalize import resolve_country, resolve_nationality

logger = logging.getLogger(__name__)

_MRZ_LENGTHS = {"TD1": (3, 30), "TD2": (2, 36), "TD3": (2, 44)}
_MRZ_DOC_TYPES = {"P": "passport", "V": "visa", "I": "national_id", "A": "national_id", "C": "national_id"}


@dataclass
class VerifiedSource:
    """Fields read from a checksum/error-correction-protected source."""
    source: str                     # "mrz" | "barcode"
    tier: int                       # precedence: barcode 1, mrz 2
    confidence: float
    fields: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    # Per-field trust; falls back to `confidence`. An MRZ's check digits
    # protect only some fields — see _mrz_field_confidence.
    field_confidence: dict[str, float] = field(default_factory=dict)

    def confidence_for(self, name: str) -> float:
        return self.field_confidence.get(name, self.confidence)


# ─────────────────────────────────────────────────────────────
# MRZ
# ─────────────────────────────────────────────────────────────

def clean_mrz_line(line: str) -> str:
    text = (line or "").upper().replace("«", "<").replace("‹", "<").replace("＜", "<")
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^A-Z0-9<]", "", text)


def _fit(line: str, width: int) -> str:
    return line.ljust(width, "<") if len(line) < width else line[:width]


def _yymmdd(value: str, *, future_leaning: bool) -> str:
    if not re.fullmatch(r"\d{6}", value or ""):
        return ""
    yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:])
    century = 2000 if future_leaning or yy <= date.today().year % 100 else 1900
    try:
        return date(century + yy, mm, dd).isoformat()
    except ValueError:
        return ""


def _mrz_format(lines: list[str]) -> Optional[str]:
    if len(lines) == 3:
        return "TD1"
    if len(lines) == 2:
        return "TD3" if len(lines[1]) > 40 else "TD2"
    return None


def _data_lines_complete(fmt: str, lines: list[str]) -> bool:
    """Agents often drop the name line's trailing '<' fillers (safe to pad
    back), but a data line missing characters can't be repaired by padding."""
    width = _MRZ_LENGTHS[fmt][1]
    data = lines[:2] if fmt == "TD1" else lines[1:]
    return all(len(l) >= width - 2 for l in data) and len(lines[-1 if fmt != "TD1" else 2]) >= 5


def _parse(fmt: str, lines: list[str]):
    width = _MRZ_LENGTHS[fmt][1]
    fitted = [_fit(l, width) for l in lines]
    if fmt == "TD1":
        return parse_td1(*fitted)
    if fmt == "TD2":
        return parse_td2(*fitted)
    return parse_td3(*fitted)


# Letters-only MRZ fields (issuing state, nationality) aren't protected by any
# check digit, so an OCR '0' for 'O' survives validation; the parser resolves
# the code, and the raw line is rebuilt with the canonical letters.
_ALPHA_SPANS = {"TD3": [(0, 2, 5), (1, 10, 13)], "TD2": [(0, 2, 5), (1, 10, 13)],
                "TD1": [(0, 2, 5), (1, 15, 18)]}
_DIGIT_TO_LETTER = str.maketrans("01258", "OIZSB")


def _canonical_lines(fmt: str, lines: list[str]) -> list[str]:
    out = list(lines)
    for idx, start, end in _ALPHA_SPANS.get(fmt, []):
        if idx < len(out) and len(out[idx]) >= end:
            line = out[idx]
            out[idx] = line[:start] + line[start:end].translate(_DIGIT_TO_LETTER) + line[end:]
    return out


def _mrz_fields(result: Any, fmt: str) -> dict[str, str]:
    surname = (result.surname or "").strip()
    given = (result.given_names or "").strip()
    sex = {"M": "Male", "F": "Female", "X": "X", "<": "X"}.get((result.sex or "").strip(), "")
    issuing_code, _ = resolve_country(result.issuing_state)
    nat_code, nat_name = resolve_nationality(result.nationality)
    code = (result.document_code or "").strip()
    fields = {
        "last_name": surname,
        "first_name": given,
        "full_name": f"{given} {surname}".strip(),
        "full_name_latin": f"{given} {surname}".strip(),
        "document_number": (result.document_number or "").strip("<"),
        "date_of_birth": _yymmdd(result.date_of_birth, future_leaning=False),
        "expiry_date": _yymmdd(result.date_of_expiry, future_leaning=True),
        "gender": sex,
        "issuing_country": issuing_code or (result.issuing_state or "").strip("<"),
        "nationality": nat_name or (result.nationality or "").strip("<"),
        "nationality_code": nat_code,
        "document_type": _MRZ_DOC_TYPES.get(code[:1], ""),
    }
    lines = _canonical_lines(fmt, list(result.raw_lines or []))
    if fmt == "TD3" and len(lines) == 2:
        # line 2 chars 29-42; idx's TD3Result.personal_number also carries the check digits
        personal = lines[1][28:42].strip("<").replace("<", " ")
        if personal and personal.strip("0 "):         # '0<<<…' is filler, not a personal number
            fields["personal_number"] = personal
    for i, line in enumerate(lines[:3], start=1):
        fields[f"mrz_line_{i}"] = line
    return {k: v for k, v in fields.items() if v}


# Which check digit protects which output field. Names, sex, nationality and
# issuing state are covered by no check digit at all in ICAO 9303.
_CHECK_FOR_FIELD = {
    "document_number": "document_number",
    "date_of_birth": "date_of_birth",
    "expiry_date": "date_of_expiry",
    "personal_number": "personal_number",
}


def _mrz_field_confidence(fields: dict[str, str], checks: dict[str, bool]) -> dict[str, float]:
    all_passed = bool(checks) and all(checks.values())
    out = {}
    for name in fields:
        check = _CHECK_FOR_FIELD.get(name)
        if check and check in checks:
            out[name] = 0.99 if checks[check] else 0.4
        else:
            out[name] = 0.9 if all_passed else 0.75
    return out


def _mrz_source(result: Any, fmt: str, read_by: str) -> VerifiedSource:
    fields = _mrz_fields(result, fmt)
    checks = dict(result.checks)
    return VerifiedSource(
        source="mrz", tier=2, confidence=0.98 if result.valid else 0.5,
        fields=fields,
        details={"format": fmt, "valid": bool(result.valid), "all_checks_passed": all(checks.values()),
                 "checks": checks, "read_by": read_by},
        field_confidence=_mrz_field_confidence(fields, checks),
    )


def verify_mrz_lines(lines: list[str]) -> Optional[VerifiedSource]:
    """Validates agent-transcribed MRZ lines. Returns a VerifiedSource with
    details['valid'] True/False, or None when the lines aren't MRZ-shaped."""
    cleaned = [clean_mrz_line(l) for l in lines if clean_mrz_line(l)]
    fmt = _mrz_format(cleaned)
    if fmt is None or not _data_lines_complete(fmt, cleaned):
        return None
    try:
        result = _parse(fmt, cleaned)
    except Exception as e:
        logger.debug("MRZ parse failed: %s", e)
        return None
    return _mrz_source(result, fmt, "agents")


_MRZ_WIDTHS = {30: "TD1", 36: "TD2", 44: "TD3"}


def _mrz_shaped(line: str) -> str:
    """A cleaned OCR line that looks like an MRZ line: 26-46 chars of
    [A-Z0-9<] with at least one filler."""
    cleaned = clean_mrz_line(line)
    if 26 <= len(cleaned) <= 46 and "<" in cleaned and len(cleaned) >= 0.9 * len(re.sub(r"\s", "", line)):
        return cleaned
    return ""


def mrz_from_ocr_text(text: str) -> tuple[Optional[VerifiedSource], bool]:
    """Checksum-validates MRZ lines found directly in OCR text (no extra OCR
    pass): every run of 2-3 consecutive MRZ-shaped lines of one width class
    is tried as TD1/TD2/TD3. -> (best source or None, MRZ-like text seen)."""
    lines = [_mrz_shaped(l) for l in (text or "").splitlines() if l.strip()]
    seen = any(lines)
    best: Optional[VerifiedSource] = None
    groups = [lines[i:i + size] for i in range(len(lines)) for size in (3, 2)]
    groups = [g for g in groups if len(g) in (2, 3) and all(g)] + _role_combinations([l for l in lines if l])
    for group in groups:
        size = len(group)
        if True:
            src = verify_mrz_lines(group)
            if src is None:
                continue
            fmt = src.details["format"]
            if (fmt == "TD1") != (size == 3):
                continue
            src.details["read_by"] = "ocr_text"
            if src.details["valid"] and src.details["all_checks_passed"]:
                return _settle_name_line(src, group, lines), True
            if best is None or (src.details["valid"] and not best.details["valid"]):
                best = src
    return best, seen


_NAME_FIELDS = ("last_name", "first_name", "full_name", "full_name_latin")


def _settle_name_line(src: VerifiedSource, group: list[str], lines: list[str]) -> VerifiedSource:
    """The MRZ name line carries no check digit, so a clean checksum says
    nothing about it. When the engines disagree on it, take the majority
    reading; with no majority, mark the MRZ name uncertain so it corroborates
    the visual name but never overrides its spelling."""
    fmt = src.details["format"]
    idx = 2 if fmt == "TD1" else 0
    width = len(group[idx])
    if fmt == "TD1":
        candidates = [l for l in lines if l and abs(len(l) - width) <= 2 and "<<" in l and re.match(r"^[A-Z]+<", l)
                      and not l[:6].isdigit() and l[:2] != group[0][:2]]
    else:
        candidates = [l for l in lines if l and abs(len(l) - width) <= 2 and l[:5] == group[0][:5]]
    names = [l.rstrip("<") for l in candidates]
    if len(set(names)) <= 1:
        return src
    counts = {n: names.count(n) for n in set(names)}
    top = max(counts.values())
    winners = [n for n, c in counts.items() if c == top]
    if len(winners) == 1 and winners[0] != group[idx].rstrip("<"):
        fixed = list(group)
        fixed[idx] = next(l for l in candidates if l.rstrip("<") == winners[0])
        better = verify_mrz_lines(fixed)
        if better is not None and better.details["valid"]:
            better.details.update(read_by="ocr_text", name_line_votes=counts)
            return better
    if len(winners) > 1:
        src.details["name_line_uncertain"] = True
        src.details["name_line_readings"] = sorted(counts)
        for name in _NAME_FIELDS:
            if name in src.fields:
                src.field_confidence[name] = 0.8     # below the 0.9 needed to override a spelling
    return src


def _role_combinations(lines: list[str], limit: int = 4) -> list[list[str]]:
    """Two OCR engines each misread a different MRZ line; combine lines by
    their role (document line / data line / name line) across engines and let
    the check digits choose. Width classes: TD1 30, TD2 36, TD3 44 (±2)."""
    def width(l: str, w: int) -> bool:
        return abs(len(l) - w) <= 2
    out: list[list[str]] = []
    td1_doc = [l for l in lines if width(l, 30) and l[:1] in "ACI" and l[1:2] in "<ABCDIPR"][:limit]
    td1_data = [l for l in lines if width(l, 30) and l[:6].isdigit()][:limit]
    td1_name = [l for l in lines if width(l, 30) and "<<" in l and re.match(r"^[A-Z]+<", l) and l not in td1_doc][:limit]
    out += [[a, b, c] for a in td1_doc for b in td1_data for c in td1_name]
    for w in (36, 44):
        head = [l for l in lines if width(l, w) and l[:1] in "PIACV" and l[1:2] in "<ABCDIPRMNOQ"][:limit]
        data = [l for l in lines if width(l, w) and re.search(r"\d{6}[0-9<][MFX<]\d{6}", l)][:limit]
        out += [[a, b] for a in head for b in data if a != b]
    return out


def search_mrz_locally(image_paths: list[str], formats: tuple[str, ...] = ("TD3", "TD1", "TD2")) -> Optional[VerifiedSource]:
    """idx's pixel-level MRZ search (slow — tens of seconds per image).
    Returns the first checksum-valid parse, else None."""
    for path in image_paths:
        for fmt in formats:
            try:
                candidates = find_mrz_candidates(path, formats=(fmt,))
            except Exception as e:
                logger.warning("local MRZ search failed on %s: %s", path, e)
                continue
            for candidate in candidates:
                if len(candidate.lines) != _MRZ_LENGTHS[fmt][0]:
                    continue
                try:
                    result = _parse(fmt, list(candidate.lines))
                except Exception:
                    continue
                if result.valid:
                    return _mrz_source(result, fmt, "local_ocr")
    return None


# ─────────────────────────────────────────────────────────────
# Barcodes
# ─────────────────────────────────────────────────────────────

def read_barcodes(image_paths: list[str]) -> tuple[Optional[VerifiedSource], list[dict[str, str]]]:
    """-> (parsed VerifiedSource if a known payload decoded, all raw hits)."""
    raw_hits: list[dict[str, str]] = []
    verified: Optional[VerifiedSource] = None
    for path in image_paths:
        try:
            hits = decode_barcodes(path)
        except Exception as e:
            logger.warning("barcode decode failed on %s: %s", path, e)
            continue
        for hit in hits:
            raw_hits.append({"format": hit.format, "text": hit.text[:1000]})
            if verified is not None:
                continue
            verified = _parse_known_payload(hit.format, hit.text)
    return verified, raw_hits


def _parse_known_payload(fmt: str, text: str) -> Optional[VerifiedSource]:
    if "PDF417" in fmt.upper():
        aamva = parse_aamva(text)
        if aamva is not None:
            country_code, _ = resolve_country(aamva.country)
            fields = {
                "full_name": aamva.full_name, "full_name_latin": aamva.full_name,
                "last_name": aamva.surname, "first_name": aamva.given_names,
                "date_of_birth": aamva.date_of_birth, "issue_date": aamva.date_of_issue,
                "expiry_date": aamva.date_of_expiry, "document_number": aamva.document_number,
                "gender": {"M": "Male", "F": "Female", "X": "X"}.get(aamva.sex, aamva.sex),
                "address": aamva.address, "region": aamva.address_state,
                "issuing_country": country_code or aamva.country,
                "document_type": "drivers_license" if aamva.subfile_type == "DL" else "national_id",
            }
            return VerifiedSource(
                source="barcode", tier=1, confidence=0.97,
                fields={k: v for k, v in fields.items() if v},
                details={"payload": "AAMVA", "license_class": aamva.license_class,
                         "restrictions": aamva.license_restrictions,
                         "endorsements": aamva.license_endorsements},
            )
    if "QR" in fmt.upper():
        eth = _parse_qr_payload(text)
        if eth.get("fan"):
            fields = {"document_number": eth["fan"], "issuing_country": "ETH",
                      "document_type": "national_id"}
            if eth.get("full_name"):
                fields["full_name"] = eth["full_name"]
                fields["full_name_latin"] = eth["full_name"]
            if eth.get("dob_ethiopian_display"):
                fields["date_of_birth_orig"] = eth["dob_ethiopian_display"]
            return VerifiedSource(source="barcode", tier=1, confidence=0.95, fields=fields,
                                  details={"payload": "ETHIOPIA_FAYDA_QR"})
    return None

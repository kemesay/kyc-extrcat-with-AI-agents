"""
Orchestrates the full extract-from-photo pipeline. Waterfall structure and
precedence discipline ported from ID-Document-Extractor's idx/pipeline.py
(1914 lines) — this is a scoped adaptation, not a line-for-line port: the
source's many small per-family AI-crosscheck enrichment functions
(_apply_passport_crosscheck, _apply_aamva_crosscheck, ...) are consolidated
into one general additive-merge step (_merge_additive) here, since our VLM
tier (idx/vision_ollama.py) already returns a full document-shaped read
rather than the source's narrower per-family crosscheck schemas.

Stages tried, in order:
  1. Local MRZ-candidate search + gate + checksum-validated TD1/TD2/TD3 parse (free)
  2. AAMVA PDF417 barcode decode (free)
  3. Ethiopia QR/barcode decode (free)
  4. Ollama VLM universal fallback (local, no cost, but slow — escalation only)
  5. Generic OCR label-alias sweep/regex (free, the floor — never returns empty)

CLAUDE.md's Precedence table: barcode > MRZ > label-anchored OCR > VLM >
regex sweep. A higher tier's field is never overwritten by a lower one —
see _merge_additive. apply_plausibility_checks() runs once, universally,
at the end, never threaded through each builder.

The MRZ gate (_mrz_gate_passes) replaces the source's earlier Gemini
classify_document call with a free, local, decisive signal: if a PDF417
barcode on the image decodes as AAMVA, any MRZ-shaped text the free search
found is a driver's license's guilloche/microprint noise, not a genuine
MRZ — a US/CA driver's license never has an MRZ (CLAUDE.md invariant #1).
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from datetime import date, datetime
from typing import Optional

from .aamva import JURISDICTION_FIELD_PROFILE, _CLASS_CODE_CANONICAL, looks_like_aamva, parse_aamva, parse_front_only, AAMVAResult
from .barcode import decode_barcodes
from .countries import resolve_country_code, resolve_nationality_code
from .ethiopia_id import classify_ethiopia_id_sides, parse_ethiopia_digital_id, EthiopiaIDResult
from .generic_ocr import extract_fields_via_sweep, extract_generic_fields, guess_document_category, ocr_full_text
from .mrz_ocr import find_mrz_candidates
from .mrz_td1 import parse_td1, TD1Result
from .mrz_td2 import parse_td2, TD2Result
from .mrz_td3 import parse_td3, TD3Result
from .schema import ExtractedIDDocument, FieldSource, FieldValue, field_value, unreadable_field
from .validation import apply_plausibility_checks
from . import vision_ollama as vision

logger = logging.getLogger(__name__)

_IDENTITY_FIELDS = (
    "full_name", "surname", "given_names", "document_number",
    "date_of_birth", "nationality_code",
)


def _yymmdd_to_iso(yymmdd: str, *, is_future_leaning: bool) -> str:
    """MRZ dates are 2-digit years with no century marker. DOB is assumed
    in the past (19xx if that reads more plausibly); expiry/issue dates on
    documents using this MRZ standard (post-1980) are assumed 20xx."""
    if not yymmdd or len(yymmdd) != 6 or not yymmdd.isdigit():
        return ""
    yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return ""
    if is_future_leaning:
        century = 2000
    else:
        current_yy = datetime.now().year % 100
        century = 2000 if yy <= current_yy else 1900
    try:
        return datetime(century + yy, mm, dd).date().isoformat()
    except ValueError:
        return ""


def _mrz_gate_passes(image_path: str, image_path_back: Optional[str] = None) -> bool:
    """Fails OPEN (keeps the MRZ candidate) when no AAMVA barcode decodes —
    an additional safety net on top of checksum validation, never a
    replacement for it."""
    for path in (image_path, image_path_back):
        if not path:
            continue
        try:
            hits = decode_barcodes(path, formats=("PDF417",))
        except Exception:
            continue
        if any(looks_like_aamva(hit.text) for hit in hits):
            return False
    return True


def _try_parse_td3(image_path: str) -> Optional[TD3Result]:
    candidates = [c for c in find_mrz_candidates(image_path, formats=("TD3",)) if len(c.lines) == 2]
    if not candidates:
        return None
    line1_votes = Counter(c.lines[0] for c in candidates)
    best_line1 = line1_votes.most_common(1)[0][0]

    seen: list[str] = []
    best_effort: Optional[TD3Result] = None
    for c in candidates:
        line2 = c.lines[1]
        if line2 in seen:
            continue
        seen.append(line2)
        result = parse_td3(best_line1, line2)
        if best_effort is None:
            best_effort = result
        if result.valid:
            return result
    return best_effort


def _try_parse_td1(image_path: str) -> Optional[TD1Result]:
    for c in find_mrz_candidates(image_path, formats=("TD1",)):
        if len(c.lines) != 3:
            continue
        result = parse_td1(*c.lines)
        if result.valid:
            return result
    return None


def _try_parse_td2(image_path: str) -> Optional[TD2Result]:
    for c in find_mrz_candidates(image_path, formats=("TD2",)):
        if len(c.lines) != 2:
            continue
        result = parse_td2(*c.lines)
        if result.valid:
            return result
    return None


def _is_essentially_empty(doc: Optional[ExtractedIDDocument]) -> bool:
    if doc is None:
        return True
    filled = sum(1 for name in _IDENTITY_FIELDS if getattr(doc, name).state == "read")
    return filled < 2


def _merge_additive(doc: ExtractedIDDocument, extra: dict, *, source: FieldSource, confidence: float) -> None:
    """CLAUDE.md Precedence: a higher-tier field is never overwritten by a
    lower one. Fills only fields that are still empty on `doc`; a field
    both sides read gets flagged for review instead of silently resolved."""
    mapping = {
        "full_name": "full_name", "surname": "surname", "given_names": "given_names",
        "sex": "sex", "date_of_birth": "date_of_birth", "date_of_issue": "date_of_issue",
        "date_of_expiry": "date_of_expiry", "document_number": "document_number",
        "nationality": "nationality_code", "address": "address",
        "issuing_authority": "issuing_authority", "place_of_birth": "place_of_birth",
        "license_class": "license_class", "license_restrictions": "license_restrictions",
        "license_endorsements": "license_endorsements", "permit_type": "permit_type",
    }
    for src_key, doc_attr in mapping.items():
        value = (extra.get(src_key) or "").strip()
        if not value:
            continue
        current: FieldValue = getattr(doc, doc_attr)
        if current.state != "read":
            setattr(doc, doc_attr, field_value(value, source=source, confidence=confidence))
        elif current.value.strip().lower() != value.lower():
            doc.human_review_required = True
            doc.warnings.append(
                f"{doc_attr}: {source} disagrees with existing {current.source} value "
                f"({current.value!r} vs {value!r}) — kept the higher-tier value"
            )
    local_name = (extra.get("full_name_local") or "").strip()
    if local_name and not doc.name_in_native_script.value:
        doc.name_in_native_script = field_value(local_name, source=source, confidence=confidence)


# ─────────────────────────────────────────────────────────────
# Document builders — one raw-result-family -> ExtractedIDDocument each
# ─────────────────────────────────────────────────────────────

def _from_td3(result: TD3Result, image_path: str) -> ExtractedIDDocument:
    doc = ExtractedIDDocument(document_category="passport", mrz_format="TD3")
    source: FieldSource = "mrz"
    confidence = 0.98 if result.valid else 0.55
    doc.extraction_method = "mrz" if result.valid else "mrz_low_confidence"
    doc.confidence = "high" if result.valid else "low"

    doc.document_number = field_value(result.document_number, source=source, confidence=confidence)
    doc.surname = field_value(result.surname, source=source, confidence=confidence)
    doc.given_names = field_value(result.given_names, source=source, confidence=confidence)
    doc.full_name = field_value(f"{result.given_names} {result.surname}".strip(), source=source, confidence=confidence)
    doc.name_order_hint = "given_first"

    doc.date_of_birth = field_value(
        _yymmdd_to_iso(result.date_of_birth, is_future_leaning=False), source=source, confidence=confidence,
        raw=result.date_of_birth,
    )
    doc.date_of_expiry = field_value(
        _yymmdd_to_iso(result.date_of_expiry, is_future_leaning=True), source=source, confidence=confidence,
        raw=result.date_of_expiry,
    )
    doc.sex = field_value(result.sex, source=source, confidence=confidence)

    country_code, country_name = resolve_country_code(result.issuing_state)
    doc.country_code = field_value(country_code, source=source, confidence=confidence)
    doc.country_name = field_value(country_name, source=source, confidence=confidence)
    nat_code, nat_name = resolve_nationality_code(result.nationality)
    doc.nationality_code = field_value(nat_code, source=source, confidence=confidence)
    doc.nationality_name = field_value(nat_name, source=source, confidence=confidence)

    if result.personal_number:
        doc.personal_number = field_value(result.personal_number, source=source, confidence=confidence)
        if result.personal_number_total:
            doc.personal_number_ocr_agreement = f"{result.personal_number_agree}/{result.personal_number_total}"

    doc.mrz_raw = list(result.raw_lines) if result.raw_lines else []
    doc.mrz_checks = dict(result.checks)
    if not result.valid:
        doc.warnings.append("MRZ found but not all checksums passed — treat as low confidence")
    return doc


def _from_td1(result: TD1Result, category: str) -> ExtractedIDDocument:
    doc = ExtractedIDDocument(document_category=category, mrz_format="TD1")
    source: FieldSource = "mrz"
    confidence = 0.98 if result.valid else 0.55
    doc.extraction_method = "mrz" if result.valid else "mrz_low_confidence"
    doc.confidence = "high" if result.valid else "low"

    doc.document_number = field_value(result.document_number, source=source, confidence=confidence)
    doc.surname = field_value(result.surname, source=source, confidence=confidence)
    doc.given_names = field_value(result.given_names, source=source, confidence=confidence)
    doc.full_name = field_value(f"{result.given_names} {result.surname}".strip(), source=source, confidence=confidence)
    doc.date_of_birth = field_value(
        _yymmdd_to_iso(result.date_of_birth, is_future_leaning=False), source=source, confidence=confidence,
        raw=result.date_of_birth,
    )
    doc.date_of_expiry = field_value(
        _yymmdd_to_iso(result.date_of_expiry, is_future_leaning=True), source=source, confidence=confidence,
        raw=result.date_of_expiry,
    )
    doc.sex = field_value(result.sex, source=source, confidence=confidence)
    country_code, country_name = resolve_country_code(result.issuing_state)
    doc.country_code = field_value(country_code, source=source, confidence=confidence)
    doc.country_name = field_value(country_name, source=source, confidence=confidence)
    nat_code, nat_name = resolve_nationality_code(result.nationality)
    doc.nationality_code = field_value(nat_code, source=source, confidence=confidence)
    doc.nationality_name = field_value(nat_name, source=source, confidence=confidence)
    doc.mrz_raw = list(result.raw_lines) if result.raw_lines else []
    doc.mrz_checks = dict(result.checks)
    if not result.valid:
        doc.warnings.append("MRZ found but not all checksums passed — treat as low confidence")
    return doc


def _from_td2(result: TD2Result, category: str) -> ExtractedIDDocument:
    doc = ExtractedIDDocument(document_category=category, mrz_format="TD2")
    source: FieldSource = "mrz"
    confidence = 0.98 if result.valid else 0.55
    doc.extraction_method = "mrz" if result.valid else "mrz_low_confidence"
    doc.confidence = "high" if result.valid else "low"

    doc.document_number = field_value(result.document_number, source=source, confidence=confidence)
    doc.surname = field_value(result.surname, source=source, confidence=confidence)
    doc.given_names = field_value(result.given_names, source=source, confidence=confidence)
    doc.full_name = field_value(f"{result.given_names} {result.surname}".strip(), source=source, confidence=confidence)
    doc.date_of_birth = field_value(
        _yymmdd_to_iso(result.date_of_birth, is_future_leaning=False), source=source, confidence=confidence,
        raw=result.date_of_birth,
    )
    doc.date_of_expiry = field_value(
        _yymmdd_to_iso(result.date_of_expiry, is_future_leaning=True), source=source, confidence=confidence,
        raw=result.date_of_expiry,
    )
    doc.sex = field_value(result.sex, source=source, confidence=confidence)
    country_code, country_name = resolve_country_code(result.issuing_state)
    doc.country_code = field_value(country_code, source=source, confidence=confidence)
    doc.country_name = field_value(country_name, source=source, confidence=confidence)
    nat_code, nat_name = resolve_nationality_code(result.nationality)
    doc.nationality_code = field_value(nat_code, source=source, confidence=confidence)
    doc.nationality_name = field_value(nat_name, source=source, confidence=confidence)
    doc.mrz_raw = list(result.raw_lines) if result.raw_lines else []
    doc.mrz_checks = dict(result.checks)
    if not result.valid:
        doc.warnings.append("MRZ found but not all checksums passed — treat as low confidence")
    return doc


def _from_aamva(result: AAMVAResult) -> ExtractedIDDocument:
    category = "drivers_license" if result.subfile_type == "DL" else "national_id"
    doc = ExtractedIDDocument(document_category=category)
    source: FieldSource = "barcode"
    confidence = 0.97
    doc.extraction_method = "barcode"
    doc.confidence = "high"

    doc.document_number = field_value(result.document_number, source=source, confidence=confidence)
    doc.surname = field_value(result.surname, source=source, confidence=confidence)
    doc.given_names = field_value(result.given_names, source=source, confidence=confidence)
    doc.full_name = field_value(result.full_name, source=source, confidence=confidence)
    doc.date_of_birth = field_value(result.date_of_birth, source=source, confidence=confidence)
    doc.date_of_issue = field_value(result.date_of_issue, source=source, confidence=confidence)
    doc.date_of_expiry = field_value(result.date_of_expiry, source=source, confidence=confidence)
    doc.sex = field_value(result.sex, source=source, confidence=confidence)
    doc.address = field_value(result.address, source=source, confidence=confidence)
    country_code, country_name = resolve_country_code(result.country)
    doc.country_code = field_value(country_code or result.country, source=source, confidence=confidence)
    doc.country_name = field_value(country_name, source=source, confidence=confidence)

    canonical = _CLASS_CODE_CANONICAL.get((result.address_state, result.license_class), "")
    doc.license_class = field_value(result.license_class, source=source, confidence=confidence)
    doc.license_class_canonical = canonical
    doc.license_restrictions = field_value(result.license_restrictions, source=source, confidence=confidence)
    doc.license_endorsements = field_value(result.license_endorsements, source=source, confidence=confidence)
    doc.restrictions_asserted_none = result.license_restrictions.strip().upper() == "NONE"
    doc.endorsements_asserted_none = result.license_endorsements.strip().upper() == "NONE"

    profile = JURISDICTION_FIELD_PROFILE.get(result.address_state, {})
    if result.eye_color:
        doc.eye_color = field_value(result.eye_color, source=source, confidence=confidence)
    if result.hair_color:
        doc.hair_color = field_value(result.hair_color, source=source, confidence=confidence)
    if result.compliance_type:
        doc.real_id_compliant = result.compliance_type.strip().upper() == "F"

    doc.extra = {
        "height": result.height, "weight": result.weight,
        "eye_color_raw": result.eye_color_raw,
        "document_discriminator": result.document_discriminator,
        "inventory_control_number": result.inventory_control_number,
        "card_revision_date": result.card_revision_date,
        "address_state": result.address_state, "address_county": result.address_county,
        "jurisdiction_field_profile": profile,
    }
    return doc


def _from_ethiopia_id(result: EthiopiaIDResult) -> ExtractedIDDocument:
    doc = ExtractedIDDocument(document_category="national_id")
    barcode_source: FieldSource = "barcode"
    ocr_source: FieldSource = "label_anchor"
    strong = bool(result.raw_qr_present or result.raw_barcode_present)
    doc.extraction_method = "barcode+ocr" if strong else "ocr_no_mrz"
    doc.confidence = "high" if strong else "medium"

    name_source = barcode_source if result.raw_qr_present else ocr_source
    name_conf = 0.9 if result.raw_qr_present else 0.55
    if result.full_name:
        doc.full_name = field_value(result.full_name, source=name_source, confidence=name_conf)
    if result.fan:
        fan_source = barcode_source if (result.raw_qr_present or result.raw_barcode_present) else ocr_source
        doc.document_number = field_value(result.fan, source=fan_source, confidence=0.9 if fan_source == "barcode" else 0.5)
    if result.dob_gregorian:
        doc.date_of_birth = field_value(result.dob_gregorian, source=ocr_source, confidence=0.6)
    if result.sex:
        doc.sex = field_value(result.sex, source=ocr_source, confidence=0.55)
    if result.date_of_issue:
        conf = 0.4 if result.date_of_issue_from_sidebar_ocr else 0.55
        doc.date_of_issue = field_value(result.date_of_issue, source=ocr_source, confidence=conf)
    if result.date_of_expiry:
        doc.date_of_expiry = field_value(result.date_of_expiry, source=ocr_source, confidence=0.55)
    if result.phone_number:
        doc.phone_number = field_value(result.phone_number, source=ocr_source, confidence=0.5)
    if result.nationality_qualifier:
        doc.nationality_qualifier = result.nationality_qualifier

    admin_levels = []
    if result.region:
        admin_levels.append({"label": "Region", "value": result.region})
    if result.zone:
        admin_levels.append({"label": "Zone", "value": result.zone})
    if result.woreda:
        admin_levels.append({"label": "Woreda", "value": result.woreda})
    doc.admin_levels = admin_levels

    doc.country_code = field_value("ETH", source=ocr_source, confidence=0.9)
    doc.country_name = field_value("ETHIOPIA", source=ocr_source, confidence=0.9)
    doc.nationality_code = field_value("ETH", source=ocr_source, confidence=0.7)
    doc.nationality_name = field_value("ETHIOPIAN", source=ocr_source, confidence=0.7)

    doc.verification_checks = dict(result.checks)
    doc.extra = {"fin": result.fin, "card_generation": result.card_generation,
                 "dob_ethiopian_display": result.dob_ethiopian_display}
    for check_name, passed in result.checks.items():
        if passed is False:
            doc.human_review_required = True
            doc.warnings.append(f"Ethiopian Digital ID cross-check failed: {check_name}")
    return doc


# ─────────────────────────────────────────────────────────────
# Fallback tiers
# ─────────────────────────────────────────────────────────────

def _universal_ai_fallback(
    image_path: str, image_path_back: Optional[str], doc_type_hint: str, trace_id: str,
) -> ExtractedIDDocument:
    raw = vision.extract_universal_document(image_path, image_path_back, doc_type_hint, trace_id=trace_id)
    doc = ExtractedIDDocument(document_category=raw.get("document_type_guess") or doc_type_hint or "unknown")
    source: FieldSource = "vlm"
    confidence = 0.55
    doc.extraction_method = "ai_vision"
    doc.confidence = "medium"

    doc.full_name = field_value(raw.get("full_name", ""), source=source, confidence=confidence)
    if raw.get("full_name_local"):
        doc.name_in_native_script = field_value(raw["full_name_local"], source=source, confidence=confidence)
    doc.surname = field_value(raw.get("surname", ""), source=source, confidence=confidence)
    doc.given_names = field_value(raw.get("given_names", ""), source=source, confidence=confidence)
    doc.sex = field_value(raw.get("sex", ""), source=source, confidence=confidence)
    doc.date_of_birth = field_value(raw.get("date_of_birth", ""), source=source, confidence=confidence)
    doc.date_of_issue = field_value(raw.get("date_of_issue", ""), source=source, confidence=confidence)
    doc.date_of_expiry = field_value(raw.get("date_of_expiry", ""), source=source, confidence=confidence)
    doc.document_number = field_value(raw.get("document_number", ""), source=source, confidence=confidence)
    nat_raw = raw.get("nationality", "")
    nat_code, nat_name = resolve_nationality_code(nat_raw)
    doc.nationality_code = field_value(nat_code or nat_raw, source=source, confidence=confidence)
    doc.nationality_name = field_value(nat_name or nat_raw, source=source, confidence=confidence)
    doc.address = field_value(raw.get("address", ""), source=source, confidence=confidence)
    doc.issuing_authority = field_value(raw.get("issuing_authority", ""), source=source, confidence=confidence)
    doc.place_of_birth = field_value(raw.get("place_of_birth", ""), source=source, confidence=confidence)
    doc.license_class = field_value(raw.get("license_class", ""), source=source, confidence=confidence)
    doc.license_restrictions = field_value(raw.get("license_restrictions", ""), source=source, confidence=confidence)
    doc.license_endorsements = field_value(raw.get("license_endorsements", ""), source=source, confidence=confidence)
    doc.permit_type = field_value(raw.get("permit_type", ""), source=source, confidence=confidence)

    country_raw = raw.get("issuing_country_guess", "")
    country_code, country_name = resolve_country_code(country_raw)
    doc.country_code = field_value(country_code or country_raw, source=source, confidence=confidence)
    doc.country_name = field_value(country_name or country_raw, source=source, confidence=confidence)

    compliance = (raw.get("real_id_compliant_guess") or "unknown").lower()
    doc.real_id_compliant = {"true": True, "false": False}.get(compliance)

    mrz_lines = [raw.get(f"mrz_line_{i}", "") for i in (1, 2, 3)]
    doc.mrz_raw = [l for l in mrz_lines if l]

    doc.warnings.append(
        "No checksum/barcode-verifiable data available for this document — extracted by local AI "
        "vision (Ollama), not independently verified."
    )
    if raw.get("notes"):
        doc.warnings.append(f"AI notes: {raw['notes']}")
    return doc


def _generic_ocr_fallback(image_path: str, doc_type_hint: str, trace_id: str) -> ExtractedIDDocument:
    """Last-resort, fully offline floor. Runs both the whole-text-blob
    regex search and the field-first sweep independently, merges whichever
    finds each field, flags disagreement — never returns an empty profile
    (CLAUDE.md invariant #5)."""
    text = ocr_full_text(image_path)
    blob_fields = extract_generic_fields(text)
    try:
        sweep_fields = extract_fields_via_sweep(image_path)
    except Exception as e:
        logger.warning("extract_fields_via_sweep failed: %s", e)
        sweep_fields = {}

    category = guess_document_category(text) or doc_type_hint or "unknown"
    doc = ExtractedIDDocument(document_category=category)
    doc.extraction_method = "ocr_generic"
    doc.confidence = "low"
    source: FieldSource = "label_anchor"

    merged: dict[str, str] = {}
    for key in set(blob_fields) | set(sweep_fields):
        blob_v = (blob_fields.get(key) or "").strip()
        sweep_v = (sweep_fields.get(key) or "").strip()
        if blob_v and sweep_v and blob_v.lower() != sweep_v.lower():
            doc.warnings.append(f"{key}: whole-text search and field sweep disagree ({blob_v!r} vs {sweep_v!r})")
            doc.human_review_required = True
        merged[key] = blob_v or sweep_v

    field_map = {
        "full_name": "full_name", "surname": "surname", "given_names": "given_names",
        "sex": "sex", "date_of_birth": "date_of_birth", "date_of_issue": "date_of_issue",
        "date_of_expiry": "date_of_expiry", "document_number": "document_number",
        "nationality": "nationality_code", "address": "address",
        "issuing_authority": "issuing_authority", "place_of_birth": "place_of_birth",
        "phone_number": "phone_number", "personal_number": "personal_number",
        "permit_type": "permit_type",
    }
    for src_key, doc_attr in field_map.items():
        value = merged.get(src_key, "")
        if value:
            setattr(doc, doc_attr, field_value(value, source=source, confidence=0.4))

    if _is_essentially_empty(doc):
        doc.warnings.append(
            "No fields could be read by any free channel and no usable AI-vision result was "
            "available either — this response reflects the image quality/format, not a bug."
        )
    return doc


# ─────────────────────────────────────────────────────────────
# Validity / coverage tier
# ─────────────────────────────────────────────────────────────

def _apply_validity_check(doc: ExtractedIDDocument, today: Optional[date] = None) -> None:
    today = today or date.today()
    expiry = doc.date_of_expiry.value if doc.date_of_expiry.state == "read" else ""
    if not expiry:
        return
    try:
        expiry_date = date.fromisoformat(expiry)
    except ValueError:
        return
    doc.document_currently_valid = expiry_date >= today
    doc.validity_checked_at = today.isoformat()
    if not doc.document_currently_valid:
        doc.warnings.append(f"Document expired on {expiry}")


def _coverage_tier(extraction_method: str, confidence: str) -> str:
    if extraction_method in ("mrz", "barcode", "barcode+ocr") and confidence == "high":
        return "A"
    if extraction_method in ("mrz_low_confidence", "ocr_no_mrz") or confidence == "medium" and extraction_method != "ai_vision":
        return "B"
    if extraction_method == "ai_vision":
        return "D"
    return "E"


# ─────────────────────────────────────────────────────────────
# Per-doc-type waterfalls
# ─────────────────────────────────────────────────────────────

def _try_passport(image_path: str, image_path_back: Optional[str], trace_id: str) -> Optional[ExtractedIDDocument]:
    if not _mrz_gate_passes(image_path, image_path_back):
        return None
    result = _try_parse_td3(image_path) or (_try_parse_td3(image_path_back) if image_path_back else None)
    if result is None:
        return None
    doc = _from_td3(result, image_path)
    if result.valid:
        mrz_hint = {
            "surname": doc.surname.value, "given_names": doc.given_names.value,
            "passport_number": doc.document_number.value,
        }
        try:
            crosscheck = vision.crosscheck_passport(image_path, mrz_hint, trace_id=trace_id)
            _merge_additive(doc, crosscheck, source="vlm_crosscheck", confidence=0.6)
        except Exception as e:
            logger.warning("passport crosscheck failed: %s", e)
    return doc


def _try_national_id(image_path: str, image_path_back: Optional[str], trace_id: str) -> Optional[ExtractedIDDocument]:
    """Barcode/QR checks first — decode_barcodes is ~0.3s and unambiguous,
    while the TD1/TD2 MRZ candidate search below is a slow multi-crop OCR
    operation (tens of seconds). Trying the cheap, decisive check first
    doesn't change which result wins (Precedence still puts barcode above
    MRZ regardless of order), it just avoids paying for an MRZ search on a
    document that plainly has no MRZ at all — confirmed against a real
    Ethiopian Digital ID, which used to pay for two failed MRZ searches
    before ever reaching its own barcode tier."""
    a, b = classify_ethiopia_id_sides(image_path, image_path_back or image_path)
    eth_result = parse_ethiopia_digital_id(a, b if image_path_back else None)
    if eth_result is not None:
        doc = _from_ethiopia_id(eth_result)
        try:
            hint = {"fan": eth_result.fan, "full_name": eth_result.full_name}
            crosscheck = vision.crosscheck_ethiopia_id(a, b if image_path_back else None, hint, trace_id=trace_id)
            mapped = {
                "full_name_local": crosscheck.get("full_name_amharic", ""),
                "phone_number": crosscheck.get("phone_number", ""),
                "address": "",
            }
            _merge_additive(doc, mapped, source="vlm_crosscheck", confidence=0.5)
        except Exception as e:
            logger.warning("ethiopia id crosscheck failed: %s", e)
        return doc

    for path in (image_path, image_path_back):
        if not path:
            continue
        for hit in decode_barcodes(path, formats=("PDF417",)):
            aamva = parse_aamva(hit.text)
            if aamva is not None:
                return _from_aamva(aamva)

    if _mrz_gate_passes(image_path, image_path_back):
        for parser, builder in ((_try_parse_td1, _from_td1), (_try_parse_td2, _from_td2)):
            result = parser(image_path) or (parser(image_path_back) if image_path_back else None)
            if result is not None:
                return builder(result, "national_id")

    return None


def _try_drivers_license(image_path: str, image_path_back: Optional[str], trace_id: str) -> Optional[ExtractedIDDocument]:
    for path in (image_path, image_path_back):
        if not path:
            continue
        for hit in decode_barcodes(path, formats=("PDF417",)):
            aamva = parse_aamva(hit.text)
            if aamva is not None:
                return _from_aamva(aamva)
    front_only = parse_front_only(image_path)
    if front_only is not None:
        return _from_aamva(front_only)
    return None


def _try_residence_permit(image_path: str, image_path_back: Optional[str], trace_id: str) -> Optional[ExtractedIDDocument]:
    if not _mrz_gate_passes(image_path, image_path_back):
        return None
    for parser, builder in ((_try_parse_td1, _from_td1), (_try_parse_td2, _from_td2)):
        result = parser(image_path) or (parser(image_path_back) if image_path_back else None)
        if result is not None:
            return builder(result, "residence_permit")
    return None


_TYPE_HANDLERS = {
    "passport": _try_passport,
    "national_id": _try_national_id,
    "drivers_license": _try_drivers_license,
    "residence_permit": _try_residence_permit,
}


def _finalize(doc: ExtractedIDDocument, image_path: str) -> ExtractedIDDocument:
    apply_plausibility_checks(doc)
    _apply_validity_check(doc)
    doc.coverage_tier = _coverage_tier(doc.extraction_method, doc.confidence)
    doc.source_image = image_path
    return doc


def _extract_for_type(
    image_path: str, doc_type: str, image_path_back: Optional[str], trace_id: str,
) -> ExtractedIDDocument:
    handler = _TYPE_HANDLERS.get(doc_type)
    doc = handler(image_path, image_path_back, trace_id) if handler else None

    if _is_essentially_empty(doc):
        try:
            doc = _universal_ai_fallback(image_path, image_path_back, doc_type, trace_id)
        except Exception as e:
            logger.warning("universal AI fallback failed: %s", e)
            doc = None

    if _is_essentially_empty(doc):
        doc = _generic_ocr_fallback(image_path, doc_type, trace_id)

    return _finalize(doc, image_path)


def extract_id_document(
    image_path: str,
    doc_type: str = "auto",
    image_path_back: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> ExtractedIDDocument:
    """Entry point. doc_type: "passport" | "national_id" | "drivers_license"
    | "residence_permit" | "auto" (tries passport, then national_id, then
    drivers_license, in that order — pick residence_permit explicitly,
    since its MRZ format can't be distinguished from national_id's).

    "auto" tries every free/structural handler across all three doc types
    FIRST — the expensive Ollama universal fallback only fires once, after
    none of them found enough, never per-type. Nesting it inside each
    type's own attempt (as an earlier version of this function did) meant
    the first type tried (passport) would exhaust its own AI fallback
    before "auto" ever got to try national_id/drivers_license at all —
    confirmed against a real Ethiopian Digital ID: it has no MRZ, so the
    passport attempt correctly found nothing structural, but the nested
    fallback then ran an 812-second VLM read instead of falling through
    to the national_id tier's barcode-verified 12-second read."""
    trace_id = trace_id or str(uuid.uuid4())

    if doc_type == "auto":
        doc = None
        for candidate_type in ("passport", "national_id", "drivers_license"):
            doc = _TYPE_HANDLERS[candidate_type](image_path, image_path_back, trace_id)
            if not _is_essentially_empty(doc):
                break

        if _is_essentially_empty(doc):
            try:
                doc = _universal_ai_fallback(image_path, image_path_back, "auto", trace_id)
            except Exception as e:
                logger.warning("universal AI fallback failed: %s", e)
                doc = None

        if _is_essentially_empty(doc):
            doc = _generic_ocr_fallback(image_path, "auto", trace_id)

        return _finalize(doc, image_path)

    return _extract_for_type(image_path, doc_type, image_path_back, trace_id)

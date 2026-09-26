"""
Local-VLM tier — replaces ID-Document-Extractor's idx/vision.py (Gemini).

Every prompt string and JSON-field shape below is ported verbatim from the
source project's vision.py — they're empirically tuned, and swapping the
model backing them doesn't change what makes a good extraction prompt.
What's replaced is `_call_gemini_vision`'s guts: instead of the Gemini SDK
with a hard `response_json_schema` constraint, calls go through
`ollama_client.vision_chat()` (already built — has the CPU-fallback retry
on Vulkan-backend failures found earlier this session) with a
prompt-engineered JSON instruction, parsed back with a forgiving
brace-matching parser (also already proven this session, in
crewai_pipeline.py). Dropped entirely: the daily budget guard, the
non-retryable-HTTP-status taxonomy, and Gemini's safety_settings — none of
these have an Ollama equivalent or a reason to exist for a local model.

Two calling patterns, deliberately different in cost:
  - `crosscheck_passport` / `crosscheck_ethiopia_id`: small, targeted
    enrichment calls (a handful of fields) — one direct vision_chat() call
    to qwen2.5vl:7b, our strongest multilingual model. Proportionate to
    what they're actually asking for.
  - `extract_universal_document`: the last-resort full read when nothing
    else worked at all. Reuses `crewai_pipeline.extract_with_crewai()`'s
    existing 4-stage sequential pipeline wholesale — unchanged, already
    resilient — rather than rebuilding a second extraction pipeline. Its
    output schema differs from the source project's UNIVERSAL_DOCUMENT_
    SCHEMA (fewer field names, no surname/given_names split), so this
    module maps between the two shapes; pipeline.py's builder is written
    against the mapped shape below, not the raw crewai_pipeline shape.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Optional

from ollama_client import vision_chat

_UNIVERSAL_MODEL = "qwen2.5vl:7b"


def _encode_images(*paths: Optional[str]) -> list[str]:
    out = []
    for path in paths:
        if not path:
            continue
        with open(path, "rb") as f:
            out.append(base64.b64encode(f.read()).decode())
    return out


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw.replace("```json", "").replace("```", "").strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s != -1 and e > s:
        try:
            parsed = json.loads(raw[s:e])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _call_local_vision(
    image_path: str, prompt: str, field_names: list[str], *,
    extra_image_path: Optional[str] = None, model: str = _UNIVERSAL_MODEL,
) -> dict[str, Any]:
    images = _encode_images(image_path, extra_image_path)
    schema_instruction = (
        "\n\nReturn ONLY a single JSON object with exactly these keys (use \"\" for "
        "anything not visible or not applicable — never omit a key): "
        + json.dumps(field_names) + ". No markdown, no explanation, just the JSON object."
    )
    try:
        response = vision_chat(
            model=model,
            messages=[{"role": "user", "content": prompt + schema_instruction, "images": images}],
        )
        return _parse_json(response["message"]["content"])
    except Exception as e:
        print(f"  ⚠️  vision_ollama: {model} call failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Prompts — ported verbatim from vision.py
# ─────────────────────────────────────────────────────────────

PASSPORT_CROSSCHECK_PROMPT = """
You are reading the biodata page of a passport. The Machine Readable Zone
(MRZ) at the bottom has already been decoded automatically. Your job is to
read the VISUAL (printed) fields that are NOT encoded in the MRZ, and to
flag if the visible name/passport number look different from what OCR
found (in case the automated MRZ read was wrong).

MRZ already decoded (for cross-check only, may contain OCR errors -- trust
the image over this if they disagree):
{mrz_hint}

Use an empty string for anything not visible. Dates in DD MON YYYY as printed.
"""

_PASSPORT_CROSSCHECK_FIELDS = [
    "surname", "given_names", "passport_number", "place_of_birth",
    "issuing_authority", "date_of_issue", "nationality", "mrz_mismatch_notes",
]

ETHIOPIA_ID_CROSSCHECK_PROMPT = """
You are reading an Ethiopian Digital ID Card ("Fayda" / National ID),
front and back. Some fields have already been read from the card's own
QR code / barcode (far more reliable than OCR for these) -- do not
second-guess those, they are provided only so you know this document has
already been partly identified:
{hint}

Your job is every OTHER field, read directly from the printed text on
both images. This card prints two calendars side by side for every date
(Gregorian and Ethiopian) -- report the Gregorian one in
date_of_birth/date_of_issue/date_of_expiry as ISO YYYY-MM-DD, and also
give whatever the second (Ethiopian-calendar) string says verbatim in the
_ethiopian_display fields, exactly as printed, no conversion.

The front shows a "FAN" (card number, 16 digits, may be printed in groups
of 4). The back separately shows a "FIN" (personal identification number,
12 digits) -- these are two different numbers for two different purposes;
do not confuse them or copy one into the other's field.

Use an empty string for anything not visible. Transcribe Amharic text as
written, do not translate it.
"""

_ETHIOPIA_ID_CROSSCHECK_FIELDS = [
    "full_name", "full_name_amharic", "sex",
    "date_of_birth", "date_of_birth_ethiopian_display",
    "date_of_issue", "date_of_issue_ethiopian_display",
    "date_of_expiry", "date_of_expiry_ethiopian_display",
    "phone_number", "region", "region_amharic",
    "zone_or_subcity", "zone_or_subcity_amharic", "woreda", "woreda_amharic",
    "fan_as_printed", "fin_as_printed", "nationality_qualifier",
]

# Field names in our locally-produced universal-document shape — mapped
# from crewai_pipeline's EMPTY_RESULT schema, see extract_universal_document.
UNIVERSAL_DOCUMENT_FIELDS = [
    "document_type_guess", "issuing_country_guess", "issuing_region_guess",
    "full_name", "full_name_local", "surname", "given_names", "sex",
    "date_of_birth", "date_of_issue", "date_of_expiry", "document_number",
    "nationality", "address", "issuing_authority", "place_of_birth",
    "license_class", "license_restrictions", "license_endorsements",
    "permit_type", "real_id_compliant_guess",
    "mrz_line_1", "mrz_line_2", "mrz_line_3",
    "language_detected", "notes",
]


def crosscheck_passport(image_path: str, mrz_hint: dict[str, Any], trace_id: Optional[str] = None) -> dict[str, Any]:
    prompt = PASSPORT_CROSSCHECK_PROMPT.format(mrz_hint=json.dumps(mrz_hint, ensure_ascii=False))
    return _call_local_vision(image_path, prompt, _PASSPORT_CROSSCHECK_FIELDS)


def crosscheck_ethiopia_id(
    front_path: str, back_path: Optional[str], hint: dict[str, Any], trace_id: Optional[str] = None,
) -> dict[str, Any]:
    prompt = ETHIOPIA_ID_CROSSCHECK_PROMPT.format(hint=json.dumps(hint, ensure_ascii=False))
    return _call_local_vision(front_path, prompt, _ETHIOPIA_ID_CROSSCHECK_FIELDS, extra_image_path=back_path)


def extract_universal_document(
    image_path: str,
    image_path_back: Optional[str] = None,
    doc_type_hint: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> dict[str, Any]:
    """Last-resort full read when nothing else (barcode/MRZ/generic OCR)
    produced enough. Reuses crewai_pipeline's existing 4-stage sequential
    pipeline wholesale, then maps its field names onto this module's
    UNIVERSAL_DOCUMENT_FIELDS shape for pipeline.py's builder to consume.
    `image_path_back` is accepted for interface parity with the source
    project but not yet used — crewai_pipeline's pipeline is single-image
    only; a two-sided document only gets its front read by this tier today.
    """
    from crewai_pipeline import extract_with_crewai

    result = extract_with_crewai(image_path)
    data = result.get("extracted_data", {})

    mrz_lines = (data.get("mrz_code") or "").splitlines()
    mrz_lines += [""] * (3 - len(mrz_lines))

    return {
        "document_type_guess": data.get("document_type", ""),
        "issuing_country_guess": "",
        "issuing_region_guess": "",
        "full_name": data.get("full_name", ""),
        "full_name_local": data.get("full_name_local", ""),
        "surname": "",
        "given_names": "",
        "sex": data.get("gender", ""),
        "date_of_birth": data.get("date_of_birth", ""),
        "date_of_issue": data.get("issue_date", ""),
        "date_of_expiry": data.get("expiry_date", ""),
        "document_number": data.get("id_number", ""),
        "nationality": data.get("nationality", ""),
        "address": data.get("address", ""),
        "issuing_authority": data.get("issuing_authority", ""),
        "place_of_birth": data.get("place_of_birth", ""),
        "license_class": "",
        "license_restrictions": "",
        "license_endorsements": "",
        "permit_type": "",
        "real_id_compliant_guess": "unknown",
        "mrz_line_1": mrz_lines[0],
        "mrz_line_2": mrz_lines[1],
        "mrz_line_3": mrz_lines[2],
        "language_detected": data.get("language_detected", ""),
        "notes": "",
    }


def has_api_key(api_key: Optional[str] = None) -> bool:
    """Kept for interface parity with the source project's pipeline.py —
    always True: there's no API key for a local model, just a running
    Ollama service (checked by GET /health, not here)."""
    return True


def budget_exceeded() -> bool:
    """No quota for a local model — always False."""
    return False

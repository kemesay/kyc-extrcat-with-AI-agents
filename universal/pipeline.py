"""
Universal extraction orchestrator.

    ┌──────────── deterministic pre-pass (≈0.5s, no GPU) ────────────┐
    │ barcode/QR decode (zxing) → AAMVA / Fayda payloads parsed      │
    └────────────────────────────────────────────────────────────────┘
                       │ verified barcode data is handed to the
                       ▼ field extractor as ground truth
    fast_scanner → field_extractor → accuracy_verifier → layout_analyzer
     (moondream)    (qwen2.5vl)       (llava:13b)         (qwen2.5vl)
                       each stage sees the previous stage's JSON
                       ▼
                   supervisor (qwen2.5, text-only) reconciles all four
                       ▼
    ┌──────────────── deterministic verification ────────────────────┐
    │ OCR grounding: RapidOCR (run beside the agents) + tesseract in  │
    │   the document's script — hallucination / calendar / truncation │
    │ MRZ check digits (agents' lines, else local OCR re-read)        │
    │ national-ID checksums + embedded DOB/sex cross-checks           │
    │ calendar conversion + dual-calendar agreement                   │
    │ ISO 3166 country resolution, plausibility, expiry               │
    └────────────────────────────────────────────────────────────────┘
                       ▼
          per-field value / confidence / provenance / flags

mode="fast" runs only the field extractor (one qwen2.5vl call) plus the
full deterministic layer — roughly 5x quicker, less cross-agent agreement.

Every agent call goes through universal/multipass.py: its prompt is packed
to the model's measured token budget, and a small-context model is run 2-3
times over field groups when the relevant prompt knowledge doesn't fit once
(agent_passes="auto" | 1 | 2 | 3).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Union

from .grounding import collect_evidence, early_ocr_text, ground_fields
from .id_numbers import check_numbers, compact, mask_aadhaar, verhoeff_valid
from .llm import clean_unicode, encode_image
from .multipass import MAX_PASSES, Hints, run_agent
from .tokens import context_limit
from .normalize import (
    detect_script, normalize_document_type, resolve_country, resolve_nationality, strip_markers,
)
from .plausibility import check_dual_calendar, check_plausibility
from .llm import chat_json
from .prompts import (
    AGENT_ORDER, AS_PRINTED_FIELDS, AS_PRINTED_INSTRUCTIONS, EXTRACTION_FIELDS, TRANSCRIBER_AGENT,
    TRANSCRIPTION_SYSTEM, build_agents,
)
from .reconcile import (
    CRITICAL_FIELDS, VISION_AGENTS, Context, FieldResult, apply_id_number_checks, apply_verified,
    consensus, cross_script_check, guard_local_script, normalize_value, prefer_latin_full_name, script_of_name, settle_name_order,
    settle_name_parts,
    settle_names,
)
from .printed import country_from_number, country_from_text, resolve_dates, resolve_local_name, resolve_document_number, resolve_names, resolve_sex
from .verify import VerifiedSource, mrz_from_ocr_text, read_barcodes, search_mrz_locally, verify_mrz_lines

ENGINE_VERSION = "1.0.0"
MODES = ("full", "fast")
# Documents whose FRONT (data page) always carries the MRZ. ID cards and
# residence permits print it on the back, so a front-only image of one would
# only waste a ~25 s local OCR search.
_MRZ_DOC_TYPES = {"passport", "visa", "refugee_travel_document"}
_QUALITY_LEVELS = ("Excellent", "Good", "Fair", "Poor")
_SCANNER_QUALITY = {"clear": "Good", "blurry": "Fair", "partial": "Fair", "damaged": "Poor"}


# ─────────────────────────────────────────────────────────────
# Agent stage
# ─────────────────────────────────────────────────────────────

def _context_block(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n\n═══════════════════════════════════════════════\nADDITIONAL CONTEXT\n" \
           "═══════════════════════════════════════════════\n" + "\n".join(lines)


def _agent_inputs(
    name: str, accumulated: dict, outputs: dict[str, dict],
    barcode: Optional[VerifiedSource], doc_type_hint: str, two_sided: bool,
) -> tuple[dict[str, Any], str]:
    """-> (placeholder values for the prompt template, ADDITIONAL CONTEXT text).
    Values stay as dicts: the multi-pass runner renders each pass's subset."""
    context: list[str] = []
    if doc_type_hint and doc_type_hint != "auto":
        context.append(f"The submitter states this document is a: {doc_type_hint.replace('_', ' ')}.")
    if name == "field_extractor":
        if two_sided:
            context.append("Two images are provided: the front and the back of the same document.")
        scan = outputs.get("fast_scanner") or {}
        pre = {k: scan.get(k) for k in ("document_type", "issuing_country", "language_primary",
                                        "script_type", "orientation") if scan.get(k)}
        if pre:
            context.append("A fast pre-scan suggested (unverified, correct it if wrong): "
                           + json.dumps(pre, ensure_ascii=False))
        if barcode is not None:
            context.append("MACHINE-VERIFIED data decoded from this document's barcode (error-corrected, "
                           "treat as ground truth): " + json.dumps(barcode.fields, ensure_ascii=False))
    values: dict[str, Any] = {}
    if name == "accuracy_verifier":
        values["previous_extraction"] = accumulated
    elif name == "layout_analyzer":
        values["verified_extraction"] = accumulated
    elif name == "supervisor":
        values["agent_outputs"] = {a: {k: v for k, v in outputs[a].items() if v}
                                   for a in VISION_AGENTS if outputs.get(a)}
    return values, _context_block(context)


def _hints(outputs: dict[str, dict], barcode: Optional[VerifiedSource], doc_type_hint: str = "auto") -> Hints:
    """What earlier agents (and the barcode) revealed about script/country,
    so later agents' prompts carry the relevant script-specific rules."""
    texts, countries = [], []
    for out in outputs.values():
        texts += [out.get(k, "") for k in ("script_type", "language_primary", "language_secondary")]
        texts.append(detect_script(out.get("full_name_local", "") or out.get("full_name", "")))
        countries.append(resolve_country(out.get("issuing_country", ""))[0])
    if barcode is not None:
        countries.append(barcode.fields.get("issuing_country", ""))
    doc_types = [normalize_document_type(o.get("document_type", "")) for o in outputs.values()]
    if doc_type_hint and doc_type_hint != "auto":
        doc_types.append(doc_type_hint)
    return Hints.from_signals([t for t in texts if t], [c for c in countries if c]).with_doc_types(doc_types)


def _merge(accumulated: dict, new: dict) -> dict:
    """What the next stage is shown: newer non-empty values win, markers
    stripped so '[CORRECTED]' doesn't compound stage over stage."""
    merged = dict(accumulated)
    for key in EXTRACTION_FIELDS:
        value, _ = strip_markers(new.get(key, ""))
        if value:
            merged[key] = value
    return merged


def run_agents(
    image_paths: list[str], mode: str, doc_type_hint: str, barcode: Optional[VerifiedSource],
    agent_passes: Union[str, int] = "auto",
) -> tuple[dict[str, dict], list[dict]]:
    agents = build_agents()
    encoded = [encode_image(p) for p in image_paths]
    plan = AGENT_ORDER if mode == "full" else ("field_extractor",)
    outputs: dict[str, dict] = {}
    log: list[dict] = []
    accumulated: dict = {}

    for order, name in enumerate(plan, start=1):
        spec = agents.get(name)
        if spec is None:
            continue
        values, context = _agent_inputs(name, accumulated, outputs, barcode, doc_type_hint,
                                        two_sided=len(encoded) > 1)
        images = encoded[:spec.max_images] if spec.is_vision else None
        print(f"  🤖 [{order}/{len(plan)}] {name} ({spec.model}) starting...")
        started = time.time()
        entry: dict[str, Any] = {"order": order, "agent": name, "model": spec.model,
                                 "images_sent": len(images or [])}
        try:
            data, meta = run_agent(spec.model, spec.system, spec.user_template, spec.output_fields,
                                   values, context, _hints(outputs, barcode, doc_type_hint), images, agent_passes)
            entry.update(meta)
            entry["status"] = "success" if any(data.values()) else "empty"
        except Exception as e:
            data = {}
            entry.update(status="failed", error=str(e)[:300])
        entry["duration_seconds"] = round(time.time() - started, 2)
        passes = f", {entry['passes']} passes" if entry.get("passes", 1) > 1 else ""
        print(f"  {'✅' if entry['status'] == 'success' else '⚠️ '} [{order}/{len(plan)}] {name} "
              f"{entry['status']} in {entry['duration_seconds']}s{passes}")
        outputs[name] = data if entry["status"] == "success" else {}
        if name in VISION_AGENTS and outputs[name]:
            accumulated = _merge(accumulated, outputs[name])
        log.append(entry)
        if name == "field_extractor":
            log.append(_run_transcriber(spec.model, encoded[:spec.max_images], outputs))
    return outputs, log


def _run_transcriber(model: str, images: list[str], outputs: dict[str, dict]) -> dict:
    """Label-anchored raw transcription in its own short call (see prompts.TRANSCRIBER_AGENT)."""
    print(f"  🤖 [+] {TRANSCRIBER_AGENT} ({model}) starting...")
    started = time.time()
    entry: dict[str, Any] = {"agent": TRANSCRIBER_AGENT, "model": model, "images_sent": len(images)}
    try:
        data, meta = chat_json(model, TRANSCRIPTION_SYSTEM, AS_PRINTED_INSTRUCTIONS, images=images,
                               fields=AS_PRINTED_FIELDS, num_ctx=context_limit(model), num_predict=600)
        entry.update(meta, status="success" if any(data.values()) else "empty")
        outputs[TRANSCRIBER_AGENT] = data if any(data.values()) else {}
    except Exception as e:
        entry.update(status="failed", error=str(e)[:300])
    entry["duration_seconds"] = round(time.time() - started, 2)
    print(f"  {'✅' if entry['status'] == 'success' else '⚠️ '} [+] {TRANSCRIBER_AGENT} {entry['status']} "
          f"in {entry['duration_seconds']}s")
    return entry


# ─────────────────────────────────────────────────────────────
# Reconciliation
# ─────────────────────────────────────────────────────────────

def _majority(values: list[str]) -> str:
    values = [v for v in values if v]
    return Counter(values).most_common(1)[0][0] if values else ""


def _initial_context(outputs: dict[str, dict], barcode: Optional[VerifiedSource]) -> Context:
    ctx = Context()
    countries = [resolve_country(o.get("issuing_country", ""))[0] for o in outputs.values()]
    ctx.country = (barcode.fields.get("issuing_country") if barcode else "") or _majority(countries)
    ctx.calendar = _majority([o.get("calendar_system", "") for o in outputs.values()])
    return ctx


def _find_mrz(outputs: dict[str, dict], fields: dict[str, FieldResult], image_paths: list[str],
              local_search: bool, doc_type_hint: str, ocr_text: str = "") -> tuple[Optional[VerifiedSource], dict]:
    report: dict[str, Any] = {"present": False}
    # 1. MRZ lines already in the OCR text (both sides, read beside the agents) —
    #    the agents often don't transcribe an MRZ at all, especially a card's back.
    best_invalid, ocr_saw_mrz = mrz_from_ocr_text(ocr_text)
    if best_invalid is not None and best_invalid.details["valid"] and best_invalid.details["all_checks_passed"]:
        if best_invalid.details.get("name_line_uncertain") and local_search:
            # The two OCR engines disagree on the (unchecked) name line: the local
            # MRZ search re-reads the zone under several preprocessings and votes.
            print("  🔍 MRZ name line disputed — re-reading it with local OCR voting...")
            fmt = best_invalid.details["format"]
            local = search_mrz_locally(image_paths, (fmt,))
            same_doc = local is not None and \
                local.fields.get("document_number") == best_invalid.fields.get("document_number")
            if same_doc:
                local.details["settled_disputed_name_line"] = best_invalid.details.get("name_line_readings")
                return local, {"present": True, **local.details}
        return best_invalid, {"present": True, **best_invalid.details}
    agents_saw_mrz = ocr_saw_mrz
    for agent in ("accuracy_verifier", "layout_analyzer", "field_extractor", "supervisor"):
        out = outputs.get(agent) or {}
        # Only MRZ-shaped text counts: junk in an MRZ field must not trigger a
        # ~20 s local MRZ search on a document that has no MRZ at all.
        lines = [normalize_value(f"mrz_line_{i}", out.get(f"mrz_line_{i}", ""), Context())[0] for i in (1, 2, 3)]
        lines = [l for l in lines if l]
        if not lines:
            continue
        agents_saw_mrz = True
        src = verify_mrz_lines(lines)
        if src is None:
            continue
        src.details["transcribed_by"] = agent
        if src.details["valid"] and src.details["all_checks_passed"]:
            return src, {"present": True, **src.details}
        if best_invalid is None or (src.details["valid"] and not best_invalid.details["valid"]):
            best_invalid = src

    doc_type = fields["document_type"].value or doc_type_hint
    agents_failed = not any(outputs.get(a) for a in VISION_AGENTS)
    if local_search and (agents_saw_mrz or doc_type in _MRZ_DOC_TYPES or agents_failed):
        formats = ("TD3", "TD1", "TD2") if doc_type in ("passport", "") else ("TD1", "TD2", "TD3")
        print(f"  🔍 re-reading MRZ with local OCR ({'/'.join(formats)})...")
        local = search_mrz_locally(image_paths, formats)
        if local is not None:
            if best_invalid is not None:
                local.details["agents_transcription_failed_checks"] = best_invalid.details["checks"]
            return local, {"present": True, **local.details}

    if best_invalid is not None:
        return best_invalid, {"present": True, **best_invalid.details}
    if agents_saw_mrz:
        report.update(present=True, valid=False, note="agents reported MRZ text that is not a parseable MRZ")
    return None, report


def _document_quality(outputs: dict[str, dict]) -> str:
    sup = (outputs.get("supervisor") or {}).get("document_quality", "").strip().title()
    if sup in _QUALITY_LEVELS:
        return sup
    scan = (outputs.get("fast_scanner") or {}).get("image_quality", "").strip().lower()
    return next((q for k, q in _SCANNER_QUALITY.items() if k in scan), "")


def _overall_confidence(fields: dict[str, FieldResult]) -> float:
    required = ["full_name", "document_number", "date_of_birth"]
    optional = [f for f in ("expiry_date", "nationality", "issuing_country") if fields[f].value]
    scores = [fields[f].confidence if fields[f].value else 0.0 for f in required + optional]
    return round(sum(scores) / len(scores), 3)


def _completeness(fields: dict[str, FieldResult]) -> int:
    """SUPERVISOR_PROMPT Rule 8's definition: share of schema fields filled."""
    filled = sum(1 for f in EXTRACTION_FIELDS if f != "extraction_notes" and fields[f].value)
    return round(100 * filled / (len(EXTRACTION_FIELDS) - 1))


def _recheck_name_reviews(fields: dict[str, FieldResult], ctx: Context, ocr_text: str) -> None:
    """Grounding judged the agents' name before the MRZ / labels settled it; a
    review raised against a name that has since been replaced is withdrawn when
    the FINAL name is verified or printed on the document."""
    letters = re.sub(r"[^A-Z]", "", ocr_text.upper())
    for name in ("full_name", "full_name_latin", "last_name"):
        reason = f"{name}: not corroborated by OCR"
        if reason not in ctx.review_reasons:
            continue
        fr = fields[name]
        tokens = [re.sub(r"[^A-Z]", "", t) for t in fr.value.upper().split()]
        printed = bool(tokens) and all(t in letters for t in tokens if len(t) >= 2)
        if printed or {"mrz", "barcode"} & set(fr.verified_by):
            ctx.review_reasons.remove(reason)


def _clean_strings(obj: Any) -> Any:
    if isinstance(obj, str):
        return clean_unicode(obj)
    if isinstance(obj, list):
        return [_clean_strings(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _clean_strings(v) for k, v in obj.items()}
    return obj


_AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\d)")


def _mask(obj: Any) -> Any:
    if isinstance(obj, str):
        return _AADHAAR_RE.sub(lambda m: mask_aadhaar(m.group()) if verhoeff_valid(re.sub(r"\s", "", m.group()))
                               else m.group(), obj)
    if isinstance(obj, list):
        return [_mask(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _mask(v) for k, v in obj.items()}
    return obj


def extract_universal(
    image_path: str,
    image_path_back: Optional[str] = None,
    *,
    mode: str = "full",
    doc_type_hint: str = "auto",
    local_mrz_search: bool = True,
    ocr_grounding: bool = True,
    agent_passes: Union[str, int] = "auto",
    include_agent_outputs: bool = False,
    mask_sensitive: bool = True,
    trace_id: Optional[str] = None,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if agent_passes != "auto" and str(agent_passes) not in {str(i) for i in range(1, MAX_PASSES + 1)}:
        raise ValueError(f"agent_passes must be 'auto' or 1..{MAX_PASSES}")
    agent_passes = agent_passes if agent_passes == "auto" else int(agent_passes)
    started = time.time()
    trace_id = trace_id or str(uuid.uuid4())
    image_paths = [p for p in (image_path, image_path_back) if p]

    print(f"\n{'=' * 60}\n🌍 Universal extraction (mode={mode}, doc_type_hint={doc_type_hint})\n{'=' * 60}")

    timings: dict[str, float] = {}
    mark = time.time()

    def lap(stage: str) -> None:
        nonlocal mark
        timings[stage] = round(time.time() - mark, 2)
        mark = time.time()

    barcode, barcode_hits = read_barcodes(image_paths)
    lap("barcodes")
    if barcode_hits:
        print(f"  ▦ {len(barcode_hits)} barcode(s) decoded locally"
              + (f" — {barcode.details.get('payload')} payload parsed" if barcode else ""))

    with ThreadPoolExecutor(max_workers=1) as pool:
        rapid_future = pool.submit(early_ocr_text, image_paths) if ocr_grounding else None
        outputs, agent_log = run_agents(image_paths, mode, doc_type_hint, barcode, agent_passes)
        rapid_text = rapid_future.result() if rapid_future else ""
    lap("agents")  # RapidOCR runs concurrently inside this window
    vision_ok = [a for a in VISION_AGENTS if outputs.get(a)]

    # Normalize every agent's values, then take per-field consensus.
    ctx = _initial_context(outputs, barcode)
    supervisor_out = outputs.get("supervisor") or {}
    fields: dict[str, FieldResult] = {}
    for name in EXTRACTION_FIELDS:
        per_agent = {}
        for agent in vision_ok:
            raw = outputs[agent].get(name, "")
            value, flags = normalize_value(name, raw, ctx)
            per_agent[agent] = (value, flags, raw)
        sup = None
        if supervisor_out.get(name):
            sv, sf = normalize_value(name, supervisor_out[name], ctx)
            sup = (sv, sf, supervisor_out[name])
        fields[name] = consensus(name, per_agent, sup, len(vision_ok))

    # The agents' raw transcriptions, interpreted by rule (names, dates, numbers).
    verification: dict[str, Any] = {}
    ctx.country = resolve_country(fields["issuing_country"].value)[0] or ctx.country
    if fields["document_type"].value in ("", "other"):
        header_type = normalize_document_type(" ".join(rapid_text.splitlines()[:8]))
        if header_type not in ("", "other"):
            fields["document_type"] = FieldResult(value=header_type, confidence=0.7, source="ocr_text",
                                                  alternatives=[v for v in [fields["document_type"].value] if v])
    if not ctx.country:
        # Evidence order: a distinctive checksum-valid number, then the printed header.
        numbers = [fields["document_number"].value, fields["personal_number"].value,
                   *(o.get("document_number_as_printed", "") for o in outputs.values())]
        by_number, by_text = country_from_number(numbers), ""
        if not by_number:
            by_text = country_from_text(rapid_text)
        ctx.country = by_number or by_text
        if ctx.country:
            source = "id_number" if by_number else "ocr_header"
            fields["issuing_country"] = FieldResult(value=ctx.country, confidence=0.85 if by_number else 0.7,
                                                    source=source, verified_by=[source])
            ctx.warn(f"issuing_country {ctx.country} recognised from the "
                     + ("document number's national format" if by_number else "printed header"))
    verification["printed_text"] = {
        "names": resolve_names(fields, outputs, ctx, rapid_text),
        "local_name": resolve_local_name(fields, outputs, ctx),
        "dates": resolve_dates(fields, outputs, ctx, rapid_text),
        "sex": resolve_sex(fields, outputs),
        "document_number": resolve_document_number(fields, outputs, ctx, rapid_text),
    }

    # Machine-verified sources, strongest first.
    if barcode is not None:
        apply_verified(fields, barcode, ctx)
    verification["barcode"] = {"decoded": [h["format"] for h in barcode_hits],
                               "parsed_payload": barcode.details.get("payload") if barcode else None}
    if barcode_hits:
        fields["barcode_data"] = FieldResult(
            value=" | ".join(h["text"] for h in barcode_hits)[:2000], confidence=0.97,
            source="barcode", verified_by=["barcode"])
    elif fields["barcode_data"].value:
        fields["barcode_data"].confidence = 0.2
        fields["barcode_data"].flags.append("vision_models_cannot_decode_barcodes")

    lap("consensus")
    mrz, mrz_report = _find_mrz(outputs, fields, image_paths, local_mrz_search, doc_type_hint, rapid_text)
    lap("mrz")
    if mrz is not None:
        apply_verified(fields, mrz, ctx)
        if not mrz.details["valid"]:
            ctx.review("MRZ present but its check digits fail")
    verification["mrz"] = mrz_report

    ctx.country = resolve_country(fields["issuing_country"].value)[0] or ctx.country
    if ocr_grounding:
        evidence = collect_evidence(rapid_text, image_paths, ctx.country, script_of_name(fields))
        grounding = ground_fields(fields, evidence, ctx, len(vision_ok))
        verification["ocr_grounding"] = {**evidence.summary(), **grounding}
        doc_no, personal = compact(fields["document_number"].value), compact(fields["personal_number"].value)
        if personal and personal == doc_no:
            fields["personal_number"] = FieldResult()  # same number reported twice
        lap("ocr_grounding")
    id_checks = check_numbers(ctx.country, {f: fields[f].value for f in ("document_number", "personal_number")})
    verification["id_numbers"] = apply_id_number_checks(fields, id_checks, ctx)

    settle_names(fields)
    settle_name_order(fields, ctx)
    prefer_latin_full_name(fields, ctx)
    given_names = settle_name_parts(fields, ctx)
    local_rejected = guard_local_script(fields, ctx)
    verification["cross_script"] = cross_script_check(fields, ctx)
    local_rejected = local_rejected or any("dropped" in c for c in verification["cross_script"])
    _recheck_name_reviews(fields, ctx, rapid_text)
    verification["dual_calendar"] = check_dual_calendar(fields, ctx)
    verification["plausibility"] = check_plausibility(fields, ctx)
    lap("checks")

    if not fields["script_type"].value:
        script = script_of_name(fields)
        if script:
            fields["script_type"] = FieldResult(value=script, confidence=0.9, source="derived")

    nationality_code, _ = resolve_nationality(fields["nationality"].value)
    issuing_code, issuing_name = resolve_country(fields["issuing_country"].value)

    if not vision_ok:
        ctx.warn("no AI agent produced a result — only machine-readable data (if any) is reported")
        ctx.review("all AI agents failed")
    for name in ("full_name", "document_number", "date_of_birth"):
        if not fields[name].value:
            ctx.review(f"{name} could not be read")

    overall = _overall_confidence(fields)
    if overall < 0.5:
        ctx.review(f"overall confidence {overall} is below 0.5")

    extracted = {name: fields[name].value for name in EXTRACTION_FIELDS}
    extracted["given_names"] = given_names
    local = (verification.get("printed_text") or {}).get("local_name") or {}
    fixes = {c["from"]: c["to"] for c in verification.get("cross_script") or [] if "from" in c}
    respell = lambda v: " ".join(fixes.get(w, w) for w in v.split())
    extracted["surname_local"] = "" if local_rejected else respell(local.get("surname_local", ""))
    extracted["given_names_local"] = "" if local_rejected else respell(local.get("given_names_local", ""))
    # Label junk ('<NAME><NAME>', 'ERIKSSON NAME') is cleaned out of the printed-order name.
    printed_name = normalize_value("full_name", fields["full_name"].original, Context())[0]
    code, surname = issuing_code or "", fields["last_name"].value.upper()
    words = printed_name.split()
    if code and words and surname and words[0].upper() in (code + surname, "P" + code + surname, "I" + code + surname):
        printed_name = " ".join([fields["last_name"].value] + words[1:])     # an MRZ-line copy ('UTOERIKSSON …')
    extracted["full_name_as_printed"] = printed_name or fields["full_name"].value
    extracted["nationality_code"] = nationality_code
    extracted["issuing_country_name"] = issuing_name

    result: dict[str, Any] = {
        "engine": "universal",
        "engine_version": ENGINE_VERSION,
        "mode": mode,
        "trace_id": trace_id,
        "document_type": fields["document_type"].value or (doc_type_hint if doc_type_hint != "auto" else "unknown"),
        "issuing_country": issuing_code,
        "extracted_data": extracted,
        "fields": {name: fields[name].to_dict() for name in EXTRACTION_FIELDS},
        "overall_confidence": overall,
        "completeness_percent": _completeness(fields),
        "document_quality": _document_quality(outputs),
        "document_currently_valid": verification["plausibility"].get("document_currently_valid"),
        "human_review_required": bool(ctx.review_reasons),
        "review_reasons": ctx.review_reasons,
        "warnings": ctx.warnings,
        "verification": verification,
        "agents": agent_log,
        "critical_fields": list(CRITICAL_FIELDS),
        "total_duration_seconds": round(time.time() - started, 2),
        "timings_seconds": timings,
    }
    if include_agent_outputs:
        result["agent_outputs"] = outputs

    result = _clean_strings(result)
    masked = mask_sensitive and (issuing_code == "IND" or any(c.scheme == "aadhaar" for c in id_checks))
    if masked:
        result = _mask(result)
        result["masked"] = ["aadhaar_number"]

    print(f"\n✅ Universal extraction done in {result['total_duration_seconds']}s | "
          f"type={result['document_type']} country={issuing_code or '?'} "
          f"confidence={overall} review={result['human_review_required']}")
    return result

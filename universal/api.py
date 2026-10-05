"""
HTTP routes for the universal engine — mounted by main.py under /universal.

Mirrors the existing /extract/* contract (async + poll, wait=true,
callback_url webhook, sync endpoint, file_back for two-sided documents) so
a client that already integrates with /extract/* can switch by changing the
path. Nothing here touches the existing endpoints' state.

Every request goes through the durable priority queue (universal.jobqueue)
and is run by the worker pool (universal.service) — in this process by
default, or by separate `python -m universal.worker` processes on other
machines (UNIVERSAL_INPROCESS_WORKERS=0 makes this an API-only node).

Quality policy: the full agent team is the only mode unless the operator
sets UNIVERSAL_ALLOW_FAST_MODE=1.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from webhook import deliver_callback, is_valid_callback_url

from . import capture, inference, metrics
from .grounding import _tesseract_languages
from .id_numbers import supported_schemes
from .jobqueue import PRIORITIES, backend_name, get_queue
from .pipeline import ENGINE_VERSION
from .prompts import AGENT_ORDER, build_agents
from .service import WorkerPool, dedupe_key
from .tokens import context_limit

router = APIRouter(prefix="/universal", tags=["universal"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = int(float(os.environ.get("UNIVERSAL_MAX_UPLOAD_MB", "25")) * 1024 * 1024)
SYNC_TIMEOUT = float(os.environ.get("UNIVERSAL_SYNC_TIMEOUT", "900"))
EXACT_CACHE_SECONDS = float(os.environ.get("UNIVERSAL_EXACT_CACHE_SECONDS", "0"))   # 0 = off

DocType = Literal["auto", "passport", "national_id", "drivers_license", "residence_permit", "visa",
                  "refugee_travel_document", "military_id", "employee_id", "student_id", "voter_id",
                  "work_permit", "health_insurance_card", "tax_id", "other"]
Mode = Literal["full", "fast"]
Passes = Literal["auto", "1", "2", "3"]
Priority = Literal["live", "online", "batch", "bulk"]
_PASSES_DOC = ("How many times each agent may be run. auto: the fewest passes that fit ALL relevant prompt "
               "knowledge into each small-context model's budget (moondream 2K, llava 4K); 1: a single "
               "maximally-packed call; 2/3: force that many field-group passes per agent.")
_PRIORITY_DOC = ("live: customer waiting at a counter / in the app; online: portal KYC; batch: compliance "
                 "re-verification; bulk: historical backfill. Higher tiers are served first; lower tiers age "
                 "forward so they are never starved.")


class UniversalJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    engine: str = "universal"
    priority: Optional[str] = None
    queue_position: Optional[int] = None
    result: Optional[dict] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fast_allowed() -> bool:
    return os.environ.get("UNIVERSAL_ALLOW_FAST_MODE", "0") in ("1", "true", "yes")


# ── worker pool (lazy: started by the first request, never at import) ──

_pool: Optional[WorkerPool] = None
_pool_lock = threading.Lock()


def _deliver(job_id: str, callback_url: str) -> dict:
    return deliver_callback(callback_url, _response(job_id).model_dump())


def ensure_workers() -> Optional[WorkerPool]:
    global _pool
    if os.environ.get("UNIVERSAL_INPROCESS_WORKERS", "auto") in ("0", "false", "no"):
        return None
    with _pool_lock:
        if _pool is None:
            count = os.environ.get("UNIVERSAL_INPROCESS_WORKERS", "auto")
            _pool = WorkerPool(workers=None if count == "auto" else int(count), deliver=_deliver).start()
        return _pool


# ── uploads ─────────────────────────────────────────────────────────────

async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    ext = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_TYPES and ext not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {file.content_type!r}. Use JPG/PNG/WEBP/BMP/TIFF.")
    data = await file.read()
    if not data:
        raise HTTPException(400, f"{file.filename or 'upload'} is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{file.filename or 'upload'} exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    return (ext if ext in ALLOWED_SUFFIXES else ".jpg"), data


def _capture(files: dict[str, tuple[str, bytes]]) -> dict:
    try:
        sides = {side: capture.assess(io.BytesIO(data)) for side, (_, data) in files.items()}
    except Exception as e:
        raise HTTPException(400, f"image could not be decoded: {e}")
    order = {"good": 0, "fair": 1, "poor": 2}
    return {"grade": max((s["grade"] for s in sides.values()), key=order.get), "sides": sides}


def _response(job_id: str) -> UniversalJobResponse:
    q = get_queue()
    job = q.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found")
    status = job["status"]
    common = {"job_id": job_id, "status": status, "priority": job["priority_name"]}
    if status == "completed":
        result = dict(job["result"] or {})
        if job.get("callback"):
            result["callback_delivery"] = job["callback"]
        return UniversalJobResponse(**common, message="Extraction complete", result=result)
    if status == "failed":
        return UniversalJobResponse(**common, message=job.get("error") or "Unknown error")
    position = q.position(job_id) if status == "queued" else None
    return UniversalJobResponse(**common, queue_position=position, message=f"Job is {status}...")


async def _enqueue(file: UploadFile, file_back: Optional[UploadFile], options: dict, priority: str,
                   callback_url: Optional[str], reject_poor_capture: bool, fresh: bool) -> tuple[str, Optional[dict]]:
    """-> (job_id, cached result or None)."""
    if options["mode"] == "fast" and not _fast_allowed():
        raise HTTPException(400, "mode=fast is disabled on this server: quality policy runs the full agent team "
                                 "(set UNIVERSAL_ALLOW_FAST_MODE=1 to allow it)")
    files = {"front": await _read_upload(file)}
    if file_back is not None:
        files["back"] = await _read_upload(file_back)
    quality = _capture(files)
    metrics.inc("universal_capture_total", grade=quality["grade"])
    if reject_poor_capture and quality["grade"] == "poor":
        issues = sorted({i for s in quality["sides"].values() for i in s["issues"]})
        raise HTTPException(422, {"message": "capture quality is too poor to read reliably — please retake the photo",
                                  "issues": issues, "capture_quality": quality})
    key = dedupe_key(files, options, callback_url)
    q = get_queue()
    if EXACT_CACHE_SECONDS > 0 and not fresh:
        hit = q.find_completed(key, EXACT_CACHE_SECONDS)
        # Only a byte-identical, review-free result is ever reused.
        if hit and hit["result"] and not hit["result"].get("human_review_required"):
            metrics.inc("universal_exact_cache_hits_total")
            return hit["id"], {**hit["result"], "served_from": "exact_duplicate_cache", "original_job_id": hit["id"]}
    payload = {"options": options, "callback_url": callback_url, "capture_quality": quality,
               "filenames": {"front": file.filename, "back": file_back.filename if file_back else None}}
    job_id, joined = q.submit(payload, files, priority, key)
    if joined:
        metrics.inc("universal_duplicates_joined_total")
    ensure_workers()
    return job_id, None


async def _wait(job_id: str, timeout: float) -> UniversalJobResponse:
    deadline = time.monotonic() + timeout
    while True:
        response = _response(job_id)
        if response.status in ("completed", "failed"):
            return response
        if time.monotonic() > deadline:
            raise HTTPException(504, f"Job {job_id} still {response.status} after {timeout:.0f}s — "
                                     f"poll GET /universal/status/{job_id}")
        await asyncio.sleep(0.25)


def _options(mode: str, doc_type: str, local_mrz_search: bool, ocr_grounding: bool,
             include_agent_outputs: bool, agent_passes: str = "auto") -> dict:
    return {"mode": mode, "doc_type_hint": doc_type, "local_mrz_search": local_mrz_search,
            "ocr_grounding": ocr_grounding, "include_agent_outputs": include_agent_outputs,
            "agent_passes": agent_passes}


_FILE_BACK_DOC = "Optional back side of a two-sided document (order doesn't matter)."
_MODE_DOC = ("full: 5-agent team + transcriber (moondream → qwen2.5vl → llava → qwen2.5vl → qwen2.5 supervisor). "
             "fast (single qwen2.5vl read) is disabled unless the server sets UNIVERSAL_ALLOW_FAST_MODE=1. Both run "
             "the full deterministic verification layer (MRZ, barcodes, ID checksums, calendars, OCR grounding).")
_REJECT_DOC = ("Refuse a capture whose measured quality is poor (blur, darkness, glare, tiny resolution) with 422 "
               "at upload, so the customer can retake it immediately. Thresholds sit below every capture the "
               "engine has read correctly.")
_FRESH_DOC = "Bypass the exact-duplicate result cache (only relevant when the server enables it)."


@router.post("/extract", response_model=UniversalJobResponse)
async def universal_extract(
    file: UploadFile = File(...),
    file_back: Optional[UploadFile] = File(default=None, description=_FILE_BACK_DOC),
    doc_type: DocType = Query("auto", description="A hint for the agents; they still classify the document."),
    mode: Mode = Query("full", description=_MODE_DOC),
    priority: Priority = Query("online", description=_PRIORITY_DOC),
    wait: bool = Query(False, description="Block until done and return the result (not with callback_url)."),
    callback_url: Optional[str] = Query(None, description="POST the finished result to this http(s) URL."),
    local_mrz_search: bool = Query(True, description="Re-read the MRZ with local OCR when the agents' copy fails its check digits."),
    ocr_grounding: bool = Query(True, description="Cross-check agent values against local OCR (hallucination / calendar / truncation guard)."),
    include_agent_outputs: bool = Query(False, description="Include each agent's raw JSON (debugging)."),
    agent_passes: Passes = Query("auto", description=_PASSES_DOC),
    reject_poor_capture: bool = Query(False, description=_REJECT_DOC),
    fresh: bool = Query(False, description=_FRESH_DOC),
):
    """Universal ID extraction — any country, language, script or ID type."""
    if wait and callback_url:
        raise HTTPException(400, "wait=true and callback_url are mutually exclusive — pick one")
    if callback_url and not is_valid_callback_url(callback_url):
        raise HTTPException(400, "callback_url must start with http:// or https://")
    options = _options(mode, doc_type, local_mrz_search, ocr_grounding, include_agent_outputs, agent_passes)
    job_id, cached = await _enqueue(file, file_back, options, priority, callback_url, reject_poor_capture, fresh)
    if cached is not None:
        return UniversalJobResponse(job_id=job_id, status="completed", priority=priority,
                                    message="Extraction complete (identical submission)", result=cached)
    if wait:
        return await _wait(job_id, SYNC_TIMEOUT)
    response = _response(job_id)
    response.message = (f"Job queued. Result will be POSTed to {callback_url} when done." if callback_url
                        else f"Job queued. Poll GET /universal/status/{job_id}")
    return response


@router.post("/extract/sync")
async def universal_extract_sync(
    file: UploadFile = File(...),
    file_back: Optional[UploadFile] = File(default=None, description=_FILE_BACK_DOC),
    doc_type: DocType = Query("auto"),
    mode: Mode = Query("full", description=_MODE_DOC),
    priority: Priority = Query("online", description=_PRIORITY_DOC),
    local_mrz_search: bool = Query(True),
    ocr_grounding: bool = Query(True),
    include_agent_outputs: bool = Query(False),
    agent_passes: Passes = Query("auto", description=_PASSES_DOC),
    reject_poor_capture: bool = Query(False, description=_REJECT_DOC),
    fresh: bool = Query(False, description=_FRESH_DOC),
):
    """Synchronous universal extraction — returns the full result directly."""
    options = _options(mode, doc_type, local_mrz_search, ocr_grounding, include_agent_outputs, agent_passes)
    job_id, cached = await _enqueue(file, file_back, options, priority, None, reject_poor_capture, fresh)
    if cached is None:
        response = await _wait(job_id, SYNC_TIMEOUT)
        if response.status == "failed":
            raise HTTPException(500, response.message)
        cached = response.result
    return {"job_id": job_id, "status": "completed", **cached, "timestamp": _now()}


@router.get("/status/{job_id}", response_model=UniversalJobResponse)
def universal_status(job_id: str):
    return _response(job_id)


@router.delete("/job/{job_id}")
def universal_delete(job_id: str):
    outcome = get_queue().cancel(job_id)
    if outcome == "not_found":
        raise HTTPException(404, "Job not found")
    if outcome == "busy":
        raise HTTPException(409, "Job is being processed — delete it after it finishes")
    return {"message": f"Job {job_id} {outcome}"}


@router.get("/metrics", response_class=PlainTextResponse)
def universal_metrics():
    """Prometheus scrape endpoint."""
    return metrics.render()


@router.get("/health")
def universal_health():
    """Readiness: inference hosts reachable, team models resident on the GPU, queue moving."""
    team = inference.team_models()
    hosts = inference.residency()
    problems = []
    for h in hosts:
        if not h["reachable"]:
            problems.append(f"{h['host']} unreachable")
        elif h["partially_on_cpu"]:
            problems.append(f"{h['host']}: {', '.join(h['partially_on_cpu'])} partly on CPU (slow)")
    loaded = {l["model"] for h in hosts if h.get("reachable") for l in h["loaded"]}
    stats = get_queue().stats()
    pool = _pool
    return {
        "status": "ok" if not problems else "degraded",
        "problems": problems,
        "team_models": team,
        "team_models_resident": [m for m in team if m in loaded],
        "inference": {"hosts": hosts, "slots": inference.pool().snapshot(), "keep_alive": inference.KEEP_ALIVE},
        "queue": stats,
        "database": {"queue": backend_name(get_queue()),
                     "signals": ("postgresql + pgvector" if _signals_store_pgvector() else "fallback")},
        "workers": {"in_process": pool.alive() if pool else 0, "active_jobs": dict(pool.active) if pool else {},
                    "warmup": pool.warmup_report if pool else None},
    }


def _signals_store_pgvector() -> bool:
    from . import signals
    try:
        return signals.enabled() and signals.get_store().pgvector
    except Exception:
        return False


@router.post("/warmup")
def universal_warmup():
    """Load every team model on every host now (also done automatically when workers start)."""
    return {"loaded": inference.warm_up()}


@router.get("/info")
def universal_info():
    """Agents, model availability, and the deterministic coverage."""
    agents = build_agents()
    available: Optional[set] = set()
    for h in inference.residency():
        if not h["reachable"]:
            available = None
            break
    if available is not None:
        try:
            for url, client in inference.pool().clients().items():
                available |= {m.model for m in client.list().models}
        except Exception:
            available = None
    return {
        "engine": "universal",
        "engine_version": ENGINE_VERSION,
        "prompts_source": "universal_id_extraction_prompts.py",
        "agents": [
            {"order": i, "agent": name, "model": agents[name].model,
             "model_status": ("unknown — ollama unreachable" if available is None
                              else "ready" if agents[name].model in available else "not pulled"),
             "images_per_call": agents[name].max_images,
             "context_tokens": context_limit(agents[name].model) if available is not None else None,
             "output_fields": len(agents[name].output_fields)}
            for i, name in enumerate(AGENT_ORDER, start=1) if name in agents
        ],
        "deterministic_verification": {
            "mrz": "ICAO 9303 TD1/TD2/TD3 check digits (idx parsers) + local OCR re-read",
            "barcodes": "zxing-cpp: AAMVA PDF417, Ethiopian Fayda QR parsed; all others surfaced raw",
            "national_id_checksums": supported_schemes(),
            "calendars_converted": ["ethiopian", "hijri (tabular, ±1-2 days)", "persian", "thai buddhist",
                                    "minguo (ROC)", "japanese eras (令和 Reiwa, 平成 Heisei, 昭和 Showa, 大正 Taisho)"],
            "calendars_flagged_not_converted": ["hebrew", "nepali bikram sambat"],
            "ocr_grounding_engines": ["rapidocr"] + (["tesseract"] if _tesseract_languages() else []),
            "tesseract_languages": len(_tesseract_languages()),
            "privacy": "Aadhaar numbers masked to XXXX XXXX 1234 (UIDAI guidance)",
        },
        "serving": {
            "modes_allowed": ["full", "fast"] if _fast_allowed() else ["full"],
            "priorities": list(PRIORITIES),
            "exact_duplicate_cache_seconds": EXACT_CACHE_SECONDS,
        },
        "queue": get_queue().stats(),
    }

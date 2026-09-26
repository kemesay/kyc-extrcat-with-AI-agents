"""
HTTP routes for the universal engine — mounted by main.py under /universal.

Mirrors the existing /extract/* contract (async + poll, wait=true,
callback_url webhook, sync endpoint, file_back for two-sided documents) so
a client that already integrates with /extract/* can switch by changing the
path. Keeps its own job store and worker pool: nothing here touches the
existing endpoints' state.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import ollama
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from webhook import deliver_callback, is_valid_callback_url

from .grounding import _tesseract_languages
from .id_numbers import supported_schemes
from .pipeline import ENGINE_VERSION, extract_universal
from .prompts import AGENT_ORDER, build_agents
from .tokens import context_limit

router = APIRouter(prefix="/universal", tags=["universal"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DocType = Literal["auto", "passport", "national_id", "drivers_license", "residence_permit", "visa",
                  "refugee_travel_document", "military_id", "other"]
Mode = Literal["full", "fast"]
Passes = Literal["auto", "1", "2", "3"]
_PASSES_DOC = ("How many times each agent may be run. auto: the fewest passes that fit ALL relevant prompt "
               "knowledge into each small-context model's budget (moondream 2K, llava 4K); 1: a single "
               "maximally-packed call; 2/3: force that many field-group passes per agent.")

job_store: dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=2)


class UniversalJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    engine: str = "universal"
    result: Optional[dict] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _save_upload(file: UploadFile, job_id: str, suffix: str = "") -> str:
    ext = Path(file.filename or "").suffix.lower()
    if file.content_type not in ALLOWED_TYPES and ext not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type {file.content_type!r}. Use JPG/PNG/WEBP/BMP/TIFF.")
    path = UPLOAD_DIR / f"universal_{job_id}{suffix}{ext or '.jpg'}"
    path.write_bytes(await file.read())
    return str(path)


def _cleanup(*paths: Optional[str]) -> None:
    for p in paths:
        if p:
            Path(p).unlink(missing_ok=True)


def _run_job(job_id: str, front: str, back: Optional[str], options: dict, callback_url: Optional[str]) -> None:
    job_store[job_id]["status"] = "processing"
    try:
        result = extract_universal(front, back, trace_id=job_id, **options)
        job_store[job_id].update(status="completed", result=result, completed_at=_now())
    except Exception as e:
        job_store[job_id].update(status="failed", error=str(e), completed_at=_now())
    finally:
        _cleanup(front, back)
    if callback_url:
        payload = _response(job_id).model_dump()
        job_store[job_id]["callback_delivery"] = deliver_callback(callback_url, payload)


def _response(job_id: str) -> UniversalJobResponse:
    job = job_store[job_id]
    status = job["status"]
    if status == "completed":
        result = dict(job["result"])
        if job.get("callback_delivery"):
            result["callback_delivery"] = job["callback_delivery"]
        return UniversalJobResponse(job_id=job_id, status=status, message="Extraction complete", result=result)
    if status == "failed":
        return UniversalJobResponse(job_id=job_id, status=status, message=job.get("error", "Unknown error"))
    return UniversalJobResponse(job_id=job_id, status=status, message=f"Job is {status}...")


def _options(mode: str, doc_type: str, local_mrz_search: bool, ocr_grounding: bool,
             include_agent_outputs: bool, agent_passes: str = "auto") -> dict:
    return {"mode": mode, "doc_type_hint": doc_type, "local_mrz_search": local_mrz_search,
            "ocr_grounding": ocr_grounding, "include_agent_outputs": include_agent_outputs,
            "agent_passes": agent_passes}


_FILE_BACK_DOC = "Optional back side of a two-sided document (order doesn't matter)."
_MODE_DOC = ("full: 5-agent team (moondream → qwen2.5vl → llava → qwen2.5vl → qwen2.5 supervisor), "
             "~1.5-2 min. fast: single qwen2.5vl read, ~10-20 s. Both run the full deterministic "
             "verification layer (MRZ, barcodes, ID checksums, calendars, OCR grounding).")


@router.post("/extract", response_model=UniversalJobResponse)
async def universal_extract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_back: Optional[UploadFile] = File(default=None, description=_FILE_BACK_DOC),
    doc_type: DocType = Query("auto", description="A hint for the agents; they still classify the document."),
    mode: Mode = Query("full", description=_MODE_DOC),
    wait: bool = Query(False, description="Block until done and return the result (not with callback_url)."),
    callback_url: Optional[str] = Query(None, description="POST the finished result to this http(s) URL."),
    local_mrz_search: bool = Query(True, description="Re-read the MRZ with local OCR when the agents' copy fails its check digits."),
    ocr_grounding: bool = Query(True, description="Cross-check agent values against local OCR (hallucination / calendar / truncation guard)."),
    include_agent_outputs: bool = Query(False, description="Include each agent's raw JSON (debugging)."),
    agent_passes: Passes = Query("auto", description=_PASSES_DOC),
):
    """Universal ID extraction — any country, language, script or ID type."""
    if wait and callback_url:
        raise HTTPException(400, "wait=true and callback_url are mutually exclusive — pick one")
    if callback_url and not is_valid_callback_url(callback_url):
        raise HTTPException(400, "callback_url must start with http:// or https://")

    job_id = str(uuid.uuid4())
    front = await _save_upload(file, job_id)
    back = await _save_upload(file_back, job_id, "_back") if file_back is not None else None
    options = _options(mode, doc_type, local_mrz_search, ocr_grounding, include_agent_outputs, agent_passes)
    job_store[job_id] = {"job_id": job_id, "status": "queued", "filename": file.filename,
                         "options": options, "callback_url": callback_url, "created_at": _now()}

    if wait:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, _run_job, job_id, front, back, options, None)
        return _response(job_id)

    background_tasks.add_task(_run_job, job_id, front, back, options, callback_url)
    message = (f"Job queued. Result will be POSTed to {callback_url} when done." if callback_url
               else f"Job queued. Poll GET /universal/status/{job_id}")
    return UniversalJobResponse(job_id=job_id, status="queued", message=message)


@router.post("/extract/sync")
async def universal_extract_sync(
    file: UploadFile = File(...),
    file_back: Optional[UploadFile] = File(default=None, description=_FILE_BACK_DOC),
    doc_type: DocType = Query("auto"),
    mode: Mode = Query("full", description=_MODE_DOC),
    local_mrz_search: bool = Query(True),
    ocr_grounding: bool = Query(True),
    include_agent_outputs: bool = Query(False),
    agent_passes: Passes = Query("auto", description=_PASSES_DOC),
):
    """Synchronous universal extraction — returns the full result directly."""
    job_id = str(uuid.uuid4())
    front = await _save_upload(file, job_id)
    back = await _save_upload(file_back, job_id, "_back") if file_back is not None else None
    options = _options(mode, doc_type, local_mrz_search, ocr_grounding, include_agent_outputs, agent_passes)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            executor, lambda: extract_universal(front, back, trace_id=job_id, **options))
    finally:
        _cleanup(front, back)
    return {"job_id": job_id, "status": "completed", **result, "timestamp": _now()}


@router.get("/status/{job_id}", response_model=UniversalJobResponse)
def universal_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(404, f"Job {job_id} not found")
    return _response(job_id)


@router.delete("/job/{job_id}")
def universal_delete(job_id: str):
    if job_store.pop(job_id, None) is None:
        raise HTTPException(404, "Job not found")
    return {"message": f"Job {job_id} deleted"}


@router.get("/info")
def universal_info():
    """Agents, model availability, and the deterministic coverage."""
    agents = build_agents()
    try:
        available = {m.model for m in ollama.list().models}
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
        "active_jobs": sum(1 for j in job_store.values() if j["status"] in ("queued", "processing")),
    }

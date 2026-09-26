"""
ID Document Extraction API
===========================
FastAPI service wrapping idx.pipeline.extract_id_document() — a
multi-channel extraction pipeline ported from ID-Document-Extractor
(see /home/mesay/Desktop/ID-Document-Extractor), with Gemini replaced by
local Ollama vision models throughout (idx/vision_ollama.py). Channels,
tried in order, most-trustworthy first:

  1. Barcode/QR (PDF417 AAMVA, Ethiopian FAN/QR) — CRC/Reed-Solomon verified
  2. MRZ (ICAO 9303 TD1/TD2/TD3) — checksum-verified
  3. Local AI-vision crosscheck (Ollama) — enriches gaps a verified source left
  4. Ollama VLM universal fallback — full read when nothing structural matched
  5. Generic OCR label-alias sweep — the floor, never returns an empty profile

A higher tier's field is never overwritten by a lower one; disagreement
sets human_review_required rather than silently resolving. See idx/pipeline.py.

Confirmed Installed Models (ollama list):
  moondream:latest 1.7GB  — Fast Pre-Scanner (crewai_pipeline.py's own tier)
  qwen2.5vl:7b     6.0GB  — Field Extractor + Layout Analyzer (multilingual)
  llava:13b        8.0GB  — Accuracy Verifier

Machine: HP ZBook Firefly 15 G8
  RAM:       32GB DDR4
  GPU:       Intel Iris Xe (22.8GB VRAM via Vulkan)
  Storage:   1TB NVMe
  OS:        Linux Ubuntu
"""

import time
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import ollama
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from validators import sanity_check
from webhook import is_valid_callback_url, deliver_callback
from idx.pipeline import extract_id_document
from idx.schema import ExtractedIDDocument
from universal.api import router as universal_router

# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="ID Document Extraction API",
    description=(
        "Sequential collaborative AI pipeline for ID document extraction. "
        "Multilingual: English, Amharic, Arabic, and 100+ languages via qwen2.5vl."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Universal engine (any country/script/ID type) — additive, see universal/api.py.
app.include_router(universal_router)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

job_store: dict = {}
executor = ThreadPoolExecutor(max_workers=4)


# ─────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    mode_used: Optional[str] = None
    result: Optional[dict] = None


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

# validators.sanity_check() (this project's own hallucination/repeat-loop
# detector, complementary to idx/validation.py's plausibility checks) was
# written against a flat dict — this maps ExtractedIDDocument's FieldValue-
# wrapped fields onto that same flat shape so both validators can run.
def _flatten_for_sanity_check(doc: ExtractedIDDocument) -> dict:
    return {
        "full_name":         doc.full_name.value,
        "full_name_local":   doc.name_in_native_script.value,
        "id_number":         doc.document_number.value,
        "date_of_birth":     doc.date_of_birth.value,
        "expiry_date":       doc.date_of_expiry.value,
        "issue_date":        doc.date_of_issue.value,
        "nationality":       doc.nationality_name.value or doc.nationality_code.value,
        "gender":            doc.sex.value,
        "document_type":     doc.document_category,
        "issuing_authority": doc.issuing_authority.value,
        "place_of_birth":    doc.place_of_birth.value,
        "address":           doc.address.value,
        "mrz_code":          "\n".join(doc.mrz_raw),
        "language_detected": doc.name_script,
    }


def run_pipeline(image_path: str, image_path_back: Optional[str] = None, doc_type: str = "auto") -> dict:
    """Runs idx.pipeline.extract_id_document() and attaches this project's
    own sanity_check() on top of idx/validation.py's plausibility checks."""
    print(f"\n{'='*60}")
    print(f"🔎 Multi-channel extraction pipeline (doc_type={doc_type})")
    print("   barcode/QR → MRZ (checksum) → Ollama crosscheck → Ollama universal fallback → generic OCR floor")
    print(f"{'='*60}")

    start = time.time()
    doc = extract_id_document(image_path, doc_type=doc_type, image_path_back=image_path_back)
    duration = round(time.time() - start, 2)

    extracted_data = _flatten_for_sanity_check(doc)
    return {
        "document":               doc.model_dump(),
        "extracted_data":         extracted_data,
        "document_type":          doc.document_category,
        "extraction_method":      doc.extraction_method,
        "confidence":             doc.confidence,
        "coverage_tier":          doc.coverage_tier,
        "human_review_required":  doc.human_review_required,
        "warnings":               doc.warnings,
        "total_duration_seconds": duration,
        "pipeline":               "idx_pipeline",
        "validation":             sanity_check(extracted_data),
    }


# ─────────────────────────────────────────────────────────────
# Background Job
# ─────────────────────────────────────────────────────────────

def process_document_job(
    job_id: str, image_path: str, callback_url: Optional[str] = None,
    image_path_back: Optional[str] = None, doc_type: str = "auto",
):
    job_store[job_id]["status"] = "processing"
    try:
        result = run_pipeline(image_path, image_path_back=image_path_back, doc_type=doc_type)
        job_store[job_id].update({"status": "completed", **result})
        print(f"\n✅ Job {job_id} done | pipeline={result['pipeline']}")
    except Exception as e:
        print(f"\n❌ Job {job_id} failed: {e}")
        job_store[job_id].update({"status": "failed", "error": str(e)})
    finally:
        Path(image_path).unlink(missing_ok=True)
        if image_path_back:
            Path(image_path_back).unlink(missing_ok=True)

    if callback_url:
        payload = _job_status_response(job_id, job_store[job_id]).model_dump()
        print(f"  📮 delivering callback for job {job_id} → {callback_url}")
        delivery = deliver_callback(callback_url, payload)
        job_store[job_id]["callback_delivery"] = delivery
        if delivery["delivered"]:
            print(f"  ✅ callback delivered for job {job_id} (attempt {delivery['attempts']})")
        else:
            print(f"  ❌ callback delivery FAILED for job {job_id} after {delivery['attempts']} attempts: {delivery['last_error']}")


# ─────────────────────────────────────────────────────────────
# File Upload Helper
# ─────────────────────────────────────────────────────────────

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

async def save_upload(file: UploadFile, job_id: Optional[str] = None, suffix: str = "") -> tuple[str, str]:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Invalid type: {file.content_type}. Use JPG/PNG/WEBP/BMP")
    job_id = job_id or str(uuid.uuid4())
    path   = UPLOAD_DIR / f"{job_id}{suffix}{Path(file.filename).suffix or '.jpg'}"
    path.write_bytes(await file.read())
    return job_id, str(path)


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "ID Document Extraction API",
        "version": "5.0.0",
        "machine": "HP ZBook Firefly 15 G8 | 32GB RAM | Intel Iris Xe 22.8GB VRAM",
        "architecture": (
            "Multi-channel pipeline ported from ID-Document-Extractor, Gemini replaced by local "
            "Ollama vision models — see idx/pipeline.py. Channels tried in order, most-trustworthy "
            "first: barcode/QR (PDF417 AAMVA, Ethiopian FAN/QR, CRC-verified) -> MRZ (ICAO 9303 "
            "TD1/TD2/TD3, checksum-verified) -> Ollama AI-vision crosscheck (enriches gaps) -> "
            "Ollama AI-vision universal fallback -> generic OCR label-alias sweep (the floor, "
            "never returns an empty profile)."
        ),
        "document_types_supported": ["passport", "national_id", "drivers_license", "residence_permit", "auto"],
        "ollama_models": {
            "moondream:latest": "1.7GB — Fast pre-scanner (crewai_pipeline.py's own tier)",
            "qwen2.5vl:7b":     "6.0GB — Field extractor + layout analyzer (multilingual)",
            "llava:13b":        "8.0GB — Accuracy verifier",
        },
        "local_ocr_engine": "RapidOCR (ONNX, CPU-only — never competes with Ollama for GPU) + tesseract fallback",
        "languages_supported": [
            "English", "Amharic (አማርኛ)", "Arabic (عربي)",
            "Chinese", "French", "Spanish", "German",
            "Italian", "Japanese", "100+ more via the Ollama VLM tier"
        ],
        "endpoints": {
            "POST /extract/id":              "Async extraction. Default: poll GET /extract/status/{job_id}. "
                                              "wait=true: blocks for the result (not recommended — jobs can take "
                                              "10+ min). callback_url=<url>: server POSTs the result when done "
                                              "(recommended for production). doc_type: passport|national_id|"
                                              "drivers_license|residence_permit|auto (default). file_back: "
                                              "optional second image for two-sided documents.",
            "POST /extract/id/sync":         "Sync extraction — same as wait=true",
            "GET  /extract/status/{job_id}": "Poll job status",
            "GET  /health":                  "Health + model status",
            "GET  /agents":                  "Pipeline channel details",
        },
    }


@app.get("/health")
def health_check():
    required = ["moondream:latest", "qwen2.5vl:7b", "llava:13b"]
    try:
        available = [m.model for m in ollama.list().models]
        models    = {m: ("✅ ready" if m in available else "❌ not pulled") for m in required}
    except Exception:
        models    = {m: "⚠️ ollama unreachable" for m in required}

    all_ready = all("ready" in s for s in models.values())
    return {
        "status":       "healthy" if all_ready else "degraded",
        "ollama_models": models,
        "local_ocr":    "RapidOCR + tesseract (CPU, no Ollama/GPU dependency)",
        "gpu":          "Intel Iris Xe — Vulkan",
        "vram_total":   "22.8 GiB",
        "active_jobs":  sum(1 for j in job_store.values() if j["status"] == "processing"),
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.get("/agents")
def list_agents():
    return {
        "channels": [
            {
                "tier":  1,
                "name":  "barcode",
                "what":  "PDF417 (AAMVA driver's licenses/ID cards), Ethiopian Digital ID FAN (Code128) + QR",
                "trust": "CRC/Reed-Solomon error-corrected by the barcode symbology itself — fully trusted on decode",
            },
            {
                "tier":  2,
                "name":  "mrz",
                "what":  "ICAO 9303 TD1 (3x30, national ID)/TD2 (2x36)/TD3 (2x44, passport)",
                "trust": "ICAO check-digit algorithm — a candidate failing checksum is never trusted as-is",
            },
            {
                "tier":  3,
                "name":  "label_anchor",
                "what":  "Field-first per-token OCR sweep, multi-language alias ontology (idx/ontology.py)",
                "trust": "Spatially grounded — the field's own printed label was actually found nearby",
            },
            {
                "tier":  4,
                "name":  "vlm / vlm_crosscheck",
                "what":  "Ollama qwen2.5vl:7b (crosscheck/single-shot) or the full 4-stage sequential "
                         "pipeline (moondream -> qwen2.5vl:7b -> llava:13b -> qwen2.5vl:7b) for the "
                         "universal fallback — see crewai_pipeline.py",
                "trust": "Flexible, unverifiable — never outranks a checksum/barcode-verified field",
            },
            {
                "tier":  5,
                "name":  "signature_sweep",
                "what":  "Unlabeled value-shape regex match (idx/signatures.py) — last resort",
                "trust": "Weakest — a value that merely looks the right shape, no label found at all",
            },
        ],
    }


@app.post("/extract/id", response_model=JobStatusResponse)
async def extract_id_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_back: Optional[UploadFile] = File(
        default=None,
        description="Optional second image (back of card) — used by AAMVA PDF417 (usually on the "
                    "back) and the Ethiopian Digital ID (QR on back, FAN barcode on front); order "
                    "doesn't matter, both sides are classified by content.",
    ),
    doc_type: Literal["passport", "national_id", "drivers_license", "residence_permit", "auto"] = Query(
        default="auto",
        description="auto tries passport, then national_id, then drivers_license in order. Pick "
                    "residence_permit explicitly — its MRZ format can't be distinguished from national_id's.",
    ),
    wait: bool = Query(
        default=False,
        description=(
            "If true, blocks and returns the completed result directly instead of "
            "returning {status: queued} immediately. Equivalent to POST /extract/id/sync "
            "but keeps this endpoint's job_id semantics. Default false: fire-and-poll. "
            "Cannot be combined with callback_url."
        ),
    ),
    callback_url: Optional[str] = Query(
        default=None,
        description=(
            "If set, returns job_id immediately (like the default) but the server "
            "POSTs the full completed result to this URL when the job finishes, "
            "instead of you polling GET /extract/status/{job_id}. Recommended for "
            "production integrations — extractions on this hardware can take "
            "several minutes, too long to hold a synchronous connection open "
            "reliably behind most client/proxy timeouts. Must be http:// or https://."
        ),
    ),
):
    """
    Upload an ID document image (+ optional back image).
    Default (wait=false, no callback_url): returns job_id immediately — poll
      GET /extract/status/{job_id}.
    wait=true: blocks until the job completes/fails and returns the full result directly.
    callback_url=<url>: returns job_id immediately; server POSTs the full result to
      that URL when done (production-recommended — see callback_url description).

    Supported documents: National ID, Passport, Driver's License, Residence Permit
    Supported languages: English, Amharic, Arabic, and 100+ more via the AI-vision tier
    Supported formats:   JPG, PNG, WEBP, BMP
    """
    if wait and callback_url:
        raise HTTPException(400, "wait=true and callback_url are mutually exclusive — pick one")
    if callback_url and not is_valid_callback_url(callback_url):
        raise HTTPException(400, "callback_url must start with http:// or https://")

    job_id, image_path = await save_upload(file)
    image_path_back = None
    if file_back is not None:
        _, image_path_back = await save_upload(file_back, job_id=job_id, suffix="_back")

    job_store[job_id] = {
        "job_id":      job_id,
        "status":      "queued",
        "filename":    file.filename,
        "doc_type":    doc_type,
        "callback_url": callback_url,
        "created_at":  datetime.utcnow().isoformat(),
    }

    if wait:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor, process_document_job, job_id, image_path, None, image_path_back, doc_type,
        )
        return _job_status_response(job_id, job_store[job_id])

    background_tasks.add_task(process_document_job, job_id, image_path, callback_url, image_path_back, doc_type)

    message = (
        f"Job queued. Result will be POSTed to {callback_url} when done."
        if callback_url else
        f"Job queued. Poll GET /extract/status/{job_id}"
    )
    return JobStatusResponse(
        job_id=job_id,
        status="queued",
        mode_used="idx_pipeline",
        message=message,
    )


def _job_status_response(job_id: str, job: dict) -> JobStatusResponse:
    status = job["status"]

    if status == "completed":
        return JobStatusResponse(
            job_id=job_id,
            status="completed",
            mode_used=job.get("pipeline"),
            message="Extraction complete",
            result={
                "document":               job.get("document"),
                "extracted_data":         job.get("extracted_data"),
                "document_type":          job.get("document_type"),
                "extraction_method":      job.get("extraction_method"),
                "confidence":             job.get("confidence"),
                "coverage_tier":          job.get("coverage_tier"),
                "human_review_required":  job.get("human_review_required"),
                "warnings":               job.get("warnings"),
                "total_duration_seconds": job.get("total_duration_seconds"),
                "pipeline":               job.get("pipeline"),
                "validation":             job.get("validation"),
                "callback_delivery":      job.get("callback_delivery"),
                "timestamp":              job.get("created_at"),
            },
        )
    elif status == "failed":
        return JobStatusResponse(
            job_id=job_id, status="failed",
            message=job.get("error", "Unknown error"),
        )
    else:
        return JobStatusResponse(
            job_id=job_id, status=status,
            message=f"Job is {status}...",
        )


@app.get("/extract/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Poll extraction job — returns result when complete"""
    if job_id not in job_store:
        raise HTTPException(404, f"Job {job_id} not found")
    return _job_status_response(job_id, job_store[job_id])


@app.post("/extract/id/sync")
async def extract_id_sync(
    file: UploadFile = File(...),
    file_back: Optional[UploadFile] = File(default=None),
    doc_type: Literal["passport", "national_id", "drivers_license", "residence_permit", "auto"] = Query(
        default="auto",
    ),
):
    """Synchronous extraction — waits and returns full result."""
    job_id, image_path = await save_upload(file)
    image_path_back = None
    if file_back is not None:
        _, image_path_back = await save_upload(file_back, job_id=job_id, suffix="_back")
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_pipeline, image_path, image_path_back, doc_type)

        return {
            "job_id":    job_id,
            "status":    "completed",
            "mode_used": result.get("pipeline"),
            **result,
            "timestamp": datetime.utcnow().isoformat(),
        }
    finally:
        Path(image_path).unlink(missing_ok=True)
        if image_path_back:
            Path(image_path_back).unlink(missing_ok=True)


@app.delete("/extract/job/{job_id}")
def delete_job(job_id: str):
    if job_id not in job_store:
        raise HTTPException(404, "Job not found")
    del job_store[job_id]
    return {"message": f"Job {job_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

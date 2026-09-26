"""
"CrewAI" Pipeline for ID Document Extraction
============================================
4 collaborative agents — each sees and corrects the previous agent's work.

Despite the module name (kept for backward compatibility with main.py's
`from crewai_pipeline import extract_with_crewai`), this is a direct
sequential Ollama pipeline, NOT the CrewAI framework. CrewAI's Agent/Task
tool-calling loop requires the underlying model to support native
function-calling in Ollama's chat template — moondream and qwen2.5vl don't,
so every tool invocation failed with "does not support tools" and every
stage silently returned an empty JSON template. Calling each vision model
directly with the image bytes and the previous stage's JSON as text context
achieves the same "each agent sees and corrects the previous one" design
without depending on tool-calling support.

Confirmed Installed Models:
  moondream:latest  1.7GB  Agent 1 — Fast Pre-Scanner
  qwen2.5vl:7b      6.0GB  Agent 2 — Field Extractor (multilingual: Amharic, Arabic)
  llava:13b         8.0GB  Agent 3 — Accuracy Verifier
  qwen2.5vl:7b      6.0GB  Agent 4 — Layout Analyzer (same model, different role/prompt)

Flow:
  moondream → qwen2.5vl:7b → llava:13b → qwen2.5vl:7b → Final JSON
  Each stage SEES the previous stage's JSON (as text) alongside the image,
  corrects errors, fills gaps.

Machine: HP ZBook Firefly 15 G8
  Intel Iris Xe GPU: 22.8GB VRAM (Vulkan)
  Expected speed: ~45s total
"""

import base64
import json
import time

from ollama_client import vision_chat

# ─────────────────────────────────────────────────────────────
# Confirmed Models
# ─────────────────────────────────────────────────────────────

MODELS = {
    "fast_scanner":      "moondream:latest",  # 1.7GB — Agent 1
    "field_extractor":   "qwen2.5vl:7b",      # 6.0GB — Agent 2 (multilingual)
    "accuracy_verifier": "llava:13b",          # 8.0GB — Agent 3
    "layout_analyzer":   "qwen2.5vl:7b",      # 6.0GB — Agent 4 (same as field_extractor)
}

EMPTY_RESULT = {
    "full_name":         "",
    "full_name_local":   "",   # Amharic / Arabic / other script name
    "id_number":         "",
    "date_of_birth":     "",
    "expiry_date":       "",
    "issue_date":        "",
    "nationality":       "",
    "gender":            "",
    "document_type":     "",
    "issuing_authority": "",
    "place_of_birth":    "",
    "address":           "",
    "mrz_code":          "",
    "language_detected": "",
}

JSON_SCHEMA = json.dumps(EMPTY_RESULT, indent=2)

# ─────────────────────────────────────────────────────────────
# Per-stage instructions — each embeds the previous stage's JSON
# ─────────────────────────────────────────────────────────────

STAGES = [
    {
        "name":  "fast_scanner",
        "role":  "Fast Document Pre-Scanner",
        "model": MODELS["fast_scanner"],
        "instruction": f"""You are a fast document scanning agent — rapid but not deep.
Quickly scan this ID document.
1. Identify document type (passport/national ID/driver license/residence permit)
2. Identify primary language(s) on the document
3. Extract the most clearly visible fields
4. Mark uncertain fields with a '?' suffix
Return ONLY this JSON — no explanation, no markdown:
{JSON_SCHEMA}""",
    },
    {
        "name":  "field_extractor",
        "role":  "Multilingual Structured Field Extractor",
        "model": MODELS["field_extractor"],
        "instruction": f"""You are a multilingual document extraction expert trained on
identity documents from every country. You read Amharic (አማርኛ), Arabic (عربي),
English, Chinese, and 100+ languages natively.
Extract ALL fields from this ID document.
Read every script on the document: Latin, Ethiopic, Arabic, or other.
Extract full_name in the primary script AND full_name_local if bilingual.
For Ethiopian IDs: extract the Amharic name as full_name_local.
For Arabic IDs: extract the Arabic name as full_name_local.
You will be given the previous agent's draft JSON below — correct its errors,
fix any '?' fields, and fill in anything missing. Preserve non-Latin scripts
exactly, do not transliterate. Never replace a non-empty value with an empty one.
Return ONLY the corrected JSON — no explanation, no markdown:
{JSON_SCHEMA}""",
    },
    {
        "name":  "accuracy_verifier",
        "role":  "High Accuracy Verification Specialist",
        "model": MODELS["accuracy_verifier"],
        "instruction": f"""You are a forensic document examiner — the most precise reader on the team.
Carefully verify every extracted value against this document image, character by character.
Check: Latin 0↔O, 1↔I↔l, 5↔S, 8↔B, rn↔m; similar-looking Arabic letter forms;
similar Ethiopic (Amharic) characters. Validate MRZ line format and check digits if present.
You will be given the previous agent's JSON below — correct any OCR errors you find,
but never replace a non-empty value with an empty one.
Return ONLY the corrected JSON — no explanation, no markdown:
{JSON_SCHEMA}""",
    },
    {
        "name":  "layout_analyzer",
        "role":  "Document Layout and Zone Analyzer",
        "model": MODELS["layout_analyzer"],
        "instruction": f"""You are a document layout expert. You see ID documents as
structured visual grids. Ethiopian IDs have Amharic on one side and English on
the other; Arabic passports read right-to-left. You know these patterns.
Analyze this document zone by zone: HEADER (country, title, emblem), PHOTO ZONE
(text near photo, security features), DATA ZONE (all labeled fields), BOTTOM
STRIP (MRZ lines, barcodes, serial numbers), STAMPS (issuing authority), FINE
PRINT (small text, watermarks, margins).
You will be given the previous agent's verified JSON below — find and fill ANY
field still missing or empty from any zone. Do not overwrite already-filled
values. This is the FINAL DEFINITIVE result — leave nothing empty if it is
visible anywhere on the document.
Return ONLY the final JSON — no explanation, no markdown:
{JSON_SCHEMA}""",
    },
]


# ─────────────────────────────────────────────────────────────
# JSON Parser
# ─────────────────────────────────────────────────────────────

def parse_json(raw: str) -> dict:
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


def merge_stage(previous: dict, new: dict) -> dict:
    """Fill/correct from `new`, but never replace a non-empty value with an empty one."""
    merged = dict(previous)
    for field in EMPTY_RESULT:
        value = new.get(field, "")
        if isinstance(value, str):
            value = value.strip().rstrip("?")
        if value:
            merged[field] = value
    return merged


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_stage(model: str, role: str, instruction: str, image_b64: str, previous: dict) -> dict:
    """Run one vision stage: model reads the image plus the previous stage's JSON as text."""
    prompt = f"{instruction}\n\nPrevious agent's draft JSON:\n{json.dumps(previous, indent=2)}"
    try:
        response = vision_chat(
            model=model,
            messages=[{
                "role":    "user",
                "content": prompt,
                "images":  [image_b64],
            }],
        )
        return parse_json(response["message"]["content"])
    except Exception as e:
        print(f"  ❌ [CREWAI] {role} ({model}) failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def extract_with_crewai(image_path: str) -> dict:
    start = time.time()
    print(f"\n{'='*60}")
    print("🤝 Sequential Collaborative Pipeline")
    print(f"   {MODELS['fast_scanner']} → {MODELS['field_extractor']} → "
          f"{MODELS['accuracy_verifier']} → {MODELS['layout_analyzer']}")
    print("   Each agent sees and corrects the previous agent's work")
    print(f"{'='*60}\n")

    image_b64  = encode_image(image_path)
    accumulated = dict(EMPTY_RESULT)
    agents_used = []

    for order, stage in enumerate(STAGES, start=1):
        stage_start = time.time()
        print(f"  🤖 [{order}/4] {stage['role']} ({stage['model']}) starting...")
        result = run_stage(
            model=stage["model"],
            role=stage["role"],
            instruction=stage["instruction"],
            image_b64=image_b64,
            previous=accumulated,
        )
        accumulated = merge_stage(accumulated, result)
        duration = round(time.time() - stage_start, 2)
        print(f"  ✅ [{order}/4] {stage['role']} done in {duration}s")
        agents_used.append({
            "order":    order,
            "agent":    stage["name"],
            "model":    stage["model"],
            "duration_seconds": duration,
            "status":   "success" if result else "failed",
        })

    total_time = round(time.time() - start, 2)
    print(f"\n✅ Sequential pipeline complete in {total_time}s")

    return {
        "extracted_data":         accumulated,
        "total_duration_seconds": total_time,
        "pipeline":               "crewai_sequential",
        "agents_used":            agents_used,
    }

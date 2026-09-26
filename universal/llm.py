"""
Ollama calls for the universal engine.

Shares the process-wide ollama_semaphore (ollama_concurrency.py) and the
Vulkan/iGPU failure detection from ollama_client.py, so this engine and the
existing idx pipeline never oversubscribe the GPU between them. Differs from
ollama_client.vision_chat() in three ways that matter for a 37-field schema:

  - Structured output: every call passes a JSON schema as Ollama's `format`,
    so the model is grammar-constrained to emit exactly the expected keys —
    no markdown fences, no prose, no missing fields to repair afterwards.
  - Explicit num_ctx / num_predict per model, so long prompts aren't silently
    truncated and a degenerate repetition loop can't generate forever.
  - Images are EXIF-rotated and downscaled once before encoding: phone photos
    arrive sideways and at 12MP, which wastes thousands of vision tokens.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Iterable, Optional

import ollama
from PIL import Image, ImageOps

from ollama_concurrency import ollama_semaphore
from validators import has_internal_repetition

MAX_IMAGE_SIDE = 1600
# A request that outlives this is a stalled server (e.g. Ollama restarting),
# not a slow model — fail fast rather than hang the job for minutes.
REQUEST_TIMEOUT = float(os.environ.get("UNIVERSAL_OLLAMA_TIMEOUT", "240"))
_client = ollama.Client(timeout=REQUEST_TIMEOUT)
NUM_PREDICT = 1536        # a full 37-field reply is ~600-900 tokens; more means a loop
TEMPERATURE = 0.1
REPEAT_PENALTY = 1.15

_PAIR_RE = re.compile(r'"([A-Za-z0-9_]+)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def encode_image(path: str, max_side: int = MAX_IMAGE_SIDE) -> str:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def json_schema(fields: Iterable[str]) -> dict[str, Any]:
    fields = list(fields)
    return {
        "type": "object",
        "properties": {f: {"type": "string"} for f in fields},
        "required": fields,
    }


def parse_json(raw: str) -> tuple[dict[str, Any], bool]:
    """-> (data, complete). When a reply is cut off mid-object (a repetition
    loop hitting num_predict), the complete "key": "value" pairs before the
    cut are salvaged instead of discarding the whole stage."""
    text = raw.replace("```json", "").replace("```", "").strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, True
        except Exception:
            continue
    salvaged = {}
    for key, value in _PAIR_RE.findall(text):
        try:
            salvaged[key] = json.loads(f'"{value}"')
        except Exception:
            salvaged[key] = value
    return salvaged, False


def clean_unicode(text: str) -> str:
    """Models can emit lone UTF-16 surrogate escapes ('\\udcfa') that are valid
    JSON but not encodable text — they crashed a JSON response. Valid pairs
    are rejoined; lone halves become U+FFFD."""
    if not any("\ud800" <= c <= "\udfff" for c in text):
        return text
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_unicode(value).strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(v) for v in value if _as_text(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _evaluated(response: Any) -> bool:
    return bool(response.get("prompt_eval_count")) and bool((response["message"]["content"] or "").strip())


DEVICE_FAILURE_MARKERS = ("errordevicelost", "device lost", "model runner has unexpectedly stopped")


def _chat(model: str, messages: list, fmt: Any, options: dict, keep_alive: Any = None) -> Any:
    extra = {"keep_alive": keep_alive} if keep_alive is not None else {}
    try:
        return _client.chat(model=model, messages=messages, format=fmt, options=options, **extra)
    except ollama.ResponseError as e:
        # An Ollama server too old for JSON-schema `format` still honours "json".
        if isinstance(fmt, dict) and "format" in str(e).lower():
            return _client.chat(model=model, messages=messages, format="json", options=options, **extra)
        raise


def chat_json(
    model: str,
    system: str,
    user: str,
    *,
    images: Optional[list[str]] = None,
    fields: Iterable[str] = (),
    num_ctx: int = 8192,
    num_predict: int = NUM_PREDICT,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Returns (fields -> string values, call metadata). Raises on a failed
    call; an unparseable reply returns ({}, meta) rather than raising."""
    fields = list(fields)
    user_msg: dict[str, Any] = {"role": "user", "content": user}
    if images:
        user_msg["images"] = images
    messages = ([{"role": "system", "content": system}] if system else []) + [user_msg]
    options = {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": TEMPERATURE,
               "repeat_penalty": REPEAT_PENALTY}
    fmt: Any = json_schema(fields) if fields else "json"

    retried_on_cpu = False
    resampled = False
    with ollama_semaphore:
        try:
            response = _chat(model, messages, fmt, options)
        except Exception as e:
            text = str(e).lower()
            if "token repeat limit" in text:
                # A degenerate repetition loop, not a device fault: resample on the
                # GPU with a stronger penalty. (Retrying on CPU — right for the old
                # Vulkan iGPU — left the model CPU-loaded on Apple Silicon and made
                # every later call ~10x slower.)
                print(f"  ⚠️  {model}: repetition loop — resampling with a stronger repeat penalty")
                resampled = True
                response = _chat(model, messages, fmt, {**options, "repeat_penalty": 1.35, "temperature": 0.4})
            elif any(m in text for m in DEVICE_FAILURE_MARKERS):
                print(f"  ⚠️  {model}: GPU device failure ({e}) — retrying once on CPU")
                retried_on_cpu = True
                # keep_alive=0 unloads the CPU-only runner right after, so the next
                # call reloads the model onto the GPU instead of reusing it.
                response = _chat(model, messages, fmt, {**options, "num_gpu": 0}, keep_alive=0)
            else:
                raise

    if not _evaluated(response):
        # Seen right after an Ollama auto-update: HTTP 200 with no prompt
        # evaluated and no content. Retry once, then fail loudly — an empty
        # reply must never pass as "this document has no data".
        time.sleep(2)
        with ollama_semaphore:
            response = _chat(model, messages, fmt, options)
        if not _evaluated(response):
            raise RuntimeError(f"{model}: Ollama returned an empty response (no prompt evaluated)")

    parsed, complete = parse_json(response["message"]["content"])
    data = {k: _as_text(v) for k, v in parsed.items()}
    looping = [k for k, v in data.items() if has_internal_repetition(v)]
    for k in looping:
        data[k] = ""
    if fields:
        data = {f: data.get(f, "") for f in fields}
    meta = {
        "prompt_tokens": response.get("prompt_eval_count"),
        "output_tokens": response.get("eval_count"),
        "retried_on_cpu": retried_on_cpu,
        "resampled": resampled,
        "parsed": bool(parsed),
        "complete_json": complete,
    }
    if looping:
        meta["dropped_repetition_loops"] = looping
    return data, meta

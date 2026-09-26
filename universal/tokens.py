"""
Exact token accounting against the models Ollama actually serves.

Ollama has no tokenize endpoint, but a prompt run with num_predict=1 reports
prompt_eval_count exactly (and stably — the prompt cache doesn't change the
reported count). That gives:

  - context_limit(model)  the model's trained context (Ollama's modelinfo),
                          capped at MAX_CONTEXT for large-context models
  - image_tokens(model)   probed once per model: fixed-resolution encoders
                          cost a constant (llava 577, moondream 729)
  - supports_system(model) whether the chat template carries a system prompt
  - estimate(model, text) cheap bytes-based estimate, calibrated per model
  - exact_chat_tokens()   exact text tokens of a system+user message pair,
                          used only when an estimate is near the limit

Every number is cached for the process lifetime.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import uuid
from functools import lru_cache

import ollama
from PIL import Image

logger = logging.getLogger(__name__)

# Memory cap for large-context models (qwen2.5vl trains at 128K). The largest
# prompt here is ~5K tokens + ~1K answer; at 16K qwen2.5vl grew to 7.8 GB and
# the four models stopped fitting in Metal's 17.8 GB, forcing evict/reload
# cycles between agents.
MAX_CONTEXT = 8192
_FALLBACK_BYTES_PER_TOKEN = 3.0


@lru_cache(maxsize=None)
def supports_system(model: str) -> bool:
    """Whether the model's chat template has a system slot. moondream's is
    'Question: {{ .Prompt }} Answer:' — a system message is silently dropped,
    so its instructions must travel inside the user message."""
    try:
        return ".System" in (ollama.show(model).template or "")
    except Exception:
        return True


@lru_cache(maxsize=None)
def context_limit(model: str) -> int:
    try:
        info = ollama.show(model).modelinfo or {}
        trained = next((int(v) for k, v in info.items() if k.endswith(".context_length")), 0)
    except Exception as e:
        logger.warning("context length unavailable for %s: %s", model, e)
        trained = 0
    return min(trained or 4096, MAX_CONTEXT)


def _count_raw(model: str, text: str) -> int:
    # A nonce keeps the count from depending on any cached prefix.
    nonce = uuid.uuid4().hex[:6]
    opts = {"num_predict": 1, "num_ctx": context_limit(model)}   # same num_ctx as real calls: no reload
    with_text = ollama.generate(model=model, prompt=f"{nonce}\n{text}", raw=True, options=opts).prompt_eval_count
    alone = ollama.generate(model=model, prompt=nonce, raw=True, options=opts).prompt_eval_count
    return max(0, with_text - alone)


@lru_cache(maxsize=None)
def bytes_per_token(model: str) -> float:
    """Calibrated on the prompts file's own mixed-script system prompt."""
    import universal_id_extraction_prompts as P
    sample = P.SYSTEM_PROMPT
    try:
        tokens = _count_raw(model, sample)
        return len(sample.encode("utf-8")) / max(tokens, 1)
    except Exception as e:
        logger.warning("token calibration failed for %s: %s", model, e)
        return _FALLBACK_BYTES_PER_TOKEN


def estimate(model: str, text: str) -> int:
    return math.ceil(len(text.encode("utf-8")) / bytes_per_token(model)) if text else 0


@lru_cache(maxsize=None)
def image_tokens(model: str) -> int:
    """Tokens one image costs this model. Fixed-resolution encoders (llava,
    moondream) cost a constant, measured once with a probe image."""
    buf = io.BytesIO()
    Image.new("RGB", (640, 400), "white").save(buf, "JPEG")
    probe = base64.b64encode(buf.getvalue()).decode()
    opts = {"num_predict": 1, "num_ctx": context_limit(model)}
    try:
        a = ollama.chat(model=model, messages=[{"role": "user", "content": uuid.uuid4().hex[:6], "images": [probe]}],
                        options=opts).prompt_eval_count
        b = ollama.chat(model=model, messages=[{"role": "user", "content": uuid.uuid4().hex[:6]}],
                        options=opts).prompt_eval_count
        return max(0, a - b)
    except Exception as e:
        logger.warning("image token probe failed for %s: %s", model, e)
        return 1024


def image_tokens_for(model: str, width: int, height: int) -> int:
    """Dynamic-resolution models (qwen2.5-VL: one token per 28x28 patch) scale
    with the image; fixed-resolution ones use the probed constant."""
    if model.split(":", 1)[0].lower().startswith("qwen2.5vl"):
        return math.ceil(width / 28) * math.ceil(height / 28) + 2
    return image_tokens(model)


@lru_cache(maxsize=512)
def exact_chat_tokens(model: str, system: str, user: str, num_ctx: int) -> int:
    """Exact text tokens of a system+user pair through the model's chat template."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return ollama.chat(model=model, messages=messages,
                       options={"num_predict": 1, "num_ctx": num_ctx}).prompt_eval_count


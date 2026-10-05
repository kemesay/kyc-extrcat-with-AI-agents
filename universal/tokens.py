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

Every number is cached for the process lifetime. Probes run through
universal.inference like real calls: on a host that serves the model, one
call at a time, carrying keep_alive (a probe without it would reset the
model's keep-alive to Ollama's 5-minute default).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
import threading
import uuid
from typing import Optional
from functools import lru_cache

from PIL import Image

from . import inference

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
        return ".System" in (inference.show(model).template or "")
    except Exception:
        return True


@lru_cache(maxsize=None)
def context_limit(model: str) -> int:
    try:
        info = inference.show(model).modelinfo or {}
        trained = next((int(v) for k, v in info.items() if k.endswith(".context_length")), 0)
    except Exception as e:
        logger.warning("context length unavailable for %s: %s", model, e)
        trained = 0
    return min(trained or 4096, MAX_CONTEXT)


def _count_raw(model: str, text: str) -> int:
    # A nonce keeps the count from depending on any cached prefix.
    nonce = uuid.uuid4().hex[:6]
    opts = {"num_predict": 1, "num_ctx": context_limit(model)}   # same num_ctx as real calls: no reload
    with_text = inference.call(model, "generate", prompt=f"{nonce}\n{text}", raw=True, options=opts).prompt_eval_count
    alone = inference.call(model, "generate", prompt=nonce, raw=True, options=opts).prompt_eval_count
    return max(0, with_text - alone)


_OVERFLOW = re.compile(r'"n_prompt_tokens"\s*:\s*(\d+)')
_PLAUSIBLE_BYTES_PER_TOKEN = (1.0, 6.0)


def overflow_tokens(error: BaseException) -> Optional[int]:
    """The true prompt size from Ollama's 'request (N tokens) exceeds the
    available context size' error, or None for any other error."""
    m = _OVERFLOW.search(str(error))
    return int(m.group(1)) if m and "exceed" in str(error) else None


def _calibration_sample(model: str) -> str:
    """The prompts file's own mixed-script system prompt, cut to fit well inside
    the model's context: a sample longer than the window is truncated by Ollama
    and would measure too FEW tokens (moondream's 2,048 measured 7.1 bytes/token,
    halving every estimate and overflowing its calls)."""
    import universal_id_extraction_prompts as P
    budget = context_limit(model) * 2                   # bytes; ~half the window at the densest ratio seen
    out, used = [], 0
    for line in P.SYSTEM_PROMPT.splitlines(keepends=True):
        size = len(line.encode("utf-8"))
        if used + size > budget:
            break
        out.append(line)
        used += size
    return "".join(out)


@lru_cache(maxsize=None)
def bytes_per_token(model: str) -> float:
    """Calibrated on a sample that fits the model's context; an implausible
    measurement falls back to a conservative ratio rather than being trusted."""
    sample = _calibration_sample(model)
    try:
        tokens = _count_raw(model, sample)
        ratio = len(sample.encode("utf-8")) / max(tokens, 1)
    except Exception as e:
        n = overflow_tokens(e)
        if not n:
            logger.warning("token calibration failed for %s: %s", model, e)
            return _FALLBACK_BYTES_PER_TOKEN
        ratio = len(sample.encode("utf-8")) / n
    lo, hi = _PLAUSIBLE_BYTES_PER_TOKEN
    if not lo <= ratio <= hi:
        logger.warning("token calibration for %s measured %.1f bytes/token — implausible, using %.1f",
                       model, ratio, _FALLBACK_BYTES_PER_TOKEN)
        return _FALLBACK_BYTES_PER_TOKEN
    return ratio


# Learned from real calls: when Ollama reports a prompt larger than planned, the
# model's estimates are scaled up for the rest of the process (never down).
_correction: dict[str, float] = {}
_correction_lock = threading.Lock()


def learn_overflow(model: str, planned: int, actual: int) -> float:
    """Record that a prompt planned at `planned` tokens was really `actual`."""
    if planned <= 0 or actual <= planned:
        return _correction.get(model, 1.0)
    with _correction_lock:
        factor = max(_correction.get(model, 1.0), (actual / planned) * 1.05)
        _correction[model] = factor
    logger.warning("token estimates for %s corrected x%.2f (planned %d, Ollama counted %d)",
                   model, factor, planned, actual)
    return factor


def estimate(model: str, text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text.encode("utf-8")) / bytes_per_token(model) * _correction.get(model, 1.0))


@lru_cache(maxsize=None)
def image_tokens(model: str) -> int:
    """Tokens one image costs this model. Fixed-resolution encoders (llava,
    moondream) cost a constant, measured once with a probe image."""
    buf = io.BytesIO()
    Image.new("RGB", (640, 400), "white").save(buf, "JPEG")
    probe = base64.b64encode(buf.getvalue()).decode()
    opts = {"num_predict": 1, "num_ctx": context_limit(model)}
    try:
        a = inference.call(model, "chat", messages=[{"role": "user", "content": uuid.uuid4().hex[:6],
                                                      "images": [probe]}], options=opts).prompt_eval_count
        b = inference.call(model, "chat", messages=[{"role": "user", "content": uuid.uuid4().hex[:6]}],
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
    try:
        return inference.call(model, "chat", messages=messages,
                              options={"num_predict": 1, "num_ctx": num_ctx}).prompt_eval_count
    except Exception as e:
        # Over the window, Ollama refuses — but tells the exact count, which is the answer.
        n = overflow_tokens(e)
        if n is None:
            raise
        return n


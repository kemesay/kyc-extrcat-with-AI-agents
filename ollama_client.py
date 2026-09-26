"""
Shared, resilient wrapper around ollama.chat() for vision calls.

Wraps every call with the process-wide concurrency limit (ollama_concurrency.py)
and one retry, chosen by what actually failed:

  - "token repeat limit reached" — Ollama's own abort on degenerate repeating
    output. A sampling problem, not a device fault: the call is retried on the
    GPU with a stronger repeat penalty and a little more temperature so the
    model resamples out of the loop.
  - "ErrorDeviceLost" / "device lost" / "model runner has unexpectedly
    stopped" — a real GPU backend crash (first seen with the Vulkan backend on
    the earlier Intel Iris Xe machine, ollama/ollama#13086, #13964, #5387,
    #18477). Retried once on CPU, which is slow but stable.

Why the split: retrying repetition loops on CPU (the original behaviour) is
harmful on Apple Silicon. The CPU-only runner stays loaded and every later
call — from this pipeline and from the universal engine — then runs on CPU,
~10x slower, until the model unloads. The CPU retry therefore also passes
keep_alive=0 so that runner unloads right after the retry and the next call
reloads the model onto the GPU.
"""

import ollama

from ollama_concurrency import ollama_semaphore

REPETITION_MARKERS = ("token repeat limit reached",)
DEVICE_FAILURE_MARKERS = (
    "ErrorDeviceLost",
    "device lost",
    "model runner has unexpectedly stopped",
)
# Kept for backward compatibility with anything importing the old name.
GPU_BACKEND_FAILURE_MARKERS = REPETITION_MARKERS + DEVICE_FAILURE_MARKERS

RESAMPLE_OPTIONS = {"repeat_penalty": 1.35, "temperature": 0.4}


def _matches(exc: Exception, markers: tuple) -> bool:
    text = str(exc).lower()
    return any(marker.lower() in text for marker in markers)


def _is_gpu_backend_failure(exc: Exception) -> bool:
    return _matches(exc, GPU_BACKEND_FAILURE_MARKERS)


def vision_chat(model: str, messages: list, retry_on_cpu: bool = True) -> dict:
    with ollama_semaphore:
        try:
            return ollama.chat(model=model, messages=messages)
        except Exception as e:
            if _matches(e, REPETITION_MARKERS):
                print(f"  ⚠️  {model}: repetition loop ({e}) — resampling on GPU with a stronger repeat penalty...")
                return ollama.chat(model=model, messages=messages, options=RESAMPLE_OPTIONS)
            if retry_on_cpu and _matches(e, DEVICE_FAILURE_MARKERS):
                print(f"  ⚠️  {model}: GPU device failure ({e}) — retrying once on CPU (num_gpu=0)...")
                # keep_alive=0: unload the CPU-only runner afterwards so the next
                # call goes back to the GPU instead of reusing it.
                return ollama.chat(model=model, messages=messages, options={"num_gpu": 0}, keep_alive=0)
            raise

"""
Shared concurrency limiter for direct ollama.chat() calls, across ALL
pipelines and ALL concurrently-submitted jobs.

Intel Iris Xe iGPU shares system RAM as VRAM — running multiple heavy vision
model calls at once (whether from multiple agents within one pipeline run,
or from multiple extraction jobs submitted around the same time) causes
memory/bandwidth contention severe enough that generation degenerates
("token repeat limit reached") and per-call latency balloons from ~10-100s
to 200s+.

Every direct ollama.chat() call carrying an image, from any pipeline
(main.py's parallel pipeline, crewai_pipeline.py's sequential pipeline),
must acquire this semaphore first. It is process-wide, so it caps total
concurrent Ollama load regardless of how many HTTP requests/jobs are in
flight at once.
"""

import threading

OLLAMA_MAX_CONCURRENT = 2
ollama_semaphore = threading.Semaphore(OLLAMA_MAX_CONCURRENT)

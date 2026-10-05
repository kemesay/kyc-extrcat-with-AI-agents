"""
Standalone worker process — scale out without touching the API:

    python -m universal.worker [--workers N] [--no-warmup]

Shares the queue named by UNIVERSAL_QUEUE_URL (a PostgreSQL URL when the
workers run on other machines) and calls the inference hosts in
UNIVERSAL_OLLAMA_HOSTS / UNIVERSAL_MODEL_HOSTS. Run the API with
UNIVERSAL_INPROCESS_WORKERS=0 when all work is done by these processes.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading

from webhook import deliver_callback

from .jobqueue import get_queue
from .service import WorkerPool


def _deliver(job_id: str, callback_url: str) -> dict:
    job = get_queue().get(job_id) or {}
    status = job.get("status", "failed")
    payload = {"job_id": job_id, "status": status, "engine": "universal", "priority": job.get("priority_name"),
               "message": "Extraction complete" if status == "completed" else (job.get("error") or "failed"),
               "result": job.get("result")}
    return deliver_callback(callback_url, payload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m universal.worker")
    parser.add_argument("--workers", type=int, default=None, help="documents in flight (default: slots + 1)")
    parser.add_argument("--no-warmup", action="store_true", help="don't preload the team models")
    args = parser.parse_args()

    pool = WorkerPool(workers=args.workers, deliver=_deliver).start(warm_up=not args.no_warmup)
    print(f"universal worker {pool.node}: {pool.count} worker(s) on queue {type(pool.queue).__name__}")
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    while not done.wait(60):
        print(json.dumps({"active": pool.active, "queue": pool.queue.stats()}))
    print("stopping: finishing the current documents...")
    pool.stop(timeout=600)


if __name__ == "__main__":
    main()

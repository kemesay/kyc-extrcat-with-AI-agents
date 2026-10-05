"""
One queued job -> one finished result, and the pool of workers that run them.

Per job, in order:
  1. the job's images are written to a private temp dir (0700), deleted after
  2. extract_universal() runs the FULL agent team exactly as before
  3. capture quality and identity-consistency signals are added (they can
     only add review reasons, never change a field)
  4. queue timings go into the result; metrics are recorded

Workers: by default one document per inference slot — one per GPU host.
Measured on the 31-ID set (one 24 GB Mac): a second worker did NOT raise
throughput (108 s/document either way — the model calls are the bottleneck
and still run one at a time) but doubled each document's latency (173-261 s
instead of ~110 s) and interleaved two documents' calls, which forces extra
model swaps. More workers pay off only with more hosts, or with models
pinned to dedicated hosts (UNIVERSAL_MODEL_HOSTS), where documents at
different stages use different GPUs.

A worker process also keeps the machine awake (macOS: caffeinate) — a laptop
that idle-sleeps mid-document stalls it for as long as it sleeps — and
refreshes its own jobs' leases before claiming, so a job whose lease ran out
while the machine slept is never taken from a worker that is still on it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from . import capture, inference, metrics, signals
from .jobqueue import PRIORITIES, JobQueue, data_dir, get_queue

LEASE_SECONDS = float(os.environ.get("UNIVERSAL_LEASE_SECONDS", "300"))
RESULT_TTL = float(os.environ.get("UNIVERSAL_RESULT_TTL", str(24 * 3600)))
SIGNALS_RETENTION = float(os.environ.get("UNIVERSAL_SIGNALS_RETENTION_DAYS", "365")) * 86400
_ROOT = Path(__file__).resolve().parent


def code_fingerprint() -> str:
    """Changes whenever the engine or its prompts change — part of the exact-cache key."""
    h = hashlib.sha256()
    for path in sorted(_ROOT.glob("*.py")) + [_ROOT.parent / "universal_id_extraction_prompts.py"]:
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


_FINGERPRINT = None


def dedupe_key(files: dict[str, tuple[str, bytes]], options: dict, callback_url: Optional[str]) -> str:
    """SHA-256 over the exact image bytes + options: only a byte-identical
    submission ever matches. Similar-looking documents never do."""
    global _FINGERPRINT
    _FINGERPRINT = _FINGERPRINT or code_fingerprint()
    h = hashlib.sha256()
    for side in sorted(files):
        h.update(side.encode())
        h.update(hashlib.sha256(files[side][1]).digest())
    h.update(json.dumps(options, sort_keys=True).encode())
    h.update((callback_url or "").encode())
    h.update(_FINGERPRINT.encode())
    return h.hexdigest()


def process(job: dict, worker: str) -> dict:
    from .pipeline import extract_universal     # heavy import, kept off the API's import path

    work = Path(tempfile.mkdtemp(prefix=f"job_{job['id'][:8]}_", dir=data_dir()))
    os.chmod(work, 0o700)
    try:
        paths = {}
        for side, (suffix, data) in job["files"].items():
            path = work / f"{side}{suffix}"
            path.write_bytes(data)
            paths[side] = str(path)
        payload = job["payload"]
        quality = payload.get("capture_quality") or capture.assess_all(paths)
        result = extract_universal(paths["front"], paths.get("back"), trace_id=job["id"], **payload["options"])
        result["capture_quality"] = quality
        if quality["grade"] == "poor":
            issues = sorted({i for s in quality["sides"].values() for i in s["issues"]})
            result.setdefault("warnings", []).append("capture quality is poor: " + "; ".join(issues))
        if signals.enabled():
            try:
                signals.check_and_record(job["id"], result, paths)
            except Exception as e:
                # A signals-store outage must not lose the extraction — but it is visible.
                result.setdefault("verification", {})["identity_signals"] = {"checked": False, "error": str(e)[:300]}
                result.setdefault("warnings", []).append("identity-consistency check unavailable")
        started = job.get("started_at") or time.time()
        result["queue"] = {"priority": job["priority_name"], "attempt": job["attempts"], "worker": worker,
                           "wait_seconds": round(started - job["created_at"], 2),
                           "joined_duplicates": job.get("joined", 0)}
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


class WorkerPool:
    def __init__(self, queue: Optional[JobQueue] = None, workers: Optional[int] = None,
                 processor=process, deliver=None, poll_seconds: float = 1.0):
        self.queue = queue or get_queue()
        default = inference.default_workers()
        self.count = workers if workers is not None else int(os.environ.get("UNIVERSAL_WORKERS", default))
        self.processor = processor
        self.deliver = deliver
        self.poll = poll_seconds
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.node = f"{os.uname().nodename}:{os.getpid()}"
        self.active: dict[str, str] = {}
        self.warmup_report: Optional[list] = None
        self.caffeinate: Optional[subprocess.Popen] = None

    def start(self, warm_up: bool = True) -> "WorkerPool":
        if (sys.platform == "darwin" and shutil.which("caffeinate")
                and os.environ.get("UNIVERSAL_PREVENT_SLEEP", "1") not in ("0", "false", "no")):
            # Held for the life of this process: -i blocks idle sleep, -w ends it with us.
            self.caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        if warm_up and os.environ.get("UNIVERSAL_WARMUP", "1") not in ("0", "false", "no"):
            threading.Thread(target=self._warm, name="universal-warmup", daemon=True).start()
        for i in range(self.count):
            t = threading.Thread(target=self._loop, args=(f"{self.node}/w{i}",), name=f"universal-worker-{i}",
                                 daemon=True)
            t.start()
            self.threads.append(t)
        threading.Thread(target=self._housekeeping, name="universal-housekeeping", daemon=True).start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout)
        if self.caffeinate is not None:
            self.caffeinate.terminate()

    def alive(self) -> int:
        return sum(t.is_alive() for t in self.threads)

    def _warm(self) -> None:
        try:
            self.warmup_report = inference.warm_up()
        except Exception as e:
            self.warmup_report = [{"status": "failed", "error": str(e)[:200]}]

    def _housekeeping(self) -> None:
        while not self.stop_event.wait(600):
            try:
                self.queue.purge(RESULT_TTL)
                if signals.enabled():
                    signals.get_store().purge(SIGNALS_RETENTION)
            except Exception:
                traceback.print_exc()

    def _refresh_own_leases(self) -> None:
        """After a sleep every lease is past due at once; renewing this
        process's own running jobs first keeps a sibling worker from
        'reviving' them."""
        for w, job_id in list(self.active.items()):
            try:
                self.queue.heartbeat(job_id, w, LEASE_SECONDS)
            except Exception:
                traceback.print_exc()

    def _loop(self, worker: str) -> None:
        while not self.stop_event.is_set():
            try:
                self._refresh_own_leases()
                job = self.queue.claim(worker, LEASE_SECONDS)
            except Exception:
                traceback.print_exc()
                self.stop_event.wait(self.poll)
                continue
            if job is None:
                self.queue.wait_for_work(self.poll)
                continue
            self._run(job, worker)

    def _run(self, job: dict, worker: str) -> None:
        self.active[worker] = job["id"]
        beat_stop = threading.Event()

        def beat():
            while not beat_stop.wait(LEASE_SECONDS / 3):
                self.queue.heartbeat(job["id"], worker, LEASE_SECONDS)

        threading.Thread(target=beat, daemon=True).start()
        priority = job["priority_name"]
        metrics.observe("universal_job_seconds", (job.get("started_at") or time.time()) - job["created_at"],
                        phase="queue_wait", priority=priority)
        started = time.time()
        try:
            result = self.processor(job, worker)
            if not self.queue.complete(job["id"], worker, result):
                print(f"⚠️  job {job['id']}: finished, but its lease had passed to another worker — "
                      "that worker's result stands")
            metrics.record_result(result, priority)
            metrics.inc("universal_jobs_total", status="completed", priority=priority)
        except Exception as e:
            traceback.print_exc()
            self.queue.fail(job["id"], worker, f"{type(e).__name__}: {e}")
            metrics.inc("universal_jobs_total", status="failed", priority=priority)
        finally:
            beat_stop.set()
            self.active.pop(worker, None)
            metrics.observe("universal_job_seconds", time.time() - started, phase="processing", priority=priority)
        callback = job["payload"].get("callback_url")
        if callback and self.deliver:
            try:
                self.queue.set_callback(job["id"], self.deliver(job["id"], callback))
            except Exception:
                traceback.print_exc()


def queue_gauges():
    q = get_queue()
    st = q.stats()
    for name in PRIORITIES:
        yield "universal_queue_depth", {"priority": name}, st["queued"].get(name, 0)
    yield "universal_jobs_processing", {}, st["processing"]
    yield "universal_oldest_queued_seconds", {}, st["oldest_queued_seconds"]
    for h in inference.pool().snapshot():
        yield "universal_inference_busy", {"host": h["host"]}, h["busy"]


metrics.register_gauges(queue_gauges)


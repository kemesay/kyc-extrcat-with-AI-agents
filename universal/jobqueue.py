"""
Durable priority job queue for the universal engine.

Replaces the in-memory job dict: a job survives an API restart, a crashed
worker's job is picked up again when its lease expires, and any number of
worker processes (python -m universal.worker) share one queue.

  backend      PostgreSQL + pgvector — the local one set up by
               `python -m universal.localdb setup` is found automatically, or
               UNIVERSAL_DATABASE_URL / UNIVERSAL_QUEUE_URL=postgresql://...;
               claims use FOR UPDATE SKIP LOCKED, so any number of nodes share
               it. Without PostgreSQL: a SQLite file (one machine, fallback)
  priorities   live (customer at the counter) > online (app / portal KYC) >
               batch (compliance re-verification) > bulk (historical backfill).
               A lower tier ages forward: a job's sort key is its submit time
               plus its tier's offset, so live work always goes first while a
               bulk job still runs within its offset of being submitted.
  images       travel inside the queue (job_files), so a worker on another
               machine needs no shared disk; they are deleted the moment the
               job finishes, and results expire after UNIVERSAL_RESULT_TTL.
  duplicates   byte-identical images with identical options, submitted while
               the first is still queued/running (a double-tap, a client
               retry), join that job instead of running the team twice.
               Similar-looking documents are never joined: the key is a
               SHA-256 of the exact bytes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

PRIORITIES = {"live": 0, "online": 1, "batch": 2, "bulk": 3}
PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}
# Seconds a tier's jobs yield to newer higher-tier jobs (UNIVERSAL_PRIORITY_AGING="0,600,3600,21600").
_DEFAULT_AGING = (0.0, 600.0, 3600.0, 21600.0)
ACTIVE = ("queued", "processing")


def _aging() -> tuple[float, ...]:
    raw = os.environ.get("UNIVERSAL_PRIORITY_AGING", "")
    try:
        values = tuple(float(x) for x in raw.split(",")) if raw else _DEFAULT_AGING
        return values if len(values) == 4 else _DEFAULT_AGING
    except ValueError:
        return _DEFAULT_AGING


def priority_rank(name: str) -> int:
    if name not in PRIORITIES:
        raise ValueError(f"priority must be one of {list(PRIORITIES)}")
    return PRIORITIES[name]


_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS universal_jobs (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL,
        sort_key DOUBLE PRECISION NOT NULL,
        dedupe_key TEXT,
        payload TEXT NOT NULL,
        result TEXT,
        error TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 2,
        worker TEXT,
        lease_until DOUBLE PRECISION,
        created_at DOUBLE PRECISION NOT NULL,
        started_at DOUBLE PRECISION,
        completed_at DOUBLE PRECISION,
        joined INTEGER NOT NULL DEFAULT 0,
        callback TEXT)""",
    "CREATE INDEX IF NOT EXISTS universal_jobs_claim ON universal_jobs (status, sort_key)",
    "CREATE INDEX IF NOT EXISTS universal_jobs_dedupe ON universal_jobs (dedupe_key, status)",
    """CREATE TABLE IF NOT EXISTS universal_job_files (
        job_id TEXT NOT NULL,
        side TEXT NOT NULL,
        suffix TEXT NOT NULL,
        data {blob} NOT NULL,
        PRIMARY KEY (job_id, side))""",
]

_COLUMNS = ("id", "status", "priority", "sort_key", "dedupe_key", "payload", "result", "error", "attempts",
            "max_attempts", "worker", "lease_until", "created_at", "started_at", "completed_at", "joined", "callback")


class JobQueue(ABC):
    """Backend-neutral queue logic; subclasses supply the connection and SQL dialect."""

    placeholder = "?"
    blob_type = "BLOB"

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        self._wakeup = threading.Condition()
        self._local = threading.local()

    # ── dialect / connection ────────────────────────────────────────────
    @abstractmethod
    def _connect(self): ...

    @abstractmethod
    def _begin_claim(self, cur) -> None: ...

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    def _sql(self, sql: str) -> str:
        return sql.replace("?", self.placeholder)

    def _tx(self, write: bool = True):
        return _Tx(self, write)

    def init_schema(self) -> None:
        with self._tx() as cur:
            for stmt in _SCHEMA:
                cur.execute(stmt.format(blob=self.blob_type))

    # ── producer side ───────────────────────────────────────────────────
    def submit(self, payload: dict, files: dict[str, tuple[str, bytes]], priority: str = "online",
               dedupe_key: Optional[str] = None) -> tuple[str, bool]:
        """-> (job_id, joined). joined=True: an identical job was already
        queued/running; the caller's files are not stored and that job's id is
        returned (and its priority raised to this request's if higher)."""
        rank = priority_rank(priority)
        now = time.time()
        sort_key = now + _aging()[rank]
        with self._tx() as cur:
            if dedupe_key:
                cur.execute(self._sql("SELECT id, priority, sort_key FROM universal_jobs "
                                      "WHERE dedupe_key = ? AND status IN ('queued', 'processing') "
                                      "ORDER BY created_at LIMIT 1"), (dedupe_key,))
                row = cur.fetchone()
                if row:
                    cur.execute(self._sql("UPDATE universal_jobs SET joined = joined + 1, "
                                          "priority = CASE WHEN priority > ? THEN ? ELSE priority END, "
                                          "sort_key = CASE WHEN sort_key > ? THEN ? ELSE sort_key END "
                                          "WHERE id = ?"), (rank, rank, sort_key, sort_key, row[0]))
                    return row[0], True
            job_id = str(uuid.uuid4())
            cur.execute(self._sql("INSERT INTO universal_jobs (id, status, priority, sort_key, dedupe_key, payload, "
                                  "max_attempts, created_at) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)"),
                        (job_id, rank, sort_key, dedupe_key, json.dumps(payload), self.max_attempts, now))
            for side, (suffix, data) in files.items():
                cur.execute(self._sql("INSERT INTO universal_job_files (job_id, side, suffix, data) "
                                      "VALUES (?, ?, ?, ?)"), (job_id, side, suffix, data))
        with self._wakeup:
            self._wakeup.notify_all()
        return job_id, False

    # ── worker side ─────────────────────────────────────────────────────
    def claim(self, worker: str, lease_seconds: float = 600.0) -> Optional[dict]:
        """Atomically take the next job (reviving expired leases first)."""
        now = time.time()
        with self._tx() as cur:
            self._begin_claim(cur)
            # A lease that ran out means its worker died: retry, or fail for good.
            cur.execute(self._sql("UPDATE universal_jobs SET status = 'failed', completed_at = ?, "
                                  "error = 'worker lost (lease expired) on every attempt' "
                                  "WHERE status = 'processing' AND lease_until < ? AND attempts >= max_attempts"),
                        (now, now))
            cur.execute(self._sql("UPDATE universal_jobs SET status = 'queued', worker = NULL, lease_until = NULL "
                                  "WHERE status = 'processing' AND lease_until < ? AND attempts < max_attempts"),
                        (now,))
            cur.execute(self._sql(self._claim_select()))
            row = cur.fetchone()
            if not row:
                return None
            job_id = row[0]
            cur.execute(self._sql("UPDATE universal_jobs SET status = 'processing', worker = ?, lease_until = ?, "
                                  "attempts = attempts + 1, started_at = ? WHERE id = ?"),
                        (worker, now + lease_seconds, now, job_id))
            cur.execute(self._sql("SELECT side, suffix, data FROM universal_job_files WHERE job_id = ?"), (job_id,))
            files = {side: (suffix, bytes(data)) for side, suffix, data in cur.fetchall()}
        job = self.get(job_id)
        job["files"] = files
        return job

    def _claim_select(self) -> str:
        return "SELECT id FROM universal_jobs WHERE status = 'queued' ORDER BY sort_key LIMIT 1"

    def heartbeat(self, job_id: str, worker: str, lease_seconds: float = 600.0) -> bool:
        with self._tx() as cur:
            cur.execute(self._sql("UPDATE universal_jobs SET lease_until = ? WHERE id = ? AND worker = ? "
                                  "AND status = 'processing'"), (time.time() + lease_seconds, job_id, worker))
            return cur.rowcount == 1

    def complete(self, job_id: str, worker: str, result: dict) -> bool:
        return self._finish(job_id, worker, "completed", result=json.dumps(result, ensure_ascii=False))

    def fail(self, job_id: str, worker: str, error: str, retry: bool = False) -> bool:
        if retry:
            with self._tx() as cur:
                cur.execute(self._sql("UPDATE universal_jobs SET status = 'queued', worker = NULL, lease_until = NULL, "
                                      "error = ? WHERE id = ? AND worker = ? AND attempts < max_attempts"),
                            (error[:2000], job_id, worker))
                if cur.rowcount == 1:
                    return True
        return self._finish(job_id, worker, "failed", error=error[:2000])

    def _finish(self, job_id: str, worker: str, status: str, result: Optional[str] = None,
                error: Optional[str] = None) -> bool:
        with self._tx() as cur:
            cur.execute(self._sql("UPDATE universal_jobs SET status = ?, result = ?, error = ?, completed_at = ?, "
                                  "lease_until = NULL WHERE id = ? AND worker = ? AND status = 'processing'"),
                        (status, result, error, time.time(), job_id, worker))
            done = cur.rowcount == 1
            if done:
                cur.execute(self._sql("DELETE FROM universal_job_files WHERE job_id = ?"), (job_id,))
        return done

    def set_callback(self, job_id: str, delivery: dict) -> None:
        with self._tx() as cur:
            cur.execute(self._sql("UPDATE universal_jobs SET callback = ? WHERE id = ?"),
                        (json.dumps(delivery), job_id))

    def wait_for_work(self, timeout: float) -> None:
        """In-process workers wake on submit; other processes poll."""
        with self._wakeup:
            self._wakeup.wait(timeout)

    # ── readers ─────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[dict]:
        with self._tx(write=False) as cur:
            cur.execute(self._sql(f"SELECT {', '.join(_COLUMNS)} FROM universal_jobs WHERE id = ?"), (job_id,))
            row = cur.fetchone()
        if not row:
            return None
        job = dict(zip(_COLUMNS, row))
        job["payload"] = json.loads(job["payload"])
        job["result"] = json.loads(job["result"]) if job["result"] else None
        job["callback"] = json.loads(job["callback"]) if job["callback"] else None
        job["priority_name"] = PRIORITY_NAMES.get(job["priority"], str(job["priority"]))
        return job

    def position(self, job_id: str) -> Optional[int]:
        """Queued jobs that will be claimed before this one (0 = next)."""
        with self._tx(write=False) as cur:
            cur.execute(self._sql("SELECT sort_key FROM universal_jobs WHERE id = ? AND status = 'queued'"), (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(self._sql("SELECT COUNT(*) FROM universal_jobs WHERE status = 'queued' AND sort_key < ?"),
                        (row[0],))
            return int(cur.fetchone()[0])

    def find_completed(self, dedupe_key: str, max_age_seconds: float) -> Optional[dict]:
        with self._tx(write=False) as cur:
            cur.execute(self._sql("SELECT id FROM universal_jobs WHERE dedupe_key = ? AND status = 'completed' "
                                  "AND completed_at >= ? ORDER BY completed_at DESC LIMIT 1"),
                        (dedupe_key, time.time() - max_age_seconds))
            row = cur.fetchone()
        return self.get(row[0]) if row else None

    def cancel(self, job_id: str) -> str:
        """queued -> cancelled; finished -> deleted; processing -> 'busy'."""
        with self._tx() as cur:
            cur.execute(self._sql("SELECT status FROM universal_jobs WHERE id = ?"), (job_id,))
            row = cur.fetchone()
            if not row:
                return "not_found"
            if row[0] == "processing":
                return "busy"
            cur.execute(self._sql("DELETE FROM universal_job_files WHERE job_id = ?"), (job_id,))
            cur.execute(self._sql("DELETE FROM universal_jobs WHERE id = ?"), (job_id,))
            return "cancelled" if row[0] == "queued" else "deleted"

    def stats(self) -> dict:
        with self._tx(write=False) as cur:
            cur.execute("SELECT status, priority, COUNT(*) FROM universal_jobs GROUP BY status, priority")
            rows = cur.fetchall()
            cur.execute("SELECT MIN(created_at) FROM universal_jobs WHERE status = 'queued'")
            oldest = cur.fetchone()[0]
        out: dict[str, Any] = {"queued": {n: 0 for n in PRIORITIES}, "processing": 0, "completed": 0, "failed": 0}
        for status, prio, count in rows:
            if status == "queued":
                out["queued"][PRIORITY_NAMES.get(prio, str(prio))] = int(count)
            else:
                out[status] = out.get(status, 0) + int(count)
        out["oldest_queued_seconds"] = round(time.time() - oldest, 1) if oldest else 0.0
        return out

    def purge(self, result_ttl_seconds: float) -> int:
        """Finished jobs older than the TTL are deleted — results hold personal data."""
        cutoff = time.time() - result_ttl_seconds
        with self._tx() as cur:
            cur.execute(self._sql("SELECT id FROM universal_jobs WHERE status IN ('completed', 'failed') "
                                  "AND completed_at < ?"), (cutoff,))
            ids = [r[0] for r in cur.fetchall()]
            for job_id in ids:
                cur.execute(self._sql("DELETE FROM universal_job_files WHERE job_id = ?"), (job_id,))
                cur.execute(self._sql("DELETE FROM universal_jobs WHERE id = ?"), (job_id,))
        return len(ids)


class _Tx:
    """One transaction on the calling thread's connection."""

    def __init__(self, q: JobQueue, write: bool = True):
        self.q = q
        self.write = write

    def __enter__(self):
        self.conn = self.q._conn()
        self.cur = self.conn.cursor()
        if isinstance(self.conn, sqlite3.Connection):
            # Writers take the lock up front (no upgrade deadlocks); readers — status
            # polls — use a plain snapshot and never block a worker.
            self.cur.execute("BEGIN IMMEDIATE" if self.write else "BEGIN")
        return self.cur

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.cur.close()
        return False


class SQLiteJobQueue(JobQueue):
    def __init__(self, path: str, max_attempts: int = 2):
        super().__init__(max_attempts)
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()
        try:
            os.chmod(path, 0o600)                     # results carry personal data
        except OSError:
            pass

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _begin_claim(self, cur) -> None:
        pass                                          # BEGIN IMMEDIATE already serialises writers


class PostgresJobQueue(JobQueue):
    placeholder = "%s"
    blob_type = "BYTEA"

    def __init__(self, url: str, max_attempts: int = 2):
        super().__init__(max_attempts)
        self.url = url
        self.init_schema()

    def _connect(self):
        import psycopg
        return psycopg.connect(self.url)

    def _begin_claim(self, cur) -> None:
        pass

    def _claim_select(self) -> str:
        # SKIP LOCKED: concurrent workers on any node each take a different job.
        return ("SELECT id FROM universal_jobs WHERE status = 'queued' ORDER BY sort_key LIMIT 1 "
                "FOR UPDATE SKIP LOCKED")


_queue: Optional[JobQueue] = None
_queue_lock = threading.Lock()


_REPO_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """Runtime data (database, queue, keys). Anchored to the repository, not the
    working directory, so the API and every worker find the same files."""
    path = Path(os.environ.get("UNIVERSAL_DATA_DIR", _REPO_ROOT / "var" / "universal"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_postgres(url: str) -> bool:
    return url.startswith(("postgres://", "postgresql://"))


def database_url() -> str:
    """UNIVERSAL_QUEUE_URL / UNIVERSAL_DATABASE_URL, else the local PostgreSQL +
    pgvector from `python -m universal.localdb setup` (started if it is down),
    else '' (SQLite fallback)."""
    explicit = os.environ.get("UNIVERSAL_QUEUE_URL") or os.environ.get("UNIVERSAL_DATABASE_URL")
    if explicit:
        return explicit
    from . import localdb
    url = localdb.managed_url()
    if url:
        localdb.ensure_running()
    return url


def backend_name(q: "JobQueue") -> str:
    return "postgresql" if isinstance(q, PostgresJobQueue) else "sqlite"


def get_queue() -> JobQueue:
    global _queue
    with _queue_lock:
        if _queue is None:
            url = database_url()
            attempts = int(os.environ.get("UNIVERSAL_MAX_ATTEMPTS", "2"))
            if is_postgres(url):
                _queue = PostgresJobQueue(url, attempts)
            else:
                print("⚠️  universal queue: no PostgreSQL configured — using the SQLite fallback "
                      "(run: python -m universal.localdb setup)")
                _queue = SQLiteJobQueue(url.removeprefix("sqlite:///") or str(data_dir() / "queue.sqlite3"), attempts)
        return _queue


def set_queue(q: Optional[JobQueue]) -> None:
    """Tests / reconfiguration."""
    global _queue
    with _queue_lock:
        _queue = q

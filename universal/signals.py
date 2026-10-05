"""
Identity-consistency signals across submissions — fraud flags, never data.

A finished extraction is compared with earlier submissions, and a conflict
sends it to human review. Nothing here ever fills or changes a field: two
different people's cards of the same type look alike to any image
similarity measure, so similarity is only ever used to ASK a question.

  document_number_seen_with_different_identity
        the same issuing country + document number was extracted earlier
        with a different name / date of birth                    -> review
  image_reused_for_different_identity
        a near-identical image (perceptual hash within IMAGE_HAMMING bits —
        a recompressed or resized copy of the same photo) was submitted
        earlier for a different identity                         -> review
  resubmission / same_identity_other_document
        informational only

Privacy: identities are stored as HMAC-SHA256 digests under a server secret
(UNIVERSAL_SIGNALS_KEY, or a key file generated in the data dir), never as
names or numbers; images as 64-bit perceptual hashes.

Storage: the engine's PostgreSQL database (universal.jobqueue.database_url;
UNIVERSAL_SIGNALS_URL to use another), SQLite as fallback. On
PostgreSQL with the pgvector extension the image hash is a bit(64) column
searched by Hamming distance (<~>) through an HNSW index; without it — and
on SQLite — 8 band columns give an exact search: two 64-bit hashes within 7
bits of each other must agree on at least one of 8 bytes (pigeonhole).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

IMAGE_HAMMING = int(os.environ.get("UNIVERSAL_IMAGE_HAMMING", "6"))   # measured: same photo re-encoded 0-2, different documents >= 16
_BANDS = 8


# ── fingerprints ────────────────────────────────────────────────────────

def image_phash(path: str) -> int:
    """64-bit DCT perceptual hash (unsigned int)."""
    import cv2
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((32, 32), Image.LANCZOS)
    dct = cv2.dct(np.asarray(im, dtype=np.float32))[:8, :8].flatten()
    bits = dct > np.median(dct[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & (2 ** 64 - 1)).count("1")


def _bands(h: int) -> list[int]:
    return [(h >> (8 * i)) & 0xFF for i in range(_BANDS)]


def _signed(h: int) -> int:
    return h - 2 ** 64 if h >= 2 ** 63 else h


def _unsigned(h: int) -> int:
    return h + 2 ** 64 if h < 0 else h


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").upper()
    return "".join(c for c in text if not unicodedata.combining(c))


def identity_keys(extracted: dict, country: str, secret: bytes) -> tuple[str, str]:
    """-> (document key, person key); empty when the fields aren't there."""
    number = re.sub(r"[^A-Z0-9]", "", _fold(extracted.get("document_number", "")))
    name_words = sorted(re.sub(r"[^\w ]", " ", _fold(extracted.get("full_name", ""))).split())
    dob = (extracted.get("date_of_birth") or "").strip()
    mac = lambda s: hmac.new(secret, s.encode("utf-8"), hashlib.sha256).hexdigest()
    doc_key = mac(f"doc|{country}|{number}") if number else ""
    person_key = mac(f"person|{' '.join(name_words)}|{dob}") if name_words and dob else ""
    return doc_key, person_key


# ── store ───────────────────────────────────────────────────────────────

class SignalStore:
    placeholder = "?"

    def __init__(self):
        self._local = threading.local()
        self.pgvector = False

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._local.conn = self._connect()
        return conn

    def _connect(self):
        raise NotImplementedError

    def _sql(self, sql: str) -> str:
        return sql.replace("?", self.placeholder)

    def _run(self, sql: str, params: tuple = (), fetch: bool = False):
        conn = self._conn()
        cur = conn.cursor()
        try:
            cur.execute(self._sql(sql), params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    def init_schema(self) -> None:
        bands = ", ".join(f"b{i} INTEGER" for i in range(_BANDS))
        extra = ", phash_bits bit(64)" if self.pgvector else ""
        self._run(f"""CREATE TABLE IF NOT EXISTS universal_signals (
            job_id TEXT NOT NULL, side TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL,
            doc_key TEXT, person_key TEXT, phash BIGINT, {bands}{extra})""")
        self._run("CREATE INDEX IF NOT EXISTS universal_signals_doc ON universal_signals (doc_key)")
        self._run("CREATE INDEX IF NOT EXISTS universal_signals_person ON universal_signals (person_key)")
        for i in range(_BANDS):
            self._run(f"CREATE INDEX IF NOT EXISTS universal_signals_b{i} ON universal_signals (b{i})")
        if self.pgvector:
            self._run("CREATE INDEX IF NOT EXISTS universal_signals_phash_hnsw ON universal_signals "
                      "USING hnsw (phash_bits bit_hamming_ops)")
        try:                                   # added later: exact image identity
            self._run("ALTER TABLE universal_signals ADD COLUMN img_sha TEXT")
        except Exception:
            pass                               # already there
        self._run("CREATE INDEX IF NOT EXISTS universal_signals_sha ON universal_signals (img_sha)")

    def jobs_with_images(self, shas: list[str]) -> set[str]:
        """Earlier jobs that received a byte-identical image."""
        if not shas:
            return set()
        marks = ", ".join("?" for _ in shas)
        return {r[0] for r in self._run(f"SELECT DISTINCT job_id FROM universal_signals WHERE img_sha IN ({marks})",
                                        tuple(shas), fetch=True)}

    def similar_images(self, h: int, limit: int = 20) -> list[tuple[str, str, int]]:
        """-> [(job_id, person_key, distance)] within IMAGE_HAMMING bits."""
        if self.pgvector:
            bits = format(h, "064b")
            rows = self._run("SELECT job_id, person_key, phash_bits <~> ?::bit(64) AS d FROM universal_signals "
                             "WHERE phash_bits IS NOT NULL ORDER BY phash_bits <~> ?::bit(64) LIMIT ?",
                             (bits, bits, limit), fetch=True)
            return [(j, p or "", int(d)) for j, p, d in rows if d is not None and int(d) <= IMAGE_HAMMING]
        where = " OR ".join(f"b{i} = ?" for i in range(_BANDS))
        rows = self._run(f"SELECT job_id, person_key, phash FROM universal_signals WHERE phash IS NOT NULL "
                         f"AND ({where}) LIMIT 500", tuple(_bands(h)), fetch=True)
        found = [(j, p or "", hamming(h, _unsigned(int(ph)))) for j, p, ph in rows]
        return sorted([f for f in found if f[2] <= IMAGE_HAMMING], key=lambda f: f[2])[:limit]

    def by_key(self, column: str, key: str) -> list[tuple[str, str, str]]:
        assert column in ("doc_key", "person_key")
        rows = self._run(f"SELECT DISTINCT job_id, doc_key, person_key FROM universal_signals WHERE {column} = ? "
                         "LIMIT 50", (key,), fetch=True)
        return [(j, d or "", p or "") for j, d, p in rows]

    def record(self, job_id: str, doc_key: str, person_key: str, hashes: dict[str, int],
               shas: Optional[dict[str, str]] = None) -> None:
        now = time.time()
        for side, h in (hashes or {"front": None}).items():
            cols = ["job_id", "side", "created_at", "doc_key", "person_key", "phash", "img_sha"] + \
                   [f"b{i}" for i in range(_BANDS)]
            vals: list[Any] = [job_id, side, now, doc_key or None, person_key or None,
                               _signed(h) if h is not None else None, (shas or {}).get(side)] + \
                              (_bands(h) if h is not None else [None] * _BANDS)
            marks = ["?"] * len(cols)
            if self.pgvector:
                cols.append("phash_bits")
                vals.append(format(h, "064b") if h is not None else None)
                marks.append("?::bit(64)")
            self._run(f"INSERT INTO universal_signals ({', '.join(cols)}) VALUES ({', '.join(marks)})", tuple(vals))

    def purge(self, retention_seconds: float) -> None:
        self._run("DELETE FROM universal_signals WHERE created_at < ?", (time.time() - retention_seconds,))


class SQLiteSignalStore(SignalStore):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


class PostgresSignalStore(SignalStore):
    placeholder = "%s"

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        try:
            self._run("CREATE EXTENSION IF NOT EXISTS vector")
            self.pgvector = True
        except Exception:
            self.pgvector = False                     # band search still exact
        self.init_schema()

    def _connect(self):
        import psycopg
        return psycopg.connect(self.url)

    def _sql(self, sql: str) -> str:
        return sql.replace("?::", "%s::").replace("?", "%s")


# ── the check ───────────────────────────────────────────────────────────

def evaluate(store: SignalStore, job_id: str, doc_key: str, person_key: str,
             hashes: dict[str, int], shas: Optional[dict[str, str]] = None) -> list[dict]:
    """Signals for this submission against earlier ones (this job excluded).
    Earlier jobs that received the byte-identical file are the SAME submission
    read again: a different reading there is the engine's, not a second
    identity, so it is reported as information, never as fraud."""
    signals: list[dict] = []
    same_file = store.jobs_with_images(list((shas or {}).values())) - {job_id}
    if same_file:
        signals.append({"signal": "same_file_submitted_before", "severity": "info",
                        "earlier_submissions": len(same_file)})
    if doc_key and person_key:
        others = {p for j, _, p in store.by_key("doc_key", doc_key)
                  if j != job_id and j not in same_file and p and p != person_key}
        if others:
            signals.append({"signal": "document_number_seen_with_different_identity", "severity": "review",
                            "earlier_identities": len(others)})
    if person_key:
        docs = {d for j, d, _ in store.by_key("person_key", person_key) if j != job_id and d and d != doc_key}
        if docs:
            signals.append({"signal": "same_identity_other_document", "severity": "info", "documents": len(docs)})
    reused, resubmitted = set(), set()
    for side, h in hashes.items():
        for j, p, d in store.similar_images(h):
            if j == job_id or j in same_file:
                continue
            if person_key and p and p != person_key:
                reused.add((j, d))
            elif p == person_key:
                resubmitted.add(j)
    if reused:
        signals.append({"signal": "image_reused_for_different_identity", "severity": "review",
                        "earlier_submissions": len({j for j, _ in reused}),
                        "closest_distance_bits": min(d for _, d in reused)})
    if resubmitted:
        signals.append({"signal": "resubmission", "severity": "info", "earlier_submissions": len(resubmitted)})
    return signals


def _secret() -> bytes:
    env = os.environ.get("UNIVERSAL_SIGNALS_KEY", "")
    if env:
        return env.encode("utf-8")
    from .jobqueue import data_dir
    path = data_dir() / "signals.key"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    return path.read_bytes()


_store: Optional[SignalStore] = None
_store_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("UNIVERSAL_IDENTITY_SIGNALS", "1") not in ("0", "false", "no")


def get_store() -> SignalStore:
    global _store
    with _store_lock:
        if _store is None:
            from .jobqueue import database_url, is_postgres
            url = os.environ.get("UNIVERSAL_SIGNALS_URL", "") or database_url()
            if is_postgres(url):
                _store = PostgresSignalStore(url)
            else:
                from .jobqueue import data_dir
                _store = SQLiteSignalStore(str(data_dir() / "signals.sqlite3"))
        return _store


def set_store(store: Optional[SignalStore]) -> None:
    global _store
    with _store_lock:
        _store = store


def check_and_record(job_id: str, result: dict, image_paths: dict[str, str]) -> dict:
    """Adds verification.identity_signals to the result; a review-level signal
    sets human_review_required with its reason. Returns the signals block."""
    store = get_store()
    extracted = result.get("extracted_data") or {}
    doc_key, person_key = identity_keys(extracted, result.get("issuing_country") or "", _secret())
    hashes, shas = {}, {}
    for side, path in image_paths.items():
        try:
            hashes[side] = image_phash(path)
            shas[side] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except Exception:
            pass
    signals = evaluate(store, job_id, doc_key, person_key, hashes, shas)
    store.record(job_id, doc_key, person_key, hashes, shas)
    block = {"checked": True, "signals": signals, "backend": "pgvector" if store.pgvector else type(store).__name__}
    result.setdefault("verification", {})["identity_signals"] = block
    for s in signals:
        if s["severity"] == "review":
            reason = f"identity check: {s['signal'].replace('_', ' ')}"
            if reason not in result.setdefault("review_reasons", []):
                result["review_reasons"].append(reason)
            result["human_review_required"] = True
    return block

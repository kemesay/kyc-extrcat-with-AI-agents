"""Serving layer: host pool, durable queue, identity signals, capture check, metrics, API."""

import io
import os
import threading
import time

import numpy as np
import pytest
from PIL import Image, ImageFilter

from universal import capture, inference, jobqueue, metrics, signals
from universal.inference import HostPool
from universal.jobqueue import SQLiteJobQueue


# ── images ──────────────────────────────────────────────────────────────

def _card(seed: int, size=(1200, 760)) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = np.full((size[1], size[0]), 200, np.uint8)
    for _ in range(140):                                  # text-like dark strokes
        x, y = rng.integers(0, size[0] - 60), rng.integers(0, size[1] - 12)
        img[y:y + rng.integers(3, 10), x:x + rng.integers(10, 60)] = rng.integers(10, 80)
    return Image.fromarray(img).convert("RGB")


def _jpeg(im: Image.Image, quality=90) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _save(tmp_path, name, im, quality=90) -> str:
    path = tmp_path / name
    path.write_bytes(_jpeg(im, quality))
    return str(path)


# ── inference host pool ─────────────────────────────────────────────────

class _FakeClient:
    def __init__(self, url):
        self.url = url


def _pool(hosts, model_hosts=None, slots=1):
    return HostPool(hosts, model_hosts, slots, client_factory=_FakeClient)


def test_one_model_call_per_host_at_a_time():
    p = _pool(["http://gpu-a:11434"])
    running, peak, lock = [0], [0], threading.Lock()

    def call():
        with p.slot("qwen2.5vl:7b"):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1

    threads = [threading.Thread(target=call) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 1


def test_two_hosts_serve_two_documents_at_once():
    p = _pool(["http://gpu-a:11434", "http://gpu-b:11434"])
    other = []

    def second_document():
        with p.slot("m", timeout=0.5) as (b, _):
            other.append(b)

    with p.slot("m") as (a, _):
        t = threading.Thread(target=second_document)
        t.start()
        t.join()
    assert {a, *other} == {"http://gpu-a:11434", "http://gpu-b:11434"}


def test_model_pinned_to_dedicated_host():
    p = _pool(["http://gpu-a:11434"], {"llava:13b": ["http://gpu-c:11434"]})
    with p.slot("llava:13b") as (host, client):
        assert host == "http://gpu-c:11434" and client.url == host
    with p.slot("qwen2.5:7b") as (host, _):
        assert host == "http://gpu-a:11434"          # dedicated host takes no general work


def test_warm_host_preferred():
    p = _pool(["http://gpu-a:11434", "http://gpu-b:11434"])
    with p.slot("m") as (first, _):
        pass
    with p.slot("other"):
        pass
    for _ in range(3):
        with p.slot("m") as (host, _):
            assert host == first


def test_slot_is_reentrant_on_the_same_thread():
    p = _pool(["http://gpu-a:11434"])
    with p.slot("m") as (a, _):
        with p.slot("m", timeout=0.2) as (b, _):     # would deadlock without re-entrancy
            assert a == b


def test_slot_times_out_when_every_host_is_busy():
    p = _pool(["http://gpu-a:11434"])
    got = threading.Event()
    release = threading.Event()

    def hold():
        with p.slot("m"):
            got.set()
            release.wait(2)

    t = threading.Thread(target=hold)
    t.start()
    got.wait(1)
    with pytest.raises(TimeoutError):
        with p.slot("m", timeout=0.05):
            pass
    release.set()
    t.join()


def test_host_configuration_from_env(monkeypatch):
    monkeypatch.setenv("UNIVERSAL_OLLAMA_HOSTS", "gpu-a:11434, http://gpu-b:11434/")
    monkeypatch.setenv("UNIVERSAL_MODEL_HOSTS", "llava:13b=http://gpu-c:11434|gpu-d:11434;bad")
    assert inference.configured_hosts() == ["http://gpu-a:11434", "http://gpu-b:11434"]
    assert inference.configured_model_hosts() == {"llava:13b": ["http://gpu-c:11434", "http://gpu-d:11434"]}


def test_every_probe_carries_keep_alive(monkeypatch):
    calls = []

    class Client(_FakeClient):
        def generate(self, **kw):
            calls.append(kw)
            return kw

    inference.reset_pool(HostPool(["http://gpu-a:11434"], None, 1, client_factory=Client))
    try:
        inference.call("m", "generate", prompt="x")
        assert calls[0]["keep_alive"] == inference.KEEP_ALIVE and calls[0]["model"] == "m"
    finally:
        inference.reset_pool(None)


def test_warm_up_uses_agent_context_and_reverse_order(monkeypatch):
    from universal import tokens as T
    loads = []

    class Client(_FakeClient):
        def generate(self, **kw):
            loads.append((kw["model"], kw["options"]["num_ctx"], kw["keep_alive"]))

    monkeypatch.setattr(T, "context_limit", lambda m: {"scan": 2048, "read": 8192}[m])
    inference.reset_pool(HostPool(["http://gpu-a:11434"], None, 1, client_factory=Client))
    try:
        report = inference.warm_up(["scan", "read"])
    finally:
        inference.reset_pool(None)
    # first agent's model loaded last (still resident); same num_ctx as real calls (no reload)
    assert loads == [("read", 8192, inference.KEEP_ALIVE), ("scan", 2048, inference.KEEP_ALIVE)]
    assert all(r["status"] == "loaded" for r in report)


# ── durable queue ───────────────────────────────────────────────────────

@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVERSAL_PRIORITY_AGING", "0,600,3600,21600")
    return SQLiteJobQueue(str(tmp_path / "q.sqlite3"))


FILES = {"front": (".jpg", b"front-bytes")}


def test_higher_priority_claimed_first(queue):
    bulk, _ = queue.submit({"n": 1}, FILES, "bulk")
    online, _ = queue.submit({"n": 2}, FILES, "online")
    live, _ = queue.submit({"n": 3}, FILES, "live")
    order = [queue.claim("w")["id"] for _ in range(3)]
    assert order == [live, online, bulk]


def test_lower_tier_ages_forward(queue, monkeypatch):
    old = time.time() - 700                          # online job waiting longer than its 600 s offset
    monkeypatch.setattr(jobqueue.time, "time", lambda: old)
    waited, _ = queue.submit({}, FILES, "online")
    monkeypatch.setattr(jobqueue.time, "time", lambda: old + 700)
    fresh_live, _ = queue.submit({}, FILES, "live")
    assert queue.claim("w")["id"] == waited
    assert queue.claim("w")["id"] == fresh_live


def test_claim_returns_images_and_complete_deletes_them(queue):
    job_id, _ = queue.submit({"k": "v"}, {"front": (".png", b"A"), "back": (".jpg", b"B")}, "online")
    job = queue.claim("w1")
    assert job["files"] == {"front": (".png", b"A"), "back": (".jpg", b"B")}
    assert job["status"] == "processing" and job["attempts"] == 1 and job["payload"] == {"k": "v"}
    assert queue.complete(job_id, "w1", {"ok": True})
    done = queue.get(job_id)
    assert done["status"] == "completed" and done["result"] == {"ok": True}
    with queue._tx() as cur:
        cur.execute("SELECT COUNT(*) FROM universal_job_files")
        assert cur.fetchone()[0] == 0


def test_identical_submission_joins_the_running_job(queue):
    a, joined_a = queue.submit({}, FILES, "bulk", dedupe_key="k1")
    b, joined_b = queue.submit({}, FILES, "live", dedupe_key="k1")
    c, joined_c = queue.submit({}, FILES, "online", dedupe_key="k2")
    assert (joined_a, joined_b, joined_c) == (False, True, False) and a == b != c
    job = queue.get(a)
    assert job["priority_name"] == "live" and job["joined"] == 1   # raised to the waiting caller's tier


def test_finished_job_is_not_joined(queue):
    a, _ = queue.submit({}, FILES, "online", dedupe_key="k")
    queue.claim("w")
    queue.complete(a, "w", {})
    b, joined = queue.submit({}, FILES, "online", dedupe_key="k")
    assert not joined and b != a


def test_expired_lease_is_retried_then_failed(queue, monkeypatch):
    job_id, _ = queue.submit({}, FILES, "online")
    now = time.time()
    assert queue.claim("dead-1", lease_seconds=10)["id"] == job_id
    monkeypatch.setattr(jobqueue.time, "time", lambda: now + 11)
    again = queue.claim("w2", lease_seconds=10)
    assert again["id"] == job_id and again["attempts"] == 2 and again["files"] == FILES
    monkeypatch.setattr(jobqueue.time, "time", lambda: now + 30)
    assert queue.claim("w3") is None
    job = queue.get(job_id)
    assert job["status"] == "failed" and "lease expired" in job["error"]


def test_only_the_lease_holder_can_finish(queue):
    job_id, _ = queue.submit({}, FILES, "online")
    queue.claim("w1")
    assert not queue.complete(job_id, "someone-else", {})
    assert queue.get(job_id)["status"] == "processing"


def test_retryable_failure_requeues(queue):
    job_id, _ = queue.submit({}, FILES, "online")
    queue.claim("w1")
    assert queue.fail(job_id, "w1", "ollama restarting", retry=True)
    assert queue.get(job_id)["status"] == "queued"
    queue.claim("w1")
    queue.fail(job_id, "w1", "still down", retry=True)          # attempts exhausted
    assert queue.get(job_id)["status"] == "failed"


def test_position_cancel_stats_purge(queue, monkeypatch):
    a, _ = queue.submit({}, FILES, "online")
    b, _ = queue.submit({}, FILES, "online")
    assert queue.position(a) == 0 and queue.position(b) == 1
    assert queue.cancel(b) == "cancelled" and queue.get(b) is None
    st = queue.stats()
    assert st["queued"]["online"] == 1 and st["processing"] == 0
    queue.claim("w")
    assert queue.cancel(a) == "busy"
    queue.complete(a, "w", {"x": 1})
    now = time.time()
    monkeypatch.setattr(jobqueue.time, "time", lambda: now + 90000)
    assert queue.purge(86400) == 1 and queue.get(a) is None


def test_find_completed_respects_age(queue, monkeypatch):
    a, _ = queue.submit({}, FILES, "online", dedupe_key="k")
    queue.claim("w")
    queue.complete(a, "w", {"r": 1})
    assert queue.find_completed("k", 60)["id"] == a
    now = time.time()
    monkeypatch.setattr(jobqueue.time, "time", lambda: now + 120)
    assert queue.find_completed("k", 60) is None


def test_queue_is_shared_across_connections(tmp_path):
    path = str(tmp_path / "shared.sqlite3")
    producer, consumer = SQLiteJobQueue(path), SQLiteJobQueue(path)
    job_id, _ = producer.submit({"n": 1}, FILES, "online")
    got = []
    t = threading.Thread(target=lambda: got.append(consumer.claim("other-process")))
    t.start()
    t.join()
    assert got[0]["id"] == job_id


# ── identity signals ────────────────────────────────────────────────────

def test_phash_survives_reencoding_but_separates_documents(tmp_path):
    base = _card(1)
    same = _save(tmp_path, "a.jpg", base)
    reencoded = _save(tmp_path, "b.jpg", base.resize((800, 507)), quality=60)
    other = _save(tmp_path, "c.jpg", _card(2))
    h = signals.image_phash(same)
    assert signals.hamming(h, signals.image_phash(reencoded)) <= 2
    assert signals.hamming(h, signals.image_phash(other)) > signals.IMAGE_HAMMING


def test_identity_keys_are_keyed_hashes_not_data():
    doc, person = signals.identity_keys({"document_number": "ep 1234567", "full_name": "Jane Mary DOE",
                                         "date_of_birth": "1990-01-01"}, "ETH", b"secret")
    doc2, person2 = signals.identity_keys({"document_number": "EP1234567", "full_name": "DOE JANE MARY",
                                           "date_of_birth": "1990-01-01"}, "ETH", b"secret")
    assert (doc, person) == (doc2, person2)                    # spacing / case / order independent
    assert "1234567" not in doc and "DOE" not in person
    assert signals.identity_keys({"document_number": "EP1234567"}, "ETH", b"other")[0] != doc


def _store(tmp_path):
    return signals.SQLiteSignalStore(str(tmp_path / "s.sqlite3"))


def test_document_number_with_different_identity_flags_review(tmp_path):
    st = _store(tmp_path)
    st.record("job1", "DOC", "PERSON-A", {"front": 0xFFFF0000FFFF0000})
    found = signals.evaluate(st, "job2", "DOC", "PERSON-B", {"front": 0x0123456789ABCDEF})
    assert [s["signal"] for s in found] == ["document_number_seen_with_different_identity"]
    assert signals.evaluate(st, "job3", "DOC", "PERSON-A", {}) == []                      # same person again


def test_reused_image_for_different_identity_flags_review(tmp_path):
    st = _store(tmp_path)
    h = 0xF0F0F0F0AAAAAAAA
    st.record("job1", "DOC-1", "PERSON-A", {"front": h})
    near = h ^ 0b10110                                                                   # 3 bits away
    found = signals.evaluate(st, "job2", "DOC-2", "PERSON-B", {"front": near})
    assert found[0]["signal"] == "image_reused_for_different_identity" and found[0]["closest_distance_bits"] == 3
    same_person = signals.evaluate(st, "job3", "DOC-1", "PERSON-A", {"front": near})
    assert [s["signal"] for s in same_person] == ["resubmission"]
    far = h ^ 0xFFFF                                                                     # 16 bits away
    assert signals.evaluate(st, "job4", "DOC-3", "PERSON-C", {"front": far}) == []


def test_band_search_finds_every_hash_within_threshold(tmp_path):
    st = _store(tmp_path)
    rng = np.random.default_rng(7)
    h = int(rng.integers(0, 2 ** 63)) * 2 + 1
    for d in range(0, signals.IMAGE_HAMMING + 1):
        flip = sum(1 << int(b) for b in rng.choice(64, d, replace=False)) if d else 0
        st.record(f"j{d}", "", f"p{d}", {"front": h ^ flip})
    assert sorted(dist for _, _, dist in st.similar_images(h)) == list(range(signals.IMAGE_HAMMING + 1))


def test_check_and_record_adds_review(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIVERSAL_SIGNALS_KEY", "k")
    signals.set_store(_store(tmp_path))
    try:
        img = _save(tmp_path, "id.jpg", _card(3))
        first = {"issuing_country": "ETH", "review_reasons": [], "human_review_required": False,
                 "extracted_data": {"document_number": "EP1", "full_name": "A B", "date_of_birth": "1990-01-01"}}
        signals.check_and_record("j1", first, {"front": img})
        assert first["human_review_required"] is False
        second = {"issuing_country": "ETH", "review_reasons": [], "human_review_required": False,
                  "extracted_data": {"document_number": "EP1", "full_name": "C D", "date_of_birth": "1985-05-05"}}
        resaved = _save(tmp_path, "id_resaved.jpg", _card(3), quality=70)     # same photo, different file
        block = signals.check_and_record("j2", second, {"front": resaved})
        assert second["human_review_required"] is True
        assert {s["signal"] for s in block["signals"]} == {"document_number_seen_with_different_identity",
                                                           "image_reused_for_different_identity"}
        assert all("identity check" in r for r in second["review_reasons"])
    finally:
        signals.set_store(None)


# ── capture quality ─────────────────────────────────────────────────────

def test_sharp_capture_is_good_and_blurred_dark_one_poor(tmp_path):
    good = capture.assess(_save(tmp_path, "good.jpg", _card(4)))
    assert good["grade"] == "good", good
    bad_img = _card(4).filter(ImageFilter.GaussianBlur(12)).point(lambda v: v * 0.15)
    bad = capture.assess(_save(tmp_path, "bad.jpg", bad_img))
    assert bad["grade"] == "poor"
    assert "image is blurred or out of focus" in bad["issues"] and "image is too dark" in bad["issues"]


def test_capture_reads_bytes_and_reports_worst_side(tmp_path):
    tiny = capture.assess(io.BytesIO(_jpeg(_card(5, size=(260, 170)))))
    assert tiny["grade"] == "poor" and any("resolution" in i for i in tiny["issues"])
    both = capture.assess_all({"front": _save(tmp_path, "f.jpg", _card(6)),
                               "back": _save(tmp_path, "b.jpg", _card(6, size=(260, 170)))})
    assert both["grade"] == "poor" and both["sides"]["front"]["grade"] == "good"


# ── metrics ─────────────────────────────────────────────────────────────

def test_metrics_render_prometheus_text():
    metrics.reset()
    metrics.inc("universal_jobs_total", status="completed", priority="live")
    metrics.record_result({"timings_seconds": {"agents": 42.0}, "human_review_required": True,
                           "agents": [{"agent": "supervisor", "model": "qwen2.5:7b", "duration_seconds": 3.5}]},
                          "live")
    text = metrics.render()
    assert 'universal_jobs_total{priority="live",status="completed"} 1' in text
    assert 'universal_stage_seconds_bucket{stage="agents",le="45"} 1' in text
    assert 'universal_agent_seconds_count{agent="supervisor",model="qwen2.5:7b"} 1' in text
    assert 'universal_human_review_total{priority="live"} 1' in text
    assert "# TYPE universal_stage_seconds histogram" in text


# ── API end to end (stub processor: no models) ─────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from universal import api
    from universal.service import WorkerPool

    monkeypatch.setenv("UNIVERSAL_IDENTITY_SIGNALS", "0")
    q = SQLiteJobQueue(str(tmp_path / "api.sqlite3"))
    jobqueue.set_queue(q)
    seen = []

    def fake_process(job, worker):
        seen.append(job)
        time.sleep(0.05)
        return {"engine": "universal", "extracted_data": {"full_name": "JANE DOE"}, "human_review_required": False,
                "review_reasons": [], "capture_quality": job["payload"]["capture_quality"]}

    pool = WorkerPool(q, workers=1, processor=fake_process, poll_seconds=0.05).start(warm_up=False)
    monkeypatch.setattr(api, "_pool", pool)
    app = FastAPI()
    app.include_router(api.router)
    try:
        yield TestClient(app), seen
    finally:
        pool.stop()
        jobqueue.set_queue(None)


def _upload(seed=8, **kw):
    return {"file": ("id.jpg", _jpeg(_card(seed, **kw)), "image/jpeg")}


def test_api_sync_runs_through_the_queue(client):
    c, seen = client
    r = c.post("/universal/extract/sync?priority=live", files=_upload())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["extracted_data"]["full_name"] == "JANE DOE" and body["capture_quality"]["grade"] == "good"
    assert seen[0]["priority_name"] == "live" and seen[0]["payload"]["options"]["mode"] == "full"


def test_api_async_then_status(client):
    c, _ = client
    r = c.post("/universal/extract", files=_upload())
    assert r.status_code == 200 and r.json()["status"] in ("queued", "processing")
    job_id = r.json()["job_id"]
    for _ in range(100):
        s = c.get(f"/universal/status/{job_id}").json()
        if s["status"] == "completed":
            break
        time.sleep(0.05)
    assert s["status"] == "completed" and s["result"]["extracted_data"]["full_name"] == "JANE DOE"
    assert c.delete(f"/universal/job/{job_id}").status_code == 200
    assert c.get(f"/universal/status/{job_id}").status_code == 404


def test_api_fast_mode_disabled_by_quality_policy(client, monkeypatch):
    c, _ = client
    r = c.post("/universal/extract?mode=fast", files=_upload())
    assert r.status_code == 400 and "quality policy" in r.json()["detail"]
    monkeypatch.setenv("UNIVERSAL_ALLOW_FAST_MODE", "1")
    assert c.post("/universal/extract?mode=fast&wait=true", files=_upload()).status_code == 200


def test_api_rejects_poor_capture_only_when_asked(client):
    c, _ = client
    tiny = _upload(size=(260, 170))
    r = c.post("/universal/extract?reject_poor_capture=true", files=tiny)
    assert r.status_code == 422 and r.json()["detail"]["issues"]
    assert c.post("/universal/extract?wait=true", files=tiny).json()["status"] == "completed"


def test_api_double_submit_runs_once(client):
    c, seen = client
    up = _upload(seed=9)
    first = c.post("/universal/extract", files=up).json()["job_id"]
    second = c.post("/universal/extract", files=up).json()["job_id"]
    assert first == second
    c.post("/universal/extract?wait=true", files=up)
    assert sum(1 for j in seen if j["id"] == first) == 1


def test_api_metrics_and_health_shape(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(inference, "residency", lambda: [{"host": "h", "reachable": True, "loaded": [],
                                                          "partially_on_cpu": []}])
    monkeypatch.setattr(inference, "team_models", lambda: ["m"])
    c.post("/universal/extract/sync", files=_upload(seed=10))
    text = c.get("/universal/metrics").text
    assert "universal_capture_total" in text and "universal_queue_depth" in text
    health = c.get("/universal/health").json()
    assert health["status"] == "ok" and health["workers"]["in_process"] == 1


# ── PostgreSQL + pgvector (runs when UNIVERSAL_TEST_PG_URL is set) ──────

PG_URL = os.environ.get("UNIVERSAL_TEST_PG_URL", "")
pg = pytest.mark.skipif(not PG_URL, reason="set UNIVERSAL_TEST_PG_URL to run the PostgreSQL tests")


@pytest.fixture(autouse=True)
def _never_the_real_database():
    # These tests DROP tables: refuse to run them against the engine's own database.
    from universal import localdb
    if PG_URL and PG_URL == (os.environ.get("UNIVERSAL_DATABASE_URL") or localdb.managed_url()):
        pytest.fail("UNIVERSAL_TEST_PG_URL points at the engine's own database — use a throwaway one")


@pg
def test_postgres_queue_skip_locked_and_priorities():
    import psycopg
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS universal_job_files, universal_jobs")
    q = jobqueue.PostgresJobQueue(PG_URL)
    bulk, _ = q.submit({}, FILES, "bulk")
    live, _ = q.submit({}, FILES, "live", dedupe_key="d")
    assert q.submit({}, FILES, "online", dedupe_key="d") == (live, True)
    claimed, lock = [], threading.Lock()

    def worker(n):
        job = jobqueue.PostgresJobQueue(PG_URL).claim(f"w{n}")
        with lock:
            claimed.append(job["id"] if job else None)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(filter(None, claimed)) == sorted([bulk, live]) and claimed.count(None) == 1
    assert q.complete(live, next(j for j in [q.get(live)] if j)["worker"], {"ok": 1})
    assert q.get(live)["result"] == {"ok": 1} and q.stats()["processing"] == 1


@pg
def test_pgvector_hamming_search():
    import psycopg
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS universal_signals")
    st = signals.PostgresSignalStore(PG_URL)
    assert st.pgvector
    h = 0xF0F0F0F0AAAAAAAA
    st.record("job1", "DOC-1", "PERSON-A", {"front": h})
    st.record("job9", "DOC-9", "PERSON-Z", {"front": h ^ 0xFFFFFF})
    found = signals.evaluate(st, "job2", "DOC-2", "PERSON-B", {"front": h ^ 0b111})
    assert found[0]["signal"] == "image_reused_for_different_identity" and found[0]["earlier_submissions"] == 1
    st.record("job3", "DOC-1", "PERSON-C", {})
    assert signals.evaluate(st, "job4", "DOC-1", "PERSON-D", {})[0]["earlier_identities"] == 2


# ── database resolution ─────────────────────────────────────────────────

def test_database_url_prefers_explicit_then_managed(monkeypatch):
    from universal import localdb
    started = []
    monkeypatch.delenv("UNIVERSAL_QUEUE_URL", raising=False)
    monkeypatch.delenv("UNIVERSAL_DATABASE_URL", raising=False)
    monkeypatch.setattr(localdb, "managed_url", lambda: "postgresql://kyc:x@127.0.0.1:5433/kyc")
    monkeypatch.setattr(localdb, "ensure_running", lambda: started.append(1))
    assert jobqueue.database_url() == "postgresql://kyc:x@127.0.0.1:5433/kyc" and started == [1]
    monkeypatch.setenv("UNIVERSAL_DATABASE_URL", "postgresql://prod/db")
    assert jobqueue.database_url() == "postgresql://prod/db"
    monkeypatch.setenv("UNIVERSAL_QUEUE_URL", "postgresql://queue/db")
    assert jobqueue.database_url() == "postgresql://queue/db"


def test_no_database_falls_back_to_sqlite(monkeypatch, tmp_path):
    from universal import localdb
    monkeypatch.delenv("UNIVERSAL_QUEUE_URL", raising=False)
    monkeypatch.delenv("UNIVERSAL_DATABASE_URL", raising=False)
    monkeypatch.setenv("UNIVERSAL_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(localdb, "managed_url", lambda: "")
    jobqueue.set_queue(None)
    try:
        assert isinstance(jobqueue.get_queue(), SQLiteJobQueue)
    finally:
        jobqueue.set_queue(None)


def test_managed_url_read_from_database_env(monkeypatch, tmp_path):
    from universal import localdb
    monkeypatch.setenv("UNIVERSAL_DATA_DIR", str(tmp_path))
    assert localdb.managed_url() == ""
    (tmp_path / "database.env").write_text("# comment\nUNIVERSAL_DATABASE_URL=postgresql://kyc:pw@127.0.0.1:5433/kyc\n")
    assert localdb.managed_url() == "postgresql://kyc:pw@127.0.0.1:5433/kyc"


# ── worker pool behaviour measured on real runs ─────────────────────────

def test_default_is_one_document_per_host():
    one = _pool(["http://gpu-a:11434"])
    two = _pool(["http://gpu-a:11434", "http://gpu-b:11434"])
    line = _pool(["http://gpu-a:11434"], {"llava:13b": ["http://gpu-c:11434"], "qwen2.5:7b": ["http://gpu-d:11434"]})
    for p, want in ((one, 1), (two, 2), (line, 3)):
        inference.reset_pool(p)
        try:
            assert inference.default_workers() == want
        finally:
            inference.reset_pool(None)


def test_own_lease_is_renewed_after_sleep_not_stolen(tmp_path, monkeypatch):
    from universal.service import WorkerPool
    q = SQLiteJobQueue(str(tmp_path / "sleep.sqlite3"))
    job_id, _ = q.submit({}, FILES, "online")
    now = time.time()
    q.claim("node/w0", lease_seconds=10)
    pool = WorkerPool(q, workers=2, processor=lambda j, w: {}, poll_seconds=0.05)
    pool.active["node/w0"] = job_id                    # w0 is still working on it
    monkeypatch.setattr(jobqueue.time, "time", lambda: now + 3600)   # the machine slept an hour
    pool._refresh_own_leases()
    assert q.claim("node/w1", lease_seconds=10) is None                # sibling does not take it
    job = q.get(job_id)
    assert job["status"] == "processing" and job["worker"] == "node/w0" and job["attempts"] == 1


def test_same_file_read_differently_is_not_fraud(tmp_path):
    st = _store(tmp_path)
    h, sha = 0xF0F0F0F0AAAAAAAA, "a" * 64
    st.record("job1", "DOC", "PERSON-A", {"front": h}, {"front": sha})
    same = signals.evaluate(st, "job2", "DOC", "PERSON-B", {"front": h}, {"front": sha})
    assert [s["signal"] for s in same] == ["same_file_submitted_before"] and same[0]["severity"] == "info"
    edited = signals.evaluate(st, "job3", "DOC", "PERSON-C", {"front": h ^ 0b11}, {"front": "b" * 64})
    assert {s["signal"] for s in edited} == {"document_number_seen_with_different_identity",
                                             "image_reused_for_different_identity"}

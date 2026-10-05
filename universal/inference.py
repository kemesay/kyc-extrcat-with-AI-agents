"""
Where each model call runs — and the rule that keeps quality intact.

Quality rule: an inference host runs ONE model call at a time
(UNIVERSAL_HOST_SLOTS=1). Concurrent requests on one GPU share memory
bandwidth and KV space, and on this project's hardware that has produced
degenerate output ("token repeat limit"), so the engine never does it.
Every agent keeps its model, its full prompt and its full-resolution images.

Speed therefore comes from the architecture around the calls:

  hot models      keep_alive=-1 and a warm-up at worker start, so no call
                  pays a model load (UNIVERSAL_KEEP_ALIVE)
  more hosts      UNIVERSAL_OLLAMA_HOSTS=http://gpu1:11434,http://gpu2:11434 —
                  each host runs the whole agent team for its documents
  model routing   UNIVERSAL_MODEL_HOSTS="llava:13b=http://gpu3:11434;..." pins a
                  model to dedicated host(s); with several workers, documents
                  then flow through the models like an assembly line (doc A at
                  the supervisor while doc B is at the verifier) with every
                  host still running one call at a time
  warm affinity   a call goes to the free host that last ran its model, so
                  the model and its cached system-prompt prefix are reused
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import ollama

from ollama_concurrency import ollama_semaphore

_held = threading.local()
REQUEST_TIMEOUT = float(os.environ.get("UNIVERSAL_OLLAMA_TIMEOUT", "240"))
LOCAL_NAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


def _normalize(host: str) -> str:
    host = host.strip().rstrip("/")
    if host and "://" not in host:
        host = f"http://{host}"
    return host


def _keep_alive(raw: str) -> Any:
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw or None            # "30m", "24h" — Ollama duration strings


def configured_hosts() -> list[str]:
    raw = os.environ.get("UNIVERSAL_OLLAMA_HOSTS") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    return [h for h in dict.fromkeys(_normalize(x) for x in raw.split(",")) if h]


def configured_model_hosts() -> dict[str, list[str]]:
    """UNIVERSAL_MODEL_HOSTS="llava:13b=http://a:11434|http://b:11434;qwen2.5:7b=http://c:11434" """
    out: dict[str, list[str]] = {}
    for item in os.environ.get("UNIVERSAL_MODEL_HOSTS", "").split(";"):
        if "=" not in item:
            continue
        model, hosts = item.split("=", 1)
        out[model.strip()] = [_normalize(h) for h in hosts.split("|") if h.strip()]
    return out


def is_local(host: str) -> bool:
    return (urlparse(host).hostname or "") in LOCAL_NAMES


@dataclass
class _Host:
    url: str
    slots: int
    client: Any
    busy: int = 0
    calls: int = 0
    busy_seconds: float = 0.0
    last_models: list[str] = field(default_factory=list)     # most recent first


class HostPool:
    """Hands out an Ollama client for a model, blocking until a host that
    serves the model has a free slot."""

    def __init__(self, hosts: list[str], model_hosts: Optional[dict[str, list[str]]] = None,
                 slots_per_host: int = 1, client_factory=None):
        if slots_per_host < 1:
            raise ValueError("slots_per_host must be >= 1")
        factory = client_factory or (lambda url: ollama.Client(host=url, timeout=REQUEST_TIMEOUT))
        self.model_hosts = {m: list(h) for m, h in (model_hosts or {}).items()}
        dedicated = [h for hs in self.model_hosts.values() for h in hs]
        urls = list(dict.fromkeys([*hosts, *dedicated]))
        self._hosts = {u: _Host(u, slots_per_host, factory(u)) for u in urls}
        # A host dedicated to some models doesn't also take the general pool's work.
        self.general = [u for u in hosts if u not in set(dedicated)] or list(hosts)
        self._cond = threading.Condition()

    def hosts_for(self, model: str) -> list[str]:
        return self.model_hosts.get(model) or self.general

    def _pick(self, model: str, only: Optional[str] = None) -> Optional[_Host]:
        candidates = [only] if only else self.hosts_for(model)
        free = [self._hosts[u] for u in candidates if self._hosts[u].busy < self._hosts[u].slots]
        if not free:
            return None
        # Warm first (model most recently run there), then the least-used host.
        def rank(h: _Host):
            warm = h.last_models.index(model) if model in h.last_models else len(h.last_models) + 1
            return warm, h.busy, h.calls
        return min(free, key=rank)

    @contextmanager
    def slot(self, model: str, timeout: Optional[float] = None,
             on: Optional[str] = None) -> Iterator[tuple[str, Any]]:
        held = getattr(_held, "hosts", [])
        reentrant = next((u for u in held if u in ([on] if on else self.hosts_for(model))), None)
        if reentrant:
            # The thread already holds a slot on a host serving this model:
            # reuse it rather than wait on itself.
            yield reentrant, self._hosts[reentrant].client
            return
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while (host := self._pick(model, on)) is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"no inference host free for {model}")
                self._cond.wait(remaining)
            host.busy += 1
            host.calls += 1
        started = time.monotonic()
        _held.hosts = held + [host.url]
        # The local GPU is shared with the existing idx pipeline: its process-wide
        # semaphore still caps the two engines together.
        local_gate = ollama_semaphore if is_local(host.url) else nullcontext()
        try:
            with local_gate:
                yield host.url, host.client
        finally:
            _held.hosts = held
            with self._cond:
                host.busy -= 1
                host.busy_seconds += time.monotonic() - started
                host.last_models = [model] + [m for m in host.last_models if m != model][:7]
                self._cond.notify_all()

    def snapshot(self) -> list[dict]:
        with self._cond:
            return [{"host": h.url, "slots": h.slots, "busy": h.busy, "calls": h.calls,
                     "busy_seconds": round(h.busy_seconds, 1), "recent_models": list(h.last_models),
                     "dedicated_to": sorted(m for m, hs in self.model_hosts.items() if h.url in hs)}
                    for h in self._hosts.values()]

    def clients(self) -> dict[str, Any]:
        return {u: h.client for u, h in self._hosts.items()}


KEEP_ALIVE = _keep_alive(os.environ.get("UNIVERSAL_KEEP_ALIVE", "-1"))
_pool: Optional[HostPool] = None
_pool_lock = threading.Lock()


def pool() -> HostPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = HostPool(configured_hosts(), configured_model_hosts(),
                             int(os.environ.get("UNIVERSAL_HOST_SLOTS", "1")))
        return _pool


def reset_pool(new: Optional[HostPool] = None) -> None:
    """Tests / reconfiguration."""
    global _pool
    with _pool_lock:
        _pool = new


def call(model: str, method: str, **kwargs: Any) -> Any:
    """client.<method>(model=..., keep_alive=KEEP_ALIVE, ...) on a free host
    serving the model. Every request carries keep_alive: Ollama resets a
    loaded model's keep-alive to its 5-minute default on any request without
    one, which would undo the hot-model setting."""
    with pool().slot(model) as (_, client):
        return getattr(client, method)(model=model, keep_alive=KEEP_ALIVE, **kwargs)


def show(model: str) -> Any:
    """Model metadata — read-only, loads nothing, so it needs no slot."""
    client = pool().clients()[pool().hosts_for(model)[0]]
    return client.show(model)


def default_workers() -> int:
    """Documents in flight by default: one per host, never more than the
    hosts can run at once. On a single GPU a second document only waits for
    the first one's model calls (and makes the models swap more)."""
    p = pool()
    general = sum(p._hosts[u].slots for u in p.general)
    dedicated = len({u for hs in p.model_hosts.values() for u in hs})
    return max(1, general + dedicated)


def team_models() -> list[str]:
    from .prompts import AGENT_ORDER, build_agents
    agents = build_agents()
    return list(dict.fromkeys(agents[a].model for a in AGENT_ORDER if a in agents))


def warm_up(models: Optional[list[str]] = None) -> list[dict]:
    """Load each team model on every host that serves it (an empty generate
    loads the weights and applies keep_alive).

    At the SAME num_ctx the agents use: Ollama reloads a model whose context
    size changes, so a warm-up at the default context would cost a second
    load on the first real call. Loaded in REVERSE agent order: a host too
    small for the whole team evicts the least recently used model first, so
    the first agents' models are the ones still resident when a document
    arrives."""
    from .tokens import context_limit
    report = []
    p = pool()
    for model in reversed(models or team_models()):
        for url in p.hosts_for(model):
            started = time.time()
            try:
                with p.slot(model, on=url) as (_, client):
                    client.generate(model=model, prompt="", keep_alive=KEEP_ALIVE,
                                    options={"num_ctx": context_limit(model)})
                report.append({"host": url, "model": model, "status": "loaded",
                               "seconds": round(time.time() - started, 1)})
            except Exception as e:
                report.append({"host": url, "model": model, "status": "failed", "error": str(e)[:200]})
    return report


def residency() -> list[dict]:
    """What each host has loaded, and whether it is fully on the GPU — a
    model partly offloaded to CPU is several times slower."""
    out = []
    for url, client in pool().clients().items():
        try:
            models = client.ps().models
        except Exception as e:
            out.append({"host": url, "reachable": False, "error": str(e)[:200]})
            continue
        loaded = []
        for m in models:
            size, vram = int(getattr(m, "size", 0) or 0), int(getattr(m, "size_vram", 0) or 0)
            loaded.append({"model": m.model, "size_gb": round(size / 1e9, 2), "gpu_percent":
                           round(100 * vram / size) if size else None,
                           "context": getattr(m, "context_length", None)})
        out.append({"host": url, "reachable": True, "loaded": loaded,
                    "partially_on_cpu": [l["model"] for l in loaded if (l["gpu_percent"] or 100) < 100]})
    return out

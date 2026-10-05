"""
Prometheus metrics for the universal engine (text exposition format, no
extra dependency). Scraped from GET /universal/metrics.

Quality and speed are measured side by side — a latency gain that raises
the review rate shows up on the same dashboard:

  universal_jobs_total{status,priority}            finished jobs
  universal_job_seconds{phase,priority}            queue wait / processing (histogram)
  universal_stage_seconds{stage}                   pipeline stages (timings_seconds)
  universal_agent_seconds{agent,model}             each agent call, all passes
  universal_model_load_seconds{model}              load paid inside a call (models swapping)
  universal_human_review_total{priority}           results needing a person
  universal_capture_total{grade}                   capture quality at upload
  universal_identity_signals_total{signal}         fraud / consistency flags
  universal_duplicates_joined_total                double submissions joined
  universal_exact_cache_hits_total                 opt-in exact-duplicate cache
  universal_queue_depth{priority}                  gauge, read from the queue
  universal_inference_busy{host}                   gauge, calls running per host
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable, Iterable

_BUCKETS = (0.5, 1, 2.5, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600)
_lock = threading.Lock()
_counters: dict[tuple, float] = defaultdict(float)
_histograms: dict[tuple, list] = {}
_HELP = {
    "universal_jobs_total": ("counter", "Finished universal extraction jobs"),
    "universal_human_review_total": ("counter", "Results that require human review"),
    "universal_capture_total": ("counter", "Uploaded captures by quality grade"),
    "universal_identity_signals_total": ("counter", "Identity-consistency signals raised"),
    "universal_duplicates_joined_total": ("counter", "Identical submissions joined to a running job"),
    "universal_exact_cache_hits_total": ("counter", "Results served from the exact-duplicate cache"),
    "universal_job_seconds": ("histogram", "Job queue wait and processing time"),
    "universal_stage_seconds": ("histogram", "Pipeline stage duration"),
    "universal_agent_seconds": ("histogram", "Agent call duration"),
    "universal_model_load_seconds": ("histogram", "Model (re)load time paid inside a call — swapping"),
}
_gauges: list[Callable[[], Iterable[tuple[str, dict, float]]]] = []


def _key(name: str, labels: dict) -> tuple:
    return (name, tuple(sorted((k, str(v)) for k, v in labels.items())))


def inc(name: str, amount: float = 1.0, **labels) -> None:
    with _lock:
        _counters[_key(name, labels)] += amount


def observe(name: str, value: float, **labels) -> None:
    if value is None:
        return
    with _lock:
        h = _histograms.setdefault(_key(name, labels), [[0] * len(_BUCKETS), 0.0, 0])
        for i, bound in enumerate(_BUCKETS):
            if value <= bound:
                h[0][i] += 1
        h[1] += float(value)
        h[2] += 1


def register_gauges(fn: Callable[[], Iterable[tuple[str, dict, float]]]) -> None:
    """fn() -> [(metric name, labels, value)], evaluated at scrape time."""
    _gauges.append(fn)


def reset() -> None:
    with _lock:
        _counters.clear()
        _histograms.clear()


def _labels(pairs: tuple, extra: str = "") -> str:
    body = ",".join(f'{k}="{_escape(v)}"' for k, v in pairs)
    body = ",".join(p for p in (body, extra) if p)
    return f"{{{body}}}" if body else ""


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def record_result(result: dict, priority: str) -> None:
    """Everything a finished extraction tells about speed and quality."""
    for stage, secs in (result.get("timings_seconds") or {}).items():
        observe("universal_stage_seconds", secs, stage=stage)
    for entry in result.get("agents") or []:
        observe("universal_agent_seconds", entry.get("duration_seconds"),
                agent=entry.get("agent", "?"), model=entry.get("model", "?"))
        for call in entry.get("pass_details") or [entry]:
            if call.get("load_seconds"):
                observe("universal_model_load_seconds", call["load_seconds"], model=entry.get("model", "?"))
    if result.get("human_review_required"):
        inc("universal_human_review_total", priority=priority)
    for s in ((result.get("verification") or {}).get("identity_signals") or {}).get("signals", []):
        inc("universal_identity_signals_total", signal=s["signal"])


def render() -> str:
    lines: list[str] = []
    with _lock:
        counters = dict(_counters)
        histograms = {k: (list(v[0]), v[1], v[2]) for k, v in _histograms.items()}
    seen: set[str] = set()

    def header(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        kind, text = _HELP.get(name, ("gauge", name))
        lines.append(f"# HELP {name} {text}")
        lines.append(f"# TYPE {name} {kind}")

    for (name, pairs), value in sorted(counters.items()):
        header(name)
        lines.append(f"{name}{_labels(pairs)} {value:g}")
    for (name, pairs), (buckets, total, count) in sorted(histograms.items()):
        header(name)
        for bound, n in zip(_BUCKETS, buckets):
            le = 'le="%g"' % bound
            lines.append(f"{name}_bucket{_labels(pairs, le)} {n}")
        inf = 'le="+Inf"'
        lines.append(f"{name}_bucket{_labels(pairs, inf)} {count}")
        lines.append(f"{name}_sum{_labels(pairs)} {total:g}")
        lines.append(f"{name}_count{_labels(pairs)} {count}")
    for fn in _gauges:
        try:
            for name, labels, value in fn():
                header(name)
                lines.append(f"{name}{_labels(_key(name, labels)[1])} {value:g}")
        except Exception:
            continue
    return "\n".join(lines) + "\n"

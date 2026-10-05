"""
Quality gate: a performance change is accepted only if extraction quality
does not drop.

    python tests/quality_gate.py --baseline <run_dir> --candidate <run_dir> [--truth truth.json]
                                 [--allow-regressions 0]

Run dirs are tests/run_id_batch.py outputs (one <image>.json per document)
from the same images, before and after the change.

  with --truth   field-by-field accuracy of both runs, every field the
                 candidate lost (right before, wrong now) and every one it
                 gained; FAILS if the candidate scores lower or loses more
                 fields than --allow-regressions
  always         the values that changed between runs, the human-review
                 rate, and the timing (total median/p95, per stage, per agent)

truth.json: {"<image file>": {"last_name": ..., "given": "FIRST MIDDLE", "date_of_birth":
"YYYY-MM-DD", ...}} — "given" is checked as given_names, first_name, middle_name and
full_name (given + last). It holds personal data: keep it outside the repository.

The agents sample at temperature 0.1, so two runs of unchanged code can
differ on a field or two. Measure that noise once (two baseline runs gated
against each other) and set --allow-regressions to it — never higher.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

VALUE_FIELDS = ["full_name", "first_name", "middle_name", "last_name", "full_name_local", "date_of_birth", "gender",
                "nationality", "document_number", "issue_date", "expiry_date", "issuing_country", "document_type"]


def _norm(field: str, value) -> str:
    v = "".join(c for c in unicodedata.normalize("NFKD", str(value or "")).upper() if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9]", "", v) if field == "document_number" else " ".join(re.sub(r"[^A-Z0-9 ]", " ", v).split())


def _checks(truth: dict) -> dict:
    checks = {k: v for k, v in truth.items() if k != "given"}
    if "given" in truth:
        words = truth["given"].split()
        checks.update(first_name=words[0], middle_name=" ".join(words[1:]), given_names=truth["given"],
                      full_name=f"{truth['given']} {truth.get('last_name', '')}".strip())
    return checks


def load_run(run_dir: Path) -> dict[str, dict]:
    return {p.name[:-5]: json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))}


def score(run: dict[str, dict], truth: dict[str, dict]) -> dict[tuple[str, str], bool]:
    out = {}
    for doc, t in truth.items():
        if doc not in run:
            continue
        data = run[doc].get("extracted_data") or {}
        for field, want in _checks(t).items():
            out[(doc, field)] = _norm(field, data.get(field)) == _norm(field, want)
    return out


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def timing(run: dict[str, dict]) -> dict:
    totals = [r.get("total_duration_seconds") or 0 for r in run.values()]
    stages, agents = {}, {}
    for r in run.values():
        for stage, secs in (r.get("timings_seconds") or {}).items():
            stages.setdefault(stage, []).append(secs)
        for a in r.get("agents") or []:
            agents.setdefault(a.get("agent", "?"), []).append(a.get("duration_seconds") or 0)
    med = lambda d: {k: round(statistics.median(v), 2) for k, v in d.items()}
    return {"median": round(statistics.median(totals), 2) if totals else 0, "p95": round(_pct(totals, 0.95), 2),
            "stages": med(stages), "agents": med(agents),
            "reviews": sum(1 for r in run.values() if r.get("human_review_required"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--truth", type=Path)
    ap.add_argument("--allow-regressions", type=int, default=0)
    args = ap.parse_args()

    base, cand = load_run(args.baseline), load_run(args.candidate)
    common = sorted(set(base) & set(cand))
    missing = sorted(set(base) - set(cand))
    base, cand = {d: base[d] for d in common}, {d: cand[d] for d in common}
    print(f"documents compared: {len(common)}" + (f"   MISSING from candidate: {missing}" if missing else ""))

    failed = bool(missing)
    if args.truth:
        truth = json.loads(args.truth.read_text())
        sb, sc = score(base, truth), score(cand, truth)
        keys = sorted(set(sb) & set(sc))
        nb, nc = sum(sb[k] for k in keys), sum(sc[k] for k in keys)
        lost = [k for k in keys if sb[k] and not sc[k]]
        gained = [k for k in keys if sc[k] and not sb[k]]
        print(f"\nACCURACY  baseline {nb}/{len(keys)} = {100 * nb / max(len(keys), 1):.1f}%   "
              f"candidate {nc}/{len(keys)} = {100 * nc / max(len(keys), 1):.1f}%")
        for doc, field in lost:
            print(f"  LOST    {doc:34} {field:14} {cand[doc]['extracted_data'].get(field)!r} "
                  f"(was {base[doc]['extracted_data'].get(field)!r})")
        for doc, field in gained:
            print(f"  gained  {doc:34} {field:14} {cand[doc]['extracted_data'].get(field)!r}")
        if nc < nb or len(lost) > args.allow_regressions:
            failed = True

    changed = []
    for doc in common:
        b, c = base[doc].get("extracted_data") or {}, cand[doc].get("extracted_data") or {}
        for f in VALUE_FIELDS:
            if _norm(f, b.get(f)) != _norm(f, c.get(f)):
                changed.append((doc, f, b.get(f), c.get(f)))
    print(f"\nVALUES CHANGED: {len(changed)}")
    for doc, f, bv, cv in changed:
        print(f"  {doc:34} {f:16} {bv!r} -> {cv!r}")

    tb, tc = timing(base), timing(cand)
    print(f"\nHUMAN REVIEW  baseline {tb['reviews']}/{len(common)}   candidate {tc['reviews']}/{len(common)}")
    print(f"\nTIME (s)      baseline median {tb['median']} p95 {tb['p95']}   "
          f"candidate median {tc['median']} p95 {tc['p95']}")
    for kind in ("stages", "agents"):
        for k in sorted(set(tb[kind]) | set(tc[kind])):
            print(f"  {kind[:-1]:6} {k:18} {tb[kind].get(k, '-'):>8} -> {tc[kind].get(k, '-'):>8}")

    print("\nGATE:", "FAIL — quality dropped; the change is rejected" if failed else "PASS — no quality loss")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

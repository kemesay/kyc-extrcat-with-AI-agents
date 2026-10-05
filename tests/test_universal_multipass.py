"""Prompt packing and pass planning — token accounting is stubbed so these run without Ollama."""

import pytest

import universal_id_extraction_prompts as P
from universal import multipass as M
from universal import tokens as T
from universal.prompts import EXTRACTION_FIELDS, SCANNER_FIELDS


@pytest.fixture(autouse=True)
def offline_tokens(monkeypatch):
    limits = {"small:1": 2048, "mid:1": 4096, "big:1": 16384}
    monkeypatch.setattr(T, "context_limit", lambda m: limits[m])
    monkeypatch.setattr(T, "estimate", lambda m, text: len(text.encode("utf-8")) // 3 if text else 0)
    monkeypatch.setattr(T, "image_tokens_for", lambda m, w, h: 700)
    monkeypatch.setattr(T, "supports_system", lambda m: m != "small:1")
    monkeypatch.setattr(T, "exact_chat_tokens", lambda m, s, u, c: (len(s.encode()) + len(u.encode())) // 3 + 20)
    monkeypatch.setattr(M, "_image_size", lambda b64: (1400, 900))


ETH = M.Hints.from_signals(["Ethiopic", "Amharic"], ["ETH"]).with_doc_types(["national_id"])


def _plan(model, user, fields, passes="auto", hints=ETH, values=None):
    return M.plan_agent(model, P.SYSTEM_PROMPT, user, fields, values or {}, "", hints, ["img"], passes)


def test_large_lists_split_into_items_with_lead():
    blocks = M.split_blocks(P.SYSTEM_PROMPT, "system")
    # One optional block per script, so the packer keeps the document's own scripts.
    ethiopic_item = next(b for b in blocks if b.text.strip().startswith("Ethiopic/Ge'ez:"))
    latin_item = next(b for b in blocks if b.text.strip().startswith("Latin:"))
    assert not ethiopic_item.mandatory and not latin_item.mandatory
    assert "Afaan Oromo (Qubee)" in latin_item.text and "Amharic" in ethiopic_item.text


def test_rules_and_output_format_always_packed():
    (plan,) = _plan("mid:1", P.FAST_SCANNER_PROMPT, SCANNER_FIELDS, passes="1")
    assert "YOUR EXTRACTION RULES" in plan.system and "OUTPUT FORMAT" in plan.system


def test_relevant_script_rules_preferred():
    (plan,) = _plan("mid:1", P.FAST_SCANNER_PROMPT, SCANNER_FIELDS, passes="1")
    assert "Ethiopic Documents" in plan.system


def test_every_pass_fits_the_context():
    for model, user, fields in [("small:1", P.FAST_SCANNER_PROMPT, SCANNER_FIELDS),
                                ("mid:1", P.ACCURACY_VERIFICATION_PROMPT, EXTRACTION_FIELDS)]:
        for plan in _plan(model, user, fields, values={"previous_extraction": {"full_name": "A B"}}):
            assert plan.prompt_tokens_est + plan.num_predict <= T.context_limit(model)


def test_auto_uses_one_pass_when_everything_fits():
    assert len(_plan("big:1", P.FIELD_EXTRACTION_PROMPT, EXTRACTION_FIELDS)) == 1


def test_forced_split_uses_disjoint_field_groups():
    plans = _plan("small:1", P.FAST_SCANNER_PROMPT, SCANNER_FIELDS, passes="3")
    assert len(plans) == 3
    seen = [f for p in plans for f in p.fields]
    assert sorted(seen) == sorted(SCANNER_FIELDS) and len(seen) == len(set(seen))


def test_auto_keeps_one_packed_call_when_splitting_would_stay_poorly_informed():
    forced = _plan("small:1", P.FAST_SCANNER_PROMPT, SCANNER_FIELDS, passes="3")
    assert min(p.coverage for p in forced) < M.AUTO_MIN_COVERAGE
    assert len(_plan("small:1", P.FAST_SCANNER_PROMPT, SCANNER_FIELDS)) == 1


def test_auto_splits_when_every_pass_becomes_well_informed(monkeypatch):
    # Too small for one pass (58 % coverage) while two passes carry ~88 % each.
    monkeypatch.setattr(T, "context_limit", lambda m: 3800)
    plans = _plan("mid:1", P.ACCURACY_VERIFICATION_PROMPT, EXTRACTION_FIELDS,
                  values={"previous_extraction": {"full_name": "A B"}})
    assert len(plans) > 1 and min(p.coverage for p in plans) >= M.AUTO_MIN_COVERAGE


def test_forced_pass_count_respected():
    assert len(_plan("big:1", P.FIELD_EXTRACTION_PROMPT, EXTRACTION_FIELDS, passes="3")) == 3


def test_system_folded_into_user_when_template_has_no_system_slot():
    (plan,) = _plan("small:1", P.FAST_SCANNER_PROMPT, SCANNER_FIELDS, passes="1")
    assert plan.system == "" and plan.user.startswith("You are UIDX-Agent")


def test_pass_prompt_asks_only_for_its_fields():
    plans = _plan("big:1", P.FIELD_EXTRACTION_PROMPT, EXTRACTION_FIELDS, passes="2")
    for plan in plans:
        others = set(EXTRACTION_FIELDS) - set(plan.fields)
        template = plan.user[plan.user.rfind("{"):]
        assert all(f'"{f}"' not in template for f in others)


def test_placeholder_rendered_with_pass_subset():
    plans = _plan("big:1", P.ACCURACY_VERIFICATION_PROMPT, EXTRACTION_FIELDS, passes="2",
                  values={"previous_extraction": {"full_name": "ANNA", "address": "MAIN ST"}})
    identity = next(p for p in plans if "full_name" in p.fields)
    assert "ANNA" in identity.user and "MAIN ST" not in identity.user


def test_partition_balanced_and_ordered():
    buckets = M.partition(EXTRACTION_FIELDS, 3)
    assert len(buckets) == 3 and [f for b in buckets for f in b] == [
        f for _, g in M.FIELD_GROUPS for f in g if f in EXTRACTION_FIELDS]


def test_empty_ollama_response_is_retried_then_raised(monkeypatch):
    from universal import llm
    calls = []
    empty = {"message": {"content": ""}, "prompt_eval_count": None, "eval_count": 0}
    monkeypatch.setattr(llm, "_chat", lambda *a, **k: calls.append(1) or empty)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="empty response"):
        llm.chat_json("m", "", "hi", fields=["a"])
    assert len(calls) == 2


def _reference_plan(model, system, user, fields, values, context, hints, images):
    """The previous auto rule: every candidate plan exact-checked."""
    ctx = T.context_limit(model)
    image_tok = sum(T.image_tokens_for(model, *M._image_size(i)) for i in images)
    sb, ub = M.split_blocks(system, "system"), M.split_blocks(user, "user")
    plans_for = lambda n: [M._plan_pass(model, ctx, image_tok, b, sb, ub, values, context, hints)
                           for b in M.partition(fields, n)]
    single = plans_for(1)
    if single[0].coverage >= 0.999:
        return single
    for n in range(2, M.MAX_PASSES + 1):
        plans = plans_for(n)
        if min(p.coverage for p in plans) >= M.AUTO_MIN_COVERAGE:
            return plans
    return single


@pytest.mark.parametrize("model", ["small:1", "mid:1", "big:1"])
@pytest.mark.parametrize("inflation", [1.0, 1.08, 1.25])
def test_auto_plan_identical_with_fewer_exact_probes(monkeypatch, model, inflation):
    # Exact counts above the estimate make the exact check drop blocks — the
    # case where skipping a probe could matter if the shortcut were unsound.
    probes = []
    def exact(m, s, u, c):
        probes.append(1)
        return int(((len(s.encode()) + len(u.encode())) // 3 + 20) * inflation)
    monkeypatch.setattr(T, "exact_chat_tokens", exact)
    cases = [(P.FAST_SCANNER_PROMPT, SCANNER_FIELDS), (P.FIELD_EXTRACTION_PROMPT, EXTRACTION_FIELDS),
             (P.ACCURACY_VERIFICATION_PROMPT, EXTRACTION_FIELDS), (P.LAYOUT_ANALYSIS_PROMPT, EXTRACTION_FIELDS)]
    for user, fields in cases:
        probes.clear()
        ref = _reference_plan(model, P.SYSTEM_PROMPT, user, fields, {}, "", ETH, ["img"])
        ref_probes = len(probes)
        probes.clear()
        new = M.plan_agent(model, P.SYSTEM_PROMPT, user, fields, {}, "", ETH, ["img"], "auto")
        assert [(p.fields, p.system, p.user, p.num_predict) for p in new] == \
               [(p.fields, p.system, p.user, p.num_predict) for p in ref]
        assert len(probes) <= ref_probes


def test_fewest_drops_matches_one_at_a_time_loop():
    import random
    rng = random.Random(3)
    for _ in range(2000):
        n = rng.randint(1, 40)
        threshold = rng.randint(1, n + 3)                  # fits(k) for k >= threshold (monotone)
        fits = lambda k: k >= threshold
        linear = next((k for k in range(1, n + 1) if fits(k)), n)
        for guess in (1, n, rng.randint(1, n), threshold, threshold - 1, threshold + 2):
            assert M.fewest_drops(n, fits, guess) == linear


@pytest.mark.parametrize("model", ["small:1", "mid:1"])
@pytest.mark.parametrize("inflation", [1.05, 1.15, 1.3])
def test_guided_drop_search_gives_the_identical_prompt_with_fewer_counts(monkeypatch, model, inflation):
    counted = []
    def exact(m, s, u, c):
        counted.append(1)
        return int(((len(s.encode()) + len(u.encode())) // 3 + 20) * inflation)
    monkeypatch.setattr(T, "exact_chat_tokens", exact)
    ctx = T.context_limit(model)
    sb, ub = M.split_blocks(P.SYSTEM_PROMPT, "system"), M.split_blocks(P.ACCURACY_VERIFICATION_PROMPT, "user")
    plan = lambda: M._plan_pass(model, ctx, 700, EXTRACTION_FIELDS, sb, ub, {}, "", ETH)

    guided = plan()
    guided_counts = len(counted)
    # The previous behaviour: re-count after every single dropped block.
    linear = lambda n, fits, guess: next((k for k in range(1, n + 1) if fits(k)), n)
    monkeypatch.setattr(M, "fewest_drops", linear)
    counted.clear()
    one_at_a_time = plan()
    assert (guided.system, guided.user, guided.num_predict, guided.coverage, guided.blocks_omitted) == \
           (one_at_a_time.system, one_at_a_time.user, one_at_a_time.num_predict, one_at_a_time.coverage,
            one_at_a_time.blocks_omitted)
    assert guided_counts <= len(counted)


def test_seed_is_sent_only_when_configured(monkeypatch):
    from universal import llm
    sent = []
    ok = {"prompt_eval_count": 5, "message": {"content": '{"a": "x"}'}}
    monkeypatch.setattr(llm, "_chat", lambda m, msgs, fmt, options, **k: sent.append(options) or ok)
    monkeypatch.setattr(llm, "SEED", None)
    llm.chat_json("m", "", "u", fields=["a"])
    monkeypatch.setattr(llm, "SEED", 42)
    llm.chat_json("m", "", "u", fields=["a"])
    assert "seed" not in sent[0] and sent[1]["seed"] == 42


# ── token calibration and context-overflow recovery ─────────────────────

_REAL_EXACT = T.exact_chat_tokens          # captured at import, before the stubbing fixture runs

OVERFLOW_MSG = ('{"error":{"code":400,"message":"request (3035 tokens) exceeds the available context size '
                '(2048 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":3035}}')


def test_calibration_sample_fits_the_models_context(monkeypatch):
    monkeypatch.setattr(T, "context_limit", lambda m: 2048)
    sample = T._calibration_sample("small:1")
    assert 0 < len(sample.encode()) <= 2048 * 2 and P.SYSTEM_PROMPT.startswith(sample)


def test_implausible_calibration_is_not_trusted(monkeypatch):
    T.bytes_per_token.cache_clear()
    monkeypatch.setattr(T, "context_limit", lambda m: 2048)
    monkeypatch.setattr(T, "_count_raw", lambda m, text: len(text.encode()) // 7)     # a truncated count
    try:
        assert T.bytes_per_token("probe:1") == T._FALLBACK_BYTES_PER_TOKEN
    finally:
        T.bytes_per_token.cache_clear()


def test_overflow_error_gives_the_true_token_count(monkeypatch):
    assert T.overflow_tokens(RuntimeError(OVERFLOW_MSG)) == 3035
    assert T.overflow_tokens(RuntimeError("connection refused")) is None
    def refuse(*a, **k):
        raise RuntimeError(OVERFLOW_MSG)
    monkeypatch.setattr(T.inference, "call", refuse)
    assert _REAL_EXACT.__wrapped__("m", "sys", "user", 2048) == 3035


def test_agent_is_replanned_and_retried_after_a_context_overflow(monkeypatch):
    calls = []
    def chat(model, system, user, **kw):
        calls.append(len(system) + len(user))
        if len(calls) == 1:
            raise RuntimeError(OVERFLOW_MSG)
        return {f: "x" for f in kw["fields"]}, {"prompt_tokens": 900, "complete_json": True}
    monkeypatch.setattr(M, "chat_json", chat)
    monkeypatch.setattr(T, "_correction", {})
    import math
    monkeypatch.setattr(T, "estimate", lambda m, t: math.ceil(len(t.encode()) / 3 * T._correction.get(m, 1.0))
                        if t else 0)
    data, log = M.run_agent("small:1", P.SYSTEM_PROMPT_COMPACT, P.FAST_SCANNER_PROMPT, SCANNER_FIELDS, {}, "",
                            ETH, ["img"], "1")
    assert log["replanned_after_overflow"]["ollama_counted"] == 3035 and data
    assert T._correction["small:1"] > 1.0          # estimates scaled up for the rest of the process
    assert calls[1] < calls[0]                      # the second plan is smaller

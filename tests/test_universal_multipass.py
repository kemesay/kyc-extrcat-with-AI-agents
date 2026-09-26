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
    ethiopic_item = next(b for b in blocks if b.text.strip().startswith("Ethiopic:"))
    assert ethiopic_item.lead.startswith("TIER 2") and not ethiopic_item.mandatory


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
    monkeypatch.setattr(T, "context_limit", lambda m: 3300)     # just too small for one pass
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

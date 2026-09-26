"""Reconciliation, verification and grounding — no Ollama needed."""
import pytest

import universal_id_extraction_prompts as P
from universal.grounding import build_evidence, ground_fields
from universal.prompts import EXTRACTION_FIELDS, build_agents, compact_system_prompt, render
from universal.reconcile import (
    Context, FieldResult, apply_verified, consensus, normalize_value, settle_name_order, settle_names,
)
from universal.verify import verify_mrz_lines

TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]


def _fields(**values) -> dict:
    fields = {name: FieldResult() for name in EXTRACTION_FIELDS}
    for name, (value, agents) in values.items():
        fields[name] = FieldResult(value=value, confidence=0.6, source="consensus", agreeing_agents=list(agents))
    return fields


# ── prompts ────────────────────────────────────────────────────────────────

def test_every_agent_prompt_renders_despite_json_braces():
    for spec in build_agents().values():
        text = render(spec.user_template, previous_extraction="{}", verified_extraction="{}", agent_outputs="{}")
        assert "{previous_extraction}" not in text and "{verified_extraction}" not in text


def test_every_uidx_agent_draws_on_full_system_prompt():
    agents = build_agents()
    for name in ("fast_scanner", "field_extractor", "accuracy_verifier", "layout_analyzer"):
        assert agents[name].system == P.SYSTEM_PROMPT            # packer's knowledge base
    assert agents["fast_scanner"].fallback_system == compact_system_prompt()
    assert len(compact_system_prompt()) < len(P.SYSTEM_PROMPT) / 3


def test_td3_check_digit_positions_corrected():
    template = build_agents()["accuracy_verifier"].user_template
    assert "line 2, character 10 in TD3" in template and "position 15 in TD3" not in template


# ── consensus ──────────────────────────────────────────────────────────────

def _agent_values(name, **by_agent):
    ctx = Context()
    return {a: (*normalize_value(name, v, ctx), v) for a, v in by_agent.items()}


def test_majority_and_agreement_confidence():
    values = _agent_values("document_number", fast_scanner="A123456", field_extractor="A123456",
                           accuracy_verifier="A123456", layout_analyzer="A128456")
    fr = consensus("document_number", values, None, 4)
    assert fr.value == "A123456" and fr.alternatives == ["A128456"]
    assert fr.confidence > consensus("document_number", {"field_extractor": values["field_extractor"]}, None, 4).confidence


def test_supervisor_value_no_agent_produced_is_rejected():
    values = _agent_values("full_name", field_extractor="ANNA ERIKSSON", layout_analyzer="ANNA ERIKSSON")
    fr = consensus("full_name", values, ("JOHN DOE", set(), "JOHN DOE"), 2)
    assert fr.value == "ANNA ERIKSSON" and "supervisor_value_rejected" in fr.flags


def test_name_order_and_punctuation_are_agreement():
    values = _agent_values("full_name", field_extractor="ERIKSSON, ANNA MARIA", layout_analyzer="ANNA MARIA ERIKSSON")
    assert consensus("full_name", values, None, 2).agreeing_agents == ["field_extractor", "layout_analyzer"]


def test_script_rule_keeps_extractor_native_script():
    values = _agent_values("full_name_local", field_extractor="አበበ ከበደ",
                           accuracy_verifier="አብር ከቤደ", layout_analyzer="አብር ከቤደ")
    fr = consensus("full_name_local", values, None, 3)
    assert fr.value == "አበበ ከበደ" and fr.source == "script_rule"


def test_non_mrz_text_in_mrz_field_dropped():
    assert normalize_value("mrz_line_1", "11010519491231002X", Context()) == ("", {"not_mrz_shaped"})


# ── verified sources ───────────────────────────────────────────────────────

def test_mrz_per_field_trust():
    src = verify_mrz_lines([TD3[0], TD3[1][:-1] + "1"])        # composite digit wrong
    assert src.details["valid"] and not src.details["all_checks_passed"]
    assert src.confidence_for("document_number") == 0.99
    assert src.confidence_for("gender") < 0.9                  # unprotected field, imperfect read


def test_mrz_overrides_disagreeing_number_and_requests_review():
    fields = _fields(document_number=("L898902G3", ["field_extractor"]), full_name=("ANNA MARIA ERIKSSON", ["field_extractor"]))
    ctx = Context()
    apply_verified(fields, verify_mrz_lines(TD3), ctx)
    assert fields["document_number"].value == "L898902C3"
    assert fields["document_number"].alternatives == ["L898902G3"]
    assert ctx.review_reasons and "mrz" in fields["full_name"].verified_by


def test_mrz_never_replaces_visual_name_with_transliteration():
    fields = _fields(full_name=("ANNA MÄRIA ERIKSSON", ["field_extractor"]))
    apply_verified(fields, verify_mrz_lines(TD3), Context())
    assert fields["full_name"].value == "ANNA MÄRIA ERIKSSON"


# ── names ──────────────────────────────────────────────────────────────────

def test_cjk_family_name_first():
    fields = _fields(full_name_local=("王芳", ["field_extractor"]), first_name=("王", ["field_extractor"]),
                     last_name=("芳", ["field_extractor"]))
    settle_name_order(fields, Context())
    assert (fields["last_name"].value, fields["first_name"].value) == ("王", "芳")


def test_parenthesised_transliteration_split():
    fields = _fields(full_name=("መሳይ ታደሰ (MESAY TADESSE)", ["field_extractor"]))
    settle_names(fields)
    assert fields["full_name_local"].value == "መሳይ ታደሰ"
    assert fields["full_name_latin"].value == "MESAY TADESSE"


# ── OCR grounding ──────────────────────────────────────────────────────────

ETH_OCR = """ETHIOPIAN NATIONAL ID CARD
Full Name ABEBE KEBEDE TESFAYE
Date of Birth 20/01/1990 | 12/05/1982
Date of Expiry 30/06/203023/10/2022
FAN 1234 5678 9012 3456"""


def test_ethiopian_calendar_misread_corrected_by_arithmetic():
    ev = build_evidence(ETH_OCR, ["test"], "ETH")
    fields = _fields(date_of_birth=("1982-05-12", ["field_extractor", "layout_analyzer"]),
                     expiry_date=("2022-10-23", ["field_extractor"]))
    report = ground_fields(fields, ev, Context(country="ETH"), 4)
    assert fields["date_of_birth"].value == "1990-01-20"
    assert fields["expiry_date"].value == "2030-06-30"          # glued OCR dates repaired
    assert fields["date_of_birth_orig"].value == "12/05/1982"
    assert len(report["calendar_corrections"]) == 2


def test_truncated_number_completed_from_ocr():
    fields = _fields(document_number=("1234", ["field_extractor"]))
    ground_fields(fields, build_evidence(ETH_OCR, ["test"], "ETH"), Context(country="ETH"), 4)
    assert fields["document_number"].value == "1234567890123456"


def test_ocr_arbitrates_between_agent_candidates():
    fields = _fields(full_name=("ABEBE KEBDE TESFAYE", ["accuracy_verifier", "layout_analyzer"]))
    fields["full_name"].alternatives = ["ABEBE KEBEDE TESFAYE"]
    ground_fields(fields, build_evidence(ETH_OCR, ["test"], "ETH"), Context(country="ETH"), 4)
    assert fields["full_name"].value == "ABEBE KEBEDE TESFAYE"
    assert "ocr_selected_alternative" in fields["full_name"].flags


def test_chained_hallucination_dropped():
    fields = _fields(issue_date=("2019-05-20", ["accuracy_verifier", "layout_analyzer"]),
                     blood_type=("O+", ["accuracy_verifier"]))
    ctx = Context(country="ETH")
    ground_fields(fields, build_evidence(ETH_OCR, ["test"], "ETH"), ctx, 4)
    assert fields["issue_date"].value == "" and fields["issue_date"].alternatives == ["2019-05-20"]
    assert fields["blood_type"].value == ""


def test_critical_field_not_dropped_but_reviewed():
    fields = _fields(document_number=("99887766", ["accuracy_verifier"]))
    ctx = Context(country="ETH")
    ground_fields(fields, build_evidence(ETH_OCR, ["test"], "ETH"), ctx, 4)
    assert fields["document_number"].value == "99887766"
    assert any("document_number" in r for r in ctx.review_reasons)


# ── regressions found in live runs ─────────────────────────────────────────

def test_hallucinated_calendar_never_overrides_verified_dob():
    from universal.plausibility import check_dual_calendar
    fields = _fields(date_of_birth=("1974-08-12", ["field_extractor"]),
                     date_of_birth_orig=("15/AUG/1974 EC", ["field_extractor"]))
    fields["date_of_birth"].verified_by = ["mrz"]
    report = check_dual_calendar(fields, Context(country="UTO"))
    assert fields["date_of_birth"].value == "1974-08-12"
    assert "implausible_calendar_for_country" in fields["date_of_birth_orig"].flags
    assert "not used by UTO" in report["note"]


def test_verified_dob_outranks_plausible_calendar_orig():
    from universal.plausibility import check_dual_calendar
    fields = _fields(date_of_birth=("1990-03-21", ["field_extractor"]),
                     date_of_birth_orig=("12/05/1982", ["field_extractor"]))
    fields["date_of_birth"].verified_by = ["barcode"]
    check_dual_calendar(fields, Context(country="ETH"))
    assert fields["date_of_birth"].value == "1990-03-21"
    assert "inconsistent_with_verified_date_of_birth" in fields["date_of_birth_orig"].flags


def test_mrz_alpha_fields_canonicalised():
    src = verify_mrz_lines([TD3[0], TD3[1].replace("UTO", "UT0")])
    assert src.fields["mrz_line_2"] == TD3[1]


def test_document_number_with_check_digit_is_agreement():
    fields = _fields(document_number=("L898902C36", ["field_extractor"]))
    ctx = Context()
    apply_verified(fields, verify_mrz_lines(TD3), ctx)
    assert fields["document_number"].value == "L898902C3" and not ctx.review_reasons


def test_miscopied_latin_name_repaired_from_agreeing_full_name():
    fields = _fields(full_name=("ANNA MARIA ERIKSSON", ["field_extractor"]),
                     full_name_latin=("UTOERIKSSON ANNA MARIA", ["field_extractor"]))
    ctx = Context()
    apply_verified(fields, verify_mrz_lines(TD3), ctx)
    assert fields["full_name_latin"].value == "ANNA MARIA ERIKSSON" and not ctx.review_reasons


def test_mrz_copied_name_replaced_with_mrz_name():
    fields = _fields(full_name=("UTOERIKSSON> ANNA>MARIA", ["field_extractor"]))
    fields["full_name"].value = normalize_value("full_name", fields["full_name"].value, Context())[0]
    ctx = Context()
    apply_verified(fields, verify_mrz_lines(TD3), ctx)
    assert fields["full_name"].value == "ANNA MARIA ERIKSSON" and not ctx.review_reasons


def test_ocr_spelling_correction_keeps_name_structure():
    fields = _fields(last_name=("KEBDE TESFAYE", ["accuracy_verifier", "layout_analyzer"]))
    fields["last_name"].alternatives = ["KEBEDE"]
    ground_fields(fields, build_evidence(ETH_OCR, ["test"], "ETH"), Context(country="ETH"), 4)
    assert fields["last_name"].value == "KEBEDE TESFAYE"


def test_issue_date_copied_from_dob_dropped():
    from universal.plausibility import check_plausibility
    fields = _fields(date_of_birth=("1990-01-20", ["field_extractor"]), issue_date=("1990-01-20", ["layout_analyzer"]))
    check_plausibility(fields, Context())
    assert fields["issue_date"].value == ""


def test_calendar_system_reflects_detected_pairs():
    fields = _fields(calendar_system=("Gregorian", ["field_extractor"]))
    ground_fields(fields, build_evidence(ETH_OCR, ["test"], "ETH"), Context(country="ETH"), 4)
    assert fields["calendar_system"].value == "Gregorian + Ethiopian"


def test_label_only_document_number_replaced_by_primary_scheme_number():
    from universal.id_numbers import check_numbers
    from universal.reconcile import apply_id_number_checks
    fields = _fields(personal_number=("1234 5678 9012 3456", ["field_extractor"]))
    assert normalize_value("document_number", "FAN", Context())[0] == ""
    ctx = Context(country="ETH")
    checks = check_numbers("ETH", {"document_number": "", "personal_number": fields["personal_number"].value})
    apply_id_number_checks(fields, checks, ctx)
    assert fields["document_number"].value == "1234 5678 9012 3456" and fields["personal_number"].value == ""


def test_printed_latin_name_spell_corrected_even_with_local_script_name():
    fields = _fields(full_name_local=("አበበ ከበደ ተስፋዬ", ["field_extractor"]),
                     full_name_latin=("ABEBE KEBDE TESFAYE", ["accuracy_verifier"]))
    ground_fields(fields, build_evidence(ETH_OCR, ["test"], "ETH"), Context(country="ETH"), 4)
    assert fields["full_name_latin"].value == "ABEBE KEBEDE TESFAYE"


def test_impossible_unprinted_issue_date_dropped_not_escalated():
    from universal.plausibility import check_plausibility
    fields = _fields(date_of_birth=("1990-01-20", ["field_extractor"]), issue_date=("2031-05-20", ["field_extractor"]),
                     expiry_date=("2030-06-30", ["field_extractor"]))
    fields["issue_date"].flags.append("not_found_in_ocr")
    ctx = Context()
    check_plausibility(fields, ctx)
    assert fields["issue_date"].value == "" and not ctx.review_reasons


def test_impossible_but_printed_issue_date_still_reviewed():
    from universal.plausibility import check_plausibility
    fields = _fields(issue_date=("2031-05-20", ["field_extractor"]), expiry_date=("2030-06-30", ["field_extractor"]))
    ctx = Context()
    check_plausibility(fields, ctx)
    assert fields["issue_date"].value == "2031-05-20" and ctx.review_reasons


def test_ocr_evidence_reads_european_date_styles():
    ev = build_evidence("29 OKT/OCT 1972\n15 03 1992\n04 AUG/AUG 30\nDOB 06MAR1994", ["t"], "NLD")
    for iso in ("1972-10-29", "1992-03-15", "2030-08-04", "1994-03-06"):
        assert iso in ev.gregorian_dates


def test_mislabelled_ec_original_dropped_not_converted():
    from universal.plausibility import check_dual_calendar
    fields = _fields(date_of_birth=("1984-06-05", ["field_extractor"]),
                     date_of_birth_orig=("05 JUN 1984 EC", ["field_extractor"]))
    check_dual_calendar(fields, Context(country=""))
    assert fields["date_of_birth"].value == "1984-06-05" and fields["date_of_birth_orig"].value == ""


def test_prompt_example_echo_dropped_only_for_personal_fields():
    ctx = Context()
    assert normalize_value("date_of_birth_orig", "15/Meskerem/1982 EC", ctx)[0] == ""
    assert normalize_value("full_name", "MESAY TADESSE", ctx)[0] == ""
    assert normalize_value("document_type", "Passport", ctx)[0] == "passport"


def test_mrz_completes_visual_name_missing_surname():
    fields = _fields(full_name=("ANNA MARIA", ["field_extractor"]))
    ctx = Context()
    apply_verified(fields, verify_mrz_lines(TD3), ctx)
    assert fields["full_name"].value == "ANNA MARIA ERIKSSON" and not ctx.review_reasons


def test_spelling_corrected_inside_glued_ocr_token():
    fields = _fields(full_name=("HANA TADESSE BEKEIE", ["field_extractor"]))
    ground_fields(fields, build_evidence("Full Name\nHANA TADESSEBEKELE\nDOB 2/6/1995 " * 3, ["t"], "ETH"),
                  Context(country="ETH"), 1)
    assert fields["full_name"].value == "HANA TADESSE BEKELE"


def test_clean_mrz_settles_small_visual_misspelling():
    fields = _fields(last_name=("ERIKSON", ["field_extractor"]))     # one letter dropped
    ctx = Context()
    apply_verified(fields, verify_mrz_lines(TD3), ctx)
    assert fields["last_name"].value == "ERIKSSON" and not ctx.review_reasons


def test_zero_mrz_personal_number_is_filler():
    src = verify_mrz_lines(["P<ETHDOE<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                            "EP12345671ETH9001014F30010110<<<<<<<<<<<<<<04"])
    assert "personal_number" not in src.fields


# ── MRZ straight from OCR text ─────────────────────────────────────────────

TD1 = ["I<UTOD231458907<<<<<<<<<<<<<<<", "7408122F1204159UTO<<<<<<<<<<<6", "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]
TD2 = ["I<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<", "D231458907UTO7408122F1204159<<<<<<<6"]


@pytest.mark.parametrize("lines, fmt", [(TD1, "TD1"), (TD2, "TD2"), (TD3, "TD3")])
def test_mrz_read_from_ocr_text_all_formats(lines, fmt):
    from universal.verify import mrz_from_ocr_text
    src, seen = mrz_from_ocr_text("header text\n" + "\n".join(lines) + "\nfooter")
    assert seen and src.details["format"] == fmt and src.details["all_checks_passed"]
    assert src.fields["document_number"] in ("D23145890", "L898902C3")


def test_mrz_lines_combined_across_ocr_engines():
    from universal.verify import mrz_from_ocr_text
    rapid = ["<UOD231458907<<<<<<<<<<<<<<<", TD1[1], TD1[2]]          # engine 1 lost 'I<UT'
    tess = [TD1[0], "", "7408122F120415 9UTO<<<<<<<<<<<6", ""]          # engine 2 split a line
    src, _ = mrz_from_ocr_text("\n".join(rapid + tess))
    assert src.details["format"] == "TD1" and src.details["all_checks_passed"]


def test_disputed_mrz_name_line_never_overrides_visual_spelling():
    from universal.verify import mrz_from_ocr_text
    text = "\n".join(TD1 + [TD1[0], TD1[1], "ERIKSSON<<ANNA<MARTA<<<<<<<<<<"])
    src, _ = mrz_from_ocr_text(text)
    assert src.details.get("name_line_uncertain")
    fields = _fields(first_name=("ANNA MARIA", ["field_extractor"]))
    apply_verified(fields, src, Context())
    assert fields["first_name"].value == "ANNA MARIA"


def test_card_back_surname_in_first_name_replaced_by_supported_mrz_reading():
    from universal.verify import mrz_from_ocr_text
    src, _ = mrz_from_ocr_text("\n".join(TD1 + [TD1[0], TD1[1], "ERIKSSON<<ANNA<MARTA<<<<<<<<<<"]))
    fields = _fields(first_name=("ERIKSSON", ["field_extractor"]), last_name=("ERIKSSON", ["field_extractor"]),
                     middle_name=("ANNA MARIA", ["field_extractor"]))
    apply_verified(fields, src, Context())
    assert fields["first_name"].value == "ANNA MARIA"


def test_label_placeholder_is_not_a_name():
    assert normalize_value("full_name", "<NAME><NAME>", Context())[0] == ""

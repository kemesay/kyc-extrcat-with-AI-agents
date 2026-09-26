"""As-printed interpretation and the formats found in the real-ID batch — fictional values only."""

import pytest

from universal.id_numbers import check_numbers
from universal.printed import (
    best_template_fit, caps_surname_split, clean_document_number, clean_person_name, fit_template,
    resolve_dates, resolve_document_number, resolve_names,
)
from universal.prompts import EXTRACTION_FIELDS
from universal.reconcile import Context, FieldResult, apply_id_number_checks


def _fields(**values):
    fields = {n: FieldResult() for n in EXTRACTION_FIELDS}
    for n, v in values.items():
        fields[n] = FieldResult(value=v, confidence=0.6, source="consensus", agreeing_agents=["field_extractor"])
    return fields


def _out(**printed):
    return {"field_extractor": printed}


# ── names ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("MR JOHN PAUL", "JOHN PAUL"), ("Dr. Anna", "Anna"), ("ATO ABEBE", "ABEBE"), ("MULU,ALEMU", "MULU ALEMU"),
])
def test_clean_person_name(raw, expected):
    assert clean_person_name(raw) == expected


@pytest.mark.parametrize("full, split", [
    ("DOE Jane Mary", ("DOE", "Jane Mary")), ("ROE, Kim Lee", ("ROE", "Kim Lee")),
    ("Jane Mary DOE", ("DOE", "Jane Mary")), ("JANE MARY DOE", None), ("Jane Doe", None),
])
def test_caps_surname_convention(full, split):
    assert caps_surname_split(full) == split


def test_label_structure_replaces_positional_split():
    # Field 1 printed below field 2 (as on some US cards): the label number decides.
    fields = _fields(full_name="JANE Q DOE", first_name="JANE", last_name="Q")
    resolve_names(fields, _out(surname_as_labelled="DOE", given_names_as_labelled="JANE Q"), Context())
    assert (fields["last_name"].value, fields["first_name"].value, fields["middle_name"].value) == ("DOE", "JANE", "Q")


def test_caps_convention_overrides_wrong_label_guess():
    fields = _fields(full_name="DOE Jane Mary")
    resolve_names(fields, _out(surname_as_labelled="Mary", given_names_as_labelled="Jane DOE"), Context())
    assert (fields["last_name"].value, fields["first_name"].value, fields["middle_name"].value) == ("DOE", "Jane", "Mary")


def test_each_given_name_word_is_its_own_name():
    fields = _fields(first_name="JANE QUINN", last_name="DOE")
    resolve_names(fields, _out(surname_as_labelled="DOE", given_names_as_labelled="JANE QUINN"), Context())
    assert (fields["first_name"].value, fields["middle_name"].value) == ("JANE", "QUINN")


def test_surname_with_spaces_never_split():
    fields = _fields()
    resolve_names(fields, _out(surname_as_labelled="VAN DER BERG", given_names_as_labelled="ANNA MARIA"), Context())
    assert (fields["last_name"].value, fields["first_name"].value, fields["middle_name"].value) == \
        ("VAN DER BERG", "ANNA", "MARIA")


# ── dates ──────────────────────────────────────────────────────────────────

def test_us_card_dates_read_month_first():
    fields = _fields(date_of_birth="1984-02-01")         # agent read 01/02/1984 day-first
    resolve_dates(fields, _out(date_of_birth_as_printed="01/02/1984"), Context(country="USA"))
    assert fields["date_of_birth"].value == "1984-01-02"


def test_dual_calendar_pair_settles_month_first_gregorian_half():
    # EC 29/5/1987 is 6 Feb 1995 — so the Gregorian '2/6/1995' is month-first.
    fields = _fields(date_of_birth="1995-06-02")
    resolve_dates(fields, _out(date_of_birth_as_printed="29/5/1987 | 2/6/1995"), Context(country="ETH"))
    assert fields["date_of_birth"].value == "1995-02-06"
    assert fields["date_of_birth_orig"].value == "29/5/1987"


def test_verified_date_not_overridden_by_printed():
    fields = _fields(date_of_birth="1984-01-02")
    fields["date_of_birth"].verified_by = ["mrz"]
    resolve_dates(fields, _out(date_of_birth_as_printed="02/01/1984"), Context(country="FRA"))
    assert fields["date_of_birth"].value == "1984-01-02"


# ── document numbers ───────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("4D DLN 000012345678 CLASS C", "000012345678"), ("LIC# WDL00X00000B", "WDL00X00000B"),
    ("FAN 1234 5678 9012 3456", "1234 5678 9012 3456"), ("DL NO. 012345678", "012345678"),
])
def test_clean_document_number(raw, expected):
    assert clean_document_number(raw) == expected


@pytest.mark.parametrize("value, template, expected", [
    ("CA123450A", "AA99999AA", "CA12345OA"),     # zero read for letter O
    ("SMITH701150JJ9AA", "XXXXX999999XX9AA", "SMITH701150JJ9AA"),
    ("A1234 - 56789 - 01234", "A9999-99999-99999", "A1234-56789-01234"),
    ("123456", "AAXXXXXX9", None),
])
def test_fit_template(value, template, expected):
    assert fit_template(value, template) == expected


def test_can_rejected_in_favour_of_format_match():
    fields = _fields(document_number="123456")            # 6-digit card access number
    resolve_document_number(fields, _out(other_numbers_with_labels="CAN: 123456 | DOC: AB12CD345"),
                            Context(country="NLD"))
    assert fields["document_number"].value == "AB12CD345"


def test_issue_suffix_dropped_by_template():
    assert best_template_fit(["SMITH701150JJ9AA 02"], "GBR") == "SMITH701150JJ9AA"


# ── number schemes from the batch ──────────────────────────────────────────

def test_dvla_licence_number_decodes_and_checks_surname():
    (check,) = check_numbers("GBR", {"document_number": "SMITH756150JJ9AA"})
    assert (check.date_of_birth, check.sex) == ("1970-06-15", "Female")   # month 56 = June, female
    fields = _fields(first_name="SMITH", last_name="JANE")                 # swapped by the agent
    apply_id_number_checks(fields, [check], Context(country="GBR"))
    assert (fields["last_name"].value, fields["first_name"].value) == ("SMITH", "JANE")


def test_ontario_licence_number_decodes():
    (check,) = check_numbers("CAN", {"document_number": "D1234-56789-85612"})
    assert (check.date_of_birth, check.sex, check.details["surname_prefix"]) == ("1998-06-12", "Female", "D")


@pytest.mark.parametrize("number, dob, sex", [("811218-9876", "1981-12-18", "Male")])
def test_swedish_personnummer(number, dob, sex):
    (check,) = check_numbers("SWE", {"personal_number": number})
    assert check.verified and (check.date_of_birth, check.sex) == (dob, sex)


def test_belgian_national_register_number():
    (check,) = check_numbers("BEL", {"personal_number": "85.07.30-033.28"})
    assert check.verified and check.date_of_birth == "1985-07-30" and check.sex == "Male"


def test_personal_number_moved_out_of_document_number():
    (check,) = check_numbers("SWE", {"document_number": "811218-9876"})
    fields = _fields(document_number="811218-9876")
    fields["document_number"].alternatives = ["AB1234567"]
    apply_id_number_checks(fields, [check], Context(country="SWE"))
    assert fields["personal_number"].value == "811218-9876" and fields["document_number"].value == "AB1234567"


# ── OCR label evidence ─────────────────────────────────────────────────────

def test_ocr_numbered_labels_glued_to_values():
    from universal.printed import ocr_name_structure
    ocr = "WASHINGTON\n4d LIC#WDL00X\n1DOE\n2JANE QUINN\n3 DOB 01/02/1990"
    assert ocr_name_structure(ocr, ["DOE JANE QUINN"]) == ("DOE", "JANE QUINN", "ocr_numbered_labels")


def test_ocr_glued_given_names_respaced_from_agents():
    from universal.printed import ocr_name_structure
    assert ocr_name_structure("1ROE\n2KIML", ["ROE KIM L"])[:2] == ("ROE", "KIM L")


def test_ocr_comma_format_surname_line():
    from universal.printed import ocr_name_structure
    assert ocr_name_structure("NAME/NOM\nDOE,\nJANE,QUINN\n8 1 MAIN ST", ["DOE JANE QUINN"])[:2] == ("DOE", "JANE QUINN")


def test_ocr_evidence_ignored_when_letters_not_in_agents_reading():
    from universal.printed import ocr_name_structure
    assert ocr_name_structure("1CLASS\n2RESTRICTIONS", ["DOE JANE"]) is None


def test_ocr_label_evidence_outranks_positional_transcription():
    fields = _fields(full_name="DOE JANE QUINN")
    resolve_names(fields, _out(surname_as_labelled="QUINN", given_names_as_labelled="JANE DOE"), Context(),
                  ocr_text="1DOE\n2JANE QUINN")
    assert (fields["last_name"].value, fields["first_name"].value, fields["middle_name"].value) == ("DOE", "JANE", "QUINN")


def test_single_ethiopian_date_confirmed_by_ocr_gregorian_reading():
    # Only the EC half was transcribed; OCR caught the month-first Gregorian half.
    fields = _fields(expiry_date="2029-05-08")
    resolve_dates(fields, _out(expiry_date_as_printed="29/11/2021"), Context(country="ETH"), ocr_text="Expiry 8/5/2029")
    assert fields["expiry_date"].value == "2029-08-05"


def test_licence_letters_rebuild_structure():
    (check,) = check_numbers("CAN", {"document_number": "D1234-56789-85612"})
    fields = _fields(full_name="DOE JANE QUINN", last_name="JANE", first_name="QUINN")
    apply_id_number_checks(fields, [check], Context(country="CAN"))
    assert (fields["last_name"].value, fields["first_name"].value) == ("DOE", "JANE QUINN")


def test_orphan_date_placed_by_plausibility():
    fields = _fields()
    resolve_dates(fields, _out(date_of_birth_as_printed="04/18/1984 | 02/11/2026"), Context(country="USA"))
    assert fields["date_of_birth"].value == "1984-04-18" and fields["issue_date"].value == "2026-02-11"


def test_future_issue_reading_swapped():
    from datetime import date
    today = date.today()
    if today.month == 12:
        pytest.skip("needs a later month in the current year")
    # Day-first reading = 1 <month after today> (future); month-first = January (past).
    printed = f"01/{today.month + 1:02d}/{today.year}"
    fields = _fields()
    resolve_dates(fields, _out(issue_date_as_printed=printed), Context(country=""))
    assert fields["issue_date"].value == f"{today.year}-01-{today.month + 1:02d}"


def test_country_from_header_text():
    from universal.printed import country_from_text
    assert country_from_text("MARYLAND USA\nDriver's License") == "USA"
    assert country_from_text("KONINKRIJK DER NEDERLANDEN") == "NLD"
    assert country_from_text("Alberta DRIVER'S LICENCE") == "CAN"


def test_country_never_taken_from_nationality_line():
    from universal.printed import country_from_text
    assert country_from_text("RESIDENT IDENTITY CARD\nNationality: Ethiopia") != "ETH"


def test_country_from_distinctive_checksum_number():
    from universal.printed import country_from_number
    assert country_from_number(["11010519491231002X"]) == "CHN"
